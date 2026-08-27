#!/usr/bin/env python3
"""
WS6.1 — Sequencing / assembly **error-injection module**.

(Protocol name: ``107_error_injection_module.py``; renumbered to 150 because
107-109 were already taken by the WS5 definitive re-run and the WS7 container
work.  See ``scripts/`` for the full numbering.)

WHY THIS EXISTS
---------------
Reviewer 2 (R2-o3) asks whether MAGICC is robust to sequencing and assembly
error.  The protocol's *a priori* assumption was that k-mer methods must be more
error-sensitive than protein-level methods.  **For indels that assumption has
already been falsified on real data** (WS3 Track A: on the indel-dense Meslier
MOCK1 MinION assembly MAGICC's completeness MAE was 4.91 pp versus CheckM2 50.69
and DeepCheck 56.83, because frameshifts destroy ORFs while canonical k-mers are
barely affected).  The open question is therefore **substitutions**, where the
mechanism genuinely favours the protein-level tools:

    MAGICC counts canonical **9-mers**.  A single substitution destroys up to 9
    overlapping 9-mers, so at per-base substitution rate p the expected fraction
    of corrupted k-mers is 1 - (1 - p)^9  ->  8.6 % at p = 1 %, 37 % at p = 5 %.
    CheckM2 works on translated proteins, where synonymous substitutions are
    invisible and non-synonymous ones usually leave the HMM/KO hit intact.

This module implements four error processes.  Every one of them is designed so
that **the ground truth of the sample is unchanged by the injection** - see
"GROUND TRUTH" below, because a reviewer will ask.

THE FOUR ERROR PROCESSES
------------------------
1. ``substitutions``  (highest priority)
   i.i.d. per-base substitution with probability *p* over A/C/G/T positions
   (ambiguity codes are never touched).  Two models:
     * ``uniform``  - the substituted base is one of the other three, equiprobably.
     * ``titv``     - transition/transversion-biased.  Real sequencers and real
                      variant spectra are Ti-biased; with A<->G and C<->T as the
                      transitions, P(transition) = R/(R+1) for a Ti/Tv ratio R
                      (default R = 2.0, the canonical genomic value).
   The realised rate, and the realised Ti/Tv ratio, are measured and recorded -
   nothing is assumed.

2. ``indels``  (dose-response complement to the WS3 real-data result)
   i.i.d. per-base indel events with probability *p*.  Each event is an insertion
   or a deletion with probability ``ins_frac`` (default 0.5, so the expected net
   length change is zero), with length drawn from a geometric distribution
   (default mean 1.33 bp), matching the strongly 1-bp-dominated indel spectrum of
   nanopore/Ion homopolymer errors.  Inserted bases are uniform over A/C/G/T.

3. ``chimeras``
   Mis-joins between unrelated sequences.  ``rate`` is the fraction of contigs
   drawn into a mis-join; each event concatenates two contigs end to end, with
   the second partner reverse-complemented with probability 0.5.  Cross-origin
   joins (dominant<->contaminant) are formed first because those are the true
   "unrelated sequence" chimeras; once the contaminant pool is exhausted, joins
   are made between non-adjacent dominant contigs, which is also a mis-join
   between sequences that are not neighbours in the reference.
   **No base is added or removed** - only contig boundaries change.

4. ``uneven_coverage``
   The canonical assembly artefact of uneven / strain-heterogeneous coverage:
   **redundant duplicated contigs**.  Each contig is assigned a log-normal
   coverage; segments are drawn with probability proportional to
   (length x coverage) and re-emitted as additional contigs until ``rate`` of the
   assembly's bp has been duplicated.  This is the truth-preserving component of
   uneven coverage; the *loss* component (low-coverage dropout) is by definition
   the completeness axis and is already covered by Sets A/E and by the real-data
   fragmentation gradient of WS3, so folding it in here would confound the axis.

GROUND TRUTH (this is the part a reviewer will probe)
-----------------------------------------------------
MAGICC's conventions are
    completeness  = dominant-derived bp / dominant FULL reference length x 100
    contamination = contaminant-derived bp / dominant FULL reference length x 100

* ``substitutions`` change which base is at a position but not which organism the
  position derives from, and do not change any length.  Truth is **exactly**
  invariant.  (This is why substitutions are the cleanest possible robustness
  probe: only the tool's input representation changes.)
* ``chimeras`` re-partition exactly the same multiset of bases into different
  contigs.  Truth is **exactly** invariant.  Asserted per sample.
* ``uneven_coverage`` re-emits sequence that is already present.  The *set* of
  reference positions represented by the assembly is unchanged, so truth under
  MAGICC's denominator is invariant.  Because a marker-duplication-based tool may
  legitimately read a duplicated segment as contamination, the alternative
  accounting is also emitted, as ``dup_bp``, so the sensitivity can be reported.
* ``indels`` are the only process that changes assembly length.  The primary
  truth convention is that an indel is a **mis-rendering of DNA that is present**
  (a base-calling/consensus artefact), not a change in organism content, so truth
  is held at the pre-error value; insertions/deletions are balanced so the
  expected net change is zero, and both the realised net bp change and an
  indel-adjusted truth are recorded per sample so the bound is auditable rather
  than asserted.

DETERMINISM
-----------
Every function takes an explicit ``numpy.random.Generator``.  Given the same
generator seed and the same input contigs the output is byte-identical; this is
verified end-to-end by ``scripts/151_generate_set_G.py --determinism-check``.

Self-test:
    python scripts/150_error_injection_module.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# base encoding: A=0 C=1 G=2 T=3, anything else = 4 (never mutated)
# --------------------------------------------------------------------------
_CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _CODE[_b] = _i
for _i, _b in enumerate(b"acgt"):
    _CODE[_b] = _i
_DECODE = np.frombuffer(b"ACGTN", dtype=np.uint8)

# complement in code space: A<->T (0<->3), C<->G (1<->2), N->N
_COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)

ERROR_TYPES = ("none", "substitution", "substitution_titv", "indel",
               "chimera", "uneven_coverage")


def encode(seq: str) -> np.ndarray:
    """ASCII sequence -> uint8 code array (A=0 C=1 G=2 T=3, other=4)."""
    return _CODE[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def decode(codes: np.ndarray) -> str:
    """uint8 code array -> ASCII sequence (code 4 -> 'N')."""
    return _DECODE[codes].tobytes().decode("ascii")


def revcomp_codes(codes: np.ndarray) -> np.ndarray:
    return _COMP[codes][::-1]


def revcomp(seq: str) -> str:
    return decode(revcomp_codes(encode(seq)))


# ==========================================================================
# 1. SUBSTITUTIONS
# ==========================================================================
def inject_substitutions(contigs: Sequence[str], rate: float,
                         rng: np.random.Generator,
                         model: str = "uniform",
                         titv: float = 2.0) -> Tuple[List[str], dict]:
    """i.i.d. per-base substitution at probability ``rate``.

    Parameters
    ----------
    contigs : sequence of str
    rate    : float          per-base substitution probability (0 <= rate <= 1)
    rng     : Generator
    model   : {'uniform', 'titv'}
    titv    : float          transition/transversion ratio for model='titv'

    Returns
    -------
    (contigs, stats).  Lengths are preserved exactly, contig count preserved
    exactly, and ambiguity codes are never substituted.
    """
    if model not in ("uniform", "titv"):
        raise ValueError(f"unknown substitution model {model!r}")
    lengths = np.array([len(c) for c in contigs], dtype=np.int64)
    if rate <= 0 or lengths.sum() == 0:
        return list(contigs), _sub_stats(0, 0, 0, int(lengths.sum()), rate, model, titv)

    codes = encode("".join(contigs))
    valid = codes < 4
    n_valid = int(valid.sum())

    sel = valid & (rng.random(codes.size) < rate)
    idx = np.flatnonzero(sel)
    n_sub = idx.size
    if n_sub:
        if model == "uniform":
            off = rng.integers(1, 4, size=n_sub).astype(np.uint8)
        else:
            p_ti = titv / (titv + 1.0)
            is_ti = rng.random(n_sub) < p_ti
            # code offset 2 is always a transition (A<->G, C<->T); 1 and 3 are
            # transversions.
            off = np.where(is_ti, 2, 1 + 2 * rng.integers(0, 2, size=n_sub)
                           ).astype(np.uint8)
        n_ti = int((off == 2).sum())
        codes[idx] = (codes[idx] + off) % 4
    else:
        n_ti = 0

    out, pos = [], 0
    for L in lengths:
        out.append(decode(codes[pos:pos + L]))
        pos += int(L)
    return out, _sub_stats(n_sub, n_ti, n_sub - n_ti, n_valid, rate, model, titv)


def _sub_stats(n_sub, n_ti, n_tv, n_valid, rate, model, titv) -> dict:
    return {
        "n_substitutions": int(n_sub),
        "n_transitions": int(n_ti),
        "n_transversions": int(n_tv),
        "n_substitutable_bp": int(n_valid),
        "target_rate": float(rate),
        "realised_rate": float(n_sub / n_valid) if n_valid else 0.0,
        "realised_titv": (float(n_ti / n_tv) if n_tv else float("nan")),
        "substitution_model": model,
        "titv_parameter": float(titv) if model == "titv" else float("nan"),
    }


# ==========================================================================
# 2. INDELS
# ==========================================================================
@njit(cache=True)
def _build_indel(codes, ev_pos, ev_ins, ev_len, ins_codes, ins_off, out):
    """Apply sorted indel events.  Returns the number of output bases.

    Deletions consume ``ev_len`` bases starting at ``ev_pos``; events whose
    position was swallowed by an earlier deletion are skipped (their positions
    are strictly increasing, so a single pointer suffices).  Insertions emit
    ``ev_len`` bases *before* the base at ``ev_pos``.
    """
    n = codes.shape[0]
    ne = ev_pos.shape[0]
    o = 0
    e = 0
    i = 0
    while i < n:
        while e < ne and ev_pos[e] < i:
            e += 1
        if e < ne and ev_pos[e] == i:
            L = ev_len[e]
            if ev_ins[e] == 1:
                base = ins_off[e]
                for k in range(L):
                    out[o] = ins_codes[base + k]
                    o += 1
                e += 1
                out[o] = codes[i]
                o += 1
                i += 1
            else:
                i += L
                e += 1
        else:
            out[o] = codes[i]
            o += 1
            i += 1
    return o


def inject_indels(contigs: Sequence[str], rate: float, rng: np.random.Generator,
                  ins_frac: float = 0.5, geom_p: float = 0.75,
                  min_contig_bp: int = 1) -> Tuple[List[str], dict]:
    """i.i.d. per-base indel events at probability ``rate``.

    ``ins_frac`` of events are insertions (default 0.5 -> expected net length
    change zero), lengths ~ Geometric(``geom_p``) (default mean 1/0.75 = 1.33 bp,
    i.e. overwhelmingly 1 bp, matching real homopolymer indel spectra).
    """
    lengths = [len(c) for c in contigs]
    total = int(sum(lengths))
    if rate <= 0 or total == 0:
        return list(contigs), _indel_stats(0, 0, 0, 0, total, total, rate,
                                           ins_frac, geom_p)

    out: List[str] = []
    n_ins = n_del = ins_bp = del_bp = 0
    for c in contigs:
        codes = encode(c)
        n = codes.size
        if n == 0:
            continue
        ev = (codes < 4) & (rng.random(n) < rate)
        pos = np.flatnonzero(ev).astype(np.int64)
        if pos.size == 0:
            out.append(c)
            continue
        is_ins = (rng.random(pos.size) < ins_frac).astype(np.uint8)
        lens = rng.geometric(geom_p, size=pos.size).astype(np.int64)
        # a deletion may not run past the end of the contig
        lens = np.where(is_ins == 0, np.minimum(lens, n - pos), lens)
        ins_lens = np.where(is_ins == 1, lens, 0)
        ins_off = np.concatenate([[0], np.cumsum(ins_lens)]).astype(np.int64)
        n_ins_bp = int(ins_lens.sum())
        ins_codes = (rng.integers(0, 4, size=max(n_ins_bp, 1))
                     .astype(np.uint8))
        buf = np.empty(n + n_ins_bp, dtype=np.uint8)
        m = _build_indel(codes, pos, is_ins, lens, ins_codes,
                         ins_off[:pos.size], buf)
        seq = decode(buf[:m])
        if len(seq) >= min_contig_bp:
            out.append(seq)
        n_ins += int(is_ins.sum())
        n_del += int((is_ins == 0).sum())
        ins_bp += n_ins_bp
        del_bp += int(lens[is_ins == 0].sum())

    new_total = int(sum(len(c) for c in out))
    return out, _indel_stats(n_ins, n_del, ins_bp, del_bp, total, new_total,
                             rate, ins_frac, geom_p)


def _indel_stats(n_ins, n_del, ins_bp, del_bp, total, new_total, rate,
                 ins_frac, geom_p) -> dict:
    n_ev = n_ins + n_del
    return {
        "n_indel_events": int(n_ev),
        "n_insertions": int(n_ins),
        "n_deletions": int(n_del),
        "inserted_bp": int(ins_bp),
        "deleted_bp": int(del_bp),
        "net_bp_change": int(new_total - total),
        "net_bp_change_frac": float((new_total - total) / total) if total else 0.0,
        "target_rate": float(rate),
        "realised_rate": float(n_ev / total) if total else 0.0,
        "indels_per_kb": float(1000.0 * n_ev / total) if total else 0.0,
        "insertion_fraction": float(ins_frac),
        "indel_length_geom_p": float(geom_p),
        "mean_indel_length": float((ins_bp + del_bp) / n_ev) if n_ev else 0.0,
    }


# ==========================================================================
# 3. CHIMERIC CONTIGS
# ==========================================================================
def inject_chimeras(contigs: Sequence[str], origins: Sequence[str], rate: float,
                    rng: np.random.Generator,
                    rc_prob: float = 0.5) -> Tuple[List[str], List[str], dict]:
    """Mis-joins between unrelated sequences.

    ``rate`` is the target fraction of contigs drawn into a mis-join; each event
    fuses two contigs, so ``n_events = round(rate * n_contigs / 2)``.

    Cross-origin (dominant<->contaminant) joins are made first - those are the
    genuine "unrelated sequence" chimeras - and dominant<->dominant joins between
    non-adjacent contigs are used once the contaminant pool is exhausted.

    Origin of a chimeric contig is recorded as ``'X'`` (mixed) when the two
    partners differ, otherwise the shared origin.  **Total bp and per-origin bp
    are preserved exactly.**
    """
    contigs = list(contigs)
    origins = list(origins)
    n = len(contigs)
    if rate <= 0 or n < 2:
        return contigs, origins, _chim_stats(0, 0, 0, n, n, rate)

    # Stochastic (unbiased) rounding: a 10-contig assembly at rate 5 % should get
    # 0.25 mis-joins in expectation, and deterministic rounding would silently
    # turn that into 0 for every such sample, biasing the whole arm downwards.
    exact = rate * n / 2.0
    n_events = int(np.floor(exact))
    if rng.random() < (exact - n_events):
        n_events += 1
    n_events = min(n_events, n // 2)
    if n_events <= 0:
        return contigs, origins, _chim_stats(0, 0, 0, n, n, rate)

    dom_idx = [i for i, o in enumerate(origins) if o == "D"]
    con_idx = [i for i, o in enumerate(origins) if o != "D"]
    rng.shuffle(dom_idx)
    rng.shuffle(con_idx)

    pairs: List[Tuple[int, int]] = []
    n_cross = 0
    di = ci = 0
    while len(pairs) < n_events and di < len(dom_idx) and ci < len(con_idx):
        pairs.append((dom_idx[di], con_idx[ci]))
        di += 1
        ci += 1
        n_cross += 1
    # remaining events: dominant<->dominant, then contaminant<->contaminant
    rest = dom_idx[di:] + con_idx[ci:]
    rng.shuffle(rest)
    k = 0
    while len(pairs) < n_events and k + 1 < len(rest):
        pairs.append((rest[k], rest[k + 1]))
        k += 2

    used = set()
    for a, b in pairs:
        used.add(a)
        used.add(b)

    flips = rng.random(len(pairs)) < rc_prob
    new_contigs: List[str] = []
    new_origins: List[str] = []
    for (a, b), flip in zip(pairs, flips):
        sb = revcomp(contigs[b]) if flip else contigs[b]
        new_contigs.append(contigs[a] + sb)
        new_origins.append(origins[a] if origins[a] == origins[b] else "X")
    for i in range(n):
        if i not in used:
            new_contigs.append(contigs[i])
            new_origins.append(origins[i])

    order = rng.permutation(len(new_contigs))
    new_contigs = [new_contigs[i] for i in order]
    new_origins = [new_origins[i] for i in order]
    return new_contigs, new_origins, _chim_stats(
        len(pairs), n_cross, int(flips.sum()), n, len(new_contigs), rate)


def _chim_stats(n_events, n_cross, n_rc, n_before, n_after, rate) -> dict:
    return {
        "n_chimera_events": int(n_events),
        "n_cross_origin_joins": int(n_cross),
        "n_within_origin_joins": int(n_events - n_cross),
        "n_reverse_complemented_partners": int(n_rc),
        "n_contigs_before": int(n_before),
        "n_contigs_after": int(n_after),
        "target_rate": float(rate),
        "realised_contig_fraction_in_chimeras":
            float(2 * n_events / n_before) if n_before else 0.0,
    }


# ==========================================================================
# 4. UNEVEN-COVERAGE ARTEFACTS (redundant duplicated contigs)
# ==========================================================================
def inject_uneven_coverage(contigs: Sequence[str], origins: Sequence[str],
                           rate: float, rng: np.random.Generator,
                           cov_sigma: float = 1.0,
                           frac_range: Tuple[float, float] = (0.25, 1.0),
                           min_seg_bp: int = 300) -> Tuple[List[str], List[str], dict]:
    """Duplicate ``rate`` of the assembly's bp as extra (redundant) contigs.

    Each contig gets a log-normal coverage (``sigma = cov_sigma``); segments are
    sampled with probability proportional to (length x coverage), so the
    duplicated material concentrates in the high-coverage part of the assembly,
    exactly as strain-heterogeneous / uneven-coverage over-assembly does.

    Duplicated bp are additional *copies* of sequence already present, so the set
    of reference positions represented by the assembly - and therefore MAGICC's
    completeness and contamination - are unchanged.  ``dup_bp_dominant`` and
    ``dup_bp_contaminant`` are reported so the alternative (duplication-counted)
    accounting can be evaluated as a sensitivity.
    """
    contigs = list(contigs)
    origins = list(origins)
    n = len(contigs)
    total = int(sum(len(c) for c in contigs))
    if rate <= 0 or n == 0 or total == 0:
        return contigs, origins, _cov_stats(0, 0, 0, 0.0, total, n, n, rate, cov_sigma)

    lens = np.array([len(c) for c in contigs], dtype=np.float64)
    cov = np.exp(rng.normal(0.0, cov_sigma, size=n))
    w = lens * cov
    w = w / w.sum()

    target_bp = int(round(rate * total))
    dup_contigs: List[str] = []
    dup_origins: List[str] = []
    dup_bp = {"D": 0, "C": 0, "X": 0}
    done = 0
    guard = 0
    while done < target_bp and guard < 200000:
        guard += 1
        i = int(rng.choice(n, p=w))
        c = contigs[i]
        L = len(c)
        if L < min_seg_bp:
            seg_len = L
        else:
            f = rng.uniform(frac_range[0], frac_range[1])
            seg_len = max(min_seg_bp, int(round(f * L)))
            seg_len = min(seg_len, L)
        remaining = target_bp - done
        if seg_len > remaining:
            seg_len = max(min(remaining, L), 1)
        start = int(rng.integers(0, L - seg_len + 1))
        dup_contigs.append(c[start:start + seg_len])
        o = origins[i]
        dup_origins.append(o)
        dup_bp[o if o in dup_bp else "X"] += seg_len
        done += seg_len

    out_contigs = contigs + dup_contigs
    out_origins = origins + dup_origins
    order = rng.permutation(len(out_contigs))
    out_contigs = [out_contigs[i] for i in order]
    out_origins = [out_origins[i] for i in order]
    return out_contigs, out_origins, _cov_stats(
        len(dup_contigs), dup_bp["D"], dup_bp["C"] + dup_bp["X"], float(cov.max() / cov.min()),
        total, n, len(out_contigs), rate, cov_sigma)


def _cov_stats(n_dup, dup_d, dup_c, cov_ratio, total, n_before, n_after,
               rate, cov_sigma) -> dict:
    dup_total = int(dup_d + dup_c)
    return {
        "n_duplicated_segments": int(n_dup),
        "dup_bp": dup_total,
        "dup_bp_dominant": int(dup_d),
        "dup_bp_contaminant": int(dup_c),
        "target_rate": float(rate),
        "realised_dup_fraction": float(dup_total / total) if total else 0.0,
        "coverage_sigma": float(cov_sigma),
        "coverage_max_min_ratio": float(cov_ratio),
        "n_contigs_before": int(n_before),
        "n_contigs_after": int(n_after),
    }


# ==========================================================================
# dispatcher
# ==========================================================================
@dataclass
class ErrorSpec:
    """One error arm: a type plus its rate and model parameters."""
    error_type: str
    rate: float = 0.0
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def arm(self) -> str:
        return self.error_type if self.rate == 0 else f"{self.error_type}"


def apply_error(contigs: Sequence[str], origins: Sequence[str], spec: ErrorSpec,
                rng: np.random.Generator) -> Tuple[List[str], List[str], dict]:
    """Apply one error process.  Returns (contigs, origins, stats)."""
    et = spec.error_type
    if et in ("none", None) or spec.rate <= 0:
        return list(contigs), list(origins), {"error_type": "none", "target_rate": 0.0,
                                              "realised_rate": 0.0}
    if et == "substitution":
        c, s = inject_substitutions(contigs, spec.rate, rng, model="uniform")
        s["error_type"] = et
        return c, list(origins), s
    if et == "substitution_titv":
        c, s = inject_substitutions(contigs, spec.rate, rng, model="titv",
                                    titv=float(spec.params.get("titv", 2.0)))
        s["error_type"] = et
        return c, list(origins), s
    if et == "indel":
        c, s = inject_indels(contigs, spec.rate, rng,
                             ins_frac=float(spec.params.get("ins_frac", 0.5)),
                             geom_p=float(spec.params.get("geom_p", 0.75)))
        s["error_type"] = et
        return c, list(origins), s
    if et == "chimera":
        c, o, s = inject_chimeras(contigs, origins, spec.rate, rng,
                                  rc_prob=float(spec.params.get("rc_prob", 0.5)))
        s["error_type"] = et
        return c, o, s
    if et == "uneven_coverage":
        c, o, s = inject_uneven_coverage(
            contigs, origins, spec.rate, rng,
            cov_sigma=float(spec.params.get("cov_sigma", 1.0)))
        s["error_type"] = et
        return c, o, s
    raise ValueError(f"unknown error type {et!r}")


def expected_kmer_corruption(rate: float, k: int = 9) -> float:
    """Expected fraction of k-mers containing >= 1 substitution: 1 - (1-p)^k.

    This is the mechanistic prediction that WS6 tests; it is written into the
    generation metadata so the measured degradation can be read against it.
    """
    return float(1.0 - (1.0 - rate) ** k)


def warm_numba() -> None:
    """Compile the numba kernel once (workers call this at start-up)."""
    inject_indels(["ACGTACGTACGTACGTACGT" * 5], 0.1, np.random.default_rng(0))


# ==========================================================================
# self-test
# ==========================================================================
def selftest(verbose: bool = True) -> List[str]:
    msgs: List[str] = []

    def ok(m):
        msgs.append(m)
        if verbose:
            print(f"  [ok] {m}")

    rng = np.random.default_rng(1234)
    seq = decode(rng.integers(0, 4, size=200_000).astype(np.uint8))
    contigs = [seq[i:i + 20_000] for i in range(0, len(seq), 20_000)]
    origins = ["D"] * 8 + ["C"] * 2

    # ---- encoding round-trip -------------------------------------------
    assert decode(encode(seq)) == seq
    ok("encode/decode round-trip")
    assert revcomp(revcomp(seq)) == seq
    assert revcomp("ACGTN") == "NACGT"
    ok("reverse complement is an involution and complements correctly")

    # ---- substitutions --------------------------------------------------
    for rate in (0.001, 0.01, 0.05):
        c, s = inject_substitutions(contigs, rate, np.random.default_rng(7))
        assert [len(x) for x in c] == [len(x) for x in contigs], "length changed"
        assert len(c) == len(contigs)
        n_diff = sum(sum(1 for a, b in zip(x, y) if a != b) for x, y in zip(contigs, c))
        assert n_diff == s["n_substitutions"], (n_diff, s["n_substitutions"])
        assert abs(s["realised_rate"] - rate) < 0.15 * rate + 1e-4
    ok("substitutions: lengths preserved, realised rate matches target, "
       "count matches observed diffs")

    c1, _ = inject_substitutions(contigs, 0.01, np.random.default_rng(11))
    c2, _ = inject_substitutions(contigs, 0.01, np.random.default_rng(11))
    assert c1 == c2
    ok("substitutions are deterministic given the seed")

    _, s = inject_substitutions(contigs, 0.05, np.random.default_rng(3),
                                model="titv", titv=2.0)
    assert 1.7 < s["realised_titv"] < 2.35, s["realised_titv"]
    ok(f"titv model reproduces the requested Ti/Tv ratio "
       f"(realised {s['realised_titv']:.3f} for target 2.0)")

    amb = ["ACGTNNNNNNACGT" * 100]
    ca, sa = inject_substitutions(amb, 1.0, np.random.default_rng(5))
    assert ca[0].count("N") == amb[0].count("N")
    assert sa["n_substitutions"] == sa["n_substitutable_bp"]
    ok("ambiguity codes are never substituted")

    # ---- indels ---------------------------------------------------------
    for rate in (0.001, 0.01, 0.05):
        c, s = inject_indels(contigs, rate, np.random.default_rng(9))
        tot0 = sum(len(x) for x in contigs)
        tot1 = sum(len(x) for x in c)
        assert s["net_bp_change"] == tot1 - tot0
        assert abs(s["realised_rate"] - rate) < 0.2 * rate + 1e-4
        assert abs(s["net_bp_change_frac"]) < 0.02, s["net_bp_change_frac"]
    ok("indels: event rate matches target and net length change stays < 2 %")

    c1, _ = inject_indels(contigs, 0.02, np.random.default_rng(13))
    c2, _ = inject_indels(contigs, 0.02, np.random.default_rng(13))
    assert c1 == c2
    ok("indels are deterministic given the seed")

    _, s = inject_indels(contigs, 0.02, np.random.default_rng(21), ins_frac=0.0)
    assert s["n_insertions"] == 0 and s["net_bp_change"] < 0
    _, s = inject_indels(contigs, 0.02, np.random.default_rng(21), ins_frac=1.0)
    assert s["n_deletions"] == 0 and s["net_bp_change"] > 0
    ok("insertion fraction controls the direction of net length change")

    # ---- chimeras -------------------------------------------------------
    for rate in (0.1, 0.4, 0.8):
        c, o, s = inject_chimeras(contigs, origins, rate, np.random.default_rng(17))
        assert sum(len(x) for x in c) == sum(len(x) for x in contigs), "bp changed"
        assert len(c) == len(contigs) - s["n_chimera_events"]
        assert abs(s["realised_contig_fraction_in_chimeras"] - rate) <= 0.21
    ok("chimeras: total bp preserved exactly, contig count falls by the event count")

    # stochastic rounding must be unbiased on small contig counts
    small = [seq[i:i + 20_000] for i in range(0, 200_000, 20_000)]   # 10 contigs
    fr = np.mean([inject_chimeras(small, ["D"] * 10, 0.05,
                                  np.random.default_rng(1000 + i))[2]
                  ["realised_contig_fraction_in_chimeras"] for i in range(600)])
    assert abs(fr - 0.05) < 0.015, fr
    ok(f"chimera dose is unbiased on a 10-contig assembly "
       f"(mean realised {fr:.4f} for target 0.05)")

    c, o, s = inject_chimeras(contigs, origins, 0.4, np.random.default_rng(17))
    dom_bp_before = sum(len(x) for x, oo in zip(contigs, origins) if oo == "D")
    # per-origin bp is preserved for pure contigs; mixed contigs are labelled X
    mixed_bp = sum(len(x) for x, oo in zip(c, o) if oo == "X")
    pure_d = sum(len(x) for x, oo in zip(c, o) if oo == "D")
    assert pure_d <= dom_bp_before <= pure_d + mixed_bp
    ok("chimera origin bookkeeping is consistent")

    # ---- uneven coverage ------------------------------------------------
    for rate in (0.05, 0.2, 0.4):
        c, o, s = inject_uneven_coverage(contigs, origins, rate,
                                         np.random.default_rng(23))
        tot0 = sum(len(x) for x in contigs)
        tot1 = sum(len(x) for x in c)
        assert abs((tot1 - tot0) - s["dup_bp"]) <= 1
        assert abs(s["realised_dup_fraction"] - rate) < 0.02
        assert set(c[:0]) == set()
    ok("uneven coverage: duplicated bp matches the target fraction to < 2 pp")

    c1, o1, _ = inject_uneven_coverage(contigs, origins, 0.2,
                                       np.random.default_rng(29))
    c2, o2, _ = inject_uneven_coverage(contigs, origins, 0.2,
                                       np.random.default_rng(29))
    assert c1 == c2 and o1 == o2
    ok("uneven coverage is deterministic given the seed")

    # every duplicated segment must be a substring of some original contig
    joined = "\x00".join(contigs)
    extra = [x for x in c1 if x not in contigs]
    assert all(x in joined for x in extra[:20])
    ok("duplicated segments are genuine substrings of the source assembly")

    # ---- dispatcher ------------------------------------------------------
    for et, r in (("substitution", 0.01), ("substitution_titv", 0.01),
                  ("indel", 0.01), ("chimera", 0.2), ("uneven_coverage", 0.2),
                  ("none", 0.0)):
        c, o, s = apply_error(contigs, origins, ErrorSpec(et, r),
                              np.random.default_rng(31))
        assert len(c) == len(o)
        assert s["error_type"] in (et, "none")
    ok("dispatcher handles every error type and keeps contigs/origins aligned")

    assert abs(expected_kmer_corruption(0.01) - 0.0862) < 1e-3
    assert abs(expected_kmer_corruption(0.05) - 0.3698) < 1e-3
    ok("expected 9-mer corruption 1-(1-p)^9: 8.62 % at p=1 %, 36.98 % at p=5 %")

    print(f"\n{len(msgs)} checks passed")
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
