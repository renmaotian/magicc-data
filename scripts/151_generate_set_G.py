#!/usr/bin/env python3
"""
WS6.2 — Build **Set G**: the sequencing / assembly error-robustness benchmark.

(Protocol name: ``108_generate_set_G.py``; renumbered to 151 because 107-109 were
already taken.  Error processes live in ``scripts/150_error_injection_module.py``.)

DESIGN
------
Set G is a **paired dose-response** design.  For each of ``N_REFERENCES``
held-out reference genomes one *base assembly* is generated with the production
fragmentation + contamination machinery (``magicc/fragmentation.py``,
``magicc/contamination.py``), and every error arm is then applied to **that same
base assembly**.  Consequently:

  * the control arm and every error arm of a reference share an identical
    ground truth, an identical fragmentation realisation and identical
    contaminants, so the only thing that varies within a reference is the
    injected error;
  * degradation can be estimated *paired within reference*, which removes the
    between-genome variance that would otherwise swamp a 0.1 % effect;
  * cluster bootstrap over reference genomes is the correct interval, matching
    WS5.4 / R1-m16.

Arms per reference (24):

    control              rate 0                                       1
    substitution         0.1, 0.5, 1, 2, 5 %   (uniform model)         5
    substitution_titv    0.1, 0.5, 1, 2, 5 %   (Ti/Tv = 2.0)           5
    indel                0.1, 0.5, 1, 2, 5 %   (50 % ins / 50 % del)   5
    chimera              5, 10, 20, 40 % of contigs mis-joined         4
    uneven_coverage      5, 10, 20, 40 % of bp duplicated              4

Base-assembly factors are crossed over references (seeded permutation of the
5 x 5 grid): target completeness {60, 70, 80, 90, 100} %, target contamination
{0, 2, 5, 10, 20} %, fragmentation tier rotating over {high, medium, low}.
Contaminants are **cross-phylum** so that contamination detection starts from
its best-case regime - WS2/Set F already established that MAGICC is
distance-limited, and mixing that in here would confound the error axis.

Every sample satisfies the training-domain restriction of protocol section 4.4a
(contamination % <= completeness %) by construction: contamination <= 20 % and
completeness >= 50 %.  Any violation is reported separately, not silently kept.

GROUND TRUTH IS UNCHANGED BY ERROR INJECTION
--------------------------------------------
    true_completeness  = dominant-derived bp in the BASE assembly
                         / dominant full reference length x 100
    true_contamination = contaminant-derived bp in the BASE assembly
                         / dominant full reference length x 100

Substitutions change bases, not origins or lengths; chimeras re-partition
exactly the same bases; uneven-coverage duplication re-emits sequence already
present, so the set of represented reference positions is unchanged.  Indels are
the only length-changing process and are treated as a mis-rendering of DNA that
is present (they are balanced, and both the realised net bp change and an
indel-adjusted truth are stored per sample so the bound is auditable).  Two
sensitivity columns are emitted for exactly this purpose:
``true_completeness_indel_adjusted`` and ``true_contamination_dup_counted``.

LEAKAGE
-------
Dominants **and** contaminants come exclusively from the held-out test split
(``data/splits/test_finished_genomes.tsv``, a subset of
``data/splits/test_genomes.tsv``).  Finished genomes are used for the dominants
so that "full reference length" is exact.  Disjointness is *proven* by the
provenance audit in ``scripts/153_error_robustness_analysis.py`` with GCA<->GCF
cross-mapping, not asserted here.

OUTPUTS
-------
    data/benchmarks/set_G/fasta/<genome_id>.fasta
    data/benchmarks/set_G/metadata.tsv              (tool-facing labels)
    data/benchmarks/set_G/generation_metadata.tsv   (full provenance, WS7.7)
    data/benchmarks/set_G/design.tsv
    data/benchmarks/set_G/reference_selection.tsv
    data/benchmarks/set_G/generation_checkpoint.jsonl
    data/benchmarks/set_G/validation_report.json

USAGE
-----
    python scripts/151_generate_set_G.py --pilot                 # 6 refs -> set_G_pilot
    python scripts/151_generate_set_G.py --workers 10            # full Set G
    python scripts/151_generate_set_G.py --validate-only
    python scripts/151_generate_set_G.py --determinism-check 12  # delete + regenerate
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eim = _load_module(PROJECT_DIR / "scripts" / "150_error_injection_module.py",
                   "ws6_error_injection")

from magicc.contamination import generate_contaminated_sample  # noqa: E402
from magicc.fragmentation import read_fasta                     # noqa: E402

SET_NAME = "set_G"
SET_DIR = PROJECT_DIR / "data" / "benchmarks" / SET_NAME
FASTA_DIR = SET_DIR / "fasta"

SET_BASE = 9_600_000               # master seed for Set G
N_REFERENCES = 80
N_CONTAMINANT_GENOMES = 2

# --- the error arms --------------------------------------------------------
SUB_RATES = [0.001, 0.005, 0.01, 0.02, 0.05]        # 0.1, 0.5, 1, 2, 5 %
INDEL_RATES = [0.001, 0.005, 0.01, 0.02, 0.05]
CHIMERA_RATES = [0.05, 0.10, 0.20, 0.40]
COVERAGE_RATES = [0.05, 0.10, 0.20, 0.40]
TITV = 2.0

COMP_LADDER = [0.60, 0.70, 0.80, 0.90, 1.00]
CONT_LADDER = [0.0, 2.0, 5.0, 10.0, 20.0]
# All four production tiers.  The tier drives the base assembly's contig count,
# which sets the achievable dose on the chimera axis: restricting to the two
# least-fragmented tiers gives a median of ~6 contigs and therefore ~1 mis-join
# even at a 40 % rate, which is not a real test.  Rotating over all four gives a
# realistic MAG contig distribution (comparable with Set E and Set F) and a
# properly dosed chimera arm on the fragmented half of the panel.
TIERS = ["high", "medium", "low", "highly_fragmented"]

MIN_GENOME_BP = 500_000
MAX_GENOME_BP = 10_000_000


def build_arms() -> List[dict]:
    """The 24 error arms, in a fixed order (arm_index is part of the seed)."""
    arms = [{"error_type": "none", "rate": 0.0, "model": "control"}]
    for r in SUB_RATES:
        arms.append({"error_type": "substitution", "rate": r, "model": "uniform"})
    for r in SUB_RATES:
        arms.append({"error_type": "substitution_titv", "rate": r,
                     "model": f"titv_{TITV:g}"})
    for r in INDEL_RATES:
        arms.append({"error_type": "indel", "rate": r, "model": "ins0.5_geom0.75"})
    for r in CHIMERA_RATES:
        arms.append({"error_type": "chimera", "rate": r, "model": "rc0.5"})
    for r in COVERAGE_RATES:
        arms.append({"error_type": "uneven_coverage", "rate": r,
                     "model": "lognormal_sigma1.0"})
    for i, a in enumerate(arms):
        a["arm_index"] = i
    return arms


ARMS = build_arms()


# ==========================================================================
# reference selection
# ==========================================================================
def load_finished_test() -> pd.DataFrame:
    t = pd.read_csv(PROJECT_DIR / "data/splits/test_finished_genomes.tsv", sep="\t")
    t["fasta_path"] = t["fasta_path"].str.replace(
        "/path/to/home/projects/magicc2", str(PROJECT_DIR), regex=False)
    t = t[(t.genome_size >= MIN_GENOME_BP) & (t.genome_size <= MAX_GENOME_BP)]
    t = t[t.fasta_path.map(os.path.exists)]
    return t.sort_values("gtdb_accession").reset_index(drop=True)


def pick_references(t: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Round-robin over phyla so the panel is as phylogenetically broad as possible."""
    phyla = sorted(t["phylum"].unique())
    by_p = {p: np.array(t.index[t.phylum == p].to_numpy(), copy=True) for p in phyla}
    for p in phyla:
        rng.shuffle(by_p[p])
    order = np.argsort([-len(by_p[p]) for p in phyla], kind="stable")
    phyla = [phyla[i] for i in order]
    picked, k = [], 0
    while len(picked) < n:
        added = False
        for p in phyla:
            if k < len(by_p[p]):
                picked.append(by_p[p][k])
                added = True
                if len(picked) == n:
                    break
        if not added:
            break
        k += 1
    return t.loc[picked].reset_index(drop=True)


def pick_contaminants(t: pd.DataFrame, ref: pd.Series, k: int,
                      rng: np.random.Generator) -> List[str]:
    """k cross-phylum contaminant genomes from the same held-out test split."""
    cand = t.index[t["phylum"] != ref["phylum"]].to_numpy()
    if cand.size == 0:
        return []
    sel = rng.choice(cand, size=min(k, cand.size), replace=False)
    return [t.loc[i, "gtdb_accession"] for i in sel]


def build_design(t: pd.DataFrame, n_refs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SET_BASE)
    refs = pick_references(t, n_refs, rng)

    combos = list(product(COMP_LADDER, CONT_LADDER))          # 25 cells
    grid = (combos * int(np.ceil(n_refs / len(combos))))[:n_refs]
    idx = rng.permutation(n_refs)
    grid = [grid[i] for i in idx]
    tiers = (TIERS * int(np.ceil(n_refs / len(TIERS))))[:n_refs]
    tiers = [tiers[i] for i in rng.permutation(n_refs)]

    refs["ref_index"] = np.arange(len(refs))
    refs["target_completeness"] = [g[0] for g in grid]
    refs["target_contamination"] = [g[1] for g in grid]
    refs["quality_tier"] = tiers

    rows = []
    for _, r in refs.iterrows():
        crng = np.random.default_rng(SET_BASE + 400_000 + int(r.ref_index))
        cons = pick_contaminants(t, r, N_CONTAMINANT_GENOMES, crng)
        refs.loc[refs.ref_index == r.ref_index, "contaminant_accessions"] = ";".join(cons)
        for a in ARMS:
            rate_tag = ("control" if a["error_type"] == "none"
                        else f"{a['rate'] * 100:g}pct".replace(".", "p"))
            rows.append({
                "genome_id": f"G_r{int(r.ref_index):03d}_{a['error_type']}_{rate_tag}",
                "ref_index": int(r.ref_index),
                "arm_index": int(a["arm_index"]),
                "error_type": a["error_type"],
                "error_rate": float(a["rate"]),
                "error_rate_pct": float(a["rate"] * 100.0),
                "error_model": a["model"],
                "dominant_accession": r.gtdb_accession,
                "contaminant_accessions": ";".join(cons),
                "target_completeness": float(r.target_completeness),
                "target_contamination": float(r.target_contamination),
                "quality_tier": r.quality_tier,
            })
    design = pd.DataFrame(rows)
    keep = ["gtdb_accession", "ncbi_accession", "gcf_accession", "ref_index",
            "target_completeness", "target_contamination", "quality_tier",
            "contaminant_accessions", "genome_size", "contig_count", "n50_contigs",
            "checkm2_completeness", "checkm2_contamination", "domain", "phylum",
            "gtdb_taxonomy", "ncbi_assembly_level", "fasta_path"]
    return design, refs[[k for k in keep if k in refs.columns]]


# ==========================================================================
# generation
# ==========================================================================
_PATHS: Dict[str, str] = {}


def _init_paths(paths: Dict[str, str]):
    global _PATHS
    _PATHS = paths
    eim.warm_numba()


def write_fasta(contigs: Sequence[str], path: Path, prefix: str) -> Tuple[int, int]:
    tmp = str(path) + ".tmp"
    n_bp = 0
    with open(tmp, "w") as fh:
        for i, c in enumerate(contigs):
            fh.write(f">{prefix}_{i + 1} length={len(c)}\n")
            n_bp += len(c)
            for j in range(0, len(c), 80):
                fh.write(c[j:j + 80] + "\n")
    os.replace(tmp, path)
    return len(contigs), n_bp


def _generate_reference(task):
    """One worker task == one reference == its 24 arms (shared base assembly)."""
    ref_index, dom_acc, rows, fasta_dir = task
    fasta_dir = Path(fasta_dir)
    out: List[dict] = []
    try:
        r0 = rows[0]
        cons = [c for c in str(r0["contaminant_accessions"]).split(";") if c]
        dom_seq = read_fasta(_PATHS[dom_acc])
        con_seqs = [read_fasta(_PATHS[c]) for c in cons]
        dom_full = len(dom_seq)

        seed_base = SET_BASE + int(ref_index)
        seed_shuffle = SET_BASE + 2_000_000 + int(ref_index)

        base = generate_contaminated_sample(
            dominant_sequence=dom_seq,
            contaminant_sequences=con_seqs if float(r0["target_contamination"]) > 0 else [],
            target_completeness=float(r0["target_completeness"]),
            target_contamination=float(r0["target_contamination"]),
            rng=np.random.default_rng(seed_base),
            dominant_quality_tier=r0["quality_tier"],
            contaminant_quality_tier=r0["quality_tier"],
        )
        dom_contigs = base["dominant_contigs"]
        con_contigs = base["contaminant_contigs"]
        dom_bp = int(sum(len(c) for c in dom_contigs))
        con_bp = int(sum(len(c) for c in con_contigs))

        # TRUTH — computed directly from bp bookkeeping, in MAGICC's denominator.
        true_comp = 100.0 * dom_bp / dom_full
        true_cont = 100.0 * con_bp / dom_full

        base_contigs = list(dom_contigs) + list(con_contigs)
        base_origins = ["D"] * len(dom_contigs) + ["C"] * len(con_contigs)
        order = np.random.default_rng(seed_shuffle).permutation(len(base_contigs))
        base_contigs = [base_contigs[i] for i in order]
        base_origins = [base_origins[i] for i in order]
        base_bp = int(sum(len(c) for c in base_contigs))

        del dom_seq, con_seqs

        for row in rows:
            gid = row["genome_id"]
            arm = int(row["arm_index"])
            et = row["error_type"]
            rate = float(row["error_rate"])
            seed_err = SET_BASE + 1_000_000 + 1000 * int(ref_index) + arm

            params = {}
            if et == "substitution_titv":
                params["titv"] = TITV
            spec = eim.ErrorSpec(error_type=et, rate=rate, params=params)
            contigs, origins, stats = eim.apply_error(
                base_contigs, base_origins, spec, np.random.default_rng(seed_err))

            n_contigs, total_len = write_fasta(contigs, fasta_dir / f"{gid}.fasta", gid)

            dom_bp_after = int(sum(len(c) for c, o in zip(contigs, origins) if o == "D"))
            mix_bp_after = int(sum(len(c) for c, o in zip(contigs, origins) if o == "X"))
            con_bp_after = total_len - dom_bp_after - mix_bp_after
            dup_bp = int(stats.get("dup_bp", 0))
            net_indel = int(stats.get("net_bp_change", 0))

            rec = dict(row)
            rec.update({
                "seed_base": seed_base, "seed_shuffle": seed_shuffle,
                "seed_error": seed_err,
                "dominant_reference_bp": dom_full,
                "dominant_bp": dom_bp,
                "contaminant_bp": con_bp,
                "true_completeness": true_comp,
                "true_contamination": true_cont,
                # sensitivity accountings (see module docstring)
                "true_completeness_indel_adjusted":
                    100.0 * (dom_bp + net_indel * (dom_bp / base_bp if base_bp else 0.0))
                    / dom_full if et == "indel" else true_comp,
                "true_contamination_dup_counted":
                    100.0 * (con_bp + dup_bp) / dom_full if dup_bp else true_cont,
                "base_n_contigs": len(base_contigs),
                "base_total_length": base_bp,
                "n_contigs": n_contigs,
                "total_length": total_len,
                "assembly_bp_delta": total_len - base_bp,
                "bp_dominant_after": dom_bp_after,
                "bp_contaminant_after": con_bp_after,
                "bp_mixed_contigs_after": mix_bp_after,
                "dup_bp": dup_bp,
                "expected_kmer_corruption":
                    eim.expected_kmer_corruption(rate)
                    if et.startswith("substitution") else float("nan"),
                # per-error realised statistics
                "n_substitutions": int(stats.get("n_substitutions", 0)),
                "n_transitions": int(stats.get("n_transitions", 0)),
                "n_transversions": int(stats.get("n_transversions", 0)),
                "realised_titv": float(stats.get("realised_titv", float("nan"))),
                "n_indel_events": int(stats.get("n_indel_events", 0)),
                "n_insertions": int(stats.get("n_insertions", 0)),
                "n_deletions": int(stats.get("n_deletions", 0)),
                "inserted_bp": int(stats.get("inserted_bp", 0)),
                "deleted_bp": int(stats.get("deleted_bp", 0)),
                "indels_per_kb": float(stats.get("indels_per_kb", 0.0)),
                "n_chimera_events": int(stats.get("n_chimera_events", 0)),
                "n_cross_origin_joins": int(stats.get("n_cross_origin_joins", 0)),
                "n_duplicated_segments": int(stats.get("n_duplicated_segments", 0)),
                "realised_error_rate": float(
                    stats.get("realised_rate",
                              stats.get("realised_dup_fraction",
                                        stats.get(
                                            "realised_contig_fraction_in_chimeras",
                                            0.0)))),
                "error": "",
            })
            out.append(rec)
        return ref_index, out, None
    except Exception:                                            # noqa: BLE001
        import traceback
        return ref_index, out, traceback.format_exc(limit=8)


# ==========================================================================
# metadata + validation
# ==========================================================================
def write_metadata(gm: pd.DataFrame, refs: pd.DataFrame, set_dir: Path) -> None:
    rmap = refs.set_index("gtdb_accession")
    gm = gm.copy()
    gm["dominant_phylum"] = gm.dominant_accession.map(rmap["phylum"])
    gm["dominant_domain"] = gm.dominant_accession.map(rmap["domain"])
    gm["dominant_taxonomy"] = gm.dominant_accession.map(rmap["gtdb_taxonomy"])
    gm["dominant_source_split"] = "test"
    gm["contaminant_source_split"] = "test"
    gm["contaminant_relatedness"] = np.where(gm.contaminant_bp > 0,
                                             "cross_phylum", "none (uncontaminated)")
    gm["set"] = set_dir.name
    gm["sample_type"] = "set_g_" + gm.error_type.astype(str)
    gm["replicate"] = gm["arm_index"]
    gm["in_training_domain"] = gm.true_contamination <= gm.true_completeness + 1e-9
    gm["constraint_violation"] = ~gm["in_training_domain"]
    gm = gm.sort_values(["ref_index", "arm_index"]).reset_index(drop=True)
    gm.to_csv(set_dir / "generation_metadata.tsv", sep="\t", index=False)

    meta = gm[["genome_id", "true_completeness", "true_contamination",
               "dominant_accession", "dominant_phylum", "sample_type",
               "error_type", "error_rate", "error_rate_pct", "error_model",
               "n_contigs", "total_length", "ref_index", "replicate"]].copy()
    meta.to_csv(set_dir / "metadata.tsv", sep="\t", index=False)
    print(f"\nwrote {set_dir / 'metadata.tsv'} and generation_metadata.tsv "
          f"({len(gm)} rows, {gm.shape[1]} provenance columns)")


def _fasta_counts(path: Path) -> Tuple[int, int]:
    nc = bp = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                nc += 1
            else:
                bp += len(line) - 1 if line.endswith("\n") else len(line)
    return nc, bp


def validate(set_dir: Path) -> int:
    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    design = pd.read_csv(set_dir / "design.tsv", sep="\t")
    rep: Dict[str, object] = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                              "set_dir": str(set_dir), "n_samples": int(len(gm))}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        rep[name] = {"pass": bool(cond), "detail": detail}
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
        if not cond:
            ok = False

    check("all_design_rows_generated", len(gm) == len(design), f"{len(gm)}/{len(design)}")
    check("unique_genome_ids", gm.genome_id.nunique() == len(gm), f"{gm.genome_id.nunique()}")

    n_bad_ct = n_bad_bp = n_missing = 0
    for r in gm.itertuples():
        p = set_dir / "fasta" / f"{r.genome_id}.fasta"
        if not p.exists():
            n_missing += 1
            continue
        nc, bp = _fasta_counts(p)
        n_bad_ct += int(nc != r.n_contigs)
        n_bad_bp += int(bp != r.total_length)
    check("fasta_present", n_missing == 0, f"{n_missing} missing")
    check("fasta_contig_counts_match", n_bad_ct == 0, f"{n_bad_ct} mismatched")
    check("fasta_bp_match", n_bad_bp == 0, f"{n_bad_bp} mismatched")

    viol = int(gm.constraint_violation.sum())
    check("training_domain_contamination_le_completeness", viol == 0,
          f"{viol} out-of-domain samples")
    check("completeness_within_50_100",
          bool(((gm.true_completeness >= 50) & (gm.true_completeness <= 100)).all()),
          f"min {gm.true_completeness.min():.2f}, max {gm.true_completeness.max():.2f}")

    # --- TRUTH INVARIANCE: the whole point of the design --------------------
    g = gm.groupby("ref_index")
    n_var_comp = int((g.true_completeness.nunique() > 1).sum())
    n_var_cont = int((g.true_contamination.nunique() > 1).sum())
    check("truth_identical_across_all_arms_of_a_reference",
          n_var_comp == 0 and n_var_cont == 0,
          f"{n_var_comp} refs with varying completeness, {n_var_cont} with varying "
          f"contamination")

    # --- bp bookkeeping per error type --------------------------------------
    ctrl = gm[gm.error_type == "none"].set_index("ref_index")
    sub = gm[gm.error_type.str.startswith("substitution")]
    bad = 0
    for r in sub.itertuples():
        c = ctrl.loc[r.ref_index]
        bad += int(r.total_length != c.total_length or r.n_contigs != c.n_contigs)
    check("substitutions_preserve_length_and_contig_count", bad == 0,
          f"{bad}/{len(sub)} altered")

    chim = gm[gm.error_type == "chimera"]
    bad = 0
    for r in chim.itertuples():
        c = ctrl.loc[r.ref_index]
        bad += int(r.total_length != c.total_length)
    check("chimeras_preserve_total_bp_exactly", bad == 0, f"{bad}/{len(chim)} altered")
    if len(chim):
        # The dose is stochastically rounded (a 12-contig assembly at 5 % gets
        # 0.3 mis-joins in expectation), so the arm is validated in aggregate.
        agg = chim.groupby("error_rate").apply(
            lambda d: (2 * d.n_chimera_events).sum() / d.base_n_contigs.sum(),
            include_groups=False)
        dev = (agg - agg.index.values).abs()
        check("chimera_dose_unbiased_per_arm", float(dev.max()) < 0.02,
              "realised contig fraction per arm: "
              + ", ".join(f"{100 * r:g}%->{100 * v:.2f}%" for r, v in agg.items()))
        check("chimeras_reduce_contig_count",
              bool((chim.n_chimera_events.sum() > 0)),
              f"{int(chim.n_chimera_events.sum())} mis-joins total, median "
              f"{chim.n_chimera_events.median():.0f} per sample")

    cov = gm[gm.error_type == "uneven_coverage"]
    if len(cov):
        dev = (cov.dup_bp / cov.base_total_length - cov.error_rate).abs()
        check("uneven_coverage_hits_target_duplication",
              float(dev.max()) < 0.02, f"max deviation {100 * dev.max():.2f} pp")

    ind = gm[gm.error_type == "indel"]
    if len(ind):
        rel = (ind.assembly_bp_delta.abs() / ind.base_total_length)
        check("indel_net_length_change_below_1pct", float(rel.max()) < 0.01,
              f"max |net bp change| {100 * rel.max():.3f} % of assembly "
              f"(median {100 * rel.median():.4f} %)")
        radj = (ind.true_completeness_indel_adjusted - ind.true_completeness).abs()
        rep["indel_truth_sensitivity_pp"] = {
            "max": float(radj.max()), "median": float(radj.median())}
        print(f"  [INFO] indel truth sensitivity: max {radj.max():.4f} pp, "
              f"median {radj.median():.4f} pp (completeness)")

    # Realised vs target error rate.  The process is i.i.d. Bernoulli per base, so
    # the realised count is Binomial(n_bp, rate) and the *relative* deviation is
    # necessarily large at the lowest rate on the smallest genomes (rate 0.1 % on
    # 0.5 Mbp gives ~500 events, relative SD 4.5 %).  The correct check is
    # therefore a z-score against the binomial SD, plus a pooled rate check where
    # the sampling error is negligible.
    for et in ("substitution", "substitution_titv", "indel"):
        s = gm[gm.error_type == et]
        if not len(s):
            continue
        n_bp = s.base_total_length.values.astype(float)
        p = s.error_rate.values.astype(float)
        n_obs = s.realised_error_rate.values * n_bp
        z = (n_obs - n_bp * p) / np.sqrt(n_bp * p * (1 - p))
        pooled = float(n_obs.sum() / (n_bp * p).sum())
        check(f"{et}_realised_rate_matches_binomial",
              float(np.abs(z).max()) < 5.0,
              f"max |z| {np.abs(z).max():.2f} (binomial SD units), pooled "
              f"realised/target = {pooled:.5f}, max relative deviation "
              f"{100 * ((s.realised_error_rate - s.error_rate).abs() / s.error_rate).max():.2f} %")
        check(f"{et}_pooled_rate_within_0p5pct_of_target", abs(pooled - 1.0) < 0.005,
              f"pooled realised/target = {pooled:.5f}")

    titv = gm[gm.error_type == "substitution_titv"]
    if len(titv):
        # Per-sample Ti/Tv is noisy at 0.1 % (few hundred substitutions), so the
        # model is validated on the pooled ratio, where the sampling error is
        # negligible, and per-sample only where n_substitutions is large.
        pooled = titv.n_transitions.sum() / max(1, titv.n_transversions.sum())
        big = titv[titv.n_substitutions >= 20_000]
        check("titv_model_reproduces_ratio_pooled", abs(pooled - TITV) < 0.02,
              f"pooled realised Ti/Tv {pooled:.4f} for target {TITV} "
              f"({int(titv.n_transitions.sum()):,} Ti / "
              f"{int(titv.n_transversions.sum()):,} Tv)")
        if len(big):
            # multinomial sampling error on the ratio, same z-score convention as
            # the rate checks above
            zt = ((big.realised_titv - TITV)
                  / (TITV * np.sqrt(1 / big.n_transitions + 1 / big.n_transversions)))
            check("titv_per_sample_ratio_large_n",
                  float(np.abs(zt).max()) < 5.0,
                  f"n={len(big)} samples with >=20,000 substitutions, realised "
                  f"{big.realised_titv.min():.3f}-{big.realised_titv.max():.3f}, "
                  f"max |z| {np.abs(zt).max():.2f} (multinomial SD units)")
        uni = gm[gm.error_type == "substitution"]
        upool = uni.n_transitions.sum() / max(1, uni.n_transversions.sum())
        check("uniform_model_has_titv_half", abs(upool - 0.5) < 0.02,
              f"pooled Ti/Tv {upool:.4f} for the uniform model (expected 0.5)")

    # leakage: every accession used must be in the held-out test split
    test_acc = set(pd.read_csv(PROJECT_DIR / "data/splits/test_genomes.tsv",
                               sep="\t")["gtdb_accession"])
    used = set(gm.dominant_accession)
    for s in gm.contaminant_accessions.fillna(""):
        used.update(x for x in str(s).split(";") if x)
    outside = sorted(used - test_acc)
    check("all_accessions_from_test_split", len(outside) == 0,
          f"{len(outside)} outside test split")

    rep["overall_pass"] = ok
    rep["composition"] = {
        "by_error_type": gm.error_type.value_counts().to_dict(),
        "by_error_type_rate": {f"{a}|{b:g}": int(c) for (a, b), c in
                               gm.groupby(["error_type", "error_rate_pct"]).size().items()},
        "n_references": int(gm.ref_index.nunique()),
        "n_phyla": int(gm.dominant_phylum.nunique()),
        "true_completeness": {"mean": float(gm.true_completeness.mean()),
                              "min": float(gm.true_completeness.min()),
                              "max": float(gm.true_completeness.max())},
        "true_contamination": {"mean": float(gm.true_contamination.mean()),
                               "min": float(gm.true_contamination.min()),
                               "max": float(gm.true_contamination.max())},
        "assembly_bp_total": int(gm.total_length.sum()),
    }
    (set_dir / "validation_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nwrote {set_dir / 'validation_report.json'}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ==========================================================================
# determinism check
# ==========================================================================
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def determinism_check(set_dir: Path, n: int, workers: int) -> int:
    """Delete n random outputs, regenerate, and require byte-identical files."""
    print("=" * 78)
    print(f"DETERMINISM CHECK — {n} samples")
    print("=" * 78)
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    design = pd.read_csv(set_dir / "design.tsv", sep="\t")
    refs = pd.read_csv(set_dir / "reference_selection.tsv", sep="\t")
    rng = random.Random(4242)
    victims = rng.sample(list(gm.genome_id), min(n, len(gm)))
    before = {g: _sha(set_dir / "fasta" / f"{g}.fasta") for g in victims}
    for g in victims:
        (set_dir / "fasta" / f"{g}.fasta").unlink()
    print(f"  deleted {len(victims)} FASTA files across "
          f"{gm[gm.genome_id.isin(victims)].ref_index.nunique()} references")

    paths = dict(zip(refs.gtdb_accession, refs.fasta_path))
    t = load_finished_test()
    paths.update(dict(zip(t.gtdb_accession, t.fasta_path)))

    ridx = sorted(set(gm.loc[gm.genome_id.isin(victims), "ref_index"]))
    tasks = []
    for ri in ridx:
        sub = design[design.ref_index == ri]
        rows = [r for r in sub.to_dict("records") if r["genome_id"] in set(victims)]
        tasks.append((int(ri), sub.iloc[0]["dominant_accession"], rows, str(FASTA_DIR)))
    with Pool(min(workers, len(tasks)), initializer=_init_paths,
              initargs=(paths,)) as pool:
        for ri, recs, err in pool.imap_unordered(_generate_reference, tasks):
            if err:
                print(f"  ref {ri} FAILED: {err.splitlines()[-1]}")

    n_ok = n_bad = 0
    bad = []
    for g, h in before.items():
        p = set_dir / "fasta" / f"{g}.fasta"
        if p.exists() and _sha(p) == h:
            n_ok += 1
        else:
            n_bad += 1
            bad.append(g)
    print(f"\n  byte-identical: {n_ok}/{len(before)}")
    if bad:
        print(f"  MISMATCHED: {bad[:5]}")
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_checked": len(before), "n_identical": n_ok, "n_mismatched": n_bad,
           "mismatched": bad,
           "verdict": "PASS" if n_bad == 0 else "FAIL"}
    (set_dir / "determinism_check.json").write_text(json.dumps(out, indent=2))
    print("VERDICT:", out["verdict"])
    return 0 if n_bad == 0 else 1


# ==========================================================================
# main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pilot", action="store_true",
                    help="first 6 references only, into set_G_pilot (identical seeds, "
                         "so the FASTAs are byte-identical to the full run)")
    ap.add_argument("--max-refs", type=int, default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--determinism-check", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global SET_DIR, FASTA_DIR
    n_refs = args.max_refs or N_REFERENCES
    if args.pilot:
        SET_DIR = PROJECT_DIR / "data" / "benchmarks" / "set_G_pilot"
        n_refs = args.max_refs or 6
    FASTA_DIR = SET_DIR / "fasta"
    for d in (SET_DIR, FASTA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.determinism_check:
        return determinism_check(SET_DIR, args.determinism_check, args.workers)

    print("=" * 78)
    print(f"WS6.2 — Set G generation  ->  {SET_DIR}")
    print(f"  {n_refs} references x {len(ARMS)} error arms = {n_refs * len(ARMS)} samples")
    print("=" * 78)

    t = load_finished_test()
    # NOTE: the design is built for the FULL panel and then truncated, so the
    # pilot's samples are byte-identical to the corresponding full-run samples.
    design_full, refs_full = build_design(t, N_REFERENCES)
    refs = refs_full[refs_full.ref_index < n_refs].reset_index(drop=True)
    design = design_full[design_full.ref_index < n_refs].reset_index(drop=True)
    design.to_csv(SET_DIR / "design.tsv", sep="\t", index=False)
    refs.to_csv(SET_DIR / "reference_selection.tsv", sep="\t", index=False)
    print(f"design: {len(design)} samples, {len(refs)} references")
    print(f"  references span {refs.phylum.nunique()} phyla, "
          f"{refs.domain.value_counts().to_dict()}")
    print(f"  genome size Mbp: median {refs.genome_size.median() / 1e6:.2f} "
          f"[{refs.genome_size.min() / 1e6:.2f}, {refs.genome_size.max() / 1e6:.2f}]")
    print(f"  target completeness {sorted(set(refs.target_completeness))}, "
          f"contamination {sorted(set(refs.target_contamination))}")

    paths = dict(zip(t.gtdb_accession, t.fasta_path))
    needed = set(design.dominant_accession)
    for s in design.contaminant_accessions:
        needed.update(x for x in str(s).split(";") if x)
    missing = [a for a in needed if a not in paths or not os.path.exists(paths[a])]
    if missing:
        print(f"FATAL: {len(missing)} genome FASTAs missing, e.g. {missing[:3]}",
              file=sys.stderr)
        return 1
    paths = {a: paths[a] for a in needed}
    print(f"  unique source genomes: {len(paths)}")

    if not args.validate_only:
        ckpt = SET_DIR / "generation_checkpoint.jsonl"
        done: Dict[str, dict] = {}
        if ckpt.exists() and not args.force:
            with open(ckpt) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (FASTA_DIR / f"{r['genome_id']}.fasta").exists():
                        done[r["genome_id"]] = r
        todo_refs = sorted(set(design.loc[~design.genome_id.isin(done), "ref_index"]))
        print(f"\ngeneration: {len(done)} samples cached, "
              f"{len(todo_refs)} references to build")

        if todo_refs:
            tasks = []
            for ri in todo_refs:
                sub = design[design.ref_index == ri]
                rows = [r for r in sub.to_dict("records") if r["genome_id"] not in done]
                tasks.append((int(ri), sub.iloc[0]["dominant_accession"], rows,
                              str(FASTA_DIR)))
            t0 = time.time()
            fails = []
            with open(ckpt, "a") as fh, \
                    Pool(min(args.workers, len(tasks)), initializer=_init_paths,
                         initargs=(paths,)) as pool:
                for i, (ri, recs, err) in enumerate(
                        pool.imap_unordered(_generate_reference, tasks), 1):
                    for r in recs:
                        done[r["genome_id"]] = r
                        fh.write(json.dumps(r, default=str) + "\n")
                    fh.flush()
                    if err:
                        fails.append((ri, err))
                    if i % 5 == 0 or i == len(tasks):
                        print(f"    {i}/{len(tasks)} refs  ({len(done)} samples, "
                              f"{time.time() - t0:.0f}s)", flush=True)
            print(f"    generation done in {time.time() - t0:.0f}s; "
                  f"{len(fails)} reference failures")
            for ri, err in fails[:3]:
                print(f"      ref {ri}: {err.splitlines()[-1][:200]}")

        gm = pd.DataFrame([done[g] for g in design.genome_id if g in done])
        write_metadata(gm, refs, SET_DIR)

    return validate(SET_DIR)


if __name__ == "__main__":
    sys.exit(main())
