#!/usr/bin/env python3
"""
WS6.3 + WS6.4 — Set G degradation curves and the usable-error-regime boundary.

(Protocol name: ``109_error_robustness_analysis.py``; renumbered to 153.)

WHAT THIS SCRIPT DOES
---------------------
1. Runs frozen MAGICC V5 on Set G by direct ONNX inference (pattern of
   ``scripts/75_run_magicc_clean_cd.py``; the CLI is deliberately not used).
2. Loads CheckM2 / CoCoPyE / DeepCheck predictions produced by
   ``scripts/152_run_tools_set_G.sh`` -> ``scripts/91_parse_competitor_clean_cd.py``.
3. Runs a provenance audit on Set G in the style of
   ``scripts/74_provenance_audit.py`` (GCA<->GCF cross-mapped), emitting
   accession lists and a SHA256 manifest.  Disjointness is proven, not asserted.
4. **Degradation curves** (WS6.3): MAE, signed bias, R^2 and MIMAG-inspired
   classification versus error rate, per error type and per tool, with 95 %
   cluster-bootstrap CIs over reference genomes.
5. **Paired degradation** relative to each reference's own error-free control -
   the estimator that actually isolates the error effect, because control and
   arm share an identical ground truth, fragmentation realisation and
   contaminants.
6. **Head-to-head MAGICC vs CheckM2 at every rate**, with the crossover rate
   (the substitution rate at which MAGICC stops being at least as accurate)
   reported explicitly whether or not it favours MAGICC.
7. **WS6.4 applicability boundary**: the largest error rate at which the paired
   degradation is bounded, stated against real Illumina / assembly-consensus
   error rates.

CONVENTIONS (binding, protocol sections 4.4d / 5.4 / 9.13)
----------------------------------------------------------
* R^2 is ALWAYS the coefficient of determination, 1 - SS_res/SS_tot, and is
  omitted (emitted blank) wherever the truth has (near-)zero variance (R1-m19).
* Signed error = predicted - true.  Negative = UNDER-estimate.
* Two-sided paired tests; cluster bootstrap over reference genomes with
  ``fw.stable_hash()`` (CRC-32) seeds and PYTHONHASHSEED=0; Benjamini-Hochberg
  FDR; Hodges-Lehmann and Cliff's delta with CIs beside every p-value.
* MIMAG-inspired: HQ >= 90 % completeness AND < 5 % contamination;
  MQ >= 50 % AND < 10 %.
* Figures use the project's validated CVD-safe palette (no red/green
  discrimination), verified in-script; the denominator is stated in every
  caption.

USAGE
-----
    python scripts/153_error_robustness_analysis.py
    python scripts/153_error_robustness_analysis.py --set set_G_pilot --skip-provenance
    python scripts/153_error_robustness_analysis.py --magicc-only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONHASHSEED", "0")
signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load(PROJECT_DIR / "scripts" / "101_metrics_framework.py",
           "magicc_metrics_framework")

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
from scipy import stats as sps   # noqa: E402

DATA_DIR = PROJECT_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
RESULTS_DIR = PROJECT_DIR / "results" / "revision" / "set_G"
FIG_DIR = RESULTS_DIR / "figures"

SELECTED_KMERS = str(DATA_DIR / "kmer_selection" / "selected_kmers.txt")
NORMALIZATION = str(DATA_DIR / "features" / "normalization_params.json")
ONNX_MODEL = str(PROJECT_DIR / "models" / "magicc_v5.onnx")

N_BOOT = 2000
ERROR_ORDER = ["substitution", "substitution_titv", "indel", "chimera",
               "uneven_coverage"]
ERROR_LABEL = {
    "substitution": "Substitutions (uniform)",
    "substitution_titv": "Substitutions (Ti/Tv = 2)",
    "indel": "Indels (50 % ins / 50 % del)",
    "chimera": "Chimeric mis-joins",
    "uneven_coverage": "Uneven coverage (duplication)",
}
ERROR_XLABEL = {
    "substitution": "per-base substitution rate (%)",
    "substitution_titv": "per-base substitution rate (%)",
    "indel": "per-base indel rate (%)",
    "chimera": "fraction of contigs mis-joined (%)",
    "uneven_coverage": "fraction of assembly bp duplicated (%)",
}
LOG_X = {"substitution", "substitution_titv", "indel"}

# Validated CVD-safe palette already in production use by scripts/105 and 147
# (plain Okabe-Ito failed its own CVD check on this project; this is the
# verified replacement subset).
PALETTE = {
    "magicc_v5": "#0072B2",   # blue
    "checkm2":   "#D55E00",   # vermillion
    "cocopye":   "#F0E442",   # yellow
    "deepcheck": "#000000",   # black
}
MARKERS = {"magicc_v5": "o", "checkm2": "^", "cocopye": "D", "deepcheck": "v"}
TOOL_LABEL = {"magicc_v5": "MAGICC V5", "checkm2": "CheckM2 1.0.1",
              "cocopye": "CoCoPyE 0.5.0", "deepcheck": "DeepCheck"}
TOOL_ORDER = ["magicc_v5", "checkm2", "cocopye", "deepcheck"]

DENOM_NOTE = ("Denominators: completeness (%) = dominant-derived bp / dominant full "
              "reference length x 100; contamination (%) = contaminant-derived bp / "
              "dominant full reference length x 100. Both percentages use the same "
              "denominator and are independent. Ground truth is that of the "
              "error-free base assembly and is unchanged by error injection "
              "(substitutions and chimeras preserve it exactly; duplication "
              "re-emits sequence already present; indels are balanced and their "
              "residual effect on truth is < 0.05 pp, recorded per sample).")

# real-world context for WS6.4 (documented in the boundary table)
REAL_WORLD_RATES = [
    ("Illumina raw read, per base (Q30 nominal)", 0.001, "typical modern Illumina"),
    ("Illumina raw read, per base (Q20 tail)", 0.01, "poor-quality read region"),
    ("Nanopore R10.4 raw read, per base", 0.02, "modern simplex nanopore"),
    ("Short-read assembly consensus, per base", 1e-5, "SPAdes/MEGAHIT consensus"),
    ("Long-read assembly consensus, polished", 1e-4, "HiFi / polished ONT consensus"),
    ("Long-read assembly consensus, unpolished ONT", 1e-3, "older ONT consensus"),
]


# ==========================================================================
# MAGICC V5 inference (direct ONNX)
# ==========================================================================
_kc = None


def _init_worker(kmers_path):
    global _kc
    from magicc.kmer_counter import KmerCounter
    _kc = KmerCounter(kmers_path)
    _kc.count_contigs(["ACGTACGTACGTACGTACGT" * 50])


def _read_contigs(path):
    contigs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    contigs.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        contigs.append("".join(cur).upper())
    return [c for c in contigs if c]


def _feat_worker(args):
    from magicc.assembly_stats import compute_assembly_stats
    idx, path = args
    try:
        contigs = _read_contigs(path)
        if not contigs:
            return idx, None, None, "no_contigs"
        kc = _kc.count_contigs(contigs)
        asm = compute_assembly_stats(_kc.total_kmer_count(kc), kc)
        return idx, kc.astype(np.float32), asm.astype(np.float32), None
    except Exception as e:                                        # noqa: BLE001
        return idx, None, None, str(e)


def run_magicc(set_dir: Path, workers: int, force: bool = False) -> pd.DataFrame:
    out = set_dir / "magicc_v5_predictions.tsv"
    meta = pd.read_csv(set_dir / "metadata.tsv", sep="\t")
    if out.exists() and not force:
        p = pd.read_csv(out, sep="\t")
        if len(p) == len(meta) and set(p.genome_id) == set(meta.genome_id):
            print(f"  MAGICC V5: reusing {len(p)} cached predictions")
            return p

    import onnxruntime as ort
    from magicc.normalization import FeatureNormalizer

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL, so, providers=["CPUExecutionProvider"])
    normalizer = FeatureNormalizer.load(NORMALIZATION)

    work = [(i, str(set_dir / "fasta" / f"{g}.fasta"))
            for i, g in enumerate(meta.genome_id)
            if (set_dir / "fasta" / f"{g}.fasta").exists()]
    print(f"  MAGICC V5 on {len(work)} genomes ({workers} workers) ...")
    t0 = time.time()
    got = {}
    with Pool(workers, initializer=_init_worker, initargs=(SELECTED_KMERS,)) as pool:
        for idx, kmer, asm, err in pool.imap_unordered(_feat_worker, work, chunksize=8):
            if err:
                print(f"    WARNING idx={idx}: {err}")
            else:
                got[idx] = (kmer, asm)
    feat_t = time.time() - t0
    idxs = sorted(got)
    kn = normalizer.normalize_kmer(np.stack([got[i][0] for i in idxs])).astype(np.float32)
    an = normalizer.normalize_assembly(np.stack([got[i][1] for i in idxs])).astype(np.float32)
    in_names = [i.name for i in session.get_inputs()]
    out_name = session.get_outputs()[0].name
    preds = np.zeros((len(idxs), 2), dtype=np.float32)
    t1 = time.time()
    for s in range(0, len(idxs), 64):
        e = min(s + 64, len(idxs))
        preds[s:e] = session.run([out_name], {in_names[0]: kn[s:e], in_names[1]: an[s:e]})[0]
    infer_t = time.time() - t1
    print(f"    features {feat_t:.1f}s, inference {infer_t:.2f}s")

    res = meta.iloc[idxs].copy().reset_index(drop=True)
    res["pred_completeness"] = preds[:, 0]
    res["pred_contamination"] = preds[:, 1]
    res["wall_clock_s"] = feat_t + infer_t
    res["n_threads"] = workers
    res["model"] = "magicc_v5.onnx"
    tmp = str(out) + ".tmp"
    res.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, out)
    print(f"    wrote {out}")

    # cache the raw feature matrix for the mechanistic k-mer diagnostic
    np.savez_compressed(set_dir / "magicc_v5_features.npz",
                        genome_id=np.array(meta.genome_id.values[idxs], dtype=object),
                        kmer=np.stack([got[i][0] for i in idxs]),
                        summary=np.stack([got[i][1] for i in idxs]))
    return res


# ==========================================================================
# MECHANISM 1 — how much does the injected error actually corrupt MAGICC's input?
# ==========================================================================
def kmer_perturbation(set_dir: Path, out_dir: Path) -> Optional[pd.DataFrame]:
    """Measure how far the injected error moves MAGICC's actual input vector.

    The naive mechanistic prediction is that a per-base substitution rate p
    destroys a fraction 1 - (1-p)^9 of overlapping 9-mers (36.98 % at p = 5 %).
    MAGICC, however, does not see k-mer *positions*: it sees the **counts of
    9,249 selected canonical 9-mers**, where a position lost from one selected
    k-mer can be replaced by a mutated non-selected 9-mer landing back in the
    selected set.  Four complementary quantities are therefore recorded, all
    against the *same reference's* error-free assembly:

      net_mass_lost   sum(max(b-a,0)) / sum(b)  - selected-k-mer count mass that
                      is lost and NOT replaced within the same k-mer species.
      total_ratio     sum(a) / sum(b)           - does the total selected count
                      shrink, hold, or grow?
      l1_relative     sum|a-b| / sum(b)         - total movement of the raw count
                      vector, counting gains and losses alike.
      normalised_l2   Euclidean distance in the model's own input space
                      (log1p -> z-score), i.e. what the network actually ingests,
                      reported per feature dimension.

    The 7 k-mer summary features are tracked separately, because those are the
    channel through which assembly-size artefacts (duplication) reach the model.
    """
    fp = set_dir / "magicc_v5_features.npz"
    if not fp.exists():
        print("  (no cached feature matrix; skipping the k-mer diagnostic)")
        return None
    from magicc.assembly_stats import FEATURE_NAMES
    from magicc.normalization import FeatureNormalizer
    normalizer = FeatureNormalizer.load(NORMALIZATION)

    z = np.load(fp, allow_pickle=True)
    ids = list(z["genome_id"])
    K = z["kmer"].astype(np.float64)
    S = z["summary"].astype(np.float64)
    KN = normalizer.normalize_kmer(K.astype(np.float32)).astype(np.float64)
    SN = normalizer.normalize_assembly(S.astype(np.float32)).astype(np.float64)
    idx_of = {g: i for i, g in enumerate(ids)}
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    ctrl_of = {int(r.ref_index): r.genome_id
               for r in gm[gm.error_type == "none"].itertuples()}

    rows = []
    for r in gm.itertuples():
        if r.genome_id not in idx_of or r.error_type == "none":
            continue
        c = ctrl_of.get(int(r.ref_index))
        if c is None or c not in idx_of:
            continue
        ia, ib = idx_of[r.genome_id], idx_of[c]
        a, b = K[ia], K[ib]
        sb = b.sum()
        rec = {
            "genome_id": r.genome_id, "ref_index": int(r.ref_index),
            "error_type": r.error_type, "error_rate": float(r.error_rate),
            "error_rate_pct": round(100 * float(r.error_rate), 4),
            "net_mass_lost": round(float(np.maximum(b - a, 0).sum() / sb), 6) if sb else np.nan,
            "total_ratio": round(float(a.sum() / sb), 6) if sb else np.nan,
            "l1_relative": round(float(np.abs(a - b).sum() / sb), 6) if sb else np.nan,
            "cosine_similarity_to_control": round(
                float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 6),
            "n_selected_kmers_lost": int(((b > 0) & (a == 0)).sum()),
            "normalised_l2_per_dim": round(
                float(np.linalg.norm(KN[ia] - KN[ib]) / np.sqrt(KN.shape[1])), 6),
            "summary_normalised_l2_per_dim": round(
                float(np.linalg.norm(SN[ia] - SN[ib]) / np.sqrt(SN.shape[1])), 6),
            "predicted_kmer_position_corruption_1_minus_1mp9":
                round(1.0 - (1.0 - float(r.error_rate)) ** 9, 6)
                if str(r.error_type).startswith("substitution") else "",
        }
        for j, name in enumerate(FEATURE_NAMES[:S.shape[1]]):
            rec[f"summary_delta_{name}"] = round(float(SN[ia, j] - SN[ib, j]), 5)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "set_G_kmer_perturbation.tsv", sep="\t", index=False)

    agg = {"n": ("genome_id", "size"),
           "observed_kmer_corruption": ("net_mass_lost", "mean"),
           "total_count_ratio": ("total_ratio", "mean"),
           "l1_relative": ("l1_relative", "mean"),
           "cosine": ("cosine_similarity_to_control", "mean"),
           "kmers_lost": ("n_selected_kmers_lost", "mean"),
           "normalised_l2_per_dim": ("normalised_l2_per_dim", "mean"),
           "summary_l2_per_dim": ("summary_normalised_l2_per_dim", "mean")}
    for name in FEATURE_NAMES[:S.shape[1]]:
        agg[f"d_{name}"] = (f"summary_delta_{name}", "mean")
    summ = df.groupby(["error_type", "error_rate_pct"]).agg(**agg).reset_index()
    summ["predicted_kmer_position_corruption"] = [
        round(1.0 - (1.0 - r / 100.0) ** 9, 6) if str(e).startswith("substitution")
        else "" for e, r in zip(summ.error_type, summ.error_rate_pct)]
    summ.round(6).to_csv(out_dir / "set_G_kmer_perturbation_summary.tsv",
                         sep="\t", index=False)
    print("\n  MAGICC input perturbation vs the error-free control "
          "(net_lost/total_ratio/L1 on raw counts; L2 per dim in model space):")
    cols = ["error_type", "error_rate_pct", "n", "observed_kmer_corruption",
            "predicted_kmer_position_corruption", "total_count_ratio",
            "l1_relative", "normalised_l2_per_dim", "summary_l2_per_dim"]
    print(summ[cols].round(4).to_string(index=False))
    return summ


# ==========================================================================
# MECHANISM 2 — ORF disruption, from CheckM2's own gene calls
# ==========================================================================
def orf_diagnostic(set_dir: Path, out_dir: Path) -> Optional[pd.DataFrame]:
    """CheckM2's own gene-call statistics as a direct read-out of ORF disruption.

    This is the same exhibit WS3 Track A used on the real MinION assembly: a
    frameshifted assembly yields MORE and SHORTER predicted CDS and a lower
    coding density.  Here it is measured under a controlled dose.
    """
    qr = set_dir / "checkm2_output" / "quality_report.tsv"
    if not qr.exists():
        return None
    q = pd.read_csv(qr, sep="\t")
    q["genome_id"] = q["Name"].astype(str)
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    keep = [c for c in ("genome_id", "Total_Coding_Sequences", "Coding_Density",
                        "Average_Gene_Length", "GC_Content", "Genome_Size",
                        "Contig_N50", "Total_Contigs") if c in q.columns]
    m = gm[["genome_id", "ref_index", "error_type", "error_rate",
            "error_rate_pct"]].merge(q[keep], on="genome_id", how="inner")
    ctrl = m[m.error_type == "none"].set_index("ref_index")
    num = [c for c in keep if c != "genome_id"]
    rows = []
    for et, rate in sorted({(a, b) for a, b in zip(m.error_type, m.error_rate)
                            if a != "none"}):
        d = m[(m.error_type == et) & (m.error_rate == rate)].set_index("ref_index")
        c = ctrl.reindex(d.index)
        rec = {"error_type": et, "error_rate_pct": round(100 * rate, 4),
               "n": int(len(d))}
        for col in num:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.nanmean(d[col].values / c[col].values.astype(float))
            rec[f"{col}_ratio_vs_control"] = round(float(ratio), 4)
            rec[f"{col}_mean"] = round(float(np.nanmean(d[col].values)), 4)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "set_G_orf_disruption.tsv", sep="\t", index=False)
    print("\n  ORF disruption (CheckM2 gene calls, ratio vs the error-free control):")
    cols = [c for c in df.columns if c.endswith("_ratio_vs_control")
            or c in ("error_type", "error_rate_pct", "n")]
    print(df[cols].to_string(index=False))
    return df


# ==========================================================================
# Provenance audit (WS1.4 style, GCA<->GCF cross-mapped)
# ==========================================================================
_VER = re.compile(r"\.\d+$")
_NUM = re.compile(r"(\d{6,})")


def strict_norm(acc):
    if not isinstance(acc, str) or not acc:
        return None
    a = acc.strip()
    for p in ("GB_", "RS_"):
        if a.startswith(p):
            a = a[len(p):]
    return _VER.sub("", a)


def assembly_number(acc):
    if not isinstance(acc, str) or not acc:
        return None
    m = _NUM.search(acc)
    return m.group(1) if m else None


def build_crossmap():
    fg = pd.read_csv(DATA_DIR / "gtdb" / "filtered_genomes.tsv", sep="\t",
                     usecols=["gtdb_accession", "ncbi_accession", "gcf_accession"])
    cmap, groups, inconsistent = {}, defaultdict(set), 0
    for gtdb, ncbi, gcf in zip(fg.gtdb_accession, fg.ncbi_accession, fg.gcf_accession):
        key = assembly_number(ncbi) or assembly_number(gtdb) or assembly_number(gcf)
        if key is None:
            continue
        nums = set()
        for a in (gtdb, ncbi, gcf):
            s = strict_norm(a)
            if s:
                if cmap.get(s) not in (None, key):
                    inconsistent += 1
                cmap[s] = key
                groups[key].add(s)
                n = assembly_number(a)
                if n:
                    nums.add(n)
        if len(nums) > 1:
            inconsistent += 1
    return cmap, {"filtered_genomes_rows": int(len(fg)),
                  "distinct_accession_strings_mapped": len(cmap),
                  "distinct_canonical_assemblies": len(groups),
                  "rows_with_inconsistent_assembly_numbers": int(inconsistent)}


def canon(accs, cmap):
    strict, canonical, unmapped = set(), set(), 0
    for a in accs:
        s = strict_norm(a)
        if not s:
            continue
        strict.add(s)
        k = cmap.get(s) or assembly_number(a)
        if k:
            canonical.add(k)
        else:
            unmapped += 1
    return strict, canonical, unmapped


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_task(p):
    return str(p), _sha(Path(p))


def provenance_audit(set_dir: Path, out_dir: Path, workers: int,
                     manifest: bool = True) -> dict:
    print("\n" + "=" * 78)
    print(f"PROVENANCE AUDIT — {set_dir.name}")
    print("=" * 78)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = set_dir.name
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    cmap, cstats = build_crossmap()

    splits = {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(DATA_DIR / "splits" / f"{s}_genomes.tsv", sep="\t")
        accs = list(d.gtdb_accession) + [a for a in d.get("ncbi_accession", [])
                                         if isinstance(a, str)]
        splits[s] = canon(accs, cmap)

    kmer_sel = []
    for f in ("selected_bacterial_1000.tsv", "selected_archaeal_1000.tsv"):
        fp = DATA_DIR / "kmer_selection" / f
        if fp.exists():
            kmer_sel += pd.read_csv(fp, sep="\t")["gtdb_accession"].tolist()
    _, kmer_canon, _ = canon(kmer_sel, cmap)

    doms = sorted(set(gm.dominant_accession))
    cons = sorted({x for s in gm.contaminant_accessions.fillna("")
                   for x in str(s).split(";") if x})
    _, dom_canon, dom_unmapped = canon(doms, cmap)
    _, con_canon, con_unmapped = canon(cons, cmap)

    rows = []
    for label, cs, raw in (("dominants", dom_canon, doms),
                           ("contaminants", con_canon, cons)):
        r = {"group": label, "n_accessions": len(raw), "n_canonical": len(cs),
             "n_unmapped": dom_unmapped if label == "dominants" else con_unmapped}
        for s in ("train", "val", "test"):
            r[f"in_{s}"] = len(cs & splits[s][1])
        r["in_kmer_selection_set"] = len(cs & kmer_canon) if kmer_canon else "n/a"
        rows.append(r)
    ov = pd.DataFrame(rows)
    ov.to_csv(out_dir / f"{tag}_overlap_summary.tsv", sep="\t", index=False)
    print(ov.to_string(index=False))

    dom_split = {}
    for a in doms:
        _, c, _ = canon([a], cmap)
        dom_split[a] = ";".join(s for s in ("train", "val", "test")
                                if c & splits[s][1]) or "none"
    gm["dominant_split_crossmapped"] = gm.dominant_accession.map(dom_split)
    per_sample = gm.dominant_split_crossmapped.value_counts().to_dict()

    (out_dir / f"{tag}_dominants.txt").write_text("\n".join(doms) + "\n")
    (out_dir / f"{tag}_contaminants.txt").write_text("\n".join(cons) + "\n")

    def _cell(group: str, col: str) -> int:
        v = ov.loc[ov.group == group, col].iloc[0]
        try:
            return int(v)
        except (TypeError, ValueError):      # "n/a" when the k-mer set is absent
            return 0

    verdict = all(_cell(g, c) == 0
                  for g in ("dominants", "contaminants")
                  for c in ("in_train", "in_val", "in_kmer_selection_set"))

    man = {}
    if manifest:
        files = sorted((set_dir / "fasta").glob("*.fasta"))
        files += [set_dir / "metadata.tsv", set_dir / "generation_metadata.tsv",
                  set_dir / "design.tsv", set_dir / "reference_selection.tsv",
                  Path(ONNX_MODEL), Path(SELECTED_KMERS), Path(NORMALIZATION)]
        files = [f for f in files if Path(f).exists()]
        print(f"  hashing {len(files)} files ...")
        with Pool(workers) as pool:
            for path, h in pool.imap_unordered(_sha_task, [str(f) for f in files],
                                               chunksize=16):
                man[str(Path(path).relative_to(PROJECT_DIR))] = h
        with open(out_dir / f"{tag}_sha256_manifest.txt", "w") as fh:
            for k in sorted(man):
                fh.write(f"{man[k]}  {k}\n")

    rep = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "set_dir": str(set_dir),
           "crossmap_stats": cstats,
           "overlap_summary": ov.to_dict("records"),
           "per_sample_dominant_split": per_sample,
           "n_files_hashed": len(man),
           "kmer_selection_set_resolved": bool(kmer_canon),
           "DISJOINTNESS_VERDICT": "PASS" if verdict else "FAIL",
           "note": ("Dominants are restricted to data/splits/test_finished_genomes.tsv "
                    "and contaminants to the same held-out test split. Accessions are "
                    "compared after GCA<->GCF cross-mapping through "
                    "data/gtdb/filtered_genomes.tsv (277,183 assemblies), the "
                    "normalisation whose absence caused the Set C/D leakage "
                    "undercount.")}
    (out_dir / f"{tag}_provenance_audit.json").write_text(
        json.dumps(rep, indent=2, default=str))
    print(f"  DISJOINTNESS VERDICT: {rep['DISJOINTNESS_VERDICT']}")
    return rep


# ==========================================================================
# data assembly
# ==========================================================================
def load_predictions(set_dir: Path) -> Dict[str, pd.DataFrame]:
    preds = {}
    files = {"magicc_v5": "magicc_v5_predictions.tsv",
             "checkm2": "checkm2_predictions.tsv",
             "cocopye": "cocopye_predictions.tsv",
             "deepcheck": "deepcheck_predictions.tsv"}
    for tool, fn in files.items():
        p = set_dir / fn
        if not p.exists():
            print(f"  [{tool}] MISSING {fn} — tool excluded from this run")
            continue
        d = pd.read_csv(p, sep="\t")
        need = {"genome_id", "pred_completeness", "pred_contamination"}
        if not need.issubset(d.columns):
            print(f"  [{tool}] {fn} lacks {need - set(d.columns)} — excluded")
            continue
        preds[tool] = d[["genome_id", "pred_completeness", "pred_contamination"]].copy()
        print(f"  [{tool}] {len(d)} predictions")
    return preds


def build_long(set_dir: Path, preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    keep = ["genome_id", "ref_index", "arm_index", "error_type", "error_rate",
            "error_rate_pct", "error_model", "dominant_accession", "dominant_phylum",
            "dominant_domain", "quality_tier", "true_completeness",
            "true_contamination", "true_completeness_indel_adjusted",
            "true_contamination_dup_counted", "dominant_reference_bp",
            "base_n_contigs", "base_total_length", "n_contigs", "total_length",
            "realised_error_rate", "expected_kmer_corruption", "dup_bp",
            "n_substitutions", "n_indel_events", "n_chimera_events",
            "in_training_domain"]
    base = gm[[c for c in keep if c in gm.columns]]
    frames = []
    for tool, d in preds.items():
        m = base.merge(d, on="genome_id", how="inner", validate="one_to_one")
        m["tool"] = tool
        frames.append(m)
    long = pd.concat(frames, ignore_index=True)
    long["err_completeness"] = long.pred_completeness - long.true_completeness
    long["err_contamination"] = long.pred_contamination - long.true_contamination
    long["abs_err_completeness"] = long.err_completeness.abs()
    long["abs_err_contamination"] = long.err_contamination.abs()
    long["mimag"] = np.where(
        (long.pred_completeness >= 90) & (long.pred_contamination < 5), "high",
        np.where((long.pred_completeness >= 50) & (long.pred_contamination < 10),
                 "medium", "low"))
    long["mimag_true"] = np.where(
        (long.true_completeness >= 90) & (long.true_contamination < 5), "high",
        np.where((long.true_completeness >= 50) & (long.true_contamination < 10),
                 "medium", "low"))
    return long


def _r2_or_blank(t, p, min_sd: float = 1e-3):
    """Coefficient of determination, blank where the truth is (near-)constant."""
    t = np.asarray(t, float)
    if float(np.std(t)) < min_sd:
        return ""
    return round(fw.r2_coefficient_of_determination(t, p), 4)


# ==========================================================================
# WS6.3 — degradation curves
# ==========================================================================
def curve_stats(df: pd.DataFrame, seed: int) -> dict:
    bs = fw.Bootstrapper(clusters=df.ref_index.values, n_iter=N_BOOT, seed=seed)
    ec = df.err_contamination.values
    ek = df.err_completeness.values
    tc = df.true_contamination.values
    pc = df.pred_contamination.values
    tk = df.true_completeness.values
    pk = df.pred_completeness.values
    agree = (df.mimag.values == df.mimag_true.values).astype(float)
    t_hq = ((df.true_completeness >= 90) & (df.true_contamination < 5)).values
    p_hq = ((df.pred_completeness >= 90) & (df.pred_contamination < 5)).values

    def stat(idx):
        return {"comp_mae": float(np.mean(np.abs(ek[idx]))),
                "comp_bias": float(np.mean(ek[idx])),
                "cont_mae": float(np.mean(np.abs(ec[idx]))),
                "cont_bias": float(np.mean(ec[idx])),
                "mimag_agreement": float(np.mean(agree[idx]))}

    r = bs.ci_multi(stat)
    out = {"n": int(len(df)), "n_clusters": int(df.ref_index.nunique())}
    for k, v in r.items():
        out[k] = round(v["estimate"], 4)
        out[f"{k}_ci_lo"] = round(v["ci_lo"], 4)
        out[f"{k}_ci_hi"] = round(v["ci_hi"], 4)
    out["comp_r2"] = _r2_or_blank(tk, pk)
    out["cont_r2"] = _r2_or_blank(tc, pc)
    out["HQ_recall"] = (round(float((t_hq & p_hq).sum() / t_hq.sum()), 4)
                        if t_hq.sum() else "")
    out["HQ_precision"] = (round(float((t_hq & p_hq).sum() / p_hq.sum()), 4)
                           if p_hq.sum() else "")
    out["n_true_HQ"] = int(t_hq.sum())
    out["n_pred_HQ"] = int(p_hq.sum())
    out["true_comp_mean"] = round(float(np.mean(tk)), 3)
    out["true_cont_mean"] = round(float(np.mean(tc)), 3)
    out["true_comp_sd"] = round(float(np.std(tk)), 3)
    out["true_cont_sd"] = round(float(np.std(tc)), 3)
    return out


def build_curves(long: pd.DataFrame, tools: List[str]) -> pd.DataFrame:
    rows = []
    for tool in tools:
        t = long[long.tool == tool]
        ctrl = t[t.error_type == "none"]
        for et in ERROR_ORDER:
            sub = t[t.error_type == et]
            if not len(sub):
                continue
            # rate 0 == the shared control arm, included in every curve so the
            # dose-response starts from the tool's error-free performance
            for rate, d in [(0.0, ctrl)] + sorted(sub.groupby("error_rate"),
                                                  key=lambda kv: kv[0]):
                if not len(d):
                    continue
                seed = fw.stable_hash(f"setG|curve|{tool}|{et}|{rate}")
                r = {"tool": tool, "error_type": et, "error_rate": float(rate),
                     "error_rate_pct": round(100 * float(rate), 4)}
                r.update(curve_stats(d, seed))
                rows.append(r)
    return pd.DataFrame(rows)


# ==========================================================================
# paired degradation vs each reference's own error-free control
# ==========================================================================
def paired_vs_control(long: pd.DataFrame, tools: List[str]) -> pd.DataFrame:
    rows = []
    for tool in tools:
        t = long[long.tool == tool]
        ctrl = t[t.error_type == "none"].set_index("ref_index")
        for et in ERROR_ORDER:
            for rate, d in sorted(t[t.error_type == et].groupby("error_rate"),
                                  key=lambda kv: kv[0]):
                d = d.set_index("ref_index").sort_index()
                c = ctrl.reindex(d.index)
                for metric, col in (("completeness", "abs_err_completeness"),
                                    ("contamination", "abs_err_contamination")):
                    diff = (d[col].values - c[col].values)
                    scol = ("err_completeness" if metric == "completeness"
                            else "err_contamination")
                    sdiff = (d[scol].values - c[scol].values)
                    seed = fw.stable_hash(f"setG|paired|{tool}|{et}|{rate}|{metric}")
                    bs = fw.Bootstrapper(clusters=d.index.values, n_iter=N_BOOT,
                                         seed=seed)
                    mean_ci = bs.ci(lambda i, v=diff: float(np.mean(v[i])))
                    hl_ci = bs.ci(lambda i, v=diff: fw.hodges_lehmann_paired(v[i]))
                    smean = bs.ci(lambda i, v=sdiff: float(np.mean(v[i])))
                    try:
                        p = float(sps.wilcoxon(diff, alternative="two-sided",
                                               zero_method="wilcox").pvalue)
                    except ValueError:
                        p = 1.0
                    rows.append({
                        "tool": tool, "error_type": et, "error_rate": float(rate),
                        "error_rate_pct": round(100 * float(rate), 4),
                        "metric": metric, "n_pairs": int(len(diff)),
                        "delta_mae": round(mean_ci["estimate"], 4),
                        "delta_mae_ci_lo": round(mean_ci["ci_lo"], 4),
                        "delta_mae_ci_hi": round(mean_ci["ci_hi"], 4),
                        "delta_mae_hl": round(hl_ci["estimate"], 4),
                        "delta_mae_hl_ci_lo": round(hl_ci["ci_lo"], 4),
                        "delta_mae_hl_ci_hi": round(hl_ci["ci_hi"], 4),
                        "delta_signed_bias": round(smean["estimate"], 4),
                        "delta_signed_bias_ci_lo": round(smean["ci_lo"], 4),
                        "delta_signed_bias_ci_hi": round(smean["ci_hi"], 4),
                        "p_wilcoxon": p,
                    })
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh"] = fw.bh_correct(df.p_wilcoxon.values)
    return df


# ==========================================================================
# head-to-head MAGICC vs each competitor, per rate
# ==========================================================================
def head_to_head(long: pd.DataFrame, ref_tool: str, others: List[str]) -> pd.DataFrame:
    rows = []
    a_all = long[long.tool == ref_tool]
    for other in others:
        b_all = long[long.tool == other]
        for et in ERROR_ORDER + ["none"]:
            rates = ([0.0] if et == "none"
                     else sorted(a_all[a_all.error_type == et].error_rate.unique()))
            for rate in rates:
                a = a_all[(a_all.error_type == et) & (a_all.error_rate == rate)]
                b = b_all[(b_all.error_type == et) & (b_all.error_rate == rate)]
                m = a.merge(b, on="genome_id", suffixes=("_a", "_b"),
                            validate="one_to_one")
                if len(m) < 3:
                    continue
                for metric, col in (("completeness", "abs_err_completeness"),
                                    ("contamination", "abs_err_contamination")):
                    d = m[f"{col}_a"].values - m[f"{col}_b"].values
                    seed = fw.stable_hash(f"setG|h2h|{other}|{et}|{rate}|{metric}")
                    bs = fw.Bootstrapper(clusters=m.ref_index_a.values,
                                         n_iter=N_BOOT, seed=seed)
                    hl = bs.ci(lambda i, v=d: fw.hodges_lehmann_paired(v[i]))
                    dm = bs.ci(lambda i, v=d: float(np.mean(v[i])))
                    cd = bs.ci(lambda i, x=m[f"{col}_a"].values,
                               y=m[f"{col}_b"].values: fw.cliffs_delta(x[i], y[i]))
                    try:
                        p = float(sps.wilcoxon(d, alternative="two-sided",
                                               zero_method="wilcox").pvalue)
                    except ValueError:
                        p = 1.0
                    rows.append({
                        "comparator": other, "error_type": et,
                        "error_rate": float(rate),
                        "error_rate_pct": round(100 * float(rate), 4),
                        "metric": metric, "n_pairs": int(len(d)),
                        "mae_magicc": round(float(np.mean(m[f"{col}_a"])), 4),
                        "mae_comparator": round(float(np.mean(m[f"{col}_b"])), 4),
                        "delta_mae": round(dm["estimate"], 4),
                        "delta_mae_ci_lo": round(dm["ci_lo"], 4),
                        "delta_mae_ci_hi": round(dm["ci_hi"], 4),
                        "hl": round(hl["estimate"], 4),
                        "hl_ci_lo": round(hl["ci_lo"], 4),
                        "hl_ci_hi": round(hl["ci_hi"], 4),
                        "cliffs_delta": round(cd["estimate"], 4),
                        "cliffs_ci_lo": round(cd["ci_lo"], 4),
                        "cliffs_ci_hi": round(cd["ci_hi"], 4),
                        "p_wilcoxon": p,
                        "winner": ("MAGICC" if dm["ci_hi"] < 0 else
                                   other if dm["ci_lo"] > 0 else "tie"),
                    })
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh"] = fw.bh_correct(df.p_wilcoxon.values)
    return df


# ==========================================================================
# WS6.4 — applicability boundary
# ==========================================================================
def _interp_cross(rates: np.ndarray, vals: np.ndarray, thr: float) -> Optional[float]:
    """First rate at which ``vals`` crosses ``thr`` upwards, linear in log-rate."""
    for i in range(1, len(rates)):
        if vals[i - 1] < thr <= vals[i]:
            r0, r1 = rates[i - 1], rates[i]
            v0, v1 = vals[i - 1], vals[i]
            if v1 == v0:
                return float(r1)
            if r0 <= 0:
                return float(r0 + (r1 - r0) * (thr - v0) / (v1 - v0))
            lr = np.log10(r0) + (np.log10(r1) - np.log10(r0)) * (thr - v0) / (v1 - v0)
            return float(10 ** lr)
    return None


def boundary_table(paired: pd.DataFrame, h2h: pd.DataFrame,
                   curves: pd.DataFrame, tools: List[str]) -> pd.DataFrame:
    rows = []
    for tool in tools:
        for et in ERROR_ORDER:
            for metric in ("completeness", "contamination"):
                p = paired[(paired.tool == tool) & (paired.error_type == et)
                           & (paired.metric == metric)].sort_values("error_rate")
                if not len(p):
                    continue
                rates = np.r_[0.0, p.error_rate.values]
                # conservative: use the UPPER 95 % bound of the paired degradation
                upper = np.r_[0.0, p.delta_mae_ci_hi.values]
                point = np.r_[0.0, p.delta_mae.values]
                rec = {"tool": tool, "error_type": et, "metric": metric,
                       "max_rate_tested_pct": round(100 * float(rates.max()), 4)}
                for thr in (1.0, 2.0, 5.0):
                    x = _interp_cross(rates, upper, thr)
                    rec[f"rate_pct_where_upperCI_delta_mae_exceeds_{thr:g}pp"] = (
                        round(100 * x, 4) if x is not None else "")
                    y = _interp_cross(rates, point, thr)
                    rec[f"rate_pct_where_delta_mae_exceeds_{thr:g}pp"] = (
                        round(100 * y, 4) if y is not None else "")
                # first rate with a BH-significant degradation
                sig = p[(p.q_bh < 0.05) & (p.delta_mae > 0)]
                rec["first_significant_rate_pct"] = (
                    round(100 * float(sig.error_rate.min()), 4) if len(sig) else "")
                rec["delta_mae_at_max_rate"] = round(float(p.delta_mae.iloc[-1]), 4)
                rec["delta_mae_at_max_rate_ci"] = (
                    f"[{p.delta_mae_ci_lo.iloc[-1]:.3f}, {p.delta_mae_ci_hi.iloc[-1]:.3f}]")
                rows.append(rec)

    bt = pd.DataFrame(rows)

    # crossover: first rate at which MAGICC becomes significantly WORSE than a
    # competitor on the paired difference (upper CI of MAGICC-comparator > 0)
    cross = []
    for other in sorted(h2h.comparator.unique()) if len(h2h) else []:
        for et in ERROR_ORDER:
            for metric in ("completeness", "contamination"):
                s = h2h[(h2h.comparator == other) & (h2h.error_type == et)
                        & (h2h.metric == metric)].sort_values("error_rate")
                if not len(s):
                    continue
                base = h2h[(h2h.comparator == other) & (h2h.error_type == "none")
                           & (h2h.metric == metric)]
                lose = s[(s.delta_mae_ci_lo > 0) & (s.q_bh < 0.05)]
                win = s[(s.delta_mae_ci_hi < 0) & (s.q_bh < 0.05)]
                cross.append({
                    "comparator": other, "error_type": et, "metric": metric,
                    "delta_mae_at_zero_error": (round(float(base.delta_mae.iloc[0]), 4)
                                                if len(base) else ""),
                    "winner_at_zero_error": (base.winner.iloc[0] if len(base) else ""),
                    "crossover_rate_pct_magicc_becomes_worse":
                        round(100 * float(lose.error_rate.min()), 4) if len(lose) else "",
                    "highest_rate_pct_magicc_still_better":
                        round(100 * float(win.error_rate.max()), 4) if len(win) else "",
                    "delta_mae_at_max_rate": round(float(s.delta_mae.iloc[-1]), 4),
                    "delta_mae_at_max_rate_ci":
                        f"[{s.delta_mae_ci_lo.iloc[-1]:.3f}, {s.delta_mae_ci_hi.iloc[-1]:.3f}]",
                    "winner_at_max_rate": s.winner.iloc[-1],
                })
    return bt, pd.DataFrame(cross)


def real_world_context(bt: pd.DataFrame) -> pd.DataFrame:
    """Place the measured boundary against real sequencing / assembly error rates."""
    rows = []
    sub = bt[(bt.tool == "magicc_v5") & (bt.error_type == "substitution")]
    b = {}
    for _, r in sub.iterrows():
        for thr in (1.0, 2.0, 5.0):
            k = f"rate_pct_where_upperCI_delta_mae_exceeds_{thr:g}pp"
            v = r.get(k, "")
            b[(r["metric"], thr)] = float(v) if v not in ("", None) else np.nan
    for name, rate, note in REAL_WORLD_RATES:
        rec = {"context": name, "per_base_error_rate": rate,
               "per_base_error_rate_pct": 100 * rate, "note": note}
        for metric in ("completeness", "contamination"):
            for thr in (1.0, 5.0):
                lim = b.get((metric, thr), np.nan)
                rec[f"{metric}_within_{thr:g}pp_boundary"] = (
                    "yes" if (np.isnan(lim) or 100 * rate < lim) else "no")
                rec[f"{metric}_boundary_pct_{thr:g}pp"] = ("" if np.isnan(lim)
                                                           else round(lim, 4))
                rec[f"{metric}_margin_x_{thr:g}pp"] = (
                    "" if np.isnan(lim) else round(lim / (100 * rate), 2))
        rows.append(rec)
    return pd.DataFrame(rows)


# ==========================================================================
# figures
# ==========================================================================
def _xvals(rates: np.ndarray, et: str) -> np.ndarray:
    """Rate 0 cannot be drawn on a log axis; place it one decade below the min."""
    r = 100 * np.asarray(rates, float)
    if et in LOG_X:
        pos = r[r > 0]
        floor = (pos.min() / 3.0) if len(pos) else 0.01
        return np.where(r <= 0, floor, r)
    return r


def fig_curves(curves: pd.DataFrame, tools: List[str], metric: str,
               value: str, ylabel: str, stem: Path, title: str) -> None:
    plt = fw.init_matplotlib()
    ets = [e for e in ERROR_ORDER if (curves.error_type == e).any()]
    fig, axes = plt.subplots(1, len(ets), figsize=(2.45 * len(ets) + 0.7, 2.9),
                             sharey=True)
    if len(ets) == 1:
        axes = [axes]
    for ax, et in zip(axes, ets):
        for tool in tools:
            s = curves[(curves.tool == tool) & (curves.error_type == et)
                       ].sort_values("error_rate")
            if not len(s):
                continue
            x = _xvals(s.error_rate.values, et)
            ax.plot(x, s[value].values, marker=MARKERS.get(tool, "o"), ms=3.6,
                    lw=1.15, color=PALETTE.get(tool, "#444"),
                    markeredgecolor="#222222", markeredgewidth=0.4,
                    label=TOOL_LABEL.get(tool, tool))
            ax.fill_between(x, s[f"{value}_ci_lo"].values, s[f"{value}_ci_hi"].values,
                            color=PALETTE.get(tool, "#444"), alpha=0.13, linewidth=0)
        if et in LOG_X:
            ax.set_xscale("log")
            xt = _xvals(np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.05]), et)
            ax.set_xticks(xt)
            ax.set_xticklabels(["0", "0.1", "0.5", "1", "2", "5"], fontsize=6.5)
        ax.axhline(0, color="#666666", lw=0.7, ls="--")
        ax.set_title(ERROR_LABEL[et], fontsize=8)
        ax.set_xlabel(ERROR_XLABEL[et], fontsize=7)
    axes[0].set_ylabel(ylabel, fontsize=8)
    axes[-1].legend(frameon=False, fontsize=6.3, loc="best")
    fig.suptitle(title, fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_substitution_focus(curves: pd.DataFrame, paired: pd.DataFrame,
                           tools: List[str], stem: Path) -> None:
    """The headline panel: substitutions, the case where MAGICC should suffer."""
    plt = fw.init_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.95))
    panels = [("substitution", "comp_mae", "completeness MAE (pp)",
               "Completeness accuracy"),
              ("substitution", "cont_mae", "contamination MAE (pp)",
               "Contamination accuracy"),
              ("substitution", "comp_bias", "signed completeness error (pp)",
               "Completeness bias"),
              ("substitution", "mimag_agreement", "MIMAG-inspired class agreement",
               "Quality-class agreement")]
    for ax, (et, val, ylab, title) in zip(axes, panels):
        for tool in tools:
            s = curves[(curves.tool == tool) & (curves.error_type == et)
                       ].sort_values("error_rate")
            if not len(s):
                continue
            x = _xvals(s.error_rate.values, et)
            ax.plot(x, s[val].values, marker=MARKERS.get(tool, "o"), ms=3.8, lw=1.2,
                    color=PALETTE.get(tool, "#444"), markeredgecolor="#222222",
                    markeredgewidth=0.4, label=TOOL_LABEL.get(tool, tool))
            ax.fill_between(x, s[f"{val}_ci_lo"].values, s[f"{val}_ci_hi"].values,
                            color=PALETTE.get(tool, "#444"), alpha=0.13, linewidth=0)
        ax.set_xscale("log")
        xt = _xvals(np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.05]), et)
        ax.set_xticks(xt)
        ax.set_xticklabels(["0", "0.1", "0.5", "1", "2", "5"], fontsize=6.8)
        ax.set_xlabel("per-base substitution rate (%)", fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        # Real-world regime, drawn in neutral grey (never a hue that would need
        # red/green discrimination).  Nominal Illumina Q30 raw-base error is
        # 0.1 %; assembly consensus error is 10^-3 to 10^-5 per base, i.e. at or
        # below that line.  Everything real therefore lies at x <= 0.1.
        ax.axvspan(xt.min() * 0.5, 0.1, color="#DDDDDD", alpha=0.55, zorder=0,
                   linewidth=0)
        ax.axvline(0.1, color="#888888", lw=0.8, ls=(0, (2, 2)), zorder=1)
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.suptitle("Set G: degradation under per-base substitutions "
                 "(paired within reference genome)", fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_paired_degradation(paired: pd.DataFrame, tools: List[str],
                           metric: str, stem: Path) -> None:
    plt = fw.init_matplotlib()
    ets = [e for e in ERROR_ORDER if (paired.error_type == e).any()]
    fig, axes = plt.subplots(1, len(ets), figsize=(2.45 * len(ets) + 0.7, 2.9),
                             sharey=True)
    if len(ets) == 1:
        axes = [axes]
    for ax, et in zip(axes, ets):
        for tool in tools:
            s = paired[(paired.tool == tool) & (paired.error_type == et)
                       & (paired.metric == metric)].sort_values("error_rate")
            if not len(s):
                continue
            x = _xvals(s.error_rate.values, et)
            ax.plot(x, s.delta_mae.values, marker=MARKERS.get(tool, "o"), ms=3.6,
                    lw=1.15, color=PALETTE.get(tool, "#444"),
                    markeredgecolor="#222222", markeredgewidth=0.4,
                    label=TOOL_LABEL.get(tool, tool))
            ax.fill_between(x, s.delta_mae_ci_lo.values, s.delta_mae_ci_hi.values,
                            color=PALETTE.get(tool, "#444"), alpha=0.13, linewidth=0)
        if et in LOG_X:
            ax.set_xscale("log")
            xt = 100 * np.array([0.001, 0.005, 0.01, 0.02, 0.05])
            ax.set_xticks(xt)
            ax.set_xticklabels(["0.1", "0.5", "1", "2", "5"], fontsize=6.5)
        ax.axhline(0, color="#666666", lw=0.7, ls="--")
        ax.set_title(ERROR_LABEL[et], fontsize=8)
        ax.set_xlabel(ERROR_XLABEL[et], fontsize=7)
    axes[0].set_ylabel(f"Delta {metric} MAE vs the same\nreference's error-free "
                       f"assembly (pp)", fontsize=7.5)
    axes[-1].legend(frameon=False, fontsize=6.3, loc="best")
    fig.suptitle(f"Set G: paired {metric} degradation caused by the injected error",
                 fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_mechanism(kp: pd.DataFrame, orf: Optional[pd.DataFrame], stem: Path) -> None:
    """Why each tool fails: k-mer corruption vs ORF disruption, same doses."""
    plt = fw.init_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    for et, mk, ls in (("substitution", "o", "-"), ("substitution_titv", "s", "--"),
                       ("indel", "v", "-.")):
        s = kp[kp.error_type == et].sort_values("error_rate_pct")
        if not len(s):
            continue
        ax.plot(s.error_rate_pct, 100 * s.observed_kmer_corruption, marker=mk,
                ms=3.8, lw=1.2, ls=ls, color=PALETTE["magicc_v5"],
                markeredgecolor="#222222", markeredgewidth=0.4,
                label=f"measured net loss, {et.replace('_', ' ')}")
        ax.plot(s.error_rate_pct, 100 * s.l1_relative, marker=mk, ms=3.0, lw=0.9,
                ls=":", color="#7FB6DA", markeredgecolor="#222222",
                markeredgewidth=0.3, label=f"measured L1 move, {et.replace('_', ' ')}")
    p = np.logspace(-1, np.log10(5), 60)
    ax.plot(p, 100 * (1 - (1 - p / 100) ** 9), color="#777777", lw=1.1,
            ls=(0, (3, 2)), label=r"naive prediction $1-(1-p)^9$")
    ax.set_xscale("log")
    ax.set_xlabel("per-base error rate (%)", fontsize=7.5)
    ax.set_ylabel("change in selected 9-mer count mass (%)", fontsize=7.5)
    ax.set_title("MAGICC input corruption", fontsize=8.5)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", ncol=1)

    ax = axes[1]
    if orf is not None and len(orf):
        cols = [(("Total_Coding_Sequences_ratio_vs_control"), "o", "predicted CDS count"),
                (("Average_Gene_Length_ratio_vs_control"), "v", "mean gene length"),
                (("Coding_Density_ratio_vs_control"), "^", "coding density")]
        for et, ls in (("substitution", "-"), ("indel", "-.")):
            s = orf[orf.error_type == et].sort_values("error_rate_pct")
            for col, mk, lab in cols:
                if col not in s.columns:
                    continue
                ax.plot(s.error_rate_pct, s[col], marker=mk, ms=3.6, lw=1.1, ls=ls,
                        color=PALETTE["checkm2"] if et == "indel" else "#7FB6DA",
                        markeredgecolor="#222222", markeredgewidth=0.4,
                        label=f"{lab}, {et}")
        ax.axhline(1.0, color="#666666", lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("per-base error rate (%)", fontsize=7.5)
        ax.set_ylabel("ratio to the error-free assembly", fontsize=7.5)
        ax.set_title("CheckM2 gene-call disruption", fontsize=8.5)
        ax.legend(frameon=False, fontsize=5.8, loc="best", ncol=2)
    fig.suptitle("Set G mechanism: substitutions corrupt k-mers; indels destroy ORFs",
                 fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_palette_check(cvd: pd.DataFrame, stem: Path) -> None:
    plt = fw.init_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 1.5 + 0.32 * len(PALETTE)))
    for i, (k, hexv) in enumerate(PALETTE.items()):
        for j, kind in enumerate(["normal", "deuteranopia", "protanopia",
                                  "tritanopia"]):
            c = hexv if kind == "normal" else fw.simulate_cvd(hexv, kind)
            ax.add_patch(plt.Rectangle((j, -i), 0.92, 0.86, facecolor=c,
                                       edgecolor="#333333", lw=0.4))
        ax.text(-0.15, -i + 0.43, TOOL_LABEL.get(k, k), ha="right", va="center",
                fontsize=7)
    ax.set_xlim(-2.4, 4.1)
    ax.set_ylim(-len(PALETTE) + 0.05, 1.1)
    ax.set_xticks(np.arange(4) + 0.46)
    ax.set_xticklabels(["normal", "deuteranopia", "protanopia", "tritanopia"],
                       fontsize=7)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Set G palette: minimum CIE76 Delta-E {cvd.min_delta_e.min():.1f} "
                 f"across all vision types ({int(cvd.flag_too_similar.sum())} pairs "
                 f"flagged)", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ==========================================================================
# main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="set_G")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--magicc-only", action="store_true")
    ap.add_argument("--skip-provenance", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--force-magicc", action="store_true")
    args = ap.parse_args()

    global RESULTS_DIR, FIG_DIR
    set_dir = BENCHMARK_DIR / args.set
    if args.set != "set_G":
        RESULTS_DIR = PROJECT_DIR / "results" / "revision" / args.set
        FIG_DIR = RESULTS_DIR / "figures"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"WS6.3 / WS6.4 — error-robustness analysis on {args.set}")
    print(f"model: {ONNX_MODEL}")
    print("=" * 78)

    run_magicc(set_dir, args.workers, force=args.force_magicc)
    if args.magicc_only:
        print("--magicc-only: stopping after inference")
        return 0

    prov = None
    if not args.skip_provenance:
        prov = provenance_audit(set_dir,
                                PROJECT_DIR / "results" / "revision" / "provenance",
                                args.workers, manifest=not args.no_manifest)

    print("\nloading predictions ...")
    preds = load_predictions(set_dir)
    tools = [t for t in TOOL_ORDER if t in preds]
    long = build_long(set_dir, preds)
    long.to_csv(RESULTS_DIR / "set_G_long_predictions.tsv", sep="\t", index=False)
    print(f"  long frame: {len(long)} rows, {len(tools)} tools, "
          f"{long.ref_index.nunique()} reference clusters")

    print("\n[1/5] degradation curves ...")
    curves = build_curves(long, tools)
    curves.to_csv(RESULTS_DIR / "set_G_curves.tsv", sep="\t", index=False)

    print("[2/5] paired degradation vs error-free control ...")
    paired = paired_vs_control(long, tools)
    paired.to_csv(RESULTS_DIR / "set_G_paired_degradation.tsv", sep="\t", index=False)

    print("[3/5] head-to-head MAGICC vs competitors ...")
    h2h = (head_to_head(long, "magicc_v5", [t for t in tools if t != "magicc_v5"])
           if "magicc_v5" in tools else pd.DataFrame())
    h2h.to_csv(RESULTS_DIR / "set_G_head_to_head.tsv", sep="\t", index=False)

    print("[4/5] applicability boundary ...")
    bt, cross = boundary_table(paired, h2h, curves, tools)
    bt.to_csv(RESULTS_DIR / "set_G_boundary.tsv", sep="\t", index=False)
    cross.to_csv(RESULTS_DIR / "set_G_crossover.tsv", sep="\t", index=False)
    ctx = real_world_context(bt)
    ctx.to_csv(RESULTS_DIR / "set_G_real_world_context.tsv", sep="\t", index=False)

    # Stratified view: does the degradation depend on the base assembly?  The
    # contig-count stratum matters specifically for the chimera axis, where the
    # achievable dose is bounded by the number of contigs available to mis-join.
    long["contig_stratum"] = np.where(long.base_n_contigs >= 20,
                                      "base_contigs>=20", "base_contigs<20")
    strat = []
    for tool in tools:
        t = long[long.tool == tool]
        ctrl = t[t.error_type == "none"].set_index("ref_index")
        for et in ERROR_ORDER:
            for rate, d in sorted(t[t.error_type == et].groupby("error_rate"),
                                  key=lambda kv: kv[0]):
                for col in ("quality_tier", "contig_stratum"):
                    for key, grp in d.groupby(col):
                        c = ctrl.reindex(grp.ref_index.values)
                        strat.append({
                            "tool": tool, "error_type": et,
                            "error_rate_pct": round(100 * float(rate), 4),
                            "stratify_by": col,
                            "stratum": f"{key}", "n": int(len(grp)),
                            "mean_realised_dose": round(
                                float(grp.realised_error_rate.mean()), 5),
                            "mean_chimera_events": round(
                                float(grp.n_chimera_events.mean()), 2),
                            "delta_comp_mae": round(float(np.mean(
                                grp.abs_err_completeness.values
                                - c.abs_err_completeness.values)), 4),
                            "delta_cont_mae": round(float(np.mean(
                                grp.abs_err_contamination.values
                                - c.abs_err_contamination.values)), 4)})
    pd.DataFrame(strat).to_csv(RESULTS_DIR / "set_G_stratified.tsv",
                               sep="\t", index=False)

    # Duplication accounting sensitivity: MAGICC's denominator says a duplicated
    # segment adds no reference coverage, so truth is unchanged; a
    # marker-duplication tool may legitimately read it as contamination.  Both
    # accountings are reported so the definitional choice is visible.
    dup = long[long.error_type == "uneven_coverage"].copy()
    if len(dup):
        rows = []
        for (tool, rate), d in dup.groupby(["tool", "error_rate"]):
            alt = d.true_contamination_dup_counted.values
            rows.append({
                "tool": tool, "error_rate_pct": round(100 * float(rate), 4),
                "n": int(len(d)),
                "true_cont_primary": round(float(d.true_contamination.mean()), 4),
                "true_cont_dup_counted": round(float(np.mean(alt)), 4),
                "cont_mae_primary": round(float(
                    np.mean(np.abs(d.pred_contamination.values
                                   - d.true_contamination.values))), 4),
                "cont_bias_primary": round(float(
                    np.mean(d.pred_contamination.values
                            - d.true_contamination.values)), 4),
                "cont_mae_dup_counted": round(float(
                    np.mean(np.abs(d.pred_contamination.values - alt))), 4),
                "cont_bias_dup_counted": round(float(
                    np.mean(d.pred_contamination.values - alt)), 4)})
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / "set_G_duplication_accounting_sensitivity.tsv",
            sep="\t", index=False)

    print("\n[5/6] mechanism diagnostics ...")
    kperturb = kmer_perturbation(set_dir, RESULTS_DIR)
    orf = orf_diagnostic(set_dir, RESULTS_DIR)

    print("\n[6/6] figures ...")
    cvd = fw.palette_cvd_report(PALETTE, min_delta_e=20.0)
    cvd.to_csv(RESULTS_DIR / "palette_cvd_check.tsv", sep="\t", index=False)
    print(f"  palette CVD check: min Delta-E {cvd.min_delta_e.min():.1f}, "
          f"{int(cvd.flag_too_similar.sum())} pair(s) flagged")
    fig_palette_check(cvd, FIG_DIR / "fig_ws6_palette_cvd_check")
    fig_substitution_focus(curves, paired, tools,
                           FIG_DIR / "fig_ws6.1_substitution_degradation")
    fig_curves(curves, tools, "completeness", "comp_mae", "completeness MAE (pp)",
               FIG_DIR / "fig_ws6.2_comp_mae_all_error_types",
               "Set G: completeness MAE versus injected error rate")
    fig_curves(curves, tools, "contamination", "cont_mae", "contamination MAE (pp)",
               FIG_DIR / "fig_ws6.3_cont_mae_all_error_types",
               "Set G: contamination MAE versus injected error rate")
    fig_curves(curves, tools, "completeness", "comp_bias",
               "signed completeness error (pp)",
               FIG_DIR / "fig_ws6.4_comp_bias_all_error_types",
               "Set G: signed completeness error (predicted - true)")
    fig_curves(curves, tools, "contamination", "cont_bias",
               "signed contamination error (pp)",
               FIG_DIR / "fig_ws6.5_cont_bias_all_error_types",
               "Set G: signed contamination error (predicted - true)")
    fig_curves(curves, tools, "mimag", "mimag_agreement",
               "MIMAG-inspired class agreement",
               FIG_DIR / "fig_ws6.6_mimag_agreement",
               "Set G: MIMAG-inspired quality-class agreement with the truth")
    fig_paired_degradation(paired, tools, "completeness",
                           FIG_DIR / "fig_ws6.7_paired_comp_degradation")
    fig_paired_degradation(paired, tools, "contamination",
                           FIG_DIR / "fig_ws6.8_paired_cont_degradation")
    if kperturb is not None:
        fig_mechanism(kperturb, orf, FIG_DIR / "fig_ws6.9_mechanism")

    write_captions(long, curves)
    write_report(set_dir, long, curves, paired, h2h, bt, cross, ctx, kperturb,
                 orf, prov, tools)

    # ------------------------------------------------------------- summary
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "set": args.set, "model": ONNX_MODEL,
        "n_samples": int(len(gm)), "n_references": int(gm.ref_index.nunique()),
        "n_phyla": int(gm.dominant_phylum.nunique()),
        "tools": tools,
        "bootstrap": {"n_iter": N_BOOT, "cluster_unit": "reference genome",
                      "seed_source": "fw.stable_hash (CRC-32), PYTHONHASHSEED=0"},
        "provenance": (prov or {}).get("DISJOINTNESS_VERDICT", "not run"),
        "denominators": DENOM_NOTE,
    }
    if len(curves):
        for tool in tools:
            s = curves[(curves.tool == tool) & (curves.error_type == "substitution")
                       ].sort_values("error_rate")
            summary[f"{tool}_substitution_comp_mae_by_rate"] = {
                f"{100 * r:g}%": v for r, v in zip(s.error_rate, s.comp_mae)}
            summary[f"{tool}_substitution_cont_mae_by_rate"] = {
                f"{100 * r:g}%": v for r, v in zip(s.error_rate, s.cont_mae)}
    if len(cross):
        summary["crossover"] = cross.to_dict("records")
    if len(bt):
        summary["boundary_magicc_substitution"] = bt[
            (bt.tool == "magicc_v5") & (bt.error_type == "substitution")
        ].to_dict("records")
    (RESULTS_DIR / "set_G_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    # ------------------------------------------------------------- console
    print("\n" + "=" * 78)
    print("SUBSTITUTION DEGRADATION — completeness / contamination MAE (pp)")
    print("=" * 78)
    piv = curves[curves.error_type == "substitution"].pivot_table(
        index="error_rate_pct", columns="tool", values=["comp_mae", "cont_mae"])
    print(piv.round(3).to_string())
    if len(cross):
        print("\nCROSSOVER (MAGICC vs comparator, substitutions):")
        print(cross[cross.error_type == "substitution"].to_string(index=False))
    print(f"\nwrote {RESULTS_DIR}")
    return 0


def write_report(set_dir: Path, long: pd.DataFrame, curves: pd.DataFrame,
                 paired: pd.DataFrame, h2h: pd.DataFrame, bt: pd.DataFrame,
                 cross: pd.DataFrame, ctx: pd.DataFrame,
                 kp: Optional[pd.DataFrame], orf: Optional[pd.DataFrame],
                 prov: Optional[dict], tools: List[str]) -> None:
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    L: List[str] = []
    A = L.append
    A("# WS6 — Sequencing and assembly error robustness (Set G)")
    A("")
    A(f"Generated {datetime.now(timezone.utc).isoformat()} · model "
      f"`models/magicc_v5.onnx` (frozen) · addresses **R2-o3**.")
    A("")
    A("## Set G composition")
    A("")
    refs = gm[gm.error_type == "none"]
    A(f"- **{len(gm):,} assemblies** = {gm.ref_index.nunique()} held-out reference "
      f"genomes x {gm.arm_index.nunique()} error arms, spanning "
      f"{gm.dominant_phylum.nunique()} phyla "
      f"({', '.join(f'{k} {v} references' for k, v in refs.dominant_domain.value_counts().items())}), "
      f"genome size {refs.dominant_reference_bp.min() / 1e6:.2f}-"
      f"{refs.dominant_reference_bp.max() / 1e6:.2f} Mbp "
      f"(median {refs.dominant_reference_bp.median() / 1e6:.2f}), "
      f"fragmentation tiers balanced over "
      f"{', '.join(f'{k} {v}' for k, v in refs.quality_tier.value_counts().items())} "
      f"(base contig count median {refs.base_n_contigs.median():.0f}, "
      f"max {refs.base_n_contigs.max():.0f}).")
    A(f"- Dominants from `data/splits/test_finished_genomes.tsv` (complete genomes, so "
      f"the full reference length used in the denominator is exact); contaminants "
      f"cross-phylum from the same held-out test split.")
    A(f"- True completeness {gm.true_completeness.min():.1f}-"
      f"{gm.true_completeness.max():.1f} % (mean {gm.true_completeness.mean():.1f}); "
      f"true contamination {gm.true_contamination.min():.1f}-"
      f"{gm.true_contamination.max():.1f} % (mean {gm.true_contamination.mean():.1f}). "
      f"**{int(gm.constraint_violation.sum())}** samples outside the training domain "
      f"(contamination % <= completeness %).")
    A("- Every arm of a reference is applied to **one shared base assembly**, so "
      "control and error arms have an identical ground truth, fragmentation "
      "realisation and contaminants; all degradation estimates are paired within "
      "reference and every CI is a cluster bootstrap over reference genomes.")
    if prov:
        A(f"- **Provenance / disjointness verdict: {prov['DISJOINTNESS_VERDICT']}** "
          f"(GCA<->GCF cross-mapped, {prov.get('n_files_hashed', 0)} files hashed) — "
          f"`results/revision/provenance/{set_dir.name}_provenance_audit.json`.")
    A("")
    A("## Ground truth is unchanged by error injection")
    A("")
    A("Substitutions change which base occupies a position, not which organism the "
      "position derives from, and change no length — truth is exactly invariant. "
      "Chimeras re-partition exactly the same multiset of bases. Uneven-coverage "
      "duplication re-emits sequence already present, so the set of represented "
      "reference positions is unchanged. Indels are the only length-changing "
      "process; they are balanced 50/50 and the residual effect on truth is "
      f"bounded at **{float(gm[gm.error_type == 'indel'].eval('abs(true_completeness_indel_adjusted - true_completeness)').max()):.3f} pp** "
      "of completeness (column `true_completeness_indel_adjusted`). An alternative "
      "accounting in which duplicated bp are scored as contamination is emitted as "
      "`true_contamination_dup_counted`.")
    A("")
    A("## Degradation curves")
    A("")
    for et in ERROR_ORDER:
        s = curves[curves.error_type == et]
        if not len(s):
            continue
        A(f"### {ERROR_LABEL[et]}")
        A("")
        piv = s.pivot_table(index="error_rate_pct", columns="tool",
                            values=["comp_mae", "cont_mae"])
        A("```")
        A(piv.round(3).to_string())
        A("```")
        A("")
    A("## Applicability boundary (WS6.4)")
    A("")
    A("```")
    A(bt[bt.tool == "magicc_v5"].to_string(index=False))
    A("```")
    A("")
    if len(ctx):
        A("Placed against real per-base error rates:")
        A("")
        A("```")
        A(ctx.to_string(index=False))
        A("```")
        A("")
    if len(cross):
        A("## Head-to-head crossover")
        A("")
        A("```")
        A(cross.to_string(index=False))
        A("```")
        A("")
    if kp is not None and len(kp):
        A("## Mechanism")
        A("")
        A("```")
        cols = ["error_type", "error_rate_pct", "observed_kmer_corruption",
                "predicted_kmer_position_corruption", "total_count_ratio",
                "l1_relative", "normalised_l2_per_dim", "summary_l2_per_dim"]
        A(kp[cols].round(4).to_string(index=False))
        A("```")
        A("")
    if orf is not None and len(orf):
        A("```")
        c = [x for x in orf.columns if x.endswith("_ratio_vs_control")
             or x in ("error_type", "error_rate_pct", "n")]
        A(orf[c].to_string(index=False))
        A("```")
        A("")
    dp = RESULTS_DIR / "set_G_duplication_accounting_sensitivity.tsv"
    if dp.exists():
        A("## Duplication accounting sensitivity")
        A("")
        A("Under MAGICC's denominator a duplicated segment adds no reference "
          "coverage, so ground truth is unchanged; a marker-duplication tool may "
          "legitimately read the same segment as contamination. Both accountings:")
        A("")
        A("```")
        A(pd.read_csv(dp, sep="\t").to_string(index=False))
        A("```")
        A("")
    sp = RESULTS_DIR / "set_G_stratified.tsv"
    if sp.exists():
        s = pd.read_csv(sp, sep="\t")
        s = s[(s.stratify_by == "contig_stratum") & (s.error_type == "chimera")]
        if len(s):
            A("## Chimera dose by base contig count")
            A("")
            A("The achievable chimera dose is bounded by the number of contigs "
              "available to mis-join, so the arm is also reported on the "
              "fragmented half of the panel where the dose is large.")
            A("")
            A("```")
            A(s.to_string(index=False))
            A("```")
            A("")
    A("## Conventions honoured")
    A("")
    A("R^2 = coefficient of determination (1 - SS_res/SS_tot) throughout, blank where "
      "the truth has near-zero variance (R1-m19). `fw.stable_hash()` (CRC-32) for "
      "every bootstrap seed with `PYTHONHASHSEED=0`. Two-sided paired Wilcoxon tests, "
      "cluster bootstrap over reference genomes (2,000 iterations), BH correction, "
      "Hodges-Lehmann and Cliff's delta with CIs. MIMAG-inspired thresholds "
      "(completeness/contamination only). Palette CVD-verified in-script; no "
      "red/green discrimination. Denominator stated in every caption "
      "(`figures/captions.md`).")
    (RESULTS_DIR / "WS6_error_robustness_report.md").write_text("\n".join(L) + "\n")
    print(f"  wrote {RESULTS_DIR / 'WS6_error_robustness_report.md'}")


def write_captions(long: pd.DataFrame, curves: pd.DataFrame) -> None:
    n_ref = int(long.ref_index.nunique())
    n_per = int(long[long.tool == long.tool.iloc[0]].groupby("error_type").size().max())
    txt = [
        "# Set G figure captions (WS6, R2-o3)",
        "",
        f"Denominator statement (R1-M5), applying to every panel: {DENOM_NOTE}",
        "",
        f"Design: {n_ref} held-out test-split reference genomes; for each reference a "
        "single base assembly is generated and every error arm is applied to that "
        "same assembly, so the control and every error level share an identical "
        "ground truth, fragmentation realisation and contaminants. Contaminants are "
        "cross-phylum, drawn from the same held-out test split. All intervals are 95 % "
        "percentile cluster bootstraps (2,000 replicates) resampling reference "
        f"genomes; each error-rate cell contains n = {n_ref} assemblies "
        f"(one per reference).",
        "",
        "**fig_ws6.1_substitution_degradation** - the headline panel. Completeness "
        "MAE, contamination MAE, signed completeness error and MIMAG-inspired class "
        "agreement as a function of the per-base substitution rate (0, 0.1, 0.5, 1, 2, "
        "5 %), for MAGICC V5, CheckM2 1.0.1, CoCoPyE 0.5.0 and DeepCheck. The x axis "
        "is logarithmic and rate 0 is plotted at the left-hand tick. The shaded grey "
        "band, bounded by the dashed line at 0.1 %, is the regime real data occupies: "
        "nominal Illumina Q30 raw-base error is 10^-3 per base (0.1 %) and assembly "
        "consensus error is 10^-3 to 10^-5 per base, i.e. at or below that line. "
        "Substitutions do not change ground truth: they alter which base occupies a "
        "position, not which organism the position derives from, and they change no "
        "length.",
        "",
        "**fig_ws6.2 / 6.3** - completeness and contamination MAE against rate for all "
        "five error processes. **fig_ws6.4 / 6.5** - the corresponding signed errors "
        "(predicted - true; negative = under-estimate). **fig_ws6.6** - MIMAG-inspired "
        "(completeness/contamination only) quality-class agreement with the truth. "
        "**fig_ws6.7 / 6.8** - paired degradation: the change in absolute error "
        "relative to the *same reference genome's* error-free assembly, which is the "
        "estimator that isolates the injected error from between-genome variation.",
        "",
        "**fig_ws6_palette_cvd_check** - verification that the four-tool palette "
        "satisfies the editorial colour-vision-deficiency requirement (E4) under "
        "normal, deuteranopic, protanopic and tritanopic vision. No green is used "
        "anywhere, so no panel requires red/green discrimination; colour is always "
        "accompanied by a distinct marker shape.",
        "",
        "Error-process definitions are given in scripts/150_error_injection_module.py "
        "and every realised rate (substitutions injected, Ti/Tv achieved, indel events "
        "per kb, mis-joins formed, bp duplicated) is recorded per sample in "
        "data/benchmarks/set_G/generation_metadata.tsv.",
    ]
    (FIG_DIR / "captions.md").write_text("\n".join(txt) + "\n")


if __name__ == "__main__":
    sys.exit(main())
