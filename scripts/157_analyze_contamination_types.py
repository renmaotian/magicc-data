#!/usr/bin/env python3
"""
WS2.5 — Attribution analysis for Set F: does MAGICC's advantage come
specifically from **non-redundant** contamination, or uniformly across types?

(Protocol name: ``84_analyze_contamination_types.py``; renumbered to 147 - see
the header of ``154_contamination_type_module.py``.)

WHAT THIS SCRIPT DOES
---------------------
1. Runs frozen MAGICC V5 on Set F by direct ONNX inference (the pattern of
   ``scripts/075_run_magicc_clean_cd.py``; the CLI is deliberately not used).
2. Loads CheckM2 / CoCoPyE / DeepCheck predictions produced by
   ``scripts/156_run_tools_set_F.sh`` -> ``scripts/091_parse_competitor_clean_cd.py``.
   GUNC is out of scope here (WS4.2 owns it) and is a detection comparator, not
   a quantitative estimator, so it would not belong in these tables anyway.
3. Runs a provenance audit on Set F in the style of
   ``scripts/074_provenance_audit.py`` (GCA<->GCF cross-mapped), emitting
   accession lists and a SHA256 manifest.  Disjointness is proven, not asserted.
4. Produces the WS2 acceptance deliverable: a **type x distance heatmap of
   SIGNED contamination error per tool**, plus the matching tables.
5. Tests the attribution hypothesis explicitly, per cell and honestly.

CONVENTIONS (binding, protocol sections 4.4d / 5.4 / 9.13)
----------------------------------------------------------
* R^2 is ALWAYS the coefficient of determination, 1 - SS_res/SS_tot.
* Signed error = predicted - true.  Negative = UNDER-estimate.
* Two-sided paired tests; cluster bootstrap over the 100 reference genomes;
  Benjamini-Hochberg FDR; Hodges-Lehmann and Cliff's delta with CIs beside
  every p-value.
* MIMAG-inspired thresholds: HQ >= 90% completeness AND < 5% contamination;
  MQ >= 50% AND < 10%.
* Figures use the project's validated CVD-safe palette (no red/green
  discrimination), verified in-script; the denominator is stated in every
  caption.

USAGE
-----
    python scripts/157_analyze_contamination_types.py
    python scripts/157_analyze_contamination_types.py --set set_F_pilot --quick
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

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load(PROJECT_DIR / "scripts" / "101_metrics_framework.py", "magicc_metrics_framework")

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

DATA_DIR = PROJECT_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
RESULTS_DIR = PROJECT_DIR / "results" / "revision" / "set_F"
FIG_DIR = RESULTS_DIR / "figures"

SELECTED_KMERS = str(DATA_DIR / "kmer_selection" / "selected_kmers.txt")
NORMALIZATION = str(DATA_DIR / "features" / "normalization_params.json")
ONNX_MODEL = str(PROJECT_DIR / "models" / "magicc_v5.onnx")

N_BOOT = 2000
BOOT_SEED = 8_200_500
DISTANCE_ORDER = ["species", "genus", "family", "order", "class", "phylum"]
TYPE_ORDER = ["redundant", "replaced", "single"]
NONREDUNDANT = ("replaced", "single")

# Validated CVD-safe palette already in production use by scripts/105
# (Okabe-Ito failed its own CVD check on this project; this is the replacement).
PALETTE = {
    "magicc_v5": "#0072B2",   # blue
    "checkm2":   "#D55E00",   # vermillion
    "cocopye":   "#F0E442",   # yellow
    "deepcheck": "#000000",   # black
}
MARKERS = {"magicc_v5": "o", "checkm2": "^", "cocopye": "D", "deepcheck": "v"}
TOOL_LABEL = {"magicc_v5": "MAGICC V5", "checkm2": "CheckM2 1.0.1",
              "cocopye": "CoCoPyE 0.5.0", "deepcheck": "DeepCheck"}

DENOM_NOTE = ("Denominators: completeness = retained dominant bp / dominant full "
              "reference bp x 100; contamination = total contaminant bp / dominant "
              "full reference bp x 100. Both are percentages of the SAME denominator "
              "(the dominant genome's full reference length), so they are independent "
              "and contamination can exceed 100% in principle.")


# ==========================================================================
# MAGICC V5 inference (direct ONNX; pattern of scripts/75)
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
    except Exception as e:                                       # noqa: BLE001
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
    return res


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
    print("PROVENANCE AUDIT — Set F")
    print("=" * 78)
    out_dir.mkdir(parents=True, exist_ok=True)
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    cmap, cstats = build_crossmap()

    splits = {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(DATA_DIR / "splits" / f"{s}_genomes.tsv", sep="\t")
        accs = list(d.gtdb_accession) + [a for a in d.get("ncbi_accession", []) if isinstance(a, str)]
        splits[s] = canon(accs, cmap)

    kmer_sel = []
    for f in ("selected_bacterial_1000.tsv", "selected_archaeal_1000.tsv"):
        fp = DATA_DIR / "kmer_selection" / f
        if fp.exists():
            kmer_sel += pd.read_csv(fp, sep="\t")["gtdb_accession"].tolist()
    _, kmer_canon, _ = canon(kmer_sel, cmap)

    doms = sorted(set(gm.dominant_accession))
    cons = sorted({x for s in gm.donors_used.fillna("") for x in str(s).split(";") if x})
    _, dom_canon, dom_unmapped = canon(doms, cmap)
    _, con_canon, con_unmapped = canon(cons, cmap)

    rows = []
    for label, cs, raw in (("dominants", dom_canon, doms), ("contaminants", con_canon, cons)):
        r = {"group": label, "n_accessions": len(raw), "n_canonical": len(cs),
             "n_unmapped": dom_unmapped if label == "dominants" else con_unmapped}
        for s in ("train", "val", "test"):
            r[f"in_{s}"] = len(cs & splits[s][1])
        r["in_kmer_selection_set"] = len(cs & kmer_canon) if kmer_canon else "n/a"
        rows.append(r)
    ov = pd.DataFrame(rows)
    ov.to_csv(out_dir / "set_F_overlap_summary.tsv", sep="\t", index=False)
    print(ov.to_string(index=False))

    # per-sample counts (a sample is leaked if its dominant is)
    dom_split = {}
    for a in doms:
        _, c, _ = canon([a], cmap)
        dom_split[a] = ";".join(s for s in ("train", "val", "test") if c & splits[s][1]) or "none"
    gm["dominant_split_crossmapped"] = gm.dominant_accession.map(dom_split)
    per_sample = gm.dominant_split_crossmapped.value_counts().to_dict()

    (out_dir / "set_F_dominants.txt").write_text("\n".join(doms) + "\n")
    (out_dir / "set_F_contaminants.txt").write_text("\n".join(cons) + "\n")

    verdict = (int(ov.loc[ov.group == "dominants", "in_train"].iloc[0]) == 0
               and int(ov.loc[ov.group == "dominants", "in_val"].iloc[0]) == 0
               and int(ov.loc[ov.group == "contaminants", "in_train"].iloc[0]) == 0
               and int(ov.loc[ov.group == "contaminants", "in_val"].iloc[0]) == 0)

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
        with open(out_dir / "set_F_sha256_manifest.txt", "w") as fh:
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
           "note": ("Dominants and contaminants are both restricted to "
                    "data/splits/test_genomes.tsv. Accessions are compared after "
                    "GCA<->GCF cross-mapping through data/gtdb/filtered_genomes.tsv "
                    "(277,183 assemblies), the normalisation whose absence caused the "
                    "Set C/D leakage undercount.")}
    (out_dir / "set_F_provenance_audit.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"  DISJOINTNESS VERDICT: {rep['DISJOINTNESS_VERDICT']}")
    return rep


# ==========================================================================
# metrics
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
    keep = ["genome_id", "ref_index", "dominant_accession", "dominant_phylum",
            "contamination_type", "distance", "distance_index", "type_index",
            "target_contamination", "observed_completeness", "observed_contamination",
            "quality_tier", "dominant_reference_bp", "n_contigs", "total_length",
            "n_duplicated_markers", "n_replaced_markers", "n_novel_genes",
            "shortfall_bp", "in_training_domain"]
    base = gm[[c for c in keep if c in gm.columns]].rename(
        columns={"observed_completeness": "true_completeness",
                 "observed_contamination": "true_contamination"})
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
    return long


def cell_stats(df: pd.DataFrame, seed: int) -> dict:
    bs = fw.Bootstrapper(clusters=df.ref_index.values, n_iter=N_BOOT, seed=seed)
    ec = df.err_contamination.values
    ecm = df.err_completeness.values
    tc = df.true_contamination.values
    pc = df.pred_contamination.values
    tk = df.true_completeness.values
    pk = df.pred_completeness.values

    def stat(idx):
        return {
            "cont_mae": float(np.mean(np.abs(ec[idx]))),
            "cont_bias": float(np.mean(ec[idx])),
            "comp_mae": float(np.mean(np.abs(ecm[idx]))),
            "comp_bias": float(np.mean(ecm[idx])),
        }

    def slope(idx):
        """OLS slope of predicted on true contamination.

        This is the single most interpretable expression of the Cornet result:
        slope 1 = the tool tracks contamination fully, slope 0 = it is blind to
        it. Estimable here because each cell spans the 1-20% ladder.
        """
        t, p = tc[idx], pc[idx]
        v = float(np.var(t))
        if v <= 1e-12:
            return np.nan
        return float(np.cov(t, p, ddof=0)[0, 1] / v)

    r = bs.ci_multi(stat)
    sl = bs.ci(slope)
    out = {"n": int(len(df)), "n_clusters": int(df.ref_index.nunique()),
           "detection_slope": round(sl["estimate"], 4),
           "detection_slope_ci_lo": round(sl["ci_lo"], 4),
           "detection_slope_ci_hi": round(sl["ci_hi"], 4)}
    for k, v in r.items():
        out[k] = round(v["estimate"], 4)
        out[f"{k}_ci_lo"] = round(v["ci_lo"], 4)
        out[f"{k}_ci_hi"] = round(v["ci_hi"], 4)
    out["cont_r2"] = round(fw.r2_coefficient_of_determination(tc, pc), 4)
    out["comp_r2"] = round(fw.r2_coefficient_of_determination(tk, pk), 4)
    out["true_cont_mean"] = round(float(np.mean(tc)), 3)
    out["true_comp_mean"] = round(float(np.mean(tk)), 3)
    return out


def paired_compare(a: pd.DataFrame, b: pd.DataFrame, seed: int) -> dict:
    """Two-sided paired comparison of |contamination error|, a (MAGICC) vs b."""
    from scipy import stats as sps
    m = a.merge(b, on="genome_id", suffixes=("_a", "_b"), validate="one_to_one")
    d = m.abs_err_contamination_a.values - m.abs_err_contamination_b.values
    if len(d) < 3 or np.allclose(d, 0):
        return {"n_pairs": int(len(d)), "hl": 0.0, "p_wilcoxon": 1.0,
                "cliffs_delta": 0.0}
    try:
        p = float(sps.wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        p = 1.0
    bs = fw.Bootstrapper(clusters=m.ref_index_a.values, n_iter=N_BOOT, seed=seed)
    hl_ci = bs.ci(lambda idx: fw.hodges_lehmann_paired(d[idx]))
    dl_ci = bs.ci(lambda idx: fw.cliffs_delta(m.abs_err_contamination_a.values[idx],
                                              m.abs_err_contamination_b.values[idx]))
    return {"n_pairs": int(len(d)),
            "hl": round(hl_ci["estimate"], 4),
            "hl_ci_lo": round(hl_ci["ci_lo"], 4), "hl_ci_hi": round(hl_ci["ci_hi"], 4),
            "cliffs_delta": round(dl_ci["estimate"], 4),
            "cliffs_ci_lo": round(dl_ci["ci_lo"], 4),
            "cliffs_ci_hi": round(dl_ci["ci_hi"], 4),
            "p_wilcoxon": p}


def mimag_rates(df: pd.DataFrame) -> dict:
    """MIMAG-inspired: HQ >= 90% comp AND < 5% cont; MQ >= 50% AND < 10%."""
    t_hq = (df.true_completeness >= 90) & (df.true_contamination < 5)
    p_hq = (df.pred_completeness >= 90) & (df.pred_contamination < 5)
    tp = int((t_hq & p_hq).sum()); fp = int((~t_hq & p_hq).sum())
    fn = int((t_hq & ~p_hq).sum()); tn = int((~t_hq & ~p_hq).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"n": int(len(df)), "n_true_HQ": int(t_hq.sum()), "n_pred_HQ": int(p_hq.sum()),
            "HQ_tp": tp, "HQ_fp": fp, "HQ_fn": fn, "HQ_tn": tn,
            "HQ_precision": round(prec, 4) if prec == prec else "",
            "HQ_recall": round(rec, 4) if rec == rec else "",
            "false_clean_rate_5pct": round(
                float(((df.true_contamination >= 5) & (df.pred_contamination < 5)).sum()
                      / max(1, int((df.true_contamination >= 5).sum()))), 4),
            "false_dirty_rate_5pct": round(
                float(((df.true_contamination < 5) & (df.pred_contamination >= 5)).sum()
                      / max(1, int((df.true_contamination < 5).sum()))), 4)}


# ==========================================================================
# Quantitative characterisation of the taxonomic-distance axis
# ==========================================================================
def distance_characterization(set_dir: Path) -> pd.DataFrame:
    """Make the distance axis a MEASURED quantity, not just a GTDB label.

    For every (dominant, donor) pair actually used, report the fraction of donor
    genes with a reciprocal-best-hit orthologue in the dominant.  This is the
    quantity that drives the redundant/single distinction, so it is the honest
    way to show that 'species -> phylum' really is a gradient (R1-m4).
    """
    design = pd.read_csv(set_dir / "design.tsv", sep="\t")
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    used = set()
    for r in gm.itertuples():
        for d in str(getattr(r, "donors_used", "") or "").split(";"):
            if d:
                used.add((r.dominant_accession, d, r.distance))
    ann_root = BENCHMARK_DIR / "set_F" / "annotations"
    ort_root = BENCHMARK_DIR / "set_F" / "orthology"
    rows = []
    gene_cache: Dict[str, int] = {}

    def n_genes(acc):
        if acc not in gene_cache:
            p = ann_root / acc.replace("/", "_") / "genes.tsv"
            gene_cache[acc] = (sum(1 for _ in open(p)) - 1) if p.exists() else 0
        return gene_cache[acc]

    for dom, don, dist in sorted(used):
        p = ort_root / f"{dom.replace('/', '_')}__{don.replace('/', '_')}.tsv"
        if not p.exists():
            continue
        n_rbh = sum(1 for _ in open(p)) - 1
        ng = n_genes(don)
        rows.append({"distance": dist, "dominant": dom, "donor": don,
                     "n_donor_genes": ng, "n_rbh_orthologues": n_rbh,
                     "pct_donor_genes_orthologous": round(100 * n_rbh / max(1, ng), 2)})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(RESULTS_DIR / "set_F_distance_characterization.tsv",
                  sep="\t", index=False)
        summ = (df.groupby("distance")["pct_donor_genes_orthologous"]
                .agg(["count", "mean", "median",
                      lambda s: s.quantile(0.05), lambda s: s.quantile(0.95)]))
        summ.columns = ["n_pairs", "mean_pct_orthologous", "median_pct_orthologous",
                        "p05", "p95"]
        summ = summ.reindex(DISTANCE_ORDER).round(2)
        summ.to_csv(RESULTS_DIR / "set_F_distance_summary.tsv", sep="\t")
        print("\n" + "=" * 78)
        print("TAXONOMIC-DISTANCE AXIS — % of donor genes with an RBH orthologue "
              "in the dominant")
        print("=" * 78)
        print(summ.to_string())
    return df


# ==========================================================================
# Independent mechanistic validation of the three contamination types
# ==========================================================================
BACT_HMM = PROJECT_DIR / "85_bcg.hmm"
ARCH_HMM = PROJECT_DIR / "uacg.hmm"


def _dup_worker(args):
    """Prodigal + hmmsearch --cut_tc on one generated Set F assembly.

    Returns the number of single-copy core-gene FAMILIES observed >= 2 times.
    This is a genuinely independent check of the operational definitions: it
    uses the project's own SCG profiles on the finished FASTA and knows nothing
    about how the sample was built.  REDUNDANT must inflate marker duplication;
    REPLACED and SINGLE must not.
    """
    import subprocess
    import tempfile
    gid, path, domain = args
    try:
        with tempfile.TemporaryDirectory() as td:
            faa = os.path.join(td, "p.faa")
            r = subprocess.run(["prodigal", "-i", path, "-a", faa, "-p", "single",
                                "-q", "-o", os.path.join(td, "p.gff")],
                               capture_output=True, text=True, timeout=3600)
            if r.returncode != 0 or not os.path.exists(faa):
                return gid, None, None, "prodigal failed"
            hmm = ARCH_HMM if str(domain).startswith("Arch") else BACT_HMM
            tbl = os.path.join(td, "h.tbl")
            r = subprocess.run(["hmmsearch", "--cut_tc", "--tblout", tbl, "--noali",
                                "--cpu", "1", str(hmm), faa],
                               capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                return gid, None, None, "hmmsearch failed"
            fam = defaultdict(set)
            with open(tbl) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    f = line.split()
                    if len(f) < 3:
                        continue
                    fam[f[2]].add(f[0])
            n_fam = len(fam)
            n_dup = sum(1 for v in fam.values() if len(v) >= 2)
            return gid, n_fam, n_dup, None
    except Exception as e:                                       # noqa: BLE001
        return gid, None, None, f"{type(e).__name__}: {e}"


def marker_duplication_validation(set_dir: Path, long: pd.DataFrame, workers: int,
                                  per_cell: int = 6) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("INDEPENDENT MECHANISTIC VALIDATION — single-copy core-gene duplication")
    print("=" * 78)
    out = RESULTS_DIR / "set_F_marker_duplication_validation.tsv"
    base = long[long.tool == long.tool.iloc[0]].copy()
    picks = []
    rng = np.random.default_rng(BOOT_SEED + 991)
    for (ty, di), sub in base.groupby(["contamination_type", "distance"]):
        k = min(per_cell if ty != "none" else per_cell * 3, len(sub))
        sel = sub.sort_values("genome_id").iloc[
            rng.choice(len(sub), size=k, replace=False)]
        picks.append(sel)
    sel = pd.concat(picks, ignore_index=True)
    if out.exists():
        old = pd.read_csv(out, sep="\t")
        if set(sel.genome_id) <= set(old.genome_id):
            print(f"  reusing cached duplication scan ({len(old)} genomes)")
            return old
    work = [(r.genome_id, str(set_dir / "fasta" / f"{r.genome_id}.fasta"),
             r.dominant_phylum) for r in sel.itertuples()]
    print(f"  scanning {len(work)} assemblies ({workers} workers) ...")
    t0 = time.time()
    res = {}
    with Pool(workers) as pool:
        for gid, n_fam, n_dup, err in pool.imap_unordered(_dup_worker, work, chunksize=1):
            res[gid] = (n_fam, n_dup, err)
    print(f"    done in {time.time()-t0:.0f}s")
    sel["n_scg_families_detected"] = sel.genome_id.map(lambda g: res.get(g, (None,))[0])
    sel["n_scg_families_duplicated"] = sel.genome_id.map(
        lambda g: res.get(g, (None, None))[1])
    sel["scan_error"] = sel.genome_id.map(lambda g: res.get(g, (None, None, ""))[2])
    keep = ["genome_id", "ref_index", "contamination_type", "distance",
            "target_contamination", "true_contamination", "true_completeness",
            "dominant_phylum", "n_duplicated_markers", "n_replaced_markers",
            "n_novel_genes", "n_scg_families_detected", "n_scg_families_duplicated",
            "scan_error"]
    sel[[c for c in keep if c in sel.columns]].to_csv(out, sep="\t", index=False)
    ok = sel[sel.n_scg_families_duplicated.notna()]
    summ = ok.groupby("contamination_type")["n_scg_families_duplicated"].agg(
        ["count", "mean", "median", "max"]).round(2)
    print(summ.to_string())
    ctrl = float(ok[ok.contamination_type == "none"]["n_scg_families_duplicated"].mean()) \
        if (ok.contamination_type == "none").any() else float("nan")
    for ty in TYPE_ORDER:
        v = ok[ok.contamination_type == ty]["n_scg_families_duplicated"]
        if len(v):
            print(f"  {ty:10s}: +{v.mean()-ctrl:6.2f} duplicated SCG families "
                  f"vs uncontaminated control ({ctrl:.2f})")
    return sel


# ==========================================================================
# figures
# ==========================================================================
def diverging_cmap(plt):
    """Blue <-> white <-> vermillion. CVD-safe (verified in-script), no green."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "magicc_bwv", ["#0072B2", "#7FB6DA", "#F2F2F2", "#EDA678", "#D55E00"], N=256)


def heatmap_figure(cells: pd.DataFrame, tools: List[str], out_stem: Path,
                   value_col: str, title: str, unit: str) -> None:
    plt = fw.init_matplotlib()
    n = len(tools)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 1.0, 3.05), sharey=True)
    if n == 1:
        axes = [axes]
    vmax = float(np.nanmax(np.abs(cells[value_col].values))) if len(cells) else 1.0
    vmax = max(vmax, 1e-6)
    cmap = diverging_cmap(plt)
    im = None
    for ax, tool in zip(axes, tools):
        sub = cells[cells.tool == tool]
        M = np.full((len(TYPE_ORDER), len(DISTANCE_ORDER)), np.nan)
        for i, ty in enumerate(TYPE_ORDER):
            for j, di in enumerate(DISTANCE_ORDER):
                v = sub[(sub.contamination_type == ty) & (sub.distance == di)][value_col]
                if len(v):
                    M[i, j] = float(v.iloc[0])
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(DISTANCE_ORDER)))
        ax.set_xticklabels(DISTANCE_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(len(TYPE_ORDER)))
        ax.set_yticklabels(TYPE_ORDER)
        ax.set_title(TOOL_LABEL.get(tool, tool), fontsize=8.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    continue
                shade = abs(M[i, j]) / vmax
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6.4,
                        color="white" if shade > 0.6 else "#111111")
        ax.set_xlabel("taxonomic distance of contaminant")
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    axes[0].set_ylabel("contamination type")
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label(unit, fontsize=7.5)
    fig.suptitle(title, fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def bias_vs_distance_figure(cells: pd.DataFrame, tools: List[str], out_stem: Path) -> None:
    plt = fw.init_matplotlib()
    fig, axes = plt.subplots(1, len(TYPE_ORDER), figsize=(9.2, 2.7), sharey=True)
    for ax, ty in zip(axes, TYPE_ORDER):
        for tool in tools:
            sub = cells[(cells.tool == tool) & (cells.contamination_type == ty)]
            sub = sub.set_index("distance").reindex(DISTANCE_ORDER).reset_index()
            x = np.arange(len(DISTANCE_ORDER))
            y = sub.cont_bias.values
            lo = sub.cont_bias_ci_lo.values
            hi = sub.cont_bias_ci_hi.values
            ax.plot(x, y, marker=MARKERS.get(tool, "o"), ms=3.6, lw=1.1,
                    color=PALETTE.get(tool, "#444444"),
                    markeredgecolor="#222222", markeredgewidth=0.4,
                    label=TOOL_LABEL.get(tool, tool))
            ax.fill_between(x, lo, hi, color=PALETTE.get(tool, "#444444"), alpha=0.13,
                            linewidth=0)
        ax.axhline(0, color="#666666", lw=0.7, ls="--")
        ax.set_xticks(np.arange(len(DISTANCE_ORDER)))
        ax.set_xticklabels(DISTANCE_ORDER, rotation=45, ha="right")
        ax.set_title(ty, fontsize=9)
        ax.set_xlabel("taxonomic distance")
    axes[0].set_ylabel("signed contamination error\n(predicted - true, pp)")
    axes[-1].legend(frameon=False, loc="lower left", fontsize=6.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ==========================================================================
# main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="set_F")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--skip-provenance", action="store_true")
    ap.add_argument("--force-magicc", action="store_true")
    ap.add_argument("--skip-duplication-scan", action="store_true")
    ap.add_argument("--dup-per-cell", type=int, default=6)
    args = ap.parse_args()

    set_dir = BENCHMARK_DIR / args.set
    global RESULTS_DIR, FIG_DIR
    RESULTS_DIR = PROJECT_DIR / "results" / "revision" / args.set
    FIG_DIR = RESULTS_DIR / "figures"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"WS2.5 — contamination type x taxonomic distance attribution ({args.set})")
    print("=" * 78)

    # --- CVD verification of the palette actually used ---------------------
    cvd = fw.palette_cvd_report(PALETTE, min_delta_e=20.0)
    cvd.to_csv(RESULTS_DIR / "palette_cvd_check.tsv", sep="\t", index=False)
    n_flag = int(cvd.flag_too_similar.sum()) if len(cvd) else 0
    print(f"  palette CVD check: min Delta-E {cvd.min_delta_e.min():.1f} across "
          f"normal/deuteranopic/protanopic/tritanopic vision; "
          f"{n_flag} pair(s) flagged too similar")
    # the diverging heatmap ramp is checked too (blue <-> vermillion, no green)
    ramp = {"neg_end": "#0072B2", "mid": "#F2F2F2", "pos_end": "#D55E00"}
    ramp_cvd = fw.palette_cvd_report(ramp, min_delta_e=20.0)
    ramp_cvd.to_csv(RESULTS_DIR / "heatmap_ramp_cvd_check.tsv", sep="\t", index=False)
    print(f"  heatmap ramp CVD check: min Delta-E {ramp_cvd.min_delta_e.min():.1f}, "
          f"{int(ramp_cvd.flag_too_similar.sum())} pair(s) flagged")

    run_magicc(set_dir, args.workers, force=args.force_magicc)
    preds = load_predictions(set_dir)
    if "magicc_v5" not in preds:
        print("FATAL: MAGICC predictions missing", file=sys.stderr)
        return 1
    long = build_long(set_dir, preds)
    long.to_csv(RESULTS_DIR / "set_F_long_predictions.tsv", sep="\t", index=False)
    tools = [t for t in ["magicc_v5", "checkm2", "cocopye", "deepcheck"] if t in preds]
    print(f"\n  tools available: {tools}")

    prov = None
    if not args.skip_provenance:
        prov = provenance_audit(set_dir, PROJECT_DIR / "results" / "revision" / "provenance",
                                args.workers, manifest=not args.no_manifest)

    dist_char = distance_characterization(set_dir)

    dup = None
    if not args.skip_duplication_scan:
        dup = marker_duplication_validation(set_dir, long, args.workers,
                                            per_cell=args.dup_per_cell)

    contaminated = long[long.contamination_type != "none"].copy()
    controls = long[long.contamination_type == "none"].copy()

    # ---------------------------------------------------------------- cells
    print("\n" + "=" * 78)
    print("PER-CELL METRICS (type x distance x tool)")
    print("=" * 78)
    rows = []
    for tool in tools:
        for ty in TYPE_ORDER:
            for di in DISTANCE_ORDER:
                sub = contaminated[(contaminated.tool == tool)
                                   & (contaminated.contamination_type == ty)
                                   & (contaminated.distance == di)]
                if not len(sub):
                    continue
                r = {"tool": tool, "contamination_type": ty, "distance": di}
                r.update(cell_stats(sub, BOOT_SEED + fw.stable_hash(f"{tool}{ty}{di}") % 10000))
                rows.append(r)
    cells = pd.DataFrame(rows)
    cells.to_csv(RESULTS_DIR / "set_F_cells_type_x_distance.tsv", sep="\t", index=False)

    # Sensitivity: 109/1,800 samples could not reach their target contamination
    # from the donor's eligible territory (mostly SINGLE at species distance,
    # where a conspecific donor has few singleton genes). Ground truth is the
    # OBSERVED contamination, so those samples are valid - but the cell means
    # are re-reported with them removed so the reader can see it changes nothing.
    gm_short = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")[
        ["genome_id", "shortfall_bp", "target_contaminant_bp"]]
    cs = contaminated.merge(gm_short, on="genome_id", suffixes=("", "_gm"))
    cs["rel_shortfall"] = cs.shortfall_bp_gm / cs.target_contaminant_bp.clip(lower=1)
    keep = cs[cs.rel_shortfall <= 0.05]
    srows = []
    for tool in tools:
        for ty in TYPE_ORDER:
            for di in DISTANCE_ORDER:
                sub = keep[(keep.tool == tool) & (keep.contamination_type == ty)
                           & (keep.distance == di)]
                if not len(sub):
                    continue
                srows.append({"tool": tool, "contamination_type": ty, "distance": di,
                              "n": int(len(sub)),
                              "n_dropped": int(((cs.tool == tool)
                                                & (cs.contamination_type == ty)
                                                & (cs.distance == di)).sum()) - len(sub),
                              "cont_bias": round(float(sub.err_contamination.mean()), 4),
                              "cont_mae": round(float(sub.abs_err_contamination.mean()), 4)})
    pd.DataFrame(srows).to_csv(
        RESULTS_DIR / "set_F_cells_sensitivity_no_shortfall.tsv", sep="\t", index=False)

    # marginals
    marg = []
    for tool in tools:
        for key, col in (("type", "contamination_type"), ("distance", "distance")):
            for lev in (TYPE_ORDER if key == "type" else DISTANCE_ORDER):
                sub = contaminated[(contaminated.tool == tool) & (contaminated[col] == lev)]
                if not len(sub):
                    continue
                r = {"tool": tool, "margin": key, "level": lev}
                r.update(cell_stats(sub, BOOT_SEED + 7 + fw.stable_hash(f"{tool}{key}{lev}") % 9999))
                r.update({f"mimag_{k}": v for k, v in mimag_rates(sub).items()})
                marg.append(r)
        # contamination-level bands (Reviewer 2: the 0-20% band is what matters)
        for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 21)):
            sub = contaminated[(contaminated.tool == tool)
                               & (contaminated.target_contamination >= lo)
                               & (contaminated.target_contamination < hi)]
            if not len(sub):
                continue
            r = {"tool": tool, "margin": "true_contamination_band",
                 "level": f"[{lo},{min(hi,20)}]%"}
            r.update(cell_stats(sub, BOOT_SEED + 5 + fw.stable_hash(f"{tool}{lo}") % 9999))
            r.update({f"mimag_{k}": v for k, v in mimag_rates(sub).items()})
            marg.append(r)
        # per type x contamination band (the practically relevant cross-tabulation)
        for ty in TYPE_ORDER:
            for lo, hi in ((0, 5), (5, 10), (10, 21)):
                sub = contaminated[(contaminated.tool == tool)
                                   & (contaminated.contamination_type == ty)
                                   & (contaminated.target_contamination >= lo)
                                   & (contaminated.target_contamination < hi)]
                if not len(sub):
                    continue
                r = {"tool": tool, "margin": "type_x_band",
                     "level": f"{ty}|[{lo},{min(hi,20)}]%"}
                r.update(cell_stats(sub, BOOT_SEED + 3
                                    + fw.stable_hash(f"{tool}{ty}{lo}") % 9999))
                marg.append(r)
        # non-redundant pooled + overall + uncontaminated controls
        for lev, sub in (("ALL_contaminated", contaminated[contaminated.tool == tool]),
                         ("NON_REDUNDANT(replaced+single)",
                          contaminated[(contaminated.tool == tool)
                                       & contaminated.contamination_type.isin(NONREDUNDANT)]),
                         ("UNCONTAMINATED_CONTROL", controls[controls.tool == tool])):
            if not len(sub):
                continue
            r = {"tool": tool, "margin": "pooled", "level": lev}
            r.update(cell_stats(sub, BOOT_SEED + 13 + fw.stable_hash(f"{tool}{lev}") % 9999))
            r.update({f"mimag_{k}": v for k, v in mimag_rates(sub).items()})
            marg.append(r)
    margins = pd.DataFrame(marg)
    margins.to_csv(RESULTS_DIR / "set_F_marginals.tsv", sep="\t", index=False)

    show = margins[margins.margin.isin(["type", "pooled"])][
        ["tool", "margin", "level", "n", "cont_mae", "cont_mae_ci_lo", "cont_mae_ci_hi",
         "cont_bias", "cont_bias_ci_lo", "cont_bias_ci_hi", "cont_r2",
         "detection_slope", "detection_slope_ci_lo", "detection_slope_ci_hi"]]
    print(show.to_string(index=False))

    # ------------------------------------------------- paired comparisons
    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS — |contamination error|, MAGICC V5 vs each competitor")
    print("=" * 78)
    comp_rows = []
    others = [t for t in tools if t != "magicc_v5"]
    strata = ([("type", ty, contaminated[contaminated.contamination_type == ty])
               for ty in TYPE_ORDER]
              + [("distance", di, contaminated[contaminated.distance == di])
                 for di in DISTANCE_ORDER]
              + [("cell", f"{ty}|{di}",
                  contaminated[(contaminated.contamination_type == ty)
                               & (contaminated.distance == di)])
                 for ty in TYPE_ORDER for di in DISTANCE_ORDER]
              + [("pooled", "ALL", contaminated),
                 ("pooled", "NON_REDUNDANT", contaminated[
                     contaminated.contamination_type.isin(NONREDUNDANT)])])
    for kind, lev, sub in strata:
        a = sub[sub.tool == "magicc_v5"]
        if not len(a):
            continue
        for o in others:
            b = sub[sub.tool == o]
            if not len(b):
                continue
            r = {"stratum": kind, "level": lev, "tool_a": "magicc_v5", "tool_b": o}
            r.update(paired_compare(a, b, BOOT_SEED + 31
                                    + fw.stable_hash(f"{kind}{lev}{o}") % 9999))
            r["mae_a"] = round(float(a.abs_err_contamination.mean()), 4)
            r["mae_b"] = round(float(b.abs_err_contamination.mean()), 4)
            r["bias_a"] = round(float(a.err_contamination.mean()), 4)
            r["bias_b"] = round(float(b.err_contamination.mean()), 4)
            comp_rows.append(r)
    comparisons = pd.DataFrame(comp_rows)
    if len(comparisons):
        comparisons["q_bh"] = fw.bh_correct(comparisons.p_wilcoxon.values)
        comparisons["winner"] = np.where(
            comparisons.q_bh >= 0.05, "tie",
            np.where(comparisons.hl < 0, "magicc_v5", comparisons.tool_b))
    comparisons.to_csv(RESULTS_DIR / "set_F_paired_comparisons.tsv", sep="\t", index=False)
    if len(comparisons):
        print(comparisons[comparisons.stratum.isin(["type", "pooled"])][
            ["stratum", "level", "tool_b", "n_pairs", "mae_a", "mae_b", "hl",
             "hl_ci_lo", "hl_ci_hi", "cliffs_delta", "q_bh", "winner"]].to_string(index=False))

    # ------------------------------------------- attribution: type-specific?
    print("\n" + "=" * 78)
    print("ATTRIBUTION — is MAGICC's advantage specific to NON-REDUNDANT contamination?")
    print("=" * 78)
    # The contrast is PAIRED BY REFERENCE GENOME: every reference contributes to
    # both the redundant and the non-redundant arm, so an unpaired test would
    # ignore the very clustering the protocol requires us to respect (WS5.4).
    from scipy import stats as sps

    def advantage(frame, competitor):
        """|err_competitor| - |err_MAGICC| per sample; positive = MAGICC better."""
        a = frame[frame.tool == "magicc_v5"][["genome_id", "ref_index",
                                              "abs_err_contamination"]]
        b = frame[frame.tool == competitor][["genome_id", "abs_err_contamination"]]
        m = a.merge(b, on="genome_id", suffixes=("_a", "_b"))
        m["adv"] = m.abs_err_contamination_b - m.abs_err_contamination_a
        return m

    attrib = []
    arms = [("NON_REDUNDANT(replaced+single)", NONREDUNDANT),
            ("replaced", ("replaced",)), ("single", ("single",))]
    for o in others:
        for di in ["ALL"] + DISTANCE_ORDER:
            sub = contaminated if di == "ALL" else contaminated[contaminated.distance == di]
            red = advantage(sub[sub.contamination_type == "redundant"], o)
            if not len(red):
                continue
            red_by_ref = red.groupby("ref_index")["adv"].mean()
            for arm_name, arm_types in arms:
                nr = advantage(sub[sub.contamination_type.isin(arm_types)], o)
                if not len(nr):
                    continue
                nr_by_ref = nr.groupby("ref_index")["adv"].mean()
                common = sorted(set(red_by_ref.index) & set(nr_by_ref.index))
                if len(common) < 5:
                    continue
                d = (nr_by_ref.loc[common] - red_by_ref.loc[common]).values
                try:
                    p = float(sps.wilcoxon(d, alternative="two-sided",
                                           zero_method="wilcox").pvalue)
                except ValueError:
                    p = 1.0
                bs = fw.Bootstrapper(n_rows=len(d), n_iter=N_BOOT,
                                     seed=BOOT_SEED + 71
                                     + fw.stable_hash(f"{o}{di}{arm_name}") % 9999)
                ci = bs.ci(lambda idx: fw.hodges_lehmann_paired(d[idx]))
                attrib.append({
                    "competitor": o, "distance": di, "arm": arm_name,
                    "n_reference_pairs": int(len(common)),
                    "mean_advantage_redundant": round(float(red.adv.mean()), 4),
                    "mean_advantage_arm": round(float(nr.adv.mean()), 4),
                    "hl_diff_arm_minus_redundant": round(ci["estimate"], 4),
                    "diff_ci_lo": round(ci["ci_lo"], 4), "diff_ci_hi": round(ci["ci_hi"], 4),
                    "rank_biserial": round(fw.rank_biserial_paired(d), 4),
                    "p_wilcoxon_paired_by_reference": p})
    attribution = pd.DataFrame(attrib)
    if len(attribution):
        attribution["q_bh"] = fw.bh_correct(
            attribution.p_wilcoxon_paired_by_reference.values)
        attribution["verdict"] = np.where(
            attribution.q_bh >= 0.05, "uniform (no type specificity)",
            np.where(attribution.hl_diff_arm_minus_redundant > 0,
                     "MAGICC advantage LARGER on this non-redundant arm",
                     "MAGICC advantage LARGER on redundant"))
    attribution.to_csv(RESULTS_DIR / "set_F_attribution.tsv", sep="\t", index=False)
    if len(attribution):
        print(attribution[attribution.distance == "ALL"].to_string(index=False))

    # ---------------------------------------------------------- figures
    print("\nfigures ...")
    heatmap_figure(cells, tools, FIG_DIR / "fig_setF_signed_contamination_error_heatmap",
                   "cont_bias",
                   "Signed contamination error (predicted - true) by contamination type "
                   "and taxonomic distance",
                   "signed error (percentage points)")
    heatmap_figure(cells, tools, FIG_DIR / "fig_setF_contamination_mae_heatmap",
                   "cont_mae",
                   "Contamination MAE by contamination type and taxonomic distance",
                   "MAE (percentage points)")
    heatmap_figure(cells, tools, FIG_DIR / "fig_setF_signed_completeness_error_heatmap",
                   "comp_bias",
                   "Signed completeness error (predicted - true) by contamination type "
                   "and taxonomic distance",
                   "signed error (percentage points)")
    bias_vs_distance_figure(cells, tools, FIG_DIR / "fig_setF_bias_vs_distance")

    n_ref = contaminated.ref_index.nunique()
    caption = (
        "**Set F — contamination type x taxonomic distance.** Denominator of every cell: "
        f"n = {len(contaminated)//max(1,len(tools))} contaminated samples per tool "
        f"({n_ref} reference genomes x 6 taxonomic distances x 3 contamination types; "
        f"{len(contaminated[contaminated.tool==tools[0]])//18} samples per cell). "
        "Colour is the mean signed error (predicted - true) in percentage points; "
        "negative (blue) = under-estimate. Confidence intervals are cluster bootstrap "
        f"({N_BOOT} resamples) over reference genomes. " + DENOM_NOTE)
    (FIG_DIR / "captions.md").write_text(caption + "\n")

    # ------------------------------------------------------------ summary
    def pooled(tool, level):
        s = margins[(margins.tool == tool) & (margins.level == level)]
        return s.iloc[0].to_dict() if len(s) else {}

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "set": args.set, "model": ONNX_MODEL,
        "n_samples": int(len(long) / max(1, len(tools))),
        "n_contaminated": int(len(contaminated) / max(1, len(tools))),
        "n_references": int(n_ref),
        "tools": tools,
        "bootstrap": {"n_iter": N_BOOT, "seed": BOOT_SEED,
                      "unit": "reference genome (ref_index)"},
        "r2_convention": "coefficient of determination (1 - SS_res/SS_tot)",
        "denominators": DENOM_NOTE,
        "provenance": (prov or {}).get("DISJOINTNESS_VERDICT", "not run"),
        "by_type_by_tool": {
            t: {ty: {k: (margins[(margins.tool == t) & (margins.margin == "type")
                                 & (margins.level == ty)][k].iloc[0]
                         if len(margins[(margins.tool == t) & (margins.margin == "type")
                                        & (margins.level == ty)]) else None)
                     for k in ("n", "cont_mae", "cont_bias", "cont_bias_ci_lo",
                               "cont_bias_ci_hi", "cont_r2", "detection_slope",
                               "detection_slope_ci_lo", "detection_slope_ci_hi")}
                for ty in TYPE_ORDER}
            for t in tools},
        "pooled_by_tool": {t: {k: pooled(t, "ALL_contaminated").get(k)
                               for k in ("n", "cont_mae", "cont_mae_ci_lo", "cont_mae_ci_hi",
                                         "cont_bias", "cont_bias_ci_lo", "cont_bias_ci_hi",
                                         "cont_r2", "detection_slope",
                                         "detection_slope_ci_lo", "detection_slope_ci_hi",
                                         "comp_mae", "comp_bias", "comp_r2")}
                          for t in tools},
        "uncontaminated_control_by_tool": {
            t: {k: pooled(t, "UNCONTAMINATED_CONTROL").get(k)
                for k in ("n", "cont_mae", "cont_bias", "comp_mae", "comp_bias")}
            for t in tools},
        "outputs": sorted(str(p.relative_to(PROJECT_DIR))
                          for p in RESULTS_DIR.rglob("*") if p.is_file()),
    }
    if len(dist_char):
        summary["distance_axis_pct_donor_genes_orthologous"] = {
            d: round(float(dist_char[dist_char.distance == d]
                           .pct_donor_genes_orthologous.median()), 2)
            for d in DISTANCE_ORDER if (dist_char.distance == d).any()}
    if dup is not None:
        ok = dup[dup.n_scg_families_duplicated.notna()]
        summary["marker_duplication_validation"] = {
            "n_scanned": int(len(ok)),
            "mean_duplicated_scg_families_by_type": {
                k: round(float(v), 3) for k, v in
                ok.groupby("contamination_type")["n_scg_families_duplicated"]
                .mean().items()},
            "note": ("Independent Prodigal + hmmsearch --cut_tc scan of the finished "
                     "Set F assemblies against this project's own single-copy core-gene "
                     "profiles (85_bcg.hmm / uacg.hmm). It knows nothing about how each "
                     "sample was built, so it is a genuine test of the operational "
                     "definitions: REDUNDANT must inflate marker duplication, REPLACED "
                     "and SINGLE must not.")}
    (RESULTS_DIR / "set_F_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {RESULTS_DIR}/set_F_summary.json")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
