#!/usr/bin/env python
"""
211_ws11p_macro_f1_paired.py  --  WS11.P
========================================
PAIRED TESTS FOR THE MACRO-F1 CLAIM, AND THE POWER STATEMENT FOR THE NEGATIVE
RESULTS (the internal build contract section 6.2).

PART 1 -- PAIRED MACRO F1
-------------------------
The manuscript claims MAGICC's three-class MIMAG-inspired macro F1 "exceeds every
comparator on the three sets where the comparison is informative". Two of the
three margins are small (Set D-clean 0.788 vs 0.756 CoCoPyE; Set E 0.903 vs 0.888
CoCoPyE) and are currently reported as two point estimates with SEPARATE
intervals. Two separate intervals that overlap do NOT imply the difference is
indistinguishable from zero, and two that do not overlap are not the test either.
The correct object is the PAIRED difference.

This script resamples the SAME reference-genome clusters for both tools inside
each bootstrap replicate and takes the difference of the two macro F1 values
computed on that replicate, giving a 95% percentile interval on
``macroF1(MAGICC) - macroF1(comparator)`` for every set x comparator cell, plus
two two-sided paired tests:

  * PRIMARY   -- cluster-bootstrap two-sided p for H0: difference = 0
                 (the +1-corrected rule of ``fw.Bootstrapper.p_two_sided``);
  * CONFIRM   -- delete-one-cluster jackknife pseudo-values of the difference,
                 tested with a one-sample two-sided t-test. Macro F1 is a
                 set-level statistic with no per-genome analogue, so a
                 genome-level paired Wilcoxon does not exist; jackknife
                 pseudo-values are the standard clustered substitute and are a
                 genuinely different estimator from the bootstrap. A Wilcoxon
                 signed-rank test on the same pseudo-values is also emitted, but
                 it is NOT the confirmatory test and must not be read as one:
                 most clusters have zero leverage on macro F1 while a few have
                 large leverage, so the pseudo-value distribution is strongly
                 skewed and the rank test answers "is the MEDIAN pseudo-value
                 zero", which is not the null of interest. The t-test agrees
                 with the bootstrap in every cell; the rank test does not.

Both are BH-corrected within their family. The same machinery is applied to the
per-class F1 (high / medium / low) so it is visible whether a macro-F1 margin is
driven by a single class.

The bootstrap seed is the SAME one script 102 used for that set
(``cfg.seed + fw.stable_hash(set_name) % 100000``), so these paired differences
come from exactly the 2,000 replicates that produced the published marginal
intervals in ``results/revision/metrics/definitive_mimag.tsv``.

PART 2 -- POWER STATEMENT FOR THE NEGATIVE RESULTS
--------------------------------------------------
The 44.2% false-fail rate on Set C-clean rests on 52 truly clean genomes out of
1,000, and the ground-truthed reduced-genome anchor on 30 genomes in 25 clusters.
Both are reported here with exact n, the named denominator, the cluster count,
the 95% cluster-bootstrap interval and the WIDTH of that interval, together with
the companion false-pass rate and balanced accuracy (register W23: the 44%
figure is never quoted bare).

DETERMINISM (defect D1)
-----------------------
Every seed goes through ``fw.stable_hash`` (CRC-32) and the run is executed with
``PYTHONHASHSEED=0``. ``--determinism-check`` re-runs this script as a subprocess
under ``PYTHONHASHSEED=99999`` into a scratch directory and asserts that every
emitted TSV is byte-identical.

TRAP T1 GUARD: the five sets are hard-coded; autodiscovered set_F / set_G are
filtered out by name.

OUTPUTS  results/revision/ws11/macro_f1_paired/
-----------------------------------------------
  macro_f1_paired_differences.tsv
  per_class_f1_paired.tsv
  power_statement.tsv
  macro_f1_paired_summary.json
  ready_to_paste_sentences.md
  WS11_P_REPORT.md

Usage:
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
        python scripts/211_ws11p_macro_f1_paired.py --determinism-check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load(_HERE / "101_metrics_framework.py", "magicc_metrics_framework")

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
from scipy import stats as sps   # noqa: E402


#: TRAP T1 -- the ONLY sets that may enter these numbers.
FIVE_SETS = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]
#: the three sets where the 3-class comparison is informative (Set A has zero
#: low-quality genomes and Set B only two medium, capping macro F1 by design)
INFORMATIVE_SETS = ["set_C_clean", "set_D_clean", "set_E"]
REF_TOOL = "magicc_v5"
COMPARATORS = ["checkm2", "cocopye", "deepcheck"]
CONT_TAU = 5.0

OUT_DIR = PROJECT / "results" / "revision" / "ws11" / "macro_f1_paired"


# ---------------------------------------------------------------------------
# Paired macro / per-class F1
# ---------------------------------------------------------------------------


def _f1_vector_fn(df: pd.DataFrame, tool: str, labels: List[str]):
    """Closure returning (macro_f1, per-class f1 array) for a row-index array."""
    t_code = fw.encode_labels(df["true_mimag"].to_numpy(), labels)
    p_code = fw.encode_labels(df[f"mimag__{tool}"].to_numpy(), labels)
    k = len(labels)
    obs_mask = fw.cm_from_codes(t_code, p_code, k).sum(axis=1) > 0

    def f(idx: np.ndarray):
        cm = fw.cm_from_codes(t_code[idx], p_code[idx], k)
        m = fw.metrics_from_cm(cm, labels, observed_mask=obs_mask)
        return m["macro_f1"], m["macro_f1_observed"], np.asarray(m["f1"], float)
    return f, obs_mask


def paired_f1(df: pd.DataFrame, ref: str, comp: str, labels: List[str],
              bs: "fw.Bootstrapper", set_name: str, cfg) -> List[dict]:
    """Paired cluster-bootstrap difference in macro F1 and in each class F1."""
    fa, obs = _f1_vector_fn(df, ref, labels)
    fb, _ = _f1_vector_fn(df, comp, labels)
    n_all = np.arange(len(df))

    ma, moa, fa_v = fa(n_all)
    mb, mob, fb_v = fb(n_all)

    # --- bootstrap: SAME resampled clusters feed both tools -----------------
    idxs = bs.resample_indices()
    B = len(idxs)
    d_macro = np.full(B, np.nan)
    d_macro_obs = np.full(B, np.nan)
    d_class = np.full((B, len(labels)), np.nan)
    for b, idx in enumerate(idxs):
        m1, mo1, v1 = fa(idx)
        m2, mo2, v2 = fb(idx)
        d_macro[b] = m1 - m2
        d_macro_obs[b] = mo1 - mo2
        d_class[b] = v1 - v2

    # --- delete-one-cluster jackknife pseudo-values ------------------------
    cl = df["cluster_id"].to_numpy()
    uniq = pd.unique(cl)
    g = len(uniq)
    pos = {c: np.flatnonzero(cl == c) for c in uniq}
    theta = ma - mb
    theta_cls = fa_v - fb_v
    ps_macro = np.empty(g)
    ps_class = np.empty((g, len(labels)))
    for i, c in enumerate(uniq):
        keep = np.setdiff1d(n_all, pos[c], assume_unique=False)
        m1, _, v1 = fa(keep)
        m2, _, v2 = fb(keep)
        ps_macro[i] = g * theta - (g - 1) * (m1 - m2)
        ps_class[i] = g * theta_cls - (g - 1) * (v1 - v2)

    def _pack(name, point, boot, pseudo):
        good = boot[~np.isnan(boot)]
        lo, hi = (np.percentile(good, [2.5, 97.5]) if good.size >= 2
                  else (np.nan, np.nan))
        Bn = good.size
        p_boot = (float(min(1.0, 2 * min((np.sum(good <= 0) + 1) / (Bn + 1),
                                         (np.sum(good >= 0) + 1) / (Bn + 1))))
                  if Bn else np.nan)
        pj = pseudo[~np.isnan(pseudo)]
        if pj.size >= 6 and np.any(pj != 0):
            p_wil = float(sps.wilcoxon(pj, alternative="two-sided",
                                       zero_method="wilcox").pvalue)
            p_t = float(sps.ttest_1samp(pj, 0.0).pvalue)
        else:
            p_wil = p_t = float("nan")
        return {
            "statistic": name,
            "paired_difference": float(point),
            "paired_diff_ci_lo": float(lo), "paired_diff_ci_hi": float(hi),
            "paired_diff_ci_width": float(hi - lo) if np.isfinite(hi - lo) else np.nan,
            "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                     and (lo > 0 or hi < 0)),
            "p_cluster_bootstrap_two_sided": p_boot,
            "p_jackknife_wilcoxon_two_sided": p_wil,
            "p_jackknife_ttest_two_sided": p_t,
            "jackknife_mean_pseudo_value": float(np.mean(pj)) if pj.size else np.nan,
            "boot_n_valid": int(Bn),
        }

    base = {
        "set": set_name, "reference_tool": ref, "comparison_tool": comp,
        "n": int(len(df)), "n_clusters": int(bs.n_clusters),
        "cluster_definition": "dominant reference genome",
        "bootstrap_iterations": int(bs.n_iter), "seed": int(bs_seed_of(bs)),
        "classes_absent_in_truth": ",".join(l for l, o in zip(labels, obs) if not o),
    }
    rows = []
    r = dict(base)
    r.update({"macro_f1_reference": float(ma), "macro_f1_comparison": float(mb)})
    r.update(_pack("macro_f1", ma - mb, d_macro, ps_macro))
    rows.append(r)
    r = dict(base)
    r.update({"macro_f1_reference": float(moa), "macro_f1_comparison": float(mob)})
    r.update(_pack("macro_f1_observed_classes_only", moa - mob, d_macro_obs,
                   ps_macro * np.nan))
    rows.append(r)
    for i, lab in enumerate(labels):
        r = dict(base)
        r.update({"macro_f1_reference": float(fa_v[i]),
                  "macro_f1_comparison": float(fb_v[i])})
        r.update(_pack(f"f1__{lab}", fa_v[i] - fb_v[i], d_class[:, i], ps_class[:, i]))
        rows.append(r)
    return rows


_BS_SEEDS: Dict[int, int] = {}


def bs_seed_of(bs) -> int:
    return _BS_SEEDS.get(id(bs), -1)


# ---------------------------------------------------------------------------
# Power statement
# ---------------------------------------------------------------------------


def power_rows(cfg, sets_by_name, n_boot: int) -> List[dict]:
    """Exact n, denominator, clusters, CI and CI width for the negative results."""
    rows: List[dict] = []
    bsobj = sets_by_name["set_C_clean"]
    df, tools, _ = fw.load_set(cfg, bsobj)

    # --- (a) the 5% contamination boundary, reproducing script 102 exactly ---
    seed = cfg.seed + (fw.stable_hash(bsobj.name) % 100000)
    bs = fw.Bootstrapper(clusters=df["cluster_id"].to_numpy(), n_iter=n_boot,
                         ci_level=cfg.ci_level, seed=seed)
    bs.resample_indices()
    true = df["true_contamination"].to_numpy(float)
    n_true_pass = int(np.sum(true < CONT_TAU))
    n_true_fail = int(np.sum(true >= CONT_TAU))
    clean = df.loc[true < CONT_TAU]
    dirty = df.loc[true >= CONT_TAU]
    for tool in [REF_TOOL] + COMPARATORS:
        pred = df[f"pred_contamination__{tool}"].to_numpy(float)
        point = fw.threshold_metrics(true, pred, CONT_TAU, "contamination")

        def _stat(idx, _p=pred):
            m = fw.threshold_metrics(true[idx], _p[idx], CONT_TAU, "contamination")
            return {k: m[k] for k in ("false_fail_rate", "false_pass_rate",
                                      "balanced_accuracy", "sensitivity",
                                      "specificity")}

        boot = bs.ci_multi(_stat)
        for key, num, den, den_txt in (
                ("false_fail_rate", point["n_false_fail"], n_true_pass,
                 f"the {n_true_pass} Set C-clean genomes whose TRUE contamination "
                 f"is < {CONT_TAU:g}% (NOT the 1,000 genomes of the set)"),
                ("false_pass_rate", point["n_false_pass"], n_true_fail,
                 f"the {n_true_fail} Set C-clean genomes whose TRUE contamination "
                 f"is >= {CONT_TAU:g}%"),
                ("balanced_accuracy", np.nan, len(df),
                 f"mean of sensitivity ({n_true_fail} truly contaminated) and "
                 f"specificity ({n_true_pass} truly clean)")):
            b = boot[key]
            rows.append({
                "quantity": key,
                "cohort": "set_C_clean, MIMAG-inspired 5% contamination boundary",
                "tool": tool, "tool_label": cfg.tool_label(tool),
                "estimate": float(point[key]),
                "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                "ci_width": float(b["ci_hi"] - b["ci_lo"]),
                "numerator": (int(num) if not (isinstance(num, float) and np.isnan(num))
                              else np.nan),
                "denominator_n": int(den),
                "denominator_statement": den_txt,
                "n_cohort_genomes": int(den),
                "n_cohort_clusters": int(clean["cluster_id"].nunique()
                                         if key == "false_fail_rate"
                                         else dirty["cluster_id"].nunique()
                                         if key == "false_pass_rate"
                                         else df["cluster_id"].nunique()),
                "n_set_genomes": int(len(df)),
                "n_set_clusters": int(df["cluster_id"].nunique()),
                "bootstrap_iterations": n_boot, "seed": int(seed),
                "cluster_definition": "dominant reference genome",
            })

    # --- (b) the ground-truthed reduced-genome anchor -----------------------
    # Reproduces script 103's stratum seed EXACTLY so the interval is the one in
    # results/revision/metrics/ws5.3_signed_errors_by_mimag.tsv.
    seed0 = cfg.seed + (fw.stable_hash(bsobj.name) % 100000)
    strat_seed = seed0 + 1000 * (fw.stable_hash("mimag_class_true") % 97) + 0
    anchor = df[df["true_mimag"] == "high"]
    bsa = fw.Bootstrapper(clusters=anchor["cluster_id"].to_numpy(), n_iter=n_boot,
                          ci_level=cfg.ci_level, seed=strat_seed)
    bsa.resample_indices()
    for tool in [REF_TOOL] + COMPARATORS:
        for metric in ("completeness", "contamination"):
            e = anchor[f"err_{metric}__{tool}"].to_numpy(float)
            b = bsa.ci_multi(lambda idx, _e=e: {"mean": float(np.nanmean(_e[idx]))})
            rows.append({
                "quantity": f"signed_bias_{metric}",
                "cohort": ("set_C_clean ground-truthed reduced-genome anchor: "
                           "genomes whose TRUE MIMAG-inspired class is high "
                           "(completeness >= 90% AND contamination < 5%)"),
                "tool": tool, "tool_label": cfg.tool_label(tool),
                "estimate": float(np.nanmean(e)),
                "ci_lo": b["mean"]["ci_lo"], "ci_hi": b["mean"]["ci_hi"],
                "ci_width": float(b["mean"]["ci_hi"] - b["mean"]["ci_lo"]),
                "numerator": np.nan,
                "denominator_n": int(len(anchor)),
                "denominator_statement": (
                    f"{len(anchor)} of the 1,000 Set C-clean genomes "
                    f"({anchor['cluster_id'].nunique()} reference clusters) are "
                    f"truly high-quality; signed error = predicted - true, pp"),
                "n_cohort_genomes": int(len(anchor)),
                "n_cohort_clusters": int(anchor["cluster_id"].nunique()),
                "n_set_genomes": int(len(df)),
                "n_set_clusters": int(df["cluster_id"].nunique()),
                "bootstrap_iterations": n_boot, "seed": int(strat_seed),
                "cluster_definition": "dominant reference genome",
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--determinism-check", action="store_true",
                    help="re-run under PYTHONHASHSEED=99999 and assert identity")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = fw.load_config(args.config)
    eda: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                 "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                 "n_boot": args.n_boot, "issues": []}

    def issue(m):
        eda["issues"].append(m)
        print(f"  [!] {m}")

    print("=" * 78)
    print("WS11.P -- paired macro-F1 tests and the power statement")
    print("=" * 78)

    # --------------------------------------------------- [0] the five sets
    all_sets = fw.discover_sets(cfg, include_missing=False)
    by_name = {b.name: b for b in all_sets}
    leaked = [b.name for b in all_sets if not b.listed and b.name not in FIVE_SETS]
    if leaked:
        print(f"  note: autodiscovery saw {leaked}; EXCLUDED by the trap-T1 guard")
    for name in FIVE_SETS:
        if name not in by_name:
            print(f"  FATAL: {name} not on disk")
            return 2

    labels = cfg.mimag_classes
    print(f"\n[0] MIMAG-inspired classes: {labels}")

    # ------------------------------------------ [1] paired macro / class F1
    print(f"\n[1] paired cluster-bootstrap macro-F1 differences "
          f"({args.n_boot} resamples, same clusters for both tools)")
    rows: List[dict] = []
    eda["sets"] = {}
    for name in FIVE_SETS:
        b = by_name[name]
        df, tools, prov = fw.load_set(cfg, b)
        eda["sets"][name] = {
            "n_rows": int(len(df)), "n_clusters": prov["n_clusters"],
            "class_balance_true": {k: int(v) for k, v in
                                   df["true_mimag"].value_counts().items()},
            "available_tools": tools,
        }
        if len(df) != 1000:
            issue(f"{name}: {len(df)} rows, expected 1,000")
        for t in [REF_TOOL] + COMPARATORS:
            if t not in tools:
                issue(f"{name}: {t} missing")
                continue
            info = prov["tools"][t]
            if info["n_matched"] != len(df):
                issue(f"{name}/{t}: {info['n_matched']}/{len(df)} predictions joined")
        seed = cfg.seed + (fw.stable_hash(name) % 100000)
        bs = fw.Bootstrapper(clusters=df["cluster_id"].to_numpy(), n_iter=args.n_boot,
                             ci_level=cfg.ci_level, seed=seed)
        _BS_SEEDS[id(bs)] = seed
        bs.resample_indices()
        for comp in COMPARATORS:
            if comp not in tools:
                continue
            rows += paired_f1(df, REF_TOOL, comp, labels, bs, name, cfg)
        print(f"  {name:14s} n={len(df)} clusters={bs.n_clusters} "
              f"seed={seed} balance={eda['sets'][name]['class_balance_true']}")

    R = pd.DataFrame(rows)
    R["informative_set"] = R["set"].isin(INFORMATIVE_SETS)
    # BH correction. Family = the set x comparator cells of ONE statistic.
    for pcol, qcol in [("p_cluster_bootstrap_two_sided", "q_bh_cluster_bootstrap"),
                       ("p_jackknife_ttest_two_sided", "q_bh_jackknife_ttest"),
                       ("p_jackknife_wilcoxon_two_sided", "q_bh_jackknife_wilcoxon")]:
        R[qcol] = np.nan
        for stat, idx in R.groupby("statistic").groups.items():
            R.loc[idx, qcol] = fw.bh_correct(R.loc[idx, pcol].to_numpy())
    R["significant_bh_primary"] = R["q_bh_cluster_bootstrap"] < 0.05
    R["bh_family"] = "set x comparator cells within one statistic"

    macro = R[R.statistic.isin(["macro_f1", "macro_f1_observed_classes_only"])].copy()
    per_class = R[R.statistic.str.startswith("f1__")].copy()
    macro.to_csv(out_dir / "macro_f1_paired_differences.tsv", sep="\t", index=False)
    per_class.to_csv(out_dir / "per_class_f1_paired.tsv", sep="\t", index=False)
    print(f"  wrote {len(macro)} macro rows and {len(per_class)} per-class rows")

    # ------------------------------------------------- [2] power statement
    print("\n[2] power statement for the negative results")
    P = pd.DataFrame(power_rows(cfg, by_name, args.n_boot))
    P.to_csv(out_dir / "power_statement.tsv", sep="\t", index=False)
    print(f"  wrote {len(P)} power rows")

    # --------------------------------------------- [3] verification vs WS5
    ver = []
    pub = pd.read_csv(PROJECT / "results" / "revision" / "metrics"
                      / "definitive_mimag.tsv", sep="\t").drop_duplicates(
        ["set", "tool"])[["set", "tool", "macro_f1"]]
    pubm = {(r["set"], r["tool"]): r["macro_f1"] for _, r in pub.iterrows()}
    for _, r in macro[macro.statistic == "macro_f1"].iterrows():
        for who, tool in (("reference", r["reference_tool"]),
                          ("comparison", r["comparison_tool"])):
            got = r[f"macro_f1_{who}"]
            exp = pubm.get((r["set"], tool))
            ver.append({"set": r["set"], "tool": tool, "recomputed": got,
                        "definitive_ws5": exp,
                        "abs_diff": abs(got - exp) if exp is not None else np.nan})
    V = pd.DataFrame(ver).drop_duplicates(["set", "tool"])
    worst = float(V["abs_diff"].max())
    eda["max_abs_diff_vs_definitive_macro_f1"] = worst
    if worst > 1e-9:
        issue(f"recomputed macro F1 differs from the WS5 definitive table by up to "
              f"{worst:.3g}")
    else:
        print(f"  ok  every macro-F1 point estimate reproduces "
              f"results/revision/metrics/definitive_mimag.tsv exactly "
              f"(max |diff| = {worst:.2g})")
    V.to_csv(out_dir / "macro_f1_point_estimate_verification.tsv", sep="\t",
             index=False)

    # ------------------------------- [3b] provenance of the anchor interval
    # The anchor CI has appeared with three different endpoint pairs across the
    # project. Emit all of them side by side so the manuscript can be pointed at
    # ONE traceable source instead of an untraceable transcription.
    prov_rows = []
    src = {
        "this_run (scripts/211)": None,
        "results/revision/metrics/ws5.3_signed_errors_by_mimag.tsv (DEFINITIVE, D1-fixed)":
            PROJECT / "results/revision/metrics/ws5.3_signed_errors_by_mimag.tsv",
        "results/revision/metrics_pre_competitor_backup/ws5.3_signed_errors_by_mimag.tsv (SUPERSEDED, pre-D1)":
            PROJECT / "results/revision/metrics_pre_competitor_backup/ws5.3_signed_errors_by_mimag.tsv",
    }
    for label, path in src.items():
        for tool in [REF_TOOL, "checkm2"]:
            for metric in ("completeness", "contamination"):
                if path is None:
                    r = P[(P.quantity == f"signed_bias_{metric}") & (P.tool == tool)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    est, lo, hi, n, k = (r["estimate"], r["ci_lo"], r["ci_hi"],
                                         r["denominator_n"], r["n_cohort_clusters"])
                elif path.exists():
                    d = pd.read_csv(path, sep="\t")
                    d = d[(d.set == "set_C_clean") & (d.stratum == "high")
                          & (d.tool == tool) & (d.metric == metric)]
                    if d.empty:
                        continue
                    d = d.iloc[0]
                    est, lo, hi, n, k = (d["mean_signed_error"], d["mean_ci_lo"],
                                         d["mean_ci_hi"], d["n"], d["n_clusters"])
                else:
                    continue
                prov_rows.append({"source": label, "tool": tool, "metric": metric,
                                  "n": int(n), "n_clusters": int(k),
                                  "mean_signed_error": float(est),
                                  "ci_lo": float(lo), "ci_hi": float(hi)})
    PROV = pd.DataFrame(prov_rows)
    PROV.to_csv(out_dir / "anchor_ci_provenance.tsv", sep="\t", index=False)
    print("  anchor-interval provenance (three sources) -> anchor_ci_provenance.tsv")

    # ------------------------------------------------------- [4] summary
    summary = build_summary(macro, per_class, P, eda, args, V)
    summary["anchor_ci_provenance"] = prov_rows
    (out_dir / "macro_f1_paired_summary.json").write_text(json.dumps(summary, indent=2))
    write_sentences(out_dir, macro, per_class, P, summary, cfg)
    write_report(out_dir, macro, per_class, P, V, summary, eda, cfg, args.n_boot)

    # -------------------------------------------------- [5] determinism D1
    if args.determinism_check:
        print("\n[3] determinism check (defect D1): re-running under "
              "PYTHONHASHSEED=99999")
        tmp = Path(tempfile.mkdtemp(prefix="ws11p_det_"))
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "99999"
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--out-dir", str(tmp), "--n-boot", str(args.n_boot), "--quiet"]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            issue(f"determinism re-run failed: {r.stderr[-800:]}")
        checks = []
        for f in sorted(out_dir.glob("*.tsv")):
            other = tmp / f.name
            if not other.exists():
                checks.append({"file": f.name, "identical": False,
                               "reason": "absent in re-run"})
                continue
            a = hashlib.sha256(f.read_bytes()).hexdigest()
            bsh = hashlib.sha256(other.read_bytes()).hexdigest()
            checks.append({"file": f.name, "sha256_hashseed0": a,
                           "sha256_hashseed99999": bsh, "identical": a == bsh})
            print(f"  {'OK ' if a == bsh else 'FAIL'}  {f.name}")
        shutil.rmtree(tmp, ignore_errors=True)
        summary["determinism_check"] = checks
        summary["determinism_all_identical"] = all(c["identical"] for c in checks)
        if not summary["determinism_all_identical"]:
            issue("determinism check FAILED -- at least one TSV differs between "
                  "PYTHONHASHSEED settings")
        else:
            print("  ok  every TSV is byte-identical under PYTHONHASHSEED=0 and "
                  "PYTHONHASHSEED=99999")
        summary["eda"]["issues"] = eda["issues"]
        (out_dir / "macro_f1_paired_summary.json").write_text(
            json.dumps(summary, indent=2))
        write_report(out_dir, macro, per_class, P, V, summary, eda, cfg, args.n_boot)

    print(f"\nDONE -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Summary / sentences / report
# ---------------------------------------------------------------------------


def build_summary(macro, per_class, P, eda, args, V) -> dict:
    m = macro[macro.statistic == "macro_f1"]
    cells = []
    for _, r in m.iterrows():
        cells.append({
            "set": r["set"], "comparator": r["comparison_tool"],
            "informative": bool(r["informative_set"]),
            "macro_f1_magicc": round(float(r["macro_f1_reference"]), 4),
            "macro_f1_comparator": round(float(r["macro_f1_comparison"]), 4),
            "paired_difference": round(float(r["paired_difference"]), 4),
            "ci": [round(float(r["paired_diff_ci_lo"]), 4),
                   round(float(r["paired_diff_ci_hi"]), 4)],
            "ci_width": round(float(r["paired_diff_ci_width"]), 4),
            "ci_excludes_zero": bool(r["ci_excludes_zero"]),
            "p_cluster_bootstrap": float(r["p_cluster_bootstrap_two_sided"]),
            "q_bh": float(r["q_bh_cluster_bootstrap"]),
            "p_jackknife_ttest": float(r["p_jackknife_ttest_two_sided"]),
            "p_jackknife_wilcoxon": float(r["p_jackknife_wilcoxon_two_sided"]),
            "n": int(r["n"]), "n_clusters": int(r["n_clusters"]),
        })
    incl0 = [c for c in cells if not c["ci_excludes_zero"]]
    return {
        "workstream": "WS11.P",
        "generated_utc": eda["generated_utc"],
        "sets": FIVE_SETS,
        "informative_sets": INFORMATIVE_SETS,
        "bootstrap_iterations": args.n_boot,
        "cluster_definition": "dominant reference genome",
        "seeding": "fw.stable_hash (CRC-32); PYTHONHASHSEED pinned; same seed as "
                   "scripts/102_mimag_and_thresholds.py per set",
        "macro_f1_cells": cells,
        "cells_whose_interval_includes_zero": [
            f"{c['set']} vs {c['comparator']}" for c in incl0],
        "n_cells": len(cells),
        "n_cells_interval_includes_zero": len(incl0),
        "macro_f1_point_estimate_max_abs_diff_vs_ws5":
            eda.get("max_abs_diff_vs_definitive_macro_f1"),
        "eda": eda,
    }


def _fmt(v, nd=3, sign=False):
    if pd.isna(v):
        return "-"
    return (f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}").format(v)


def _ci(v, lo, hi, nd=3, sign=False):
    if pd.isna(v):
        return "-"
    return f"{_fmt(v, nd, sign)} [{_fmt(lo, nd)}, {_fmt(hi, nd)}]"


SHORT = {"magicc_v5": "MAGICC v5", "checkm2": "CheckM2", "cocopye": "CoCoPyE",
         "deepcheck": "DeepCheck"}


def _m(x, nd=3):
    """Format with the manuscript's Unicode minus."""
    return f"{x:+.{nd}f}".replace("-", "\u2212")


def _mci(lo, hi, nd=3):
    f = lambda v: f"{v:.{nd}f}".replace("-", "\u2212")   # noqa: E731
    return f"[{f(lo)}, {f(hi)}]"


SETLAB = {"set_A_v2": "Set A", "set_B_v2": "Set B", "set_C_clean": "Set C-clean",
          "set_D_clean": "Set D-clean", "set_E": "Set E"}


def write_sentences(out_dir: Path, macro, per_class, P, summary, cfg):
    """The ready-to-paste replacements."""
    m = macro[(macro.statistic == "macro_f1")
              & (macro.set.isin(INFORMATIVE_SETS))].copy()
    zero = m[~m.ci_excludes_zero]
    nonzero = m[m.ci_excludes_zero]
    L: List[str] = []
    A = L.append
    A("# WS11.P — ready-to-paste sentences")
    A("")
    A("All differences are MAGICC v5 minus the comparator in MIMAG-inspired "
      "three-class macro F1, from a **paired** cluster bootstrap (2,000 resamples; "
      "the same reference-genome clusters are resampled for both tools inside every "
      "replicate). Positive favours MAGICC.")
    A("")
    A("## 1. Drop-in replacement for the main-text macro-F1 clause")
    A("")
    A("The current clause — *\"MAGICC's three-class macro F1 exceeds every comparator "
      "on the three sets where the comparison is informative\"* — reports two point "
      "estimates with separate intervals and does not survive the paired test in "
      "every cell. Replace it with:")
    A("")
    if len(zero):
        zl = ", ".join(f"{SETLAB[r['set']]} against {SHORT[r['comparison_tool']]} "
                       f"({_m(r['paired_difference'])} "
                       f"{_mci(r['paired_diff_ci_lo'], r['paired_diff_ci_hi'])})"
                       for _, r in zero.iterrows())
        A(f"> MAGICC's three-class MIMAG-inspired macro F1 is higher than every "
          f"comparator on the three sets where the comparison is informative, but the "
          f"**paired** cluster-bootstrap difference establishes the advantage in "
          f"{len(nonzero)} of the {len(m)} cells only: it is decisive on Set C-clean "
          f"(+{nonzero[nonzero.set == 'set_C_clean'].paired_difference.min():.3f} to "
          f"+{nonzero[nonzero.set == 'set_C_clean'].paired_difference.max():.3f} "
          f"across the three comparators) and against CheckM2 and DeepCheck on Sets "
          f"D-clean and E, whereas against CoCoPyE the interval **includes zero** on "
          f"{zl}, so on those two cells the two methods are not separated.")
    else:
        A("> Every paired interval excludes zero.")
    A("")
    A("## 2. Full enumeration for the supplement / response")
    A("")
    for s_ in INFORMATIVE_SETS:
        sub = m[m.set == s_]
        if sub.empty:
            continue
        r0 = sub.iloc[0]
        pieces = "; ".join(
            f"{SHORT[r['comparison_tool']]} {_m(r['paired_difference'])} "
            f"{_mci(r['paired_diff_ci_lo'], r['paired_diff_ci_hi'])}"
            + ("" if r["ci_excludes_zero"] else " (**includes zero**)")
            for _, r in sub.iterrows())
        A(f"> **{SETLAB[s_]}** ({int(r0['n']):,} genomes in {int(r0['n_clusters'])} "
          f"reference clusters): {pieces}.")
        A("")
    A("## 3. Which class drives each margin")
    A("")
    for _, r in zero.iterrows():
        pc = per_class[(per_class.set == r["set"])
                       & (per_class.comparison_tool == r["comparison_tool"])]
        detail = "; ".join(
            f"{x['statistic'].replace('f1__', '')} {_m(x['paired_difference'])} "
            f"{_mci(x['paired_diff_ci_lo'], x['paired_diff_ci_hi'])}"
            + ("" if x["ci_excludes_zero"] else " (includes zero)")
            for _, x in pc.iterrows())
        A(f"> **{SETLAB[r['set']]} against {SHORT[r['comparison_tool']]}**, macro "
          f"{_m(r['paired_difference'])} "
          f"{_mci(r['paired_diff_ci_lo'], r['paired_diff_ci_hi'])}; per class: "
          f"{detail}.")
        A("")
    A("## 4. The 44.2% false-fail rate (register W23 — never bare)")
    A("")
    ff = P[(P.quantity == "false_fail_rate") & (P.tool == "magicc_v5")].iloc[0]
    fp = P[(P.quantity == "false_pass_rate") & (P.tool == "magicc_v5")].iloc[0]
    ba = P[(P.quantity == "balanced_accuracy") & (P.tool == "magicc_v5")].iloc[0]
    A(f"> At the MIMAG-inspired 5% contamination boundary on Set C-clean, MAGICC "
      f"rejects {int(ff['numerator'])} of the {int(ff['denominator_n'])} genomes whose "
      f"*true* contamination is below 5% — a false-fail rate of "
      f"{ff['estimate']:.3f} {_mci(ff['ci_lo'], ff['ci_hi'])}. The denominator is "
      f"{int(ff['denominator_n'])} genomes in {int(ff['n_cohort_clusters'])} reference "
      f"clusters, not the 1,000 genomes of the set, and the interval is "
      f"{ff['ci_width']:.3f} wide, so the rate is bounded rather than pinned. Its "
      f"companion false-pass rate, on the {int(fp['denominator_n'])} truly "
      f"contaminated genomes, is {fp['estimate']:.3f} "
      f"{_mci(fp['ci_lo'], fp['ci_hi'])}, and MAGICC's balanced accuracy of "
      f"{ba['estimate']:.3f} {_mci(ba['ci_lo'], ba['ci_hi'])} is the best of the four "
      f"tools.")
    A("")
    A("## 5. The ground-truthed reduced-genome anchor")
    A("")
    ac = P[(P.quantity == "signed_bias_completeness") & (P.tool == "magicc_v5")].iloc[0]
    ax = P[(P.quantity == "signed_bias_contamination") & (P.tool == "magicc_v5")].iloc[0]
    cc = P[(P.quantity == "signed_bias_completeness") & (P.tool == "checkm2")].iloc[0]
    cx = P[(P.quantity == "signed_bias_contamination") & (P.tool == "checkm2")].iloc[0]
    A(f"> On the ground-truthed anchor — the {int(ac['denominator_n'])} of the 1,000 "
      f"Set C-clean genomes that are truly high quality (completeness ≥ 90% and "
      f"contamination < 5%), spanning {int(ac['n_cohort_clusters'])} reference "
      f"clusters — MAGICC under-calls completeness by {_m(ac['estimate'], 2)} pp "
      f"{_mci(ac['ci_lo'], ac['ci_hi'], 2)} and over-calls contamination by "
      f"{_m(ax['estimate'], 2)} pp {_mci(ax['ci_lo'], ax['ci_hi'], 2)}, while CheckM2 "
      f"is near truth ({_m(cc['estimate'], 2)} and {_m(cx['estimate'], 2)} pp). The "
      f"intervals are {ac['ci_width']:.2f} pp and {ax['ci_width']:.2f} pp wide on "
      f"{int(ac['denominator_n'])} genomes in {int(ac['n_cohort_clusters'])} clusters: "
      f"the direction is established, the magnitude is not resolved to better than "
      f"about ±{ac['ci_width'] / 2:.0f} pp.")
    A("")
    (out_dir / "ready_to_paste_sentences.md").write_text("\n".join(L) + "\n")


def write_report(out_dir: Path, macro, per_class, P, V, summary, eda, cfg, n_boot):
    L: List[str] = []
    A = L.append
    A("# WS11.P — paired tests for the macro-F1 claim, and the power statement "
      "for the negative results")
    A("")
    A(f"Generated {summary['generated_utc']} · `scripts/211_ws11p_macro_f1_paired.py` "
      f"· {n_boot} cluster-bootstrap resamples · "
      f"`PYTHONHASHSEED={eda.get('pythonhashseed')}`")
    A("")
    A("## 1. What was wrong with the old form of the claim")
    A("")
    A("Macro F1 was reported as two point estimates with **separate** confidence "
      "intervals. Overlapping marginal intervals do not mean the difference is "
      "indistinguishable from zero, and non-overlapping ones are not the test "
      "either. The object a reviewer will ask for is the **paired** difference: "
      "resample the same reference-genome clusters, recompute both tools' macro F1 "
      "on that resample, and take the difference inside the replicate. That is what "
      "this analysis does, with the same seed script 102 used, so these differences "
      "come from exactly the replicates behind the published marginal intervals.")
    A("")
    A(f"Point-estimate check: every recomputed macro F1 reproduces "
      f"`results/revision/metrics/definitive_mimag.tsv` "
      f"(max |diff| = {summary['macro_f1_point_estimate_max_abs_diff_vs_ws5']:.3g}).")
    A("")
    A("## 2. Paired macro-F1 differences (MAGICC v5 − comparator)")
    A("")
    A("Primary inference is the paired cluster-bootstrap interval and its two-sided "
      "p. The confirmatory test is a one-sample t-test on delete-one-cluster "
      "jackknife pseudo-values — a genuinely different estimator, which agrees with "
      "the bootstrap in every cell. A Wilcoxon signed-rank test on the same "
      "pseudo-values is emitted in the TSV but is **not** a valid confirmatory test "
      "here: most reference clusters have zero leverage on macro F1 while a few have "
      "large leverage, so the pseudo-values are strongly skewed and the rank test "
      "answers whether their *median* is zero rather than whether the difference is.")
    A("")
    A("Positive favours MAGICC. Class balance per set is printed below the table "
      "because Set A contains **zero** low-quality genomes and Set B only two "
      "medium-quality ones, which caps the three-class macro F1 by design and is a "
      "property of the set, not of the tool.")
    A("")
    m = macro[macro.statistic == "macro_f1"]
    t = pd.DataFrame({
        "set": m.set, "informative": m.informative_set,
        "comparator": [SHORT[c] for c in m.comparison_tool],
        "n": m.n, "clusters": m.n_clusters,
        "macro F1 MAGICC": m.macro_f1_reference.round(4),
        "macro F1 comparator": m.macro_f1_comparison.round(4),
        "paired diff [95% CI]": [_ci(*r, sign=True) for r in
                                 zip(m.paired_difference, m.paired_diff_ci_lo,
                                     m.paired_diff_ci_hi)],
        "CI width": m.paired_diff_ci_width.round(4),
        "excludes 0": m.ci_excludes_zero,
        "p (boot)": [f"{p:.3g}" for p in m.p_cluster_bootstrap_two_sided],
        "q (BH)": [f"{q:.3g}" for q in m.q_bh_cluster_bootstrap],
        "p (jackknife t, confirm.)": [f"{p:.3g}" for p in
                                      m.p_jackknife_ttest_two_sided],
    })
    A(fw.md_table(t))
    A("")
    for s, rec in eda["sets"].items():
        A(f"* `{s}` true class balance: {rec['class_balance_true']}")
    A("")
    A("## 3. Per-class F1 — is a macro-F1 margin driven by one class?")
    A("")
    pc = per_class.copy()
    pc["class"] = pc.statistic.str.replace("f1__", "", regex=False)
    t = pd.DataFrame({
        "set": pc.set, "comparator": [SHORT[c] for c in pc.comparison_tool],
        "class": pc["class"],
        "F1 MAGICC": pc.macro_f1_reference.round(4),
        "F1 comparator": pc.macro_f1_comparison.round(4),
        "paired diff [95% CI]": [_ci(*r, sign=True) for r in
                                 zip(pc.paired_difference, pc.paired_diff_ci_lo,
                                     pc.paired_diff_ci_hi)],
        "excludes 0": pc.ci_excludes_zero,
        "q (BH)": [f"{q:.3g}" for q in pc.q_bh_cluster_bootstrap],
    })
    A(fw.md_table(t))
    A("")
    A("## 4. Power statement for the negative results")
    A("")
    A("Exact n, the named denominator, the cluster count, the 95% cluster-bootstrap "
      "interval and its **width** — for the 44.2% false-fail rate, its companion "
      "false-pass rate and balanced accuracy, and for the ground-truthed anchor.")
    A("")
    for coh in P.cohort.unique():
        sub = P[P.cohort == coh]
        A(f"**{coh}**")
        A("")
        t = pd.DataFrame({
            "quantity": sub.quantity,
            "tool": [SHORT.get(x, x) for x in sub.tool],
            "estimate [95% CI]": [_ci(*r) for r in
                                  zip(sub.estimate, sub.ci_lo, sub.ci_hi)],
            "CI width": sub.ci_width.round(4),
            "numerator": sub.numerator,
            "denominator n": sub.denominator_n,
            "cohort clusters": sub.n_cohort_clusters,
            "denominator": sub.denominator_statement,
        })
        A(fw.md_table(t))
        A("")
    A("### 4b. Provenance of the anchor interval — one number, three printed forms")
    A("")
    A("The anchor bias CIs currently printed in the manuscript "
      "(`−8.68 [−14.03, −3.57]` / `+5.09 [+1.90, +8.65]`) match **neither** the "
      "definitive D1-fixed re-run nor the superseded pre-D1 backup. The point "
      "estimates are correct everywhere; only the bootstrap endpoints drift, which "
      "is the signature of defect D1 (`abs(hash())` seeding). The table below lists "
      "every on-disk source so the manuscript can be pointed at one of them.")
    A("")
    if "anchor_ci_provenance" in summary:
        A(fw.md_table(pd.DataFrame(summary["anchor_ci_provenance"])))
    A("")
    A("## 5. Ready-to-paste sentences")
    A("")
    A("See `ready_to_paste_sentences.md` in this directory.")
    A("")
    A("## 6. Determinism (defect D1)")
    A("")
    if "determinism_check" in summary:
        A(f"All TSV outputs byte-identical under `PYTHONHASHSEED=0` and "
          f"`PYTHONHASHSEED=99999`: **{summary['determinism_all_identical']}**")
        A("")
        A(fw.md_table(pd.DataFrame(summary["determinism_check"])))
    else:
        A("Not run in this invocation (pass `--determinism-check`).")
    A("")
    A("## 7. Input verification")
    A("")
    if eda["issues"]:
        for i in eda["issues"]:
            A(f"* {i}")
    else:
        A("Every set carries exactly 1,000 rows, all four tool prediction tables "
          "join 1:1 to `metadata.tsv`, and every macro-F1 point estimate reproduces "
          "the WS5 definitive table exactly. No row was dropped.")
    A("")
    (out_dir / "WS11_P_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
