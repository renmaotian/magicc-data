#!/usr/bin/env python
"""
104_clustered_statistics.py  --  WS5.4 + WS5.5
==============================================
WS5.4  Statistical overhaul demanded by Reviewer 1 (minor comment 16):

       * two-sided paired Wilcoxon signed-rank tests replace the one-sided tests
         used in the submitted manuscript;
       * clustering by dominant reference genome is accounted for by (a) a
         cluster bootstrap that resamples reference genomes rather than
         individual simulated genomes and (b) a cluster-mean paired test whose
         unit of analysis IS the reference genome;
       * Benjamini-Hochberg FDR correction across the whole family of
         comparisons produced in one run;
       * effect sizes reported alongside every p-value: Cliff's delta,
         matched-pairs rank-biserial correlation, and the Hodges-Lehmann median
         paired difference with a cluster-bootstrap CI.

WS5.5  Table S2 rebuilt: MAE and R^2 per set per tool with bootstrap 95% CIs,
       omitting R^2 (with an explicit printed reason) wherever the true value
       has zero variance, and separating the coefficient of determination from
       the squared Pearson correlation.

    python scripts/104_clustered_statistics.py [--config ...] [--n-boot 2000]
                                               [--n-boot-slow 1000]
                                               [--tiers primary reported secondary]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent


def _load_framework():
    spec = importlib.util.spec_from_file_location(
        "magicc_metrics_framework", _HERE / "101_metrics_framework.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["magicc_metrics_framework"] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
from scipy import stats as sps   # noqa: E402


# ---------------------------------------------------------------------------
# WS5.4  paired comparison with clustering
# ---------------------------------------------------------------------------


def paired_comparison(df: pd.DataFrame, ref: str, comp: str, metric: str,
                      cfg, bs: "fw.Bootstrapper", bs_slow: "fw.Bootstrapper",
                      cluster_col: str = "cluster_id") -> Optional[dict]:
    """Two-sided, clustered, effect-size-annotated comparison of two tools.

    ``metric`` is one of ``abs_err_completeness`` / ``abs_err_contamination``.
    Paired difference d_i = error(ref)_i - error(comp)_i, so **negative d means
    the reference tool (MAGICC) is more accurate**.
    """
    ca, cb = f"{metric}__{ref}", f"{metric}__{comp}"
    if ca not in df.columns or cb not in df.columns:
        return None
    a = df[ca].to_numpy(float)
    b = df[cb].to_numpy(float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10:
        return None
    d_all = a - b

    # ---- naive (genome-level, independence-assuming) two-sided Wilcoxon -----
    d = d_all[ok]
    if np.all(d == 0):
        p_naive, w_stat = 1.0, np.nan
    else:
        wr = sps.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        p_naive, w_stat = float(wr.pvalue), float(wr.statistic)

    # ---- cluster-mean paired Wilcoxon (unit of analysis = reference genome) --
    cl = df[cluster_col].to_numpy()
    g = pd.DataFrame({"cl": cl[ok], "a": a[ok], "b": b[ok]}).groupby("cl").mean()
    dc = (g["a"] - g["b"]).to_numpy()
    if dc.size >= 6 and np.any(dc != 0):
        wrc = sps.wilcoxon(dc, alternative="two-sided", zero_method="wilcox")
        p_cluster_mean, w_cluster = float(wrc.pvalue), float(wrc.statistic)
    else:
        p_cluster_mean, w_cluster = float("nan"), float("nan")

    # ---- cluster bootstrap on the median / mean paired difference -----------
    def med(idx):
        x = d_all[idx]
        x = x[~np.isnan(x)]
        return float(np.median(x)) if x.size else np.nan

    def mean(idx):
        x = d_all[idx]
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    ci_med = bs.ci(med)
    ci_mean = bs.ci(mean)
    p_boot_med = bs.p_two_sided(med)[0]
    p_boot_mean = bs.p_two_sided(mean)[0]

    # ---- Hodges-Lehmann median paired difference (Walsh averages) -----------
    def hl(idx):
        x = d_all[idx]
        x = x[~np.isnan(x)]
        return fw.hodges_lehmann_paired(x)

    if bs_slow is None:
        # secondary/legacy sets: point estimate only (the O(n^2) Walsh-average
        # bootstrap is the run's bottleneck and these sets carry no claims)
        ci_hl = {"estimate": hl(np.arange(len(d_all))), "ci_lo": np.nan,
                 "ci_hi": np.nan}
    else:
        ci_hl = bs_slow.ci(hl)

    # ---- effect sizes ------------------------------------------------------
    delta = fw.cliffs_delta(a[ok], b[ok])
    ci_delta = bs.ci(lambda idx: fw.cliffs_delta(a[idx][~np.isnan(a[idx])],
                                                 b[idx][~np.isnan(b[idx])]),
                     point=delta)
    r_rb = fw.rank_biserial_paired(d)

    def mag(x):
        ax = abs(x)
        return ("negligible" if ax < 0.147 else "small" if ax < 0.33
                else "medium" if ax < 0.474 else "large")

    return {
        "metric": metric, "reference_tool": ref, "comparison_tool": comp,
        "n_pairs": int(ok.sum()), "n_clusters": int(pd.unique(cl[ok]).size),
        "mae_reference": float(np.mean(a[ok])), "mae_comparison": float(np.mean(b[ok])),
        "mean_paired_difference": ci_mean["estimate"],
        "mean_diff_ci_lo": ci_mean["ci_lo"], "mean_diff_ci_hi": ci_mean["ci_hi"],
        "median_paired_difference": ci_med["estimate"],
        "median_diff_ci_lo": ci_med["ci_lo"], "median_diff_ci_hi": ci_med["ci_hi"],
        "hodges_lehmann_diff": ci_hl["estimate"],
        "hl_ci_lo": ci_hl["ci_lo"], "hl_ci_hi": ci_hl["ci_hi"],
        "cliffs_delta": delta, "cliffs_delta_ci_lo": ci_delta["ci_lo"],
        "cliffs_delta_ci_hi": ci_delta["ci_hi"],
        "cliffs_delta_magnitude": mag(delta),
        "rank_biserial_paired": r_rb,
        "wilcoxon_statistic_naive": w_stat,
        "p_wilcoxon_two_sided_naive": p_naive,
        "wilcoxon_statistic_cluster_mean": w_cluster,
        "n_clusters_used_cluster_mean": int(dc.size),
        "p_wilcoxon_two_sided_cluster_mean": p_cluster_mean,
        "p_cluster_bootstrap_median": p_boot_med,
        "p_cluster_bootstrap_mean": p_boot_mean,
        "favours": ("reference" if ci_mean["estimate"] < 0 else
                    "comparison" if ci_mean["estimate"] > 0 else "tie"),
        "boot_clustered": bs.clustered, "boot_n_iter": bs.n_iter,
        "boot_n_iter_slow": (bs_slow.n_iter if bs_slow is not None else 0),
    }


# ---------------------------------------------------------------------------
# WS5.5  accuracy table with CIs
# ---------------------------------------------------------------------------


def accuracy_rows(df: pd.DataFrame, tools: List[str], bsobj, cfg,
                  bs: "fw.Bootstrapper") -> List[dict]:
    rows = []
    for tool in tools:
        for metric in ("completeness", "contamination"):
            y = df[f"true_{metric}"].to_numpy(float)
            yh = df[f"pred_{metric}__{tool}"].to_numpy(float)
            ok = ~(np.isnan(y) | np.isnan(yh))
            if ok.sum() == 0:
                continue
            var = float(np.var(y[ok]))
            constant = var == 0.0
            boot = bs.ci_multi(lambda idx, _y=y, _h=yh: {
                "mae": fw.mae(_y[idx], _h[idx]),
                "rmse": fw.rmse(_y[idx], _h[idx]),
                "bias": fw.bias(_y[idx], _h[idx]),
                "r2": fw.r2_coefficient_of_determination(_y[idx], _h[idx]),
                "r2_pearson_sq": fw.r2_pearson_squared(_y[idx], _h[idx])})
            rec = {
                "set": bsobj.name, "set_label": bsobj.label, "status": bsobj.status,
                "tier": bsobj.tier, "tool": tool, "tool_label": cfg.tool_label(tool),
                "tool_role": cfg.tool_role(tool), "metric": metric,
                "n": int(ok.sum()), "n_clusters": bs.n_clusters,
                "true_value_variance": var,
                "true_value_constant": constant,
                "true_value_min": float(np.min(y[ok])), "true_value_max": float(np.max(y[ok])),
                "mae": boot["mae"]["estimate"], "mae_ci_lo": boot["mae"]["ci_lo"],
                "mae_ci_hi": boot["mae"]["ci_hi"],
                "rmse": boot["rmse"]["estimate"], "rmse_ci_lo": boot["rmse"]["ci_lo"],
                "rmse_ci_hi": boot["rmse"]["ci_hi"],
                "bias": boot["bias"]["estimate"], "bias_ci_lo": boot["bias"]["ci_lo"],
                "bias_ci_hi": boot["bias"]["ci_hi"],
            }
            if constant:
                rec.update({
                    "r2": np.nan, "r2_ci_lo": np.nan, "r2_ci_hi": np.nan,
                    "r2_pearson_sq": np.nan,
                    "r2_omitted_reason": (
                        f"R^2 is undefined: the true {metric} is constant at "
                        f"{y[ok][0]:.4g}% for all {int(ok.sum())} genomes in this set, "
                        f"so the total sum of squares is zero and no fraction of "
                        f"variance can be explained. MAE, RMSE and mean signed error "
                        f"remain well defined and are reported instead.")})
            else:
                rec.update({
                    "r2": boot["r2"]["estimate"], "r2_ci_lo": boot["r2"]["ci_lo"],
                    "r2_ci_hi": boot["r2"]["ci_hi"],
                    "r2_pearson_sq": boot["r2_pearson_sq"]["estimate"],
                    "r2_pearson_sq_ci_lo": boot["r2_pearson_sq"]["ci_lo"],
                    "r2_pearson_sq_ci_hi": boot["r2_pearson_sq"]["ci_hi"],
                    "r2_omitted_reason": ""})
            rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_set(cfg, bsobj, n_boot, n_boot_slow, cluster_col="cluster_id"):
    df, tools, _ = fw.load_set(cfg, bsobj)
    if not tools:
        return [], [], pd.DataFrame()
    seed = cfg.seed + (fw.stable_hash(bsobj.name) % 100000)
    bs = fw.Bootstrapper(clusters=df[cluster_col].to_numpy(), n_iter=n_boot,
                         ci_level=cfg.ci_level, seed=seed)
    bs.resample_indices()
    if bsobj.tier in ("primary", "reported"):
        bs_slow = fw.Bootstrapper(clusters=df[cluster_col].to_numpy(),
                                  n_iter=n_boot_slow, ci_level=cfg.ci_level,
                                  seed=seed + 1)
        bs_slow.resample_indices()
    else:
        bs_slow = None

    st = cfg.raw["statistics"]
    ref = st["reference_tool"]
    tests = []
    if ref in tools:
        for comp in st["comparison_tools"]:
            if comp not in tools:
                continue
            for metric in st["metrics"]:
                r = paired_comparison(df, ref, comp, metric, cfg, bs, bs_slow,
                                      cluster_col)
                if r is None:
                    continue
                r.update({"set": bsobj.name, "set_label": bsobj.label,
                          "status": bsobj.status, "tier": bsobj.tier,
                          "cluster_column": cluster_col})
                tests.append(r)
    acc = accuracy_rows(df, tools, bsobj, cfg, bs)
    return tests, acc, df


def pooled_accuracy(cfg, set_names, tools_common, label, n_boot):
    """MAE / R^2 pooled over several sets (for the 'Overall' Table S2c row)."""
    frames = []
    for b in fw.discover_sets(cfg, include_missing=False):
        if b.name not in set_names:
            continue
        df, tools, _ = fw.load_set(cfg, b)
        keep = ["cluster_id", "true_completeness", "true_contamination"]
        for t in tools_common:
            if f"pred_completeness__{t}" not in df.columns:
                return []
            keep += [f"pred_completeness__{t}", f"pred_contamination__{t}"]
        sub = df[keep].copy()
        sub["cluster_id"] = b.name + "|" + sub["cluster_id"].astype(str)
        frames.append(sub)
    if len(frames) != len(set_names):
        return []
    big = pd.concat(frames, ignore_index=True)
    bs = fw.Bootstrapper(clusters=big["cluster_id"].to_numpy(), n_iter=n_boot,
                         ci_level=cfg.ci_level, seed=cfg.seed + 777)
    bs.resample_indices()

    pooled_obj = type("PooledSet", (), {
        "name": label, "label": f"POOLED: {', '.join(set_names)}",
        "status": "pooled", "tier": "pooled", "design": ""})()
    return accuracy_rows(big, list(tools_common), pooled_obj, cfg, bs)


def verify_published(cfg) -> pd.DataFrame:
    """Compare every recomputed MAE / R^2 / MIMAG F1 against the published value.

    Reads the outputs of scripts 102 and 104 that are already on disk, so it can
    be run on its own with ``--verify-only``.
    """
    pv = cfg.raw.get("published_reference_values")
    if not pv:
        return pd.DataFrame()
    A = pd.read_csv(cfg.out_dir / "ws5.5_table_S2_rebuilt.tsv", sep="\t")
    tol, tol_r2 = float(pv["tolerance_pp"]), float(pv["tolerance_r2"])
    rows = []

    def look(setname, tool, metric, col):
        m = A[(A["set"] == setname) & (A.tool == tool) & (A.metric == metric)]
        return (float(m[col].iloc[0]), m) if len(m) else (np.nan, m)

    for src, sname, tool, metric, published in pv.get("mae", []):
        got, m = look(sname, tool, metric, "mae")
        lo = float(m.mae_ci_lo.iloc[0]) if len(m) else np.nan
        hi = float(m.mae_ci_hi.iloc[0]) if len(m) else np.nan
        rows.append({"source": src, "quantity": "MAE", "set": sname, "tool": tool,
                     "metric": metric, "published": published, "recomputed": got,
                     "recomputed_ci_lo": lo, "recomputed_ci_hi": hi,
                     "difference": got - published if np.isfinite(got) else np.nan,
                     "agrees_within_tolerance": bool(np.isfinite(got) and
                                                     abs(got - published) <= tol),
                     "published_inside_recomputed_ci": bool(
                         np.isfinite(lo) and lo <= published <= hi),
                     "note": "" if np.isfinite(got) else "not recomputed (set/tool absent)"})
    for src, sname, tool, metric, published in pv.get("r2", []):
        got, m = look(sname, tool, metric, "r2")
        pear, _ = look(sname, tool, metric, "r2_pearson_sq")
        note = ""
        agrees = bool(np.isfinite(got) and abs(got - published) <= tol_r2)
        if not agrees and np.isfinite(pear) and abs(pear - published) <= tol_r2:
            note = ("published value matches the SQUARED PEARSON CORRELATION r^2, "
                    "not the coefficient of determination R^2 - metric mix-up")
        rows.append({"source": src, "quantity": "R2", "set": sname, "tool": tool,
                     "metric": metric, "published": published, "recomputed": got,
                     "recomputed_ci_lo": float(m.r2_ci_lo.iloc[0]) if len(m) and "r2_ci_lo" in m else np.nan,
                     "recomputed_ci_hi": float(m.r2_ci_hi.iloc[0]) if len(m) and "r2_ci_hi" in m else np.nan,
                     "recomputed_pearson_r2": pear,
                     "difference": got - published if np.isfinite(got) else np.nan,
                     "agrees_within_tolerance": agrees, "note": note})
    mp = cfg.out_dir / "ws5.1_mimag_overall_metrics.tsv"
    if mp.exists():
        M = pd.read_csv(mp, sep="\t")
        for src, snames, tool, plo, phi, variant in pv.get("mimag_macro_f1_ranges", []):
            sub = M[(M["set"].isin(snames)) & (M.tool == tool)]
            if sub.empty:
                continue
            got = sub[variant].to_numpy()
            rows.append({
                "source": src, "quantity": f"MIMAG {variant} range over "
                                           f"{'+'.join(snames)}",
                "set": "+".join(snames), "tool": tool, "metric": "-",
                "published": f"{plo:.2f}-{phi:.2f}",
                "recomputed": f"{got.min():.3f}-{got.max():.3f}",
                "difference": np.nan,
                "agrees_within_tolerance": bool(abs(got.min() - plo) <= 0.01 and
                                                abs(got.max() - phi) <= 0.01),
                "note": "published as a RANGE over two sets; Reviewer 1 asks for "
                        "per-set values with CIs, now in ws5.1_mimag_overall_metrics.tsv"})
    return pd.DataFrame(rows)


def write_verification(cfg, V: pd.DataFrame):
    if V.empty:
        return
    V.to_csv(cfg.out_dir / "verification_vs_published.tsv", sep="\t", index=False)
    bad = V[~V.agrees_within_tolerance.astype(bool)]
    L = ["# Verification of every recomputed number against the submitted "
         "manuscript and the progress document", "",
         f"{len(V)} published values checked; "
         f"{int(V.agrees_within_tolerance.astype(bool).sum())} reproduce within "
         f"tolerance ({cfg.raw['published_reference_values']['tolerance_pp']} pp for "
         f"MAE, {cfg.raw['published_reference_values']['tolerance_r2']} for R^2); "
         f"**{len(bad)} do not** and are listed first.", ""]
    if len(bad):
        L += ["## Discrepancies", "", fw.md_table(bad.reindex(columns=[
            "source", "quantity", "set", "tool", "metric", "published",
            "recomputed", "recomputed_pearson_r2", "difference", "note"])), ""]
    L += ["## All checks", "", fw.md_table(V.reindex(columns=[
        "source", "quantity", "set", "tool", "metric", "published", "recomputed",
        "recomputed_ci_lo", "recomputed_ci_hi", "difference",
        "agrees_within_tolerance", "published_inside_recomputed_ci", "note"])), ""]
    (cfg.out_dir / "verification_vs_published.md").write_text("\n".join(L) + "\n")
    print(f"[104] verification: {len(V)} published values, {len(bad)} discrepancies "
          f"-> verification_vs_published.{{tsv,md}}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-boot-slow", type=int, default=None)
    ap.add_argument("--tiers", nargs="*", default=["primary", "reported", "secondary"])
    ap.add_argument("--skip-phylum-sensitivity", action="store_true")
    ap.add_argument("--reports-only", action="store_true",
                    help="re-render the markdown reports from the TSVs already "
                         "on disk (no recomputation)")
    ap.add_argument("--verify-only", action="store_true",
                    help="only re-run the published-value verification against "
                         "the TSVs already on disk")
    args = ap.parse_args(argv)

    cfg = fw.load_config(args.config)
    if args.verify_only:
        write_verification(cfg, verify_published(cfg))
        return 0
    if args.reports_only:
        o = cfg.out_dir
        T = pd.read_csv(o / "ws5.4_clustered_tests.tsv", sep="\t")
        A = pd.read_csv(o / "ws5.5_table_S2_rebuilt.tsv", sep="\t")
        sp = o / "ws5.4_phylum_clustered_sensitivity.tsv"
        S = pd.read_csv(sp, sep="\t") if sp.exists() else pd.DataFrame()
        names = list(dict.fromkeys(T["set"]).keys()) + [
            n for n in A["set"].drop_duplicates() if n not in set(T["set"])]
        allsets = {b.name: b for b in fw.discover_sets(cfg, include_missing=False)}
        sets = [allsets[n] for n in names if n in allsets]
        write_reports(cfg, T, A, S, sets,
                      int(T.boot_n_iter.max()) if "boot_n_iter" in T else cfg.n_boot,
                      int(T.boot_n_iter_slow.max()) if "boot_n_iter_slow" in T else cfg.n_boot_slow,
                      cfg.raw["statistics"]["bh_alpha"])
        write_verification(cfg, verify_published(cfg))
        print(f"[104] re-rendered reports from existing TSVs in {o}")
        return 0
    n_boot = args.n_boot or cfg.n_boot
    n_slow = args.n_boot_slow or cfg.n_boot_slow
    sets = fw.discover_sets(cfg, include_missing=False, tiers=args.tiers)

    tests, acc = [], []
    for b in sets:
        print(f"  [104] {b.name} ...", flush=True)
        t, a, _ = process_set(cfg, b, n_boot, n_slow)
        tests += t
        acc += a

    # sensitivity: cluster by dominant phylum instead of dominant genome
    sens = []
    if not args.skip_phylum_sensitivity:
        for b in sets:
            if b.tier not in ("primary", "reported"):
                continue
            print(f"  [104] {b.name} (phylum-clustered sensitivity) ...", flush=True)
            df, tools, _ = fw.load_set(cfg, b)
            if not tools:
                continue
            df = df.copy()
            df["phylum_cluster"] = df["dominant_phylum"].astype(str)
            n_phy = df["phylum_cluster"].nunique()
            if n_phy < 5:
                print(f"      skipped: only {n_phy} distinct phyla - a cluster "
                      f"bootstrap over <5 clusters is not interpretable")
                continue
            seed = cfg.seed + 31 + (fw.stable_hash(b.name) % 100000)
            bs = fw.Bootstrapper(clusters=df["phylum_cluster"].to_numpy(),
                                 n_iter=n_boot, ci_level=cfg.ci_level, seed=seed)
            bs.resample_indices()
            bs_slow = fw.Bootstrapper(clusters=df["phylum_cluster"].to_numpy(),
                                      n_iter=min(200, n_slow), ci_level=cfg.ci_level,
                                      seed=seed + 1)
            bs_slow.resample_indices()
            st = cfg.raw["statistics"]
            if st["reference_tool"] not in tools:
                continue
            for comp in st["comparison_tools"]:
                if comp not in tools:
                    continue
                for metric in st["metrics"]:
                    r = paired_comparison(df, st["reference_tool"], comp, metric,
                                          cfg, bs, bs_slow, "phylum_cluster")
                    if r is None:
                        continue
                    r.update({"set": b.name, "status": b.status, "tier": b.tier,
                              "cluster_column": "dominant_phylum"})
                    sens.append(r)

    T = pd.DataFrame(tests)
    # ---- Benjamini-Hochberg within each tier family -----------------------
    T["bh_family"] = T["tier"]
    for pcol, qcol in [("p_wilcoxon_two_sided_naive", "q_bh_wilcoxon_naive"),
                       ("p_wilcoxon_two_sided_cluster_mean", "q_bh_wilcoxon_cluster_mean"),
                       ("p_cluster_bootstrap_mean", "q_bh_cluster_bootstrap_mean"),
                       ("p_cluster_bootstrap_median", "q_bh_cluster_bootstrap_median")]:
        T[qcol] = np.nan
        for fam, idx in T.groupby("bh_family").groups.items():
            T.loc[idx, qcol] = fw.bh_correct(T.loc[idx, pcol].to_numpy())
    alpha = cfg.raw["statistics"]["bh_alpha"]
    # PRIMARY decision rule: rank-based paired test with the reference genome as
    # the unit of analysis, BH-corrected within the family.
    T["significant_bh_primary"] = T["q_bh_wilcoxon_cluster_mean"] < alpha
    # CONFIRMATORY: cluster bootstrap of the difference in MAE (= mean paired
    # difference), which is the quantity the manuscript's accuracy claims are about.
    T["significant_bh_cluster_bootstrap_mean"] = T["q_bh_cluster_bootstrap_mean"] < alpha
    T["significant_bh_naive"] = T["q_bh_wilcoxon_naive"] < alpha
    T["significant_all_three_bh"] = (
        T["significant_bh_naive"] & T["significant_bh_primary"] &
        T["significant_bh_cluster_bootstrap_mean"])

    A = pd.DataFrame(acc)
    # pooled rows for version reconciliation with the published tables
    manuscript5 = ["set_A_v2", "set_B_v2", "set_C", "set_D", "set_E"]
    present = {b.name for b in sets}
    pooled = []
    if set(manuscript5) <= present:
        pooled += pooled_accuracy(cfg, manuscript5,
                                  ["magicc_v5", "magicc_v4", "magicc_v3", "checkm2",
                                   "cocopye", "deepcheck"],
                                  "POOLED_manuscript_5_sets", n_boot)
    clean5 = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]
    if set(clean5) <= present:
        # Prefer the largest tool set that is present on ALL FIVE sets, so that
        # every pooled MAE is computed on exactly the same genomes (paired
        # comparison). magicc_v3/v4 were never run on the clean C/D rebuilds, so
        # the 4-tool intersection (v5 + the three competitors) is the headline row.
        for tools_common in (["magicc_v5", "magicc_v4", "magicc_v3", "checkm2",
                              "cocopye", "deepcheck"],
                             ["magicc_v5", "checkm2", "cocopye", "deepcheck"],
                             ["magicc_v5"]):
            p = pooled_accuracy(cfg, clean5, tools_common, "POOLED_leakage_free_5_sets",
                                n_boot)
            if p:
                pooled += p
                break
    if pooled:
        A = pd.concat([A, pd.DataFrame(pooled)], ignore_index=True)

    o = cfg.out_dir
    order = {b.name: i for i, b in enumerate(sets)}
    torder = {t: i for i, t in enumerate(cfg.tools)}
    T["_s"] = T["set"].map(order)
    T["_t"] = T["comparison_tool"].map(torder)
    T = T.sort_values(["_s", "metric", "_t"]).drop(columns=["_s", "_t"])
    A["_s"] = A["set"].map(order).fillna(999)
    A["_t"] = A["tool"].map(torder)
    A = A.sort_values(["_s", "metric", "_t"]).drop(columns=["_s", "_t"])

    T.to_csv(o / "ws5.4_clustered_tests.tsv", sep="\t", index=False)
    A.to_csv(o / "ws5.5_table_S2_rebuilt.tsv", sep="\t", index=False)
    if sens:
        pd.DataFrame(sens).to_csv(o / "ws5.4_phylum_clustered_sensitivity.tsv",
                                  sep="\t", index=False)
    write_reports(cfg, T, A, pd.DataFrame(sens), sets, n_boot, n_slow, alpha)
    write_verification(cfg, verify_published(cfg))
    print(f"[104] wrote ws5.4_* and ws5.5_* to {o}")
    return 0


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

METHODS_TEXT = """\
### Statistical methods (text for the Methods section)

Every tool-vs-tool comparison is a **two-sided** test; the one-sided Wilcoxon
signed-rank tests of the original submission have been withdrawn because they
presuppose the direction of the effect they are used to establish.

For each benchmark set and each accuracy measure (absolute completeness error,
absolute contamination error) we form the paired per-genome difference
d_i = |error_MAGICC,i| - |error_competitor,i|, so that d_i < 0 means MAGICC was
more accurate on genome i. Three complementary tests are reported:

1. **Genome-level two-sided Wilcoxon signed-rank test.** This treats the
   simulated genomes as independent and is included only for comparability with
   the original submission; it is anti-conservative whenever several simulated
   genomes derive from the same reference genome.

2. **Cluster-mean two-sided Wilcoxon signed-rank test.** Absolute errors are
   first averaged within each dominant reference genome; the test is then run on
   the resulting per-reference means, so the unit of analysis is the reference
   genome and the number of independent units equals the number of distinct
   reference genomes (e.g. 100 for Sets C-clean/D-clean, which contain 10
   independent simulations per reference).

3. **Cluster bootstrap.** Reference genomes (clusters), not individual simulated
   genomes, are resampled with replacement 2,000 times, taking all simulated
   genomes belonging to each sampled reference. Percentile 95% confidence
   intervals and two-sided p-values (p = 2 x min[Pr(theta* <= 0), Pr(theta* >= 0)],
   with a +1/(B+1) continuity correction, so the smallest attainable p-value is
   ~1/1000 at 2,000 replicates) are obtained for the mean paired difference -
   which is exactly the difference in MAE between the two tools - and for the
   median paired difference. A coverage simulation with 40 clusters of 10
   observations and an intraclass correlation of 0.8 gives 95.0% coverage for the
   cluster bootstrap versus 54.0% for the naive bootstrap (script 101
   --selftest), confirming that ignoring clustering understates uncertainty
   severely in exactly the design used for Sets C-clean/D-clean.

**Which test is the primary one.** Test 2 (cluster-mean Wilcoxon) is the primary
decision rule: it is rank-based, two-sided, and treats the reference genome as
the sampling unit, which is precisely what Reviewer 1 asks for. Test 3 on the
mean paired difference is the confirmatory test, because the difference in MAE is
the quantity the manuscript's accuracy claims are about. Test 1 is reported for
comparability only. A comparison is called robust only when all three survive BH
correction (column `significant_all_three_bh`). Note that the bootstrap p-value
for the MEDIAN paired difference can be large even when the mean difference is
decisively non-zero: when most paired differences are near zero and the advantage
comes from the tails (as for completeness on Set A), the median of d is close to
zero by construction. That is a property of the statistic, not a failure of the
test, which is why the mean (dMAE) version is the confirmatory one.

A sensitivity analysis repeats (3) with clusters defined by dominant **phylum**
instead of dominant genome, i.e. the most conservative taxonomic grouping the
data support.

**Multiplicity.** Benjamini-Hochberg FDR correction at q < 0.05 is applied
across the entire family of tests reported in one run, separately for the
leakage-free ("primary") sets, for the superseded leaky Sets C/D ("reported"),
and for the secondary/legacy sets, so that the primary family is not diluted by
withdrawn results.

**Effect sizes.** Every p-value is accompanied by (i) the Hodges-Lehmann
estimator of the median paired difference (median of the Walsh averages of d)
in percentage points, with a cluster-bootstrap 95% CI; (ii) Cliff's delta
comparing the two absolute-error distributions, with a cluster-bootstrap 95% CI
and the conventional magnitude labels (|delta| < 0.147 negligible, < 0.33 small,
< 0.474 medium, else large); and (iii) the matched-pairs rank-biserial
correlation, the effect size that corresponds directly to the signed-rank
statistic. Negative Hodges-Lehmann differences and negative Cliff's delta both
indicate that MAGICC is the more accurate tool. To keep run time acceptable the
O(n^2) Walsh-average bootstrap CI for the Hodges-Lehmann estimator is computed
for the primary and reported (superseded) sets only; for the secondary/legacy
sets the Hodges-Lehmann point estimate is given without an interval.
"""


def _ci(v, lo, hi, nd=2, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return f.format(v)
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def write_reports(cfg, T, A, S, sets, n_boot, n_slow, alpha):
    den = fw.caption_denominator(cfg)

    # ---------------- WS5.4 ----------------
    L = ["# WS5.4 - Two-sided, cluster-aware, multiplicity-corrected statistics", "",
         den, "", METHODS_TEXT, "",
         f"Bootstrap replicates: {n_boot} (median/mean/Cliff's delta), "
         f"{n_slow} (Hodges-Lehmann). BH alpha = {alpha}.", "",
         "**Sign convention: negative values favour MAGICC.**", ""]
    for fam in ["primary", "reported", "secondary"]:
        sub = T[T.bh_family == fam]
        if sub.empty:
            continue
        title = {"primary": "Primary family (leakage-free sets)",
                 "reported": "Superseded family (Sets C/D - training-set leakage; "
                             "reported for transparency only)",
                 "secondary": "Secondary family (v1 sets and motivating sets)"}[fam]
        L += [f"## {title}", "",
              f"Family size m = {len(sub)} tests; BH applied within this family.", ""]
        for metric in sub.metric.drop_duplicates():
            sm = sub[sub.metric == metric]
            t = pd.DataFrame({
                "set": sm["set"],
                "MAGICC v5 vs": sm.comparison_tool,
                "n / clusters": [f"{a}/{b}" for a, b in zip(sm.n_pairs, sm.n_clusters)],
                "MAE ref": sm.mae_reference.round(3),
                "MAE comp": sm.mae_comparison.round(3),
                "HL median diff (pp) [95% CI]": [
                    _ci(*r, sign=True) for r in zip(sm.hodges_lehmann_diff,
                                                    sm.hl_ci_lo, sm.hl_ci_hi)],
                "Cliff delta [95% CI]": [
                    _ci(*r, nd=3, sign=True) for r in zip(sm.cliffs_delta,
                                                          sm.cliffs_delta_ci_lo,
                                                          sm.cliffs_delta_ci_hi)],
                "|delta|": sm.cliffs_delta_magnitude,
                "r_rb": sm.rank_biserial_paired.round(3),
                "mean diff = dMAE (pp) [95% CI]": [
                    _ci(*r, sign=True) for r in zip(sm.mean_paired_difference,
                                                    sm.mean_diff_ci_lo,
                                                    sm.mean_diff_ci_hi)],
                "p naive": [f"{p:.2g}" for p in sm.p_wilcoxon_two_sided_naive],
                "p cluster-mean (PRIMARY)": [f"{p:.2g}" for p in sm.p_wilcoxon_two_sided_cluster_mean],
                "q BH cluster-mean (PRIMARY)": [f"{q:.2g}" for q in sm.q_bh_wilcoxon_cluster_mean],
                "p cluster-boot (dMAE)": [f"{p:.2g}" for p in sm.p_cluster_bootstrap_mean],
                "q BH cluster-boot (dMAE)": [f"{q:.2g}" for q in sm.q_bh_cluster_bootstrap_mean],
                "survives BH (all 3)": sm.significant_all_three_bh,
                "favours": sm.favours,
            })
            L += [f"### {metric}", "", fw.md_table(t), ""]
    if not S.empty:
        L += ["## Sensitivity analysis: clusters defined by dominant phylum", "",
              "The most conservative grouping the data support (clusters = phyla, "
              "so only a handful of independent units per set). Reported because "
              "Reviewer 1 asks for clustering by reference genome **or taxonomic "
              "group**.", ""]
        for metric in S.metric.drop_duplicates():
            sm = S[S.metric == metric]
            t = pd.DataFrame({
                "set": sm["set"], "MAGICC v5 vs": sm.comparison_tool,
                "n phyla (clusters)": sm.n_clusters,
                "HL median diff (pp) [95% CI]": [
                    _ci(*r, sign=True) for r in zip(sm.hodges_lehmann_diff,
                                                    sm.hl_ci_lo, sm.hl_ci_hi)],
                "median diff [95% CI]": [
                    _ci(*r, sign=True) for r in zip(sm.median_paired_difference,
                                                    sm.median_diff_ci_lo,
                                                    sm.median_diff_ci_hi)],
                "p cluster-boot (phylum)": [f"{p:.2g}" for p in sm.p_cluster_bootstrap_median],
                "p Wilcoxon phylum-mean": [f"{p:.2g}" for p in sm.p_wilcoxon_two_sided_cluster_mean],
            })
            L += [f"### {metric}", "", fw.md_table(t), ""]
    (cfg.out_dir / "ws5.4_clustered_tests.md").write_text("\n".join(L) + "\n")

    # ---------------- WS5.5 ----------------
    L = ["# WS5.5 - Table S2 rebuilt with confidence intervals", "",
         den, "",
         f"MAE, RMSE, mean signed error (bias) and R^2 with percentile bootstrap "
         f"95% CIs ({n_boot} replicates, clusters resampled by dominant reference "
         f"genome). All values in percentage points except R^2.", "",
         "**R^2 is the coefficient of determination, R^2 = 1 - SS_res/SS_tot.** "
         "It is omitted, with the reason printed, wherever the true value has zero "
         "variance. The squared Pearson correlation r^2 is reported in a separate "
         "column because it is NOT the same quantity and the submitted "
         "supplementary table mixes the two (see the discrepancy note below).", ""]
    for b in list(sets) + [type("P", (), {"name": n, "label": n, "status": "pooled",
                                          "tier": "pooled", "design": ""})()
                           for n in A[A.tier == "pooled"]["set"].drop_duplicates()]:
        sub = A[A["set"] == b.name]
        if sub.empty:
            continue
        flag = "  **[SUPERSEDED - training-set leakage]**" if getattr(b, "status", "") == "superseded_leaky" else ""
        L += [f"## {b.name} - {getattr(b, 'label', b.name)}{flag}", ""]
        for metric in ("completeness", "contamination"):
            sm = sub[sub.metric == metric]
            if sm.empty:
                continue
            const = bool(sm.true_value_constant.iloc[0])
            L += [f"### {metric}", ""]
            if const:
                L += ["> " + sm.r2_omitted_reason.iloc[0], ""]
            t = pd.DataFrame({
                "tool": sm.tool_label, "n": sm.n.astype(int),
                "MAE [95% CI]": [_ci(*r) for r in zip(sm.mae, sm.mae_ci_lo, sm.mae_ci_hi)],
                "RMSE [95% CI]": [_ci(*r) for r in zip(sm.rmse, sm.rmse_ci_lo, sm.rmse_ci_hi)],
                "mean signed error [95% CI]": [
                    _ci(*r, sign=True) for r in zip(sm.bias, sm.bias_ci_lo, sm.bias_ci_hi)],
                "R^2 (coeff. of determination) [95% CI]": [
                    ("omitted - true value constant" if const else _ci(*r, nd=3))
                    for r in zip(sm.r2, sm.get("r2_ci_lo", sm.r2 * np.nan),
                                 sm.get("r2_ci_hi", sm.r2 * np.nan))],
                "r^2 (squared Pearson) [95% CI]": [
                    ("undefined" if const else _ci(*r, nd=3))
                    for r in zip(sm.r2_pearson_sq,
                                 sm.get("r2_pearson_sq_ci_lo", sm.r2_pearson_sq * np.nan),
                                 sm.get("r2_pearson_sq_ci_hi", sm.r2_pearson_sq * np.nan))],
            })
            L += [fw.md_table(t), ""]
    (cfg.out_dir / "ws5.5_table_S2_rebuilt.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
