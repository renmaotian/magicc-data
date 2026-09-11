#!/usr/bin/env python
"""
103_signed_errors.py  --  WS5.3
===============================
Signed-error distributions (predicted - true) for completeness and
contamination, per tool, stratified by benchmark set, MIMAG-inspired quality
class, true-contamination bin, dominant phylum, dominant domain, and
contamination-source relatedness (within-phylum vs cross-phylum).

Answers Reviewer 2's "other comment 2": MAE hides the direction of the error;
the CheckM2 study reports signed-error distributions, and so should this paper.
Mean and median signed error are reported with cluster-bootstrap 95% CIs so the
direction of any systematic bias, not just its magnitude, is testable.

    python scripts/103_signed_errors.py [--config ...] [--n-boot 2000]
                                        [--tiers primary reported secondary]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

METRICS = ("completeness", "contamination")
MIN_STRATUM_N = 5


def _describe(e: np.ndarray) -> Dict[str, float]:
    e = e[~np.isnan(e)]
    if e.size == 0:
        return {}
    q = np.percentile(e, [5, 25, 50, 75, 95])
    return {
        "n": int(e.size),
        "mean_signed_error": float(e.mean()),
        "median_signed_error": float(q[2]),
        "hodges_lehmann_signed_error": fw.hodges_lehmann_paired(e) if e.size <= 4000 else float("nan"),
        "sd_signed_error": float(e.std(ddof=1)) if e.size > 1 else float("nan"),
        "iqr_signed_error": float(q[3] - q[1]),
        "q05": float(q[0]), "q25": float(q[1]), "q75": float(q[3]), "q95": float(q[4]),
        "min": float(e.min()), "max": float(e.max()),
        "mae": float(np.abs(e).mean()),
        "frac_within_1pp": float(np.mean(np.abs(e) <= 1)),
        "frac_within_5pp": float(np.mean(np.abs(e) <= 5)),
        "frac_within_10pp": float(np.mean(np.abs(e) <= 10)),
        "frac_overestimated": float(np.mean(e > 0)),
        "frac_underestimated": float(np.mean(e < 0)),
    }


def _stratum_rows(df: pd.DataFrame, tools: List[str], bsobj, cfg,
                  n_boot: int, stratum_kind: str, stratum_value: str,
                  seed: int) -> List[dict]:
    """All (tool x metric) signed-error statistics for one stratum of one set."""
    if len(df) == 0:
        return []
    bs = fw.Bootstrapper(clusters=df["cluster_id"].to_numpy(), n_iter=n_boot,
                         ci_level=cfg.ci_level, seed=seed)
    bs.resample_indices()
    out = []
    for tool in tools:
        for metric in METRICS:
            col = f"err_{metric}__{tool}"
            if col not in df.columns:
                continue
            e = df[col].to_numpy(float)
            d = _describe(e)
            if not d:
                continue
            boot = bs.ci_multi(lambda idx, _e=e: {
                "mean": float(np.nanmean(_e[idx])),
                "median": float(np.nanmedian(_e[idx]))})
            # two-sided test of zero bias
            good = e[~np.isnan(e)]
            if good.size >= 6 and np.any(good != 0):
                p_wil = float(sps.wilcoxon(good, alternative="two-sided",
                                           zero_method="wilcox").pvalue)
            else:
                p_wil = float("nan")
            p_boot = bs.p_two_sided(lambda idx, _e=e: float(np.nanmean(_e[idx])))[0]
            rec = {
                "set": bsobj.name, "set_label": bsobj.label, "status": bsobj.status,
                "tier": bsobj.tier, "stratum_kind": stratum_kind,
                "stratum": stratum_value, "tool": tool,
                "tool_label": cfg.tool_label(tool), "tool_role": cfg.tool_role(tool),
                "metric": metric, "n_clusters": bs.n_clusters,
                "small_n_flag": bool(d["n"] < 10),
            }
            rec.update(d)
            rec.update({
                "mean_ci_lo": boot["mean"]["ci_lo"], "mean_ci_hi": boot["mean"]["ci_hi"],
                "median_ci_lo": boot["median"]["ci_lo"],
                "median_ci_hi": boot["median"]["ci_hi"],
                "bias_direction": ("over" if d["mean_signed_error"] > 0 else
                                   "under" if d["mean_signed_error"] < 0 else "none"),
                "mean_ci_excludes_zero": bool(
                    not np.isnan(boot["mean"]["ci_lo"]) and
                    (boot["mean"]["ci_lo"] > 0 or boot["mean"]["ci_hi"] < 0)),
                "p_wilcoxon_zero_bias_two_sided": p_wil,
                "p_cluster_bootstrap_zero_mean_bias": p_boot,
            })
            out.append(rec)
    return out


def process_set(cfg, bsobj, n_boot: int) -> Tuple[List[dict], pd.DataFrame]:
    df, tools, _ = fw.load_set(cfg, bsobj)
    if not tools:
        return [], pd.DataFrame()
    df = df.copy()
    df["phylum_group"] = fw.phylum_group(df["dominant_phylum"],
                                         cfg.raw["strata"]["min_phylum_n"])
    seed0 = cfg.seed + (fw.stable_hash(bsobj.name) % 100000)
    rows: List[dict] = []
    rows += _stratum_rows(df, tools, bsobj, cfg, n_boot, "overall", "all", seed0)
    strata = [("mimag_class_true", "true_mimag"),
              ("true_contamination_bin", "true_cont_bin"),
              ("dominant_phylum", "phylum_group"),
              ("dominant_domain", "dominant_domain"),
              ("contamination_relatedness", "relatedness")]
    for kind, col in strata:
        for i, (val, g) in enumerate(df.groupby(col, sort=True, observed=True)):
            if len(g) < MIN_STRATUM_N:
                # still report the descriptive statistics, without a bootstrap CI
                for tool in tools:
                    for metric in METRICS:
                        e = g[f"err_{metric}__{tool}"].to_numpy(float)
                        d = _describe(e)
                        if not d:
                            continue
                        rows.append({"set": bsobj.name, "set_label": bsobj.label,
                                     "status": bsobj.status, "tier": bsobj.tier,
                                     "stratum_kind": kind, "stratum": str(val),
                                     "tool": tool, "tool_label": cfg.tool_label(tool),
                                     "tool_role": cfg.tool_role(tool), "metric": metric,
                                     "n_clusters": int(g.cluster_id.nunique()),
                                     "small_n_flag": True,
                                     "note": f"n < {MIN_STRATUM_N}: no bootstrap CI",
                                     **d})
                continue
            rows += _stratum_rows(g, tools, bsobj, cfg, n_boot, kind, str(val),
                                  seed0 + 1000 * (fw.stable_hash(kind) % 97) + i)

    # long-format per-genome signed errors (Source Data / plotting)
    keep = ["genome_id", "dominant_accession", "dominant_phylum", "dominant_domain",
            "dominant_split", "relatedness", "true_mimag", "true_cont_bin",
            "true_completeness", "true_contamination", "cluster_id"]
    keep = [c for c in keep if c in df.columns]
    long = []
    for tool in tools:
        for metric in METRICS:
            sub = df[keep].copy()
            sub["set"] = bsobj.name
            sub["status"] = bsobj.status
            sub["tier"] = bsobj.tier
            sub["tool"] = tool
            sub["metric"] = metric
            sub["predicted"] = df[f"pred_{metric}__{tool}"].to_numpy()
            sub["true"] = df[f"true_{metric}"].to_numpy()
            sub["signed_error"] = df[f"err_{metric}__{tool}"].to_numpy()
            long.append(sub)
    return rows, pd.concat(long, ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--tiers", nargs="*", default=["primary", "reported", "secondary"])
    ap.add_argument("--no-long", action="store_true",
                    help="skip the per-genome long-format export")
    args = ap.parse_args(argv)

    cfg = fw.load_config(args.config)
    n_boot = args.n_boot or cfg.n_boot
    sets = fw.discover_sets(cfg, include_missing=False, tiers=args.tiers)
    all_rows, longs = [], []
    for b in sets:
        print(f"  [103] {b.name} ...", flush=True)
        r, lg = process_set(cfg, b, n_boot)
        if not r:
            print(f"  [103] {b.name}: no predictions, skipped")
            continue
        all_rows += r
        if not args.no_long:
            longs.append(lg)

    d = pd.DataFrame(all_rows)
    order = {b.name: i for i, b in enumerate(sets)}
    torder = {t: i for i, t in enumerate(cfg.tools)}
    d["_s"] = d["set"].map(order)
    d["_t"] = d["tool"].map(torder)
    d = d.sort_values(["_s", "stratum_kind", "stratum", "metric", "_t"]).drop(
        columns=["_s", "_t"])

    o = cfg.out_dir
    front = ["set", "set_label", "status", "tier", "stratum_kind", "stratum",
             "tool", "tool_label", "tool_role", "metric", "n", "n_clusters",
             "small_n_flag", "mean_signed_error", "mean_ci_lo", "mean_ci_hi",
             "median_signed_error", "median_ci_lo", "median_ci_hi",
             "hodges_lehmann_signed_error", "sd_signed_error", "iqr_signed_error",
             "q05", "q25", "q75", "q95", "min", "max", "mae",
             "frac_within_1pp", "frac_within_5pp", "frac_within_10pp",
             "frac_overestimated", "frac_underestimated", "bias_direction",
             "mean_ci_excludes_zero", "p_wilcoxon_zero_bias_two_sided",
             "p_cluster_bootstrap_zero_mean_bias"]
    d = d.reindex(columns=front + [c for c in d.columns if c not in front])
    d.to_csv(o / "ws5.3_signed_errors_all_strata.tsv", sep="\t", index=False)
    for kind, fname in [("overall", "ws5.3_signed_errors_overall.tsv"),
                        ("mimag_class_true", "ws5.3_signed_errors_by_mimag.tsv"),
                        ("true_contamination_bin", "ws5.3_signed_errors_by_cont_bin.tsv"),
                        ("dominant_phylum", "ws5.3_signed_errors_by_phylum.tsv"),
                        ("dominant_domain", "ws5.3_signed_errors_by_domain.tsv"),
                        ("contamination_relatedness", "ws5.3_signed_errors_by_relatedness.tsv")]:
        d[d.stratum_kind == kind].to_csv(o / fname, sep="\t", index=False)
    if longs:
        lg = pd.concat(longs, ignore_index=True)
        lg.to_csv(o / "ws5.3_signed_errors_per_genome.tsv.gz", sep="\t",
                  index=False, compression="gzip")
        print(f"  [103] per-genome long file: {len(lg):,} rows")

    write_report(cfg, d, sets, n_boot)
    print(f"[103] wrote ws5.3_* to {o}")
    return 0


def _ci(v, lo, hi, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return f"{v:+.{nd}f}"
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def write_report(cfg, d, sets, n_boot):
    den = fw.caption_denominator(cfg)
    L = ["# WS5.3 - Signed error distributions (predicted - true)", "",
         "Reviewer 2 (other comment 2): MAE quantifies the magnitude of the "
         "deviation but not its direction. Positive signed error = the tool "
         "OVERESTIMATES; negative = the tool UNDERESTIMATES. All values are in "
         "percentage points (pp).", "",
         den, "",
         f"Mean and median signed errors carry percentile bootstrap 95% CIs "
         f"({n_boot} replicates) with clusters resampled by dominant reference "
         f"genome. `p_wilcoxon_zero_bias_two_sided` is a two-sided Wilcoxon "
         f"signed-rank test of H0: median signed error = 0 that treats genomes as "
         f"independent; `p_cluster_bootstrap_zero_mean_bias` is the "
         f"cluster-bootstrap two-sided p-value for H0: mean signed error = 0 and "
         f"is the one to quote when several genomes share a reference genome.", "",
         "Strata reported in the TSV outputs: `overall`, `mimag_class_true`, "
         "`true_contamination_bin`, `dominant_phylum`, `dominant_domain`, "
         "`contamination_relatedness`.", ""]
    ov = d[(d.stratum_kind == "overall")]
    for b in sets:
        g = ov[ov["set"] == b.name]
        if g.empty:
            continue
        flag = "  **[SUPERSEDED - training-set leakage]**" if b.status == "superseded_leaky" else ""
        L += [f"## {b.name} - {b.label}{flag}", ""]
        for metric in METRICS:
            gm = g[g.metric == metric]
            if gm.empty:
                continue
            t = pd.DataFrame({
                "tool": gm.tool_label, "n": gm.n.astype(int),
                "mean signed error [95% CI]": [_ci(*r) for r in zip(
                    gm.mean_signed_error, gm.mean_ci_lo, gm.mean_ci_hi)],
                "median [95% CI]": [_ci(*r) for r in zip(
                    gm.median_signed_error, gm.median_ci_lo, gm.median_ci_hi)],
                "SD": gm.sd_signed_error.round(2), "IQR": gm.iqr_signed_error.round(2),
                "q05..q95": [f"{a:+.1f} .. {b2:+.1f}" for a, b2 in zip(gm.q05, gm.q95)],
                "% over": (100 * gm.frac_overestimated).round(1),
                "% within 5pp": (100 * gm.frac_within_5pp).round(1),
                "direction": gm.bias_direction,
                "p (cluster boot.)": [f"{p:.2g}" for p in gm.p_cluster_bootstrap_zero_mean_bias],
            })
            L += [f"### {metric} signed error (pp)", "", fw.md_table(t), ""]

    # stratified highlights for the primary tool
    ref = cfg.raw["statistics"]["reference_tool"]
    L += ["## Stratified signed error, all tools, primary and reported sets", ""]
    for kind, title in [("mimag_class_true", "by true MIMAG-inspired class"),
                        ("true_contamination_bin", "by true contamination bin"),
                        ("contamination_relatedness", "by contamination-source relatedness"),
                        ("dominant_domain", "by domain")]:
        L += [f"### Mean signed error {title} (pp; point estimates only - the "
              f"cluster-bootstrap 95% CIs for every cell are in "
              f"ws5.3_signed_errors_all_strata.tsv)", ""]
        sub = d[(d.stratum_kind == kind) & (d.tier.isin(["primary", "reported"]))]
        for metric in METRICS:
            sm = sub[sub.metric == metric]
            if sm.empty:
                continue
            piv = sm.pivot_table(index=["set", "stratum"], columns="tool",
                                 values="mean_signed_error", aggfunc="first")
            nn = sm.pivot_table(index=["set", "stratum"], columns="tool",
                                values="n", aggfunc="first")
            piv = piv.reindex(columns=[t for t in cfg.tools if t in piv.columns])
            piv = piv.round(2).reset_index()
            piv.insert(2, "n", [int(nn.loc[(s, st)].max()) for s, st in
                                zip(piv["set"], piv["stratum"])])
            L += [f"**{metric}**", "", fw.md_table(piv), ""]
    (cfg.out_dir / "ws5.3_signed_errors.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
