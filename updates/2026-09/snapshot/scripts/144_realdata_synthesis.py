#!/usr/bin/env python3
"""
WS3.9 — cross-dataset synthesis for the three real-data cohorts with genuine ground
truth (Meslier MOCK1, Zymo isolates, NCBI Tier-1 same-BioSample pairs).

Emits one synthesis table plus the figures, all captions carrying the denominator
convention explicitly (R1-M5). Colours use the project's CVD-safe Okabe-Ito subset from
scripts/config_revision_metrics.yaml; every colour is paired with a redundant marker
shape, and no red/green discrimination is required (editorial E4).

Outputs (results/revision/real_data/):
    synthesis_table.tsv          real-data performance for every tool, one view
    synthesis_table.md           the same, formatted
    figures/fig_ws3_realdata_synthesis.{png,pdf}
    figures/fig_ws3_fragmentation_gradient.{png,pdf}
    figures/fig_ws3_signed_errors.{png,pdf}

Usage: python scripts/144_realdata_synthesis.py
"""
from __future__ import annotations

import importlib.util
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/media/Data_1/tianrm/projects/magicc2")
RD = ROOT / "results" / "revision" / "real_data"
FIG = RD / "figures"

_spec = importlib.util.spec_from_file_location(
    "mf", ROOT / "scripts" / "101_metrics_framework.py")
mf = importlib.util.module_from_spec(_spec)
sys.modules["mf"] = mf
_spec.loader.exec_module(mf)

COLOUR = {"MAGICC V5": "#0072B2", "CheckM2 1.0.1": "#D55E00",
          "CoCoPyE 0.5.0": "#F0E442", "DeepCheck": "#000000"}
MARKER = {"MAGICC V5": "o", "CheckM2 1.0.1": "^", "CoCoPyE 0.5.0": "D",
          "DeepCheck": "v"}
EDGE = {"CoCoPyE 0.5.0": "#7f7000"}
TOOLS = ["MAGICC V5", "CheckM2 1.0.1", "CoCoPyE 0.5.0", "DeepCheck"]

DENOM_CAPTION = (
    "Denominators (identical for ground truth and for every tool): completeness = "
    "retained dominant-organism bp / dominant reference FULL length x 100; "
    "contamination = contaminant bp / dominant reference FULL length x 100. "
    "MIMAG-inspired thresholds (completeness/contamination only): high quality "
    ">=90 % completeness AND <5 % contamination; medium quality >=50 % AND <10 %. "
    "R2 is the coefficient of determination (1 - SS_res/SS_tot), never squared Pearson.")

PRIMARY = {
    "meslier": ("primary_leakage_free_comp>=50_ORF_intact",
                "Meslier MOCK1 (leakage-free, >=50 %, ORF-intact assemblies)"),
    "zymo": ("zymo_8_isolates", "Zymo isolate drafts (n=8)"),
    "ncbi_pairs": ("primary_leakage_free", "NCBI Tier-1 pairs (leakage-free)"),
}
SECONDARY = {
    "meslier": [("primary_leakage_free_comp>=50",
                 "Meslier MOCK1 (leakage-free, >=50 %, all 7 assemblies)"),
                ("ORF_compromised_assembly_only",
                 "Meslier MOCK1 (indel-dense MinION assembly only)"),
                ("all_MOCK1_comp>=50",
                 "Meslier MOCK1 (all organisms, >=50 %, all 7 assemblies)")],
    "ncbi_pairs": [("all_tier1_pairs", "NCBI Tier-1 pairs (all)")],
}


def load(track, fname):
    p = RD / track / fname
    return pd.read_csv(p, sep="\t") if p.exists() else None


def build_synthesis():
    rows = []
    for track, groups in (("meslier", [PRIMARY["meslier"]] + SECONDARY["meslier"]),
                          ("zymo", [PRIMARY["zymo"]]),
                          ("ncbi_pairs", [PRIMARY["ncbi_pairs"]]
                                         + SECONDARY["ncbi_pairs"])):
        m = load(track, "metrics_by_cohort.tsv")
        mi = load(track, "mimag_by_cohort.tsv")
        if m is None:
            continue
        for cohort, label in groups:
            sub = m[m.cohort == cohort]
            for _, r in sub.iterrows():
                d = {"dataset": label, "track": track, "cohort": cohort,
                     "tool": r["tool"], "n": r["n"], "n_clusters": r["n_clusters"],
                     "cluster_unit": r["cluster_unit"],
                     "true_comp_mean": r["true_comp_mean"],
                     "true_cont_mean": r["true_cont_mean"],
                     "comp_MAE": r["comp_mae"],
                     "comp_MAE_95CI": f"[{r['comp_mae_ci_lo']}, {r['comp_mae_ci_hi']}]",
                     "comp_bias": r["comp_bias"],
                     "comp_bias_95CI": f"[{r['comp_bias_ci_lo']}, "
                                       f"{r['comp_bias_ci_hi']}]",
                     "comp_R2_CoD": (r["comp_r2"]
                                     if r.get("comp_r2_interpretable", True)
                                     else "n/a (SS_tot~0)"),
                     "cont_MAE": r["cont_mae"],
                     "cont_MAE_95CI": f"[{r['cont_mae_ci_lo']}, {r['cont_mae_ci_hi']}]",
                     "cont_bias": r["cont_bias"],
                     "cont_bias_95CI": f"[{r['cont_bias_ci_lo']}, "
                                       f"{r['cont_bias_ci_hi']}]",
                     "cont_R2_CoD": (r["cont_r2"]
                                     if r.get("cont_r2_interpretable", True)
                                     else "n/a (SS_tot~0)"),
                     "true_comp_sd": r.get("true_comp_sd"),
                     "true_cont_sd": r.get("true_cont_sd")}
                if mi is not None:
                    q = mi[(mi.cohort == cohort) & (mi.tool == r["tool"])]
                    if len(q):
                        q = q.iloc[0]
                        d.update({"true_HQ": q["true_HQ"], "pred_HQ": q["pred_HQ"],
                                  "MIMAG_class_agreement_pct":
                                      q["class_agreement_pct"],
                                  "false_fail_5pct_rate": q["false_fail_5pct_rate"]})
                rows.append(d)
    return pd.DataFrame(rows)


def fig_fragmentation(plt):
    g = load("meslier", "fragmentation_gradient.tsv")
    if g is None:
        return None
    qc = load("meslier", "assembly_sequence_qc.tsv")
    bad = (set(qc.loc[qc.orf_integrity_compromised, "assembly"])
           if qc is not None else set())
    panels = ["balanced_panel_comp>=50_leakage_free", "balanced_panel_comp>=50"]
    panels = [p for p in panels if p in set(g.panel)]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.7 * len(panels), 5.4),
                             sharex=True)
    axes = np.atleast_2d(axes)
    if len(panels) == 1:
        axes = axes.reshape(2, 1)
    for j, pan in enumerate(panels):
        gg = g[g.panel == pan]
        npan = int(gg["n_panel_organisms"].iloc[0])
        for i, metric, ylab in ((0, "comp_mae", "Completeness MAE (pp)"),
                                (1, "cont_mae", "Contamination MAE (pp)")):
            ax = axes[i, j]
            for t in TOOLS:
                s = gg[gg.tool == t].sort_values("assembly_median_bin_n50")
                if not len(s):
                    continue
                x = s["assembly_median_bin_n50"].values / 1000.0
                # solid line through the ORF-intact assemblies only; the indel-dense
                # assembly is a point on a different axis and is drawn detached
                keep = ~s["assembly"].isin(bad).values
                ax.plot(x[keep], s[metric].values[keep], marker=MARKER[t],
                        color=COLOUR[t], lw=1.3, ms=5,
                        markeredgecolor=EDGE.get(t, COLOUR[t]),
                        markeredgewidth=0.8, label=t)
                ax.fill_between(x[keep], s[f"{metric}_ci_lo"].values[keep],
                                s[f"{metric}_ci_hi"].values[keep],
                                color=COLOUR[t], alpha=0.12, lw=0)
                if (~keep).any():
                    ax.scatter(x[~keep], s[metric].values[~keep], marker=MARKER[t],
                               s=70, facecolors="none", edgecolors=COLOUR[t],
                               linewidths=1.6, zorder=5)
            ax.set_xscale("log")
            ax.set_ylabel(ylab if j == 0 else "")
            if i == 0:
                ax.set_title(f"{pan.replace('balanced_panel_', '')}\n"
                             f"({npan} organisms recovered in all 7 assemblies)",
                             fontsize=8)
            if i == 1:
                ax.set_xlabel("Median bin N50 (kb, log scale)\n"
                              "more fragmented  ←            →  less fragmented")
            for a in sorted(bad):
                xa = gg[gg.assembly == a]["assembly_median_bin_n50"]
                if len(xa):
                    ax.axvline(float(xa.iloc[0]) / 1000.0, ls=":", lw=0.9,
                               c="#888888", zorder=0)
    axes[0, 0].legend(frameon=False, loc="upper left")
    if bad:
        axes[0, -1].annotate(
            f"open markers / dotted line:\n{', '.join(sorted(bad))} — indel-dense,\n"
            "ORFs frameshifted (see assembly_sequence_qc.tsv)",
            xy=(0.97, 0.70), xycoords="axes fraction", ha="right", va="top",
            fontsize=6.5, color="#444444")
    return fig


def fig_scatter(plt):
    p = load("meslier", "per_bin_results.tsv")
    if p is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    sub = p[p.true_completeness >= 50]
    for ax, (metric, lab, lim) in zip(
            axes, (("completeness", "completeness (%)", (40, 105)),
                   ("contamination", "contamination (%)", (-0.5, 25)))):
        for t, key in (("MAGICC V5", "magicc"), ("CheckM2 1.0.1", "checkm2")):
            c = f"{key}_{metric}"
            if c not in sub.columns:
                continue
            ax.scatter(sub[f"true_{metric}"], sub[c], s=13, alpha=0.6,
                       marker=MARKER[t], c=COLOUR[t],
                       edgecolors=EDGE.get(t, "none"), linewidths=0.3, label=t)
        ax.plot(lim, lim, ls="--", lw=0.8, c="#666666", zorder=0)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel(f"True {lab}")
        ax.set_ylabel(f"Predicted {lab}")
        ax.set_title(f"Meslier MOCK1 bins, true completeness >=50 % (n={len(sub)})",
                     fontsize=8)
    axes[0].legend(frameon=False, loc="lower right")
    return fig


def fig_signed(plt):
    frames = []
    for track, (cohort, label) in PRIMARY.items():
        f = load(track, "per_bin_results.tsv")
        if f is None:
            f = load(track, "per_pair_results.tsv")
        if f is None:
            continue
        if track == "meslier":
            qc = load(track, "assembly_sequence_qc.tsv")
            bad = (set(qc.loc[qc.orf_integrity_compromised, "assembly"])
                   if qc is not None else set())
            f = f[(f.leakage_free) & (f.true_completeness >= 50)
                  & (~f.assembly.isin(bad))]
        elif track == "ncbi_pairs":
            f = f[f.leakage_free]
            f = f.rename(columns={"true_contamination_upper": "true_contamination"})
        frames.append((label, f))
    if not frames:
        return None
    fig, axes = plt.subplots(2, len(frames), figsize=(3.1 * len(frames), 5.0))
    axes = np.atleast_2d(axes)
    if len(frames) == 1:
        axes = axes.reshape(2, 1)
    for j, (label, f) in enumerate(frames):
        for i, metric, ylab in ((0, "completeness", "Signed completeness error (pp)"),
                                (1, "contamination", "Signed contamination error (pp)")):
            ax = axes[i, j]
            data, labs, cols = [], [], []
            for t, key in zip(TOOLS, ["magicc", "checkm2", "cocopye", "deepcheck"]):
                c = f"{key}_{metric}"
                if c not in f.columns or f[c].isna().all():
                    continue
                e = (f[c] - f[f"true_{metric}"]).dropna().values
                data.append(e)
                labs.append(t.split()[0])
                cols.append(COLOUR[t])
            if not data:
                continue
            bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                            medianprops=dict(color="white", lw=1.2))
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
                patch.set_edgecolor("#333333")
                patch.set_linewidth(0.7)
            for k, (e, c) in enumerate(zip(data, cols), start=1):
                if len(e) <= 60:
                    x = np.random.default_rng(7).normal(k, 0.06, size=len(e))
                    ax.scatter(x, e, s=6, c="#333333", alpha=0.5, zorder=3)
            ax.axhline(0, ls="--", lw=0.8, c="#666666", zorder=0)
            ax.set_xticks(range(1, len(labs) + 1))
            ax.set_xticklabels(labs, rotation=30, ha="right")
            ax.set_ylabel(ylab if j == 0 else "")
            if i == 0:
                ax.set_title(f"{label}\n(n={len(f)})", fontsize=8)
    return fig


def fig_synthesis(plt, syn):
    prim = [lbl for _, lbl in PRIMARY.values()]
    s = syn[syn.dataset.isin(prim)]
    if not len(s):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    ds = [d for d in prim if d in set(s.dataset)]
    ypos = {d: i for i, d in enumerate(ds)}
    for ax, metric, lab in ((axes[0], "comp", "Completeness MAE (pp)"),
                            (axes[1], "cont", "Contamination MAE (pp)")):
        for k, t in enumerate(TOOLS):
            sub = s[s.tool == t]
            if not len(sub):
                continue
            y = [ypos[d] + (k - 1.5) * 0.17 for d in sub.dataset]
            x = sub[f"{metric}_MAE"].values
            ci = sub[f"{metric}_MAE_95CI"].apply(
                lambda v: [float(z) for z in v.strip("[]").split(",")])
            lo = np.array([c[0] for c in ci])
            hi = np.array([c[1] for c in ci])
            ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt=MARKER[t], ms=5.5,
                        color=COLOUR[t], ecolor=COLOUR[t], elinewidth=1.1,
                        capsize=2, lw=0, markeredgecolor=EDGE.get(t, "#333333"),
                        markeredgewidth=0.6, label=t)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels([d.replace(" (", "\n(") for d in ds], fontsize=7)
        ax.set_xlabel(lab)
        ax.grid(axis="x", ls=":", lw=0.5, alpha=0.5)
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle("Real-data accuracy with genuine ground truth "
                 "(primary leakage-free cohorts, 95 % cluster-bootstrap CI)",
                 fontsize=9)
    return fig


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    syn = build_synthesis()
    syn.to_csv(RD / "synthesis_table.tsv", sep="\t", index=False)
    print(f"wrote {RD/'synthesis_table.tsv'} ({len(syn)} rows)")

    cols = ["dataset", "tool", "n", "n_clusters", "comp_MAE", "comp_MAE_95CI",
            "comp_bias", "comp_R2_CoD", "cont_MAE", "cont_MAE_95CI", "cont_bias",
            "cont_R2_CoD"]
    md = ["# WS3 Track A — real-data performance with genuine ground truth\n",
          f"_Generated {datetime.now(timezone.utc).isoformat()}_\n",
          DENOM_CAPTION + "\n", mf.md_table(syn[cols]), ""]
    (RD / "synthesis_table.md").write_text("\n".join(md))
    print(f"wrote {RD/'synthesis_table.md'}")

    plt = mf.init_matplotlib()
    caps = []
    for name, fn, cap in (
            ("fig_ws3_fragmentation_gradient", fig_fragmentation,
             "MOCK1 fragmentation gradient. Seven independent assemblies of the SAME "
             "real mock community (Meslier et al. 2022) spanning median bin N50 "
             "4.0 kb - 2.01 Mb, evaluated on a balanced panel of organisms recovered in "
             "every assembly. Lines are per-tool MAE; shading is the 95 % "
             "cluster-bootstrap CI (clusters = reference organisms). " + DENOM_CAPTION),
            ("fig_ws3_meslier_scatter", fig_scatter,
             "Predicted vs true completeness and contamination for Meslier MOCK1 "
             "reference-anchored bins with true completeness >=50 %. Dashed line is "
             "identity. " + DENOM_CAPTION),
            ("fig_ws3_signed_errors", fig_signed,
             "Signed error distributions (predicted - true) for the primary "
             "leakage-free cohort of each real-data dataset. Boxes show median and "
             "IQR, whiskers 1.5x IQR; individual points are overlaid where n<=60 "
             "(editorial policy E9). " + DENOM_CAPTION),
            ("fig_ws3_realdata_synthesis", lambda p: fig_synthesis(p, syn),
             "Cross-dataset synthesis of real-data accuracy. Points are MAE with 95 % "
             "cluster-bootstrap CI. " + DENOM_CAPTION)):
        try:
            fig = fn(plt)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED — {type(e).__name__}: {e}")
            continue
        if fig is None:
            print(f"  {name}: skipped (inputs absent)")
            continue
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        caps.append(f"### {name}\n\n{cap}\n")
        print(f"  wrote {FIG/name}.png/.pdf")
    (FIG / "captions.md").write_text("\n".join(caps))

    with open(RD / "synthesis_summary.json", "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "n_rows": len(syn), "datasets": sorted(syn.dataset.unique()),
                   "denominator_caption": DENOM_CAPTION}, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
