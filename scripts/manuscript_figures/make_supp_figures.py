#!/usr/bin/env python3
"""Build supplementary figures S1-S17 for the MAGICC Nature Communications
resubmission (v2).

S1-S15 keep their previous numbering and content and are only repaletted onto
the author-mandated palette of ``figstyle`` -- the ORIGINAL submission's
four-tool set restored by the internal build contract 9.4 (MAGICC #d62728 red,
CheckM2 #1f77b4 blue, CoCoPyE #2ca02c green, DeepCheck #9467bd purple).  S16 and
S17 are new composites carrying the panels that left main-text Figures 4 and 5;
they are drawn by ``panels_taxonomy`` and ``panels_realdata``, the same modules
the main figures use, so no number can diverge between the two.

Every figure is re-plotted from the underlying result TSVs so that the whole
supplementary panel shares one style (`figstyle.py`: 7 pt sans-serif, bold
lowercase panel labels, no top/right spines, the author-mandated palette with
marker shape and line style as redundant cues, pdf.fonttype 42) and one
resolution (vector PDF + 400 dpi PNG).  Series that encode something other than
a tool -- the five Set G error processes in S11e-g, the contamination types in
S16b-c -- are drawn from ``figstyle.AUX``, whose hues sit well away from all
four tool hues so that they cannot be read as a tool.

Re-runnable from scratch.  Every input is checked before use and a missing file
raises immediately.

Usage:
    PYTHONHASHSEED=0 python nature_communications/resubmission2/scripts/make_supp_figures.py
"""

from __future__ import annotations

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs  # noqa: E402
from figstyle import (AUX, PALETTE, MARKERS, EDGE, LINESTYLES, TOOL_LABEL,  # noqa: E402
                      DENOM_NOTE, add_panel_label, save_fig, hline0, diag)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

warnings.filterwarnings("ignore", message="findfont")

PROJECT = fs.PROJECT
RESULTS = fs.RESULTS
BENCH = fs.BENCH
OUT = fs.SUPP_FIG_DIR
os.makedirs(OUT, exist_ok=True)

DIVERGING_CMAP = LinearSegmentedColormap.from_list("magicc_div", fs.DIVERGING)

# Tool ordering used everywhere (MAGICC first, then the three comparators).
TOOLS = ["magicc", "checkm2", "cocopye", "deepcheck"]

# file-name stem for each tool inside data/benchmarks/<set>/
PRED_FILE = {"magicc": "magicc_v5_predictions.tsv",
             "checkm2": "checkm2_predictions.tsv",
             "cocopye": "cocopye_predictions.tsv",
             "deepcheck": "deepcheck_predictions.tsv"}

# tool key used inside results/revision/**/*.tsv
TSV_TOOL = {"magicc": "magicc_v5", "checkm2": "checkm2",
            "cocopye": "cocopye", "deepcheck": "deepcheck"}
CAMI_TOOL = {"magicc": "MAGICC_V5", "checkm2": "CheckM2",
             "cocopye": "CoCoPyE", "deepcheck": "DeepCheck"}

# The five leakage-free benchmark sets (PLAN.md).  n and n_clusters are read
# back from results/revision/metrics/definitive_table_5set.tsv at run time.
SETS = [
    ("A", "set_A_v2", "Set A (completeness gradient)"),
    ("B", "set_B_v2", "Set B (contamination gradient)"),
    ("C", "set_C_clean", "Set C-clean (Patescibacteriota)"),
    ("D", "set_D_clean", "Set D-clean (Archaea)"),
    ("E", "set_E", "Set E (realistic mixture)"),
]

CAPTIONS: "dict[str, str]" = {}


# ---------------------------------------------------------------------------
# I/O helpers -- fail loudly
# ---------------------------------------------------------------------------
def require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"MISSING REQUIRED INPUT: {path}")
    return path


def read_tsv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(require(path), sep="\t", **kw)


def caption(name: str, text: str) -> None:
    """Register a caption; the denominator sentence is appended automatically."""
    CAPTIONS[name] = text.strip() + " " + DENOM_NOTE


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def violin_group(ax, values_by_tool, x_centre, width=0.78, showbox=True):
    """Draw one violin+box per tool, centred on x_centre.

    values_by_tool: dict tool -> 1-D array (may be empty).
    """
    ntool = len(TOOLS)
    slot = width / ntool
    for i, tool in enumerate(TOOLS):
        v = np.asarray(values_by_tool.get(tool, []), dtype=float)
        v = v[np.isfinite(v)]
        pos = x_centre - width / 2 + slot * (i + 0.5)
        if v.size == 0:
            continue
        if v.size >= 8 and np.ptp(v) > 1e-9:
            parts = ax.violinplot([v], positions=[pos], widths=slot * 0.95,
                                  showextrema=False, showmedians=False)
            for b in parts["bodies"]:
                b.set_facecolor(PALETTE[tool])
                b.set_edgecolor(EDGE.get(tool, PALETTE[tool]))
                b.set_linewidth(0.3)
                b.set_alpha(0.45)
        if showbox:
            bp = ax.boxplot([v], positions=[pos], widths=slot * 0.34,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="white", lw=0.7),
                            whiskerprops=dict(lw=0.4, color="0.3"),
                            capprops=dict(lw=0.4, color="0.3"),
                            boxprops=dict(lw=0.3))
            for patch in bp["boxes"]:
                patch.set_facecolor(PALETTE[tool])
                patch.set_edgecolor(EDGE.get(tool, "0.25"))
        ax.plot([pos], [np.mean(v)], marker=MARKERS[tool], ms=2.4,
                mfc="white", mec="0.15", mew=0.5, ls="none", zorder=5)


def tool_legend(ax, loc="upper left", ncol=1, extra=None, **kw):
    handles = [Line2D([], [], color=PALETTE[t], marker=MARKERS[t], ls="none",
                      ms=3.2, mec=EDGE.get(t, PALETTE[t]), mew=0.5,
                      label=TOOL_LABEL[t]) for t in TOOLS]
    if extra:
        handles += list(extra)
    ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=False,
              handletextpad=0.4, borderpad=0.2, labelspacing=0.3, **kw)


def cat_axis(ax, labels, rotation=0, ha="center"):
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.set_xlim(-0.6, len(labels) - 0.4)


def line_pe(tool):
    """Kept for call-site compatibility.

    The previous round's yellow CoCoPyE needed a dark path-effect stroke to stay
    visible on white; the author-mandated palette has no such colour, so
    ``figstyle.EDGE`` is empty and this returns nothing.
    """
    if tool in EDGE:
        import matplotlib.patheffects as pe
        return [pe.Stroke(linewidth=2.0, foreground=EDGE[tool]), pe.Normal()]
    return []


def errbars(ax, x, y, lo, hi, tool, ms=3.2, **kw):
    style = dict(color=PALETTE[tool], marker=MARKERS[tool], ms=ms,
                 ls="none", lw=0.8, capsize=1.6, elinewidth=0.7,
                 mec=EDGE.get(tool, PALETTE[tool]), mew=0.5)
    style.update(kw)
    y = np.asarray(y, float)
    yerr = np.vstack([y - np.asarray(lo, float), np.asarray(hi, float) - y])
    yerr[~np.isfinite(yerr)] = 0.0
    yerr = np.clip(yerr, 0, None)
    ax.errorbar(x, y, yerr=yerr, **style)


# ---------------------------------------------------------------------------
# Shared metadata: per-set n and reference-cluster counts
# ---------------------------------------------------------------------------
def set_counts() -> "dict[str, tuple[int, int]]":
    d = read_tsv(os.path.join(RESULTS, "metrics", "definitive_table_5set.tsv"))
    d = d[(d.tier == "primary") & (d.tool == "magicc_v5")
          & (d.metric == "completeness")]
    out = {r["set"]: (int(r["n"]), int(r["n_clusters"])) for _, r in d.iterrows()}
    for _, sdir, _ in SETS:
        if sdir not in out:
            raise KeyError(f"no n/n_clusters row for {sdir} in definitive_table_5set.tsv")
    return out


SET_N = set_counts()


def load_set_predictions(sdir: str) -> "dict[str, pd.DataFrame]":
    """Load the four tools' predictions for one benchmark set.

    Every prediction file already carries `true_completeness` /
    `true_contamination`; they are re-verified against `metadata.tsv`.
    """
    base = os.path.join(BENCH, sdir)
    meta = read_tsv(os.path.join(base, "metadata.tsv"))
    out = {}
    for tool in TOOLS:
        df = read_tsv(os.path.join(base, PRED_FILE[tool]))
        need = {"genome_id", "true_completeness", "true_contamination",
                "pred_completeness", "pred_contamination"}
        missing = need - set(df.columns)
        if missing:
            raise KeyError(f"{base}/{PRED_FILE[tool]} missing columns {missing}")
        chk = df.merge(meta[["genome_id", "true_completeness", "true_contamination"]],
                       on="genome_id", suffixes=("", "_meta"), validate="one_to_one")
        for m in ("completeness", "contamination"):
            dev = np.abs(chk[f"true_{m}"] - chk[f"true_{m}_meta"]).max()
            if dev > 1e-6:
                raise ValueError(f"{sdir}/{tool}: truth disagrees with metadata "
                                 f"for {m} (max deviation {dev})")
        out[tool] = df
    return out


def mae(df: pd.DataFrame, metric: str) -> float:
    return float(np.mean(np.abs(df[f"pred_{metric}"] - df[f"true_{metric}"])))


# ---------------------------------------------------------------------------
# S1-S5  Per-set predicted vs true completeness and contamination
# ---------------------------------------------------------------------------
def fig_S1_to_S5():
    for idx, (slab, sdir, stitle) in enumerate(SETS, start=1):
        name = f"Figure_S{idx}"
        print(f"[{name}] per-set predicted vs true -- {sdir}")
        data = load_set_predictions(sdir)
        n, nclust = SET_N[sdir]

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
        fig.subplots_adjust(wspace=0.30, left=0.085, right=0.985,
                            top=0.86, bottom=0.14)

        offscale = {}
        maes = {}
        # draw the worst-behaved tools first so MAGICC stays visible on top
        draw_order = ["deepcheck", "cocopye", "checkm2", "magicc"]
        for ax, metric, label in zip(axes, ("completeness", "contamination"),
                                     ("completeness", "contamination")):
            for tool in draw_order:
                df = data[tool]
                x = df[f"true_{metric}"].to_numpy(float)
                y = df[f"pred_{metric}"].to_numpy(float)
                kw = dict(s=5.0, alpha=0.35, marker=MARKERS[tool],
                          linewidths=0.0, rasterized=True,
                          color=PALETTE[tool], label=TOOL_LABEL[tool])
                if tool in EDGE:
                    kw.update(edgecolors=EDGE[tool], linewidths=0.15)
                ax.scatter(x, y, **kw)
                out = int(np.sum((y < -5) | (y > 105)))
                offscale.setdefault(metric, {})[tool] = out
                maes.setdefault(metric, {})[tool] = mae(df, metric)
            diag(ax, 0, 100)
            ax.set_xlabel(f"True {label} (%)")
            ax.set_ylabel(f"Predicted {label} (%)")
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_title(f"{label.capitalize()} — {stitle}", fontsize=7, pad=3)

        add_panel_label(axes[0], "a", x=-0.14, y=1.10)
        add_panel_label(axes[1], "b", x=-0.14, y=1.10)

        for ax, metric in zip(axes, ("completeness", "contamination")):
            handles = [Line2D([], [], color=PALETTE[t], marker=MARKERS[t],
                              ls="none", ms=3.0, mec=EDGE.get(t, PALETTE[t]),
                              mew=0.4,
                              label=f"{TOOL_LABEL[t]}  MAE {maes[metric][t]:.2f}")
                       for t in TOOLS]
            ax.legend(handles=handles, loc="upper left", frameon=False,
                      fontsize=5.5, handletextpad=0.3, borderpad=0.15,
                      labelspacing=0.25)

        save_fig(fig, name, out_dir=OUT)

        off_txt = []
        for metric in ("completeness", "contamination"):
            for t in TOOLS:
                k = offscale[metric][t]
                if k:
                    off_txt.append(f"{TOOL_LABEL[t]} {k} {metric} point"
                                   f"{'s' if k != 1 else ''}")
        off_sentence = ("" if not off_txt else
                        " Predictions falling outside the plotted 0-100 % range "
                        "(unconstrained regression output, passed through "
                        "unclipped) are drawn off-scale: " + "; ".join(off_txt) + ".")
        mae_sentence = "; ".join(
            f"{TOOL_LABEL[t]} {maes['completeness'][t]:.2f} / "
            f"{maes['contamination'][t]:.2f}" for t in TOOLS)
        caption(name,
                f"**Supplementary Figure S{idx} | Predicted versus true genome "
                f"quality on {stitle}, all four tools.** "
                f"**a**, Predicted versus true completeness. **b**, Predicted "
                f"versus true contamination. Each point is one benchmark genome "
                f"(n = {n:,} genomes drawn from {nclust:,} reference-genome "
                f"clusters); the dashed line is y = x. Each tool carries its "
                f"own marker shape as a redundant, non-colour cue, and is named "
                f"in the legend. Statistics: mean absolute error (MAE, percentage "
                f"points) computed per genome over all n = {n:,} genomes and "
                f"printed in the legend; completeness / contamination MAE = "
                f"{mae_sentence}. Set C-clean and Set D-clean are the rebuilt, "
                f"leakage-free replacements for the Sets C and D of the original "
                f"submission.{off_sentence}")


# ---------------------------------------------------------------------------
# S6  Signed-error distributions by MIMAG-inspired class, true-contamination
#     band and contaminant relatedness (pooled leakage-free panel)
# ---------------------------------------------------------------------------
LEAKFREE = [s[1] for s in SETS]
CONT_BINS = ["0 (uncontaminated)", "(0,5)", "[5,10)", "[10,20)",
             "[20,40)", "[40,100]"]
CONT_BIN_LABEL = {"0 (uncontaminated)": "0", "(0,5)": "(0,5)", "[5,10)": "[5,10)",
                  "[10,20)": "[10,20)", "[20,40)": "[20,40)",
                  "[40,100]": "[40,100]"}
RELATEDNESS = ["none (uncontaminated)", "within_phylum", "cross_phylum"]
REL_LABEL = {"none (uncontaminated)": "uncontaminated",
             "within_phylum": "within-phylum", "cross_phylum": "cross-phylum"}
MIMAG_ORDER = ["high", "medium", "low"]


def load_per_genome_signed_errors() -> pd.DataFrame:
    p = os.path.join(RESULTS, "metrics", "ws5.3_signed_errors_per_genome.tsv.gz")
    d = pd.read_csv(require(p), sep="\t")
    d = d[d.set.isin(LEAKFREE) & d.tool.isin(TSV_TOOL.values())].copy()
    inv = {v: k for k, v in TSV_TOOL.items()}
    d["tool_key"] = d["tool"].map(inv)
    if d["tool_key"].isna().any():
        raise ValueError("unmapped tool in ws5.3_signed_errors_per_genome")
    return d


def fig_S6():
    name = "Figure_S6"
    print(f"[{name}] signed-error distributions by stratum")
    d = load_per_genome_signed_errors()
    comp = d[d.metric == "completeness"]
    cont = d[d.metric == "contamination"]
    n_gen = int(comp[comp.tool_key == "magicc"].shape[0])
    n_clust = int(sum(comp[comp.set == sd].cluster_id.nunique()
                      for sd in LEAKFREE))

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.9))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.055,
                        hspace=0.42)

    # a -- completeness signed error by true MIMAG-inspired class
    ax = axes[0]
    ns_a = []
    for i, cls in enumerate(MIMAG_ORDER):
        sub = comp[comp.true_mimag == cls]
        violin_group(ax, {t: sub[sub.tool_key == t].signed_error.to_numpy()
                          for t in TOOLS}, i)
        ns_a.append(int(sub[sub.tool_key == "magicc"].shape[0]))
    hline0(ax)
    cat_axis(ax, [f"{c}\n(n = {n:,})" for c, n in zip(MIMAG_ORDER, ns_a)])
    ax.set_ylabel("Signed completeness error (pp)")
    ax.set_xlabel("True MIMAG-inspired quality class")
    ax.set_ylim(-45, 45)
    add_panel_label(ax, "a", x=-0.065, y=1.14)
    tool_legend(ax, loc="lower left", ncol=4, fontsize=6)

    # b -- contamination signed error by true-contamination band
    ax = axes[1]
    ns_b = []
    for i, b in enumerate(CONT_BINS):
        sub = cont[cont.true_cont_bin == b]
        violin_group(ax, {t: sub[sub.tool_key == t].signed_error.to_numpy()
                          for t in TOOLS}, i)
        ns_b.append(int(sub[sub.tool_key == "magicc"].shape[0]))
    hline0(ax)
    cat_axis(ax, [f"{CONT_BIN_LABEL[b]}\n(n = {n:,})"
                  for b, n in zip(CONT_BINS, ns_b)])
    ax.set_ylabel("Signed contamination error (pp)")
    ax.set_xlabel("True contamination band (%)")
    ax.set_ylim(-85, 45)
    add_panel_label(ax, "b", x=-0.065, y=1.14)
    tool_legend(ax, loc="lower left", ncol=4, fontsize=6)

    # c -- contamination signed error by contaminant relatedness
    ax = axes[2]
    ns_c = []
    for i, r in enumerate(RELATEDNESS):
        sub = cont[cont.relatedness == r]
        violin_group(ax, {t: sub[sub.tool_key == t].signed_error.to_numpy()
                          for t in TOOLS}, i)
        ns_c.append(int(sub[sub.tool_key == "magicc"].shape[0]))
    hline0(ax)
    cat_axis(ax, [f"{REL_LABEL[r]}\n(n = {n:,})"
                  for r, n in zip(RELATEDNESS, ns_c)])
    ax.set_ylabel("Signed contamination error (pp)")
    ax.set_xlabel("Relatedness of the contaminant to the dominant genome")
    ax.set_ylim(-85, 45)
    add_panel_label(ax, "c", x=-0.065, y=1.14)
    tool_legend(ax, loc="lower left", ncol=4, fontsize=6)

    save_fig(fig, name, out_dir=OUT)

    caption(name,
            "**Supplementary Figure S6 | Signed-error distributions stratified "
            "by MIMAG-inspired quality class, true-contamination band and "
            "contaminant relatedness.** Signed error is predicted minus true, in "
            "percentage points (pp); values above the dashed zero line are "
            "overestimates. **a**, Completeness signed error by the genome's "
            "*true* MIMAG-inspired class (high, >=90 % completeness and <5 % "
            "contamination; medium, >=50 % and <10 %; low, otherwise). "
            "**b**, Contamination signed error by true-contamination band. "
            "**c**, Contamination signed error by the taxonomic relatedness of "
            "the contaminant source to the dominant genome. Data are pooled over "
            f"the five leakage-free benchmark sets (Set A, Set B, Set C-clean, "
            f"Set D-clean, Set E; n = {n_gen:,} genomes per tool, "
            f"{n_clust:,} reference-genome clusters). Statistics: each element "
            "is a distribution, not a summary bar - the shaded shape is a kernel "
            "density estimate, the box gives the median and interquartile range "
            "with 1.5x IQR whiskers, and the open marker is the arithmetic mean; "
            "per-stratum n (per tool) is printed on the axis. Underlying "
            "per-genome values and their cluster-bootstrap confidence intervals "
            "are in results/revision/metrics/ws5.3_signed_errors_*.tsv.")


# ---------------------------------------------------------------------------
# S7  Computational performance (WS8)
# ---------------------------------------------------------------------------
SPEED_TOOL = {"magicc": "magicc", "checkm2": "checkm2",
              "cocopye": "cocopye", "deepcheck": "deepcheck"}
SPEED_LABEL = dict(TOOL_LABEL)
SPEED_LABEL["deepcheck"] = "DeepCheck (inference only)"


def fig_S7():
    import json
    name = "Figure_S7"
    print(f"[{name}] computational performance")
    sp = os.path.join(RESULTS, "speed")
    summ = read_tsv(os.path.join(sp, "matched_thread_summary.tsv"))
    runs = read_tsv(os.path.join(sp, "matched_thread_runs.tsv"))
    scal = read_tsv(os.path.join(sp, "scaling_efficiency.tsv"))
    with open(require(os.path.join(sp, "reconciliation_40s_vs_97.5s.json"))) as fh:
        rec = json.load(fh)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.83, bottom=0.24,
                        wspace=0.42)

    # a -- wall clock vs threads, matched 100-genome cell, warm cache
    ax = axes[0]
    for tool in TOOLS:
        s = summ[(summ.tool == SPEED_TOOL[tool]) & (summ.input_set == "set_E_100")
                 & (summ.cache == "warm")].sort_values("threads")
        if s.empty:
            raise ValueError(f"no warm set_E_100 rows for {tool}")
        ax.plot(s.threads, s.wall_median_s, color=PALETTE[tool],
                marker=MARKERS[tool], ms=3.2, lw=0.9,
                mec=EDGE.get(tool, PALETTE[tool]), mew=0.5,
                label=SPEED_LABEL[tool])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 8, 16, 32])
    ax.set_xticklabels(["1", "8", "16", "32"])
    ax.set_ylim(top=6e4)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Wall clock (s), 100 genomes")
    ax.set_title("Matched-thread wall clock", fontsize=7, pad=3)
    ax.legend(loc="upper right", frameon=False, fontsize=5.4,
              handletextpad=0.35, labelspacing=0.25, borderpad=0.15)
    add_panel_label(ax, "a", x=-0.24, y=1.16)

    # b -- peak RSS, every repeat plotted, 32-thread matched cell
    ax = axes[1]
    rss_note = {}
    for i, tool in enumerate(TOOLS):
        for j, (iset, lab) in enumerate((("set_E_100", "100"),
                                         ("set_E_full", "1,000"))):
            r = runs[(runs.tool == SPEED_TOOL[tool]) & (runs.input_set == iset)
                     & (runs.cache == "warm") & (runs.threads == 32)]
            if r.empty:
                continue
            pos = i + (j - 0.5) * 0.34
            y = r.peak_rss_gb.to_numpy(float)
            ax.plot(np.full(y.size, pos), y, ls="none", marker=MARKERS[tool],
                    ms=3.2, mfc=PALETTE[tool] if j else "none",
                    mec=EDGE.get(tool, "0.2") if j
                    else EDGE.get(tool, PALETTE[tool]),
                    mew=0.7)
            rss_note[(tool, lab)] = float(np.median(y))
    ax.set_yscale("log")
    cat_axis(ax, [TOOL_LABEL[t] for t in TOOLS], rotation=30, ha="right")
    ax.set_ylabel("Peak RSS (GB)")
    ax.set_title("Peak memory, 32 threads", fontsize=7, pad=3)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.2,
                              mfc="none", mec="0.35", mew=0.7,
                              label="100 genomes"),
                       Line2D([], [], marker="o", ls="none", ms=3.2,
                              mfc="0.35", mec="0.2", label="1,000 genomes")],
              loc="lower right", frameon=False, fontsize=5.4,
              handletextpad=0.35, labelspacing=0.25, borderpad=0.15)
    add_panel_label(ax, "b", x=-0.24, y=1.16)

    # c -- reconciliation of the withdrawn timings
    ax = axes[2]
    v5 = summ[(summ.tool == "magicc") & (summ.input_set == "set_E_full")
              & (summ.cache == "warm") & (summ.threads == 1)]
    if v5.empty:
        raise ValueError("no warm 1-thread set_E_full MAGICC row")
    v5_1thr = float(v5.iloc[0].wall_median_s)
    v5_32 = summ[(summ.tool == "magicc") & (summ.input_set == "set_E_full")
                 & (summ.cache == "warm") & (summ.threads == 32)]
    v5_32thr = float(v5_32.iloc[0].wall_median_s)
    items = [
        ('"40 s"\nsubmitted main text',
         float(rec["figure_40_s"]["arithmetic_check_s"]), True, "0.55"),
        ("74.4 s\nV3, script 40",
         float(rec["figure_74_4_s_and_7_9_s"]["one_thread_s"]), False,
         AUX["mauve"]),
        ("97.5 s\nV3, submitted Table S4",
         float(rec["figure_97_5_s"]["value_s"]), False, AUX["mauve"]),
        (f"{v5_1thr:.1f} s\nV5, this work, 1 thread",
         v5_1thr, False, PALETTE["magicc"]),
        (f"{v5_32thr:.2f} s\nV5, this work, 32 threads",
         v5_32thr, False, PALETTE["magicc"]),
    ]
    x = np.arange(len(items))
    for xi, (lab, val, withdrawn, col) in zip(x, items):
        ax.plot([xi, xi], [0.5, val], color=col, lw=0.9, zorder=1)
        ax.plot([xi], [val], marker="X" if withdrawn else "o", ms=4.6,
                color=col, mec="0.15", mew=0.5, zorder=3)
        ax.annotate(f"{val:.1f}" if val >= 10 else f"{val:.2f}",
                    (xi, val), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=5.6, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(2, 400)
    cat_axis(ax, [i[0] for i in items], rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=5.0)
    ax.set_ylabel("Wall clock (s), 1,000 genomes")
    ax.set_title("Reconciliation of the withdrawn timings", fontsize=7, pad=3)
    ax.legend(handles=[Line2D([], [], marker="X", ls="none", ms=4.0,
                              color="0.55", mec="0.15",
                              label="never measured (withdrawn)")],
              loc="upper right", frameon=False, fontsize=5.2,
              handletextpad=0.3, borderpad=0.15)
    add_panel_label(ax, "c", x=-0.24, y=1.16)

    save_fig(fig, name, out_dir=OUT)

    eff = scal[(scal.tool == "magicc") & (scal.threads == 32)
               & (scal.input_set == "set_E_full")]
    if eff.empty:
        raise ValueError("no MAGICC 32-thread set_E_full scaling row")
    eff_pct = float(eff.parallel_efficiency_pct.iloc[0])
    caption(name,
            "**Supplementary Figure S7 | Computational performance measured on "
            "matched hardware, and reconciliation of the withdrawn timings.** "
            "All runs are on one 48-CPU host verified idle before each cell; "
            "wall clock is end-to-end (`/usr/bin/time -v` around the complete "
            "process, including interpreter start, imports, model load, input "
            "read and output write). **a**, Median wall clock against thread "
            "count for the matched 100-genome cell of Set E, warm page cache; "
            "n = 3 repeats per cell, log-log axes. **b**, Peak resident set size "
            "in the matched 32-thread cell; every individual repeat is plotted "
            "(n = 1-3 per cell) for the 100-genome and the 1,000-genome input. "
            "**c**, What each published MAGICC timing actually measured. The "
            "cross marks the withdrawn \"1,000 genomes in 40 s\" of the "
            "submitted main text: no run of that duration exists: the value is "
            "1,000 genomes divided by 1,451 genomes min-1 thread-1, itself the "
            "arithmetic mean of five per-set rates taken from an internal "
            "compute-phase timer. The 74.4 s and 97.5 s figures are genuine "
            "end-to-end measurements but were made on magicc_v3.onnx, neither "
            "the model of the submitted manuscript nor the released one. "
            f"Statistics: medians of repeated runs; MAGICC parallel efficiency "
            f"at 32 threads on the 1,000-genome cell is {eff_pct:.0f} % "
            "(8.1x speed-up over one thread), so speed-up is not linear in "
            "thread count. Ratios between tools are quoted only within a "
            "matched cell (same input, same thread count, same host state). "
            "Reference-genome clustering does not apply here: the unit of "
            "replication is a timed process, not a genome. "
            "Source: results/revision/speed/{matched_thread_summary.tsv, "
            "matched_thread_runs.tsv, scaling_efficiency.tsv, "
            "reconciliation_40s_vs_97.5s.json}. n = 1,000 genomes "
            "(4,806,980,767 bp) for the full Set E cell.")


# ---------------------------------------------------------------------------
# S8  Circularity safeguard (set H, WS1.11)
# ---------------------------------------------------------------------------
SEVERITY = ["mild", "moderate", "severe"]


def fig_S8():
    name = "Figure_S8"
    print(f"[{name}] circularity safeguard, set H")
    circ = os.path.join(RESULTS, "circularity")
    arm = read_tsv(os.path.join(circ, "ws1_11_arm_metrics.tsv"))
    paired = read_tsv(os.path.join(circ, "ws1_11_paired_arm_tests.tsv"))
    mech = read_tsv(os.path.join(circ, "ws1_11_reference_incompleteness_mechanism.tsv"))
    perref = read_tsv(os.path.join(circ, "ws1_11_per_reference_errors.tsv"))
    reflev = read_tsv(os.path.join(circ, "ws1_11_reference_level_scores.tsv"))

    # per-sample predictions for the violin panels
    base = os.path.join(BENCH, "set_H_ncbi")
    meta = read_tsv(os.path.join(base, "metadata.tsv"))[
        ["genome_id", "arm", "pair_id", "checkm2_severity"]]
    per = {}
    for tool in TOOLS:
        df = read_tsv(os.path.join(base, PRED_FILE[tool]))
        df = df.merge(meta, on="genome_id", validate="one_to_one",
                      suffixes=("", "_m"))
        if "arm_m" in df.columns:
            df["arm"] = df["arm_m"]
        df["comp_err"] = df.pred_completeness - df.true_completeness
        df["cont_err"] = df.pred_contamination - df.true_contamination
        per[tool] = df

    fig = plt.figure(figsize=(7.2, 8.1))
    gs = fig.add_gridspec(3, 2, left=0.085, right=0.985, top=0.955,
                          bottom=0.065, hspace=0.55, wspace=0.30)

    ARMS = [("H_pass", "would have PASSED"), ("H_fail", "would have FAILED")]

    # a, b -- signed error by arm
    for k, (col, metric, ylab, ylim) in enumerate(
            [("comp_err", "completeness", "Signed completeness error (pp)", (-40, 45)),
             ("cont_err", "contamination", "Signed contamination error (pp)", (-60, 45))]):
        ax = fig.add_subplot(gs[0, k])
        for i, (a, _) in enumerate(ARMS):
            violin_group(ax, {t: per[t].loc[per[t].arm == a, col].to_numpy()
                              for t in TOOLS}, i)
        hline0(ax)
        cat_axis(ax, ["would have\nPASSED", "would have\nFAILED"])
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.set_xlabel("CheckM2-based curation filter")
        add_panel_label(ax, "ab"[k], x=-0.17, y=1.14)
        if k == 0:
            tool_legend(ax, loc="lower left", ncol=2, fontsize=5.6)

    # c -- paired matched-pair difference D (fail - pass) in absolute error
    ax = fig.add_subplot(gs[1, 0])
    pa = paired[(paired.subset == "all pairs") & (paired.statistic == "abs_error")]
    for j, metric in enumerate(("completeness", "contamination")):
        for i, tool in enumerate(TOOLS):
            r = pa[(pa.metric_name == metric) & (pa.tool == TSV_TOOL[tool])]
            if r.empty:
                raise ValueError(f"missing paired arm test for {tool}/{metric}")
            r = r.iloc[0]
            pos = j + (i - 1.5) * 0.19
            errbars(ax, [pos], [r.D_fail_minus_pass], [r.ci_lo], [r.ci_hi], tool)
    hline0(ax)
    cat_axis(ax, ["completeness", "contamination"])
    ax.set_ylabel("D = MAE(failed) − MAE(passed) (pp)")
    ax.set_xlabel("Metric")
    add_panel_label(ax, "c", x=-0.17, y=1.14)
    tool_legend(ax, loc="upper left", ncol=2, fontsize=5.6)

    # d -- MAE by CheckM2-deficit severity stratum, completeness
    ax = fig.add_subplot(gs[1, 1])
    for i, sev in enumerate(SEVERITY):
        for j, (grp, off) in enumerate(((f"H_pass:matched_to_{sev}", -0.19),
                                        (f"H_fail:{sev}", 0.19))):
            for k2, tool in enumerate(TOOLS):
                r = arm[(arm.tool == TSV_TOOL[tool]) & (arm.group == grp)]
                if r.empty:
                    continue
                r = r.iloc[0]
                pos = i + off + (k2 - 1.5) * 0.075
                errbars(ax, [pos], [r.comp_mae], [r.comp_mae_ci_lo],
                        [r.comp_mae_ci_hi], tool, ms=2.6,
                        mfc="white" if j == 0 else PALETTE[tool])
    cat_axis(ax, [f"{s}\n(n = {int(arm[(arm.tool=='magicc_v5') & (arm.group=='H_fail:'+s)].n.iloc[0]):,})"
                  for s in SEVERITY])
    ax.set_ylabel("Completeness MAE (pp)")
    ax.set_xlabel("CheckM2-deficit severity of the reference")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.0, mfc="white",
                              mec="0.3", label="matched control (passed)"),
                       Line2D([], [], marker="o", ls="none", ms=3.0, mfc="0.3",
                              mec="0.2", label="would have failed")],
              loc="upper right", frameon=False, fontsize=5.4,
              handletextpad=0.35, labelspacing=0.25, borderpad=0.15)
    add_panel_label(ax, "d", x=-0.17, y=1.14)

    # e -- reference-incompleteness mechanism
    ax = fig.add_subplot(gs[2, 0])
    xs = np.linspace(0, perref.checkm2_completeness_deficit.max(), 50)
    for tool in TOOLS:
        sub = perref[perref.tool == TSV_TOOL[tool]]
        ax.plot(sub.checkm2_completeness_deficit, sub.comp_bias, ls="none",
                marker=MARKERS[tool], ms=1.9, alpha=0.30, rasterized=True,
                color=PALETTE[tool], mec="none")
        m = mech[(mech.tool == TSV_TOOL[tool]) & (mech.y == "comp_bias")]
        if m.empty:
            continue
        m = m.iloc[0]
        circular = not bool(m.independent_of_the_x_variable)
        ax.plot(xs, m.ols_intercept + m.ols_slope * xs, color=PALETTE[tool],
                lw=1.2, ls=":" if circular else "-", path_effects=line_pe(tool),
                label=f"{TOOL_LABEL[tool]} {m.ols_slope:+.2f}"
                      + (" (circular)" if circular else ""))
    hline0(ax)
    ax.set_xlabel("CheckM2 completeness deficit of the reference (pp below 98 %)")
    ax.set_ylabel("Per-reference mean signed\ncompleteness error (pp)")
    ax.set_ylim(-25, 35)
    ax.legend(loc="upper right", frameon=False, fontsize=5.2,
              handletextpad=0.35, labelspacing=0.25, borderpad=0.15)
    add_panel_label(ax, "e", x=-0.17, y=1.14)

    # f -- unmodified deposited assemblies (reference-level scores)
    ax = fig.add_subplot(gs[2, 1])
    est = [("magicc_v5_completeness", "magicc", "completeness"),
           ("checkm2_local_completeness", "checkm2", "completeness"),
           ("cocopye_completeness", "cocopye", "completeness"),
           ("magicc_v5_contamination", "magicc", "contamination"),
           ("checkm2_local_contamination", "checkm2", "contamination"),
           ("cocopye_contamination", "cocopye", "contamination")]
    ticks, labels = [], []
    for i, (key, tool, metric) in enumerate(est):
        r = reflev[reflev.estimate == key]
        if r.empty:
            raise ValueError(f"missing reference-level estimate {key}")
        r = r.iloc[0]
        errbars(ax, [i], [r.D_fail_minus_pass], [r.ci_lo], [r.ci_hi], tool)
        ticks.append(i)
        labels.append(f"{TOOL_LABEL[tool]}\n{metric[:4]}.")
    hline0(ax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=5.2)
    ax.set_xlim(-0.6, len(est) - 0.4)
    ax.set_ylabel("Failed − passed on the unmodified\ndeposited assemblies (pp)")
    ax.set_xlabel("Estimator (n = 200 matched reference pairs)")
    add_panel_label(ax, "f", x=-0.17, y=1.14)

    save_fig(fig, name, out_dir=OUT)

    d_comp = pa[(pa.metric_name == "completeness") & (pa.tool == "magicc_v5")].iloc[0]
    d_cont = pa[(pa.metric_name == "contamination") & (pa.tool == "magicc_v5")].iloc[0]
    caption(name,
            "**Supplementary Figure S8 | Circularity safeguard (set H): the "
            "CheckM2-based reference curation does not manufacture MAGICC's "
            "advantage.** Set H holds 4,000 simulations built from 400 NCBI "
            "reference genomes forming 200 taxonomy- and size-matched pairs: one "
            "member of each pair would have passed the project's CheckM2-based "
            "curation filter (>=98 % completeness, <=2 % contamination) and the "
            "other would have failed it; both members share identical target "
            "completeness/contamination draws. **a**, **b**, Signed error "
            "(predicted minus true, pp) by arm, for completeness and "
            "contamination; each element is a distribution (kernel density, box "
            "= median and IQR with 1.5x IQR whiskers, open marker = mean), n = "
            "2,000 simulations from 200 references per arm. **c**, The primary "
            "estimator: D, the mean per-pair difference in absolute error "
            "between the would-have-failed reference and its matched control. "
            "D > 0 means the tool is worse on the genomes the curation removed. "
            "**d**, Completeness MAE by the severity of the reference's CheckM2 "
            "deficit (mild n = 870, moderate n = 670, severe n = 460 "
            "simulations, with their matched controls). **e**, Per-reference "
            "mean signed completeness error against how far the reference fell "
            "below the 98 % criterion, over all 400 references; the fitted OLS "
            "slope is annotated. For MAGICC, CoCoPyE and DeepCheck the x axis is "
            "independent evidence; for CheckM2 it is its own estimate, so its "
            "slope (dotted) is circular by construction and shown only for "
            "contrast. **f**, The same paired contrast measured on the "
            "unmodified deposited assemblies rather than on simulations; "
            "DeepCheck is absent from this panel because it was not run at the "
            "reference level. "
            "Statistics: 95 % percentile cluster bootstrap over the 200 matched "
            "reference pairs, 2,000 resamples; two-sided paired Wilcoxon tests "
            "with Benjamini-Hochberg correction. MAGICC's paired penalty is "
            f"D = {d_comp.D_fail_minus_pass:+.2f} pp "
            f"[{d_comp.ci_lo:.2f}, {d_comp.ci_hi:.2f}] for completeness and "
            f"{d_cont.D_fail_minus_pass:+.2f} pp "
            f"[{d_cont.ci_lo:.2f}, {d_cont.ci_hi:.2f}] for contamination, both "
            "far below the pre-registered 2.0 pp materiality threshold, so the "
            "curation introduces no material bias. Set H MAEs are not comparable "
            "across experiments: its reference set was deliberately not "
            "quality-curated.")


# ---------------------------------------------------------------------------
# S9  GUNC as a DETECTION comparator (WS4.2)
#     GUNC is never given a MAE: pass/fail, sensitivity/specificity and CSS
#     rank correlation only.  Every rate names its power stratum.
# ---------------------------------------------------------------------------
GUNC_SETS = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]
GUNC_SET_LABEL = {"set_A_v2": "Set A", "set_B_v2": "Set B",
                  "set_C_clean": "Set C-clean", "set_D_clean": "Set D-clean",
                  "set_E": "Set E"}
DB_ORDER = ["progenomes_2.1", "gtdb_95"]
DB_LABEL = {"progenomes_2.1": "proGenomes 2.1", "gtdb_95": "GTDB r95"}
DB_COLOR = {"progenomes_2.1": PALETTE["gunc"], "gtdb_95": PALETTE["truth"]}
DB_MARKER = {"progenomes_2.1": "s", "gtdb_95": "o"}

DETECTOR_KEY = {"gunc": "GUNC 1.1.1 (pass.GUNC == False)",
                "magicc": "MAGICC V5 (pred contamination >= 5 %)",
                "checkm2": "CheckM2 (pred contamination >= 5 %)"}
DET_ORDER = ["gunc", "magicc", "checkm2"]

SCORE_KEY = {"gunc_css": "GUNC 1.1.1 CSS",
             "gunc_portion": "GUNC 1.1.1 contamination_portion",
             "magicc": "MAGICC V5 predicted contamination",
             "checkm2": "CheckM2 predicted contamination"}
SCORE_STYLE = {"gunc_css": (PALETTE["gunc"], "s"),
               "gunc_portion": (PALETTE["truth"], "P"),
               "magicc": (PALETTE["magicc"], "o"),
               "checkm2": (PALETTE["checkm2"], "^")}
SCORE_LABEL = {"gunc_css": "GUNC CSS", "gunc_portion": "GUNC contamination_portion",
               "magicc": "MAGICC V5 predicted cont.",
               "checkm2": "CheckM2 predicted cont."}


def fig_S9():
    name = "Figure_S9"
    print(f"[{name}] GUNC detection comparator")
    g = os.path.join(RESULTS, "gunc")
    power = read_tsv(os.path.join(g, "gunc_power_audit.tsv"))
    strat = read_tsv(os.path.join(g, "gunc_stratified_passfail.tsv"))
    agree = read_tsv(os.path.join(g, "gunc_threshold_agreement.tsv"))
    css = read_tsv(os.path.join(g, "gunc_css_correlations.tsv"))
    dbs = read_tsv(os.path.join(g, "gunc_db_sensitivity.tsv"))
    peff = read_tsv(os.path.join(g, "gunc_power_effect.tsv"))

    # Canvas width 7.15 in, not 7.2 in. savefig.bbox="tight" saves the union
    # of the canvas and every artist that overflows it, so S9 on a full-width
    # canvas rendered 180.49 mm against the 183 mm Nature Communications cap
    # and the 180 mm working target this build holds every figure to; at 7.15
    # in it renders 179.25 mm. Font sizes are unchanged in points, the canvas
    # was stepped down and re-measured 0.05 in at a time, and no new text
    # collision appears at any width down to 7.00 in. No plotted value moves:
    # the artist ledger is byte-identical to the full-width build.
    fig = plt.figure(figsize=(7.15, 9.6))
    gs = fig.add_gridspec(4, 2, left=0.09, right=0.985, top=0.96,
                          bottom=0.065, hspace=0.62, wspace=0.32,
                          height_ratios=[1, 1, 1, 0.85])

    # a -- power audit
    ax = fig.add_subplot(gs[0, 0])
    for db in DB_ORDER:
        xs, ys = [], []
        for i, s in enumerate(GUNC_SETS):
            r = power[(power.set == s) & (power.db == db)]
            if r.empty:
                continue
            xs.append(i)
            ys.append(float(r.frac_powered.iloc[0]) * 100)
        ax.plot(xs, ys, ls="none", marker=DB_MARKER[db], ms=4.2,
                color=DB_COLOR[db], mec="0.2", mew=0.4, label=DB_LABEL[db])
    ax.axhline(100, color="0.75", lw=0.5, ls=":")
    cat_axis(ax, [GUNC_SET_LABEL[s] for s in GUNC_SETS], rotation=30, ha="right")
    ax.set_ylabel("Genomes in the powered\nstratum (%)")
    ax.set_ylim(60, 104)
    ax.set_title("Power audit (n = 1,000 per run)", fontsize=7, pad=3)
    ax.legend(loc="lower left", frameon=False, fontsize=5.6,
              handletextpad=0.35, borderpad=0.15)
    add_panel_label(ax, "a", x=-0.20, y=1.16)

    # b -- clean-genome fail rate, all strata vs powered, with denominators
    ax = fig.add_subplot(gs[0, 1])
    clean = strat[(strat.true_contamination_band_pct == "0-5")
                  & (strat.power_stratum.isin(["all", "powered"]))]
    ticklab = []
    for i, s in enumerate(GUNC_SETS):
        for j, db in enumerate(DB_ORDER):
            for k, stratum in enumerate(["all", "powered"]):
                r = clean[(clean.set == s) & (clean.db == db)
                          & (clean.power_stratum == stratum)]
                if r.empty or int(r.n.iloc[0]) == 0:
                    continue
                r = r.iloc[0]
                pos = i + (j - 0.5) * 0.34 + (k - 0.5) * 0.15
                ax.plot([pos], [float(r.gunc_fail_rate) * 100],
                        marker=DB_MARKER[db], ms=3.6, color=DB_COLOR[db],
                        mfc=DB_COLOR[db] if stratum == "powered" else "none",
                        mec=DB_COLOR[db], mew=0.8, ls="none")
        nrow = clean[(clean.set == s) & (clean.power_stratum == "all")]
        ticklab.append(f"{GUNC_SET_LABEL[s]}\n(n = {int(nrow.n.iloc[0]):,})"
                       if not nrow.empty else GUNC_SET_LABEL[s])
    cat_axis(ax, ticklab, rotation=30, ha="right")
    ax.set_ylabel("GUNC fail rate on truly\nclean genomes (%)")
    ax.set_ylim(-4, 100)
    ax.set_title("False positives on genuinely clean genomes\n"
                 "(true contamination < 5 %)", fontsize=6.6, pad=3)
    ax.legend(handles=[Line2D([], [], marker="s", ls="none", ms=3.6,
                              mfc="none", mec=DB_COLOR["progenomes_2.1"], mew=0.8,
                              label="proGenomes 2.1, all strata"),
                       Line2D([], [], marker="s", ls="none", ms=3.6,
                              mfc=DB_COLOR["progenomes_2.1"],
                              mec=DB_COLOR["progenomes_2.1"],
                              label="proGenomes 2.1, powered"),
                       Line2D([], [], marker="o", ls="none", ms=3.6,
                              mfc="none", mec=DB_COLOR["gtdb_95"], mew=0.8,
                              label="GTDB r95, all strata"),
                       Line2D([], [], marker="o", ls="none", ms=3.6,
                              mfc=DB_COLOR["gtdb_95"], mec=DB_COLOR["gtdb_95"],
                              label="GTDB r95, powered")],
              loc="upper left", frameon=False, fontsize=5.0,
              handletextpad=0.3, labelspacing=0.2, borderpad=0.15)
    add_panel_label(ax, "b", x=-0.20, y=1.16)

    # c, d -- sensitivity and specificity at the 5 % boundary, all strata
    for k, (metric, ciname, lab) in enumerate(
            (("sensitivity_recall", "sensitivity", "Sensitivity"),
             ("specificity", "specificity", "Specificity"))):
        ax = fig.add_subplot(gs[1, k])
        sub = agree[(agree.threshold_pct == 5.0)
                    & (agree.power_stratum == "all")
                    & (agree.db == "progenomes_2.1")]
        det_sets = [x for x in GUNC_SETS if x in set(sub.set)]
        for i, s in enumerate(det_sets):
            for j, det in enumerate(DET_ORDER):
                r = sub[(sub.set == s) & (sub.detector == DETECTOR_KEY[det])]
                if r.empty:
                    continue
                r = r.iloc[0]
                pos = i + (j - 1) * 0.22
                col = PALETTE["gunc"] if det == "gunc" else PALETTE[det]
                mk = "s" if det == "gunc" else MARKERS[det]
                lo, hi = r[f"{ciname}_ci_lo"], r[f"{ciname}_ci_hi"]
                y = float(r[metric])
                ax.errorbar([pos], [y],
                            yerr=[[max(y - lo, 0)], [max(hi - y, 0)]],
                            color=col, marker=mk, ms=3.2, ls="none",
                            capsize=1.5, elinewidth=0.7, mec="0.2", mew=0.4)
        cat_axis(ax, [GUNC_SET_LABEL[s] for s in det_sets], rotation=30,
                 ha="right")
        ax.set_ylabel(f"{lab} at the 5 % boundary")
        ax.set_ylim(-0.03, 1.06)
        ax.set_title(f"{lab}, proGenomes 2.1, all strata\n"
                     "(Set A omitted: no contaminated genomes)",
                     fontsize=6.4, pad=3)
        if k == 0:
            ax.legend(handles=[
                Line2D([], [], marker="s", ls="none", ms=3.2,
                       color=PALETTE["gunc"], mec="0.2", label="GUNC 1.1.1"),
                Line2D([], [], marker=MARKERS["magicc"], ls="none", ms=3.2,
                       color=PALETTE["magicc"], mec="0.2", label="MAGICC V5"),
                Line2D([], [], marker=MARKERS["checkm2"], ls="none", ms=3.2,
                       color=PALETTE["checkm2"], mec="0.2", label="CheckM2")],
                loc="lower left", frameon=False, fontsize=5.4,
                handletextpad=0.3, labelspacing=0.2, borderpad=0.15)
        add_panel_label(ax, "cd"[k], x=-0.20, y=1.16)

    # e -- Spearman rank correlation against true contamination
    ax = fig.add_subplot(gs[2, 0])
    sub = css[(css.power_stratum == "all") & (css.db == "progenomes_2.1")]
    css_sets = [s for s in GUNC_SETS if s != "set_A_v2"]
    for i, s in enumerate(css_sets):
        for j, key in enumerate(["gunc_css", "gunc_portion", "magicc", "checkm2"]):
            r = sub[(sub.set == s) & (sub.score == SCORE_KEY[key])]
            if r.empty:
                continue
            r = r.iloc[0]
            col, mk = SCORE_STYLE[key]
            pos = i + (j - 1.5) * 0.19
            y = float(r.spearman_rho_vs_true_contamination)
            ax.errorbar([pos], [y],
                        yerr=[[max(y - r.spearman_ci_lo, 0)],
                              [max(r.spearman_ci_hi - y, 0)]],
                        color=col, marker=mk, ms=3.2, ls="none", capsize=1.5,
                        elinewidth=0.7, mec="0.2", mew=0.4)
    hline0(ax)
    cat_axis(ax, [GUNC_SET_LABEL[s] for s in css_sets], rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ vs true contamination")
    ax.set_ylim(-0.34, 1.05)
    ax.set_title("Rank agreement with truth\n(Set A omitted: truth is constant 0 %)",
                 fontsize=6.6, pad=3)
    ax.legend(handles=[Line2D([], [], marker=SCORE_STYLE[k][1], ls="none", ms=3.2,
                              color=SCORE_STYLE[k][0], mec="0.2",
                              label=SCORE_LABEL[k])
                       for k in ["gunc_css", "gunc_portion", "magicc", "checkm2"]],
              loc="lower left", ncol=2, frameon=False, fontsize=5.0,
              handletextpad=0.3, labelspacing=0.2, borderpad=0.15)
    add_panel_label(ax, "e", x=-0.20, y=1.16)

    # f -- proGenomes 2.1 -> GTDB r95, paired deltas
    ax = fig.add_subplot(gs[2, 1])
    wanted = [("fraction powered", "Δ fraction powered"),
              ("detection sensitivity at >=5 % contamination", "Δ sensitivity"),
              ("detection specificity at >=5 % contamination", "Δ specificity")]
    db_sets = ["set_C_clean", "set_D_clean", "set_E"]
    mks = ["o", "^", "D"]
    for i, (key, lab) in enumerate(wanted):
        for j, s in enumerate(db_sets):
            r = dbs[(dbs.set == s) & (dbs.metric == key)
                    & (dbs.power_stratum == "all_paired")]
            if r.empty:
                continue
            r = r.iloc[0]
            pos = i + (j - 1) * 0.24
            y = float(r.delta_gtdb95_minus_progenomes)
            ax.errorbar([pos], [y],
                        yerr=[[max(y - r.delta_ci_lo, 0)],
                              [max(r.delta_ci_hi - y, 0)]],
                        color=PALETTE["magicc"] if j == 0 else
                        (PALETTE["checkm2"] if j == 1 else PALETTE["truth"]),
                        marker=mks[j], ms=3.2, ls="none", capsize=1.5,
                        elinewidth=0.7, mec="0.2", mew=0.4)
    hline0(ax)
    cat_axis(ax, [w[1] for w in wanted], rotation=20, ha="right")
    ax.set_ylabel("GTDB r95 − proGenomes 2.1")
    ax.set_title("Database swap, paired on identical genomes", fontsize=6.6, pad=3)
    ax.legend(handles=[Line2D([], [], marker=mks[j], ls="none", ms=3.2,
                              color=[PALETTE["magicc"], PALETTE["checkm2"],
                                     PALETTE["truth"]][j], mec="0.2",
                              label=GUNC_SET_LABEL[s])
                       for j, s in enumerate(db_sets)],
              loc="lower left", frameon=False, fontsize=5.4,
              handletextpad=0.3, labelspacing=0.2, borderpad=0.15)
    add_panel_label(ax, "f", x=-0.20, y=1.16)

    # g -- the power effect and what the database swap does to it
    ax = fig.add_subplot(gs[3, :])
    PE_DET = {"gunc": "GUNC 1.1.1 (pass.GUNC == False)",
              "magicc": "MAGICC V5 (pred >= 5 %)",
              "checkm2": "CheckM2 (pred >= 5 %)"}
    pe_sets = [x for x in GUNC_SETS if x in set(peff.set)]
    ticks, ticklab = [], []
    set_centres = []
    pos = 0.0
    for si, st in enumerate(pe_sets):
        for det in DET_ORDER:
            for db in DB_ORDER:
                r = peff[(peff.set == st) & (peff.db == db)
                         & (peff.threshold_pct == 5.0)
                         & (peff.detector == PE_DET[det])]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = float(r.delta_sensitivity_powered_minus_unpowered)
                xp = pos + (DB_ORDER.index(db) - 0.5) * 0.26
                col = PALETTE["gunc"] if det == "gunc" else PALETTE[det]
                mk = "s" if det == "gunc" else MARKERS[det]
                ax.errorbar([xp], [y],
                            yerr=[[max(y - r.delta_ci_lo, 0)],
                                  [max(r.delta_ci_hi - y, 0)]],
                            color=col, marker=mk, ms=3.4, ls="none",
                            mfc=col if db == "gtdb_95" else "none",
                            mec=col, mew=0.8, capsize=1.5, elinewidth=0.7)
            ticks.append(pos)
            ticklab.append({"gunc": "GUNC", "magicc": "MAGICC",
                            "checkm2": "CheckM2"}[det])
            pos += 1.0
        set_centres.append(pos - 2.0)
        if si < len(pe_sets) - 1:
            ax.axvline(pos - 0.5, color="0.85", lw=0.5)
        pos += 0.5
    hline0(ax)
    lo9, hi9 = ax.get_ylim()
    ax.set_ylim(lo9, hi9 + 0.30 * (hi9 - lo9))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklab, rotation=25, ha="right", fontsize=5.4)
    ax.set_xlim(-0.8, ticks[-1] + 0.8)
    for c, st in zip(set_centres, pe_sets):
        ax.text(c, ax.get_ylim()[1], GUNC_SET_LABEL[st],
                ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("Δ sensitivity at 5 %\n(powered − unpowered)")
    ax.set_title("The power effect, and what the database swap does to it",
                 fontsize=6.8, pad=12)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.4,
                              mfc="none", mec="0.35", mew=0.8,
                              label="proGenomes 2.1"),
                       Line2D([], [], marker="o", ls="none", ms=3.4,
                              mfc="0.35", mec="0.35", label="GTDB r95")],
              loc="upper right", frameon=False, fontsize=5.4,
              handletextpad=0.3, labelspacing=0.2, borderpad=0.15)
    add_panel_label(ax, "g", x=-0.095, y=1.20)

    save_fig(fig, name, out_dir=OUT)

    # numbers quoted in the caption, read back from the files
    dC = clean[(clean.set == "set_D_clean") & (clean.db == "progenomes_2.1")]
    d_all = dC[dC.power_stratum == "all"].iloc[0]
    d_pow = dC[dC.power_stratum == "powered"].iloc[0]
    aC = clean[(clean.set == "set_A_v2") & (clean.power_stratum == "all")].iloc[0]
    powC = power[(power.set == "set_C_clean")]
    frac_pro = float(powC[powC.db == "progenomes_2.1"].frac_powered.iloc[0])
    frac_gtdb = float(powC[powC.db == "gtdb_95"].frac_powered.iloc[0])
    dsC = dbs[(dbs.set == "set_C_clean") & (dbs.power_stratum == "all_paired")]
    dsen = dsC[dsC.metric == "detection sensitivity at >=5 % contamination"].iloc[0]
    dspe = dsC[dsC.metric == "detection specificity at >=5 % contamination"].iloc[0]

    caption(name,
            "**Supplementary Figure S9 | GUNC as a detection comparator: power "
            "audit, sensitivity and specificity at the 5 % boundary, and the "
            "proGenomes 2.1 versus GTDB r95 database swap.** GUNC is compared "
            "only as a *detector* - pass/fail and rank agreement - and is never "
            "assigned a mean absolute error, because its clade separation score "
            "(CSS) is not an estimate of percent contamination. 8 runs x 1,000 "
            "genomes = 8,000 GUNC calls (GUNC 1.1.1, DIAMOND 2.1.24, Prodigal "
            "2.6.3); every genome was scored and none dropped. A genome is "
            "*powered* when genes_retained_index > 0.4 and "
            "reference_representation_score >= 0.5; every rate below names its "
            "stratum, because pass/fail differs between them. "
            "GTDB r95 was run only on Set C-clean, Set D-clean and Set E; "
            "proGenomes 2.1 already powers 98.4 % of Set A and 99.3 % of Set B, "
            "so there was no headroom to test. "
            "**a**, Fraction of genomes in the powered stratum, per set and "
            f"database (Set C-clean {frac_pro*100:.1f} % on proGenomes 2.1 "
            f"versus {frac_gtdb*100:.1f} % on GTDB r95). **b**, GUNC fail rate "
            "on genomes whose *true* contamination is below 5 %, shown for both "
            "power strata; open markers are all strata, filled markers the "
            "powered stratum, and the denominator (the number of truly clean "
            "genomes) is printed on the axis. On Set D-clean GUNC fails "
            f"{float(d_all.gunc_fail_rate)*100:.1f} % of genuinely clean "
            f"archaeal genomes across all strata "
            f"({int(d_all.n_fail)}/{int(d_all.n)}) and "
            f"{float(d_pow.gunc_fail_rate)*100:.1f} % in the powered stratum "
            f"({int(d_pow.n_fail)}/{int(d_pow.n)}); on the mainstream Set A "
            f"control the same rate is {float(aC.gunc_fail_rate)*100:.1f} % "
            f"({int(aC.n_fail)}/{int(aC.n):,}), so the failure mode is reference "
            "representation, not the tool. **c**, **d**, Sensitivity and "
            "specificity of GUNC, MAGICC V5 and CheckM2 at the MIMAG-inspired "
            "5 % contamination boundary (proGenomes 2.1, all strata). GUNC "
            "out-detects CheckM2 on recall everywhere; CheckM2 buys perfect "
            "specificity on the novel-lineage sets, GUNC does not. **e**, "
            "Spearman rank correlation with true contamination; Set A is omitted "
            "because its true contamination is uniformly 0 % and rho is "
            "undefined (R1-m19). **f**, Effect of swapping the GUNC reference "
            "database, paired on identical genomes. On Set C-clean GTDB r95 buys "
            f"{dsen.delta_gtdb95_minus_progenomes:+.3f} "
            f"[{dsen.delta_ci_lo:.3f}, {dsen.delta_ci_hi:.3f}] sensitivity for "
            f"{dspe.delta_gtdb95_minus_progenomes:+.3f} "
            f"[{dspe.delta_ci_lo:.3f}, {dspe.delta_ci_hi:.3f}] specificity. "
            "**g**, The power effect itself: the change in detection sensitivity "
            "at the 5 % boundary between the powered and the unpowered stratum, "
            "for each detector and set; open markers are proGenomes 2.1 and "
            "filled markers GTDB r95, so the panel shows both how much of GUNC's "
            "behaviour is a power effect and how much of that the database swap "
            "removes. Source: `results/revision/gunc/gunc_power_effect.tsv`. "
            "Statistics: 95 % percentile cluster bootstrap over reference "
            "genomes, 2,000 resamples; n = 1,000 genomes per set per database, "
            "drawn from 798 (Set A), 803 (Set B), 100 (Set C-clean), 100 "
            "(Set D-clean) and 785 (Set E) reference-genome clusters.")


# ---------------------------------------------------------------------------
# S10  Set F: contamination MAE and completeness signed-error heatmaps
# ---------------------------------------------------------------------------
F_TYPES = ["redundant", "replaced", "single"]
F_TYPE_LABEL = {"redundant": "redundant", "replaced": "replaced",
                "single": "single-copy"}
F_DIST = ["species", "genus", "family", "order", "class", "phylum"]


def _heat(ax, mat, cmap, norm, xlabels, ylabels, fmt="{:.1f}", show_y=True):
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=5.4)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels if show_y else [""] * len(ylabels), fontsize=5.4)
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.5)
    ax.tick_params(which="minor", length=0)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=4.6,
                    color="white" if lum < 0.5 else "0.1")
    return im


def fig_S10():
    name = "Figure_S10"
    print(f"[{name}] Set F heatmaps")
    f = read_tsv(os.path.join(RESULTS, "set_F", "set_F_cells_type_x_distance.tsv"))

    def grid(tool, col):
        m = np.full((len(F_TYPES), len(F_DIST)), np.nan)
        for i, ty in enumerate(F_TYPES):
            for j, d in enumerate(F_DIST):
                r = f[(f.tool == TSV_TOOL[tool]) & (f.contamination_type == ty)
                      & (f.distance == d)]
                if not r.empty:
                    m[i, j] = float(r[col].iloc[0])
        return m

    mae_all = np.concatenate([grid(t, "cont_mae").ravel() for t in TOOLS])
    bias_all = np.concatenate([grid(t, "comp_bias").ravel() for t in TOOLS])
    mae_norm = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(mae_all))
    b = float(np.nanmax(np.abs(bias_all)))
    bias_norm = TwoSlopeNorm(vmin=-b, vcenter=0.0, vmax=b)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4))
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.13,
                        hspace=0.75, wspace=0.30)

    for j, tool in enumerate(TOOLS):
        ax = axes[0, j]
        im1 = _heat(ax, grid(tool, "cont_mae"), fs.SEQUENTIAL_CMAP, mae_norm,
                    F_DIST, [F_TYPE_LABEL[t] for t in F_TYPES], show_y=(j == 0))
        ax.set_title(TOOL_LABEL[tool], fontsize=7, pad=3)
        if j == 0:
            ax.set_ylabel("Contamination type")
            add_panel_label(ax, "a", x=-0.55, y=1.30)
    for j, tool in enumerate(TOOLS):
        ax = axes[1, j]
        im2 = _heat(ax, grid(tool, "comp_bias"), DIVERGING_CMAP, bias_norm,
                    F_DIST, [F_TYPE_LABEL[t] for t in F_TYPES], show_y=(j == 0))
        ax.set_title(TOOL_LABEL[tool], fontsize=7, pad=3)
        ax.set_xlabel("Taxonomic distance", fontsize=6)
        if j == 0:
            ax.set_ylabel("Contamination type")
            add_panel_label(ax, "b", x=-0.55, y=1.30)

    cax1 = fig.add_axes([0.915, 0.545, 0.014, 0.34])
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("Contamination MAE (pp)", fontsize=5.8)
    cb1.ax.tick_params(labelsize=5)
    cax2 = fig.add_axes([0.915, 0.135, 0.014, 0.34])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label("Signed completeness error (pp)", fontsize=5.8)
    cb2.ax.tick_params(labelsize=5)

    save_fig(fig, name, out_dir=OUT)

    n_cell = int(f.n.iloc[0])
    n_clust = int(f.n_clusters.iloc[0])
    caption(name,
            "**Supplementary Figure S10 | Set F: contamination mean absolute "
            "error and signed completeness error across the contamination-type "
            "x taxonomic-distance grid.** Set F holds 1,800 contaminated samples "
            "built from 100 reference clusters, crossing three contamination "
            "types (redundant, the contaminant duplicates single-copy marker "
            "genes already present; replaced, the contaminant substitutes them; "
            "single-copy, the contaminant contributes them once) with six levels "
            "of taxonomic distance between contaminant and dominant genome. "
            "**a**, Contamination MAE per cell, one heatmap per tool, on a "
            "common perceptually uniform, colour-vision-deficiency optimised "
            "scale. **b**, Signed completeness error (predicted minus true) per "
            "cell, on a common blue-to-red diverging scale centred on zero; "
            "blue is underestimation and red overestimation, and every cell "
            "prints its own value, so no hue discrimination is required. "
            "Statistics: cell means over "
            f"n = {n_cell} samples per cell ({n_clust} reference clusters per "
            "cell, one sample per cluster); the printed value is the cell "
            "estimate in percentage points. Per-cell 95 % cluster-bootstrap "
            "confidence intervals and detection slopes are in "
            "results/revision/set_F/set_F_cells_type_x_distance.tsv. Set F is "
            "deliberately excluded from the pooled five-set benchmark: its "
            "stratification does not fit that panel's design.")


# ---------------------------------------------------------------------------
# S11  Set G: sequencing and assembly error robustness
# ---------------------------------------------------------------------------
G_TYPES = ["substitution", "substitution_titv", "indel", "chimera",
           "uneven_coverage"]
# Error PROCESSES, not tools.  Drawn from figstyle.AUX so that no curve in the
# mechanism row can be mistaken for one of the four tools plotted in a-d; the
# previous round drew "uneven-coverage duplication" in exactly MAGICC's red.
G_TYPE_STYLE = {"substitution": (AUX["navy"], "o"),
                "substitution_titv": (AUX["cyan"], "s"),
                "indel": (AUX["orange"], "^"),
                "chimera": (AUX["slate"], "D"),
                "uneven_coverage": (AUX["mauve"], "v")}
G_TYPE_LABEL = {"substitution": "substitutions",
                "substitution_titv": "substitutions (Ti/Tv 2:1)",
                "indel": "indels",
                "chimera": "chimeric joins",
                "uneven_coverage": "uneven-coverage duplication"}


def fig_S11():
    name = "Figure_S11"
    print(f"[{name}] Set G error robustness")
    g = os.path.join(RESULTS, "set_G")
    curves = read_tsv(os.path.join(g, "set_G_curves.tsv"))
    kmer = read_tsv(os.path.join(g, "set_G_kmer_perturbation_summary.tsv"))
    orf = read_tsv(os.path.join(g, "set_G_orf_disruption.tsv"))
    bound = read_tsv(os.path.join(g, "set_G_boundary.tsv"))

    # Canvas width 7.05 in, not 7.2 in. savefig.bbox="tight" saves the union
    # of the canvas and every artist that overflows it, so S11 on a full-width
    # canvas rendered 183.39 mm against the 183 mm Nature Communications cap
    # and the 180 mm working target this build holds every figure to; at 7.05
    # in it renders 179.86 mm. Font sizes are unchanged in points, the canvas
    # was stepped down and re-measured 0.05 in at a time, and no new text
    # collision appears at any width down to 6.95 in. No plotted value moves:
    # the artist ledger is byte-identical to the full-width build.
    fig = plt.figure(figsize=(7.05, 9.0))
    gs = fig.add_gridspec(5, 5, left=0.085, right=0.99, top=0.955,
                          bottom=0.055, hspace=0.85, wspace=0.50)

    rows = [("comp_mae", "Completeness\nMAE (pp)", "a", False),
            ("comp_bias", "Completeness signed\nerror (pp)", "b", True),
            ("cont_mae", "Contamination\nMAE (pp)", "c", False),
            ("cont_bias", "Contamination signed\nerror (pp)", "d", True)]

    for ri, (col, ylab, letter, zero) in enumerate(rows):
        ymin = min(curves[f"{col}_ci_lo"].min(), 0 if zero else curves[col].min())
        ymax = curves[f"{col}_ci_hi"].max()
        pad = 0.05 * (ymax - ymin)
        for ci, et in enumerate(G_TYPES):
            ax = fig.add_subplot(gs[ri, ci])
            for tool in TOOLS:
                sub = curves[(curves.tool == TSV_TOOL[tool])
                             & (curves.error_type == et)].sort_values("error_rate_pct")
                if sub.empty:
                    raise ValueError(f"no Set G rows for {tool}/{et}")
                ax.fill_between(sub.error_rate_pct, sub[f"{col}_ci_lo"],
                                sub[f"{col}_ci_hi"], color=PALETTE[tool],
                                alpha=0.16, lw=0)
                ax.plot(sub.error_rate_pct, sub[col], color=PALETTE[tool],
                        marker=MARKERS[tool], ms=2.4, lw=0.9,
                        ls=LINESTYLES[tool],
                        mec=EDGE.get(tool, PALETTE[tool]), mew=0.4,
                        path_effects=line_pe(tool), label=TOOL_LABEL[tool])
            if zero:
                hline0(ax)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.tick_params(labelsize=5)
            if ri == 0:
                ax.set_title(G_TYPE_LABEL[et], fontsize=5.8, pad=3)
            if ri == 3:
                ax.set_xlabel("Error rate (%)", fontsize=5.8)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=6)
                add_panel_label(ax, letter, x=-0.62, y=1.30)
            else:
                ax.set_yticklabels([])
            if ri == 0 and ci == 3:
                ax.legend(loc="upper left", frameon=False, fontsize=4.8,
                          handletextpad=0.3, labelspacing=0.2, borderpad=0.1)

    # mechanism row
    ax = fig.add_subplot(gs[4, 0:2])
    for et in G_TYPES:
        sub = kmer[kmer.error_type == et].sort_values("error_rate_pct")
        col, mk = G_TYPE_STYLE[et]
        ax.plot(sub.error_rate_pct, sub.observed_kmer_corruption * 100,
                color=col, marker=mk, ms=2.4, lw=0.9, mec="0.2", mew=0.3,
                label=G_TYPE_LABEL[et])
    ax.set_xlabel("Error rate (%)", fontsize=6)
    ax.set_ylabel("Observed 9-mer\ncorruption (%)", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.legend(loc="upper left", frameon=False, fontsize=4.8,
              handletextpad=0.3, labelspacing=0.2, borderpad=0.1)
    add_panel_label(ax, "e", x=-0.26, y=1.24)

    ax = fig.add_subplot(gs[4, 2:4])
    for et in G_TYPES:
        sub = orf[orf.error_type == et].sort_values("error_rate_pct")
        col, mk = G_TYPE_STYLE[et]
        ax.plot(sub.error_rate_pct,
                sub.Total_Coding_Sequences_ratio_vs_control * 100,
                color=col, marker=mk, ms=2.4, lw=0.9, mec="0.2", mew=0.3,
                label=G_TYPE_LABEL[et])
    ax.axhline(100, color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("Error rate (%)", fontsize=6)
    ax.set_ylabel("Predicted ORFs, % of\nthe error-free control", fontsize=6)
    ax.tick_params(labelsize=5)
    add_panel_label(ax, "f", x=-0.26, y=1.24)

    ax = fig.add_subplot(gs[4, 4])
    BCOL = "rate_pct_where_upperCI_delta_mae_exceeds_1pp"
    btypes = [t for t in G_TYPES if t != "chimera"]
    for i, et in enumerate(btypes):
        for tool in TOOLS:
            r = bound[(bound.tool == TSV_TOOL[tool]) & (bound.error_type == et)
                      & (bound.metric == "completeness")]
            if r.empty:
                continue
            v = r[BCOL].iloc[0]
            if not np.isfinite(v):
                continue
            ax.plot([float(v)], [i], marker=MARKERS[tool], ms=3.0,
                    color=PALETTE[tool], mec=EDGE.get(tool, "0.2"), mew=0.4,
                    ls="none")
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(btypes)))
    ax.set_yticklabels([G_TYPE_LABEL[t].replace(" (Ti/Tv 2:1)", "\nTi/Tv 2:1")
                        .replace("uneven-coverage ", "uneven-cov.\n")
                        for t in btypes], fontsize=4.8)
    ax.set_ylim(-0.6, len(btypes) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Completeness applicability\nboundary (% error, log)",
                  fontsize=5.4)
    ax.tick_params(labelsize=4.8)
    ax.grid(axis="x", lw=0.3, alpha=0.3)
    add_panel_label(ax, "g", x=-0.95, y=1.24)

    save_fig(fig, name, out_dir=OUT)

    n_per_cell = int(curves.n.iloc[0])
    n_clust = int(curves.n_clusters.iloc[0])
    caption(name,
            "**Supplementary Figure S11 | Set G: robustness to sequencing and "
            "assembly error, and the mechanism behind it.** Five error processes "
            "were injected into otherwise unchanged benchmark genomes at "
            "increasing rates: base substitutions (uniform), base substitutions "
            "with a 2:1 transition/transversion ratio, indels, chimeric contig "
            "joins, and uneven-coverage duplication (assembly redundancy). "
            "**a**, Completeness MAE against error rate, one column per error "
            "type, all four tools. **b**, Completeness signed error (predicted "
            "minus true). **c**, Contamination MAE. **d**, Contamination signed "
            "error. Shaded ribbons are 95 % cluster-bootstrap confidence "
            "intervals. **e**, The mechanism: the fraction of the 9,249 selected "
            "canonical 9-mers actually corrupted at each error rate; base-level "
            "errors destroy k-mers, whereas chimeric joins and duplication leave "
            "the k-mer set essentially intact. **f**, The complementary "
            "marker-gene mechanism: predicted open reading frames as a "
            "percentage of the error-free control, which is what the "
            "annotation-based tools consume. **g**, The measured applicability "
            "boundary for each tool: the lowest error rate at which the upper "
            "bound of the 95 % confidence interval on the completeness MAE "
            "increase over the error-free control first exceeds 1 pp (log axis). "
            "Chimeric joins are omitted because no tool ever crosses that "
            "boundary within the tested range. Uneven-coverage duplication is the "
            "unfavourable result: MAGICC is the only tool that degrades, because "
            "its inputs are absolute k-mer counts. Statistics: cluster bootstrap "
            f"over reference genomes, 2,000 resamples; n = {n_per_cell} genomes "
            f"per error-rate cell from {n_clust} reference-genome clusters. Set G "
            "is deliberately excluded from the pooled five-set benchmark.")


# ---------------------------------------------------------------------------
# S12  CAMI II external benchmark
# ---------------------------------------------------------------------------
CAMI_DATASETS = [("marine", "CAMI II marine"),
                 ("strain_madness", "CAMI II strain-madness")]
COMP_BANDS = ["50-60", "60-70", "70-80", "80-90", "90-95", "95-100"]


def fig_S12():
    name = "Figure_S12"
    print(f"[{name}] CAMI II")
    a = os.path.join(RESULTS, "cami2", "analysis")
    gold = read_tsv(os.path.join(a, "cami2_gold_by_completeness_decile.tsv"))
    coh = read_tsv(os.path.join(a, "cami2_accuracy_by_cohort.tsv"))
    mim = read_tsv(os.path.join(a, "cami2_mimag.tsv"))
    lp = read_tsv(os.path.join(a, "cami2_long_predictions.tsv"))

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 2, left=0.09, right=0.985, top=0.955,
                          bottom=0.065, hspace=0.55, wspace=0.28)

    # a, b -- gold-bin completeness MAE by true-completeness band
    for k, (ds, dlab) in enumerate(CAMI_DATASETS):
        ax = fig.add_subplot(gs[0, k])
        sub = gold[gold.dataset == ds]
        ns = []
        for i, band in enumerate(COMP_BANDS):
            for tool in TOOLS:
                r = sub[(sub.completeness_band == band)
                        & (sub.tool == CAMI_TOOL[tool])]
                if r.empty:
                    continue
                pos = i + (TOOLS.index(tool) - 1.5) * 0.19
                ax.plot([pos], [float(r.comp_mae.iloc[0])], marker=MARKERS[tool],
                        ms=3.0, color=PALETTE[tool],
                        mec=EDGE.get(tool, "0.2"), mew=0.4, ls="none")
            rr = sub[(sub.completeness_band == band)
                     & (sub.tool == CAMI_TOOL["magicc"])]
            ns.append(int(rr.n.iloc[0]) if not rr.empty else 0)
        cat_axis(ax, [f"{b}\n({n:,})" for b, n in zip(COMP_BANDS, ns)],
                 rotation=45, ha="right")
        ax.set_ylabel("Completeness MAE (pp)")
        ax.set_xlabel("True completeness band (%)")
        ax.set_title(f"{dlab}: gold-standard (pure) bins", fontsize=6.8, pad=3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.32)
        if k == 0:
            tool_legend(ax, loc="upper center", ncol=4, fontsize=5.4)
        add_panel_label(ax, "ab"[k], x=-0.19, y=1.16)

    # c, d -- mixed-bin predicted vs true contamination
    mixed = lp[(lp.binset == "mixed") & (lp.all_tools_scored)
               & (lp.leakage_free)]
    mixed_n = {}
    for k, (ds, dlab) in enumerate(CAMI_DATASETS):
        ax = fig.add_subplot(gs[1, k])
        sub = mixed[mixed.dataset == ds]
        for tool in ["deepcheck", "cocopye", "checkm2", "magicc"]:
            t = sub[sub.tool == CAMI_TOOL[tool]]
            kw = dict(s=3.0, alpha=0.35, marker=MARKERS[tool], linewidths=0.0,
                      rasterized=True, color=PALETTE[tool])
            if tool in EDGE:
                kw.update(edgecolors=EDGE[tool], linewidths=0.15)
            ax.scatter(t.true_contamination, t.pred_contamination, **kw)
        mixed_n[ds] = int(sub[sub.tool == CAMI_TOOL["magicc"]].shape[0])
        hi = float(np.nanpercentile(sub.true_contamination, 99.5))
        hi = max(20.0, np.ceil(hi / 10) * 10)
        diag(ax, 0, hi)
        ax.set_xlim(-0.03 * hi, 1.03 * hi)
        ax.set_ylim(-0.03 * hi, 1.03 * hi)
        ax.set_xlabel("True contamination (%)")
        ax.set_ylabel("Predicted contamination (%)")
        ax.set_title(f"{dlab}: constructed mixed bins\n"
                     f"(n = {mixed_n[ds]:,} bins per tool)", fontsize=6.8, pad=3)
        add_panel_label(ax, "cd"[k], x=-0.19, y=1.16)

    # e, f -- MIMAG-inspired false-clean and false-fail at 5 %
    for k, (col, lab) in enumerate((("false_clean_rate_at_5pct",
                                     "False-clean rate at 5 %"),
                                    ("false_fail_rate_at_5pct",
                                     "False-fail rate at 5 %"))):
        ax = fig.add_subplot(gs[2, k])
        cells, ticklab = [], []
        for ds, dlab in CAMI_DATASETS:
            for bs in ("gold", "mixed"):
                cells.append((ds, bs))
                ticklab.append(f"{dlab.replace('CAMI II ', '')}\n{bs}")
        for i, (ds, bs) in enumerate(cells):
            vals_here = []
            for tool in TOOLS:
                r = mim[(mim.dataset == ds) & (mim.binset == bs)
                        & (mim.tool == CAMI_TOOL[tool])]
                if r.empty:
                    continue
                v = r[col].iloc[0]
                if not np.isfinite(v):
                    continue
                vals_here.append(v)
                pos = i + (TOOLS.index(tool) - 1.5) * 0.19
                ax.plot([pos], [float(v)], marker=MARKERS[tool], ms=3.0,
                        color=PALETTE[tool], mec=EDGE.get(tool, "0.2"),
                        mew=0.4, ls="none")
            if not vals_here:
                ax.text(i, 0.5, "not defined\n(no truly\ncontaminated bins)",
                        ha="center", va="center", fontsize=5.0, color="0.45")
        cat_axis(ax, ticklab, rotation=30, ha="right")
        ax.set_ylabel(lab)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(("Truly contaminated bins called clean" if k == 0
                      else "Truly clean bins called contaminated"),
                     fontsize=6.8, pad=3)
        if k == 0:
            tool_legend(ax, loc="upper left", ncol=2, fontsize=5.4)
        add_panel_label(ax, "ef"[k], x=-0.19, y=1.16)

    save_fig(fig, name, out_dir=OUT)

    gold_n = {ds: int(mim[(mim.dataset == ds) & (mim.binset == "gold")
                          & (mim.tool == CAMI_TOOL["magicc"])].n.iloc[0])
              for ds, _ in CAMI_DATASETS}
    PRIM = "primary_in_domain_scoreable_LEAKAGE_FREE"

    def nclust(ds, bs):
        r = coh[(coh.dataset == ds) & (coh.binset == bs) & (coh.cohort == PRIM)
                & (coh.tool == CAMI_TOOL["magicc"])
                & (coh.metric == "completeness")]
        if r.empty:
            raise ValueError(f"no cohort row for {ds}/{bs}")
        return int(r.n_clusters.iloc[0])
    caption(name,
            "**Supplementary Figure S12 | CAMI II external replication: "
            "gold-standard bin completeness, constructed mixed-bin "
            "contamination, and MIMAG-inspired threshold errors.** Truth is "
            "CAMI II's own contig-to-source-genome assignment "
            "(`gsa_mapping.tsv`); only the grouping of contigs into bins is "
            "ours, and the completeness/contamination denominators are the "
            "MAGICC convention throughout. **a**, **b**, Completeness MAE on the "
            "gold-standard (pure, 0 % contamination) bins, by true-completeness "
            "band, for the marine and strain-madness datasets; the per-band bin "
            "count is printed on the axis. Contamination R2 is deliberately not "
            "reported for these bins: the true value is exactly 0 for every bin, "
            "so the total sum of squares is zero and R2 is undefined (R1-m19). "
            "**c**, **d**, Predicted versus true contamination on the "
            "constructed mixed bins, restricted to the leakage-free cohort that "
            "all four tools scored; the dashed line is y = x, and points "
            "collapsing onto the x axis are contamination that is present but "
            "undetected. **e**, **f**, MIMAG-inspired threshold errors at the "
            "5 % contamination boundary, shown as an obligatory pair: the "
            "false-clean rate (truly contaminated bins called clean) and the "
            "false-fail rate (truly clean bins called contaminated). Statistics: "
            "per-bin errors; cluster-bootstrap confidence intervals, bias by "
            "taxonomic distance, detection slopes and paired Hodges-Lehmann "
            "tests are in results/revision/cami2/analysis/; the cluster unit is "
            "the dominant CAMI II source genome. n = "
            f"{gold_n['marine']:,} marine gold bins "
            f"({nclust('marine', 'gold'):,} clusters) and "
            f"{gold_n['strain_madness']:,} strain-madness gold bins "
            f"({nclust('strain_madness', 'gold'):,} clusters); "
            f"{mixed_n['marine']:,} marine mixed bins "
            f"({nclust('marine', 'mixed'):,} clusters) and "
            f"{mixed_n['strain_madness']:,} strain-madness mixed bins "
            f"({nclust('strain_madness', 'mixed'):,} clusters), per tool. "
            "39.4 % of the marine source genomes were present in MAGICC's "
            "training data and are excluded here; the strain-madness dataset has "
            "zero leakage. Bins below the 50 % completeness floor are censored "
            "by the reporting convention, at rates that must be quoted with "
            "their denominator (see Table S14).")


# ---------------------------------------------------------------------------
# S13  Reduced genomes: lineage-relative size effect and MIMAG-threshold impact
# ---------------------------------------------------------------------------
def _parse_ci(s):
    """'[-21.32, -0.56]' -> (-21.32, -0.56)"""
    if not isinstance(s, str) or not s.strip().startswith("["):
        return (np.nan, np.nan)
    lo, hi = s.strip()[1:-1].split(",")
    return float(lo), float(hi)


def fig_S13():
    name = "Figure_S13"
    print(f"[{name}] reduced genomes")
    r = os.path.join(RESULTS, "real_data", "reduced_genome")
    lin = read_tsv(os.path.join(r, "lineage_relative_size_deltas.tsv"))
    mim = read_tsv(os.path.join(r, "mimag_threshold_impact.tsv"))
    rev = read_tsv(os.path.join(r, "reviewer_genera_deltas.tsv"))

    # Canvas width 6.95 in, not 7.2 in. savefig.bbox="tight" saves the union
    # of the canvas and every artist that overflows it, so S13 on a full-width
    # canvas rendered 185.93 mm against the 183 mm Nature Communications cap
    # and the 180 mm working target this build holds every figure to; at 6.95
    # in it renders 179.83 mm. Font sizes are unchanged in points, the canvas
    # was stepped down and re-measured 0.05 in at a time, and 6.90 in is the
    # first step at which a new text collision appears, so this stops one step
    # short of it. No plotted value moves: the artist ledger is byte-identical
    # to the full-width build.
    fig, axes = plt.subplots(2, 2, figsize=(6.95, 6.0))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.20,
                        hspace=0.95, wspace=0.30)

    xs = np.arange(len(lin))
    xlabels = [f"{v}\n(n = {int(n):,}; {int(c)} clusters)"
               for v, n, c in zip(lin.log2_size_vs_phylum_bin, lin.n,
                                  lin.n_family_clusters)]

    # a -- lineage-relative completeness delta
    ax = axes[0, 0]
    lo, hi = zip(*[_parse_ci(v) for v in lin.delta_completeness_ci95])
    errbars(ax, xs, lin.delta_completeness_median.to_numpy(), lo, hi, "magicc",
            ms=3.6)
    hline0(ax)
    cat_axis(ax, xlabels, rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=4.8)
    ax.set_ylabel("MAGICC − CheckM2\ncompleteness (pp)")
    ax.set_xlabel("log2 genome size relative to the phylum median", fontsize=6)
    ax.set_title("Lineage-relative size effect, completeness", fontsize=7, pad=3)
    add_panel_label(ax, "a", x=-0.20, y=1.18)

    # b -- lineage-relative contamination delta + the >=5 % disagreement rate
    ax = axes[0, 1]
    lo, hi = zip(*[_parse_ci(v) for v in lin.delta_contamination_ci95])
    errbars(ax, xs, lin.delta_contamination_median.to_numpy(), lo, hi, "magicc",
            ms=3.6)
    hline0(ax)
    cat_axis(ax, xlabels, rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=4.8)
    ax.set_ylabel("MAGICC − CheckM2\ncontamination (pp)")
    ax.set_xlabel("log2 genome size relative to the phylum median", fontsize=6)
    ax.set_title("Lineage-relative size effect, contamination", fontsize=7, pad=3)
    ax2 = ax.twinx()
    ax2.plot(xs, lin.pct_magicc_cont_ge5_while_checkm2_lt5, ls="none",
             marker="s", ms=3.2, color=AUX["mauve"], mec="0.2", mew=0.4)
    ax2.set_ylabel("MAGICC ≥5 % while CheckM2 <5 % (%)", fontsize=6)
    ax2.tick_params(axis="y", labelsize=5.5)
    ax2.spines["right"].set_visible(True)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    lo0, hi0 = ax.get_ylim()
    ax.set_ylim(lo0, hi0 + 0.42 * (hi0 - lo0))
    ax2.set_ylim(0, 100 * (1 + 0.42))
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.2,
                              color=PALETTE["magicc"], label="median Δ (left axis)"),
                       Line2D([], [], marker="s", ls="none", ms=3.2,
                              color=AUX["mauve"], label="disagreement rate (right axis)")],
              loc="upper left", frameon=False, fontsize=5.2,
              handletextpad=0.3, labelspacing=0.2, borderpad=0.15)
    add_panel_label(ax, "b", x=-0.20, y=1.18)

    # c -- MIMAG high-quality fraction, CheckM2 vs MAGICC
    ax = axes[1, 0]
    groups = ["size_stratified:<1Mb", "size_stratified:1-2Mb",
              "size_stratified:2-3Mb", "size_stratified:3-5Mb",
              "size_stratified:>5Mb", "control_genera(all 8)",
              "reviewer_genera(all 5)"]
    glabel = {"size_stratified:<1Mb": "<1 Mb", "size_stratified:1-2Mb": "1–2 Mb",
              "size_stratified:2-3Mb": "2–3 Mb", "size_stratified:3-5Mb": "3–5 Mb",
              "size_stratified:>5Mb": ">5 Mb",
              "control_genera(all 8)": "8 non-reduced\ncontrol genera",
              "reviewer_genera(all 5)": "5 reduced\nreviewer genera"}
    ticklab = []
    for i, gname in enumerate(groups):
        row = mim[mim.group == gname]
        if row.empty:
            raise ValueError(f"missing MIMAG threshold group {gname}")
        row = row.iloc[0]
        ax.plot([i - 0.13], [row.pct_HQ_by_CheckM2], marker=MARKERS["checkm2"],
                ms=3.4, color=PALETTE["checkm2"], mec="0.2", mew=0.4, ls="none")
        ax.plot([i + 0.13], [row.pct_HQ_by_MAGICC], marker=MARKERS["magicc"],
                ms=3.4, color=PALETTE["magicc"], mec="0.2", mew=0.4, ls="none")
        ax.plot([i - 0.13, i + 0.13], [row.pct_HQ_by_CheckM2, row.pct_HQ_by_MAGICC],
                color="0.6", lw=0.5, zorder=0)
        ticklab.append(f"{glabel[gname]}\n(n = {int(row.n):,})")
    cat_axis(ax, ticklab, rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=4.8)
    ax.set_ylabel("Genomes called high quality (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("MIMAG-inspired high-quality fraction", fontsize=7, pad=3)
    ax.legend(handles=[Line2D([], [], marker=MARKERS["checkm2"], ls="none",
                              ms=3.4, color=PALETTE["checkm2"], label="CheckM2"),
                       Line2D([], [], marker=MARKERS["magicc"], ls="none",
                              ms=3.4, color=PALETTE["magicc"], label="MAGICC")],
              loc="upper left", frameon=False, fontsize=5.4,
              handletextpad=0.3, labelspacing=0.2, borderpad=0.15)
    add_panel_label(ax, "c", x=-0.20, y=1.18)

    # d -- disagreement rates at the MIMAG thresholds
    ax = axes[1, 1]
    for i, gname in enumerate(groups):
        row = mim[mim.group == gname].iloc[0]
        for j, (col, cikey, col_c, mk) in enumerate((
                ("false_fail_rate_cont5_vs_CheckM2", "false_fail_ci95",
                 PALETTE["magicc"], "o"),
                ("HQ_downgrade_rate_vs_CheckM2", "HQ_downgrade_ci95",
                 AUX["mauve"], "s"))):
            lo_, hi_ = _parse_ci(row[cikey])
            y = float(row[col])
            ax.errorbar([i + (j - 0.5) * 0.26], [y],
                        yerr=[[max(y - lo_, 0)], [max(hi_ - y, 0)]],
                        color=col_c, marker=mk, ms=3.2, ls="none", capsize=1.5,
                        elinewidth=0.7, mec="0.2", mew=0.4)
    cat_axis(ax, ticklab, rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=4.8)
    ax.set_ylabel("Rate of disagreement with CheckM2")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Impact at the MIMAG-inspired thresholds", fontsize=7, pad=3)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.2,
                              color=PALETTE["magicc"],
                              label="MAGICC ≥5 % contamination\nwhere CheckM2 says <5 %"),
                       Line2D([], [], marker="s", ls="none", ms=3.2,
                              color=AUX["mauve"],
                              label="CheckM2 high quality\ndowngraded by MAGICC")],
              loc="upper right", frameon=False, fontsize=5.0,
              handletextpad=0.3, labelspacing=0.4, borderpad=0.15)
    add_panel_label(ax, "d", x=-0.20, y=1.18)

    save_fig(fig, name, out_dir=OUT)

    tot_n = int(lin.n.sum())
    tot_c = int(mim[mim.group == "reviewer_genera(all 5)"].n.iloc[0])
    caption(name,
            "**Supplementary Figure S13 | Reduced genomes: the lineage-relative "
            "size effect and its impact at the MIMAG-inspired thresholds.** "
            "These are real catalogue MAGs with no ground truth, so every "
            "quantity below is a *disagreement* between MAGICC and CheckM2, not "
            "an error of either. **a**, Median MAGICC-minus-CheckM2 completeness "
            "against genome size expressed relative to the median size of the "
            "genome's own phylum (log2 ratio), so the effect is not confounded "
            "by which lineages happen to be small. **b**, The same contrast for "
            "contamination (left axis) together with the fraction of genomes in "
            "each bin where MAGICC calls >=5 % contamination while CheckM2 calls "
            "<5 % (right axis, mauve squares). **c**, Fraction of genomes each "
            "tool calls high quality (MIMAG-inspired: >=90 % completeness and "
            "<5 % contamination) for the five size strata, the eight non-reduced "
            "control genera and the five reduced genera raised by the reviewer; "
            "the grey connector links the two tools on the same genomes. "
            "**d**, The two disagreement rates behind panel c. Statistics: "
            "medians with 95 % percentile cluster-bootstrap confidence intervals "
            "over family-level clusters, 5,000 resamples. n = "
            f"{tot_n:,} catalogue MAGs across the five lineage-relative size "
            f"bins (44-154 family clusters per bin) and n = {tot_c} MAGs in the "
            "five reduced reviewer genera. The ground-truthed anchor for this "
            "disagreement is Figure 7b: on genuinely clean Set C-clean "
            "Patescibacteria, MAGICC is the tool in error.")


# ---------------------------------------------------------------------------
# S14  Foodborne pathogen mixtures (the submitted Figure 4), re-run under V5
# ---------------------------------------------------------------------------
PATH_DIR = os.path.join(BENCH, "pathogen_analysis_v5")

SE_CONFIGS = [("100se_5lm", "100 % S.e.\n+ 5 % L.m."),
              ("100se_20lm", "100 % S.e.\n+ 20 % L.m."),
              ("100se_50lm", "100 % S.e.\n+ 50 % L.m."),
              ("100se_100lm", "100 % S.e.\n+ 100 % L.m."),
              ("60se_40lm", "60 % S.e.\n+ 40 % L.m.")]
LM_CONFIGS = [("100lm_5se", "100 % L.m.\n+ 5 % S.e."),
              ("100lm_20se", "100 % L.m.\n+ 20 % S.e."),
              ("100lm_50se", "100 % L.m.\n+ 50 % S.e."),
              ("100lm_100se", "100 % L.m.\n+ 100 % S.e."),
              ("60lm_40se", "60 % L.m.\n+ 40 % S.e.")]


def _pathogen_frames():
    """Per-replicate MAGICC V5 and CheckM2 values for both mixture directions."""
    se = read_tsv(os.path.join(PATH_DIR, "exact_synthetic",
                               "replicate_summary.tsv"))
    se_c2 = read_tsv(os.path.join(PATH_DIR, "exact_synthetic",
                                  "checkm2_per_replicate.tsv"))
    se = se.merge(se_c2[["config", "replicate", "Completeness", "Contamination"]],
                  on=["config", "replicate"], validate="one_to_one")
    se = se.rename(columns={"Completeness": "checkm2_completeness",
                            "Contamination": "checkm2_contamination"})
    lm = read_tsv(os.path.join(PATH_DIR, "exact_synthetic_lm_dominant",
                               "replicate_summary.tsv"))
    for df, cfgs in ((se, SE_CONFIGS), (lm, LM_CONFIGS)):
        have = set(df.config)
        missing = {c for c, _ in cfgs} - have
        if missing:
            raise ValueError(f"pathogen configs missing from disk: {missing}")
        for c in have:
            k = int((df.config == c).sum())
            if k != 10:
                raise ValueError(f"config {c} has {k} replicates, expected 10")
    return se, lm


def _strip_panel(ax, df, cfgs, metric, ylabel, title):
    """Per-replicate strip of MAGICC and CheckM2 against the true value."""
    truth_handle = None
    for i, (cfg, lab) in enumerate(cfgs):
        sub = df[df.config == cfg]
        if sub.empty:
            continue
        true_v = float(sub[f"true_{metric}"].iloc[0])
        h = ax.hlines(true_v, i - 0.38, i + 0.38, color=PALETTE["truth"],
                      lw=1.4, zorder=1)
        truth_handle = h
        for j, (tool, col) in enumerate((("magicc", f"pred_{metric}"),
                                         ("checkm2", f"checkm2_{metric}"))):
            y = sub[col].to_numpy(float)
            x = i + (j - 0.5) * 0.34 + np.linspace(-0.06, 0.06, y.size)
            ax.plot(x, y, ls="none", marker=MARKERS[tool], ms=2.6,
                    color=PALETTE[tool], mec="0.15", mew=0.3, alpha=0.85,
                    zorder=3)
            ax.plot([i + (j - 0.5) * 0.34], [np.mean(y)], marker="_", ms=8,
                    color=PALETTE[tool], mew=1.4, zorder=4)
    cat_axis(ax, [lab for _, lab in cfgs])
    ax.tick_params(axis="x", labelsize=5.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7, pad=3)
    return truth_handle


def fig_S14():
    name = "Figure_S14"
    print(f"[{name}] foodborne pathogen mixtures under V5")
    se, lm = _pathogen_frames()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.14,
                        hspace=0.62, wspace=0.26)

    _strip_panel(axes[0, 0], se, SE_CONFIGS, "completeness",
                 "Completeness (%)",
                 "S. enterica dominant, L. monocytogenes contaminant")
    axes[0, 0].set_ylim(45, 110)
    add_panel_label(axes[0, 0], "a", x=-0.14, y=1.16)

    th = _strip_panel(axes[0, 1], se, SE_CONFIGS, "contamination",
                      "Contamination (%)",
                      "S. enterica dominant, L. monocytogenes contaminant")
    axes[0, 1].set_ylim(-5, 115)
    add_panel_label(axes[0, 1], "b", x=-0.14, y=1.16)

    _strip_panel(axes[1, 0], lm, LM_CONFIGS, "completeness",
                 "Completeness (%)",
                 "L. monocytogenes dominant, S. enterica contaminant")
    axes[1, 0].set_ylim(45, 110)
    add_panel_label(axes[1, 0], "c", x=-0.14, y=1.16)

    _strip_panel(axes[1, 1], lm, LM_CONFIGS, "contamination",
                 "Contamination (%)",
                 "L. monocytogenes dominant, S. enterica contaminant")
    axes[1, 1].set_ylim(-5, 115)
    add_panel_label(axes[1, 1], "d", x=-0.14, y=1.16)

    handles = [Line2D([], [], color=PALETTE["truth"], lw=1.4, label="Truth"),
               Line2D([], [], color=PALETTE["magicc"], marker=MARKERS["magicc"],
                      ls="none", ms=3.2, label="MAGICC (V5)"),
               Line2D([], [], color=PALETTE["checkm2"], marker=MARKERS["checkm2"],
                      ls="none", ms=3.2, label="CheckM2")]
    axes[0, 1].legend(handles=handles, loc="upper left", frameon=False,
                      fontsize=5.6, handletextpad=0.35, labelspacing=0.25,
                      borderpad=0.15)

    save_fig(fig, name, out_dir=OUT)

    def cell(df, cfg, col):
        return float(df.loc[df.config == cfg, col].mean())

    caption(name,
            "**Supplementary Figure S14 | Foodborne pathogen mixtures, re-run "
            "under the released V5 model.** Exact two-contig synthetic mixtures "
            "of *Salmonella enterica* (GCF_001302605.1) and *Listeria "
            "monocytogenes* (GCF_000021185.1); no fragmentation is applied, so "
            "the true completeness and contamination are exact by construction. "
            "**a**, **b**, Completeness and contamination when *S. enterica* is "
            "the dominant genome and *L. monocytogenes* the contaminant. "
            "**c**, **d**, The reciprocal design, *L. monocytogenes* dominant. "
            "Every one of the 10 replicates per configuration is plotted "
            "individually (n = 10 per configuration per tool, 100 mixtures in "
            "total); the horizontal tick is the arithmetic mean and the grey bar "
            "is the exact truth. Statistics: replicate-level values, no "
            "resampling required - the truth is exact and the replicate spread "
            "is the only source of variation. Two results are visible and both "
            "belong in the record. CheckM2's contamination underestimate grows "
            "with the true contamination and is far worse when the dominant "
            "genome is the smaller of the pair: at 100 % L. monocytogenes plus "
            f"100 % S. enterica CheckM2 reports {cell(lm, '100lm_100se', 'checkm2_contamination'):.1f} % "
            f"against a true 100 % while MAGICC reports "
            f"{cell(lm, '100lm_100se', 'pred_contamination'):.1f} %; at 60 % "
            "S. enterica plus 40 % L. monocytogenes CheckM2 reports "
            f"{cell(se, '60se_40lm', 'checkm2_completeness'):.1f} % completeness "
            f"and {cell(se, '60se_40lm', 'checkm2_contamination'):.1f} % "
            "contamination against a true 60 % / 40 %, where MAGICC reports "
            f"{cell(se, '60se_40lm', 'pred_completeness'):.1f} % / "
            f"{cell(se, '60se_40lm', 'pred_contamination'):.1f} %. Against that, "
            "MAGICC over-calls mid-range contamination: at a true 20 % it "
            f"reports {cell(se, '100se_20lm', 'pred_contamination'):.1f} % and at "
            f"a true 50 % it reports {cell(se, '100se_50lm', 'pred_contamination'):.1f} % "
            "with S. enterica dominant. This figure is a controlled two-organism "
            "demonstration only. It supports no claim about the quality of any "
            "public genome database, and the '315 rejected genomes' figure of "
            "the original submission - which came from the superseded V4 "
            "analysis - is withdrawn.")


# ---------------------------------------------------------------------------
# S15  MIMAG-inspired confusion matrices, per set and tool
# ---------------------------------------------------------------------------
def fig_S15():
    name = "Figure_S15"
    print(f"[{name}] MIMAG-inspired confusion matrices")
    cm = read_tsv(os.path.join(RESULTS, "metrics",
                               "ws5.1_mimag_confusion_matrices.tsv"))
    cm = cm[cm.set.isin(LEAKFREE) & cm.tool.isin(TSV_TOOL.values())]
    f1 = read_tsv(os.path.join(RESULTS, "metrics",
                               "ws5.1_mimag_overall_metrics.tsv"))

    fig, axes = plt.subplots(len(SETS), len(TOOLS), figsize=(7.2, 8.8))
    fig.subplots_adjust(left=0.135, right=0.885, top=0.945, bottom=0.055,
                        hspace=0.60, wspace=0.28)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    f1col = next((c for c in f1.columns if "macro" in c.lower()
                  and "f1" in c.lower()), None)

    for ri, (slab, sdir, stitle) in enumerate(SETS):
        for ci, tool in enumerate(TOOLS):
            ax = axes[ri, ci]
            sub = cm[(cm.set == sdir) & (cm.tool == TSV_TOOL[tool])]
            if sub.empty:
                raise ValueError(f"no confusion matrix for {sdir}/{tool}")
            mat = np.full((3, 3), np.nan)
            cnt = np.zeros((3, 3), dtype=int)
            for _, r in sub.iterrows():
                i = MIMAG_ORDER.index(r.true_class)
                j = MIMAG_ORDER.index(r.pred_class)
                mat[i, j] = float(r.row_frac)
                cnt[i, j] = int(r.n)
            im = ax.imshow(mat, cmap=fs.SEQUENTIAL_CMAP, norm=norm, aspect="auto")
            for i in range(3):
                if not np.isfinite(mat[i]).any():
                    ax.text(1, i, "no genomes of this true class",
                            ha="center", va="center", fontsize=4.2, color="0.4")
                for j in range(3):
                    if not np.isfinite(mat[i, j]):
                        continue
                    rgba = im.cmap(im.norm(mat[i, j]))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    ax.text(j, i, f"{mat[i, j]*100:.0f}%\n{cnt[i, j]:,}",
                            ha="center", va="center", fontsize=4.2,
                            color="white" if lum < 0.5 else "0.1")
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(MIMAG_ORDER if ri == len(SETS) - 1
                               else [""] * 3, fontsize=4.8, rotation=30,
                               ha="right")
            ax.set_yticklabels(MIMAG_ORDER if ci == 0 else [""] * 3, fontsize=4.8)
            ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
            ax.grid(which="minor", color="white", lw=0.5)
            ax.tick_params(which="minor", length=0)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_visible(False)
            title = TOOL_LABEL[tool]
            if f1col is not None:
                fr = f1[(f1.set == sdir) & (f1.tool == TSV_TOOL[tool])]
                if not fr.empty and np.isfinite(fr[f1col].iloc[0]):
                    title += f"\nmacro F1 {float(fr[f1col].iloc[0]):.3f}"
            ax.set_title(title, fontsize=5.4, pad=2)
            if ci == 0:
                ax.set_ylabel(f"{stitle.split(' (')[0]}\nTrue class", fontsize=5.6)
                add_panel_label(ax, "abcde"[ri], x=-0.62, y=1.34)
            if ri == len(SETS) - 1:
                ax.set_xlabel("Predicted class", fontsize=5.6)

    cax = fig.add_axes([0.90, 0.30, 0.014, 0.40])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Fraction of the true class (row-normalised)", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    save_fig(fig, name, out_dir=OUT)

    caption(name,
            "**Supplementary Figure S15 | MIMAG-inspired quality-class "
            "confusion matrices for every tool on every leakage-free benchmark "
            "set.** Rows within each matrix are the genome's true class and "
            "columns the class the tool assigns; cells are row-normalised, so "
            "each row sums to 100 %, and both the percentage and the underlying "
            "genome count are printed. Classes are MIMAG-*inspired*: high, "
            ">=90 % completeness and <5 % contamination; medium, >=50 % and "
            "<10 %; low, otherwise. The rRNA and tRNA criteria of the strict "
            "MIMAG definition cannot be evaluated from these assemblies and are "
            "not applied. **a**, Set A. **b**, Set B. **c**, Set C-clean. "
            "**d**, Set D-clean. **e**, Set E. Statistics: n = 1,000 genomes per "
            "set per tool, drawn from 798 (Set A), 803 (Set B), 100 "
            "(Set C-clean), 100 (Set D-clean) and 785 (Set E) reference-genome "
            "clusters; three-class macro F1 with 95 % cluster-bootstrap "
            "confidence intervals is in "
            "results/revision/metrics/ws5.1_mimag_overall_metrics.tsv. Class "
            "balance differs sharply between sets - Set A contains no "
            "low-quality genomes and Set B only two medium-quality ones, which "
            "caps the attainable macro F1 - so matrices must be read within a "
            "set, not across sets.")


# ---------------------------------------------------------------------------
# Captions file
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# S16  Contamination type, taxonomic distance and the CAMI II replication
#      -- the panels of the previous round's Figures 4 and 5 that are not in
#      main-text Figure 4.  Drawn by panels_taxonomy so that the numbers cannot
#      diverge from the main figure.
# ---------------------------------------------------------------------------
def fig_S16():
    import panels_taxonomy as ptx
    import figledger as fl

    fl.require_all()
    # Canvas width 6.75 in, not fl.DOUBLE_COL (7.205 in = 183 mm).
    # savefig.bbox="tight" saves the union of the canvas and every artist that
    # overflows it, so S16 on a full-width canvas rendered 190.17 mm against
    # the 183 mm Nature Communications cap and the 180 mm working target this
    # build holds every figure to; at 6.75 in it renders 178.84 mm. Font sizes
    # are unchanged in points, the canvas was stepped down and re-measured
    # 0.05 in at a time, and no new text collision appears at any width down
    # to 6.70 in. No plotted value moves: the artist ledger is byte-identical
    # to the full-width build.
    fig = plt.figure(figsize=(6.75, 8.3))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.02, 0.80, 1.05],
                          hspace=0.62, wspace=0.30,
                          left=0.150, right=0.985, top=0.955, bottom=0.065)

    ax_a = fig.add_subplot(gs[0, :])
    att = ptx.draw_family_vs_phylum(ax_a, fig="S16", panel="a")
    add_panel_label(ax_a, "a", x=-0.175, y=1.10)

    ax_b, heat = ptx.draw_type_distance_heatmaps(fig, gs[1, :], fig="S16", panel="b")
    add_panel_label(ax_b, "b", x=-0.42, y=1.30)

    ax_c = fig.add_subplot(gs[2, 0])
    slopes = ptx.draw_detection_slope(ax_c, fig="S16", panel="c")
    add_panel_label(ax_c, "c", x=-0.30, y=1.10)

    ax_d = fig.add_subplot(gs[2, 1])
    ptx.draw_cami_distance(ax_d, fig="S16", panel="d")
    add_panel_label(ax_d, "d", x=-0.30, y=1.10)

    fl.report_verification("Figure S16")

    pat = att["Patescibacteriota_families"]
    heli = att["Campylobacterota_Helicobacteraceae"]  # noqa: F841 (kept: ledger key)
    sp = slopes["species"]
    cm, cs = slopes["cami"]["marine"], slopes["cami"]["strain_madness"]
    caption("Figure_S16", (
        "**Supplementary Figure S16 | Contamination type, taxonomic distance and "
        "the CAMI II replication: the panels not shown in Figure 4.** "
        "**a**, Paired attenuation of the completeness difference-in-differences "
        "(DiD) from phylum-level to family-level novelty. All three models (V5, "
        "family holdout, phylum holdout) were run on the same family evaluation "
        "genomes against a common in-distribution control restricted to the 800 "
        "samples / 80 references whose dominant phylum lies outside the "
        "leave-phylum-out panel, with a shared cluster bootstrap (2,000 resamples) "
        "and Benjamini-Hochberg correction. Arrows run from the phylum-holdout DiD "
        "to the family-holdout DiD. The phylum-holdout DiD plotted here is "
        "recomputed on the family evaluation genomes against the common control "
        "and is not the leave-phylum-out DiD of Fig. 4a (trap T6). "
        f"Patescibacteriota attenuates from {pat['phylum']:.2f} to "
        f"{pat['family']:.2f} pp; Helicobacteraceae does not attenuate. "
        + "; ".join(f"{ptx.FAMILY_LABEL[g].replace(chr(10), ' ')} "
                    f"{v['attenuation']:+.2f} pp ({v['pct_retained']:.0f} % retained, "
                    f"p = {v['p']:.3f})" for g, v in att.items()) + ". "
        "Source: `results/revision/holdout_family/family_vs_phylum_did.tsv`. "
        "**b**, Signed contamination error (predicted minus true, pp) over the "
        "fully crossed 3 contamination-type x 6 taxonomic-distance grid of Set F, "
        "one heatmap per tool (1,800 contaminated samples; n = 100 samples / 100 "
        "reference clusters per cell). Contamination types follow CRACOT: "
        "redundant, donor genes added while the acceptor keeps its homologue; "
        "replaced, the same with the genuine acceptor sequence deleted; single, "
        "donor singletons with no acceptor homologue. Colour is a diverging "
        "blue-to-red ramp centred at zero; blue is under-calling, red "
        "over-calling, and every cell additionally prints its value so the panel "
        "does not depend on colour. All three annotation- and marker-based tools "
        "over-call redundant contamination and under-call the two non-redundant "
        "types; MAGICC is the only type-invariant tool, and it loses outright on "
        "redundant contamination. Attribution test: MAGICC's advantage over "
        "CheckM2 in absolute contamination "
        f"error moves from {heat['adv_redundant']:+.2f} pp on redundant "
        f"contamination (MAGICC loses) to {heat['adv_nonredundant']:+.2f} pp on "
        "the two non-redundant classes, paired by reference genome (n = 100 pairs, "
        f"two-sided Wilcoxon p = {heat['p'] * 1e8:.1f} x 10^-8), so the advantage "
        "is created entirely by the non-redundant classes. Source: "
        "`results/revision/set_F/set_F_cells_type_x_distance.tsv`, "
        "`results/revision/set_F/set_F_attribution.tsv`. "
        "**c**, MAGICC detection slope (ordinary least squares of predicted on "
        "true contamination; 1.0 = perfect detection, 0 = blind) against distance, "
        "shown per contamination type because pooling conceals the worst cell. At "
        "species distance the three types span "
        f"{sp['replaced'][0]:+.3f} [{sp['replaced'][1]:.3f}, {sp['replaced'][2]:.3f}] "
        f"(replaced, indistinguishable from zero), {sp['single'][0]:+.3f} "
        f"[{sp['single'][1]:.3f}, {sp['single'][2]:.3f}] (single) and "
        f"{sp['redundant'][0]:+.3f} [{sp['redundant'][1]:.3f}, "
        f"{sp['redundant'][2]:.3f}] (redundant), so the correct species range is "
        f"-0.01 to 0.18, not the pooled {sp['pooled']:+.2f} alone. Grey dashes, "
        "pooled over types. Open markers, the same quantity measured "
        "independently on CAMI II's well-controlled species cell (marine "
        f"{cm[0]:.3f} [{cm[1]:.3f}, {cm[2]:.3f}], n = 147 bins / 85 clusters; "
        f"strain-madness {cs[0]:.3f} [{cs[1]:.3f}, {cs[2]:.3f}], n = 372 / 205); "
        "they are drawn with a small horizontal offset inside the species column "
        "for legibility. Source: "
        "`results/revision/set_F/set_F_cells_type_x_distance.tsv`, "
        "`results/revision/set_F/set_F_marginals.tsv`, "
        "`results/revision/cami2/analysis/cami2_wellcontrolled_by_distance.tsv`. "
        "**d**, External replication of the distance gradient. MAGICC signed "
        "contamination error against distance on Set F (grey crosses, our "
        "simulation) and on the two CAMI II short-read datasets, whose genome "
        "selection, read simulation, assembly and gold standard are all CAMI's - "
        "only the grouping of contigs is ours. The cohort is leakage-free (marine "
        "39.35 % of source genomes leaked and excluded; strain-madness 0 %), "
        "in-domain and >=50 % completeness: marine n = 195/146/168/146/161/170 "
        "bins over 94/75/85/69/90/96 dominant-genome clusters, strain-madness "
        "n = 375 bins per distance over 205/163/178/155/153/162 clusters. Error "
        "bars are 95 % percentile cluster-bootstrap CIs (2,000 resamples). Source: "
        "`results/revision/cami2/analysis/cami2_mixed_by_distance.tsv`, "
        "`results/revision/cami2/analysis/cami2_setF_comparison.tsv`, "
        "`results/revision/set_F/set_F_marginals.tsv`. "
        "Errors are in percentage points (pp) throughout."))
    save_fig(fig, "Figure_S16", out_dir=OUT)


# ---------------------------------------------------------------------------
# S17  Error robustness and recalibration -- the panels of the previous round's
#      Figures 6 and 7 that are not in main-text Figure 5.
# ---------------------------------------------------------------------------
def fig_S17():
    import panels_realdata as prd

    D6 = prd.load6()
    D7 = prd.load7()
    # Canvas width 7.15 in, not prd.DOUBLE_COL (7.205 in = 183 mm).
    # savefig.bbox="tight" saves the union of the canvas and every artist that
    # overflows it, so S17 on a full-width canvas rendered 180.05 mm against
    # the 183 mm Nature Communications cap and the 180 mm working target this
    # build holds every figure to; at 7.15 in it renders 178.69 mm. Font sizes
    # are unchanged in points, the canvas was stepped down and re-measured
    # 0.05 in at a time, and no text collision appears at any width down to
    # 7.10 in. No plotted value moves: the artist ledger is byte-identical to
    # the full-width build.
    fig = plt.figure(figsize=(7.15, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.34,
                          left=0.085, right=0.985, top=0.895, bottom=0.085)

    ax_a1, ax_a2 = prd.draw_orf_cohorts(fig, gs[0, 0], D6)
    add_panel_label(ax_a1, "a", x=-0.42, y=1.24)

    ax_b = fig.add_subplot(gs[0, 1])
    b = prd.draw_dose_substitution(ax_b, D6)
    add_panel_label(ax_b, "b", x=-0.20, y=1.24)

    ax_c = fig.add_subplot(gs[1, 0])
    c = prd.draw_dose_indel(ax_c, D6)
    add_panel_label(ax_c, "c", x=-0.20, y=1.24)

    ax_d = fig.add_subplot(gs[1, 1])
    dd = prd.draw_recalibration(ax_d, D7)
    add_panel_label(ax_d, "d", x=-0.34, y=1.24)

    fig.legend(handles=prd.tool_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.4,
               handletextpad=0.35, columnspacing=1.4)

    prd.verification_report()
    orf, mino = D6["orf"], D6["mino"]
    d0, d1, d2 = dd["drows"]
    e_lo = int(round(float(f"{D6['cons_lo']:e}".split('e')[1])))
    e_hi = int(round(float(f"{D6['cons_hi']:e}".split('e')[1])))
    caption("Figure_S17", (
        "**Supplementary Figure S17 | Error robustness and recalibration: the "
        "panels not shown in Figure 5.** "
        "**a**, The Meslier MOCK1 mock community separated by per-base accuracy "
        f"rather than by fragmentation. Left, the {int(orf['n'].iloc[0])} "
        "leakage-free bins with true completeness >=50 % from the "
        f"{int(orf['n_clusters'].iloc[0])} ORF-intact organisms; right, the "
        f"{int(mino['n'].iloc[0])} bins of the indel-dense MinION assembly "
        f"({int(mino['n_clusters'].iloc[0])} organisms). MinION carries "
        f"{D6['minion_indel']:.2f} indels per kb, against "
        f"{D6['low_indel_lo']:.2f}-{D6['low_indel_hi']:.2f} for the "
        f"{D6['n_low']} PacBio/Illumina/MGI assemblies. The {D6['n_ion']} Ion "
        f"Torrent assemblies carry a comparable indel load "
        f"({D6['ion_indel_lo']:.2f}-{D6['ion_indel_hi']:.2f}) but are far more "
        f"fragmented (median bin N50 at most {D6['ion_n50_hi'] / 1e3:,.1f} kb), so "
        "MinION is the only assembly that combines a high indel rate with long "
        f"contigs (median bin N50 {D6['minion_n50'] / 1e3:,.0f} kb) and the only "
        "one whose ORFs are measurably broken: CheckM2's own output gives it "
        f"median coding density {D6['minion_cd']:.3f} and median gene length "
        f"{D6['minion_gl']:.0f} bp, against {D6['orf_cd_lo']:.3f}-"
        f"{D6['orf_cd_hi']:.3f} for the six ORF-intact assemblies. Filled "
        "markers, completeness MAE; open markers, signed completeness bias; bars, "
        "95 % cluster-bootstrap confidence intervals clustered on reference "
        f"organism. On ORF-intact assemblies MAGICC "
        f"({orf.loc['magicc', 'comp_mae']:.2f} pp) and CheckM2 "
        f"({orf.loc['checkm2', 'comp_mae']:.2f} pp) are not significantly "
        "different (paired two-sided Wilcoxon on per-bin absolute errors, "
        f"{D6['d_ns']:+.2f} pp [{D6['ci_ns'][0]:+.2f}, {D6['ci_ns'][1]:+.2f}], "
        f"p = {D6['p_ns']:.3f}); the large gap is confined to the indel-dense "
        f"MinION cohort, where the protein-based tools collapse (CheckM2 "
        f"{mino.loc['checkm2', 'comp_mae']:.2f} pp and DeepCheck "
        f"{mino.loc['deepcheck', 'comp_mae']:.2f} pp, with biases "
        f"{D6['minion_bias_c2']:+.2f} and {D6['minion_bias_dc']:+.2f} pp) because "
        "frameshifts destroy open reading frames. The claim this panel supports "
        "is robustness to indel-dense long-contig assemblies, not a blanket "
        "accuracy advantage. Source: "
        "`results/revision/real_data/meslier/metrics_by_cohort.tsv`, "
        "`.../paired_tests.tsv`, `.../assembly_sequence_qc.tsv`. "
        "**b**, **c**, Set G dose-response for uniform per-base substitutions "
        "(**b**) and indels (**c**). 1,920 assemblies = 80 held-out reference "
        "genomes x 24 error arms, paired within reference: one base assembly per "
        "reference, every arm applied to that same assembly, so truth, "
        "fragmentation realisation and contaminants are identical across arms. "
        "Points are completeness MAE and error bars 95 % percentile "
        "cluster-bootstrap confidence intervals (2,000 iterations) over the 80 "
        "references; n = 80 per dose. Vertical dashed lines mark the measured "
        "1-pp applicability boundary, defined as the rate at which the upper 95 % "
        "confidence limit of the paired degradation first exceeds 1 pp: in **b** "
        f"MAGICC {b['b_magicc']:.2f} % (red) and CheckM2 {b['b_checkm2']:.2f} % "
        "(blue); in **c** MAGICC "
        f"{c['b_magicc']:.2f} % against CheckM2 {c['b_checkm2']:.3f} %, 110x "
        "lower. The grey band in **b** is the per-base error rate of real "
        f"assemblies: assembly consensus error 10^{e_lo}-10^{e_hi} per base, and "
        f"nominal Illumina Q30 raw-base error {b['illumina_pct']:g} %, so a real "
        f"assembly sits {b['illumina_margin']:.1f}x (Illumina raw base) to about "
        "1,000x (short-read consensus) below MAGICC's boundary. The third arm of "
        "the same experiment, uneven-coverage duplication, is Fig. 5c. Source: "
        "`results/revision/set_G/set_G_curves.tsv`, "
        "`.../set_G_paired_degradation.tsv`, `.../set_G_boundary.tsv`, "
        "`.../set_G_real_world_context.tsv`. "
        "**d**, The size correction does not transfer to novel lineages. Signed "
        "completeness bias of the leave-phylum-out holdout model on three unseen "
        "phyla: before recalibration (grey cross), after a size-conditioned "
        f"recalibrator fitted blind on {dd['n_fitB']:,} in-distribution "
        "predictions of the same model and transferred to the unseen phylum "
        "(brown square), and after an oracle recalibrator fitted on the held-out "
        "lineage itself (red circle); n about 1,000 genomes per group. Blind "
        f"transfer removes only {d0[4]:.1f} % of the Patescibacteriota bias "
        f"({d0[1]:+.2f} to {d0[2]:+.2f} pp) and {d1[4]:.1f} % of the DPANN bias "
        f"({d1[1]:+.2f} to {d1[2]:+.2f} pp), while the oracle removes "
        f"{d0[5]:.1f} % and {d1[5]:.1f} % "
        f"(residual bias {d0[3]:+.2f} and {d1[3]:+.2f} pp). Blind transfer also "
        f"harms a lineage that needed no correction (Campylobacterota "
        f"{d2[1]:+.2f} to {d2[2]:+.2f} pp). The information needed to correct the "
        "error is present in the observables, but the mapping is "
        "lineage-specific: the same lineage-specificity that breaks "
        "generalization also breaks the fix. Source: "
        "`results/revision/real_data/reduced_genome/mitigation/"
        "mitigation_headline.json`, derived from `.../recalibration_ceiling.tsv` "
        "and `.../recalibration_novel_lineage.tsv`. "
        "Errors are in percentage points (pp) throughout."))
    save_fig(fig, "Figure_S17", out_dir=OUT)


PANEL_LETTERS = set("abcdefgh")


def _tidy_caption(text: str) -> str:
    """the internal build contract 2.1: bold only for the leading figure number and panel letters.

    The registered captions open with ``**Supplementary Figure SN | Title.**``
    and mark panel letters with ``**a**``.  This rewrites the opening to
    ``**Figure SN.** Title.`` and strips every other bold span, leaving the
    single-letter panel markers alone.
    """
    m = re.match(r"\*\*Supplementary Figure (S\d+) \| (.*?)\*\*\s*", text, flags=re.S)
    if m:
        head = f"**Figure {m.group(1)}.** {m.group(2).strip()}"
        rest = text[m.end():]
    else:
        head, rest = "", text
    seg = rest.split("**")
    if len(seg) % 2 == 0:
        raise ValueError("unbalanced bold markers in caption: " + rest[:80])
    out = []
    for i, piece in enumerate(seg):
        if i % 2 == 1 and piece in PANEL_LETTERS:
            out.append("**" + piece + "**")
        else:
            out.append(piece)
    rest = "".join(out)
    return (head + " " + rest).strip() if head else rest.strip()


def write_captions():
    path = os.path.join(OUT, "captions_supp.md")
    order = [f"Figure_S{i}" for i in range(1, 18)]
    missing = [k for k in order if k not in CAPTIONS]
    if missing:
        raise RuntimeError(f"no caption registered for {missing}")
    lines = [
        "# Supplementary figure captions",
        "",
        "MAGICC, Nature Communications resubmission.",
        "",
        "Generated by `nature_communications/resubmission3/scripts/"
        "make_supp_figures.py`; every number below is read back from the result "
        "file that the corresponding panel is plotted from. Panel letters are "
        "bold lowercase and no other emphasis is used. Every tool comparison "
        f"uses the original submission's palette (MAGICC {PALETTE['magicc']} "
        f"red, CheckM2 {PALETTE['checkm2']} blue, CoCoPyE {PALETTE['cocopye']} "
        f"green, DeepCheck {PALETTE['deepcheck']} purple; reference series "
        f"{PALETTE['truth']} grey, GUNC {PALETTE['gunc']} mauve, holdout models "
        f"{PALETTE['holdout']} brown), with marker shape and line style as "
        "redundant cues: MAGICC circle and solid line, CheckM2 triangle and "
        "long dash, CoCoPyE diamond and dots, DeepCheck inverted triangle and "
        "dash-dot. Series that encode something other than a tool - the five "
        "Set G error processes in S11e-g and the contamination types in "
        "S16b-c - are drawn in auxiliary hues chosen to sit far from all four "
        "tool hues. The palette is not colour-vision-deficiency safe and is not "
        "claimed to be: CheckM2 and DeepCheck are a blue/purple pair whose "
        f"CIE76 Delta-E falls to {fs.delta_e(PALETTE['checkm2'], PALETTE['deepcheck'], 'protanopia'):.2f} "
        "under protanopic simulation, and MAGICC and CoCoPyE a red/green pair "
        f"that falls to {fs.delta_e(PALETTE['magicc'], PALETTE['cocopye'], 'deuteranopia'):.2f} "
        "under deuteranopic simulation, against a screening threshold of 15. No "
        "panel may therefore be read by colour alone, and none needs to be: "
        "every series is directly labelled in its panel legend and every "
        "heatmap cell prints its own value. The measured Delta-E of every pair "
        "under normal, deuteranopic, protanopic and tritanopic simulation, the "
        "panels in which each flagged pair co-occurs, and the redundant cue "
        "that carries it, are in `figures/palette_cvd_check.tsv` and "
        "`figures/colour_statement.md`. Errors are in percentage points (pp). "
        "\"MIMAG-inspired\" thresholds are used throughout because the rRNA/tRNA "
        "criteria of the strict MIMAG definition cannot be evaluated from these "
        "assemblies. R2 is the coefficient of determination, never a squared "
        "Pearson correlation, and is omitted with a recorded reason wherever the "
        "true value has (near-)zero variance.",
        "",
        "---",
        "",
    ]
    for k in order:
        lines.append(f"## {k}")
        lines.append("")
        lines.append(_tidy_caption(CAPTIONS[k]))
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.basename(path)} ({len(order)} captions)")


# ---------------------------------------------------------------------------
def main():
    print(f"Output directory: {OUT}")
    fig_S1_to_S5()
    fig_S6()
    fig_S7()
    fig_S8()
    fig_S9()
    fig_S10()
    fig_S11()
    fig_S12()
    fig_S13()
    fig_S14()
    fig_S15()
    fig_S16()
    fig_S17()
    write_captions()
    made = sorted(f for f in os.listdir(OUT) if f.endswith((".pdf", ".png")))
    print(f"\n{len(made)} figure files written to {OUT}")
    expect = {f"Figure_S{i}.{ext}" for i in range(1, 18) for ext in ("pdf", "png")}
    missing = expect - set(made)
    if missing:
        raise RuntimeError(f"missing outputs: {sorted(missing)}")
    print("All supplementary figures S1-S17 generated successfully.")


if __name__ == "__main__":
    main()
