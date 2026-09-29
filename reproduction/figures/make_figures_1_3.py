#!/usr/bin/env python3
"""Render resubmission7 main Figures1–3 from frozen verified sources.

Figure1 contains the comparator-only motivation (a–e); Figure2 is an editable
workflow; Figure3 contains the five-set accuracy/classification benchmark.
Scientific source identifiers are preserved. Captions are exported separately
from canonical Markdown by sync_canonical_captions.py.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import (  # noqa: E402
    ARROW_INK,
    BENCH,
    DENOM_NOTE,
    EDGE,
    FIG_DIR,
    MARKERS,
    PALETTE,
    PHASE_ACCENT,
    PHASE_BAND,
    RESULTS,
    TOOL_LABEL,
    add_panel_label,
    apply_style,
    bar_colour,
    delta_e,
    diag,
    hline0,
    readable_ink,
    save_fig,
    tint,
)
import figvalues  # noqa: E402

apply_style()

# ---------------------------------------------------------------------------
# Paths (repointed from create_figures_v5.py)
# ---------------------------------------------------------------------------
BASE = BENCH
MOTIV = os.path.join(BASE, "motivating_v2")
METRICS = os.path.join(RESULTS, "metrics")
OUT_DIR = FIG_DIR
DRAFT_DIR = os.path.join(
    "/tmp/claude-1001/-media-Data-1-tianrm-projects-magicc2",
    "57115812-63e1-4d54-a7fa-5cec7e0f00c8", "scratchpad", "fig_draft")

DRAFT = False

# Result files that must exist
F_WIDE = os.path.join(METRICS, "definitive_table_5set_wide.tsv")
F_LONG = os.path.join(METRICS, "definitive_table_5set.tsv")
F_MIMAG = os.path.join(METRICS, "definitive_mimag.tsv")
F_THRESH = os.path.join(METRICS, "definitive_thresholds.tsv")
F_S2 = os.path.join(METRICS, "ws5.5_table_S2_rebuilt.tsv")
F_REL = os.path.join(METRICS, "ws5.3_signed_errors_by_relatedness.tsv")

# ---------------------------------------------------------------------------
# Benchmark panel definitions
# ---------------------------------------------------------------------------
MOTIV_SETS = [("A", "motivating_v2_set_A", "set_A"),
              ("B", "motivating_v2_set_B", "set_B"),
              ("C", "motivating_v2_set_C", "set_C")]
MOTIV_TOOLS = ["checkm2", "cocopye", "deepcheck"]

BENCH_SETS = [("Set A", "set_A_v2", "set_A_v2"),
              ("Set B", "set_B_v2", "set_B_v2"),
              ("Set C-clean", "set_C_clean", "set_C_clean"),
              ("Set D-clean", "set_D_clean", "set_D_clean"),
              ("Set E", "set_E", "set_E")]
BENCH_TOOLS = ["magicc_v5", "checkm2", "cocopye", "deepcheck"]

# Trap T1: sets F and G are deliberately unregistered and must never appear.
FORBIDDEN_SETS = ("set_F", "set_G")

_DISCREPANCIES: list[str] = []


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def require(path: str) -> str:
    """Fail loudly on a missing input."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"required input file is missing: {path}")
    return path


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(require(path), sep="\t")


def load_predictions(set_dir: str, tool: str) -> pd.DataFrame:
    """Predictions merged with the ground truth in `metadata.tsv`.

    Truth always comes from `metadata.tsv` (WS5 verified that the truth columns
    carried inside every prediction file are identical to it, max deviation 0.0).
    """
    fname = "magicc_v5_predictions.tsv" if tool == "magicc_v5" else f"{tool}_predictions.tsv"
    pred = read_tsv(os.path.join(set_dir, fname))
    meta = read_tsv(os.path.join(set_dir, "metadata.tsv"))
    for col in ("genome_id", "pred_completeness", "pred_contamination"):
        if col not in pred.columns:
            raise KeyError(f"{set_dir}/{fname} lacks column {col}")
    out = pred[["genome_id", "pred_completeness", "pred_contamination"]].merge(
        meta[["genome_id", "true_completeness", "true_contamination"]],
        on="genome_id", validate="1:1")
    if len(out) != len(pred) or len(out) != len(meta):
        raise ValueError(f"genome_id mismatch between predictions and metadata in {set_dir}")
    return out


def abs_err(df: pd.DataFrame, metric: str) -> np.ndarray:
    return np.abs(df[f"pred_{metric}"].to_numpy() - df[f"true_{metric}"].to_numpy())


def check_value(label: str, recomputed: float, recorded: float, tol: float = 0.01) -> float:
    """Compare a recomputed number with the recorded one; the file value wins."""
    if not np.isfinite(recorded):
        return recorded
    if abs(recomputed - recorded) > tol:
        msg = (f"{label}: recomputed {recomputed:.4f} vs recorded {recorded:.4f} "
               f"(delta {recomputed - recorded:+.4f}) -> using the recorded value")
        _DISCREPANCIES.append(msg)
        print("  ! " + msg)
    return recorded


# Kept for call-site compatibility.  Box and bar FILLS now come from the
# original submission's pastel companions (figstyle.BAR_PALETTE, reached
# through figstyle.bar_colour), not from an ad-hoc tint of the line colour.
LIGHTEN: "dict[str, float]" = {}


def lighten(color: str, f: float = 0.58) -> tuple:
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def tool_edge(tool: str) -> str:
    return EDGE.get(tool, PALETTE[tool])


def box_series(ax, positions, arrays, tool, width):
    """Box (IQR) with 5th-95th percentile whiskers, no fliers, tool-coloured."""
    col = PALETTE[tool]
    ec = tool_edge(tool)
    ax.boxplot(
        arrays, positions=positions, widths=width, whis=(5, 95),
        showfliers=False, patch_artist=True, manage_ticks=False,
        boxprops=dict(facecolor=bar_colour(tool), edgecolor=ec, linewidth=0.4),
        medianprops=dict(color=ec, linewidth=0.7),
        whiskerprops=dict(color=ec, linewidth=0.4),
        capprops=dict(color=ec, linewidth=0.4),
        zorder=2,
    )


def point_ci(ax, x, value, lo, hi, tool, ms=3.4, zorder=6, filled=True):
    """Point estimate with its 95 % cluster-bootstrap CI, tool marker shape."""
    col = PALETTE[tool]
    ec = tool_edge(tool)
    yerr = None
    if np.isfinite(lo) and np.isfinite(hi):
        yerr = np.array([[max(0.0, value - lo)], [max(0.0, hi - value)]])
    ax.errorbar([x], [value], yerr=yerr, fmt=MARKERS[tool], ms=ms,
                mfc=col if filled else "white", mec=ec, mew=0.6,
                ecolor=ec, elinewidth=0.8, capsize=1.6, capthick=0.6,
                zorder=zorder, linestyle="none")


def tool_handles(tools, ms=3.4):
    return [Line2D([], [], marker=MARKERS[t], color=PALETTE[t],
                   markerfacecolor=PALETTE[t], markeredgecolor=tool_edge(t),
                   markeredgewidth=0.6, markersize=ms, linestyle="none",
                   label=TOOL_LABEL[t]) for t in tools]


def scatter_tool(ax, x, y, tool, s=8, alpha=0.35):
    """Predicted-vs-true scatter, tool colour + tool marker shape."""
    kw = dict(s=s, alpha=alpha, color=PALETTE[tool], marker=MARKERS[tool],
              label=TOOL_LABEL[tool], edgecolors="none", rasterized=True)
    if tool in EDGE:                      # yellow needs a dark outline
        kw.update(edgecolors=EDGE[tool], linewidths=0.15, alpha=min(1.0, alpha + 0.10))
    ax.scatter(x, y, **kw)


def _save(fig, name):
    if DRAFT:
        os.makedirs(DRAFT_DIR, exist_ok=True)
        p = os.path.join(DRAFT_DIR, f"{name}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"  draft saved {p}")
        return [p]
    return save_fig(fig, name, out_dir=OUT_DIR)


# ---------------------------------------------------------------------------
# Result-file accessors
# ---------------------------------------------------------------------------
def s2_row(s2: pd.DataFrame, set_name: str, tool: str, metric: str) -> pd.Series:
    r = s2[(s2["set"] == set_name) & (s2["tool"] == tool) & (s2["metric"] == metric)]
    if len(r) != 1:
        raise ValueError(f"{len(r)} rows in ws5.5_table_S2_rebuilt.tsv for "
                         f"{set_name}/{tool}/{metric} (expected 1)")
    return r.iloc[0]


def wide_row(w: pd.DataFrame, set_name: str, tool: str) -> pd.Series:
    r = w[(w["set"] == set_name) & (w["tool"] == tool)]
    if len(r) != 1:
        raise ValueError(f"{len(r)} rows in definitive_table_5set_wide.tsv for "
                         f"{set_name}/{tool} (expected 1)")
    return r.iloc[0]


def mimag_row(m: pd.DataFrame, set_name: str, tool: str) -> pd.Series:
    r = m[(m["set"] == set_name) & (m["tool"] == tool)]
    if len(r) == 0:
        raise ValueError(f"no rows in definitive_mimag.tsv for {set_name}/{tool}")
    if r["macro_f1"].nunique() != 1:
        raise ValueError(f"macro F1 not constant across classes for {set_name}/{tool}")
    return r.iloc[0]


def thresh_row(t: pd.DataFrame, set_name: str, tool: str) -> pd.Series:
    r = t[(t["set"] == set_name) & (t["tool"] == tool)
          & (t["criterion"] == "contamination") & (t["threshold"] == 5.0)]
    if len(r) != 1:
        raise ValueError(f"{len(r)} rows in definitive_thresholds.tsv for "
                         f"{set_name}/{tool} at the 5 % contamination boundary")
    return r.iloc[0]


def rel_row(rel: pd.DataFrame, stratum: str, tool: str) -> pd.Series:
    r = rel[(rel["set"] == "set_E") & (rel["metric"] == "contamination")
            & (rel["stratum"] == stratum) & (rel["tool"] == tool)]
    if len(r) != 1:
        raise ValueError(f"{len(r)} rows in ws5.3_signed_errors_by_relatedness.tsv "
                         f"for set_E/{stratum}/{tool}")
    return r.iloc[0]


# =========================================================================
# Schematic: Figure 2 panel a  (verbatim from create_figures_v5.py)
# =========================================================================
def _draw_figure1_panel_a(ax):
    """Draw the motivating sets schematic on the given axes (black/grey only)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.0)
    ax.axis("off")

    bw = 2.8
    bh = 2.2
    by = 0.6
    positions = [(0.3, by), (3.6, by), (6.9, by)]
    set_info = [
        ("M1 (completeness gradient)", "comp: 50-100%  cont: 0%", "1,000"),
        ("M2 (contamination gradient)", "comp: 100%  cont: 0-80%", "1,000"),
        ("M3 (realistic mix)", "comp: 50-100%  cont: 0-100%", "1,000"),
    ]

    for (x, y), (title, desc, n) in zip(positions, set_info):
        rect = FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.1",
                              facecolor="#f0f0f0", edgecolor="black", linewidth=0.7)
        ax.add_patch(rect)
        ax.text(x + bw / 2, y + bh - 0.15, title, ha="center", va="top",
                fontsize=7.2, fontweight="bold", color="black")
        ax.text(x + bw / 2, y + bh - 0.50, desc,
                ha="center", va="top", fontsize=6, color="#333333")

    gradient_y = by + 0.55
    visual_h = 0.40

    set_a_labels = [50, 60, 70, 80, 90, 100]
    for i, (alpha, label) in enumerate(zip([0.15, 0.30, 0.45, 0.60, 0.80, 1.0], set_a_labels)):
        rx = 0.6 + i * 0.38
        rect = Rectangle((rx, gradient_y), 0.34, visual_h,
                         facecolor="black", alpha=alpha, edgecolor="#999999",
                         linewidth=0.3)
        ax.add_patch(rect)
        ax.text(rx + 0.17, gradient_y + visual_h + 0.06, str(label),
                ha="center", va="bottom", fontsize=5, color="#333333")

    set_b_labels = [0, 20, 40, 60, 80]
    for i, (alpha, label) in enumerate(zip([0.0, 0.25, 0.5, 0.75, 1.0], set_b_labels)):
        rx = 3.9 + i * 0.44
        rect = Rectangle((rx, gradient_y), 0.40, visual_h,
                         facecolor="black", alpha=max(0.05, alpha),
                         edgecolor="#999999", linewidth=0.3)
        ax.add_patch(rect)
        ax.text(rx + 0.20, gradient_y + visual_h + 0.06, str(label),
                ha="center", va="bottom", fontsize=5, color="#333333")

    rng = np.random.RandomState(42)
    for _ in range(30):
        cx = 7.2 + rng.uniform(0, 2.2)
        cy = gradient_y + rng.uniform(0, visual_h)
        shade = rng.choice(["black", "#666666", "#999999"])
        ax.plot(cx, cy, "o", markersize=2, color=shade, alpha=0.5)

    for (x, y), (_, _, n) in zip(positions, set_info):
        ax.text(x + bw / 2, y + 0.10, f"n = {n}", ha="center", va="bottom",
                fontsize=5.4, color="#555555", style="italic")


# =========================================================================
# Schematic: Figure 3 panel a  (same drawing code; Sets C/D relabelled to the
# test-reference rebuilds, which is the only scientific change)
# =========================================================================
def _draw_figure3_panel_a(ax):
    """Draw the test-reference benchmark sets schematic, Figure 2a style."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    set_info = [
        ("Set A\n(completeness gradient)", "comp: 50-100%\ncont: 0%", "1,000", "798"),
        ("Set B\n(contamination gradient)", "comp: 100%\ncont: 0-80%", "1,000", "803"),
        ("Set C-clean\n(Patescibacteriota)", "held-out test split\nmixed quality", "1,000", "100"),
        ("Set D-clean\n(Archaea)", "held-out test split\nmixed quality", "1,000", "100"),
        ("Set E\n(realistic mix)", "comp: 50-100%\ncont: 0-100%", "1,000", "785"),
    ]

    box_w = 1.72
    box_h = 1.8
    gap = 0.30
    total_w = 5 * box_w + 4 * gap
    start_x = (10 - total_w) / 2
    by = 0.3
    positions = [start_x + i * (box_w + gap) for i in range(5)]

    for i, (title, desc, n, ncl) in enumerate(set_info):
        bx = positions[i]
        rect = FancyBboxPatch((bx, by), box_w, box_h,
                              boxstyle="round,pad=0.08",
                              facecolor="#f0f0f0", edgecolor="black",
                              linewidth=0.7)
        ax.add_patch(rect)
        ax.text(bx + box_w / 2, by + box_h - 0.06, title,
                ha="center", va="top", fontsize=5.6, fontweight="bold",
                color="black")
        ax.text(bx + box_w / 2, by + box_h - 0.66, desc,
                ha="center", va="top", fontsize=4.8, color="#333333")
        ax.text(bx + box_w / 2, by + 0.04, f"n = {n} | {ncl} clusters",
                ha="center", va="bottom", fontsize=4.6, color="#555555",
                style="italic")

    gradient_y = by + 0.36
    visual_h = 0.26
    margin_inner = 0.12
    inner_w = box_w - 2 * margin_inner

    bx_a = positions[0]
    for j, alpha in enumerate([0.15, 0.30, 0.45, 0.60, 0.80, 1.0]):
        rw = inner_w / 6
        rx = bx_a + margin_inner + j * rw
        rect = Rectangle((rx, gradient_y), rw * 0.88, visual_h,
                         facecolor="black", alpha=alpha, edgecolor="#999999",
                         linewidth=0.3)
        ax.add_patch(rect)

    bx_b = positions[1]
    for j, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        rw = inner_w / 5
        rx = bx_b + margin_inner + j * rw
        rect = Rectangle((rx, gradient_y), rw * 0.88, visual_h,
                         facecolor="black", alpha=max(0.05, alpha),
                         edgecolor="#999999", linewidth=0.3)
        ax.add_patch(rect)

    rng = np.random.RandomState(42)
    for bx_s in (positions[2], positions[3], positions[4]):
        for _ in range(25):
            cx = bx_s + margin_inner + rng.uniform(0, inner_w)
            cy = gradient_y + rng.uniform(0, visual_h)
            shade = rng.choice(["black", "#666666", "#999999"])
            ax.plot(cx, cy, "o", markersize=1.5, color=shade, alpha=0.5)


# =========================================================================
# FIGURE 1
# =========================================================================
def make_figure1(store: dict):
    print("Creating Figure 1: comparator-only motivating analysis (a-e) ...")

    s2 = read_tsv(F_S2)
    rel = read_tsv(F_REL)

    # ---- data -----------------------------------------------------------
    data = {}
    for slabel, s2name, sdir in MOTIV_SETS:
        data[slabel] = {t: load_predictions(os.path.join(MOTIV, sdir), t)
                        for t in MOTIV_TOOLS}

    stats = {}          # (set, tool, metric) -> dict(mae, lo, hi, n, k)
    for slabel, s2name, _ in MOTIV_SETS:
        for tool in MOTIV_TOOLS:
            for metric in ("completeness", "contamination"):
                r = s2_row(s2, s2name, tool, metric)
                mae = check_value(f"Fig2 {s2name}/{tool}/{metric} MAE",
                                  float(np.mean(abs_err(data[slabel][tool], metric))),
                                  float(r["mae"]))
                stats[(slabel, tool, metric)] = dict(
                    mae=mae, lo=float(r["mae_ci_lo"]), hi=float(r["mae_ci_hi"]),
                    n=int(r["n"]), k=int(r["n_clusters"]))

    store["fig1_motiv_n"] = {s: (stats[(s, "checkm2", "completeness")]["n"],
                                 stats[(s, "checkm2", "completeness")]["k"])
                             for s, _, _ in MOTIV_SETS}
    store["fig1_stats"] = stats

    # ---- layout ---------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 6.1))
    gs = fig.add_gridspec(5, 2, height_ratios=[0.62, 0.25, 1.0, 0.40, 1.0],
                          hspace=0, wspace=0.28,
                          left=0.085, right=0.985, top=0.965, bottom=0.055)

    # ---- panel a --------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    add_panel_label(ax_a, "a", x=-0.03, y=1.16)
    _draw_figure1_panel_a(ax_a)

    # ---- panels b-e: the three-tool motivating panels -------------------
    # MAGICC is deliberately absent from b-e, but the three tools that are
    # present keep the hues they carry everywhere else in the paper (the
    # four-tool BENCH palette, minus MAGICC's red).  v4's separate
    # MOTIV_COLORS set is retired: it recoloured CoCoPyE teal here and green
    # in panel f, inside a single figure.
    # ---- panels b, c ----------------------------------------------------
    width = 0.22
    offs = {t: (i - 1) * width for i, t in enumerate(MOTIV_TOOLS)}
    xs = np.arange(len(MOTIV_SETS))

    for panel, metric, ylab, title, ytop in (
            ("b", "completeness", "Absolute completeness error (pp)", "", 41),
            ("c", "contamination", "Absolute contamination error (pp)", "", 78)):
        ax = fig.add_subplot(gs[2, 0 if panel == "b" else 1])
        add_panel_label(ax, panel, x=-0.155, y=1.13)
        for tool in MOTIV_TOOLS:
            pos = xs + offs[tool]
            arrays = [abs_err(data[s][tool], metric) for s, _, _ in MOTIV_SETS]
            box_series(ax, pos, arrays, tool, width * 0.80)
            for xi, (slabel, _, _) in zip(pos, MOTIV_SETS):
                st = stats[(slabel, tool, metric)]
                point_ci(ax, xi, st["mae"], st["lo"], st["hi"], tool)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"M{i+1}" for i, _ in enumerate(MOTIV_SETS)])
        ax.set_xlim(-0.5, len(MOTIV_SETS) - 0.5)
        ax.set_ylim(0, ytop)
        ax.set_ylabel(ylab)
        if title:
            ax.set_title(title, fontsize=7, pad=3)
        ax.legend(handles=tool_handles(MOTIV_TOOLS), loc="upper left",
                  frameon=False, fontsize=6, handletextpad=0.4,
                  borderaxespad=0.2, labelspacing=0.25)

    # ---- panels d, e ----------------------------------------------------
    for panel, slabel, title in (("d", "B", "Pred. vs. true cont. (M2)"),
                                 ("e", "C", "Pred. vs. true cont. (M3)")):
        ax = fig.add_subplot(gs[4, 0 if panel == "d" else 1])
        add_panel_label(ax, panel, x=-0.155, y=1.13)
        for tool in ["checkm2", "cocopye", "deepcheck"]:
            df = data[slabel][tool]
            scatter_tool(ax, df["true_contamination"], df["pred_contamination"],
                         tool, s=6)
        diag(ax, 0, 100)
        ax.set_xlabel("True contamination (%)")
        ax.set_ylabel("Predicted contamination (%)")
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xlim(-5, 105)
        # Stage-selected comparator estimates are retained even above100%.
        # The previous105% limit silently hid valid high predictions.
        ymax = max(float(data[slabel][t]["pred_contamination"].max()) for t in MOTIV_TOOLS)
        ax.set_ylim(-5, max(105, np.ceil((ymax + 5) / 50) * 50))
        ax.legend(handles=tool_handles(MOTIV_TOOLS), loc="upper left",
                  frameon=False, fontsize=6, handletextpad=0.4,
                  borderaxespad=0.2, labelspacing=0.25)

    return _save(fig, "Figure_1")


# =========================================================================
# FIGURE 2 -- MAGICC workflow schematic
#
# Every box, arrow and label COORDINATE is unchanged from
# manuscript3/scripts/create_figures_v5.py::make_figure2.  The scientific
# content is unchanged from the previous round.  What changed here is colour
# only (BUILD_CONTRACT.md 4.2):
#
#   1. One accent per phase, in phase order: #4E79A7, #F28E2B, #59A14F,
#      #4E79A7, #E15759 (figstyle.PHASE_ACCENT).  Boxes are a 15 % tint of the
#      phase accent over white, outlined in the saturated accent, with BLACK
#      text; the box each phase hands on to the next keeps its heavier outline.
#      No dark box with reversed-out white text and no pure-grey box remains.
#   2. The phase grouping panel is a 6 % tint of the same accent and carries a
#      header band with the phase name.
#   3. The prediction head is the only solid-filled box, so the reader's eye
#      lands on the output.
#   4. Arrows are #4D4D4D at 0.8 pt throughout; cross-phase flow is now marked
#      by a dashed line style rather than by a lighter colour, because the
#      contract fixes one arrow colour.
#   5. TEXT CONTRAST.  The two fills that carry reversed-out white text -- the
#      header bands and the solid prediction head -- are drawn in
#      figstyle.PHASE_BAND, the darkened companions of the same five hues,
#      because white on the undarkened accents misses the WCAG AA 4.5:1 target
#      (white on #E15759 measures 3.68:1).  Only this schematic furniture is
#      darkened: the box borders, the box tints and the panel tints still come
#      from PHASE_ACCENT, so every phase hue is on the canvas at full
#      saturation, and the four series hex codes of figstyle.PALETTE -- which
#      encode the tools -- are untouched.  Every text/fill pair in this figure
#      is re-measured into figures/palette_cvd_check.tsv on each build.
#
# Carried over unchanged from the previous round:
#   * the V5 synthetic stream (1,200,000 samples; train 1,000,000 / val 100,000
#     / test 100,000), source project_progress_and_results.md sec. 3.0
#     "Phase 4 -- Synthetic data" and data/features/magicc_v5_features.h5;
#   * no GTDB release label, because none is asserted under results/revision/;
#   * a 5 pt in-box type floor.
#
# Style constants from the original palette specification are fixed here.
# The optional palette-screen report is outside scientific figure replay.
# =========================================================================
PHASE_TINT_BOX = .15   # 0.15
PHASE_TINT_PANEL = .06  # 0.06
BOX_INK = "#111111"                  # "#111111"
OUTPUT_FILL = PHASE_BAND[4]          # "#A6383A"
OUTPUT_INK = "#FFFFFF"            # "#FFFFFF"


def make_figure2(store: dict):
    from make_workflow import make_workflow
    return make_workflow()


# =========================================================================
# FIGURE 3
# =========================================================================
def make_figure3(store: dict):
    print("Creating Figure 3: test-reference five-set benchmark ...")

    wide = read_tsv(F_WIDE)
    mimag = read_tsv(F_MIMAG)
    thresh = read_tsv(F_THRESH)

    for df, name in ((wide, "definitive_table_5set_wide.tsv"),
                     (mimag, "definitive_mimag.tsv"),
                     (thresh, "definitive_thresholds.tsv")):
        bad = [s for s in df["set"].unique() if str(s) in FORBIDDEN_SETS]
        if bad:
            raise ValueError(f"{name} unexpectedly contains {bad} (trap T1)")

    data = {}
    for slabel, skey, sdir in BENCH_SETS:
        data[slabel] = {t: load_predictions(os.path.join(BASE, sdir), t)
                        for t in BENCH_TOOLS}

    mae = {}
    for slabel, skey, _ in BENCH_SETS:
        for tool in BENCH_TOOLS:
            r = wide_row(wide, skey, tool)
            for metric in ("completeness", "contamination"):
                v = check_value(f"Fig3 {skey}/{tool}/{metric} MAE",
                                float(np.mean(abs_err(data[slabel][tool], metric))),
                                float(r[f"{metric}_mae"]))
                mae[(slabel, tool, metric)] = dict(
                    mae=v, lo=float(r[f"{metric}_mae_ci_lo"]),
                    hi=float(r[f"{metric}_mae_ci_hi"]),
                    n=int(r["n"]), k=int(r["n_clusters"]))
    store["fig3_mae"] = mae

    # Trap T3: R^2 is the coefficient of determination, never squared Pearson.
    r2_cc = float(wide_row(wide, "set_C_clean", "magicc_v5")["completeness_r2_cod"])
    d = data["Set C-clean"]["magicc_v5"]
    y, yhat = d["true_completeness"].to_numpy(), d["pred_completeness"].to_numpy()
    cod = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    check_value("Fig3 set_C_clean MAGICC completeness R2 (CoD)", float(cod), r2_cc)
    if abs(r2_cc - 0.656) < 0.005:
        raise ValueError("set_C_clean completeness R2 is the squared Pearson value "
                         "0.656; the coefficient of determination 0.611 is required (T3)")
    store["fig3_r2_cc"] = r2_cc

    f1 = {}
    for slabel, skey, _ in BENCH_SETS:
        for tool in BENCH_TOOLS:
            r = mimag_row(mimag, skey, tool)
            f1[(slabel, tool)] = dict(f1=float(r["macro_f1"]),
                                      lo=float(r["macro_f1_ci_lo"]),
                                      hi=float(r["macro_f1_ci_hi"]),
                                      balance=str(r["class_balance_true"]))
    store["fig3_f1"] = f1

    thr = {}
    for slabel, skey, _ in BENCH_SETS:
        for tool in BENCH_TOOLS:
            r = thresh_row(thresh, skey, tool)
            thr[(slabel, tool)] = dict(
                ff=float(r["false_fail_rate"]), ff_lo=float(r["false_fail_rate_ci_lo"]),
                ff_hi=float(r["false_fail_rate_ci_hi"]),
                fp=float(r["false_pass_rate"]), fp_lo=float(r["false_pass_rate_ci_lo"]),
                fp_hi=float(r["false_pass_rate_ci_hi"]),
                n_clean=int(r["n_true_pass"]), n_cont=int(r["n_true_fail"]),
                bal=float(r["balanced_accuracy"]))
    store["fig3_thr"] = thr

    # hard check: Set C-clean denominators
    cc = thr[("Set C-clean", "magicc_v5")]
    if (cc["n_clean"], cc["n_cont"]) != (52, 948):
        raise ValueError(f"Set C-clean 5 % threshold denominators are "
                         f"{cc['n_clean']}/{cc['n_cont']}, expected 52/948")

    # ---- layout ---------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.52, 1.0, 1.0, 0.95],
                          hspace=0.58, wspace=0.46,
                          left=0.075, right=0.985, top=0.965, bottom=0.06)

    # ---- panel a --------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :])
    add_panel_label(ax_a, "a", x=-0.022, y=1.20)
    _draw_figure3_panel_a(ax_a)

    # ---- panels b, c ----------------------------------------------------
    width = 0.19
    offs = {t: o for t, o in zip(BENCH_TOOLS, [-0.30, -0.10, 0.10, 0.30])}
    xs = np.arange(len(BENCH_SETS))
    set_labels = [s for s, _, _ in BENCH_SETS]

    for panel, metric, ylab, title, ytop in (
            ("b", "completeness", "Absolute completeness error (pp)", "", 42),
            ("c", "contamination", "Absolute contamination error (pp)", "", 88)):
        ax = fig.add_subplot(gs[1, 0:2] if panel == "b" else gs[1, 2:4])
        add_panel_label(ax, panel, x=-0.105, y=1.13)
        for tool in BENCH_TOOLS:
            pos = xs + offs[tool]
            arrays = [abs_err(data[s][tool], metric) for s in set_labels]
            box_series(ax, pos, arrays, tool, width * 0.82)
            for xi, s in zip(pos, set_labels):
                st = mae[(s, tool, metric)]
                point_ci(ax, xi, st["mae"], st["lo"], st["hi"], tool, ms=3.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(set_labels)
        ax.set_xlim(-0.55, len(BENCH_SETS) - 0.45)
        ax.set_ylim(0, ytop)
        ax.set_ylabel(ylab)
        if title:
            ax.set_title(title, fontsize=7, pad=3)
        ax.legend(handles=tool_handles(BENCH_TOOLS), loc="upper left", ncol=2,
                  frameon=False, fontsize=5.8, handletextpad=0.4,
                  columnspacing=0.9, labelspacing=0.25, borderaxespad=0.2)

    # ---- panel d: the two rebuilt sets ----------------------------------
    for j, (slabel, title) in enumerate((("Set C-clean", "Pred. vs. true cont.\n(Set C-clean)"),
                                         ("Set D-clean", "Pred. vs. true cont.\n(Set D-clean)"))):
        ax = fig.add_subplot(gs[2, j])
        if j == 0:
            add_panel_label(ax, "d", x=-0.36, y=1.16)
        for tool in ["deepcheck", "cocopye", "checkm2", "magicc_v5"]:
            df = data[slabel][tool]
            scatter_tool(ax, df["true_contamination"], df["pred_contamination"],
                         tool, s=5)
        diag(ax, 0, 100)
        ax.set_xlabel("True cont. (%)", fontsize=6.5)
        ax.set_ylabel("Pred. cont. (%)", fontsize=6.5)
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_xlim(-5, 105)
        ymax = max(float(data[s][t]["pred_contamination"].max())
                   for s in ["Set C-clean", "Set D-clean"] for t in BENCH_TOOLS)
        ytop = max(105, np.ceil((ymax + 5) / 50) * 50)
        ax.set_ylim(-5, ytop)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks(np.arange(0, ytop + 1, 50))
        if j == 0:
            ax.legend(handles=tool_handles(BENCH_TOOLS, ms=2.8), loc="upper left",
                      frameon=False, fontsize=5.0, handletextpad=0.3,
                      labelspacing=0.2, borderaxespad=0.15)

    # ---- panel e: MIMAG-inspired macro F1 -------------------------------
    ax = fig.add_subplot(gs[2, 2:4])
    add_panel_label(ax, "e", x=-0.105, y=1.13)
    for tool in BENCH_TOOLS:
        for s in set_labels:
            st = f1[(s, tool)]
            point_ci(ax, set_labels.index(s) + offs[tool], st["f1"],
                     st["lo"], st["hi"], tool, ms=3.0)
    ax.plot([-0.42, 0.42], [2 / 3, 2 / 3], ls=":", lw=0.7, color="0.4", zorder=1)
    ax.text(0.0, 2 / 3 + 0.025, "ceiling 0.67", fontsize=5.0,
            ha="center", va="bottom", color="0.35")
    ax.set_xticks(xs)
    ax.set_xticklabels(set_labels)
    ax.set_xlim(-0.55, len(BENCH_SETS) - 0.45)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("MIMAG-inspired macro F1\n(3 classes, n = 1,000 per set)")
    ax.set_title("Three-class quality assignment", fontsize=7, pad=3)
    ax.legend(handles=tool_handles(BENCH_TOOLS), loc="lower left", ncol=1,
              frameon=False, fontsize=5.8, handletextpad=0.4,
              labelspacing=0.25, borderaxespad=0.2)

    # ---- panel f: the false-fail / false-pass pair ----------------------
    ax = fig.add_subplot(gs[3, :])
    add_panel_label(ax, "f", x=-0.062, y=1.13)
    for tool in BENCH_TOOLS:
        for i, s in enumerate(set_labels):
            st = thr[(s, tool)]
            x = i + offs[tool]
            if np.isfinite(st["ff"]) and np.isfinite(st["fp"]):
                ax.plot([x, x], [st["ff"], st["fp"]], color="0.55", lw=0.5, zorder=3)
            if np.isfinite(st["ff"]):
                point_ci(ax, x, st["ff"], st["ff_lo"], st["ff_hi"], tool,
                         ms=3.4, filled=True)
            if np.isfinite(st["fp"]):
                point_ci(ax, x, st["fp"], st["fp_lo"], st["fp_hi"], tool,
                         ms=3.4, filled=False)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{s}\n{thr[(s, 'magicc_v5')]['n_clean']:,} truly clean\n"
         f"{thr[(s, 'magicc_v5')]['n_cont']:,} truly contaminated"
         for s in set_labels], fontsize=5.2)
    ax.set_xlim(-0.55, len(BENCH_SETS) - 0.45)
    ax.set_ylim(-0.03, 1.0)
    ax.set_ylabel("Rate at the 5 % contamination\nboundary")
    ax.set_title("5 % contamination boundary", fontsize=7, pad=3)
    style_handles = [
        Line2D([], [], marker="o", color="0.3", markerfacecolor="0.3",
               markeredgecolor="0.3", markersize=3.4, linestyle="none",
               label="False-fail (of truly clean)"),
        Line2D([], [], marker="o", color="0.3", markerfacecolor="white",
               markeredgecolor="0.3", markersize=3.4, linestyle="none",
               label="False-pass (of truly contaminated)"),
    ]
    ax.legend(handles=tool_handles(BENCH_TOOLS) + style_handles, loc="upper left",
              ncol=3, frameon=False, fontsize=5.6, handletextpad=0.4,
              columnspacing=1.0, labelspacing=0.25, borderaxespad=0.2)

    return _save(fig, "Figure_3")


# =========================================================================
# Captions -- one file per figure under figures/caption_parts/, assembled into
# figures/captions_main.md by make_captions_main.py.
#
# BUILD_CONTRACT 2.1: bold is allowed only for the leading "**Figure N.**" and
# for panel letters.  BUILD_CONTRACT 4.3: every number deleted from a canvas is
# carried here.
# =========================================================================
CAPTION_PARTS = os.path.join(FIG_DIR, "caption_parts")


def _write_part(name: str, lines: "list[str]") -> str:
    os.makedirs(CAPTION_PARTS, exist_ok=True)
    path = os.path.join(CAPTION_PARTS, f"{name}.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
    print(f"  wrote {os.path.basename(path)}")
    return path


# =========================================================================
# main
# =========================================================================
def main():
    global DRAFT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", action="store_true",
                    help="fast low-dpi PNG only, written to a scratch directory")
    ap.add_argument("--only", choices=["1", "2", "3"], default=None,
                    help="render a single figure")
    args = ap.parse_args()
    DRAFT = args.draft

    for f in (F_WIDE, F_LONG, F_MIMAG, F_THRESH, F_S2, F_REL):
        require(f)
    for _, _, sdir in MOTIV_SETS:
        for tool in MOTIV_TOOLS:
            require(os.path.join(MOTIV, sdir, f"{tool}_predictions.tsv"))
        require(os.path.join(MOTIV, sdir, "metadata.tsv"))
    for _, _, sdir in BENCH_SETS:
        require(os.path.join(BASE, sdir, "magicc_v5_predictions.tsv"))
        for tool in ("checkm2", "cocopye", "deepcheck"):
            require(os.path.join(BASE, sdir, f"{tool}_predictions.tsv"))
        require(os.path.join(BASE, sdir, "metadata.tsv"))
    print("All required input files verified.")

    store: dict = {}
    if args.only in (None, "1"):
        make_figure1(store)
    if args.only in (None, "2"):
        make_figure2(store)
    if args.only in (None, "3"):
        make_figure3(store)
    if args.only is None and not DRAFT:
        figvalues.dump(OUT_DIR, "fig1_3", store)

    print()
    if _DISCREPANCIES:
        print(f"{len(_DISCREPANCIES)} value discrepancy/ies between recomputed and recorded:")
        for m in _DISCREPANCIES:
            print("  - " + m)
    else:
        print("All recomputed values match the recorded result files to <0.01.")


if __name__ == "__main__":
    main()
