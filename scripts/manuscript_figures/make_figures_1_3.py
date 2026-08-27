#!/usr/bin/env python3
"""Main-text Figures 1-3 for the MAGICC Nature Communications resubmission.

Style continuity
----------------
Panel layouts, schematic-drawing code, axis conventions and figure proportions
are taken from the script that produced the submitted figures,
`manuscript3/scripts/create_figures_v5.py` (its hard-coded absolute workspace
paths are repointed to `/path/to/magicc`).  Two deliberate
deviations, both editorial requirements:

  * E4 / the internal build contract 9.4 -- the tool palette is the ORIGINAL
    submission's set, restored in `figstyle.py` (MAGICC #d62728 red,
    CheckM2 #1f77b4 blue, CoCoPyE #2ca02c green, DeepCheck #9467bd purple;
    pastel companions for box and bar fills).  The author's instruction is
    read strictly, so this is the palette for EVERY tool comparison: the
    three-tool motivating panels b-e, from which MAGICC is deliberately
    absent, draw the same CheckM2 / CoCoPyE / DeepCheck hues as panel f and
    as the rest of the paper, simply with MAGICC's red missing.  v4's
    separate `MOTIV_COLORS` set is retired -- it gave CoCoPyE a different
    hue in b-e (teal) from the one it has in f (green), inside one figure,
    and its CheckM2/DeepCheck pair was the worst in the audit (CIE76
    Delta-E 0.60 under protanopia).  CheckM2 and DeepCheck remain a
    blue/purple pair and MAGICC and CoCoPyE a red/green pair, both below the
    15-unit CIE76 Delta-E screen under simulated deficiency, so marker
    shape, line style and direct labelling carry the separation; the
    measured Delta-E of every pair is in `figures/palette_cvd_check.tsv` and
    `figures/colour_statement.md`.
  * E9 -- bar charts of MAE are replaced by distribution-revealing plots
    (box: IQR, whiskers 5th-95th percentile, no fliers) with the MAE drawn as a
    point estimate with its 95 % cluster-bootstrap CI on top.
  * the internal build contract 4.3 -- no explanatory sentence, statistical annotation block
    or per-point commentary is drawn onto the canvas.  Only axis labels, tick
    labels, panel letters, legends and at most one short direct label per panel
    survive; every number removed from a canvas is carried by the caption.

Figure 1  Existing tools underestimate contamination from divergent sources
          a  schematic of the three motivating dataset profiles (verbatim)
          b  completeness error, CheckM2/CoCoPyE/DeepCheck x motivating A,B,C
          c  contamination error, same layout
          d  predicted vs true contamination, motivating Set B
          e  predicted vs true contamination, motivating Set C
          f  NEW: contamination signed bias by contaminant relatedness, Set E
             (the only panel in which MAGICC appears)

Figure 2  MAGICC workflow schematic.  Box, arrow and label coordinates are
          unchanged; recoloured to one accent per phase (the internal build contract 4.2:
          #4E79A7, #F28E2B, #59A14F, #4E79A7, #E15759) with light tinted fills,
          saturated same-hue borders, black in-box text and accent-filled phase
          header bands.  The prediction head is the only solid-filled box.  The
          V5 synthetic-data stream (1,200,000 samples) and the removal of the
          unverifiable GTDB release label are carried over unchanged.  The
          palette check is written to `figures/palette_cvd_check.tsv` by
          `make_palette_cvd_check.py`, which this script calls.

Figure 3  Benchmark performance on the LEAKAGE-FREE five-set panel
          Set A = set_A_v2, Set B = set_B_v2, Set C-clean, Set D-clean, Set E
          a  schematic of the five benchmark profiles
          b  per-genome absolute completeness error by tool x set
          c  per-genome absolute contamination error by tool x set
          d  predicted vs true contamination, Set C-clean and Set D-clean
          e  MIMAG-inspired three-class macro F1 by tool x set
          f  false-fail / false-pass at the 5 % contamination boundary (paired)

Summary statistics are READ FROM the finalized result files under
`results/revision/metrics/`; anything recomputed from the raw predictions is
checked against the recorded value and any deviation >0.01 is reported.

Sets F and G never appear (trap T1).  R^2 is always the coefficient of
determination (trap T3): set_C_clean MAGICC completeness R^2 = 0.611.

Usage
-----
    python make_figures_1_3.py            # final, 400 dpi PNG + vector PDF
    python make_figures_1_3.py --draft    # fast low-dpi PNG only, to a scratch dir
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
import make_palette_cvd_check  # noqa: E402

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
# Schematic: Figure 1 panel a  (verbatim from create_figures_v5.py)
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
        ("Set A (completeness gradient)", "comp: 50-100%  cont: 0%", "1,000"),
        ("Set B (contamination gradient)", "comp: 100%  cont: 0-80%", "1,000"),
        ("Set C (realistic mix)", "comp: 50-100%  cont: 0-100%", "1,000"),
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
# leakage-free rebuilds, which is the only scientific change)
# =========================================================================
def _draw_figure3_panel_a(ax):
    """Draw the leakage-free benchmark sets schematic, Figure 1a style."""
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
    print("Creating Figure 1: motivating analysis (a-e) + relatedness (f) ...")

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
                mae = check_value(f"Fig1 {s2name}/{tool}/{metric} MAE",
                                  float(np.mean(abs_err(data[slabel][tool], metric))),
                                  float(r["mae"]))
                stats[(slabel, tool, metric)] = dict(
                    mae=mae, lo=float(r["mae_ci_lo"]), hi=float(r["mae_ci_hi"]),
                    n=int(r["n"]), k=int(r["n_clusters"]))

    store["fig1_motiv_n"] = {s: (stats[(s, "checkm2", "completeness")]["n"],
                                 stats[(s, "checkm2", "completeness")]["k"])
                             for s, _, _ in MOTIV_SETS}
    store["fig1_stats"] = stats

    # Set E relatedness strata (panel f)
    strata = [("none (uncontaminated)", "Uncontaminated"),
              ("within_phylum", "Within-phylum contaminant"),
              ("cross_phylum", "Cross-phylum contaminant")]
    setE = {t: load_predictions(os.path.join(BASE, "set_E"), t) for t in BENCH_TOOLS}
    meta_E = read_tsv(os.path.join(BASE, "set_E", "metadata.tsv"))

    def relatedness_of(row):
        # config_revision_metrics.yaml: relatedness_rule "from_sample_type";
        # rows with zero true contamination always become "none (uncontaminated)"
        if row["true_contamination"] == 0:
            return "none (uncontaminated)"
        st = str(row["sample_type"])
        if "within_phylum" in st:
            return "within_phylum"
        if "cross_phylum" in st:
            return "cross_phylum"
        raise ValueError(f"cannot resolve relatedness for sample_type={st!r}")

    meta_E = meta_E.assign(relatedness=meta_E.apply(relatedness_of, axis=1))
    rel_map = dict(zip(meta_E["genome_id"], meta_E["relatedness"]))

    relstats = {}
    for skey, _ in strata:
        for tool in BENCH_TOOLS:
            r = rel_row(rel, skey, tool)
            df = setE[tool]
            sel = df["genome_id"].map(rel_map) == skey
            signed = (df.loc[sel, "pred_contamination"]
                      - df.loc[sel, "true_contamination"]).to_numpy()
            if len(signed) != int(r["n"]):
                raise ValueError(f"Set E {skey}/{tool}: n={len(signed)} recomputed vs "
                                 f"{int(r['n'])} recorded")
            mean = check_value(f"Fig1f set_E/{skey}/{tool} mean signed error",
                               float(signed.mean()), float(r["mean_signed_error"]))
            relstats[(skey, tool)] = dict(
                mean=mean, lo=float(r["mean_ci_lo"]), hi=float(r["mean_ci_hi"]),
                n=int(r["n"]), k=int(r["n_clusters"]), values=signed)
    store["fig1_rel"] = relstats

    # ---- layout ---------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.62, 1.0, 1.0, 0.95],
                          hspace=0.56, wspace=0.28,
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
        ax = fig.add_subplot(gs[1, 0 if panel == "b" else 1])
        add_panel_label(ax, panel, x=-0.155, y=1.13)
        for tool in MOTIV_TOOLS:
            pos = xs + offs[tool]
            arrays = [abs_err(data[s][tool], metric) for s, _, _ in MOTIV_SETS]
            box_series(ax, pos, arrays, tool, width * 0.80)
            for xi, (slabel, _, _) in zip(pos, MOTIV_SETS):
                st = stats[(slabel, tool, metric)]
                point_ci(ax, xi, st["mae"], st["lo"], st["hi"], tool)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"Set {s}" for s, _, _ in MOTIV_SETS])
        ax.set_xlim(-0.5, len(MOTIV_SETS) - 0.5)
        ax.set_ylim(0, ytop)
        ax.set_ylabel(ylab)
        if title:
            ax.set_title(title, fontsize=7, pad=3)
        ax.legend(handles=tool_handles(MOTIV_TOOLS), loc="upper left",
                  frameon=False, fontsize=6, handletextpad=0.4,
                  borderaxespad=0.2, labelspacing=0.25)

    # ---- panels d, e ----------------------------------------------------
    for panel, slabel, title in (("d", "B", "Pred. vs. true cont. (Set B)"),
                                 ("e", "C", "Pred. vs. true cont. (Set C)")):
        ax = fig.add_subplot(gs[2, 0 if panel == "d" else 1])
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
        ax.set_ylim(-5, 105)
        ax.legend(handles=tool_handles(MOTIV_TOOLS), loc="upper left",
                  frameon=False, fontsize=6, handletextpad=0.4,
                  borderaxespad=0.2, labelspacing=0.25)

    # ---- panel f --------------------------------------------------------
    ax = fig.add_subplot(gs[3, :])
    add_panel_label(ax, "f", x=-0.062, y=1.13)
    fwidth = 0.15
    foffs = {t: o for t, o in zip(BENCH_TOOLS, [-0.27, -0.09, 0.09, 0.27])}
    hline0(ax)
    for tool in BENCH_TOOLS:
        means = []
        for i, (skey, _) in enumerate(strata):
            st = relstats[(skey, tool)]
            x = i + foffs[tool]
            box_series(ax, [x], [st["values"]], tool, fwidth)
            point_ci(ax, x, st["mean"], st["lo"], st["hi"], tool, ms=3.2)
            means.append((x, st["mean"]))
        ax.plot([m[0] for m in means], [m[1] for m in means],
                color=PALETTE[tool], lw=0.7, ls="-", zorder=5, alpha=0.85)

    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(
        [f"{lab}\nn = {relstats[(k, 'magicc_v5')]['n']:,} "
         f"({relstats[(k, 'magicc_v5')]['k']:,} clusters)" for k, lab in strata],
        fontsize=6)
    ax.set_xlim(-0.55, len(strata) - 0.45)
    ax.set_ylim(-82, 58)
    ax.set_ylabel("Contamination signed error (pp)\npredicted - true")
    ax.set_title("Set E", fontsize=7, pad=3)

    # the internal build contract 4.3: the within- -> cross-phylum mini table that used to be
    # drawn here is deleted; its eight numbers are carried by the caption.
    ax.legend(handles=tool_handles(BENCH_TOOLS), loc="upper right", ncol=2,
              frameon=False, fontsize=6, handletextpad=0.4, columnspacing=1.0,
              labelspacing=0.25, borderaxespad=0.3)

    return _save(fig, "Figure_1")


# =========================================================================
# FIGURE 2 -- MAGICC workflow schematic
#
# Every box, arrow and label COORDINATE is unchanged from
# manuscript3/scripts/create_figures_v5.py::make_figure2.  The scientific
# content is unchanged from the previous round.  What changed here is colour
# only (the internal build contract 4.2):
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
#     / test 100,000), source the internal project log sec. 3.0
#     "Phase 4 -- Synthetic data" and data/features/magicc_v5_features.h5;
#   * no GTDB release label, because none is asserted under results/revision/;
#   * a 5 pt in-box type floor.
#
# The palette screen lives in make_palette_cvd_check.py and is written to
# figures/palette_cvd_check.tsv.
# =========================================================================
PHASE_TINT_BOX = make_palette_cvd_check.PHASE_TINT_FRAC   # 0.15
PHASE_TINT_PANEL = make_palette_cvd_check.PHASE_TINT_PANEL  # 0.06
BOX_INK = make_palette_cvd_check.BOX_INK                  # "#111111"
OUTPUT_FILL = make_palette_cvd_check.OUTPUT_FILL          # "#A6383A"
OUTPUT_INK = make_palette_cvd_check.OUTPUT_INK            # "#FFFFFF"


def make_figure2(store: dict):
    """Create Figure 2: MAGICC pipeline workflow diagram.

    Layout is that of manuscript3/scripts/create_figures_v5.py::make_figure2;
    see the section header above for the colour rules of the internal build contract 4.2.
    """
    print("Creating Figure 2: MAGICC workflow (one accent per phase) ...")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.7))
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-1.2, 6.5)
    ax.axis("off")

    # One accent per phase; boxes are a light tint of it, borders the accent.
    # `band` is the darkened companion of the same hue and is used only where
    # white text sits on the fill (header bands, prediction head).
    accent = {f"p{i + 1}": PHASE_ACCENT[i] for i in range(5)}
    band = {f"p{i + 1}": PHASE_BAND[i] for i in range(5)}
    colors = {k: tint(v, PHASE_TINT_BOX) for k, v in accent.items()}
    colors["out"] = OUTPUT_FILL
    edge = dict(accent)
    edge["out"] = OUTPUT_FILL
    fg = {k: BOX_INK for k in accent}
    fg["out"] = OUTPUT_INK
    panel = {k: tint(v, PHASE_TINT_PANEL) for k, v in accent.items()}
    flow = ARROW_INK                     # every arrow, within and across phases

    def add_box(x, y, w, h, text, key, fontsize=5.0, bold=False,
                text_color=None, edge_color=None, linewidth=0.7,
                linespacing=1.15):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             facecolor=colors[key],
                             edgecolor=edge_color or edge[key],
                             linewidth=linewidth)
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight,
                color=text_color or fg[key],
                linespacing=linespacing, multialignment="center")

    def arrow(x1, y1, x2, y2, lw=0.8, style="->", dashed=False):
        props = dict(arrowstyle=style, color=flow, lw=lw)
        if dashed:
            props["linestyle"] = (0, (2.6, 1.6))
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=props)

    def cross_arrow(x1, y1, x2, y2):
        arrow(x1, y1, x2, y2, lw=0.8, style="->", dashed=True)

    def phase_bg(x, y, w, h, label, key):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=panel[key], edgecolor=accent[key],
                              linewidth=0.5, alpha=1.0)
        ax.add_patch(rect)
        band_h = 0.66
        head = FancyBboxPatch((x, y + h - band_h), w, band_h,
                              boxstyle="round,pad=0.06",
                              facecolor=band[key], edgecolor=band[key],
                              linewidth=0.5)
        ax.add_patch(head)
        ax.text(x + w / 2, y + h - band_h / 2, label, ha="center", va="center",
                fontsize=5.5, fontweight="bold",
                color=readable_ink(band[key]), linespacing=1.25)

    PX = [0.0, 3.6, 7.2, 10.5, 14.0]
    PW = [3.3, 3.3, 3.0, 3.2, 4.2]
    BW = [2.8, 2.8, 2.5, 2.7, 3.6]
    BX = [PX[i] + (PW[i] - BW[i]) / 2 for i in range(5)]
    ROW = [4.2, 2.8, 1.4, 0.0]
    BH = 0.85

    bg_y = -0.3
    bg_h = 6.4
    phase_labels = [
        "Phase 1\nData Curation",
        "Phase 2\nK-mer Selection",
        "Phase 3\nTraining Synthesis",
        "Phase 4\nFeature Extraction",
        "Phase 5\nNeural Network",
    ]
    for i in range(5):
        phase_bg(PX[i], bg_y, PW[i], bg_h, phase_labels[i], f"p{i + 1}")

    # Phase 1 -- no GTDB release number is asserted; only counts that are
    # traceable to data/gtdb/filtered_genomes.tsv and data/splits/.
    i = 0
    add_box(BX[i], ROW[0], BW[i], BH, "GTDB\n732,475 genomes",
            "p1", fontsize=5.5, bold=True)
    add_box(BX[i], ROW[1], BW[i], BH,
            "Quality filters\ncomp >98%, cont <2%\n<100 contigs\nN50 >20 kbp",
            "p1", fontsize=5.0)
    add_box(BX[i], ROW[2], BW[i], BH, "277,183 pass\nquality filters",
            "p1", fontsize=5.5, bold=True)
    add_box(BX[i], ROW[3], BW[i], BH,
            "100,000 sampled\nacross 110 phyla\n(stratified sampling)",
            "p1", fontsize=5.0, bold=True, linewidth=1.4)
    cx1 = BX[i] + BW[i] / 2
    arrow(cx1, ROW[0], cx1, ROW[1] + BH)
    arrow(cx1, ROW[1], cx1, ROW[2] + BH)
    arrow(cx1, ROW[2], cx1, ROW[3] + BH)

    # Phase 2
    i = 1
    add_box(BX[i], ROW[0], BW[i], BH,
            "2,000 representatives\n(1,000 bacterial +\n1,000 archaeal)",
            "p2", fontsize=5.0)
    add_box(BX[i], ROW[1], BW[i], BH,
            "Core gene identification\nProdigal + HMMER\n(85 bacterial +\n128 archaeal HMMs)",
            "p2", fontsize=5.0)
    add_box(BX[i], ROW[2], BW[i], BH, "9-mer counting\n(KMC3)",
            "p2", fontsize=5.5)
    add_box(BX[i], ROW[3], BW[i], BH,
            "Top 9,249 canonical\nk-mers\n(by prevalence)",
            "p2", fontsize=5.0, bold=True, linewidth=1.4)
    cx2 = BX[i] + BW[i] / 2
    arrow(cx2, ROW[0], cx2, ROW[1] + BH)
    arrow(cx2, ROW[1], cx2, ROW[2] + BH)
    arrow(cx2, ROW[2], cx2, ROW[3] + BH)
    cross_arrow(BX[0] + BW[0], ROW[3] + BH / 2, BX[1], ROW[0] + BH / 2)

    # Phase 3 -- V5 synthetic stream (magicc_v5_features.h5): 1,200,000 samples,
    # train 1,000,000 = 800,000 V4 + 100,000 comp 100 % / cont 0 % + 100,000
    # comp 100 % / cont 0-10 %; validation 100,000; test 100,000.
    i = 2
    add_box(BX[i], ROW[0], BW[i], BH,
            "100,000 reference\ngenomes fragmented\n(4 quality tiers)",
            "p3", fontsize=5.0)
    add_box(BX[i], ROW[1], BW[i], BH,
            "Contamination\ninjection: within +\ncross-phylum\n(Dirichlet)",
            "p3", fontsize=5.0)
    add_box(BX[i], ROW[2], BW[i], BH, "1,200,000\nsynthetic genomes",
            "p3", fontsize=5.0, bold=True)
    add_box(BX[i], ROW[3], BW[i], BH,
            "1,000,000 train\n100,000 validation\n100,000 test",
            "p3", fontsize=5.0, bold=True, linewidth=1.4)
    cx3 = BX[i] + BW[i] / 2
    arrow(cx3, ROW[0], cx3, ROW[1] + BH)
    arrow(cx3, ROW[1], cx3, ROW[2] + BH)
    arrow(cx3, ROW[2], cx3, ROW[3] + BH)
    cross_arrow(BX[0] + BW[0], ROW[3] + BH / 2, BX[2], ROW[0] + BH / 2)

    # Phase 4
    i = 3
    add_box(BX[i], ROW[0], BW[i], BH, "Input FASTA",
            "p4", fontsize=5.5, bold=True)
    add_box(BX[i], ROW[1], BW[i], BH,
            "K-mer counting\n9,249 canonical\n9-mers (Numba\nrolling hash)",
            "p4", fontsize=5.0)
    add_box(BX[i], ROW[2], BW[i], BH, "7 k-mer summary\nfeatures",
            "p4", fontsize=5.0)
    add_box(BX[i], ROW[3], BW[i], BH,
            "Normalization\n(Z-score, log,\nmin-max, robust)",
            "p4", fontsize=5.0, linewidth=1.4)
    cx4 = BX[i] + BW[i] / 2
    arrow(cx4, ROW[0], cx4, ROW[1] + BH)
    arrow(cx4 + BW[i] / 2 - 0.15, ROW[0], cx4 + BW[i] / 2 - 0.15, ROW[2] + BH,
          dashed=True)
    arrow(cx4 - 0.4, ROW[1], cx4 - 0.4, ROW[3] + BH, dashed=True)
    arrow(cx4 + 0.4, ROW[2], cx4 + 0.4, ROW[3] + BH)
    ax.text(cx4, ROW[0] + BH + 0.12, "(also inference pipeline)",
            ha="center", va="bottom", fontsize=5.0, color=BOX_INK,
            style="italic")
    cross_arrow(BX[1] + BW[1], ROW[3] + BH / 2, BX[3], ROW[1] + BH / 2)
    cross_arrow(BX[2] + BW[2], ROW[3] + BH / 2, BX[3], ROW[3] + BH / 2)

    # Phase 5
    i = 4
    add_box(BX[i], ROW[0], BW[i], BH,
            "K-mer branch\n9,249→4,096→1,024→256\n(FC layers)",
            "p5", fontsize=5.0)
    add_box(BX[i], ROW[1], BW[i], BH,
            "Fusion layer\nconcat(256 + 16 = 272)\n→ 128 → 64",
            "p5", fontsize=5.0, bold=True, linewidth=1.4)
    add_box(BX[i], ROW[2], BW[i], BH,
            "K-mer stat branch\n7 → 32 → 16\n(FC layers)",
            "p5", fontsize=5.0)
    add_box(BX[i], ROW[3], BW[i], BH,
            "Prediction head\nCompleteness [50–100%]\nContamination [0–100%]",
            "out", fontsize=5.0, bold=True, linewidth=1.2)
    cx5 = BX[i] + BW[i] / 2
    arrow(cx5, ROW[0], cx5, ROW[1] + BH)
    arrow(cx5, ROW[2] + BH, cx5, ROW[1])
    arrow(cx5, ROW[1], cx5, ROW[3] + BH, dashed=True)
    cross_arrow(BX[3] + BW[3], ROW[1] + BH / 2, BX[4], ROW[0] + BH / 2)
    cross_arrow(BX[3] + BW[3], ROW[2] + BH / 2, BX[4], ROW[2] + BH / 2)

    return _save(fig, "Figure_2")



# =========================================================================
# FIGURE 3
# =========================================================================
def make_figure3(store: dict):
    print("Creating Figure 3: leakage-free five-set benchmark ...")

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
        ax.set_ylim(-5, 105)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
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
# the internal build contract 2.1: bold is allowed only for the leading "**Figure N.**" and
# for panel letters.  the internal build contract 4.3: every number deleted from a canvas is
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


def write_captions(store: dict) -> "list[str]":
    stats = store["fig1_stats"]
    rel = store["fig1_rel"]
    mae = store["fig3_mae"]
    f1 = store["fig3_f1"]
    thr = store["fig3_thr"]
    nk = store["fig1_motiv_n"]

    def rl(stratum, tool):
        return rel[(stratum, tool)]

    cc = thr[("Set C-clean", "magicc_v5")]
    tot_n = sum(mae[(s, "magicc_v5", "completeness")]["n"] for s, _, _ in BENCH_SETS)
    tot_k = sum(mae[(s, "magicc_v5", "completeness")]["k"] for s, _, _ in BENCH_SETS)
    paths = []

    # ---------------- Figure 1 ----------------
    lines = []
    add = lines.append
    add("**Figure 1.** Existing tools underestimate contamination from divergent sources.")
    add("")
    add("**a**, Design of the three motivating datasets, each n = 1,000 synthetic genomes: "
        f"Set A, a completeness gradient from 50 % to 100 % without contamination ({nk['A'][1]} "
        f"reference-genome clusters); Set B, a contamination gradient from 0 % to 80 % at 100 % "
        f"completeness ({nk['B'][1]} clusters); Set C, a mixture spanning 50-100 % completeness "
        f"and 0-100 % contamination ({nk['C'][1]} clusters).")
    add("")
    add("**b**, Absolute completeness error for CheckM2, CoCoPyE and DeepCheck on the three "
        "datasets. Boxes, interquartile range and median; whiskers, 5th-95th percentiles; the "
        "overlaid marker is the mean absolute error (MAE) with its 95 % confidence "
        f"interval. Set C: CheckM2 {stats[('C','checkm2','completeness')]['mae']:.2f} pp "
        f"[{stats[('C','checkm2','completeness')]['lo']:.2f}-{stats[('C','checkm2','completeness')]['hi']:.2f}], "
        f"CoCoPyE {stats[('C','cocopye','completeness')]['mae']:.2f} pp, DeepCheck "
        f"{stats[('C','deepcheck','completeness')]['mae']:.2f} pp.")
    add("")
    add("**c**, Absolute contamination error, same datasets, tools and glyphs. Contamination "
        "error is an order of magnitude larger than completeness error for all three tools: "
        f"Set B CheckM2 {stats[('B','checkm2','contamination')]['mae']:.2f} pp "
        f"[{stats[('B','checkm2','contamination')]['lo']:.2f}-{stats[('B','checkm2','contamination')]['hi']:.2f}], "
        f"CoCoPyE {stats[('B','cocopye','contamination')]['mae']:.2f} pp, DeepCheck "
        f"{stats[('B','deepcheck','contamination')]['mae']:.2f} pp.")
    add("")
    add("**d**, **e**, Predicted versus true contamination on Sets B and C (n = 1,000 "
        f"genomes, {nk['B'][1]} and {nk['C'][1]} clusters); one point per genome, dashed line = "
        "identity. All three tools fall below identity across the whole gradient.")
    add("")
    add("**f**, Signed contamination error (predicted - true) on Set E by the taxonomic "
        "relatedness of the injected contaminant; boxes and markers as in **b**, and the line "
        "traces each tool across strata. Strata: uncontaminated n = "
        f"{rl('none (uncontaminated)','magicc_v5')['n']:,} genomes "
        f"({rl('none (uncontaminated)','magicc_v5')['k']:,} clusters); within-phylum "
        f"n = {rl('within_phylum','magicc_v5')['n']:,} ({rl('within_phylum','magicc_v5')['k']:,}); "
        f"cross-phylum n = {rl('cross_phylum','magicc_v5')['n']:,} "
        f"({rl('cross_phylum','magicc_v5')['k']:,}). Mean signed error within-phylum to "
        f"cross-phylum: CheckM2 {rl('within_phylum','checkm2')['mean']:.2f} to "
        f"{rl('cross_phylum','checkm2')['mean']:.2f} pp, CoCoPyE "
        f"{rl('within_phylum','cocopye')['mean']:.2f} to {rl('cross_phylum','cocopye')['mean']:.2f}, "
        f"DeepCheck {rl('within_phylum','deepcheck')['mean']:.2f} to "
        f"{rl('cross_phylum','deepcheck')['mean']:.2f}, MAGICC "
        f"{rl('within_phylum','magicc_v5')['mean']:.2f} to "
        f"{rl('cross_phylum','magicc_v5')['mean']:.2f} pp. Marker-based tools lose signal as the "
        "contaminant diverges; MAGICC's bias is small and near-flat across this coarse contrast, "
        "which Fig. 4d refines into six ranks. MAGICC appears in this panel only.")
    add("")
    add("Confidence intervals are 95 % cluster bootstraps over the dominant reference genome "
        "(2,000 iterations); errors in pp. " + DENOM_NOTE)
    paths.append(_write_part("Figure_1", lines))

    over = []
    add = over.append
    add("## Figure 1 - material removed from the legend")
    add("")
    add("**Panel a.** Shading in the schematic indicates the simulated completeness (Set A) or "
        "contamination (Set B) level; dots indicate the unstructured mixture of Set C.")
    add("")
    add("**Panels b, c.** Fliers are omitted from the boxes. The cluster bootstrap uses 2,000 "
        "iterations clustered on the dominant reference genome, and n = 1,000 genomes per "
        f"dataset ({nk['A'][1]}, {nk['B'][1]} and {nk['C'][1]} clusters for Sets A, B and C). "
        "Panel **c** shares the denominators of panel **b**. MAEs and confidence intervals are "
        "read from `results/revision/metrics/ws5.5_table_S2_rebuilt.tsv`.")
    add("")
    add("**Panels d, e.** Points are drawn from "
        "`data/benchmarks/motivating_v2/set_{A,B,C}/` predictions and `metadata.tsv`.")
    add("")
    add("**Panel f.** Marker-based tools recover signal when the contaminant is a close relative "
        "and lose it as the contaminant diverges. This two-level contrast is too coarse to "
        "support invariance to taxonomic distance; it is refined into a six-rank gradient in "
        "Fig. 4d. Source: `results/revision/metrics/ws5.3_signed_errors_by_relatedness.tsv`.")
    paths.append(_write_part("Figure_1_overflow", over))

    # ---------------- Figure 2 ----------------
    # The three accessibility numbers quoted below are read out of the same
    # screen that writes figures/palette_cvd_check.tsv, so the caption and the
    # TSV cannot disagree.
    _pairs = list(itertools.combinations(sorted(set(PHASE_ACCENT)), 2))
    f2_de_normal = min(delta_e(a, b) for a, b in _pairs)
    f2_de_deut = min(delta_e(a, b, "deuteranopia") for a, b in _pairs)
    _tc = make_palette_cvd_check.build()
    _tc = _tc[_tc["check"] == "text_contrast"].copy()
    _tc["cr"] = _tc["contrast_ratio"].astype(float)
    if not bool(_tc["pass"].all()):
        raise ValueError("Figure 2 has a text/fill pair below the WCAG AA 4.5:1 "
                         "target; see figures/palette_cvd_check.tsv")
    f2_n_text = len(_tc)
    _worst = _tc.loc[_tc["cr"].idxmin()]
    f2_cr_min = float(_worst["cr"])
    f2_cr_who = f"{_worst['item_a']} on {_worst['item_b']}".replace("#", "")

    lines = []
    add = lines.append
    add("**Figure 2.** The MAGICC workflow, from reference curation to training and inference.")
    add("")
    add("Phase 1, reference curation: the 732,475 genomes of GTDB are filtered on CheckM2 "
        "completeness >98 %, contamination <2 %, <100 contigs and N50 >20 kbp to 277,183 "
        "high-quality genomes, from which 100,000 are drawn by square-root-proportional "
        "stratified sampling across 110 phyla. Phase 2, k-mer feature selection: 2,000 "
        "representatives (1,000 bacterial, 1,000 archaeal) are annotated with Prodigal and "
        "searched with HMMER against 85 bacterial and 128 archaeal core-gene HMMs, 9-mers are "
        "counted with KMC3, and the 9,249 most prevalent canonical 9-mers are retained. "
        "Phase 3, training-data synthesis: the curated genomes are fragmented into four quality "
        "tiers and injected with within- and cross-phylum contamination under a Dirichlet "
        "allocation, giving the 1,200,000 synthetic genomes on which the released V5 model was "
        "trained (1,000,000 training, 100,000 validation, 100,000 test). Phase 4, feature "
        "extraction, which is also the inference pipeline: an input FASTA is converted to counts "
        "of the 9,249 selected 9-mers plus 7 k-mer summary features, then normalised. Phase 5, "
        "the network: a k-mer branch (9,249 -> 4,096 -> 1,024 -> 256) and a k-mer-statistics "
        "branch (7 -> 32 -> 16) are concatenated (272 -> 128 -> 64) and read out by a two-headed "
        "regressor predicting completeness (50-100 %) and contamination (0-100 %). Prodigal, "
        "HMMER and KMC3 appear only in the one-off feature-selection stage; inference is "
        "annotation-free and consists of Phases 4 and 5 alone. Each phase carries one accent "
        "colour; boxes are a light tint of it with a saturated border and black text, and the "
        "prediction head is the only solid-filled box. Solid arrows are within-phase flow, "
        "dashed arrows cross-phase. Every box is labelled in text at 5 pt or larger, so no step "
        f"depends on colour, and all {f2_n_text} text-on-fill pairs clear the WCAG AA 4.5:1 "
        f"target (tightest {f2_cr_min:.2f}:1); the phase-accent separations are in "
        "`figures/palette_cvd_check.tsv`. " + DENOM_NOTE)
    paths.append(_write_part("Figure_2", lines))

    over = []
    add = over.append
    add("## Figure 2 - material removed from the legend")
    add("")
    add("**Schematic.** The GTDB release is stated in Methods. The minimum pairwise CIE76 dE "
        f"over the distinct phase accents is {f2_de_normal:.1f} under normal vision and "
        f"{f2_de_deut:.1f} under deuteranopic simulation. The 1,200,000 synthetic genomes "
        "are `data/features/magicc_v5_features.h5`; the 1,000,000 training samples are the "
        "800,000 of the earlier V4 stream plus 100,000 further samples at 100 % completeness and "
        "0 % contamination and 100,000 at 100 % completeness and 0-10 % contamination. The "
        "9,249-mer counts are produced by a Numba rolling hash at inference. Phase accents in "
        "order are Phase 1 and Phase 4 blue, Phase 2 orange, Phase 3 green, Phase 5 red; the "
        "phase header bands and the prediction head, the only fills carrying reversed-out white "
        "text, are drawn in a darkened companion of the same hue so that their labels are "
        f"legible. The tightest text-on-fill pair is {f2_cr_who}. Both the phase-accent "
        "separations and the text contrasts are recorded in `figures/palette_cvd_check.tsv`.")
    paths.append(_write_part("Figure_2_overflow", over))

    # ---------------- Figure 3 ----------------
    lines = []
    add = lines.append
    add("**Figure 3.** Benchmark performance on the leakage-free five-set panel.")
    add("")
    add(f"The leakage-free panel is {tot_n:,} synthetic genomes over {tot_k:,} "
        "set-by-reference clusters, n = 1,000 per set: Set A completeness gradient, "
        "Set B contamination gradient, Set C-clean Patescibacteriota, Set D-clean Archaea and "
        "Set E realistic mixture.")
    add("")
    add("**a**, Design of the five datasets; genome and cluster counts are printed per set.")
    add("")
    add("**b**, Per-genome absolute completeness error by tool and dataset. Boxes, interquartile "
        "range and median; whiskers, 5th-95th percentiles; the overlaid marker is the MAE with "
        "its 95 % confidence interval. On Set C-clean, MAGICC "
        f"{mae[('Set C-clean','magicc_v5','completeness')]['mae']:.2f} pp "
        f"[{mae[('Set C-clean','magicc_v5','completeness')]['lo']:.2f}-"
        f"{mae[('Set C-clean','magicc_v5','completeness')]['hi']:.2f}] versus CheckM2 "
        f"{mae[('Set C-clean','checkm2','completeness')]['mae']:.2f} pp; MAGICC's completeness "
        f"R2 there is {store['fig3_r2_cc']:.3f}.")
    add("")
    add("**c**, Per-genome absolute contamination error, same layout and glyphs. The gap is "
        "largest on the two novel-lineage sets (Set C-clean: "
        f"MAGICC {mae[('Set C-clean','magicc_v5','contamination')]['mae']:.2f} pp "
        f"[{mae[('Set C-clean','magicc_v5','contamination')]['lo']:.2f}-"
        f"{mae[('Set C-clean','magicc_v5','contamination')]['hi']:.2f}] versus CheckM2 "
        f"{mae[('Set C-clean','checkm2','contamination')]['mae']:.2f}, CoCoPyE "
        f"{mae[('Set C-clean','cocopye','contamination')]['mae']:.2f} and DeepCheck "
        f"{mae[('Set C-clean','deepcheck','contamination')]['mae']:.2f} pp).")
    add("")
    add("**d**, Predicted versus true contamination on Set C-clean and Set D-clean (n = 1,000 "
        "genomes, 100 clusters each); one point per genome, all four tools, dashed line = "
        "identity.")
    add("")
    add("**e**, MIMAG-inspired three-class quality assignment (high, completeness >=90 % and "
        "contamination <5 %; medium, >=50 % and <10 %; low, otherwise). Macro F1 with its 95 % "
        "confidence interval. Set A contains no low-quality genomes, so its "
        "macro F1 cannot exceed 0.667 (dotted line). MAGICC leads on every dataset (Set C-clean "
        f"{f1[('Set C-clean','magicc_v5')]['f1']:.3f} "
        f"[{f1[('Set C-clean','magicc_v5')]['lo']:.3f}-{f1[('Set C-clean','magicc_v5')]['hi']:.3f}] "
        f"versus {f1[('Set C-clean','checkm2')]['f1']:.3f}, {f1[('Set C-clean','cocopye')]['f1']:.3f} "
        f"and {f1[('Set C-clean','deepcheck')]['f1']:.3f}).")
    add("")
    add("**f**, Error rates at the 5 % contamination boundary, always read as a pair: filled "
        "markers, false-fail rate; open markers, false-pass rate. Denominators are printed under "
        "each dataset. On Set C-clean "
        f"(52 truly clean, 948 truly contaminated) MAGICC pairs a false-fail rate of {cc['ff']:.3f} "
        f"[{cc['ff_lo']:.3f}-{cc['ff_hi']:.3f}] with a false-pass rate of {cc['fp']:.3f} and the "
        f"best balanced accuracy of the four tools ({cc['bal']:.3f} versus "
        f"{thr[('Set C-clean','checkm2')]['bal']:.3f}, {thr[('Set C-clean','cocopye')]['bal']:.3f} "
        f"and {thr[('Set C-clean','deepcheck')]['bal']:.3f}).")
    add("")
    add("Confidence intervals are 95 % cluster bootstraps on the dominant reference genome "
        "(2,000 iterations); R2 is the coefficient of determination; errors in pp. " + DENOM_NOTE)
    paths.append(_write_part("Figure_3", lines))

    over = []
    add = over.append
    add("## Figure 3 - material removed from the legend")
    add("")
    add("**Panel set-up.** Sets C-clean and D-clean replace Sets C and D of the original "
        "submission, whose dominant genomes overlapped the training split; all references here "
        "are drawn from the held-out test split only. Per-set "
        "cluster counts are Set A "
        f"{mae[('Set A','magicc_v5','completeness')]['k']}, Set B "
        f"{mae[('Set B','magicc_v5','completeness')]['k']}, Set C-clean "
        f"{mae[('Set C-clean','magicc_v5','completeness')]['k']}, Set D-clean "
        f"{mae[('Set D-clean','magicc_v5','completeness')]['k']} and Set E "
        f"{mae[('Set E','magicc_v5','completeness')]['k']}.")
    add("")
    add("**Panels b, c.** Fliers are omitted from the boxes; the cluster bootstrap uses 2,000 "
        "iterations clustered on the dominant reference genome, n = 1,000 genomes per dataset. "
        "Panel **c** shares the denominators of panel **b**.")
    add("")
    add("**Panel e.** n = 1,000 genomes per dataset. The Set A ceiling is labelled "
        "'ceiling 0.67' on the canvas, and Set B contains only two medium-quality genomes.")
    add("")
    add("**Panel f.** The false-fail rate counts truly clean genomes called contaminated and "
        "the false-pass rate truly contaminated genomes called clean; the thin vertical line "
        "joins one tool's two rates so that neither can be read alone. The false-fail "
        "denominator is the number of genomes whose true contamination is <5 %, the "
        "false-pass denominator the number whose true contamination is >=5 %. Error bars "
        "are 95 % cluster-bootstrap "
        "confidence intervals. CheckM2 and DeepCheck reach a false-fail rate of 0.000 on Set "
        f"C-clean only by passing {thr[('Set C-clean','checkm2')]['fp']:.1%} and "
        f"{thr[('Set C-clean','deepcheck')]['fp']:.1%} of genuinely contaminated genomes. Set A "
        "contains no truly contaminated genomes, so no false-pass rate is defined there and no "
        "open marker is drawn.")
    add("")
    add("**Sources.** MAEs and confidence intervals from "
        "`results/revision/metrics/definitive_table_5set_wide.tsv`; macro F1 from "
        "`definitive_mimag.tsv`; threshold rates and denominators from "
        "`definitive_thresholds.tsv`; per-genome points from "
        "`data/benchmarks/{set_A_v2,set_B_v2,set_C_clean,set_D_clean,set_E}/` predictions and "
        "`metadata.tsv`.")
    paths.append(_write_part("Figure_3_overflow", over))
    return paths

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
        write_captions(store)
        make_palette_cvd_check.main()
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
