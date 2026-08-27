#!/usr/bin/env python3
"""Shared figure style for the MAGICC Nature Communications resubmission (v3).

Palette
-------
the internal build contract section 9 item 4 reverts every tool comparison to the
palette of the ORIGINAL submission (`manuscript3/scripts/create_figures_v5.py`,
`BENCH_COLORS`).  The hex codes below are copied from that script verbatim; they
are fixed by the author and must not be changed to make a colour-vision check
pass.

The author's instruction is read strictly: `BENCH_COLORS` is the palette for
EVERY tool comparison, the three-tool motivating panels of Figure 1 (b-e)
included.  v4's separate three-tool `MOTIV_COLORS` set is therefore **retired**,
for two substantive reasons.  First, a tool must carry one colour throughout the
paper; under `MOTIV_COLORS` CoCoPyE was teal in Fig. 1b-e and green in Fig. 1f,
a reading hazard inside a single figure.  Second, `MOTIV_COLORS` contained the
worst pair in the whole audit -- motivating CheckM2 #1f77b4 against DeepCheck
#7570b3 at CIE76 Delta-E 0.60 under protanopia and only 21.79 under normal
vision.  Figure 1b-e now draws CheckM2 #1f77b4, CoCoPyE #2ca02c and DeepCheck
#9467bd: the hues those tools carry everywhere else, simply with MAGICC's red
absent.

Four-tool benchmark palette (`BENCH_COLORS`) -- used for EVERY tool comparison,
main text and supplementary, in every panel including Figure 1b-e::

    MAGICC     #d62728  red      marker o   solid
    CheckM2    #1f77b4  blue     marker ^   long dash
    CoCoPyE    #2ca02c  green    marker D   dotted
    DeepCheck  #9467bd  purple   marker v   dash-dot

Pastel companions (`BENCH_BAR_COLORS`), used only as bar / box / patch fills so
that a filled area never competes with the saturated line or marker::

    MAGICC #e88a8a   CheckM2 #6baed6   CoCoPyE #78c679   DeepCheck #b5a8d4

Non-tool series keep the neutral assignments used throughout the package::

    truth / reference  #888888 grey (bar #b0b0b0)   marker x
    GUNC               #B07AA1 mauve                marker s
    holdout model      #9C755F brown                marker s

`AUX` carries the categorical hues for series that encode something OTHER than
a tool (the two metrics in the difference-in-differences panels, the three
contamination types, the five Set G error processes).  They are deliberately
chosen to sit away from all four tool hues so that no auxiliary series can be
read as a tool.

This palette is NOT colour-vision-deficiency safe and nothing here claims that
it is.  Two pairs fall below the 15-unit CIE76 Delta-E screening threshold:

  * CheckM2 #1f77b4 vs DeepCheck #9467bd -- Delta-E 5.31 under protanopia,
    7.81 under deuteranopia (38.19 under normal vision);
  * MAGICC #d62728 vs CoCoPyE #2ca02c -- Delta-E 7.28 under deuteranopia
    (37.42 protanopic, 119.77 normal).

Separation is therefore carried by *redundant*, non-colour cues in every panel:

  * marker shape (above) is drawn in every panel and in every legend;
  * line style differs wherever two series share an axis;
  * every series is directly labelled, and heatmap cells print their value.

The measured Delta-E for every pair under normal, deuteranopic, protanopic and
tritanopic simulation, the panels in which each failing pair actually
co-occurs, and the redundant cue that carries it are written to
``figures/palette_cvd_check.tsv`` by ``scripts/make_palette_cvd_check.py`` and
described in ``figures/colour_statement.md``.

No palette entry is light enough to need a dark outline on white, so ``EDGE``
is empty and ``EDGE.get(tool, PALETTE[tool])`` simply returns the series
colour.

Import from figure scripts:

    from figstyle import (PALETTE, BAR_PALETTE, MARKERS, TOOL_LABEL, AUX,
                          add_panel_label, save_fig, DENOM_NOTE, apply_style)
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = "/path/to/magicc"
RESULTS = os.path.join(PROJECT, "results", "revision")
BENCH = os.path.join(PROJECT, "data", "benchmarks")
RESUB = os.path.join(PROJECT, "nature_communications", "resubmission3")
# The two output directories may be redirected (integrity runs write a second
# copy of every figure to a scratch tree and diff the value ledgers).
FIG_DIR = os.environ.get("MAGICC_FIG_DIR") or os.path.join(RESUB, "figures")
SUPP_FIG_DIR = (os.environ.get("MAGICC_SUPP_FIG_DIR")
                or os.path.join(RESUB, "supp_figures"))
# Set to "1" to write a per-figure ledger of every plotted coordinate next to
# the figure, used to prove that a recolour changed no number.
ARTIST_LEDGER = os.environ.get("MAGICC_ARTIST_LEDGER") == "1"

# ---------------------------------------------------------------------------
# Author-mandated tool palette (the internal build contract section 9 item 4)
# ---------------------------------------------------------------------------
PALETTE = {
    "magicc": "#d62728",      # red
    "magicc_v5": "#d62728",
    "checkm2": "#1f77b4",     # blue
    "cocopye": "#2ca02c",     # green
    "deepcheck": "#9467bd",   # purple
    "gunc": "#B07AA1",        # mauve (GUNC comparator figure only)
    "truth": "#888888",       # grey
    "holdout": "#9C755F",     # brown (validation-artefact models)
}

# Pastel companions -- bar, box and patch FILLS only, never a line or a marker.
BAR_PALETTE = {
    "magicc": "#e88a8a",
    "magicc_v5": "#e88a8a",
    "checkm2": "#6baed6",
    "cocopye": "#78c679",
    "deepcheck": "#b5a8d4",
    "gunc": "#d5b6cd",
    "truth": "#b0b0b0",
    "holdout": "#c8ae9d",
}

MARKERS = {
    "magicc": "o",
    "magicc_v5": "o",
    "checkm2": "^",
    "cocopye": "D",
    "deepcheck": "v",
    "gunc": "s",
    "truth": "x",
    "holdout": "s",
}

# Line style is a second redundant cue wherever two series share an axis.
LINESTYLES = {
    "magicc": "-",
    "magicc_v5": "-",
    "checkm2": (0, (4, 1.4)),
    "cocopye": (0, (1.2, 1.2)),
    "deepcheck": (0, (5, 1.2, 1, 1.2)),
    "gunc": (0, (3, 1, 1, 1, 1, 1)),
    "truth": (0, (2.5, 1.5)),
    "holdout": (0, (1, 1)),
}

# ---------------------------------------------------------------------------
# Auxiliary categorical hues -- series that encode something that is NOT a tool
# (metric, contamination type, injected error process).  Every one of them is
# at least 26 CIE76 Delta-E from all four tool hues under normal vision, so an
# auxiliary series can never be mistaken for a tool, and each carries its own
# marker shape.
# ---------------------------------------------------------------------------
AUX = {
    "navy": "#08306B",
    "orange": "#F28E2B",
    "slate": "#4D4D4D",
    "cyan": "#17becf",
    "mauve": "#B07AA1",
}

# No palette entry is light enough to need a dark outline on white, so this is
# deliberately empty; it is kept so that ``EDGE.get(tool, PALETTE[tool])`` and
# ``if tool in EDGE`` keep working in the figure scripts.
EDGE: "dict[str, str]" = {}

TOOL_LABEL = {
    "magicc": "MAGICC",
    "magicc_v5": "MAGICC",
    "checkm2": "CheckM2",
    "cocopye": "CoCoPyE",
    "deepcheck": "DeepCheck",
    "gunc": "GUNC",
    "truth": "Truth",
    "holdout": "Holdout model",
}

TOOL_ORDER = ["magicc", "checkm2", "cocopye", "deepcheck"]

# Ordered lightness ramp for MIMAG-inspired classes (inherently CVD-safe)
MIMAG_COLORS = {"high": "#08306B", "medium": "#4292C6", "low": "#BDD7E7"}

# Diverging ramp for signed-error heatmaps: blue (under) - neutral - red (over),
# anchored on the two extreme hues of the reverted palette.  Every cell of every
# heatmap that uses this ramp also prints its own value.
DIVERGING = ["#1f77b4", "#DCE3EC", "#F2F2F2", "#F6C6A8", "#d62728"]

SEQUENTIAL_CMAP = "cividis"   # CVD-optimised

# Figure 2 workflow schematic: one accent per phase, in phase order.  Figure 2
# encodes no tool, so the internal build contract section 9 item 4 (which reverts the
# TOOL comparisons) does not reach it and the internal build contract section 4.2 still
# governs these five accents.  They are schematic furniture and never data.
PHASE_ACCENT = ["#4E79A7", "#F28E2B", "#59A14F", "#4E79A7", "#E15759"]

# Darkened companions of the same five hues, used ONLY as the schematic's
# header-band fills and as the fill of its one solid box.  White label text on
# the undarkened accents misses the WCAG AA 4.5:1 target (white on #E15759 is
# 3.68:1, on #59A14F 3.16:1, on #F28E2B 2.42:1), so the *furniture* is darkened
# rather than the text left unreadable.  Hue is preserved and these values never
# encode data.  Measured white-on-fill contrast, recomputed into
# figures/palette_cvd_check.tsv on every build:
#     #3D5F84  6.63:1     #A85F16  4.87:1
#     #3F7439  5.57:1     #A6383A  6.47:1
PHASE_BAND = ["#3D5F84", "#A85F16", "#3F7439", "#3D5F84", "#A6383A"]

ARROW_INK = "#4D4D4D"

# Sentinel default property cycle.  Every series in this package sets its own
# colour explicitly; anything that falls through to matplotlib's default cycle
# is a bug, and this magenta makes it impossible to miss in a rendered figure.
# ``scripts/`` is screened for it after every build.
CYCLE_SENTINEL = "#FF00FF"

# ---------------------------------------------------------------------------
# rcParams -- Nature Communications / Nature Methods conventions
# ---------------------------------------------------------------------------
STYLE = {
    "font.family": "sans-serif",
    # Arial first.  This host has no Arial, so the next two entries are its
    # metric-compatible clones (identical advance widths and glyph shapes);
    # DejaVu Sans is the last-resort matplotlib default and is NOT
    # Arial-metric, so a build that falls through to it should be re-run on a
    # host with Arial installed.
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
                        "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    # 0.05 in of padding on both sides added 2.54 mm to every rendered page and
    # pushed four figures past the 183 mm Nature Communications cap.  Exactly 0
    # is too tight: matplotlib's mathtext measurement under-reports the width of
    # Figure 5's "log$_2$(size / phylum median)" axis label by a little over
    # 0.02 in, so at 0 the closing parenthesis was shaved off at the page edge
    # and at 0.02 in it still landed on the last pixel column.  0.025 in
    # (0.635 mm per side, 1.27 mm on the width) is the smallest padding that
    # leaves every glyph of every figure clear of the page boundary, verified by
    # scanning the outer pixel row and column of all 22 rendered PNGs.
    "savefig.pad_inches": 0.025,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,   # TrueType, editable text in the PDF
    "ps.fonttype": 42,
    # Not a palette: a tripwire (see CYCLE_SENTINEL above).
    "axes.prop_cycle": cycler(color=[CYCLE_SENTINEL]),
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


apply_style()


def tint(hex_colour: str, frac: float = 0.15) -> str:
    """`frac` of `hex_colour` composited over white, as a hex string."""
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    out = [int(round(255 + (c - 255) * frac)) for c in rgb]
    return "#{:02X}{:02X}{:02X}".format(*out)


def bar_colour(tool: str) -> str:
    """Pastel fill for `tool` (BENCH_BAR_COLORS)."""
    return BAR_PALETTE.get(tool, tint(PALETTE[tool], 0.55))


def tool_kw(tool: str, **extra):
    """Standard keyword arguments for plotting one tool's series."""
    kw = dict(color=PALETTE[tool], marker=MARKERS[tool], label=TOOL_LABEL[tool])
    if tool in EDGE:
        kw["markeredgecolor"] = EDGE[tool]
        kw["markeredgewidth"] = 0.5
    kw.update(extra)
    return kw


# ---------------------------------------------------------------------------
# Colour-vision-deficiency arithmetic (Machado, Oliveira & Fernandes 2009,
# severity 1.0, applied in LINEAR RGB; CIE76 Delta-E in CIE L*a*b* under D65).
# Mirrors scripts/101_metrics_framework.py exactly so that any script can run
# the check without importing the analysis framework.
# ---------------------------------------------------------------------------
CVD_MATRICES = {
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
VISION = ("normal", "deuteranopia", "protanopia", "tritanopia")


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate_cvd(hex_colour: str, kind: str) -> str:
    if kind == "normal":
        return hex_colour.upper()
    lin = srgb_to_linear(hex_to_rgb(hex_colour))
    out = linear_to_srgb(CVD_MATRICES[kind] @ lin)
    return "#{:02X}{:02X}{:02X}".format(*(np.round(out * 255).astype(int)))


def hex_to_lab(h: str) -> np.ndarray:
    xyz = _RGB2XYZ @ srgb_to_linear(hex_to_rgb(h)) / _WHITE_D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a: str, b: str, vision: str = "normal") -> float:
    return float(np.linalg.norm(hex_to_lab(simulate_cvd(a, vision))
                                - hex_to_lab(simulate_cvd(b, vision))))


def relative_luminance(h: str) -> float:
    r = srgb_to_linear(hex_to_rgb(h))
    return float(0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2])


def contrast_ratio(a: str, b: str) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def readable_ink(background: str, dark: str = "#111111", light: str = "#FFFFFF") -> str:
    """Whichever of `dark` / `light` has the higher WCAG contrast on `background`."""
    return dark if contrast_ratio(dark, background) >= contrast_ratio(light, background) else light


# ---------------------------------------------------------------------------
# Denominator note -- must appear in every figure caption (R1-M5)
# ---------------------------------------------------------------------------
DENOM_NOTE = (
    "Denominators: completeness (%) = retained dominant-genome bp / full reference "
    "length of the dominant genome x 100; contamination (%) = total contaminant bp / "
    "the same denominator x 100. Both share one denominator and are independent "
    "measures."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_panel_label(ax, label, x=-0.14, y=1.10, fontsize=9):
    """Bold lowercase panel label in the Nature style."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# Plotted-geometry ledger
#
# Colour is the ONLY thing this rebuild is allowed to change, so every figure
# can be proved unchanged by dumping the data coordinates of every artist it
# contains and diffing the dump against the same dump from the previous build.
# Nothing colour-valued is recorded, so an identical dump means an identical
# figure up to colour.  Enabled by MAGICC_ARTIST_LEDGER=1.
# ---------------------------------------------------------------------------
def _num(a) -> str:
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        return ""
    return ",".join("nan" if not np.isfinite(v) else f"{v:.9g}" for v in a)


def artist_ledger_rows(fig, name: str) -> "list[tuple[str, str]]":
    rows: "list[tuple[str, str]]" = []
    for ai, ax in enumerate(fig.axes):
        k = f"{name}|ax{ai:02d}"
        rows.append((f"{k}|xlim", _num(ax.get_xlim())))
        rows.append((f"{k}|ylim", _num(ax.get_ylim())))
        rows.append((f"{k}|xscale", ax.get_xscale()))
        rows.append((f"{k}|yscale", ax.get_yscale()))
        rows.append((f"{k}|xlabel", ax.get_xlabel()))
        rows.append((f"{k}|ylabel", ax.get_ylabel()))
        rows.append((f"{k}|title", ax.get_title()))
        rows.append((f"{k}|xticks", _num(ax.get_xticks())))
        rows.append((f"{k}|yticks", _num(ax.get_yticks())))
        rows.append((f"{k}|xticklabels",
                     "␟".join(t.get_text() for t in ax.get_xticklabels())))
        rows.append((f"{k}|yticklabels",
                     "␟".join(t.get_text() for t in ax.get_yticklabels())))
        for li, ln in enumerate(ax.get_lines()):
            rows.append((f"{k}|line{li:03d}|xy", _num(ln.get_xydata())))
            rows.append((f"{k}|line{li:03d}|marker", str(ln.get_marker())))
            rows.append((f"{k}|line{li:03d}|ls", str(ln.get_linestyle())))
        for ci, col in enumerate(ax.collections):
            try:
                off = np.asarray(col.get_offsets(), dtype=float)
            except Exception:
                off = np.empty((0, 2))
            if off.size:
                rows.append((f"{k}|coll{ci:03d}|offsets", _num(off)))
            try:
                verts = [p.vertices for p in col.get_paths()]
            except Exception:
                verts = []
            if verts:
                rows.append((f"{k}|coll{ci:03d}|verts",
                             _num(np.vstack([v for v in verts if v.size]))))
            arr = col.get_array()
            if arr is not None:
                rows.append((f"{k}|coll{ci:03d}|array",
                             _num(np.asarray(arr, dtype=float))))
        for pi, pa in enumerate(ax.patches):
            try:
                v = pa.get_path().transformed(pa.get_patch_transform()).vertices
            except Exception:
                v = np.empty((0, 2))
            rows.append((f"{k}|patch{pi:03d}|verts", _num(v)))
        for ii, im in enumerate(ax.images):
            rows.append((f"{k}|image{ii:03d}|array",
                         _num(np.asarray(im.get_array(), dtype=float))))
            rows.append((f"{k}|image{ii:03d}|extent", _num(im.get_extent())))
        for ti, tx in enumerate(ax.texts):
            rows.append((f"{k}|text{ti:03d}",
                         f"{_num(tx.get_position())}␟{tx.get_text()}"))
        leg = ax.get_legend()
        if leg is not None:
            rows.append((f"{k}|legend",
                         "␟".join(t.get_text() for t in leg.get_texts())))
    for ti, tx in enumerate(fig.texts):
        rows.append((f"{name}|figtext{ti:03d}",
                     f"{_num(tx.get_position())}␟{tx.get_text()}"))
    return rows


def _write_artist_ledger(fig, name: str, out_dir: str) -> None:
    d = os.path.join(out_dir, "_values")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}_artists.tsv"), "w") as fh:
        fh.write("key\tvalue\n")
        for k, v in artist_ledger_rows(fig, name):
            fh.write(f"{k}\t{v}\n")


def save_fig(fig, name, out_dir=None, png=True, pdf=True):
    """Save a figure as a 400-dpi PNG (for the Word file) and a vector PDF."""
    out_dir = out_dir or FIG_DIR
    os.makedirs(out_dir, exist_ok=True)
    if ARTIST_LEDGER:
        _write_artist_ledger(fig, name, out_dir)
    paths = []
    if pdf:
        p = os.path.join(out_dir, f"{name}.pdf")
        fig.savefig(p)
        paths.append(p)
    if png:
        p = os.path.join(out_dir, f"{name}.png", )
        fig.savefig(p, dpi=400)
        paths.append(p)
    plt.close(fig)
    print("  saved " + ", ".join(os.path.basename(p) for p in paths))
    return paths


def hline0(ax, **kw):
    """Zero reference line for signed-error panels."""
    style = dict(color="0.35", lw=0.6, ls="--", zorder=0)
    style.update(kw)
    ax.axhline(0, **style)


def diag(ax, lo=0, hi=100, **kw):
    """1:1 diagonal for predicted-versus-true panels."""
    style = dict(color="0.4", lw=0.6, ls="--", zorder=0)
    style.update(kw)
    ax.plot([lo, hi], [lo, hi], **style)
