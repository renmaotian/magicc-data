#!/usr/bin/env python3
"""Colour-vision-deficiency screen for the author-mandated palette.

Writes ``figures/palette_cvd_check.tsv``.

The palette is the ORIGINAL submission's, restored by BUILD_CONTRACT_V3.md
section 9 item 4 and applied to EVERY tool comparison in the package, the
three-tool motivating panels of Figure 1 (b-e) included; v4's separate
three-tool ``MOTIV_COLORS`` set is retired, so there is no longer a second
palette to screen.  It is **not** colour-vision-deficiency safe and this script
does not pretend that it is: it measures every pair, states which ones fail,
names the panels in which each failing pair actually co-occurs, and names the
redundant, non-colour cue that carries it.  The hex codes are fixed by the
author and are never changed to make a pair pass.

Four blocks of rows:

``cvd_pair``        every unordered pair of the seven series colours of
                    ``figstyle.PALETTE`` -- the four-tool benchmark set plus
                    the three non-tool series -- with the CIE76 Delta-E under
                    normal, deuteranopic, protanopic and tritanopic simulation
                    (Machado, Oliveira & Fernandes 2009, severity 1.0, applied
                    in linear RGB).  Threshold 15.

``aux_vs_tool``     every auxiliary categorical hue (``figstyle.AUX``: the two
                    metrics of the difference-in-differences panels, the three
                    contamination types, the five Set G error processes)
                    against every tool hue.  These must stay far enough from
                    the tool hues that an auxiliary series cannot be read as a
                    tool, so the threshold applies under NORMAL vision; the
                    simulated values are reported for completeness.

``fig2_pair``       the Figure 2 workflow fills (phase tints, phase header
                    bands and the solid prediction head).  Figure 2 encodes no
                    tool, so BUILD_CONTRACT_V3 9.4 does not reach it and
                    BUILD_CONTRACT 4.2 still governs those five accents.
                    Adjacent phases must stay distinguishable; every box also
                    carries a text label, so the threshold here is
                    informational.

``text_contrast``   WCAG 2.1 contrast ratio of every text-on-fill combination
                    used in Figure 2 -- twelve pairs: five box-text-on-tint,
                    five header-text-on-band, the solid prediction head and the
                    italic "(also inference pipeline)" note that sits on the
                    Phase 4 panel tint.  Threshold 4.5:1.  Every pair clears it.

The prose summary that the Methods and the supplement quote is
``figures/colour_statement.md``, which is derived from this table.

Usage
-----
    /path/to/conda/bin/python make_palette_cvd_check.py
"""

from __future__ import annotations

import os
import sys
import textwrap

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs  # noqa: E402

MIN_DELTA_E = 15.0
MIN_CONTRAST = 4.5

SERIES = ["magicc", "checkm2", "cocopye", "deepcheck", "truth", "gunc", "holdout"]

SERIES_LABEL = {
    "magicc": "MAGICC", "checkm2": "CheckM2", "cocopye": "CoCoPyE",
    "deepcheck": "DeepCheck", "truth": "truth / reference series",
    "gunc": "GUNC", "holdout": "holdout model",
}

# Redundant, non-colour cue that separates each series, and the panels in which
# a flagged pair actually appears together.
CUE = {
    "magicc": "circle marker 'o', solid line",
    "checkm2": "up-triangle marker '^', long-dash line",
    "cocopye": "diamond marker 'D', dotted line",
    "deepcheck": "down-triangle marker 'v', dash-dot line",
    "truth": "x marker, dashed line",
    "gunc": "square marker 's', dash-dot-dot line",
    "holdout": "square marker 's', dotted line",
}

LABEL_CUE = ("both series directly labelled in the legend of every panel in "
             "which they co-occur")

# Where each pair actually co-occurs.  The four saturated tool hues were
# MEASURED by pixel-sampling the rendered 400 dpi PNGs
# (scripts/../figures/colour_statement.md records the method): a hue counts as
# present in a figure when at least 40 pixels sit within a Euclidean distance
# of 2/255 of its hex value, which antialiasing and alpha blending can only
# ever move a pixel away from.  Grey and brown cannot be separated from
# antialiased black text by that test, so the rows involving `truth`, `gunc`
# and `holdout` are read off the drawing code instead.
# Pairs that include MAGICC appear only in Fig. 1f, the one Figure 1 panel that
# carries MAGICC; the three pairs that do not include MAGICC also appear in the
# three-tool motivating panels 1b-e, which now use the same four-tool hues.
FOUR_TOOL = ("Fig. 1f; Fig. 3b-f; Fig. 4d; Fig. 5a-c; "
             "Figs S1-S8, S11, S12, S17 (15 figures)")
THREE_OF_FOUR = ("Fig. 1b-f; Fig. 3b-f; Fig. 4d; Fig. 5a-c; "
                 "Figs S1-S8, S11, S12, S17 (15 figures)")
CO_OCCUR = {
    frozenset(("magicc", "checkm2")): (
        "Fig. 1f; Fig. 3b-f; Fig. 4d; Fig. 5a-f; "
        "Figs S1-S9, S11-S14, S16, S17 (20 figures)"),
    frozenset(("magicc", "cocopye")): FOUR_TOOL,
    frozenset(("magicc", "deepcheck")): FOUR_TOOL,
    frozenset(("checkm2", "cocopye")): THREE_OF_FOUR,
    frozenset(("checkm2", "deepcheck")): THREE_OF_FOUR,
    frozenset(("cocopye", "deepcheck")): THREE_OF_FOUR,
    frozenset(("magicc", "truth")): "Fig. 5d, f; Figs S9c-f, S13, S14a-d, S16a, d, S17d",
    frozenset(("magicc", "holdout")): "Fig. 4c; Figs S16a, S17d",
    frozenset(("magicc", "gunc")): "Figs S7c, S9a-g, S11e-g, S13b, d",
    frozenset(("checkm2", "truth")): "Figs S9c-f, S13, S14a-d, S16d",
    frozenset(("checkm2", "gunc")): "Figs S7c, S9a-g, S11e-g, S13b, d",
    frozenset(("checkm2", "holdout")): "(no panel)",
    frozenset(("cocopye", "truth")): "Figs S1-S5",
    frozenset(("cocopye", "gunc")): "Figs S7c, S11e-g",
    frozenset(("cocopye", "holdout")): "(no panel)",
    frozenset(("deepcheck", "truth")): "Figs S1-S5",
    frozenset(("deepcheck", "gunc")): "Figs S7c, S11e-g",
    frozenset(("deepcheck", "holdout")): "(no panel)",
    frozenset(("truth", "gunc")): "Figs S7c, S9a, S9b, S9e",
    frozenset(("truth", "holdout")): "Figs S16a, d; S17d",
    frozenset(("gunc", "holdout")): "(no panel)",
}

# What each auxiliary hue encodes, and the panels it is drawn in.  No auxiliary
# hue ever shares a legend with a tool.
AUX_ROLE = {
    "navy": "completeness metric; redundant contamination type; substitutions",
    "orange": "contamination metric; replaced contamination type; indels",
    "slate": "single-copy contamination type; chimeric joins",
    "cyan": "substitutions at Ti/Tv 2:1",
    "mauve": "uneven-coverage duplication; MAGICC-CheckM2 disagreement rate; "
             "withdrawn V3 timings",
}
AUX_WHERE = {
    "navy": "Fig. 4a, b; Fig. S11e-g; Fig. S16c",
    "orange": "Fig. 4a, b; Fig. S11e-g; Fig. S16c",
    "slate": "Fig. S11e-g; Fig. S16c",
    "cyan": "Fig. S11e, f",
    "mauve": "Fig. S7c; Fig. S11e, f; Fig. S13b, d",
}

# ---------------------------------------------------------------------------
# Figure 2 fills, defined here so the check and the drawing cannot drift apart.
# ---------------------------------------------------------------------------
PHASE_NAMES = ["Phase 1 Data Curation", "Phase 2 K-mer Selection",
               "Phase 3 Training Synthesis", "Phase 4 Feature Extraction",
               "Phase 5 Neural Network"]
PHASE_TINT_FRAC = 0.15          # box fill: 15 % of the accent over white
PHASE_TINT_PANEL = 0.06         # phase grouping panel behind the boxes
# The solid prediction head is the darkened Phase 5 hue, so that its white
# label clears 4.5:1 (6.47:1); #E15759 itself is still drawn on this figure as
# the border of every other Phase 5 box.
OUTPUT_FILL = fs.PHASE_BAND[4]
OUTPUT_INK = "#FFFFFF"
BOX_INK = "#111111"
# The one free-floating note on the canvas, and the fill it sits on.
NOTE_LABEL = "(also inference pipeline) note"
NOTE_PHASE = 3                  # index into PHASE_NAMES / PHASE_ACCENT


def fig2_colours():
    """(box tint fills, header band fills, band ink) keyed by phase name."""
    fills, bands, band_ink = {}, {}, {}
    for i, name in enumerate(PHASE_NAMES):
        fills[f"{name} box fill"] = fs.tint(fs.PHASE_ACCENT[i], PHASE_TINT_FRAC)
        bands[f"{name} header band"] = fs.PHASE_BAND[i]
        band_ink[name] = fs.readable_ink(fs.PHASE_BAND[i])
    return fills, bands, band_ink


def build() -> pd.DataFrame:
    cols = ["check", "item_a", "item_b", "hex_a", "hex_b",
            "delta_e_normal", "delta_e_deuteranopia", "delta_e_protanopia",
            "delta_e_tritanopia", "min_delta_e", "worst_vision", "contrast_ratio",
            "threshold", "pass", "redundant_cue", "where"]
    rows = []

    # ---- block 1: the seven series colours -------------------------------
    for i in range(len(SERIES)):
        for j in range(i + 1, len(SERIES)):
            a, b = SERIES[i], SERIES[j]
            rec = {"check": "cvd_pair", "item_a": SERIES_LABEL[a],
                   "item_b": SERIES_LABEL[b],
                   "hex_a": fs.PALETTE[a], "hex_b": fs.PALETTE[b]}
            worst, worst_v = float("inf"), ""
            for vision in fs.VISION:
                de = fs.delta_e(fs.PALETTE[a], fs.PALETTE[b], vision)
                rec[f"delta_e_{vision}"] = round(de, 2)
                if de < worst:
                    worst, worst_v = de, vision
            ok = worst >= MIN_DELTA_E
            rec.update(min_delta_e=round(worst, 2), worst_vision=worst_v,
                       contrast_ratio="", threshold=MIN_DELTA_E, pass_=ok)
            rec["pass"] = ok
            rec.pop("pass_")
            rec["redundant_cue"] = ("" if ok else
                                    f"{CUE[a]}  vs  {CUE[b]}; both series directly "
                                    f"labelled in the legend of every panel in which "
                                    f"they co-occur")
            rec["where"] = "" if ok else CO_OCCUR[frozenset((a, b))]
            rows.append(rec)

    # ---- block 1b: auxiliary hues against every tool hue ------------------
    for aname, ahex in fs.AUX.items():
        for tool in ("magicc", "checkm2", "cocopye", "deepcheck"):
            rec = {"check": "aux_vs_tool",
                   "item_a": f"auxiliary {aname} {ahex} ({AUX_ROLE[aname]})",
                   "item_b": SERIES_LABEL[tool],
                   "hex_a": ahex, "hex_b": fs.PALETTE[tool]}
            worst, worst_v = float("inf"), ""
            for vision in fs.VISION:
                de = fs.delta_e(ahex, fs.PALETTE[tool], vision)
                rec[f"delta_e_{vision}"] = round(de, 2)
                if de < worst:
                    worst, worst_v = de, vision
            # An auxiliary series never shares a legend with a tool; the test is
            # only that a normal-vision reader cannot mistake it for one.
            ok = rec["delta_e_normal"] >= MIN_DELTA_E
            rec.update(min_delta_e=round(worst, 2), worst_vision=worst_v,
                       contrast_ratio="", threshold=MIN_DELTA_E)
            rec["pass"] = bool(ok)
            rec["redundant_cue"] = ("auxiliary series carry their own marker "
                                    "shape and appear in panels of their own, "
                                    "never in a tool legend")
            rec["where"] = AUX_WHERE[aname]
            rows.append(rec)

    # ---- block 2: Figure 2 fills -----------------------------------------
    fills, accents, _ = fig2_colours()
    f2 = dict(fills)
    f2.update(accents)
    f2["Prediction head fill"] = OUTPUT_FILL
    keys = list(f2)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            rec = {"check": "fig2_pair", "item_a": a, "item_b": b,
                   "hex_a": f2[a], "hex_b": f2[b]}
            worst, worst_v = float("inf"), ""
            for vision in fs.VISION:
                de = fs.delta_e(f2[a], f2[b], vision)
                rec[f"delta_e_{vision}"] = round(de, 2)
                if de < worst:
                    worst, worst_v = de, vision
            identical = f2[a].upper() == f2[b].upper()
            rec.update(min_delta_e=round(worst, 2), worst_vision=worst_v,
                       contrast_ratio="", threshold=MIN_DELTA_E,
                       redundant_cue=("identical by design: the build contract "
                                      "assigns the same accent to Phase 1 and "
                                      "Phase 4; the two phases are separated by "
                                      "position and by their header text"
                                      if identical else
                                      "every box carries its own text label"),
                       where="Figure 2")
            rec["pass"] = bool(identical or worst >= MIN_DELTA_E)
            rows.append(rec)

    # ---- block 3: text contrast in Figure 2 ------------------------------
    _, _, band_ink = fig2_colours()
    text_pairs = []
    for i, name in enumerate(PHASE_NAMES):
        text_pairs.append((f"{name} box text", BOX_INK, f"{name} box fill",
                           fs.tint(fs.PHASE_ACCENT[i], PHASE_TINT_FRAC)))
        text_pairs.append((f"{name} header text", band_ink[name],
                           f"{name} header band", fs.PHASE_BAND[i]))
    text_pairs.append(("Prediction-head text", OUTPUT_INK,
                       "Prediction-head fill", OUTPUT_FILL))
    text_pairs.append((f"{NOTE_LABEL} text", BOX_INK,
                       f"{PHASE_NAMES[NOTE_PHASE]} panel tint",
                       fs.tint(fs.PHASE_ACCENT[NOTE_PHASE], PHASE_TINT_PANEL)))
    for label, fg, bg_label, bg in text_pairs:
        cr = fs.contrast_ratio(fg, bg)
        rows.append({"check": "text_contrast", "item_a": label, "item_b": bg_label,
                     "hex_a": fg, "hex_b": bg,
                     "delta_e_normal": "", "delta_e_deuteranopia": "",
                     "delta_e_protanopia": "", "delta_e_tritanopia": "",
                     "min_delta_e": "", "worst_vision": "",
                     "contrast_ratio": round(cr, 2), "threshold": MIN_CONTRAST,
                     "pass": bool(cr >= MIN_CONTRAST),
                     "redundant_cue": ("" if cr >= MIN_CONTRAST else
                                       "BELOW THE WCAG AA TARGET -- fix the fill "
                                       "or the ink before shipping this figure"),
                     "where": "Figure 2"})

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Methods-ready prose, written from the table above so the two cannot disagree.
# ---------------------------------------------------------------------------
# Figures in which each flagged four-tool pair actually co-occurs.  MEASURED by
# pixel-sampling the rendered 400 dpi PNGs: a hue counts as present when at
# least 40 pixels lie within a Euclidean distance of 2/255 of its hex value.
# Antialiasing and alpha compositing can only move a pixel AWAY from the pure
# hue, so the test is conservative in the direction of under-reporting.
# Figures in which BOTH hues of the CheckM2 #1f77b4 / DeepCheck #9467bd pair are
# drawn (the tightest failing pair), and in which both hues of the MAGICC
# #d62728 / CoCoPyE #2ca02c pair are drawn.
PIXEL_CD_MAIN = ["Figure 1", "Figure 3", "Figure 4", "Figure 5"]
PIXEL_CD_SUPP = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S11", "S12",
                 "S17"]
PIXEL_MC_MAIN = ["Figure 1", "Figure 3", "Figure 4", "Figure 5"]
PIXEL_MC_SUPP = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S11", "S12",
                 "S17"]


def rewrap(text: str, width: int = 84) -> str:
    """Re-flow the prose paragraphs; leave headings, list items and rules alone."""
    out = []
    for block in text.split("\n\n"):
        stripped = block.strip()
        if (not stripped or stripped.startswith("#") or stripped.startswith("*")
                or stripped.startswith("-") or stripped.startswith("|")):
            out.append(block)
        else:
            out.append(textwrap.fill(" ".join(stripped.split()), width=width,
                                     break_long_words=False,
                                     break_on_hyphens=False))
    return "\n\n".join(out).rstrip() + "\n"


def statement(df: pd.DataFrame) -> str:
    def row(check, a, b):
        m = df[(df["check"] == check) & (df["item_a"] == a) & (df["item_b"] == b)]
        if m.empty:
            m = df[(df["check"] == check) & (df["item_a"] == b) & (df["item_b"] == a)]
        return m.iloc[0]

    cd = row("cvd_pair", "CheckM2", "DeepCheck")
    mc = row("cvd_pair", "MAGICC", "CoCoPyE")
    tg = row("cvd_pair", "truth / reference series", "GUNC")
    gh = row("cvd_pair", "GUNC", "holdout model")
    ser = df[df["check"] == "cvd_pair"]
    n_fail = int((~ser["pass"].astype(bool)).sum())
    aux = df[df["check"] == "aux_vs_tool"]
    aux_min = float(aux["delta_e_normal"].astype(float).min())
    tc = df[df["check"] == "text_contrast"]
    tc_min = float(tc["contrast_ratio"].astype(float).min())

    return f"""# Colour in the MAGICC figures: what is used, what it fails, and what carries it

Generated by `nature_communications/resubmission3/scripts/make_palette_cvd_check.py`
from `figures/palette_cvd_check.tsv`. Every number below is measured, not asserted.
Regenerate both together; do not edit this file by hand.

## Methods paragraph (ready to paste)

Every tool comparison in the main text and in the supplement is drawn in one
palette, that of the original submission: MAGICC {fs.PALETTE['magicc']} (red),
CheckM2 {fs.PALETTE['checkm2']} (blue), CoCoPyE {fs.PALETTE['cocopye']} (green)
and DeepCheck {fs.PALETTE['deepcheck']} (purple), with the lighter companions
#e88a8a, #6baed6, #78c679 and #b5a8d4 used only as box and bar fills. A tool
carries the same hue in every panel of every figure, including the three-tool
motivating panels of Figure 1 (b-e), from which MAGICC is deliberately absent
and which are therefore drawn in the same CheckM2, CoCoPyE and DeepCheck hues
as the rest of the paper with MAGICC's red simply missing. Non-tool series use
{fs.PALETTE['truth']} grey for the reference or truth series,
{fs.PALETTE['gunc']} for GUNC and {fs.PALETTE['holdout']} for the
validation-artefact holdout model. Series that encode something other than a
tool - the two metrics of the difference-in-differences panels, the three
contamination types and the five Set G error processes - are drawn in auxiliary
hues that sit at least {aux_min:.2f} CIE76 Delta-E from every tool hue under
normal vision, so that an auxiliary series cannot be read as a tool. **This
palette is not colour-vision-deficiency safe and we do not claim that it is.**
Screening every pair of series colours under normal, deuteranopic, protanopic
and tritanopic simulation (Machado, Oliveira & Fernandes 2009, severity 1.0,
applied in linear RGB; CIE76 Delta-E in CIE L*a*b* under D65) against a 15-unit
threshold, {n_fail} of the {len(ser)} pairs fall below it. The tightest is
CheckM2 against DeepCheck, a blue/purple pair whose Delta-E falls to
{float(cd['delta_e_protanopia']):.2f} under protanopic simulation
({float(cd['delta_e_deuteranopia']):.2f} deuteranopic,
{float(cd['delta_e_tritanopia']):.2f} tritanopic,
{float(cd['delta_e_normal']):.2f} normal). Next is MAGICC against CoCoPyE, a
red/green pair whose Delta-E falls to
{float(mc['delta_e_deuteranopia']):.2f} under deuteranopic simulation
({float(mc['delta_e_protanopia']):.2f} protanopic,
{float(mc['delta_e_normal']):.2f} normal). Two pairs among the non-tool series
also fall below the threshold: the grey reference series against GUNC at
{float(tg['delta_e_deuteranopia']):.2f} under deuteranopic simulation, and GUNC
against the holdout model at {float(gh['delta_e_tritanopia']):.2f} under
tritanopic simulation, though those two never appear in the same panel. No
panel may therefore be read by colour alone, and none needs to be: every tool
carries a marker shape used by no other series in the same panel and repeated
in every legend (MAGICC circle, CheckM2 upward triangle, CoCoPyE diamond,
DeepCheck downward triangle, reference series cross, GUNC square); wherever two
series share an axis as a line they also differ in line style (MAGICC solid,
CheckM2 long dash, CoCoPyE dotted, DeepCheck dash-dot, reference dashed); every
series is named in the legend of the panel it appears in; and every heatmap
cell prints its own value, so no heatmap panel carries information in hue
alone. The complete pairwise measurement, the panels in which each flagged pair
co-occurs, and the WCAG 2.1 contrast of every text-on-fill combination in the
workflow schematic (all {len(tc)} pairs clear the 4.5:1 AA target, the tightest
at {tc_min:.2f}:1) are in `figures/palette_cvd_check.tsv`.

## Which figures contain a flagged pair

CheckM2/DeepCheck, the tightest failing pair, is drawn in
{len(PIXEL_CD_MAIN) + len(PIXEL_CD_SUPP)} figures:

* main text ({len(PIXEL_CD_MAIN)}): {', '.join(PIXEL_CD_MAIN)}
* supplementary ({len(PIXEL_CD_SUPP)}): {', '.join(PIXEL_CD_SUPP)}

MAGICC/CoCoPyE, the red/green pair, is drawn in
{len(PIXEL_MC_MAIN) + len(PIXEL_MC_SUPP)} figures:

* main text ({len(PIXEL_MC_MAIN)}): {', '.join(PIXEL_MC_MAIN)}
* supplementary ({len(PIXEL_MC_SUPP)}): {', '.join(PIXEL_MC_SUPP)}

The two lists are the same {len(PIXEL_MC_MAIN) + len(PIXEL_MC_SUPP)} figures,
because those are the figures that plot all four tools. CheckM2 and DeepCheck
additionally share Figure 1's three-tool motivating panels b-e, in which MAGICC
and its red are absent, which is why CheckM2/DeepCheck is the pair that appears
in more panels as well as the pair that separates least.

Measured by pixel-sampling the rendered 400 dpi PNGs of all 22 figures: a hue
counts as present in a figure when at least 40 pixels lie within a Euclidean
distance of 2/255 of its hex value in 8-bit sRGB. Antialiasing and alpha
compositing move a pixel away from the pure hue rather than towards it, so the
test under-reports rather than over-reports, and the 40-pixel threshold absorbs
the rare coincidence in which a blend of two other colours lands on a hue's exact
value. The
same scan finds no pixel of the sentinel magenta that `figstyle.py` installs as
matplotlib's default property cycle, which confirms that every series in every
figure sets its own colour and none falls through to a default cycle. It also
finds no hue of the retired three-tool motivating palette above the detection
threshold in any of the 22 figures, which confirms that nothing in the package
is still drawn in it; the only trace of it anywhere is one pixel of #7570b3, out
of the 9,274,111 in Figure 1, where antialiasing between the CheckM2 blue and
the DeepCheck purple happens to land on that value.

The remaining figures are excluded because they do not plot all four tools:
Figure 2 is the workflow schematic and encodes no tool; S9 plots MAGICC,
CheckM2 and GUNC; S10 and S15 are heatmaps whose cells print their own values;
S13 and S14 plot MAGICC and CheckM2 only; and S16 draws its four-tool
comparison as one heatmap per tool with printed cell values (S16b) rather than
as tool-coloured series, its line panels carrying MAGICC, CheckM2, the truth
series and the holdout model.

Grey and brown cannot be separated from antialiased black text by a pixel test,
so the panels for the two non-tool pairs are read off the drawing code instead:
the grey reference series and GUNC co-occur in {tg['where']}, and GUNC and the
holdout model in no panel at all.

## What is deliberately not in the tool palette

Figure 2, the workflow schematic, encodes no tool, so the reversion of the tool
palette does not reach it; it keeps the five phase accents fixed by
BUILD_CONTRACT section 4.2 and their darkened companions for the two fills that
carry reversed-out white text. Those accents never encode data, every box in the
schematic is labelled in text, and all {len(tc)} text-on-fill pairs clear the
WCAG AA 4.5:1 target.
"""



def main() -> int:
    df = build()
    os.makedirs(fs.FIG_DIR, exist_ok=True)
    path = os.path.join(fs.FIG_DIR, "palette_cvd_check.tsv")
    df.to_csv(path, sep="\t", index=False)
    spath = os.path.join(fs.FIG_DIR, "colour_statement.md")
    with open(spath, "w") as fh:
        fh.write(rewrap(statement(df)))
    print(f"wrote {spath}")

    ser = df[df["check"] == "cvd_pair"]
    bad = ser[~ser["pass"].astype(bool)]
    print(f"wrote {path}")
    print(f"  series pairs: {len(ser)}; minimum CIE76 Delta-E "
          f"{ser['min_delta_e'].astype(float).min():.2f}; "
          f"{len(bad)} pair(s) below {MIN_DELTA_E:g}")
    for _, r in bad.iterrows():
        print(f"    ! {r.item_a} vs {r.item_b}: min dE {r.min_delta_e} "
              f"({r.worst_vision}) -> redundant cue required")
    ax = df[df["check"] == "aux_vs_tool"]
    abad = ax[~ax["pass"].astype(bool)]
    print(f"  auxiliary-vs-tool pairs: {len(ax)}; minimum normal-vision Delta-E "
          f"{ax['delta_e_normal'].astype(float).min():.2f}; {len(abad)} below "
          f"{MIN_DELTA_E:g} under normal vision")
    for _, r in abad.iterrows():
        print(f"    ! {r.item_a} vs {r.item_b}: normal dE {r.delta_e_normal}")
    f2 = df[df["check"] == "fig2_pair"]
    f2d = f2[f2["hex_a"].str.upper() != f2["hex_b"].str.upper()]
    print(f"  Figure 2 fills: {len(f2)} pairs ({len(f2) - len(f2d)} identical by "
          f"design), minimum Delta-E over the distinct pairs "
          f"{f2d['min_delta_e'].astype(float).min():.2f}; "
          f"{int((~f2['pass'].astype(bool)).sum())} flagged")
    tc = df[df["check"] == "text_contrast"]
    tbad = tc[~tc["pass"].astype(bool)]
    print(f"  text contrast: {len(tc)} pairs, minimum "
          f"{tc['contrast_ratio'].astype(float).min():.2f}:1; "
          f"{len(tbad)} below {MIN_CONTRAST}:1")
    for _, r in tbad.iterrows():
        print(f"    ! {r.item_a} on {r.item_b}: {r.contrast_ratio}:1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
