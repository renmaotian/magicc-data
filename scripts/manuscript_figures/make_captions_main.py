#!/usr/bin/env python3
"""Assemble ``figures/captions_main.md`` from the per-figure caption parts.

Each figure builder writes its own legend to ``figures/caption_parts/Figure_N.md``
as it renders, so the caption can never quote a number the figure did not read
from ``results/revision/``.  This script concatenates the five parts, checks the
BUILD_CONTRACT rules that apply to legends, and writes the single file the Word
builder consumes.

Checks enforced (a failure is fatal):
  * all five parts exist;
  * every caption ends with ``figstyle.DENOM_NOTE``;
  * the only bold spans are the leading ``**Figure N.**`` and single panel
    letters ``**a**`` (BUILD_CONTRACT 2.1);
  * no caption cites `cold_vs_warm.tsv`, and none reports R2 0.656 for
    set_C_clean completeness (traps T3/T4).

    /path/to/conda/bin/python make_captions_main.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs  # noqa: E402
from figstyle import DENOM_NOTE, FIG_DIR  # noqa: E402

MAX_LEGEND_WORDS = 350          # Nature Communications, per legend

PARTS = os.path.join(FIG_DIR, "caption_parts")
OUT = os.path.join(FIG_DIR, "captions_main.md")
OUT_OVERFLOW = os.path.join(FIG_DIR, "caption_overflow.md")
FIGURES = [1, 2, 3, 4, 5]

def _de(a, b, vision):
    """Measured CIE76 Delta-E, so the legend header can never quote a stale number."""
    return fs.delta_e(fs.PALETTE[a], fs.PALETTE[b], vision)


HEADER = f"""# Main-text figure captions, Figures 1-5

MAGICC, Nature Communications resubmission (resubmission3).

Assembled by `nature_communications/resubmission3/scripts/make_captions_main.py`
from the caption parts written by `make_figures_1_3.py`, `make_figure_4.py` and
`make_figure_5.py`. Every number below was read from a file under
`results/revision/` by the script that drew the panel it describes; nothing here
is transcribed by hand.

Bold is used only for the leading figure number and for panel letters. Errors
and disagreements are in percentage points (pp). R2 is the coefficient of
determination, never a squared Pearson correlation. Quality thresholds are
"MIMAG-inspired" because the rRNA/tRNA criteria of the strict MIMAG definition
cannot be evaluated from these assemblies.

Colour. Every tool comparison uses one palette, that of the original
submission: MAGICC {fs.PALETTE["magicc"]} red, CheckM2 {fs.PALETTE["checkm2"]}
blue, CoCoPyE {fs.PALETTE["cocopye"]} green, DeepCheck
{fs.PALETTE["deepcheck"]} purple, with the pastel companions #e88a8a, #6baed6,
#78c679 and #b5a8d4 as box and bar fills; the reference series is
{fs.PALETTE["truth"]} grey, GUNC {fs.PALETTE["gunc"]} mauve and the holdout
model {fs.PALETTE["holdout"]} brown. A tool keeps the same hue in every panel of
every figure, including the three-tool motivating panels of Figure 1 (b-e),
from which MAGICC is deliberately absent and which therefore carry the same
CheckM2, CoCoPyE and DeepCheck hues as the rest of the paper with MAGICC's red
simply missing. This palette is not colour-vision-deficiency safe and is not
claimed to be: CheckM2 and DeepCheck are a blue/purple pair whose CIE76 Delta-E
falls to {_de("checkm2", "deepcheck", "protanopia"):.2f} under protanopic and
{_de("checkm2", "deepcheck", "deuteranopia"):.2f} under deuteranopic
simulation, and MAGICC and CoCoPyE a red/green pair that falls to
{_de("magicc", "cocopye", "deuteranopia"):.2f} under deuteranopic simulation,
against a screening threshold of 15. No panel may therefore be read by colour
alone, and none needs to be: marker shape and line style are redundant cues
throughout (MAGICC circle and solid line, CheckM2 triangle and long dash,
CoCoPyE diamond and dots, DeepCheck inverted triangle and dash-dot), every
series is directly labelled in its panel legend, and every heatmap cell prints
its own value. The measured Delta-E of every pair under normal, deuteranopic,
protanopic and tritanopic simulation, the panels in which each flagged pair
co-occurs, and the redundant cue that carries it are in
`figures/palette_cvd_check.tsv` and `figures/colour_statement.md`.

---
"""

OVERFLOW_HEADER = """# Main-text figure legends: material moved out of the legend

MAGICC, Nature Communications resubmission (resubmission3).

Nature Communications caps each figure legend at 350 words. The five main-text
legends in `figures/captions_main.md` are at or under that cap; this file holds
every sentence that had to leave them, labelled by figure and by panel, so that
it can be moved into the corresponding supplementary figure note. Nothing here
is new and nothing was dropped: every block is written by the same script that
draws the panel, and every number is still read from the same file under
`results/revision/`.

Kept in the legend in every case: what the panel shows, the n and the
denominators, and the confidence intervals a reader needs to interpret it.
Moved here: source-file paths, per-panel restatements of numbers that also
appear in the Results or in a supplementary table, bootstrap and estimator
detail, mechanism prose, and withdrawal notes.

---
"""

BOLD = re.compile(r"\*\*(.+?)\*\*", flags=re.S)


def word_count(text: str) -> int:
    """Words in a legend as an editor counts them: markdown emphasis stripped."""
    return len(re.sub(r"[*`]", "", text).split())


def check(fig: int, text: str) -> None:
    n = word_count(text)
    if n > MAX_LEGEND_WORDS:
        raise SystemExit(f"Figure {fig}: legend is {n} words, cap is "
                         f"{MAX_LEGEND_WORDS} (BUILD_CONTRACT_V3 1)")
    if not text.rstrip().endswith(DENOM_NOTE):
        raise SystemExit(f"Figure {fig}: caption does not end with figstyle.DENOM_NOTE")
    if not text.startswith(f"**Figure {fig}.**"):
        raise SystemExit(f"Figure {fig}: caption does not open with '**Figure {fig}.**'")
    allowed = {f"Figure {fig}."} | set("abcdefgh")
    bad = [m for m in BOLD.findall(text) if m not in allowed]
    if bad:
        raise SystemExit(f"Figure {fig}: disallowed bold span(s) {bad}")
    low = text.lower()
    if "cold_vs_warm.tsv" in low:
        raise SystemExit(f"Figure {fig}: cites cold_vs_warm.tsv (trap T4)")
    if "0.656" in text:
        raise SystemExit(f"Figure {fig}: quotes the squared-Pearson 0.656 (trap T3)")


def main() -> int:
    blocks = []
    for fig in FIGURES:
        path = os.path.join(PARTS, f"Figure_{fig}.md")
        if not os.path.isfile(path):
            raise SystemExit(
                f"missing {path}; run make_figures_1_3.py, make_figure_4.py and "
                f"make_figure_5.py first")
        text = open(path).read().strip()
        check(fig, text)
        blocks.append(text)

    with open(OUT, "w") as fh:
        fh.write(HEADER + "\n" + "\n\n---\n\n".join(blocks) + "\n")

    # Everything trimmed out of the five legends to meet the 350-word cap,
    # labelled by figure and panel so it can be moved into the corresponding
    # supplementary figure note without re-deriving a single number.
    over = []
    for fig in FIGURES:
        path = os.path.join(PARTS, f"Figure_{fig}_overflow.md")
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}; re-run the figure builders")
        over.append(open(path).read().strip())
    with open(OUT_OVERFLOW, "w") as fh:
        fh.write(OVERFLOW_HEADER + "\n" + "\n\n---\n\n".join(over) + "\n")
    print(f"wrote {OUT_OVERFLOW}  ({len(over)} blocks, "
          f"{sum(word_count(b) for b in over):,} words)")
    counts = {f: word_count(b) for f, b in zip(FIGURES, blocks)}
    print(f"wrote {OUT}  ({len(blocks)} captions, {sum(counts.values()):,} words)")
    for f, n in counts.items():
        print(f"  Figure {f}: {n} words  (cap {MAX_LEGEND_WORDS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
