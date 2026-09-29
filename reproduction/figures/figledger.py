#!/usr/bin/env python3
"""Source-file registry, plotted-value ledger and small drawing helpers shared
by the taxonomy panels (holdout, Set F, CAMI II).

Lifted unchanged from ``make_figures_4_5.py`` of the previous round so that the
main-text figure and the supplementary composite that now split those panels
between them read their numbers through exactly the same code path.

Integrity: every value that is drawn or annotated is written to ``LEDGER`` as
it is drawn, then re-read from its source TSV with an independent ``csv``
reader and compared by :func:`verify` before the calling script exits.  A
missing input, an ambiguous row or a mismatch is a hard failure.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figstyle import AUX, DIVERGING, PALETTE, RESULTS, tint  # noqa: E402

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry / vocabulary
# ---------------------------------------------------------------------------
MM = 1.0 / 25.4
DOUBLE_COL = 183 * MM          # 7.205 in -- Nature double column

DISTANCES = ["species", "genus", "family", "order", "class", "phylum"]
TYPES = ["redundant", "replaced", "single"]
SETF_TOOLS = ["magicc_v5", "checkm2", "cocopye", "deepcheck"]

# Metric colours (leave-one-out DiD panels).  These identify a METRIC, not a
# tool, so they come from figstyle.AUX -- deep navy and orange, both far from
# every tool hue of the restored palette (navy is 31 CIE76 Delta-E from CheckM2
# blue under normal vision, orange 45 from MAGICC red) and 86-118 Delta-E apart
# from each other under all four simulations.  Marker shape is redundant.
METRIC_STYLE = {
    "completeness": dict(color=AUX["navy"], marker="o", label="completeness"),
    "contamination": dict(color=AUX["orange"], marker="^", label="contamination"),
}
# Contamination-type colours: navy / orange / dark slate, again from AUX so that
# a type can never be read as a tool; marker shape and line style redundant.
TYPE_STYLE = {
    "redundant": dict(color=AUX["navy"], marker="o", ls="-"),
    "replaced": dict(color=AUX["orange"], marker="^", ls=(0, (4, 1.4))),
    "single": dict(color=AUX["slate"], marker="v", ls=(0, (1.2, 1.2))),
}

# ---------------------------------------------------------------------------
# Taxonomic-rank styling for the WS11.G genus -> family -> phylum ladder.
#
# Within a ladder panel every marker carries ONE metric hue (METRIC_STYLE, from
# figstyle.AUX), so colour still means "metric" exactly as it does everywhere
# else in the package and a rank can never be read as a tool.  Rank -- an
# ORDINAL variable -- is carried by an ordered lightness ramp of that single
# hue applied to the marker FACE only, with the full-saturation hue always kept
# as the marker edge so that even the lightest face stays legible on white.
# Three further redundant cues repeat the same ordering: marker shape, the
# confidence-interval line style, and the fixed row position inside each
# group's band (genus on top, phylum at the bottom).
# ---------------------------------------------------------------------------
RANKS = ("genus", "family", "phylum")
RANK_FACE = {"genus": 0.45, "family": 0.70, "phylum": 1.00}     # tint over white
RANK_LINE = {"genus": 0.66, "family": 0.84, "phylum": 1.00}
RANK_MARKER = {"genus": "o", "family": "s", "phylum": "D"}
RANK_LS = {"genus": (0, (1.2, 1.0)), "family": (0, (3.0, 1.2)), "phylum": "-"}
RANK_DY = {"genus": +0.28, "family": 0.0, "phylum": -0.28}
RANK_LABEL = {"genus": "genus held out", "family": "family held out",
              "phylum": "phylum held out"}


def rank_style(metric: str, rank: str) -> dict:
    """Marker/line keywords for one rung of the ladder in one metric."""
    hue = METRIC_STYLE[metric]["color"]
    return dict(color=tint(hue, RANK_LINE[rank]), marker=RANK_MARKER[rank],
                mfc=tint(hue, RANK_FACE[rank]), mec=hue, ls=RANK_LS[rank])


FAMILY_LABEL = {
    "Bacteroidota_Muribaculaceae": "Muribaculaceae\n(Bacteroidota)",
    "Bacteroidota_Flavobacteriaceae": "Flavobacteriaceae\n(Bacteroidota)",
    "Campylobacterota_Helicobacteraceae": "Helicobacteraceae\n(Campylobacterota)",
    "Campylobacterota_Arcobacteraceae": "Arcobacteraceae\n(Campylobacterota)",
    "Halobacteriota_halophilic": "Haloarculaceae +\nHaloferacaceae\n(Halobacteriota)",
    "Patescibacteriota_families": "19 CPR families\n(Patescibacteriota)",
}

# ---------------------------------------------------------------------------
# Source files -- absolute, all under results/revision
# ---------------------------------------------------------------------------
SRC = {
    "phy_did": Path(RESULTS) / "holdout" / "lineage_novelty_effect_did.tsv",
    "phy_h2h": Path(RESULTS) / "holdout" / "head_to_head_by_group.tsv",
    "phy_ref": Path(RESULTS) / "holdout" / "per_reference_errors.tsv",
    "fam_did": Path(RESULTS) / "holdout_family" / "lineage_novelty_effect_did.tsv",
    "fam_h2h": Path(RESULTS) / "holdout_family" / "head_to_head_by_group.tsv",
    "fam_vs_phy": Path(RESULTS) / "holdout_family" / "family_vs_phylum_did.tsv",
    "setF_cells": Path(RESULTS) / "set_F" / "set_F_cells_type_x_distance.tsv",
    "setF_marg": Path(RESULTS) / "set_F" / "set_F_marginals.tsv",
    "setF_paired": Path(RESULTS) / "set_F" / "set_F_paired_comparisons.tsv",
    "setF_attr": Path(RESULTS) / "set_F" / "set_F_attribution.tsv",
    "setF_dist": Path(RESULTS) / "set_F" / "set_F_distance_summary.tsv",
    "cami_setF": Path(RESULTS) / "cami2" / "analysis" / "cami2_setF_comparison.tsv",
    "cami_mixed": Path(RESULTS) / "cami2" / "analysis" / "cami2_mixed_by_distance.tsv",
    "cami_slopes": Path(RESULTS) / "cami2" / "analysis" / "cami2_detection_slopes.tsv",
    "cami_wc": Path(RESULTS) / "cami2" / "analysis" / "cami2_wellcontrolled_by_distance.tsv",
    "cami_paired": Path(RESULTS) / "cami2" / "analysis" / "cami2_paired_tests.tsv",
    # WS11.G leave-genus-out: the three-rung ladder.  The family and phylum
    # columns of `gen_ladder` are RECOMPUTED on the genus evaluation genomes
    # against the common control; they are NOT the WS1.9 / WS1.6 headline DiDs
    # and the two must never be mixed (trap T6).
    "gen_ladder": Path(RESULTS) / "holdout_genus" / "genus_vs_family_vs_phylum_did.tsv",
    "gen_ctrl": Path(RESULTS) / "holdout_genus" / "ladder_report.md",
    "gen_sample": Path(RESULTS) / "holdout_genus" / "ladder_per_sample.tsv.gz",
    "gen_panel": Path(RESULTS) / "holdout_genus" / "ws11g_consolidated.json",
}


def require(path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"required input missing: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"required input is empty: {p}")
    return p


def require_all() -> int:
    for key in SRC:
        require(SRC[key])
    return len(SRC)


def read_tsv(key: str) -> pd.DataFrame:
    return pd.read_csv(require(SRC[key]), sep="\t")


def parse_ci(text) -> tuple:
    """'[6.961, 9.1024]' -> (6.961, 9.1024)."""
    lo, hi = ast.literal_eval(str(text).strip())
    return float(lo), float(hi)


def md_table(key: str, heading: str) -> "list[dict]":
    """Rows of the first markdown table that follows `heading` in SRC[key]."""
    lines = require(SRC[key]).read_text().splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(heading))
    except StopIteration:
        raise KeyError(f"heading {heading!r} not in {SRC[key]}") from None
    grid = []
    for ln in lines[start + 1:]:
        t = ln.strip()
        if t.startswith("|"):
            grid.append([c.strip() for c in t.strip("|").split("|")])
        elif grid:
            break
    if len(grid) < 3:
        raise ValueError(f"no markdown table under {heading!r} in {SRC[key]}")
    return [dict(zip(grid[0], row)) for row in grid[2:]]


def ladder_panel_facts() -> dict:
    """Panel-design facts of WS11.G, read from the consolidated record."""
    panel = json.loads(require(SRC["gen_panel"]).read_text())["panel"]
    removed = int(panel["n_removed"]["train"])
    surviving = int(panel["surviving"]["train"])
    return dict(n_genera=len(panel["panel_taxa"]),
                n_families=len(panel["panel_parent_families"]),
                n_phyla=len(panel["panel_parent_phyla"]),
                train_removed=removed,
                train_total=removed + surviving,
                pct_genus=float(panel["pct_removed"]["train"]),
                pct_family=float(panel["ws1_9_pct_removed_train"]),
                pct_phylum=float(panel["ws1_6_pct_removed_train"]),
                ladder_complete=bool(panel["ladder_complete"]))


def ladder_control_deltas() -> dict:
    """Common-control raw dMAE (holdout - V5) for the three holdout models.

    The genus, family and phylum rungs of the ladder are only comparable
    because all three holdout models are differenced against ONE control: the
    in-distribution samples whose dominant phylum lies outside the WS1.6
    leave-phylum-out panel, so the control is in-distribution for every model.

    The three deltas are taken from the analysis script's own
    ``ladder_report.md`` table AND recomputed here, independently, from the
    per-sample ladder file.  A disagreement larger than the report's rounding
    (5e-4 pp) is fatal, so this value cannot drift from its source.
    """
    published = {r["model"]: {"completeness": float(r["completeness dMAE"]),
                              "contamination": float(r["contamination dMAE"])}
                 for r in md_table("gen_ctrl", "## Control deltas vs V5")}
    if set(published) != {"genusHO", "familyHO", "phylumHO"}:
        raise ValueError(f"unexpected control models {sorted(published)}")

    panel = set(json.loads(require(SRC["gen_panel"]).read_text())
                ["panel"]["ws16_phylum_panel"])
    with gzip.open(require(SRC["gen_sample"]), "rt") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    n_all = len(rows)
    ctrl = [r for r in rows if r["group"] == "in_distribution"
            and r["dominant_phylum"] not in panel]
    if not ctrl:
        raise ValueError("common control is empty")
    worst = 0.0
    for metric in ("completeness", "contamination"):
        truth = np.array([float(r["true_" + metric]) for r in ctrl])
        base = np.abs(np.array([float(r["V5_" + metric]) for r in ctrl]) - truth).mean()
        for model, got in published.items():
            mae = np.abs(np.array([float(r[model + "_" + metric])
                                   for r in ctrl]) - truth).mean()
            worst = max(worst, abs(float(mae - base) - got[metric]))
    if worst > 5e-4:
        raise AssertionError(
            f"common-control dMAE disagrees with {SRC['gen_ctrl'].name} by "
            f"{worst:.2e} pp (rounding tolerance 5e-4)")
    return dict(deltas=published, n=len(ctrl), n_all=n_all,
                n_refs=len({r["dominant_accession"] for r in ctrl}),
                max_abs_diff=worst)


# ---------------------------------------------------------------------------
# Verification ledger
# ---------------------------------------------------------------------------
class Ledger:
    """Records every plotted / annotated number with a pointer to its source."""

    def __init__(self):
        self.rows = []

    def rec(self, figure, panel, what, value, key, column, where, tol=1e-6):
        v = value if isinstance(value, str) else float(value)
        self.rows.append(dict(figure=figure, panel=panel, what=what, value=v,
                              key=key, column=column, where=dict(where), tol=tol))
        return v

    def rec_many(self, figure, panel, what, values, key, column, wheres, tol=1e-6):
        return [self.rec(figure, panel, f"{what}[{i}]", v, key, column, w, tol)
                for i, (v, w) in enumerate(zip(values, wheres))]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"figure": r["figure"], "panel": r["panel"],
                              "what": r["what"], "value": r["value"],
                              "file": SRC[r["key"]].name, "column": r["column"]}
                             for r in self.rows])


LEDGER = Ledger()


def _raw_rows(key):
    with open(require(SRC[key]), newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _extract(raw: str, column: str):
    """column may be 'x', or 'x[0]' / 'x[1]' for a '[lo, hi]' cell."""
    if column.endswith("]") and "[" in column:
        idx = int(column[column.rindex("[") + 1:-1])
        return float(ast.literal_eval(raw.strip())[idx])
    return float(raw)


def verify(ledger: Ledger = None) -> pd.DataFrame:
    """Re-read every source with a plain csv reader and compare."""
    ledger = ledger or LEDGER
    cache, out = {}, []
    for r in ledger.rows:
        rows = cache.setdefault(r["key"], _raw_rows(r["key"]))
        hits = [row for row in rows
                if all(str(row.get(k, "\0")).strip() == str(v) for k, v in r["where"].items())]
        status, src_val, note = "OK", np.nan, ""
        if len(hits) != 1:
            status, note = "FAIL", f"{len(hits)} rows match {r['where']}"
        else:
            cell = hits[0].get(r["column"].split("[")[0])
            if cell is None:
                status, note = "FAIL", f"no column {r['column']}"
            elif isinstance(r["value"], str):
                src_val = str(cell).strip()
                if src_val != r["value"]:
                    status, note = "FAIL", f"plotted {r['value']!r} vs source {src_val!r}"
            else:
                try:
                    src_val = _extract(cell, r["column"])
                except Exception as exc:                      # noqa: BLE001
                    status, note = "FAIL", f"unparsable {cell!r}: {exc}"
                else:
                    if not np.isfinite(src_val) or abs(src_val - r["value"]) > r["tol"]:
                        status = "FAIL"
                        note = f"plotted {r['value']!r} vs source {src_val!r}"
        out.append(dict(figure=r["figure"], panel=r["panel"], what=r["what"],
                        plotted=r["value"], source=src_val,
                        file=SRC[r["key"]].name, column=r["column"],
                        status=status, note=note))
    return pd.DataFrame(out)


def report_verification(label: str) -> pd.DataFrame:
    report = verify(LEDGER)
    n_bad = int((report.status == "FAIL").sum())
    print(f"  {label}: {len(report)} plotted values checked against "
          f"{report.file.nunique()} source files -- {n_bad} mismatch(es)")
    if n_bad:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 70)
        print(report[report.status == "FAIL"].to_string())
        raise AssertionError(f"{n_bad} plotted value(s) do not match their source")
    return report


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def diverging_cmap():
    return LinearSegmentedColormap.from_list("magicc_bwv", DIVERGING, N=256)


def num(v, spec="+.2f") -> str:
    """Format a number with a typographic minus sign (U+2212)."""
    return format(float(v), spec).replace("-", "−")


def style_ax(ax):
    ax.tick_params(length=2.2, width=0.5, pad=1.6)
    return ax


def vline0(ax, **kw):
    style = dict(color="0.35", lw=0.6, ls="--", zorder=0)
    style.update(kw)
    ax.axvline(0, **style)


def dot_ci(ax, x, y, lo, hi, *, color, marker, ms=3.4, lw=0.9, mfc=None,
           mec=None, zorder=3, label=None, ls="-", mew=0.6):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, ls=ls, solid_capstyle="butt",
            zorder=zorder)
    ax.plot([x], [y], marker=marker, ms=ms, color=color, lw=0,
            mfc=color if mfc is None else mfc,
            mec=color if mec is None else mec, mew=mew, zorder=zorder + 1,
            label=label)
