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
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figstyle import AUX, DIVERGING, PALETTE, RESULTS  # noqa: E402

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
           mec=None, zorder=3, label=None):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    ax.plot([x], [y], marker=marker, ms=ms, color=color, lw=0,
            mfc=color if mfc is None else mfc,
            mec=color if mec is None else mec, mew=0.6, zorder=zorder + 1,
            label=label)
