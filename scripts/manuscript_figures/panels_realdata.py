#!/usr/bin/env python3
"""Panel painters for the real-data, error-robustness and genome-size family.

These are the panels of the previous round's Figure 6 (real data with known
composition, Set G error robustness) and Figure 7 (genome size and the size
channel).  In this round they are split between main-text Figure 5 and
supplementary Figure S17, so they live here and both builders call them.

The data-loading code (``load6``/``load7``) is lifted verbatim from
``make_figures_6_7.py``, including every ``check()`` of a plotted value against
the value recorded in ``project_progress_and_results.md``: that is what
guarantees the recomposition changed no number.  The drawing code differs only
in the palette (BUILD_CONTRACT 4.1, with line style as a second redundant cue)
and in BUILD_CONTRACT 4.3 de-annotation -- every note box, banner, commentary
paragraph and per-point annotation is deleted and its numbers handed to the
caller for the caption.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs  # noqa: E402
from figstyle import (EDGE, LINESTYLES, MARKERS, PALETTE, TOOL_LABEL,  # noqa: E402
                      hline0)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
RES = fs.RESULTS
REAL = os.path.join(RES, "real_data")
MESLIER = os.path.join(REAL, "meslier")
RED = os.path.join(REAL, "reduced_genome")
MIT = os.path.join(RED, "mitigation")
SETG = os.path.join(RES, "set_G")
CAMI = os.path.join(RES, "cami2", "analysis")

MM = 1.0 / 25.4
DOUBLE_COL = 183 * MM          # 183 mm = 7.205 in (Nature double column)

TOOLS4 = ["magicc", "checkm2", "cocopye", "deepcheck"]
# tool_key used inside the WS3 result files -> figstyle key
KEYMAP = {"magicc": "magicc", "magicc_v5": "magicc", "MAGICC V5": "magicc",
          "MAGICC_V5": "magicc", "checkm2": "checkm2", "CheckM2 1.0.1": "checkm2",
          "CheckM2": "checkm2", "cocopye": "cocopye", "CoCoPyE 0.5.0": "cocopye",
          "CoCoPyE": "cocopye", "deepcheck": "deepcheck", "DeepCheck": "deepcheck"}

_VERIFY: "list[tuple]" = []


def req(path: str) -> str:
    """Return *path*, or die loudly if it is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"REQUIRED INPUT MISSING: {path}")
    return path


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(req(path), sep="\t")


def read_json(path: str):
    with open(req(path)) as fh:
        return json.load(fh)


def parse_ci(s) -> "tuple[float, float]":
    """'[-14.96, -9.81]' -> (-14.96, -9.81).  NaN-safe."""
    if isinstance(s, (list, tuple)):
        return float(s[0]), float(s[1])
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return (np.nan, np.nan)
    lo, hi = str(s).strip().strip("[]").split(",")
    return float(lo), float(hi)


def check(label: str, got: float, expect: float, tol: float = 0.011) -> float:
    """Record a verification of a plotted value against the documented value."""
    ok = (got is not None and expect is not None
          and not np.isnan(got) and abs(got - expect) <= tol)
    _VERIFY.append((label, got, expect, ok))
    return got


def verification_report(exit_on_fail: bool = True) -> int:
    print("\n" + "=" * 96)
    print(f"{'VERIFICATION (plotted value vs value recorded in the progress doc)':^96}")
    print("=" * 96)
    print(f"{'check':<62}{'plotted':>13}{'expected':>13}{'':>8}")
    bad = 0
    for label, got, exp, ok in _VERIFY:
        if not ok:
            bad += 1
            print(f"{label:<62}{got:>13.4f}{exp:>13.4f}{'  ** MISMATCH':>8}")
    print("=" * 96)
    print(f"{len(_VERIFY) - bad}/{len(_VERIFY)} checks passed"
          + ("" if bad == 0 else f"  ({bad} MISMATCH)"))
    if bad and exit_on_fail:
        raise AssertionError(f"{bad} plotted value(s) disagree with the record")
    return bad


def tkw(tool: str, **extra):
    kw = dict(color=PALETTE[tool], marker=MARKERS[tool], label=TOOL_LABEL[tool],
              ls=LINESTYLES[tool])
    if tool in EDGE:
        kw["markeredgecolor"] = EDGE[tool]
        kw["markeredgewidth"] = 0.5
    kw.update(extra)
    return kw


def dot_ci(ax, x, lo, hi, y, tool, filled=True, ms=4.0, lw=0.9, zorder=3):
    """Horizontal point-with-95 %-CI at row *y*, tool-coloured, shape = tool."""
    col = PALETTE[tool]
    ax.plot([lo, hi], [y, y], color=col, lw=lw, solid_capstyle="butt", zorder=zorder)
    ax.plot([x], [y], marker=MARKERS[tool], ms=ms,
            mfc=col if filled else "white",
            mec=EDGE.get(tool, col) if filled else col,
            mew=0.7, ls="none", zorder=zorder + 1)


def tool_legend_handles(open_label=None, filled_label=None):
    h = [Line2D([], [], color=PALETTE[t], marker=MARKERS[t], ms=4, lw=1.2,
                ls=LINESTYLES[t], markeredgecolor=EDGE.get(t, PALETTE[t]),
                markeredgewidth=0.5, label=TOOL_LABEL[t]) for t in TOOLS4]
    if filled_label:
        h.append(Line2D([], [], color="0.25", marker="o", ms=4, ls="none",
                        label=filled_label))
    if open_label:
        h.append(Line2D([], [], color="0.25", marker="o", ms=4, ls="none",
                        mfc="white", mec="0.25", label=open_label))
    return h


YPOS = {t: 3 - i for i, t in enumerate(TOOLS4)}


# ===========================================================================
#  Data loading -- lifted verbatim from make_figures_6_7.py
# ===========================================================================
def load6():
    # ---------------- data --------------------------------------------------
    grad = read_tsv(os.path.join(MESLIER, "fragmentation_gradient.tsv"))
    slopes = read_tsv(os.path.join(MESLIER, "fragmentation_slopes.tsv"))
    cohorts = read_tsv(os.path.join(MESLIER, "metrics_by_cohort.tsv"))
    paired = read_tsv(os.path.join(MESLIER, "paired_tests.tsv"))
    fp = read_tsv(os.path.join(MESLIER, "per_organism_contamination_fp.tsv"))
    synth = read_tsv(os.path.join(REAL, "synthesis_table_with_cami2.tsv"))
    cami_acc = read_tsv(os.path.join(CAMI, "cami2_accuracy_by_cohort.tsv"))
    curves = read_tsv(os.path.join(SETG, "set_G_curves.tsv"))
    degr = read_tsv(os.path.join(SETG, "set_G_paired_degradation.tsv"))
    bound = read_tsv(os.path.join(SETG, "set_G_boundary.tsv"))
    cross = read_tsv(os.path.join(SETG, "set_G_crossover.tsv"))
    ctx = read_tsv(os.path.join(SETG, "set_G_real_world_context.tsv"))
    aqc = read_tsv(os.path.join(MESLIER, "assembly_sequence_qc.tsv")).set_index("assembly")
    ncbi_merge = read_json(os.path.join(REAL, "ncbi_pairs", "predictions_merge_report.json"))
    kpert = read_tsv(os.path.join(SETG, "set_G_kmer_perturbation_summary.tsv"))

    PANEL = "balanced_panel_comp>=50"          # 37 organisms in every assembly
    g = grad[grad["panel"] == PANEL].copy()
    g["tool"] = g["tool_key"].map(KEYMAP)
    g_orf = g[~g["orf_integrity_compromised"].astype(bool)]
    if g_orf.empty:
        raise ValueError("no ORF-intact rows in the fragmentation gradient")
    n_panel_org = int(g_orf["n_panel_organisms"].iloc[0])
    n50_span = (g_orf["assembly_median_bin_n50"].max()
                / g_orf["assembly_median_bin_n50"].min())

    sl = slopes[slopes["panel"] == PANEL].copy()
    sl["tool"] = sl["tool"].map(KEYMAP)
    sl_comp = sl[sl["metric"] == "comp_mae"].set_index("tool")
    sl_lf = slopes[slopes["panel"] == PANEL + "_leakage_free"].copy()
    sl_lf["tool"] = sl_lf["tool"].map(KEYMAP)
    sl_lf = sl_lf[sl_lf["metric"] == "comp_mae"].set_index("tool")
    n_lf_org = int(grad[grad["panel"] == PANEL + "_leakage_free"]["n_panel_organisms"].iloc[0])
    pacbio = g_orf[g_orf["assembly"] == "pacbio"].set_index("tool")
    check("6a leakage-free panel MAGICC slope", float(sl_lf.loc["magicc", "slope_per_log10_N50"]), -1.29, 0.006)
    check("6a leakage-free panel CheckM2 slope", float(sl_lf.loc["checkm2", "slope_per_log10_N50"]), -1.84, 0.006)
    check("6a PacBio CheckM2 comp MAE", float(pacbio.loc["checkm2", "comp_mae"]), 0.30, 0.006)
    check("6a PacBio MAGICC comp MAE", float(pacbio.loc["magicc", "comp_mae"]), 1.73, 0.006)

    check("6a MAGICC slope per log10 N50", float(sl_comp.loc["magicc", "slope_per_log10_N50"]), -1.08, 0.006)
    check("6a CheckM2 slope per log10 N50", float(sl_comp.loc["checkm2", "slope_per_log10_N50"]), -2.40, 0.006)
    check("6a N50 fold range (~1,100x)", n50_span, 1127.0, 2.0)
    check("6a balanced-panel organisms", n_panel_org, 37, 0)

    orf_ok = aqc[~aqc["orf_integrity_compromised"].astype(bool)]
    low_ind = orf_ok[orf_ok["indel_bp_per_kb"] < 1.0]
    ion_ind = orf_ok[orf_ok["indel_bp_per_kb"] >= 1.0]
    check("6b MinION indel rate (bp per kb)", float(aqc.loc["minion", "indel_bp_per_kb"]), 8.95, 0.006)
    check("6b MinION median coding density", float(aqc.loc["minion", "median_coding_density"]), 0.754, 0.0006)
    check("6b MinION median gene length (bp)", float(aqc.loc["minion", "median_gene_length"]), 132.19, 0.01)
    kp = kpert[(kpert.error_type == "uneven_coverage") & (kpert.error_rate_pct == 40.0)].iloc[0]
    check("6f duplication: unique k-mer count shift (z)", float(kp["d_unique_kmer_count"]), 0.000, 0.0006)
    check("6f duplication: log10 total k-mer count shift (z)", float(kp["d_log10_total_kmer_count"]), 0.15, 0.006)
    check("6f duplication: duplicate k-mer count shift (z)", float(kp["d_duplicate_kmer_count"]), 0.51, 0.006)

    coh = cohorts.copy()
    coh["tool"] = coh["tool_key"].map(KEYMAP)
    C_ORF = "primary_leakage_free_comp>=50_ORF_intact"
    C_MIN = "ORF_compromised_assembly_only"
    orf = coh[coh["cohort"] == C_ORF].set_index("tool")
    mino = coh[coh["cohort"] == C_MIN].set_index("tool")
    check("6b ORF-intact MAGICC comp MAE", float(orf.loc["magicc", "comp_mae"]), 3.96, 0.006)
    check("6b ORF-intact CheckM2 comp MAE", float(orf.loc["checkm2", "comp_mae"]), 4.53, 0.006)
    check("6b MinION MAGICC comp MAE", float(mino.loc["magicc", "comp_mae"]), 4.91, 0.006)
    check("6b MinION CheckM2 comp MAE", float(mino.loc["checkm2", "comp_mae"]), 50.69, 0.006)
    check("6b MinION DeepCheck comp MAE", float(mino.loc["deepcheck", "comp_mae"]), 56.83, 0.006)

    pt = paired[(paired["cohort"] == C_ORF) & (paired["metric"] == "completeness")
                & (paired["tool_b"] == "CheckM2 1.0.1")].iloc[0]
    p_ns = float(pt["wilcoxon_p_two_sided"])
    d_ns = float(pt["mean_diff_a_minus_b"])
    ci_ns = (float(pt["diff_ci_lo"]), float(pt["diff_ci_hi"]))
    check("6b ORF-intact paired p (MAGICC vs CheckM2, completeness)", p_ns, 0.686, 0.001)
    check("6b ORF-intact paired difference (pp)", d_ns, -0.58, 0.006)

    # per-organism contamination false positives (context sentence for panel b/c)
    clean = fp[fp["true_cont_max"] < 5.0]
    n_clean_org = len(clean)
    n_ge5 = {t: int((clean[f"{t}_cont_median"] >= 5).sum())
             for t in ["magicc", "checkm2", "cocopye", "deepcheck"]}
    n_ge2 = {t: int((clean[f"{t}_cont_median"] >= 2).sum())
             for t in ["magicc", "checkm2", "cocopye", "deepcheck"]}

    # --- data-availability guard -------------------------------------------
    # The NCBI strain-matched draft/complete pairs are deliberately NOT in this
    # figure: the competitor predictions were never merged for that cohort, so
    # no four-tool comparison exists.  Fail loudly if that ever changes, so the
    # exclusion is revisited rather than silently kept.
    ncbi_absent = [t for t in ("checkm2", "cocopye", "deepcheck")
                   if str(ncbi_merge.get(t)) == "absent"]
    if len(ncbi_absent) != 3:
        raise ValueError(
            "ncbi_pairs/predictions_merge_report.json no longer reports all three "
            f"competitors as absent (got {ncbi_merge}); the NCBI strain-matched "
            "pairs were excluded from Figure 6c for exactly that reason - revisit "
            "panel c before re-running.")
    n_ncbi = int(ncbi_merge.get("magicc_matched", 0))

    ZY = "Zymo isolate drafts (n=8)"
    SM = "CAMI II strain-madness, gold-standard pure bins (leakage-free, >=50 %)"
    syn = synth.copy()
    syn["tool_k"] = syn["tool"].map(KEYMAP)
    zy = syn[syn["dataset"] == ZY].set_index("tool_k")
    sm = syn[syn["dataset"] == SM].set_index("tool_k")
    if zy.empty or sm.empty:
        raise ValueError("Zymo / CAMI II strain-madness rows not found")
    check("6c Zymo MAGICC comp MAE", float(zy.loc["magicc", "comp_MAE"]), 1.75, 0.006)
    check("6c Zymo DeepCheck comp MAE (best)", float(zy.loc["deepcheck", "comp_MAE"]), 0.98, 0.006)
    check("6c Zymo CheckM2 cont MAE (best)", float(zy.loc["checkm2", "cont_MAE"]), 0.17, 0.006)
    check("6c CAMI-SM gold MAGICC comp MAE", float(sm.loc["magicc", "comp_MAE"]), 1.30, 0.006)
    check("6c CAMI-SM gold CheckM2 comp MAE", float(sm.loc["checkm2", "comp_MAE"]), 13.96, 0.006)
    check("6c CAMI-SM gold MAGICC cont MAE", float(sm.loc["magicc", "cont_MAE"]), 0.50, 0.006)
    check("6c CAMI-SM gold CheckM2 cont MAE", float(sm.loc["checkm2", "cont_MAE"]), 3.02, 0.006)

    # cross-check the strain-madness numbers against the CAMI II analysis table
    ca = cami_acc[(cami_acc["dataset"] == "strain_madness")
                  & (cami_acc["binset"] == "gold")
                  & (cami_acc["cohort"] == "primary_in_domain_scoreable_LEAKAGE_FREE")].copy()
    ca["tool_k"] = ca["tool"].map(KEYMAP)
    for t in TOOLS4:
        a = float(ca[(ca.tool_k == t) & (ca.metric == "completeness")]["mae"].iloc[0])
        b = float(sm.loc[t, "comp_MAE"])
        check(f"6c CAMI-SM cross-source agreement, {TOOL_LABEL[t]} comp MAE", a, b, 0.002)

    cur = curves.copy()
    cur["tool_k"] = cur["tool"].map(KEYMAP)
    sub = cur[cur.error_type == "substitution"]
    ind = cur[cur.error_type == "indel"]
    dup = cur[cur.error_type == "uneven_coverage"]
    # documented dose ladders (progress doc S3.13); (rate %, expected MAE)
    DOC = {
        "substitution": (sub, {"magicc": [(0.0, 4.40), (2.0, 6.91), (5.0, 12.18)],
                               "checkm2": [(0.0, 5.11), (2.0, 14.54), (5.0, 31.07)],
                               "cocopye": [(0.0, 4.23), (2.0, 4.70), (5.0, 6.11)],
                               "deepcheck": [(0.0, 5.91), (2.0, 15.42), (5.0, 33.29)]}),
        "indel": (ind, {"magicc": [(0.1, 4.38), (1.0, 4.97), (5.0, 11.93)],
                        "checkm2": [(0.1, 15.25), (1.0, 46.58), (5.0, 54.63)],
                        "cocopye": [(0.1, 4.44), (1.0, 6.45), (5.0, 11.30)],
                        "deepcheck": [(0.1, 18.00), (1.0, 54.61), (5.0, 36.42)]}),
        "duplication": (dup, {"magicc": [(0.0, 4.40), (5.0, 5.37), (10.0, 7.67),
                                         (20.0, 12.17), (40.0, 15.99)],
                              "checkm2": [(0.0, 5.11), (5.0, 5.27), (10.0, 5.12),
                                          (20.0, 5.20), (40.0, 5.52)],
                              "cocopye": [(0.0, 4.23), (5.0, 4.35), (10.0, 4.53),
                                          (20.0, 4.85), (40.0, 4.89)],
                              "deepcheck": [(0.0, 5.91), (5.0, 5.84), (10.0, 6.07),
                                            (20.0, 5.78), (40.0, 6.11)]}),
    }
    for nm, (df_, exp) in DOC.items():
        for t, pairs in exp.items():
            d_ = df_[df_.tool_k == t].set_index("error_rate_pct")
            for rate, ev in pairs:
                check(f"6d-f {nm} {TOOL_LABEL[t]} comp MAE at {rate:g} %",
                      float(d_.loc[rate, "comp_mae"]), ev, 0.006)
    # W13 note: the CoCoPyE / DeepCheck duplication rows checked above are the
    # CORRECTED values read from set_G_curves.tsv (progress doc W13 records the
    # superseded transcription 4.32/4.40/4.62 and 5.90/5.87/5.99).

    dmg = dup[dup.tool_k == "magicc"].set_index("error_rate_pct")
    check("6f MAGICC completeness bias at 5 % duplication (pp)", float(dmg.loc[5.0, "comp_bias"]), 2.43, 0.006)
    check("6f MAGICC completeness bias at 40 % duplication (pp)", float(dmg.loc[40.0, "comp_bias"]), 11.29, 0.006)

    bd = bound.copy()
    def bval(tool, et, metric="completeness"):
        r = bd[(bd.tool == tool) & (bd.error_type == et) & (bd.metric == metric)]
        return float(r["rate_pct_where_upperCI_delta_mae_exceeds_1pp"].iloc[0])
    b_sub = check("6d MAGICC substitution boundary (%)", bval("magicc_v5", "substitution"), 0.96, 0.006)
    b_ind = check("6e MAGICC indel boundary (%)", bval("magicc_v5", "indel"), 0.86, 0.006)
    b_dup = check("6f MAGICC duplication boundary (%)", bval("magicc_v5", "uneven_coverage"), 2.87, 0.006)
    b_sub_c2 = check("6d CheckM2 substitution boundary (%)", bval("checkm2", "substitution"), 0.17, 0.006)
    b_ind_c2 = check("6e CheckM2 indel boundary (%)", bval("checkm2", "indel"), 0.008, 0.001)
    b_dup_c2 = check("6f CheckM2 duplication boundary (%)", bval("checkm2", "uneven_coverage"), 29.95, 0.006)

    xo = cross[(cross.comparator == "checkm2") & (cross.error_type == "uneven_coverage")
               & (cross.metric == "completeness")].iloc[0]
    xover = float(xo["crossover_rate_pct_magicc_becomes_worse"])
    dmax = float(xo["delta_mae_at_max_rate"])
    dmax_ci = parse_ci(xo["delta_mae_at_max_rate_ci"])
    check("6f duplication crossover vs CheckM2 (%)", xover, 10.0, 0.001)
    check("6f duplication delta MAE at 40 % (pp)", dmax, 10.47, 0.006)

    dq = degr[(degr.tool == "magicc_v5") & (degr.error_type == "uneven_coverage")
              & (degr.metric == "completeness") & (degr.error_rate_pct == 5.0)].iloc[0]
    check("6f MAGICC paired degradation at 5 % duplication (pp)", float(dq["delta_mae"]), 0.98, 0.006)

    ill = ctx[ctx["context"].str.contains("Illumina raw read, per base .Q30", regex=True)].iloc[0]
    ill_pct = float(ill["per_base_error_rate_pct"])
    ill_margin = float(ill["completeness_margin_x_1pp"])
    cons = ctx[ctx["context"].str.contains("consensus")]
    cons_lo = float(cons["per_base_error_rate"].min())
    cons_hi = float(cons["per_base_error_rate"].max())
    check("6d Illumina raw base error (%)", ill_pct, 0.1, 1e-9)
    check("6d Illumina margin to boundary (x)", ill_margin, 9.64, 0.02)
    check("6d consensus error lower bound (per base)", cons_lo, 1e-5, 1e-9)
    check("6d consensus error upper bound (per base)", cons_hi, 1e-3, 1e-9)
    # ---- assembled ------------------------------------------------------
    return dict(n_ncbi=n_ncbi, n_clean_org=n_clean_org, n_ge5=n_ge5, n_ge2=n_ge2,
                n_panel_org=n_panel_org, n50_span=n50_span,
                sl_m=float(sl_comp.loc["magicc", "slope_per_log10_N50"]),
                sl_c=float(sl_comp.loc["checkm2", "slope_per_log10_N50"]),
                sl_m_p=float(sl_comp.loc["magicc", "p_two_sided"]),
                sl_c_p=float(sl_comp.loc["checkm2", "p_two_sided"]),
                orf=orf, mino=mino, zy=zy, sm=sm, d_ns=d_ns, ci_ns=ci_ns, p_ns=p_ns,
                b_sub=b_sub, b_ind=b_ind, b_dup=b_dup, b_sub_c2=b_sub_c2,
                b_ind_c2=b_ind_c2, b_dup_c2=b_dup_c2, xover=xover, dmax=dmax,
                dmax_ci=dmax_ci, ill_pct=ill_pct, ill_margin=ill_margin,
                cons_lo=cons_lo, cons_hi=cons_hi,
                sl_lf_m=float(sl_lf.loc["magicc", "slope_per_log10_N50"]),
                sl_lf_c=float(sl_lf.loc["checkm2", "slope_per_log10_N50"]),
                n_lf_org=n_lf_org,
                pb_m=float(pacbio.loc["magicc", "comp_mae"]),
                pb_c=float(pacbio.loc["checkm2", "comp_mae"]),
                zy_bias_lo=float(zy["comp_bias"].min()), zy_bias_hi=float(zy["comp_bias"].max()),
                minion_indel=float(aqc.loc["minion", "indel_bp_per_kb"]),
                minion_cd=float(aqc.loc["minion", "median_coding_density"]),
                minion_gl=float(aqc.loc["minion", "median_gene_length"]),
                minion_n50=float(aqc.loc["minion", "median_bin_n50"]),
                low_indel_lo=float(low_ind["indel_bp_per_kb"].min()),
                low_indel_hi=float(low_ind["indel_bp_per_kb"].max()),
                ion_indel_lo=float(ion_ind["indel_bp_per_kb"].min()),
                ion_indel_hi=float(ion_ind["indel_bp_per_kb"].max()),
                ion_n50_hi=float(ion_ind["median_bin_n50"].max()),
                n_low=len(low_ind), n_ion=len(ion_ind),
                orf_cd_lo=float(orf_ok["median_coding_density"].min()),
                orf_cd_hi=float(orf_ok["median_coding_density"].max()),
                dup_kmer=(float(kp["d_unique_kmer_count"]), float(kp["d_log10_total_kmer_count"]),
                          float(kp["d_duplicate_kmer_count"])),
                minion_bias_c2=float(mino.loc["checkm2", "comp_bias"]),
                minion_bias_dc=float(mino.loc["deepcheck", "comp_bias"]),
                dup_ctrl=float(dmg.loc[0.0, "comp_mae"]),
                dup_max=float(dmg.loc[40.0, "comp_mae"]),
                dup_bias_max=float(dmg.loc[40.0, "comp_bias"]),
                sub_ctrl=float(sub[sub.tool_k == "magicc"].set_index("error_rate_pct").loc[0.0, "comp_mae"]),
                # frames the panel painters need
                g_orf=g_orf, sl_comp=sl_comp, sub=sub, ind=ind, dup=dup)

def load7():
    sizeb = read_tsv(os.path.join(RED, "size_bin_deltas.tsv"))
    linrel = read_tsv(os.path.join(RED, "lineage_relative_size_deltas.tsv"))
    revg = read_tsv(os.path.join(RED, "reviewer_genera_deltas.tsv"))
    recon = read_tsv(os.path.join(RED, "catalogue_baseline_reconciliation.tsv"))
    anchor = read_tsv(os.path.join(RED, "set_C_clean_crosscheck.tsv"))
    mech = read_tsv(os.path.join(RED, "mechanism_models.tsv"))
    phi_p = read_tsv(os.path.join(MIT, "intervention_phi_pooled.tsv"))
    phi_s = read_tsv(os.path.join(MIT, "intervention_phi_by_reference_size.tsv"))
    head = read_json(os.path.join(MIT, "mitigation_headline.json"))
    m1 = read_tsv(os.path.join(MIT, "mechanism_m1_matched_completeness.tsv"))

    # ---- panel a data ------------------------------------------------------
    bins = sizeb[sizeb["label"].str.startswith("size_stratified:")].copy()
    bins["short"] = bins["label"].str.replace("size_stratified:", "", regex=False)
    ctrls = sizeb[~sizeb["label"].str.startswith("size_stratified:")
                  & (sizeb["label"] != "ALL non-reduced control genera")].copy()
    ctrl_all = sizeb[sizeb["label"] == "ALL non-reduced control genera"].iloc[0]
    revs = revg[revg["label"] != "ALL five reviewer genera"].copy()
    rev_all = revg[revg["label"] == "ALL five reviewer genera"].iloc[0]
    n_mag = int(bins["n"].sum())
    n_ctrl = int(ctrls["n"].sum())
    n_rev = int(revs["n"].sum())
    check("7a MAGs in the size-stratified cohort", n_mag, 750, 0)
    check("7a control genera genomes", n_ctrl, 320, 0)
    check("7a <1 Mb delta completeness (pp)", float(bins.iloc[0]["delta_completeness_MAGICC_minus_CheckM2_median"]), -12.60, 0.006)
    check("7a 1-2 Mb delta completeness (pp)", float(bins.iloc[1]["delta_completeness_MAGICC_minus_CheckM2_median"]), -2.42, 0.006)
    check("7a 2-3 Mb delta completeness (pp)", float(bins.iloc[2]["delta_completeness_MAGICC_minus_CheckM2_median"]), 0.28, 0.006)
    check("7a 3-5 Mb delta completeness (pp)", float(bins.iloc[3]["delta_completeness_MAGICC_minus_CheckM2_median"]), 0.65, 0.006)
    check("7a >5 Mb delta completeness (pp)", float(bins.iloc[4]["delta_completeness_MAGICC_minus_CheckM2_median"]), 0.64, 0.006)
    check("7a five reviewer genera pooled delta (pp)", float(rev_all["delta_completeness_MAGICC_minus_CheckM2_median"]), -14.75, 0.006)

    dose = mech[mech.analysis == "dose_response_log10size"]
    d_mag = dose[(dose.pool == "size_stratified_MAGs") & (dose.target == "delta_completeness")].iloc[0]
    d_ctl = dose[(dose.pool == "non_reduced_control_genera") & (dose.target == "delta_completeness")].iloc[0]
    slope_mag = float(d_mag["slope_per_log10Mbp"])
    slope_mag_ci = parse_ci(d_mag["slope_ci95"])
    n_clu = int(d_mag["n_clusters"])
    slope_ctl = float(d_ctl["slope_per_log10Mbp"])
    p_ctl = float(d_ctl["slope_p"])
    p_mag = float(d_mag["slope_p"])
    r2_mag = float(d_mag["R2_coefficient_of_determination"])
    check("7a OLS p, MAGs", np.log10(p_mag), np.log10(4.34e-21), 0.02)
    check("7a OLS R2 (CoD), MAGs", r2_mag, 0.183, 0.0006)
    check("7a OLS slope, MAGs (pp per log10 Mbp)", slope_mag, 15.31, 0.006)
    check("7a OLS slope CI low", slope_mag_ci[0], 12.31, 0.006)
    check("7a OLS slope CI high", slope_mag_ci[1], 18.31, 0.006)
    check("7a family clusters", n_clu, 361, 0)
    check("7a control-genera slope (pp per log10 Mbp)", slope_ctl, -5.29, 0.006)
    check("7a control-genera slope p", p_ctl, 0.202, 0.001)
    check("7a lineage-relative delta at >=2x reduced (pp)",
          float(linrel.iloc[0]["delta_completeness_median"]), -13.07, 0.006)
    check("7a MAGICC-fails-CheckM2-passes at >=2x reduced (%)",
          float(linrel.iloc[0]["pct_magicc_cont_ge5_while_checkm2_lt5"]), 48.6, 0.06)
    check("7a MAGICC-fails-CheckM2-passes, lineage-typical (%)",
          float(linrel[linrel.log2_size_vs_phylum_bin.str.contains("typical")]
                ["pct_magicc_cont_ge5_while_checkm2_lt5"].iloc[0]), 9.7, 0.06)

    resid = (recon.assign(a=recon["residual_completeness_pp"].abs())
             .groupby("catalogue")["a"].mean().sort_values())
    check("7a SPIRE mean |residual| vs reviewer deltas (pp)", float(resid.get("SPIRE v1", np.nan)), 4.24, 0.02)

    # ---- panel b data ------------------------------------------------------
    a_row = anchor[anchor["stratum"].str.startswith("clean AND HQ-truth")].iloc[0]
    n_anch, ncl_anch = int(a_row["n"]), int(a_row["n_reference_clusters"])
    anchor_vals = []
    for tool, tag in [("magicc", "MAGICC"), ("checkm2", "CheckM2")]:
        for metric, mt in [("completeness", "comp"), ("contamination", "cont")]:
            v = float(a_row[f"{tag}_minus_truth_{mt}_mean"])
            lo, hi = parse_ci(a_row[f"{tag}_minus_truth_{mt}_mean_ci95"])
            anchor_vals.append((tool, metric, v, lo, hi))
    check("7b MAGICC completeness error vs truth (pp)", anchor_vals[0][2], -8.68, 0.006)
    check("7b MAGICC completeness CI low", anchor_vals[0][3], -14.03, 0.006)
    check("7b MAGICC completeness CI high", anchor_vals[0][4], -3.57, 0.006)
    check("7b MAGICC contamination error vs truth (pp)", anchor_vals[1][2], 5.09, 0.006)
    check("7b CheckM2 completeness error vs truth (pp)", anchor_vals[2][2], -0.70, 0.006)
    check("7b CheckM2 contamination error vs truth (pp)", anchor_vals[3][2], -1.16, 0.006)
    check("7b anchor n", n_anch, 30, 0)
    check("7b anchor reference clusters", ncl_anch, 25, 0)

    m1r = m1[(m1.subset == "clean_AND_near_complete") & (m1.size_predictor == "z_log10_reference_Mbp")]
    pr2_m = float(m1r[m1r.tool == "MAGICC_V5"]["partial_R2_increment"].iloc[0])
    pr2_c = float(m1r[m1r.tool == "CheckM2"]["partial_R2_increment"].iloc[0])
    n_m1 = int(m1r[m1r.tool == "MAGICC_V5"]["n"].iloc[0])
    ncl_m1 = int(m1r[m1r.tool == "MAGICC_V5"]["n_reference_clusters"].iloc[0])
    n_corpus = int(m1[(m1.subset == "all") & (m1.tool == "MAGICC_V5")]["n"].iloc[0])
    ncl_corpus = int(m1[(m1.subset == "all") & (m1.tool == "MAGICC_V5")]["n_reference_clusters"].iloc[0])
    check("7b controlled partial R2, MAGICC (W6: 0.036 not 0.313)", pr2_m, 0.036, 0.0006)
    check("7b controlled partial R2, CheckM2", pr2_c, 0.011, 0.0006)

    # ---- panel c data ------------------------------------------------------
    pv = phi_p.set_index("variant")
    rows_c = []
    for var, lab in [("coherent [LOCAL x0.85-x1.18]", "pooled (all inputs)"),
                     ("kmer_only [LOCAL x0.85-x1.18]", "k-mer branch only"),
                     ("assembly_only [LOCAL x0.85-x1.18]", "7 summary features only")]:
        r = pv.loc[var]
        lo, hi = parse_ci(r["ci95"])
        rows_c.append((lab, float(r["phi_pooled_regression"]), lo, hi))
    n_phi = int(pv.loc["coherent [LOCAL x0.85-x1.18]", "n"])
    ncl_phi = int(pv.loc["coherent [LOCAL x0.85-x1.18]", "n_reference_clusters"])
    L_elast = float(pv.loc["coherent [LOCAL x0.85-x1.18]", "implied_L_elasticity_to_observed_size"])
    check("7c phi pooled", rows_c[0][1], 0.699, 0.0011)
    check("7c phi pooled CI low", rows_c[0][2], 0.687, 0.0011)
    check("7c phi pooled CI high", rows_c[0][3], 0.711, 0.0011)
    check("7c phi k-mer branch", rows_c[1][1], 0.644, 0.0011)
    check("7c phi summary features", rows_c[2][1], 0.057, 0.0011)
    check("7c implied reference-length elasticity", L_elast, 0.30, 0.002)

    rows_s = []
    for _, r in phi_s.iterrows():
        lo, hi = parse_ci(r["ci95"])
        rows_s.append((str(r["reference_size_stratum"]), float(r["phi_coherent_LOCAL"]),
                       lo, hi, int(r["n"]), int(r["n_reference_clusters"])))
    phis = [v for _, v, _, _, _, _ in rows_s]
    check("7c phi stratum minimum", min(phis), 0.638, 0.002)
    check("7c phi stratum maximum", max(phis), 0.719, 0.002)

    # ---- panel d data ------------------------------------------------------
    nlr = {d["group"]: d for d in head["novel_lineage_bias_removed"]}
    orc = {d["stratum"]: d for d in head["ceiling_oracle_within_lineage"]}
    trf = {d["stratum"]: d for d in head["ceiling_novel_transfer"]}
    groups_d = ["group:Patescibacteriota", "group:DPANN", "group:Campylobacterota"]
    dlab = {"group:Patescibacteriota": "Patescibacteriota\n(unseen phylum)",
            "group:DPANN": "DPANN\n(unseen phylum)",
            "group:Campylobacterota": "Campylobacterota\n(unseen, needed no fix)"}
    drows = []
    for gkey in groups_d:
        raw = float(nlr[gkey]["raw_comp_bias_pp"])
        tra = float(trf[gkey]["comp_bias_recal"])
        ora = float(orc[gkey]["comp_bias_recal"])
        drows.append((gkey, raw, tra, ora,
                      float(nlr[gkey]["transfer_removes_pct"]),
                      float(nlr[gkey]["oracle_removes_pct"]),
                      int(nlr[gkey]["n"])))
    check("7d Patescibacteriota raw bias (pp)", drows[0][1], -30.00, 0.006)
    check("7d Patescibacteriota after blind transfer (pp)", drows[0][2], -25.50, 0.006)
    check("7d Patescibacteriota after oracle (pp)", drows[0][3], -0.27, 0.006)
    check("7d Patescibacteriota % removed, transfer", drows[0][4], 15.0, 0.06)
    check("7d Patescibacteriota % removed, oracle", drows[0][5], 99.1, 0.06)
    check("7d DPANN raw bias (pp)", drows[1][1], -23.65, 0.006)
    check("7d DPANN after blind transfer (pp)", drows[1][2], -20.44, 0.006)
    check("7d DPANN after oracle (pp)", drows[1][3], -0.54, 0.006)
    check("7d DPANN % removed, transfer (14 % to nearest per cent)", drows[1][4], 13.5, 0.06)
    check("7d DPANN % removed, oracle (98 % to nearest per cent)", drows[1][5], 97.7, 0.06)
    n_fitB = int(head["ceiling_fit_corpora"]["fitB"]["n"])
    check("7d blind-transfer fitting corpus n", n_fitB, 47015, 0)
    # ---- assembled ------------------------------------------------------
    return dict(n_mag=n_mag, n_ctrl=n_ctrl, n_rev=n_rev, n_clu=n_clu,
                slope_mag=slope_mag, slope_mag_ci=slope_mag_ci,
                slope_ctl=slope_ctl, p_ctl=p_ctl, p_mag=p_mag, r2_mag=r2_mag,
                bins=bins, linrel=linrel,
                resid=resid, n_anch=n_anch, ncl_anch=ncl_anch,
                anchor_vals=anchor_vals, pr2_m=pr2_m, pr2_c=pr2_c, n_m1=n_m1,
                ncl_m1=ncl_m1, n_corpus=n_corpus, ncl_corpus=ncl_corpus,
                rows_c=rows_c, rows_s=rows_s, n_phi=n_phi, ncl_phi=ncl_phi,
                L_elast=L_elast, drows=drows, n_fitB=n_fitB, phis=phis,
                # frames the panel painters need
                ctrls=ctrls, revs=revs, dlab=dlab, rev_all=rev_all,
                ctrl_all=ctrl_all)

# ===========================================================================
#  Panel painters
#
#  BUILD_CONTRACT 4.3: no note box, banner, commentary paragraph or per-point
#  annotation is drawn.  Each painter returns the numbers it no longer draws.
# ===========================================================================
def draw_fragmentation(ax, D, title="Meslier MOCK1 fragmentation gradient"):
    """Completeness MAE vs contig N50 (previous round Fig. 6a)."""
    for t in TOOLS4:
        d_ = D["g_orf"][D["g_orf"].tool == t].sort_values("assembly_median_bin_n50")
        yerr = np.vstack([d_["comp_mae"] - d_["comp_mae_ci_lo"],
                          d_["comp_mae_ci_hi"] - d_["comp_mae"]])
        ax.errorbar(d_["assembly_median_bin_n50"] / 1e3, d_["comp_mae"], yerr=yerr,
                    elinewidth=0.55, capsize=1.2, ms=3.6, lw=1.0,
                    **tkw(t, markeredgewidth=0.5))
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 100, 1000])
    ax.set_xticklabels(["2", "5", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("Assembly contig N50 (kb, log scale)\n"
                  "fragmentation increases to the left  $\\longleftarrow$")
    ax.set_ylabel("Completeness MAE (pp)")
    ax.set_ylim(-0.6, 11.6)
    if title:
        ax.set_title(title, fontsize=6.6, pad=3)
    return dict(slope_magicc=D["sl_m"], slope_checkm2=D["sl_c"],
                p_magicc=D["sl_m_p"], p_checkm2=D["sl_c_p"],
                n_organisms=D["n_panel_org"], n50_span=D["n50_span"])


def draw_orf_cohorts(fig, spec, D, panel_gap=0.10):
    """ORF-intact vs indel-dense MinION cohorts (previous round Fig. 6b)."""
    gsb = spec.subgridspec(1, 2, width_ratios=[1.0, 1.18], wspace=panel_gap)
    ax1 = fig.add_subplot(gsb[0, 0])
    ax2 = fig.add_subplot(gsb[0, 1], sharey=ax1)
    orf, mino = D["orf"], D["mino"]
    for axx, tab, ttl in [
            (ax1, orf, f"ORF-intact\nn = {int(orf['n'].iloc[0])} bins, "
                       f"{int(orf['n_clusters'].iloc[0])} organisms"),
            (ax2, mino, f"indel-dense MinION\nn = {int(mino['n'].iloc[0])} bins, "
                        f"{int(mino['n_clusters'].iloc[0])} organisms")]:
        for t in TOOLS4:
            r = tab.loc[t]
            dot_ci(axx, float(r["comp_mae"]), float(r["comp_mae_ci_lo"]),
                   float(r["comp_mae_ci_hi"]), YPOS[t] + 0.17, t, filled=True, ms=4.0)
            dot_ci(axx, float(r["comp_bias"]), float(r["comp_bias_ci_lo"]),
                   float(r["comp_bias_ci_hi"]), YPOS[t] - 0.17, t, filled=False, ms=4.0)
        axx.axvline(0, color="0.4", lw=0.6, ls="--", zorder=0)
        axx.set_title(ttl, fontsize=6.1, pad=3.0)
        axx.set_ylim(-0.75, 4.35)
        axx.set_xlabel("Completeness error (pp)", fontsize=6.2)
    ax1.set_yticks(list(YPOS.values()))
    ax1.set_yticklabels([TOOL_LABEL[t] for t in TOOLS4], fontsize=6)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(axis="y", length=0)
    ax1.set_xlabel("Completeness error (pp)\nfilled: |error| (MAE)\nopen: signed bias",
                   fontsize=6.2)
    ax1.set_xlim(-9.8, 9.8)
    ax1.set_xticks([-8, -4, 0, 4, 8])
    ax2.set_xlim(-68, 68)
    ax2.set_xticks([-60, -30, 0, 30, 60])
    return ax1, ax2


def draw_known_composition(fig, spec, D, panel_gap=0.10):
    """Zymo isolate pairs + CAMI II strain-madness gold bins (prev. Fig. 6c)."""
    gsc = spec.subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=panel_gap)
    ax1 = fig.add_subplot(gsc[0, 0])
    ax2 = fig.add_subplot(gsc[0, 1], sharey=ax1)
    zy, sm = D["zy"], D["sm"]
    for axx, tab, ttl in [
            (ax1, zy, f"ZymoBIOMICS isolate drafts\nn = {int(zy['n'].iloc[0])} pairs, "
                      f"{int(zy['n_clusters'].iloc[0])} strains"),
            (ax2, sm, f"CAMI II strain-madness gold bins\nn = {int(sm['n'].iloc[0])} bins, "
                      f"{int(sm['n_clusters'].iloc[0])} source genomes")]:
        for t in TOOLS4:
            r = tab.loc[t]
            clo, chi = parse_ci(r["comp_MAE_95CI"])
            klo, khi = parse_ci(r["cont_MAE_95CI"])
            dot_ci(axx, float(r["comp_MAE"]), clo, chi, YPOS[t] + 0.17, t, filled=True)
            dot_ci(axx, float(r["cont_MAE"]), klo, khi, YPOS[t] - 0.17, t, filled=False)
        axx.set_title(ttl, fontsize=6.1, pad=3.0)
        axx.set_ylim(-0.75, 4.35)
        axx.set_xlabel("MAE (pp)", fontsize=6.2)
    ax1.set_yticks(list(YPOS.values()))
    ax1.set_yticklabels([TOOL_LABEL[t] for t in TOOLS4], fontsize=6)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(axis="y", length=0)
    ax1.set_xlabel("MAE (pp)\nfilled: completeness\nopen: contamination", fontsize=6.2)
    ax1.set_xlim(-0.15, 3.3)
    ax1.set_xticks([0, 1, 2, 3])
    ax2.set_xlim(-1.0, 22.5)
    ax2.set_xticks([0, 5, 10, 15, 20])
    return ax1, ax2


def _dose_panel(ax, df_, xlab, xmax, boundaries, title=None, shade=None):
    for t in TOOLS4:
        d_ = df_[df_.tool_k == t].sort_values("error_rate_pct")
        yerr = np.vstack([d_["comp_mae"] - d_["comp_mae_ci_lo"],
                          d_["comp_mae_ci_hi"] - d_["comp_mae"]])
        ax.errorbar(d_["error_rate_pct"], d_["comp_mae"], yerr=yerr,
                    elinewidth=0.55, capsize=1.2, ms=3.6, lw=1.0,
                    **tkw(t, markeredgewidth=0.5))
    if shade is not None:
        ax.axvspan(shade[0], shade[1], color="0.82", alpha=0.55, lw=0, zorder=0)
    for xb, col in boundaries:
        ax.axvline(xb, color=col, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-xmax * 0.025, xmax)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Completeness MAE (pp)")
    if title:
        ax.set_title(title, fontsize=6.6, pad=3)
    return ax


def draw_dose_substitution(ax, D):
    """Set G per-base substitution dose-response (previous round Fig. 6d)."""
    _dose_panel(ax, D["sub"], "Substitution rate (% of bases)", 5.25,
                [(D["b_sub"], PALETTE["magicc"]), (D["b_sub_c2"], PALETTE["checkm2"])],
                title="Set G: uniform substitutions", shade=(0.0, D["ill_pct"]))
    ax.set_ylim(0, 40)
    return dict(b_magicc=D["b_sub"], b_checkm2=D["b_sub_c2"],
                illumina_pct=D["ill_pct"], illumina_margin=D["ill_margin"],
                consensus=(D["cons_lo"], D["cons_hi"]))


def draw_dose_indel(ax, D):
    """Set G indel dose-response (previous round Fig. 6e)."""
    _dose_panel(ax, D["ind"], "Indel rate (% of bases)", 5.25,
                [(D["b_ind"], PALETTE["magicc"]), (D["b_ind_c2"], PALETTE["checkm2"])],
                title="Set G: indels")
    ax.set_ylim(0, 62)
    return dict(b_magicc=D["b_ind"], b_checkm2=D["b_ind_c2"])


def draw_dose_duplication(ax, D, mark_unfavourable=True):
    """Set G uneven-coverage duplication dose-response (previous round Fig. 6f).

    This is the unfavourable result.  The direction is marked by one short
    label, not by a sentence drawn on the canvas and not by a background wash:
    a full-panel tint reads as an error state rather than as a caution, so the
    panel keeps the plain axes of every other dose-response panel.
    """
    _dose_panel(ax, D["dup"], "Assembly bp duplicated (%)", 42,
                [(D["b_dup"], PALETTE["magicc"]), (D["xover"], "0.35")],
                title="Set G: uneven-coverage duplication")
    ax.set_ylim(0, 22.0)
    if mark_unfavourable:
        ax.text(0.015, 0.965, "unfavourable: only MAGICC degrades",
                transform=ax.transAxes, fontsize=5.7, ha="left", va="top",
                color=PALETTE["magicc"], fontweight="bold")
    return dict(b_magicc=D["b_dup"], b_checkm2=D["b_dup_c2"], crossover=D["xover"],
                dmax=D["dmax"], dmax_ci=D["dmax_ci"], bias_max=D["dup_bias_max"],
                mae_ctrl=D["dup_ctrl"], mae_max=D["dup_max"], kmer=D["dup_kmer"])


# Panel-d cohort styling.  No tool series appears in this panel -- it plots one
# MAGICC-minus-CheckM2 difference for three cohorts of genomes -- so a cohort
# must not borrow a tool's cue.  The reviewer-genera cohort was previously drawn
# in CheckM2 blue with CoCoPyE's diamond, which invited exactly that misreading.
# Both context cohorts now use the neutral reference grey #888888
# (figstyle PALETTE["truth"]) and are separated from each other by marker and
# fill: open square for the non-reduced controls, solid plus for the reviewer
# genera.  'P' is used by no tool in any figure.  Grey rather than the brown
# #9C755F because, against the restored MAGICC red #d62728, grey holds a CIE76
# dE of 36.06 at its worst simulation (protanopia) while brown falls to 19.56.
COHORT_STYLE = {
    "controls": dict(color=PALETTE["truth"], marker="s", ms=3.2,
                     mfc="white", mew=0.7),
    "reviewer": dict(color=PALETTE["truth"], marker="P", ms=4.0,
                     mfc=PALETTE["truth"], mew=0.5),
}


def draw_size_disagreement(fig, spec, D, width_ratios=(1.55, 1.0), gap=0.09):
    """MAGICC-minus-CheckM2 completeness by genome size (previous Fig. 7a)."""
    gsa = spec.subgridspec(1, 2, width_ratios=list(width_ratios), wspace=gap)
    ax1 = fig.add_subplot(gsa[0, 0])
    ax2 = fig.add_subplot(gsa[0, 1], sharey=ax1)
    bins, ctrls, revs, linrel = D["bins"], D["ctrls"], D["revs"], D["linrel"]

    def eb(ax, x, row, dcol, ccol, **kw):
        y = float(row[dcol])
        lo, hi = parse_ci(row[ccol])
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], elinewidth=0.6,
                    capsize=1.3, ls="none", **kw)

    xb = bins["genome_size_median_bp"] / 1e6
    yb = bins["delta_completeness_MAGICC_minus_CheckM2_median"]
    lo = np.array([parse_ci(s)[0] for s in bins["delta_completeness_ci95"]])
    hi = np.array([parse_ci(s)[1] for s in bins["delta_completeness_ci95"]])
    ax1.errorbar(xb, yb, yerr=np.vstack([yb - lo, hi - yb]), color=PALETTE["magicc"],
                 marker="o", ms=4.2, lw=1.1, elinewidth=0.65, capsize=1.4, zorder=4)
    for _, r in ctrls.iterrows():
        eb(ax1, float(r["genome_size_median_bp"]) / 1e6, r,
           "delta_completeness_MAGICC_minus_CheckM2_median", "delta_completeness_ci95",
           zorder=3, **COHORT_STYLE["controls"])
    for _, r in revs.iterrows():
        eb(ax1, float(r["genome_size_median_bp"]) / 1e6, r,
           "delta_completeness_MAGICC_minus_CheckM2_median", "delta_completeness_ci95",
           zorder=5, **COHORT_STYLE["reviewer"])
    hline0(ax1)
    ax1.set_xscale("log")
    ax1.set_xlim(0.62, 8.2)
    ax1.set_xticks([0.8, 1, 2, 3, 5, 8])
    ax1.set_xticklabels(["0.8", "1", "2", "3", "5", "8"])
    ax1.set_xlabel("Median genome size (Mbp, log scale)", fontsize=6.2)
    ax1.set_ylabel("MAGICC $-$ CheckM2 completeness\nDISAGREEMENT (pp), median [95 % CI]",
                   fontsize=6.5)
    ax1.set_ylim(-41, 16)
    ax1.legend(handles=[
        Line2D([], [], color=PALETTE["magicc"], marker="o", ms=4, lw=1.1,
               label=f"MAGs, n = {D['n_mag']}"),
        Line2D([], [], ls="none", label=f"controls, n = {D['n_ctrl']}",
               **COHORT_STYLE["controls"]),
        Line2D([], [], ls="none", label=f"reviewer genera, n = {D['n_rev']}",
               **COHORT_STYLE["reviewer"])],
        loc="lower right", bbox_to_anchor=(1.015, -0.015), fontsize=5.2,
        handletextpad=0.3, labelspacing=0.18, borderpad=0.12)

    ylr = linrel["delta_completeness_median"].values
    llo = np.array([parse_ci(s)[0] for s in linrel["delta_completeness_ci95"]])
    lhi = np.array([parse_ci(s)[1] for s in linrel["delta_completeness_ci95"]])
    xlr = np.arange(len(ylr))
    ax2.errorbar(xlr, ylr, yerr=np.vstack([ylr - llo, lhi - ylr]),
                 color=PALETTE["magicc"], marker="o", ms=4.0, lw=1.1,
                 elinewidth=0.65, capsize=1.4)
    hline0(ax2)
    ax2.set_xticks(xlr)
    ax2.set_xticklabels(["<-1", "-1 to -0.5", "-0.5 to -0.25",
                         "-0.25 to +0.25", ">+0.25"], fontsize=5.2,
                        rotation=45, ha="right", rotation_mode="anchor")
    ax2.set_xlim(-0.5, len(ylr) - 0.5)
    ax2.set_xlabel("log$_2$(size / phylum median)", fontsize=6.2)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(axis="y", length=0)
    ax1.set_title("Catalogue MAGs, no ground truth", fontsize=6.6, pad=3)
    typ = linrel[linrel.log2_size_vs_phylum_bin.str.contains("typical")]
    return ax1, ax2, dict(
        slope=D["slope_mag"], slope_ci=D["slope_mag_ci"], p=D["p_mag"],
        r2=D["r2_mag"], slope_ctrl=D["slope_ctl"], p_ctrl=D["p_ctl"],
        n_mag=D["n_mag"], n_clu=D["n_clu"], n_ctrl=D["n_ctrl"], n_rev=D["n_rev"],
        spire=float(D["resid"].get("SPIRE v1")),
        gtdb=float(D["resid"].get("GTDB r220")),
        uhgg=float(D["resid"].get("UHGG v2.0.2")),
        reduced_delta=float(linrel.iloc[0]["delta_completeness_median"]),
        pct_reduced=float(linrel.iloc[0]["pct_magicc_cont_ge5_while_checkm2_lt5"]),
        pct_typical=float(typ["pct_magicc_cont_ge5_while_checkm2_lt5"].iloc[0]),
        bin_deltas=list(bins["delta_completeness_MAGICC_minus_CheckM2_median"]),
        rev_all=float(D["rev_all"]["delta_completeness_MAGICC_minus_CheckM2_median"]))


def draw_anchor(ax, D):
    """Signed error vs truth on genuinely clean Set C-clean (prev. Fig. 7b)."""
    ymap = {("magicc", "completeness"): 3, ("checkm2", "completeness"): 2,
            ("magicc", "contamination"): 1, ("checkm2", "contamination"): 0}
    for tool, metric, v, lo_, hi_ in D["anchor_vals"]:
        dot_ci(ax, v, lo_, hi_, ymap[(tool, metric)], tool,
               filled=(metric == "completeness"), ms=4.4)
    ax.axvline(0, color="0.35", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(list(range(4)))
    ax.set_yticklabels(["CheckM2\ncontamination", "MAGICC\ncontamination",
                        "CheckM2\ncompleteness", "MAGICC\ncompleteness"], fontsize=5.9)
    ax.set_ylim(-0.9, 3.6)
    ax.set_xlim(-16.5, 10.5)
    ax.set_xlabel("Signed error vs TRUTH (pp), mean [95 % CI]")
    ax.set_title(f"Ground-truthed anchor: clean, HQ Set C-clean\n"
                 f"Patescibacteriota (n = {D['n_anch']}, {D['ncl_anch']} clusters)",
                 fontsize=6.2, pad=3)
    av = {(t, m): (v, lo, hi) for t, m, v, lo, hi in D["anchor_vals"]}
    return dict(av=av, pr2_m=D["pr2_m"], pr2_c=D["pr2_c"], n_m1=D["n_m1"],
                ncl_m1=D["ncl_m1"], n_corpus=D["n_corpus"], ncl_corpus=D["ncl_corpus"])


def draw_phi(ax, D, label_points=False):
    """Size-channel elasticity phi, read-only intervention (prev. Fig. 7c)."""
    yrow, labels = [], []
    y = 0
    for lab, v, lo_, hi_ in D["rows_c"]:
        yrow.append((y, lab, v, lo_, hi_, "branch"))
        labels.append(lab)
        y -= 1
    y -= 0.7
    for lab, v, lo_, hi_, nn, ncl in D["rows_s"]:
        yrow.append((y, lab, v, lo_, hi_, "stratum"))
        labels.append(lab)
        y -= 1
    for yy, lab, v, lo_, hi_, kind in yrow:
        col = PALETTE["magicc"] if kind == "branch" else PALETTE["checkm2"]
        mk = "o" if kind == "branch" else "s"
        ax.plot([lo_, hi_], [yy, yy], color=col, lw=1.0, solid_capstyle="butt")
        ax.plot([v], [yy], marker=mk, ms=4.0, color=col, mfc=col, mec=col, ls="none")
        if label_points:
            ax.text(v, yy + 0.30, f"{v:.3f}", fontsize=5.4, ha="center", va="bottom",
                    color=col)
    ax.set_yticks([r[0] for r in yrow])
    ax.set_yticklabels(labels, fontsize=5.9)
    ax.set_ylim(min(r[0] for r in yrow) - 0.7, 0.9)
    ax.set_xlim(-0.03, 1.16)
    ax.axvline(1.0, color="0.35", lw=0.7, ls="--")
    ax.axvline(0.0, color="0.35", lw=0.7, ls=":")
    ax.set_xlabel("Size-channel elasticity $\\varphi$\n"
                  "d log(predicted completeness) / d log(assembly size)", fontsize=6.0)
    ax.text(1.012, 0.985, "$\\varphi$ = 1:\nno size\nleak",
            transform=ax.get_xaxis_transform(),
            fontsize=5.3, ha="left", va="top", color="0.35", linespacing=1.25)
    ax.text(-0.42, yrow[0][0] + 0.62, "by input branch", fontsize=5.6,
            ha="left", va="center", color="0.3", style="italic",
            transform=ax.get_yaxis_transform())
    ax.text(-0.42, yrow[3][0] + 0.62, "by reference-size stratum", fontsize=5.6,
            ha="left", va="center", color="0.3", style="italic",
            transform=ax.get_yaxis_transform())
    return dict(rows_c={lab: (v, lo, hi) for lab, v, lo, hi in D["rows_c"]},
                rows_s=D["rows_s"], phis=D["phis"], n_phi=D["n_phi"],
                ncl_phi=D["ncl_phi"], L_elast=D["L_elast"])


def draw_recalibration(ax, D):
    """Recalibration transfer to novel lineages vs oracle (prev. Fig. 7d)."""
    yy = [2, 1, 0]
    drows, dlab = D["drows"], D["dlab"]
    for i, (gkey, raw, tra, ora, pct_t, pct_o, nn) in enumerate(drows):
        y = yy[i]
        ax.plot([raw, ora], [y, y], color="0.75", lw=0.8, zorder=1)
        ax.plot([raw], [y], marker="x", ms=4.6, mew=1.0, color=PALETTE["truth"],
                ls="none", zorder=3)
        ax.plot([tra], [y], marker="s", ms=4.2, color=PALETTE["holdout"],
                mec=PALETTE["holdout"], ls="none", zorder=3)
        ax.plot([ora], [y], marker="o", ms=4.2, color=PALETTE["magicc"],
                mec=PALETTE["magicc"], ls="none", zorder=3)
        if abs(tra - raw) > 1.5:
            ax.annotate("", xy=(tra, y + 0.18), xytext=(raw, y + 0.18),
                        arrowprops=dict(arrowstyle="->", lw=0.6,
                                        color=PALETTE["holdout"]), zorder=2)
        ax.text(tra, y + 0.30, f"transfer {pct_t:.1f} %", fontsize=5.3,
                ha="center", va="bottom", color=PALETTE["holdout"])
        ax.text(ora, y - 0.32, f"oracle {pct_o:.1f} %", fontsize=5.3,
                ha="center", va="top", color=PALETTE["magicc"])
    ax.axvline(0, color="0.35", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(yy)
    ax.set_yticklabels([dlab[g[0]] for g in drows], fontsize=5.8)
    ax.set_ylim(-1.75, 2.8)
    ax.set_xlim(-36, 17)
    ax.set_xlabel("Signed completeness bias (pp)")
    ax.legend(handles=[
        Line2D([], [], color=PALETTE["truth"], marker="x", ms=4.6, ls="none",
               mew=1.0, label="raw holdout model"),
        Line2D([], [], color=PALETTE["holdout"], marker="s", ms=4.2, ls="none",
               label="blind transfer\n(recalibrator fitted on seen lineages)"),
        Line2D([], [], color=PALETTE["magicc"], marker="o", ms=4.2, ls="none",
               label="oracle fitted on the lineage itself")],
        loc="lower left", bbox_to_anchor=(-0.02, -0.02), fontsize=5.3,
        handletextpad=0.35, labelspacing=0.25, borderpad=0.15)
    return dict(drows=drows, n_fitB=D["n_fitB"])
