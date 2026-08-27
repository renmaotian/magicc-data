#!/usr/bin/env python3
"""Panel painters for the taxonomy family of results.

These are the panels of the previous round's Figure 4 (leave-one-lineage-out
holdout) and Figure 5 (contamination type x taxonomic distance, replicated on
CAMI II).  In this round they are split between main-text Figure 4 and
supplementary Figure S16, so they live here and both builders call them.  The
drawing code is the previous round's, with two changes only:

  * the author-mandated palette of ``figstyle`` (the internal build contract 4.1), with line
    style added as a second redundant cue wherever series share an axis; and
  * the internal build contract 4.3 de-annotation -- every free-floating sentence,
    statistical annotation block, "better tool" header row and per-point
    n/MAE/CI string is deleted.  Each painter returns a dict of the numbers it
    no longer draws, so the caller's caption can carry them.

No number changed value: every plotted value still goes through
``figledger.LEDGER`` and is re-read from its source TSV by
``figledger.verify``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figstyle import (LINESTYLES, MARKERS, PALETTE, TOOL_LABEL,  # noqa: E402
                      add_panel_label, hline0, tool_kw)
from figledger import (DISTANCES, FAMILY_LABEL, LEDGER, METRIC_STYLE,  # noqa: E402
                       SETF_TOOLS, TYPES, TYPE_STYLE, diverging_cmap, dot_ci,
                       num, parse_ci, read_tsv, style_ax, vline0)

from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


# ===========================================================================
# Leave-one-lineage-out holdout
# ===========================================================================
def _did_panel(ax, did, h2h, key_did, key_h2h, fig, panel, ylabels, xlabel,
               xlim, ylabelsize=None):
    d = did.sort_values("comp_did", ascending=True).reset_index(drop=True)
    for i, row in d.iterrows():
        for metric, col, cicol, off in (
                ("completeness", "comp_did", "comp_did_ci95", +0.17),
                ("contamination", "cont_did", "cont_did_ci95", -0.17)):
            lo, hi = parse_ci(row[cicol])
            v = LEDGER.rec(fig, panel, f"{row.group} {metric} DiD", row[col],
                           key_did, col, {"group": row.group})
            LEDGER.rec(fig, panel, f"{row.group} {metric} DiD CI lo", lo,
                       key_did, f"{cicol}[0]", {"group": row.group})
            LEDGER.rec(fig, panel, f"{row.group} {metric} DiD CI hi", hi,
                       key_did, f"{cicol}[1]", {"group": row.group})
            st = METRIC_STYLE[metric]
            dot_ci(ax, v, i + off, lo, hi, color=st["color"], marker=st["marker"])

    # in-distribution control: its DiD is 0 by construction, so plot the raw
    # delta-MAE that DEFINES the baseline (design-validity check).
    ctrl = h2h[h2h.group == "in_distribution"].iloc[0]
    c_comp = LEDGER.rec(fig, panel, "control comp dMAE", ctrl.d_comp_mae,
                        key_h2h, "d_comp_mae", {"group": "in_distribution"})
    c_cont = LEDGER.rec(fig, panel, "control cont dMAE", ctrl.d_cont_mae,
                        key_h2h, "d_cont_mae", {"group": "in_distribution"})
    c_p = LEDGER.rec(fig, panel, "control comp p", ctrl.paired_d_comp_p,
                     key_h2h, "paired_d_comp_p", {"group": "in_distribution"})
    yctrl = -1.15
    ax.axhline(-0.55, color="0.75", lw=0.5, ls=":", zorder=0)
    for v, metric, off in ((c_comp, "completeness", +0.17),
                           (c_cont, "contamination", -0.17)):
        st = METRIC_STYLE[metric]
        ax.plot([v], [yctrl + off], marker=st["marker"], ms=3.6, lw=0,
                mfc="white", mec=st["color"], mew=0.8, zorder=4)
    ax.set_yticks(list(np.arange(len(d))) + [yctrl])
    ax.set_yticklabels(ylabels(d) + ["in-distribution\ncontrol"])
    ax.get_yticklabels()[-1].set_color("0.30")
    if ylabelsize:
        ax.tick_params(axis="y", labelsize=ylabelsize)
    vline0(ax)
    ax.set_xlabel(xlabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.9, len(d) + 0.95)
    hs = [Line2D([], [], color=METRIC_STYLE[m]["color"], marker=METRIC_STYLE[m]["marker"],
                 ms=3.4, lw=0.9, label=METRIC_STYLE[m]["label"]) for m in METRIC_STYLE]
    hs.append(Line2D([], [], color="0.35", marker="o", ms=3.6, lw=0, mfc="white",
                     mec="0.35", label="in-distribution control (ΔMAE)"))
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.012, 1.03),
              handlelength=1.4, borderpad=0.2, labelspacing=0.22, ncol=2,
              columnspacing=0.8)
    return dict(control_comp_dmae=c_comp, control_cont_dmae=c_cont,
                control_comp_p=c_p, did=d)


def draw_phylum_did(ax, fig="4", panel="a"):
    """Leave-phylum-out difference-in-differences (previous round Fig. 4a)."""
    return _did_panel(
        style_ax(ax), read_tsv("phy_did"), read_tsv("phy_h2h"), "phy_did",
        "phy_h2h", fig, panel, lambda d: list(d.group),
        "leave-phylum-out difference-in-differences (pp)", (-2.2, 26.5))


def draw_family_did(ax, fig="4", panel="b"):
    """Leave-family-out difference-in-differences (previous round Fig. 4b)."""
    return _did_panel(
        style_ax(ax), read_tsv("fam_did"), read_tsv("fam_h2h"), "fam_did",
        "fam_h2h", fig, panel, lambda d: [FAMILY_LABEL[g] for g in d.group],
        "leave-family-out difference-in-differences (pp)", (-1.2, 14.9),
        ylabelsize=5.2)


def draw_family_vs_phylum(ax, fig="S16", panel="a"):
    """Paired family-vs-phylum attenuation (previous round Fig. 4c)."""
    style_ax(ax)
    fvp = read_tsv("fam_vs_phy")
    d = fvp.sort_values("comp_attenuation", ascending=True).reset_index(drop=True)
    fam_c, phy_c = PALETTE["magicc"], PALETTE["truth"]
    removed = {}
    for i, row in d.iterrows():
        fam = LEDGER.rec(fig, panel, f"{row.group} family DiD", row.comp_did_family,
                         "fam_vs_phy", "comp_did_family", {"group": row.group})
        phy = LEDGER.rec(fig, panel, f"{row.group} phylum DiD (recomputed)",
                         row.comp_did_phylum, "fam_vs_phy", "comp_did_phylum",
                         {"group": row.group})
        att = LEDGER.rec(fig, panel, f"{row.group} attenuation", row.comp_attenuation,
                         "fam_vs_phy", "comp_attenuation", {"group": row.group})
        pat = LEDGER.rec(fig, panel, f"{row.group} attenuation p", row.comp_p_attenuation,
                         "fam_vs_phy", "comp_p_attenuation", {"group": row.group})
        flo, fhi = parse_ci(row.comp_ci95_family)
        plo, phi = parse_ci(row.comp_ci95_phylum)
        LEDGER.rec(fig, panel, f"{row.group} family CI lo", flo,
                   "fam_vs_phy", "comp_ci95_family[0]", {"group": row.group})
        LEDGER.rec(fig, panel, f"{row.group} family CI hi", fhi,
                   "fam_vs_phy", "comp_ci95_family[1]", {"group": row.group})
        LEDGER.rec(fig, panel, f"{row.group} phylum CI lo", plo,
                   "fam_vs_phy", "comp_ci95_phylum[0]", {"group": row.group})
        LEDGER.rec(fig, panel, f"{row.group} phylum CI hi", phi,
                   "fam_vs_phy", "comp_ci95_phylum[1]", {"group": row.group})
        ax.annotate("", xy=(fam, i), xytext=(phy, i),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.55",
                                    mutation_scale=5, shrinkA=2.2, shrinkB=2.2))
        dot_ci(ax, phy, i, plo, phi, color=phy_c, marker="s", ms=3.2, mec="0.25")
        dot_ci(ax, fam, i, flo, fhi, color=fam_c, marker="o", ms=3.4)
        removed[row.group] = dict(attenuation=att, pct_retained=row.comp_pct_of_phylum_effect,
                                  p=pat, family=fam, phylum=phy)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([FAMILY_LABEL[g] for g in d.group])
    ax.tick_params(axis="y", labelsize=5.2)
    vline0(ax)
    ax.set_xlabel("completeness difference-in-differences (pp)")
    ax.set_xlim(-1.0, 30.5)
    ax.set_ylim(-0.75, len(d) - 0.10)
    hs = [Line2D([], [], color=fam_c, marker="o", ms=3.4, lw=0.9,
                 label="family holdout"),
          Line2D([], [], color=phy_c, marker="s", ms=3.2, lw=0.9, mec="0.25",
                 label="phylum holdout, recomputed on the family panel")]
    ax.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.005, -0.02),
              handlelength=1.4, borderpad=0.2, labelspacing=0.25)
    return removed


def draw_head_to_head(ax, fig="4", panel="c", label_groups=("Patescibacteriota",)):
    """Raw signed completeness bias, production V5 vs holdout (prev. Fig. 4d).

    ``label_groups`` names the groups that keep an on-canvas direct label; the
    build contract allows at most one per panel.
    """
    style_ax(ax)
    did_p = read_tsv("phy_did")
    ref_p = read_tsv("phy_ref")
    order = ["in_distribution"] + list(did_p.sort_values("comp_did", ascending=True).group)
    v5_c, ho_c = PALETTE["magicc"], PALETTE["holdout"]
    means = {}
    for i, grp in enumerate(order):
        sub = ref_p[ref_p.group == grp]
        if sub.empty:
            raise ValueError(f"no per-reference rows for group {grp}")
        for vals, off, colr in ((sub.V5_comp_bias.values, +0.19, v5_c),
                                (sub.holdout_comp_bias.values, -0.19, ho_c)):
            parts = ax.violinplot([vals], positions=[i + off], vert=False,
                                  widths=0.34, showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(colr)
                body.set_edgecolor(colr)
                body.set_alpha(0.42)
                body.set_linewidth(0.4)
        m_v5 = LEDGER.rec(fig, panel, f"{grp} V5 comp bias", float(sub.V5_comp_bias.mean()),
                          "phy_h2h", "V5_comp_bias", {"group": grp}, tol=5e-4)
        m_ho = LEDGER.rec(fig, panel, f"{grp} holdout comp bias",
                          float(sub.holdout_comp_bias.mean()),
                          "phy_h2h", "HO_comp_bias", {"group": grp}, tol=5e-4)
        n_ref = LEDGER.rec(fig, panel, f"{grp} n references", float(len(sub)),
                           "phy_h2h", "n_refs", {"group": grp})
        means[grp] = dict(v5=m_v5, holdout=m_ho, n_refs=int(n_ref))
        ax.plot([m_v5], [i + 0.19], marker="o", ms=3.4, color=v5_c, mec="white",
                mew=0.5, lw=0, zorder=5)
        ax.plot([m_ho], [i - 0.19], marker="s", ms=3.2, color=ho_c, mec="0.15",
                mew=0.5, lw=0, zorder=5)
        if grp in label_groups:
            ax.text(m_ho + 1.8, i - 0.19, f"{num(m_ho)} pp", fontsize=5.4, va="center",
                    ha="left", color=ho_c, fontweight="bold", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8))
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([g if g != "in_distribution" else "in-distribution\ncontrol"
                        for g in order])
    ax.axhline(0.5, color="0.75", lw=0.5, ls=":", zorder=0)
    ax.get_yticklabels()[0].set_color("0.30")
    vline0(ax)
    ax.set_xlabel("signed completeness error, predicted − true (pp)")
    ax.set_xlim(-58, 31)
    ax.set_ylim(-0.95, len(order) + 0.62)
    hs = [Line2D([], [], color=v5_c, marker="o", ms=3.4, lw=0, mec="white", mew=0.5,
                 label="production V5"),
          Line2D([], [], color=ho_c, marker="s", ms=3.2, lw=0, mec="0.15", mew=0.5,
                 label="leave-phylum-out holdout")]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.012, 1.03),
              handlelength=1.0, borderpad=0.2, labelspacing=0.22, ncol=2,
              columnspacing=0.9)
    return means


# ===========================================================================
# Contamination type x taxonomic distance
# ===========================================================================
def draw_type_distance_heatmaps(fig_obj, subplot_spec, fig="S16", panel="b"):
    """3 types x 6 distances signed-bias heatmaps, four tools (prev. Fig. 5a)."""
    cells = read_tsv("setF_cells")
    inner = GridSpecFromSubplotSpec(1, 5, subplot_spec=subplot_spec,
                                    width_ratios=[1, 1, 1, 1, 0.045], wspace=0.13)
    grid = {t: cells[cells.tool == t].set_index(["contamination_type", "distance"])
            for t in SETF_TOOLS}
    vmax = float(np.nanmax(np.abs(cells.cont_bias.values)))
    cmap = diverging_cmap()
    ax0, im = None, None
    xedge = np.arange(len(DISTANCES) + 1) - 0.5
    yedge = np.arange(len(TYPES) + 1) - 0.5
    for k, tool in enumerate(SETF_TOOLS):
        ax = fig_obj.add_subplot(inner[0, k])
        ax0 = ax0 or ax
        mat = np.zeros((len(TYPES), len(DISTANCES)))
        for i, ty in enumerate(TYPES):
            for j, dist in enumerate(DISTANCES):
                v = float(grid[tool].loc[(ty, dist), "cont_bias"])
                mat[i, j] = LEDGER.rec(fig, panel, f"{tool} {ty} {dist} bias", v,
                                       "setF_cells", "cont_bias",
                                       {"tool": tool, "contamination_type": ty,
                                        "distance": dist})
        im = ax.pcolormesh(xedge, yedge, mat, cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading="flat", linewidth=0, rasterized=False)
        for i in range(len(TYPES)):
            for j in range(len(DISTANCES)):
                ax.text(j, i, num(mat[i, j], "+.1f"), ha="center", va="center",
                        fontsize=4.9,
                        color="white" if abs(mat[i, j]) > 0.62 * vmax else "0.10")
        ax.set_xlim(xedge[0], xedge[-1])
        ax.set_ylim(yedge[-1], yedge[0])      # 'redundant' stays on top
        ax.set_xticks(range(len(DISTANCES)))
        ax.set_xticklabels(DISTANCES, rotation=45, ha="right", fontsize=5.6)
        ax.set_yticks(range(len(TYPES)))
        ax.set_yticklabels(TYPES if k == 0 else [], fontsize=6)
        ax.set_title(TOOL_LABEL[tool], fontsize=6.8, pad=2.6)
        ax.tick_params(length=1.6, width=0.5, pad=1.2)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.4)
        if k == 0:
            ax.set_ylabel("contamination type", fontsize=6.4, labelpad=2)
        ax.set_xlabel("taxonomic distance", fontsize=6.4, labelpad=1.5)
    cax = fig_obj.add_subplot(inner[0, 4])
    cb = fig_obj.colorbar(im, cax=cax)
    cb.set_label("signed contamination error,\npredicted − true (pp)", fontsize=6, labelpad=3)
    cb.ax.tick_params(labelsize=5.4, length=1.8, width=0.4)
    cb.outline.set_linewidth(0.4)

    # attribution numbers: no longer drawn on the canvas, returned for the caption
    attr = read_tsv("setF_attr")
    aw = {"competitor": "checkm2", "distance": "ALL",
          "arm": "NON_REDUNDANT(replaced+single)"}
    arow = attr[(attr.competitor == "checkm2") & (attr.distance == "ALL")
                & (attr.arm == "NON_REDUNDANT(replaced+single)")].iloc[0]
    a_red = LEDGER.rec(fig, panel, "MAGICC-CheckM2 advantage, redundant",
                       arow.mean_advantage_redundant, "setF_attr",
                       "mean_advantage_redundant", aw, tol=1e-4)
    a_non = LEDGER.rec(fig, panel, "MAGICC-CheckM2 advantage, non-redundant",
                       arow.mean_advantage_arm, "setF_attr", "mean_advantage_arm",
                       aw, tol=1e-4)
    a_p = LEDGER.rec(fig, panel, "attribution p", arow.p_wilcoxon_paired_by_reference,
                     "setF_attr", "p_wilcoxon_paired_by_reference", aw, tol=1e-12)
    return ax0, dict(adv_redundant=a_red, adv_nonredundant=a_non, p=a_p, vmax=vmax)


def draw_setF_bias_by_distance(ax, fig="4", panel="d"):
    """Signed contamination error vs taxonomic distance, four tools (prev. 5b)."""
    style_ax(ax)
    marg = read_tsv("setF_marg")
    xs = np.arange(len(DISTANCES))
    magicc_bias = {}
    for k, tool in enumerate(SETF_TOOLS):
        sub = marg[(marg.tool == tool) & (marg.margin == "distance")].set_index("level")
        y, lo, hi = [], [], []
        for dist in DISTANCES:
            where = {"tool": tool, "margin": "distance", "level": dist}
            y.append(LEDGER.rec(fig, panel, f"{tool} {dist} bias", sub.loc[dist, "cont_bias"],
                                "setF_marg", "cont_bias", where))
            lo.append(LEDGER.rec(fig, panel, f"{tool} {dist} bias lo",
                                 sub.loc[dist, "cont_bias_ci_lo"],
                                 "setF_marg", "cont_bias_ci_lo", where))
            hi.append(LEDGER.rec(fig, panel, f"{tool} {dist} bias hi",
                                 sub.loc[dist, "cont_bias_ci_hi"],
                                 "setF_marg", "cont_bias_ci_hi", where))
        y, lo, hi = map(np.asarray, (y, lo, hi))
        if tool == "magicc_v5":
            magicc_bias = dict(zip(DISTANCES, y))
        kw = tool_kw(tool, ms=3.2, lw=1.0, ls=LINESTYLES[tool])
        ax.errorbar(xs + (k - 1.5) * 0.055, y, yerr=[y - lo, hi - y],
                    elinewidth=0.7, capsize=1.4, capthick=0.7,
                    ecolor=kw["color"], **kw)
    hline0(ax)

    # the distance axis is a MEASURED gradient, not a label (set_F_distance_summary)
    dsum = read_tsv("setF_dist").set_index("distance")
    orth = [LEDGER.rec(fig, panel, f"{d} % orthologous", dsum.loc[d, "mean_pct_orthologous"],
                       "setF_dist", "mean_pct_orthologous", {"distance": d})
            for d in DISTANCES]
    # paired MAGICC-vs-CheckM2 verdict per distance: read, verified, NOT drawn
    pc = read_tsv("setF_paired")
    pc = pc[(pc.stratum == "distance") & (pc.tool_a == "magicc_v5")
            & (pc.tool_b == "checkm2")].set_index("level")
    winners, hls = {}, {}
    for d in DISTANCES:
        pw = {"stratum": "distance", "level": d, "tool_a": "magicc_v5",
              "tool_b": "checkm2"}
        hls[d] = LEDGER.rec(fig, panel, f"{d} paired HL vs CheckM2", pc.loc[d, "hl"],
                            "setF_paired", "hl", pw)
        winners[d] = LEDGER.rec(fig, panel, f"{d} paired winner", str(pc.loc[d, "winner"]),
                                "setF_paired", "winner", pw)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}\n{o:.0f} %" for d, o in zip(DISTANCES, orth)],
                       rotation=0, ha="center")
    ax.set_xlabel("taxonomic distance of the contaminant\n"
                  "(tick: mean % of donor genes with an orthologue)", fontsize=6.4)
    ax.set_ylabel("signed contamination error (pp)")
    ax.set_xlim(-0.45, len(DISTANCES) - 0.4)
    ax.set_ylim(-10.3, 2.6)
    ax.legend(loc="lower right", handlelength=1.8, borderpad=0.2, labelspacing=0.25,
              ncol=2, columnspacing=0.9, bbox_to_anchor=(1.008, -0.02))
    return dict(pct_orthologous=dict(zip(DISTANCES, orth)), winners=winners, hl=hls,
                magicc_bias=magicc_bias)


def draw_detection_slope(ax, fig="S16", panel="c"):
    """MAGICC detection slope vs distance, per contamination type (prev. 5c)."""
    style_ax(ax)
    cells = read_tsv("setF_cells")
    marg = read_tsv("setF_marg")
    cami_wc = read_tsv("cami_wc")
    xs = np.arange(len(DISTANCES))
    mg = cells[cells.tool == "magicc_v5"].set_index(["contamination_type", "distance"])
    for ty in TYPES:
        y, lo, hi = [], [], []
        for dist in DISTANCES:
            where = {"tool": "magicc_v5", "contamination_type": ty, "distance": dist}
            y.append(LEDGER.rec(fig, panel, f"{ty} {dist} slope",
                                mg.loc[(ty, dist), "detection_slope"],
                                "setF_cells", "detection_slope", where))
            lo.append(LEDGER.rec(fig, panel, f"{ty} {dist} slope lo",
                                 mg.loc[(ty, dist), "detection_slope_ci_lo"],
                                 "setF_cells", "detection_slope_ci_lo", where))
            hi.append(LEDGER.rec(fig, panel, f"{ty} {dist} slope hi",
                                 mg.loc[(ty, dist), "detection_slope_ci_hi"],
                                 "setF_cells", "detection_slope_ci_hi", where))
        y, lo, hi = map(np.asarray, (y, lo, hi))
        st = TYPE_STYLE[ty]
        ax.errorbar(xs + (TYPES.index(ty) - 1) * 0.075, y, yerr=[y - lo, hi - y],
                    color=st["color"], marker=st["marker"], ms=3.2, lw=1.0,
                    ls=st["ls"], elinewidth=0.7, capsize=1.4, capthick=0.7,
                    ecolor=st["color"], label=ty)
    pooled = marg[(marg.tool == "magicc_v5") & (marg.margin == "distance")].set_index("level")
    yp = [LEDGER.rec(fig, panel, f"pooled {dist} slope", pooled.loc[dist, "detection_slope"],
                     "setF_marg", "detection_slope",
                     {"tool": "magicc_v5", "margin": "distance", "level": dist})
          for dist in DISTANCES]
    ax.plot(xs, yp, color=PALETTE["truth"], marker="x", ms=3.2, lw=0.9, ls="--",
            label="pooled over types")

    # external replication of the species cell (CAMI II, well-controlled cohort)
    wc = cami_wc[cami_wc.tool == "MAGICC_V5"].set_index(["dataset", "distance_rank"])
    ext = {}
    for dsname, short, dx, mk in (("marine", "CAMI II marine*", 0.17, "o"),
                                  ("strain_madness", "CAMI II strain-madness*", 0.34, "D")):
        where = {"dataset": dsname, "tool": "MAGICC_V5", "distance_rank": "species"}
        s = LEDGER.rec(fig, panel, f"CAMI II {dsname} species slope (w.c.)",
                       wc.loc[(dsname, "species"), "slope"], "cami_wc", "slope", where)
        slo = LEDGER.rec(fig, panel, f"CAMI II {dsname} species slope lo",
                         wc.loc[(dsname, "species"), "slope_lo"], "cami_wc", "slope_lo", where)
        shi = LEDGER.rec(fig, panel, f"CAMI II {dsname} species slope hi",
                         wc.loc[(dsname, "species"), "slope_hi"], "cami_wc", "slope_hi", where)
        ext[dsname] = (s, slo, shi)
        ax.errorbar([dx], [s], yerr=[[s - slo], [shi - s]], color="#4D4D4D",
                    marker=mk, ms=3.0, mfc="white", mew=0.7, lw=0,
                    elinewidth=0.7, capsize=1.4, capthick=0.7, ecolor="#4D4D4D",
                    label=short, zorder=6)
    ax.axhline(1.0, color="0.55", lw=0.6, ls=":", zorder=0)
    ax.text(2.55, 1.02, "perfect detection", fontsize=5.2,
            color="0.40", ha="center", va="bottom")
    hline0(ax)
    ax.text(1.95, 0.025, "no detection", fontsize=5.2,
            color="0.40", ha="center", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(DISTANCES, rotation=30, ha="right")
    ax.set_xlabel("taxonomic distance of the contaminant")
    ax.set_ylabel("MAGICC detection slope\n(OLS of predicted on\ntrue contamination)")
    ax.set_xlim(-0.38, len(DISTANCES) - 0.32)
    ax.set_ylim(-0.42, 1.60)
    hnd, lab = ax.get_legend_handles_labels()
    want = ["redundant", "replaced", "single", "pooled over types",
            "CAMI II marine*", "CAMI II strain-madness*"]
    idx = [lab.index(w) for w in want]
    ax.legend([hnd[i] for i in idx], want, loc="lower right", handlelength=1.8,
              borderpad=0.2, labelspacing=0.22, bbox_to_anchor=(1.008, -0.015))
    ax.text(0.34, -0.32, "* well-controlled species cell", fontsize=4.9,
            color="#4D4D4D", ha="left", va="bottom")
    rep = mg.loc[("replaced", "species")]
    red = mg.loc[("redundant", "species")]
    sing = mg.loc[("single", "species")]
    return dict(species=dict(replaced=(rep.detection_slope, rep.detection_slope_ci_lo,
                                       rep.detection_slope_ci_hi),
                             single=(sing.detection_slope, sing.detection_slope_ci_lo,
                                     sing.detection_slope_ci_hi),
                             redundant=(red.detection_slope, red.detection_slope_ci_lo,
                                        red.detection_slope_ci_hi),
                             pooled=float(pooled.loc["species", "detection_slope"])),
                cami=ext)


def draw_cami_distance(ax, fig="S16", panel="d"):
    """Set F and CAMI II signed contamination bias vs distance (prev. Fig. 5d)."""
    style_ax(ax)
    marg = read_tsv("setF_marg")
    cami_mix = read_tsv("cami_mixed")
    read_tsv("cami_setF")            # cross-check source, re-read by verify()
    xs = np.arange(len(DISTANCES))
    setf = marg[(marg.tool == "magicc_v5") & (marg.margin == "distance")].set_index("level")
    y = [LEDGER.rec(fig, panel, f"Set F {dist} bias", setf.loc[dist, "cont_bias"],
                    "setF_marg", "cont_bias",
                    {"tool": "magicc_v5", "margin": "distance", "level": dist})
         for dist in DISTANCES]
    lo = [setf.loc[dist, "cont_bias_ci_lo"] for dist in DISTANCES]
    hi = [setf.loc[dist, "cont_bias_ci_hi"] for dist in DISTANCES]
    for dist, v in zip(DISTANCES, y):        # cross-check against the CAMI II table
        LEDGER.rec(fig, panel, f"Set F {dist} bias (cross-check)", v, "cami_setF",
                   "ws2_setF_cont_bias_pp",
                   {"distance_rank": dist, "dataset": "CAMI_II_marine"}, tol=6e-3)
    ax.errorbar(xs, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
                color=PALETTE["truth"], marker="x", ms=3.4, lw=1.0, ls="--",
                elinewidth=0.7, capsize=1.4, capthick=0.7, ecolor=PALETTE["truth"],
                label="Set F (our simulation)")
    cami_style = (("marine", "CAMI_II_marine", "CAMI II marine", PALETTE["magicc"],
                   "o", "-", -0.075),
                  ("strain_madness", "CAMI_II_strain_madness", "CAMI II strain-madness",
                   PALETTE["holdout"], "s", (0, (1, 1)), +0.075))
    mix = cami_mix[cami_mix.tool == "MAGICC_V5"].set_index(["dataset", "distance_rank"])
    for dsname, cmpname, label, colr, mk, ls, dx in cami_style:
        yy, ll, hh = [], [], []
        for dist in DISTANCES:
            where = {"dataset": dsname, "tool": "MAGICC_V5", "distance_rank": dist}
            yy.append(LEDGER.rec(fig, panel, f"{dsname} {dist} bias",
                                 mix.loc[(dsname, dist), "cont_bias"],
                                 "cami_mixed", "cont_bias", where))
            ll.append(LEDGER.rec(fig, panel, f"{dsname} {dist} bias lo",
                                 mix.loc[(dsname, dist), "cont_bias_lo"],
                                 "cami_mixed", "cont_bias_lo", where))
            hh.append(LEDGER.rec(fig, panel, f"{dsname} {dist} bias hi",
                                 mix.loc[(dsname, dist), "cont_bias_hi"],
                                 "cami_mixed", "cont_bias_hi", where))
            LEDGER.rec(fig, panel, f"{dsname} {dist} bias (cross-check)", yy[-1],
                       "cami_setF", "cami2_cont_bias_pp",
                       {"distance_rank": dist, "dataset": cmpname}, tol=1e-3)
        yy, ll, hh = map(np.asarray, (yy, ll, hh))
        ax.errorbar(xs + dx, yy, yerr=[yy - ll, hh - yy], color=colr, marker=mk,
                    ms=3.2, lw=1.0, ls=ls, elinewidth=0.7, capsize=1.4, capthick=0.7,
                    ecolor=colr, label=label,
                    mec="0.15" if mk == "s" else colr, mew=0.5)
    hline0(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(DISTANCES, rotation=30, ha="right")
    ax.set_xlabel("taxonomic distance of the contaminant")
    ax.set_ylabel("MAGICC signed contamination\nerror (pp)")
    ax.set_xlim(-0.45, len(DISTANCES) - 0.4)
    ax.set_ylim(-15.9, 5.4)
    ax.legend(loc="lower right", handlelength=1.8, borderpad=0.2, labelspacing=0.28,
              bbox_to_anchor=(1.005, -0.015))
    return dict(setF=dict(zip(DISTANCES, y)))


def draw_cami_paired_hl(ax, fig="4", panel="e"):
    """Paired MAGICC-vs-CheckM2 Hodges-Lehmann difference (prev. Fig. 5e)."""
    style_ax(ax)
    cami_pt = read_tsv("cami_paired")
    pt = cami_pt[(cami_pt.tool_b == "CheckM2") & (cami_pt.metric == "contamination")]
    pt = pt.set_index(["dataset", "binset"])
    rows = [("strain_madness", "mixed::distant_order_class_phylum",
             "distant\n(order + class + phylum)", 0.0),
            ("strain_madness", "mixed::close_species_genus", "close\n(species + genus)", 1.0),
            ("marine", "mixed::distant_order_class_phylum",
             "distant\n(order + class + phylum)", 2.4),
            ("marine", "mixed::close_species_genus", "close\n(species + genus)", 3.4)]
    XLO, XHI = -7.4, 5.4
    ax.axvspan(0, XHI, color=PALETTE["checkm2"], alpha=0.055, lw=0, zorder=0)
    ax.axvspan(XLO, 0, color=PALETTE["magicc"], alpha=0.055, lw=0, zorder=0)
    yticks, ylabels, out = [], [], {}
    for dsname, binset, label, ypos in rows:
        where = {"dataset": dsname, "binset": binset, "tool_b": "CheckM2",
                 "metric": "contamination", "tool_a": "MAGICC_V5"}
        hl = LEDGER.rec(fig, panel, f"{dsname} {binset} HL",
                        pt.loc[(dsname, binset), "hodges_lehmann"],
                        "cami_paired", "hodges_lehmann", where)
        lo = LEDGER.rec(fig, panel, f"{dsname} {binset} HL lo",
                        pt.loc[(dsname, binset), "hl_lo"], "cami_paired", "hl_lo", where)
        hi = LEDGER.rec(fig, panel, f"{dsname} {binset} HL hi",
                        pt.loc[(dsname, binset), "hl_hi"], "cami_paired", "hl_hi", where)
        n = LEDGER.rec(fig, panel, f"{dsname} {binset} n",
                       pt.loc[(dsname, binset), "n_pairs"],
                       "cami_paired", "n_pairs", where)
        mae_a = LEDGER.rec(fig, panel, f"{dsname} {binset} MAGICC MAE",
                           pt.loc[(dsname, binset), "mean_abs_err_a"],
                           "cami_paired", "mean_abs_err_a", where, tol=1e-5)
        mae_b = LEDGER.rec(fig, panel, f"{dsname} {binset} CheckM2 MAE",
                           pt.loc[(dsname, binset), "mean_abs_err_b"],
                           "cami_paired", "mean_abs_err_b", where, tol=1e-5)
        close = "close" in binset
        colr = PALETTE["checkm2"] if close else PALETTE["magicc"]
        dot_ci(ax, hl, ypos, lo, hi, color=colr, marker="^" if close else "o",
               ms=3.8, lw=1.1)
        yticks.append(ypos)
        ylabels.append(label)
        out[(dsname, "close" if close else "distant")] = dict(
            hl=hl, lo=lo, hi=hi, n=int(n), mae_magicc=mae_a, mae_checkm2=mae_b)
    vline0(ax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5.6)
    ax.set_ylim(-0.85, 4.55)
    ax.set_xlim(XLO, XHI)
    ax.text(XLO + 0.12, 3.95, "CAMI II marine", fontsize=6.0, color="0.15",
            ha="left", va="bottom")
    ax.text(XLO + 0.12, 1.55, "CAMI II strain-madness", fontsize=6.0, color="0.15",
            ha="left", va="bottom")
    ax.set_xlabel("Hodges–Lehmann difference in |contamination error|,\n"
                  "MAGICC − CheckM2 (pp), paired on identical bins\n"
                  "← favours MAGICC                    favours CheckM2 →")
    ax.axhline(2.20, color="0.80", lw=0.5, ls=":", zorder=0)
    return out
