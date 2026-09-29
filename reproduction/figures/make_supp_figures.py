#!/usr/bin/env python3
"""Render eight nonredundant supplementary figures for resubmission7.

The approved panel map is qa/retention_contract.json. Scientific data and
statistics are unchanged; removed displays remain indexed machine-readable
sources. Captions are exported from the canonical supplementary Markdown.
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

# The five-set test-reference benchmark panel (PLAN.md).  n and n_clusters are read
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
        assert df.genome_id.is_unique and meta.genome_id.is_unique
        assert set(df.genome_id) == set(meta.genome_id), (sdir, tool, 'prediction/truth IDs differ')
        if not {'true_completeness', 'true_contamination'} <= set(df.columns):
            df = df.merge(meta[['genome_id', 'true_completeness', 'true_contamination']], on='genome_id', validate='one_to_one')
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
    name = "Figure_S1"
    print(f"[{name}] signed-error distributions by stratum")
    d = load_per_genome_signed_errors()
    comp = d[d.metric == "completeness"]
    cont = d[d.metric == "contamination"]
    n_gen = int(comp[comp.tool_key == "magicc"].shape[0])
    n_clust = int(comp.dominant_accession.nunique())

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
    ax.set_ylim(float(comp.signed_error.min()) - 5, float(comp.signed_error.max()) + 5)
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
    ax.set_ylim(float(cont.signed_error.min()) - 5, float(cont.signed_error.max()) + 5)
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
    ax.set_ylim(float(cont.signed_error.min()) - 5, float(cont.signed_error.max()) + 5)
    add_panel_label(ax, "c", x=-0.065, y=1.14)
    tool_legend(ax, loc="lower left", ncol=4, fontsize=6)

    save_fig(fig, name, out_dir=OUT)



# ---------------------------------------------------------------------------
# S7  Computational performance (WS8)
# ---------------------------------------------------------------------------
SPEED_TOOL = {"magicc": "magicc", "checkm2": "checkm2",
              "cocopye": "cocopye", "deepcheck": "deepcheck"}
SPEED_LABEL = dict(TOOL_LABEL)
SPEED_LABEL["deepcheck"] = "DeepCheck (inference only)"


def fig_S7():
    import panels_timing_current as pt
    name='Figure_S6'
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.7))
    fig.subplots_adjust(left=.085,right=.99,top=.82,bottom=.25,wspace=.5)
    runs,summary=pt.draw(axes[0], 'set_E_100')
    # Keep the scope note clear of CheckM2's one-thread point and segment.
    scope_notes=[t for t in axes[0].texts if t.get_text().startswith('DeepCheck: inference only')]
    assert len(scope_notes)==1
    scope_notes[0].set_x(.98)
    scope_notes[0].set_horizontalalignment('right')
    pt.draw(axes[1], 'set_E_full')
    ax=axes[2]
    for i,tool in enumerate(TOOLS):
        for j,input_set in enumerate(['set_E_100','set_E_full']):
            r=runs[(runs.tool==tool)&(runs.input_set==input_set)&(runs.threads==32)]
            x=i+(j-.5)*.3+np.linspace(-.04,.04,len(r))
            ax.scatter(x,r.peak_rss_gb,s=12,marker=MARKERS[tool],
                facecolor=PALETTE[tool] if j else 'white',edgecolor=PALETTE[tool],linewidth=.6)
    ax.set_yscale('log');ax.set_ylabel('Peak RSS (GB)')
    ax.set_xticks(range(4),[TOOL_LABEL[t] for t in TOOLS],rotation=35,ha='right',fontsize=5.8)
    ax.set_title('Memory: 32 threads',fontsize=6.6,pad=3)
    ax.legend(handles=[Line2D([],[],marker='o',ls='none',mfc='white',mec='.4',ms=3,label='100 genomes'),Line2D([],[],marker='o',ls='none',color='.4',ms=3,label='1,000 genomes')],fontsize=5.2,loc='lower right')
    for ax,letter in zip(axes,'abc'):
        add_panel_label(ax,letter,x=-.26,y=1.2)
    fig.legend(handles=[Line2D([],[],color=PALETTE[t],marker=MARKERS[t],ls=LINESTYLES[t],ms=3,label=TOOL_LABEL[t]) for t in TOOLS],loc='upper center',ncol=4,fontsize=6)
    save_fig(fig,name,out_dir=OUT)

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
        ["genome_id", "arm", "pair_id", "checkm2_severity", "true_completeness", "true_contamination"]]
    per = {}
    for tool in TOOLS:
        df = read_tsv(os.path.join(base, PRED_FILE[tool]))
        assert df.genome_id.is_unique and set(df.genome_id) == set(meta.genome_id)
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
        low = min(float(per[t][col].min()) for t in TOOLS)
        high = max(float(per[t][col].max()) for t in TOOLS)
        ax.set_ylim(min(ylim[0], low - 3), max(ylim[1], high + 3))
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
    ax.set_ylim(-25, 50)
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
    name = "Figure_S3"
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
                "uneven_coverage": "Localized exact\nduplication"}


def fig_S11():
    name = "Figure_S5"
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
                ax.set_xlabel("Added / original bp (%)" if et == "uneven_coverage" else "Error rate (%)", fontsize=5.8)
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
    ax.set_ylim(-0.5, max(14.0, float(kmer.observed_kmer_corruption.max()) * 105))
    ax.tick_params(labelsize=5)
    ax.legend(loc="upper right", frameon=False, fontsize=4.8,
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
    ax.set_xlabel("Interpolated 1-pp\ndegradation crossing\n(%; log scale)",
                  fontsize=5.4)
    ax.tick_params(labelsize=4.8)
    ax.grid(axis="x", lw=0.3, alpha=0.3)
    add_panel_label(ax, "g", x=-0.95, y=1.24)

    save_fig(fig, name, out_dir=OUT)

    n_per_cell = int(curves.n.iloc[0])
    n_clust = int(curves.n_clusters.iloc[0])


# ---------------------------------------------------------------------------
# S12  CAMI II external benchmark
# ---------------------------------------------------------------------------
CAMI_DATASETS = [("marine", "CAMI II marine"),
                 ("strain_madness", "CAMI II strain-madness")]
COMP_BANDS = ["50-60", "60-70", "70-80", "80-90", "90-95", "95-100"]





# ---------------------------------------------------------------------------
# S13  Reduced genomes: lineage-relative size effect and MIMAG-threshold impact
# ---------------------------------------------------------------------------
def _parse_ci(s):
    """'[-21.32, -0.56]' -> (-21.32, -0.56)"""
    if not isinstance(s, str) or not s.strip().startswith("["):
        return (np.nan, np.nan)
    lo, hi = s.strip()[1:-1].split(",")
    return float(lo), float(hi)





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





# ---------------------------------------------------------------------------
# S15  MIMAG-inspired confusion matrices, per set and tool
# ---------------------------------------------------------------------------
def fig_S15():
    name = "Figure_S2"
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
    # Each panel validates its actual Set F/CAMI inputs via read_tsv.
    # No removed historical holdout panel is required by this figure.
    fig=plt.figure(figsize=(6.75,5.8))
    gs=fig.add_gridspec(2,2,height_ratios=[.8,1.05],hspace=.72,wspace=.34,
        left=.15,right=.985,top=.92,bottom=.09)
    ax_a,heat=ptx.draw_type_distance_heatmaps(fig,gs[0,:],fig="S4",panel="a")
    add_panel_label(ax_a,"a",x=-.42,y=1.30)
    ax_b=fig.add_subplot(gs[1,0]);ptx.draw_detection_slope(ax_b,fig="S4",panel="b")
    add_panel_label(ax_b,"b",x=-.30,y=1.10)
    ax_c=fig.add_subplot(gs[1,1]);ptx.draw_cami_distance(ax_c,fig="S4",panel="c")
    add_panel_label(ax_c,"c",x=-.30,y=1.10)
    fl.report_verification("Figure S4")
    save_fig(fig,"Figure_S4",out_dir=OUT)



# ---------------------------------------------------------------------------
# S17  Error robustness and recalibration -- the panels of the previous round's
#      Figures 6 and 7 that are not in main-text Figure 5.
# ---------------------------------------------------------------------------
def fig_S17():
    import panels_realdata as prd
    D7=prd.load7()
    fig=plt.figure(figsize=(7.1,5.9))
    gs=fig.add_gridspec(2,2,height_ratios=[.9,1],hspace=.8,wspace=.4,
        left=.09,right=.98,top=.96,bottom=.09)
    ax_a=fig.add_subplot(gs[0,:]);prd.draw_anchor(ax_a,D7)
    add_panel_label(ax_a,"a",x=-.11,y=1.24)
    ax_b1,ax_b2,values=prd.draw_size_disagreement(fig,gs[1,0],D7)
    add_panel_label(ax_b1,"b",x=-.28,y=1.24)
    ax_c=fig.add_subplot(gs[1,1]);prd.draw_phi(ax_c,D7)
    add_panel_label(ax_c,"c",x=-.30,y=1.24)
    prd.verification_report()
    save_fig(fig,"Figure_S7",out_dir=OUT)



PANEL_LETTERS = set("abcdefgh")








# ---------------------------------------------------------------------------
def main():
    print(f"Output directory: {OUT}")
    for draw in [fig_S6,fig_S15,fig_S9,fig_S16,fig_S11,fig_S7,fig_S17,fig_S8]:draw()
    expected={f"Figure_S{i}.{ext}" for i in range(1,9) for ext in ('png','pdf')}
    actual={p for p in os.listdir(OUT) if p.endswith(('.png','.pdf'))}
    assert actual==expected,(actual-expected,expected-actual)
    print('All eight supplementary figures generated; captions come from canonical Markdown.')



if __name__ == "__main__":
    main()
