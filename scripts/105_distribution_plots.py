#!/usr/bin/env python
"""
105_distribution_plots.py  --  WS5.6 + WS5.7
============================================
Replaces every bar-chart-style summary with a plot that shows the distribution of
the underlying data (editorial requirement E9), using a colour-vision-deficiency
safe palette with no red/green discrimination (editorial requirement E4, verified
by simulating deuteranopia / protanopia / tritanopia in script 101), and states
the denominator of every percentage in every caption (WS5.7 / Reviewer 1 M5).

Rules applied
-------------
* n >= 10 in a group  -> violin (kernel density) + box (median, IQR, 1.5xIQR
  whiskers) + mean marker.
* n < 10 in a group   -> every individual data point is drawn, plus the median.
* Derived scalar metrics that have no per-genome distribution (macro F1, false
  pass rate) are shown as the point estimate with its cluster-bootstrap
  distribution drawn as a violin and the 95% percentile interval as whiskers.
* Colour is never the only channel: tool order and marker shape are redundant cues.

    python scripts/105_distribution_plots.py [--config ...] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent


def _load_framework():
    spec = importlib.util.spec_from_file_location(
        "magicc_metrics_framework", _HERE / "101_metrics_framework.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["magicc_metrics_framework"] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

plt = fw.init_matplotlib()

CAPTIONS: List[str] = []
SMALL_N = 10


# ---------------------------------------------------------------------------
# drawing primitives
# ---------------------------------------------------------------------------


def dist_at(ax, x: float, vals: np.ndarray, colour: str, width: float = 0.72,
            marker: str = "o", edge: str = "#222222", rng=None):
    """Violin+box (n>=10) or all individual points (n<10) centred at x."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return
    if v.size >= SMALL_N and np.ptp(v) > 0:
        parts = ax.violinplot([v], positions=[x], widths=width,
                              showextrema=False, showmedians=False)
        for b in parts["bodies"]:
            b.set_facecolor(colour)
            b.set_edgecolor(edge)
            b.set_alpha(0.45)
            b.set_linewidth(0.4)
        bp = ax.boxplot([v], positions=[x], widths=width * 0.33, whis=1.5,
                        showfliers=False, patch_artist=True, manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor="white", edgecolor=edge, linewidth=0.6)
        for k in ("whiskers", "caps"):
            for ln in bp[k]:
                ln.set(color=edge, linewidth=0.6)
        for md in bp["medians"]:
            md.set(color=edge, linewidth=1.1)
        ax.plot([x], [v.mean()], marker=marker, ms=3.2, mfc=colour, mec=edge,
                mew=0.5, ls="none", zorder=5)
    else:
        r = rng if rng is not None else np.random.default_rng(0)
        jit = (r.random(v.size) - 0.5) * width * 0.6
        ax.plot(x + jit, v, marker=marker, ms=3.6, mfc=colour, mec=edge,
                mew=0.5, ls="none", alpha=0.95, zorder=4)
        ax.plot([x - width / 3, x + width / 3], [np.median(v)] * 2,
                color=edge, lw=1.2, zorder=5)


def point_ci_at(ax, x: float, est: float, lo: float, hi: float, colour: str,
                marker: str = "o", boot: Optional[np.ndarray] = None,
                width: float = 0.72, edge: str = "#222222"):
    """Point estimate + interval, with the bootstrap distribution as a violin."""
    if boot is not None:
        b = np.asarray(boot, float)
        b = b[~np.isnan(b)]
        if b.size >= SMALL_N and np.ptp(b) > 0:
            parts = ax.violinplot([b], positions=[x], widths=width,
                                  showextrema=False, showmedians=False)
            for p in parts["bodies"]:
                p.set_facecolor(colour)
                p.set_edgecolor(edge)
                p.set_alpha(0.35)
                p.set_linewidth(0.4)
    if not (np.isnan(lo) or np.isnan(hi)):
        ax.plot([x, x], [lo, hi], color=edge, lw=0.9, zorder=4)
        for y in (lo, hi):
            ax.plot([x - width / 5, x + width / 5], [y, y], color=edge, lw=0.9,
                    zorder=4)
    ax.plot([x], [est], marker=marker, ms=4.2, mfc=colour, mec=edge, mew=0.6,
            ls="none", zorder=6)


def tool_legend(fig, cfg, tools: Sequence[str], ncol: int = 6, y: float = -0.02):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=cfg.tool_marker(t), ls="none",
                      mfc=cfg.tool_colour(t), mec="#222222", mew=0.6, ms=5,
                      label=cfg.tool_label(t)) for t in tools]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y))


def union_tools(cfg, sets, frames) -> List[str]:
    have = set()
    for b in sets:
        have |= set(frames[b.name][1])
    return [t for t in cfg.tools if t in have]


def robust_ylim(values: Sequence[np.ndarray], lo_pct=0.5, hi_pct=99.5, pad=0.08):
    """Display range from pooled percentiles; extreme tails are clipped."""
    v = np.concatenate([np.asarray(x, float).ravel() for x in values if len(x)])
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None, False
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi <= lo:
        return None, False
    span = hi - lo
    clipped = bool((v < lo - pad * span).any() or (v > hi + pad * span).any())
    return (lo - pad * span, hi + pad * span), clipped


def zero_line(ax):
    ax.axhline(0, color="#555555", lw=0.7, ls=(0, (4, 3)), zorder=1)


def set_title_for(b) -> str:
    s = b.name
    if getattr(b, "status", "") == "superseded_leaky":
        s += "\n[SUPERSEDED: leakage]"
    return s


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------


def load_all(cfg, tiers=("primary", "reported")):
    sets, frames = [], {}
    for b in fw.discover_sets(cfg, include_missing=False, tiers=list(tiers)):
        df, tools, _ = fw.load_set(cfg, b)
        if not tools:
            continue
        df = df.copy()
        df["phylum_group"] = fw.phylum_group(df["dominant_phylum"],
                                             cfg.raw["strata"]["min_phylum_n"])
        sets.append(b)
        frames[b.name] = (df, tools)
    return sets, frames


DEN = None  # filled in main()


def cap(text: str) -> str:
    return f"{text}\n\n{DEN}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_signed_error_overall(cfg, sets, frames):
    metrics = ["completeness", "contamination"]
    ncol = len(sets)
    used = union_tools(cfg, sets, frames)
    fig, axes = plt.subplots(2, ncol, figsize=(1.75 * ncol + 1.1, 5.4),
                             sharey="row", squeeze=False)
    clipped_any = False
    for i, metric in enumerate(metrics):
        pool = [frames[b.name][0][f"err_{metric}__{t}"].to_numpy(float)
                for b in sets for t in frames[b.name][1]]
        lim, clip = robust_ylim(pool)
        clipped_any |= clip
        for j, b in enumerate(sets):
            df, tools = frames[b.name]
            tl = [t for t in cfg.tools if t in tools]
            ax = axes[i][j]
            zero_line(ax)
            for k, t in enumerate(used):
                if t not in tl:
                    continue
                dist_at(ax, used.index(t), df[f"err_{metric}__{t}"].to_numpy(float),
                        cfg.tool_colour(t), marker=cfg.tool_marker(t))
            ax.set_xlim(-0.7, len(used) - 0.3)
            if lim:
                ax.set_ylim(*lim)
            ax.set_xticks(range(len(used)))
            ax.set_xticklabels([cfg.tool_short(t) for t in used] if i == 1 else [],
                               rotation=90, fontsize=6.2)
            if i == 0:
                ax.set_title(set_title_for(b), fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{metric} signed error\n(predicted - true, pp)")
    tool_legend(fig, cfg, used, y=-0.075)
    extra = (" The y-axis is limited to the pooled 0.5-99.5 percentile range so "
             "that the informative part of every distribution is legible; a small "
             "number of extreme values (chiefly DeepCheck contamination "
             "overestimates above +100 pp) fall outside the plotted range and are "
             "given in full in ws5.3_signed_errors_overall.tsv."
             if clipped_any else "")
    fw.save_figure(fig, cfg, "fig_ws5.3_signed_error_by_set", cap(
        "Signed error distributions (predicted minus true, percentage points) for "
        "completeness (top row) and contamination (bottom row), one column per "
        "benchmark set, one violin per tool. Violins are kernel density estimates "
        "over all genomes of the set (n = 1,000 for Sets A, B, C, D, E, C-clean and "
        "D-clean); the inset box shows the median and interquartile range with "
        "whiskers at 1.5x IQR; the filled marker is the mean. The dashed line "
        "marks zero error: mass above it indicates systematic OVERestimation, "
        "below it systematic UNDERestimation. Sets labelled SUPERSEDED were built "
        "with dominant genomes drawn from the training split and are shown only "
        "for transparency. Colours come from a colour-vision-deficiency-safe "
        "palette (no red/green pair; minimum CIE76 Delta-E of 24.6 between any "
        "two tool colours under normal, deuteranopic, protanopic and tritanopic "
        "vision) and marker shape is a redundant cue." + extra), CAPTIONS)
    plt.close(fig)


def fig_signed_error_by_cont_bin(cfg, sets, frames):
    labels = cfg.raw["strata"]["contamination_bins"]["labels"]
    sets2 = [b for b in sets if frames[b.name][0]["true_cont_bin"].nunique() > 1]
    if not sets2:
        return
    used = union_tools(cfg, sets2, frames)
    fig, axes = plt.subplots(len(sets2), 1, figsize=(7.4, 2.15 * len(sets2)),
                             sharex=True, squeeze=False)
    for i, b in enumerate(sets2):
        ax = axes[i][0]
        df, tools = frames[b.name]
        tl = [t for t in cfg.tools if t in tools]
        zero_line(ax)
        lim, _ = robust_ylim([df[f"err_contamination__{t}"].to_numpy(float)
                              for t in tl], 1.0, 99.0)
        w = 0.82 / max(1, len(used))
        if lim:
            ax.set_ylim(*lim)
        for bi, lab in enumerate(labels):
            g = df[df.true_cont_bin == lab]
            if len(g) == 0:
                continue
            for t in tl:
                k = used.index(t)
                x = bi + (k - (len(used) - 1) / 2) * w
                dist_at(ax, x, g[f"err_contamination__{t}"].to_numpy(float),
                        cfg.tool_colour(t), width=w * 0.92,
                        marker=cfg.tool_marker(t))
            ax.annotate(f"n={len(g)}", xy=(bi, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=6, color="#444444")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylabel("contamination\nsigned error (pp)")
        ax.set_title(set_title_for(b).replace("\n", "  "), fontsize=8, loc="left")
    axes[-1][0].set_xlabel("true contamination bin (%)")
    tool_legend(fig, cfg, used, y=-0.015)
    fw.save_figure(fig, cfg, "fig_ws5.3_signed_error_by_contamination_bin", cap(
        "Contamination signed error (predicted minus true, percentage points) "
        "stratified by TRUE contamination level, one panel per benchmark set. "
        "Reviewer 2 (major comment 2) notes that genomes with 20-80% "
        "contamination would be discarded regardless of the exact estimate; this "
        "figure therefore resolves the practically relevant low-contamination "
        "bins separately from the extreme ones. Violin = kernel density of all "
        "genomes in the bin, box = median and IQR (whiskers 1.5x IQR), filled "
        "marker = mean; groups with fewer than 10 genomes are drawn as individual "
        "points with a median bar. n above each group is the number of genomes in "
        "that bin. Negative values indicate UNDERestimation of contamination."),
        CAPTIONS)
    plt.close(fig)


def fig_signed_error_by_stratum(cfg, sets, frames, col: str, name: str,
                                title: str, metric: str = "contamination",
                                max_levels: int = 8):
    rows = []
    for b in sets:
        df, tools = frames[b.name]
        for lv in df[col].dropna().unique():
            rows.append((b, lv, (df[col] == lv).sum()))
    if not rows:
        return
    levels = (pd.DataFrame(rows, columns=["b", "lv", "n"]).groupby("lv")["n"].sum()
              .sort_values(ascending=False).head(max_levels).index.tolist())
    used = union_tools(cfg, sets, frames)
    fig, axes = plt.subplots(len(sets), 1, figsize=(7.4, 2.05 * len(sets)),
                             sharex=True, squeeze=False)
    for i, b in enumerate(sets):
        ax = axes[i][0]
        df, tools = frames[b.name]
        tl = [t for t in cfg.tools if t in tools]
        zero_line(ax)
        lim, _ = robust_ylim([df[f"err_{metric}__{t}"].to_numpy(float) for t in tl],
                             1.0, 99.0)
        if lim:
            ax.set_ylim(*lim)
        w = 0.82 / max(1, len(used))
        for bi, lv in enumerate(levels):
            g = df[df[col] == lv]
            if len(g) == 0:
                continue
            for t in tl:
                k = used.index(t)
                x = bi + (k - (len(used) - 1) / 2) * w
                dist_at(ax, x, g[f"err_{metric}__{t}"].to_numpy(float),
                        cfg.tool_colour(t), width=w * 0.92,
                        marker=cfg.tool_marker(t),
                        rng=np.random.default_rng(bi * 97 + k))
            ax.annotate(f"n={len(g)}", xy=(bi, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=6, color="#444444")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(l) for l in levels], rotation=20, ha="right")
        ax.set_ylabel(f"{metric}\nsigned error (pp)")
        ax.set_title(set_title_for(b).replace("\n", "  "), fontsize=8, loc="left")
    tool_legend(fig, cfg, used, y=-0.015)
    fw.save_figure(fig, cfg, name, cap(
        f"{title} Values are {metric} signed errors (predicted minus true, "
        f"percentage points); the dashed line is zero error. Violin = kernel "
        f"density, box = median and interquartile range (whiskers 1.5x IQR), "
        f"filled marker = mean; strata with fewer than 10 genomes are drawn as "
        f"individual points with a median bar, per editorial policy. n above each "
        f"group is the number of genomes in that stratum."), CAPTIONS)
    plt.close(fig)


def fig_mimag_macro_f1(cfg, sets, frames, n_boot: int):
    labels = cfg.mimag_classes
    used = union_tools(cfg, sets, frames)
    fig, axes = plt.subplots(1, len(sets), figsize=(1.75 * len(sets) + 1.0, 3.8),
                             sharey=True, squeeze=False)
    for j, b in enumerate(sets):
        ax = axes[0][j]
        df, tools = frames[b.name]
        tl = [t for t in cfg.tools if t in tools]
        bs = fw.Bootstrapper(clusters=df["cluster_id"].to_numpy(), n_iter=n_boot,
                             ci_level=cfg.ci_level,
                             seed=cfg.seed + (fw.stable_hash(b.name) % 100000))
        bs.resample_indices()
        tcode = fw.encode_labels(df["true_mimag"].to_numpy(), labels)
        obs = fw.cm_from_codes(tcode, tcode, len(labels)).sum(axis=1) > 0
        n_obs = int(obs.sum())
        for t in tl:
            k = used.index(t)
            pcode = fw.encode_labels(df[f"mimag__{t}"].to_numpy(), labels)

            def stat(idx, _p=pcode):
                cm = fw.cm_from_codes(tcode[idx], _p[idx], len(labels))
                return fw.metrics_from_cm(cm, labels, observed_mask=obs)["macro_f1_observed"]

            vals = np.array([stat(idx) for idx in bs.resample_indices()])
            est = stat(np.arange(len(df)))
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            point_ci_at(ax, k, est, lo, hi, cfg.tool_colour(t),
                        marker=cfg.tool_marker(t), boot=vals)
        ax.set_xticks(range(len(used)))
        ax.set_xticklabels([cfg.tool_short(t) for t in used], rotation=90,
                           fontsize=6.2)
        ax.set_xlim(-0.7, len(used) - 0.3)
        ax.set_title(f"{set_title_for(b)}\n({n_obs} classes present)", fontsize=7.5)
        ax.set_ylim(0, 1.02)
        if j == 0:
            ax.set_ylabel("macro-averaged F1\n(classes present in the truth)")
    tool_legend(fig, cfg, used, y=-0.09)
    fw.save_figure(fig, cfg, "fig_ws5.1_mimag_macro_f1", cap(
        "Macro-averaged F1 for the MIMAG-inspired 3-class quality assignment "
        "(high: completeness >= 90% AND contamination < 5%; medium: completeness "
        ">= 50% AND contamination < 10%; low: all others), reported PER BENCHMARK "
        "SET rather than as a range. Filled marker = point estimate; whiskers = "
        "percentile 95% confidence interval; the shaded violin behind each marker "
        "is the full cluster-bootstrap distribution (clusters = dominant "
        "reference genomes), which is what replaces a bar chart here. The average "
        "is taken over the quality classes that actually occur in the ground truth "
        "of each set (Set A contains no low-quality genomes by design), so it is "
        "not deflated by structurally empty classes. Note that the full MIMAG "
        "standard additionally requires rRNA and tRNA criteria that cannot be "
        "evaluated from completeness and contamination estimates."), CAPTIONS)
    plt.close(fig)


def fig_confusion_grid(cfg, sets, frames):
    labels = cfg.mimag_classes
    tools_all = []
    for b in sets:
        tools_all += [t for t in frames[b.name][1] if t not in tools_all]
    tools_all = [t for t in cfg.tools if t in tools_all]
    nr, nc = len(sets), len(tools_all)
    fig, axes = plt.subplots(nr, nc, figsize=(1.25 * nc + 1.0, 1.25 * nr + 0.9),
                             squeeze=False)
    cmap = cfg.raw["palette"]["sequential_cmap"]
    im = None
    for i, b in enumerate(sets):
        df, tools = frames[b.name]
        for j, t in enumerate(tools_all):
            ax = axes[i][j]
            if t not in tools:
                ax.axis("off")
                continue
            cm = fw.confusion_matrix(df["true_mimag"], df[f"mimag__{t}"], labels)
            rn = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
            im = ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
            for r in range(len(labels)):
                for c in range(len(labels)):
                    ax.text(c, r, f"{cm[r, c]}", ha="center", va="center",
                            fontsize=5.6,
                            color="white" if rn[r, c] < 0.55 else "black")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels([l[0].upper() for l in labels], fontsize=6)
            ax.set_yticklabels([l[0].upper() for l in labels] if j == 0 else [],
                               fontsize=6)
            if i == 0:
                ax.set_title(cfg.tool_label(t).split(" (")[0], fontsize=6.8)
            if j == 0:
                ax.set_ylabel(set_title_for(b).replace("\n", " "), fontsize=6.2)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cb.set_label("fraction of the true class (row-normalised)", fontsize=7)
    fw.save_figure(fig, cfg, "fig_ws5.1_mimag_confusion_matrices", cap(
        "Confusion matrices for the MIMAG-inspired quality assignment, one panel "
        "per benchmark set (rows) and tool (columns). Rows are the true class, "
        "columns the predicted class; H = high, M = medium, L = low. Cell shading "
        "is the fraction of the true class (row-normalised, so rows sum to 1) on "
        "the perceptually uniform, colour-vision-deficiency-optimised `cividis` "
        "scale; the printed number is the genome count. Off-diagonal cells above "
        "the diagonal are quality DOWNgrades, below the diagonal are UPgrades; "
        "the lower-left cell of each matrix therefore counts low-quality genomes "
        "wrongly promoted to high quality, the single most consequential error "
        "for genome curation."), CAPTIONS)
    plt.close(fig)


def fig_threshold_rates(cfg):
    p = cfg.out_dir / "ws5.2_threshold_analysis.tsv"
    if not p.exists():
        print("  [105] ws5.2_threshold_analysis.tsv missing - run 102 first")
        return
    d = pd.read_csv(p, sep="\t")
    d = d[d.tier.isin(["primary", "reported"])]
    specs = [("contamination", 5.0), ("contamination", 10.0),
             ("completeness", 50.0), ("completeness", 90.0)]
    sets = list(dict.fromkeys(d["set"]))
    used = [t for t in cfg.tools if t in set(d.tool)]
    fig, axes = plt.subplots(2, len(specs), figsize=(2.15 * len(specs) + 0.8, 5.4),
                             sharex=True, squeeze=False)
    for j, (crit, tau) in enumerate(specs):
        for i, rate in enumerate(["false_pass_rate", "false_fail_rate"]):
            ax = axes[i][j]
            sub = d[(d.criterion == crit) & (d.threshold == tau)]
            for si, sname in enumerate(sets):
                g = sub[sub["set"] == sname]
                tl = [t for t in cfg.tools if t in set(g.tool)]
                w = 0.78 / max(1, len(used))
                for t in tl:
                    k = used.index(t)
                    r = g[g.tool == t]
                    if r.empty or not np.isfinite(r[rate].iloc[0]):
                        continue
                    x = si + (k - (len(used) - 1) / 2) * w
                    point_ci_at(ax, x, r[rate].iloc[0], r[f"{rate}_ci_lo"].iloc[0],
                                r[f"{rate}_ci_hi"].iloc[0], cfg.tool_colour(t),
                                marker=cfg.tool_marker(t), width=w * 0.9)
            ax.set_xticks(range(len(sets)))
            ax.set_xticklabels([s.replace("set_", "") for s in sets], rotation=35,
                               ha="right", fontsize=6.5)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(("contamination < " if crit == "contamination"
                              else "completeness >= ") + f"{tau:g}%", fontsize=8)
            if j == 0:
                ax.set_ylabel("false-PASS rate\n(bad genome retained)" if i == 0
                              else "false-FAIL rate\n(good genome discarded)")
    tool_legend(fig, cfg, used, y=-0.02)
    fw.save_figure(fig, cfg, "fig_ws5.2_threshold_error_rates", cap(
        "QC decision errors at the thresholds that matter in practice. A genome "
        "PASSES the contamination criterion when contamination < tau and the "
        "completeness criterion when completeness >= tau. Top row: false-PASS "
        "rate = n(truly fails AND predicted passes) / n(truly fails), i.e. the "
        "fraction of genomes that should have been discarded but were retained "
        "(equals 1 - sensitivity). Bottom row: false-FAIL rate = n(truly passes "
        "AND predicted fails) / n(truly passes), i.e. usable genomes wrongly "
        "discarded (equals 1 - specificity). Filled markers are point estimates; "
        "whiskers are percentile 95% confidence intervals from 2,000 bootstrap "
        "replicates with clusters resampled by dominant reference genome; the "
        "shaded violin is the bootstrap distribution. Panels where one class is "
        "empty by design (e.g. no contaminated genomes in Set A) are blank."),
        CAPTIONS)
    plt.close(fig)


def fig_mae_dotplot(cfg):
    p = cfg.out_dir / "ws5.5_table_S2_rebuilt.tsv"
    if not p.exists():
        print("  [105] ws5.5_table_S2_rebuilt.tsv missing - run 104 first")
        return
    d = pd.read_csv(p, sep="\t")
    d = d[d.tier.isin(["primary", "reported"])]
    sets = list(dict.fromkeys(d["set"]))
    used = [t for t in cfg.tools if t in set(d.tool)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), squeeze=False)
    for j, metric in enumerate(["completeness", "contamination"]):
        ax = axes[0][j]
        sub = d[d.metric == metric]
        for si, sname in enumerate(sets):
            g = sub[sub["set"] == sname]
            tl = [t for t in cfg.tools if t in set(g.tool)]
            w = 0.8 / max(1, len(used))
            for t in tl:
                k = used.index(t)
                r = g[g.tool == t]
                if r.empty:
                    continue
                x = si + (k - (len(used) - 1) / 2) * w
                point_ci_at(ax, x, r.mae.iloc[0], r.mae_ci_lo.iloc[0],
                            r.mae_ci_hi.iloc[0], cfg.tool_colour(t),
                            marker=cfg.tool_marker(t), width=w * 0.8)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s.replace("set_", "") for s in sets], rotation=35,
                           ha="right", fontsize=7)
        ax.set_ylabel(f"{metric} MAE (pp)")
        ax.set_title(metric, fontsize=8)
    tool_legend(fig, cfg, used, y=-0.06)
    fw.save_figure(fig, cfg, "fig_ws5.5_mae_with_confidence_intervals", cap(
        "Mean absolute error per benchmark set and tool, replacing the bar chart "
        "of the original submission. Filled markers are point estimates; whiskers "
        "are percentile 95% confidence intervals from 2,000 bootstrap replicates "
        "with clusters resampled by dominant reference genome (so that multiple "
        "simulations from one reference genome are resampled together). MAE is a "
        "single number per set and tool and therefore has no per-genome "
        "distribution of its own; the distribution of the underlying per-genome "
        "errors is shown in fig_ws5.3_signed_error_by_set and "
        "fig_ws5.3_signed_error_by_contamination_bin."), CAPTIONS)
    plt.close(fig)


def fig_effect_sizes(cfg):
    p = cfg.out_dir / "ws5.4_clustered_tests.tsv"
    if not p.exists():
        print("  [105] ws5.4_clustered_tests.tsv missing - run 104 first")
        return
    d = pd.read_csv(p, sep="\t")
    d = d[d.bh_family.isin(["primary", "reported"])]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 0.30 * len(d) / 2 + 1.7),
                             sharey=True, squeeze=False)
    for j, metric in enumerate(["abs_err_completeness", "abs_err_contamination"]):
        ax = axes[0][j]
        sub = d[d.metric == metric].reset_index(drop=True)
        ylab = []
        for i, r in sub.iterrows():
            y = len(sub) - 1 - i
            c = cfg.tool_colour(r.comparison_tool)
            ax.plot([r.cliffs_delta_ci_lo, r.cliffs_delta_ci_hi], [y, y],
                    color="#222222", lw=0.9)
            sig = bool(r.get("significant_all_three_bh", False))
            ax.plot([r.cliffs_delta], [y], marker=cfg.tool_marker(r.comparison_tool),
                    ms=5, mfc=c if sig else "white", mec="#222222", mew=0.7,
                    ls="none")
            lbl = f"{r['set'].replace('set_', '')} vs {cfg.tool_short(r.comparison_tool)}"
            if str(r.get("status", "")) == "superseded_leaky":
                lbl += "  [SUPERSEDED]"
            ylab.append((y, lbl))
        ax.axvline(0, color="#555555", lw=0.7, ls=(0, (4, 3)))
        for v in (-0.474, -0.33, -0.147, 0.147, 0.33, 0.474):
            ax.axvline(v, color="#DDDDDD", lw=0.5, zorder=0)
        ax.set_yticks([y for y, _ in ylab])
        ax.set_yticklabels([l for _, l in ylab], fontsize=6)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("Cliff's delta  (< 0 favours MAGICC v5)")
        ax.set_title(metric.replace("abs_err_", "") + " absolute error", fontsize=8)
    fw.save_figure(fig, cfg, "fig_ws5.4_effect_sizes_cliffs_delta", cap(
        "Effect sizes for every MAGICC v5 vs comparator test that entered the "
        "Benjamini-Hochberg families. Cliff's delta compares the two absolute-"
        "error distributions: delta < 0 means MAGICC v5's errors are "
        "stochastically smaller. Horizontal bars are percentile 95% confidence "
        "intervals from 2,000 cluster-bootstrap replicates (clusters = dominant "
        "reference genomes). Filled markers indicate comparisons that remain "
        "significant at q < 0.05 after BH correction under ALL THREE tests "
        "(genome-level two-sided Wilcoxon signed-rank, reference-genome-level "
        "two-sided Wilcoxon signed-rank on cluster means, and the cluster "
        "bootstrap of the difference in MAE); open markers do not. Faint vertical "
        "guides mark the conventional |delta| magnitude boundaries 0.147 "
        "(negligible/small), 0.33 (small/medium) and 0.474 (medium/large)."),
        CAPTIONS)
    plt.close(fig)


def fig_palette_check(cfg):
    groups = ["tools", "mimag"]
    visions = ["normal", "deuteranopia", "protanopia", "tritanopia"]
    rows = sum(len(cfg.raw["palette"][g]) for g in groups)
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * rows + 1.0))
    y = 0
    yt, yl = [], []
    for g in groups:
        for k, hexv in cfg.raw["palette"][g].items():
            for x, vis in enumerate(visions):
                c = hexv if vis == "normal" else fw.simulate_cvd(hexv, vis)
                ax.add_patch(plt.Rectangle((x, y), 0.94, 0.86, facecolor=c,
                                          edgecolor="#333333", lw=0.4))
            yt.append(y + 0.43)
            yl.append(f"{g}: {k}  {hexv}")
            y += 1
    ax.set_xlim(0, len(visions))
    ax.set_ylim(0, y)
    ax.set_xticks([x + 0.47 for x in range(len(visions))])
    ax.set_xticklabels(visions, fontsize=7)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6.5)
    ax.set_frame_on(False)
    ax.tick_params(length=0)
    rep = fw.palette_cvd_report({**cfg.raw["palette"]["tools"]})
    ax.set_title(f"Palette under simulated colour vision deficiency\n"
                 f"minimum CIE76 Delta-E between any two tool colours over all "
                 f"four vision types = {rep.min_delta_e.min():.1f} "
                 f"({int(rep.flag_too_similar.sum())} pairs below the "
                 f"{cfg.raw['palette']['min_delta_e']:.0f} threshold)", fontsize=8)
    fw.save_figure(fig, cfg, "fig_ws5.6_palette_cvd_check", cap(
        "Verification that the figure palette satisfies the editorial "
        "colour-vision-deficiency requirement. Each row is one palette entry, "
        "rendered as seen under normal trichromatic vision and under simulated "
        "dichromatic vision (Machado, Oliveira & Fernandes 2009 transformation "
        "matrices at severity 1.0, applied in linear RGB). No green is used "
        "anywhere in the palette, so no figure requires red/green "
        "discrimination. The 6 tool colours were selected by exhaustive search "
        "over all 6-colour subsets of the Okabe-Ito palette to maximise the "
        "minimum CIE76 Delta-E across all four vision types."), CAPTIONS)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    global DEN
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = fw.load_config(args.config)
    n_boot = args.n_boot or cfg.n_boot
    DEN = ("Denominators, stated explicitly per Reviewer 1 (major comment 5): "
           + fw.caption_denominator(cfg).replace("Denominators: ", "", 1))

    sets, frames = load_all(cfg)
    print(f"[105] {len(sets)} sets: {[b.name for b in sets]}")

    fig_signed_error_overall(cfg, sets, frames)
    fig_signed_error_by_cont_bin(cfg, sets, frames)
    fig_signed_error_by_stratum(
        cfg, sets, frames, "true_mimag", "fig_ws5.3_signed_error_by_mimag_class",
        "Contamination signed error stratified by the TRUE MIMAG-inspired quality "
        "class of the genome, one panel per benchmark set.")
    fig_signed_error_by_stratum(
        cfg, sets, frames, "relatedness",
        "fig_ws5.3_signed_error_by_contamination_relatedness",
        "Contamination signed error stratified by the taxonomic relatedness "
        "between the contaminant genomes and the dominant genome, one panel per "
        "benchmark set. `none (uncontaminated)` marks genomes with zero true "
        "contamination.")
    fig_signed_error_by_stratum(
        cfg, sets, frames, "phylum_group", "fig_ws5.3_signed_error_by_phylum",
        "Contamination signed error stratified by the phylum of the dominant "
        "genome (phyla with fewer than "
        f"{cfg.raw['strata']['min_phylum_n']} genomes in a set are pooled as "
        "'Other'), one panel per benchmark set.")
    fig_signed_error_by_stratum(
        cfg, sets, frames, "true_mimag",
        "fig_ws5.3_completeness_signed_error_by_mimag_class",
        "Completeness signed error stratified by the TRUE MIMAG-inspired quality "
        "class, one panel per benchmark set.", metric="completeness")
    fig_mimag_macro_f1(cfg, sets, frames, n_boot)
    fig_confusion_grid(cfg, sets, frames)
    fig_threshold_rates(cfg)
    fig_mae_dotplot(cfg)
    fig_effect_sizes(cfg)
    fig_palette_check(cfg)

    (cfg.fig_dir / "captions.md").write_text(
        "# Figure captions (WS5.6 / WS5.7)\n\n"
        "Every caption below states the denominator used for each percentage, as "
        "required by Reviewer 1 (major comment 5). Every figure is saved as both "
        "PNG (300 dpi) and PDF (vector, Type-42 fonts).\n\n"
        + "\n".join(CAPTIONS) + "\n")
    print(f"[105] wrote {len(CAPTIONS)} figures (PNG + PDF) to {cfg.fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
