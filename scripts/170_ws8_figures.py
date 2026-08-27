#!/usr/bin/env python3
"""
WS8 figures — colour-vision-deficiency safe (E4).

Palette is the project's validated CVD-optimised Okabe-Ito subset taken from
scripts/config_revision_metrics.yaml (the same one scripts/105_distribution_plots.py
uses; min CIE76 Delta-E = 24.6 across normal, deuteranopic, protanopic and
tritanopic vision). No green is used, so no red/green discrimination is ever
required, and colour is always accompanied by a redundant cue (marker shape and
direct line labels).

Panels:
  ws8_fig1_wallclock_vs_threads.(png|pdf)  wall clock vs thread count, log y
  ws8_fig2_peak_rss.(png|pdf)              peak RSS, every repeat plotted
  ws8_fig3_reconciliation.(png|pdf)        the 40 s / 97.5 s / 1,451 provenance
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402
import yaml                                  # noqa: E402

PROJECT = Path("/path/to/magicc")
SPEED = PROJECT / "results" / "revision" / "speed"
FIGS = SPEED / "figures"
CFG = PROJECT / "scripts" / "config_revision_metrics.yaml"

TOOL_KEY = {"magicc": "magicc_v5", "checkm2": "checkm2",
            "cocopye": "cocopye", "deepcheck": "deepcheck"}
TOOL_LABEL = {"magicc": "MAGICC (V5)", "checkm2": "CheckM2",
              "cocopye": "CoCoPyE", "deepcheck": "DeepCheck (inference only)"}
ORDER = ["magicc", "checkm2", "cocopye", "deepcheck"]

# runs whose page cache was NOT warm (first touch after the host reboot);
# identified in results/revision/speed/cold_start_forensics.tsv by non-zero
# `/usr/bin/time -v` File system inputs. Kept in every median; marked in fig 1.
COLD_CELLS = {
    "magicc__t32__set_E_100__warm__r1.json",
    "magicc__t32__set_E_full__warm__r1.json",
    "deepcheck__t32__set_E_full__warm__r1.json",
    "checkm2__t8__set_E_100__warm__r1.json",
    "cocopye__t32__set_E_100__warm__r1.json",
}


def load_palette():
    cfg = yaml.safe_load(CFG.read_text())
    return cfg["palette"]["tools"], cfg["palette"]["markers"]


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def fig1(summary: pd.DataFrame, runs: pd.DataFrame, colours, markers, manifest):
    warm = summary[(summary.cache == "warm") & (summary.tool.isin(ORDER))]
    sets = [s for s in ["set_E_100", "set_E_full"] if s in warm.input_set.unique()]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.2), squeeze=False)
    for ax, iset in zip(axes[0], sets):
        sub = warm[warm.input_set == iset]
        n_gen = int(sub.n_genomes.max())
        gbp = manifest[iset]["total_sequence_bp"] / 1e9
        for tool in ORDER:
            t = sub[sub.tool == tool].sort_values("threads")
            if t.empty:
                continue
            c = colours[TOOL_KEY[tool]]
            edge = "#333333" if tool == "cocopye" else c
            ax.plot(t.threads, t.wall_median_s, marker=markers[TOOL_KEY[tool]],
                    color=c, markeredgecolor=edge, markeredgewidth=0.9,
                    linewidth=1.8, markersize=7, label=TOOL_LABEL[tool], zorder=3)
            ax.fill_between(t.threads, t.wall_min_s, t.wall_max_s,
                            color=c, alpha=0.18, linewidth=0, zorder=2)
            r = runs[(runs.tool == tool) & (runs.input_set == iset) & (runs.cache == "warm")]
            cold = r[r.cell_file.isin(COLD_CELLS)] if "cell_file" in r.columns else r.iloc[0:0]
            hot = r.drop(cold.index)
            ax.scatter(hot.threads, hot.wall_clock_s, s=9, color=c, edgecolor=edge,
                       linewidth=0.4, alpha=0.85, zorder=4)
            if not cold.empty:
                ax.scatter(cold.threads, cold.wall_clock_s, s=52, facecolor="none",
                           edgecolor="#222222", linewidth=1.1, marker="o", zorder=6)
                for _, cr in cold.iterrows():
                    dy = 8 if tool in ("magicc", "checkm2") else -14
                    ax.annotate("cold cache", (cr.threads, cr.wall_clock_s),
                                textcoords="offset points", xytext=(-7, dy),
                                ha="right", fontsize=6.4, color="#222222")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Threads")
        ax.set_ylabel("End-to-end wall clock (s, log scale)")
        ax.set_title(f"{iset}  —  n = {n_gen} genomes, {gbp:.3f} Gbp", fontsize=9)
        ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.35)
        for lbl in [60, 3600]:
            ax.axhline(lbl, color="#999999", linewidth=0.6, linestyle=":", zorder=1)
        ax.text(0.02, 0.02, "dotted lines: 1 min, 1 h", transform=ax.transAxes,
                fontsize=6.5, color="#666666")
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Matched-hardware, matched-input end-to-end wall clock\n"
                 "median (line) with full min–max range (band); every individual repeat plotted",
                 fontsize=9.5, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.text(0.5, -0.02,
             "Wall clock = process start to results written, including model/database load. "
             "48-core host, uncompressed FASTA on local ext4, page cache warm except where marked.\n"
             "Ringed points are the first run to touch that input set or database after the host "
             "reboot and read it from disk; they are retained in every median and range.\n"
             "CVD-safe palette (min CIE76 dE = 33.4 across normal, deuteranopic, protanopic and "
             "tritanopic vision); marker shape is redundant with colour.",
             ha="center", va="top", fontsize=6.5, color="#555555", linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"ws8_fig1_wallclock_vs_threads.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig2(runs: pd.DataFrame, colours, markers, manifest):
    base = runs[(runs.cache == "warm") & (runs.tool.isin(ORDER))]
    sets = [s for s in ["set_E_100", "set_E_full"] if not base[base.input_set == s].empty]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(5.0 * len(sets), 4.2), squeeze=False)
    for ax, iset in zip(axes[0], sets):
        warm = base[base.input_set == iset]
        n_gen = int(warm.n_genomes.max())
        gbp = manifest[iset]["total_sequence_bp"] / 1e9
        for i, tool in enumerate(ORDER):
            t = warm[warm.tool == tool]
            if t.empty:
                continue
            c = colours[TOOL_KEY[tool]]
            edge = "#333333" if tool == "cocopye" else c
            jit = (np.random.default_rng(1).random(len(t)) - 0.5) * 0.22
            ax.scatter(np.full(len(t), i) + jit, t.peak_rss_gb, s=30,
                       color=c, edgecolor=edge, linewidth=0.7,
                       marker=markers[TOOL_KEY[tool]], zorder=3)
            med = t.peak_rss_gb.median()
            ax.plot([i - 0.28, i + 0.28], [med, med], color="#222222",
                    linewidth=1.6, zorder=4)
            ax.annotate(f"{med:.2f} GB", (i, med), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([TOOL_LABEL[t].replace(" (inference only)", "\n(inference only)")
                            for t in ORDER], fontsize=7.5)
        ax.set_yscale("log")
        ax.set_ylim(0.2, 40)
        ax.set_ylabel("Peak resident set size (GB, log scale)")
        ax.set_title(f"{iset}  —  n = {n_gen:,} genomes, {gbp:.3f} Gbp", fontsize=9)
        ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.35)
    fig.suptitle("Peak memory — every run at every thread count plotted", fontsize=9.5)
    fig.text(0.5, -0.01,
             "Horizontal bar = median across all thread counts and repeats. "
             "`/usr/bin/time -v` Maximum resident set size.\n"
             "DeepCheck's figure excludes the CheckM2 run it requires to produce its input; "
             "on the full set only MAGICC and DeepCheck were run at all four thread counts.",
             ha="center", va="top", fontsize=6.5, color="#555555", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"ws8_fig2_peak_rss.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig3(rec: dict, summary: pd.DataFrame, colours):
    """The R1-m11 reconciliation as one picture."""
    blue = colours["magicc_v5"]
    grey = colours["magicc_v3"]
    sky = colours["magicc_v4"]

    labels, vals, cols, notes = [], [], [], []

    labels.append("“40 s”\nmain text")
    vals.append(rec["figure_40_s"]["arithmetic_check_s"])
    cols.append(grey)
    notes.append("NOT MEASURED\n1,000 ÷ 1,451 min⁻¹")

    labels.append("74.4 s\nscript 40, V3")
    vals.append(rec["figure_74_4_s_and_7_9_s"]["one_thread_s"])
    cols.append(sky)
    notes.append("end-to-end,\ndirect python -m")

    labels.append("97.5 s\nTable S4, V3")
    vals.append(rec["figure_97_5_s"]["value_s"])
    cols.append(sky)
    notes.append("end-to-end,\nvia `conda run`")

    new = summary[(summary.tool == "magicc") & (summary.input_set == "set_E_full")
                  & (summary.cache == "warm") & (summary.threads == 1)]
    if not new.empty:
        r = new.iloc[0]
        labels.append(f"{r.wall_median_s:.1f} s\nWS8, V5, n={int(r.n_repeats)}")
        vals.append(r.wall_median_s)
        cols.append(blue)
        notes.append("end-to-end,\nidle host,\nload recorded")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=cols, edgecolor="#333333", linewidth=0.7, width=0.62)
    bars[0].set_hatch("///")
    for i, (b, n) in enumerate(zip(bars, notes)):
        ax.annotate(f"{vals[i]:.1f} s", (b.get_x() + b.get_width() / 2, vals[i]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.5, fontweight="bold")
        ax.annotate(n, (b.get_x() + b.get_width() / 2, 2), ha="center",
                    va="bottom", fontsize=6.8, color="white" if i else "#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Seconds for 1,000 genomes (Set E), 1 thread")
    ax.set_title("R1-m11: what each published MAGICC timing actually measured",
                 fontsize=9.5)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.35)
    fig.text(0.5, -0.04,
             "Hatched bar is not a measurement: it is 1,000 genomes divided by "
             "1,451 genomes/min/thread, which was itself the arithmetic MEAN OF FIVE "
             "PER-SET RATES computed from MAGICC's internal compute-phase timer "
             "(feature extraction + inference only). Pooled over the same five sets that "
             "timer gives 1,066 genomes/min/thread, not 1,451.",
             ha="center", fontsize=6.5, color="#555555", wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"ws8_fig3_reconciliation.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    style()
    colours, markers = load_palette()
    summary = pd.read_csv(SPEED / "matched_thread_summary.tsv", sep="\t")
    runs = pd.read_csv(SPEED / "matched_thread_runs.tsv", sep="\t")
    manifest = json.loads((SPEED / "inputs" / "input_manifest.json").read_text())
    rec = json.loads((SPEED / "reconciliation_40s_vs_97.5s.json").read_text())

    fig1(summary, runs, colours, markers, manifest)
    fig2(runs, colours, markers, manifest)
    fig3(rec, summary, colours)

    # CVD verification of the exact colours used, via the project's own checker
    sys.path.insert(0, str(PROJECT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fw", PROJECT / "scripts" / "101_metrics_framework.py")
    fw = importlib.util.module_from_spec(spec)
    sys.modules["fw"] = fw          # dataclasses in 101 resolve via sys.modules
    spec.loader.exec_module(fw)
    used = {k: colours[TOOL_KEY[k]] for k in ORDER}
    rep = fw.palette_cvd_report(used)
    rep.to_csv(SPEED / "figures" / "palette_cvd_report.tsv", sep="\t", index=False)
    bad = rep[rep.min_delta_e < 20.0]
    print(f"[cvd] palette used: {used}")
    print(f"[cvd] min Delta-E over all pairs x all vision types = "
          f"{rep.min_delta_e.min():.1f} ({'PASS' if bad.empty else 'FAIL'})")
    print(f"figures -> {FIGS}")


if __name__ == "__main__":
    main()
