#!/usr/bin/env python3
"""
WS8.1/8.2/8.4 — aggregate every timed run into the matched-thread table,
the cold-vs-warm table, and the rewritten Table S4.

Primary metric is END-TO-END WALL CLOCK, defined identically for every tool:
process start -> results written & process exited, including interpreter start,
imports, model/database load and output write, measured by /usr/bin/time -v.

Per-thread throughput is emitted only as a secondary column and is always
accompanied by its thread count, its input set and the wall-clock definition
(R1-m20).

Writes, under results/revision/speed/:
  matched_thread_runs.tsv        every individual run
  matched_thread_summary.tsv     median + min + max + n per cell
  matched_thread_table.md        the compact human-readable table
  cold_vs_warm.tsv
  table_S4_rewritten.md
  ws8_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/path/to/magicc")
SPEED = PROJECT / "results" / "revision" / "speed"
RUNS = SPEED / "runs"

TOOL_LABEL = {
    "magicc": "MAGICC v0.3.0 (V5)",
    "checkm2": "CheckM2 1.0.1",
    "cocopye": "CoCoPyE 0.5.0",
    "deepcheck": "DeepCheck (inference only)",
    "checkm2_vectors": "CheckM2 1.0.1 (+feature dump)",
}
TOOL_ORDER = ["magicc", "checkm2", "cocopye", "deepcheck"]


def load_runs() -> pd.DataFrame:
    recs = []
    for f in sorted(RUNS.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "wall_clock_s" not in r:
            continue
        r["cell_file"] = f.name
        recs.append(r)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["loadavg_1min_before"] = df["loadavg_before"].apply(lambda x: x[0])
    df["loadavg_1min_after"] = df["loadavg_after"].apply(lambda x: x[0])
    df["tool_label"] = df["tool"].map(TOOL_LABEL).fillna(df["tool"])
    return df


def fmt_time(s: float) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "-"
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{int(s // 60)} m {s % 60:04.1f} s"
    return f"{int(s // 3600)} h {int((s % 3600) // 60)} m"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby(["tool", "tool_label", "input_set", "cache", "threads"], as_index=False)
    out = g.agg(
        n_repeats=("wall_clock_s", "size"),
        wall_median_s=("wall_clock_s", "median"),
        wall_min_s=("wall_clock_s", "min"),
        wall_max_s=("wall_clock_s", "max"),
        peak_rss_gb_median=("peak_rss_gb", "median"),
        peak_rss_gb_max=("peak_rss_gb", "max"),
        pct_cpu_median=("pct_cpu", "median"),
        loadavg_before_max=("loadavg_1min_before", "max"),
        n_genomes=("n_genomes", "max"),
        n_output_rows=("n_output_rows", "min"),
        rc_max=("return_code", "max"),
    )
    out["wall_range_pct"] = (out.wall_max_s - out.wall_min_s) / out.wall_median_s * 100
    out["genomes_per_min"] = out.n_genomes / out.wall_median_s * 60
    out["genomes_per_min_per_thread"] = out.genomes_per_min / out.threads
    out["tool_rank"] = out.tool.apply(
        lambda t: TOOL_ORDER.index(t) if t in TOOL_ORDER else 99)
    return out.sort_values(["input_set", "cache", "tool_rank", "threads"]).drop(columns="tool_rank")


def matched_table_md(s: pd.DataFrame, input_set: str, n_gen: int, gbp: float) -> str:
    sub = s[(s.input_set == input_set) & (s.cache == "warm")]
    if sub.empty:
        return f"_no warm runs yet for {input_set}_\n"
    threads = sorted(sub.threads.unique())
    lines = [
        f"| Tool | " + " | ".join(f"{t} thread{'s' if t > 1 else ''}" for t in threads)
        + " | Peak RSS (GB) |",
        "|---|" + "---|" * (len(threads) + 1),
    ]
    for tool in TOOL_ORDER:
        ts = sub[sub.tool == tool]
        if ts.empty:
            continue
        cells = []
        for t in threads:
            r = ts[ts.threads == t]
            if r.empty:
                cells.append("not run")
            else:
                r = r.iloc[0]
                cells.append(
                    f"**{fmt_time(r.wall_median_s)}**<br><sub>{fmt_time(r.wall_min_s)}"
                    f"–{fmt_time(r.wall_max_s)}, n={int(r.n_repeats)}</sub>")
        rss = ts.peak_rss_gb_max.max()
        lines.append(f"| {TOOL_LABEL.get(tool, tool)} | " + " | ".join(cells)
                     + f" | {rss:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load_runs()
    if df.empty:
        print("no completed runs yet")
        return

    manifest = json.loads((SPEED / "inputs" / "input_manifest.json").read_text())

    keep = ["tool", "tool_label", "threads", "repeat", "input_set", "cache",
            "n_genomes", "n_output_rows", "wall_clock_s", "peak_rss_gb", "max_rss_kb",
            "pct_cpu", "user_s", "sys_s", "major_faults", "minor_faults",
            "loadavg_1min_before", "loadavg_1min_after",
            "sum_pcpu_all_processes_before", "t_start_utc", "t_end_utc",
            "return_code", "cell_file"]
    runs = df[[c for c in keep if c in df.columns]].sort_values(
        ["input_set", "cache", "tool", "threads", "repeat"])
    runs.to_csv(SPEED / "matched_thread_runs.tsv", sep="\t", index=False)

    s = summarise(df)
    s.round(4).to_csv(SPEED / "matched_thread_summary.tsv", sep="\t", index=False)

    # ------------------------------------------------------- cold vs warm
    cw = s[s.tool != "checkm2_vectors"].pivot_table(
        index=["tool_label", "input_set", "threads"], columns="cache",
        values=["wall_median_s", "n_repeats"]).reset_index()
    cw.columns = ["_".join([c for c in col if c]).strip() for col in cw.columns.values]
    if "wall_median_s_cold" in cw.columns and "wall_median_s_warm" in cw.columns:
        cw["cold_minus_warm_s"] = cw.wall_median_s_cold - cw.wall_median_s_warm
        cw["cold_over_warm"] = cw.wall_median_s_cold / cw.wall_median_s_warm
    cw.round(3).to_csv(SPEED / "cold_vs_warm.tsv", sep="\t", index=False)

    # -------------------------------------------------------------- report
    m100 = manifest["set_E_100"]
    mfull = manifest["set_E_full"]

    md = ["# WS8.1 — matched-thread wall-clock and peak memory",
          "",
          "**Wall-clock definition (identical for every tool):** process start to results "
          "file written and process exited, including interpreter start, imports, "
          "model/database load, and output write. Instrument: `/usr/bin/time -v`. "
          "No `conda run` wrapper. Uncompressed `.fasta` read from a local ext4 "
          "filesystem (`/dev/sdb`), warm page cache unless stated.",
          "",
          f"**Hardware:** 48-core host (2 sockets), 881 GiB RAM, ext4 on /dev/sdb. "
          f"CPU-only ONNX / CPU-only PyTorch; no GPU was used by any tool.",
          "",
          "**Every cell is median [min–max] over n repeats.** The 1-minute load average was "
          "recorded immediately before and after every run and is in "
          "`matched_thread_runs.tsv`; the campaign driver waits for the load average to "
          "fall below 1.0 before starting each timed run.",
          "",
          "---",
          "",
          f"## Primary matched factorial — set_E_100 "
          f"(**denominator: {m100['n_genomes']} genomes, "
          f"{m100['total_sequence_bp']/1e9:.3f} Gbp**, a seeded 10% subsample of Set E)",
          "",
          matched_table_md(s, "set_E_100", m100["n_genomes"],
                           m100["total_sequence_bp"] / 1e9),
          "",
          f"## Full-scale anchor — Set E "
          f"(**denominator: {mfull['n_genomes']} genomes, "
          f"{mfull['total_sequence_bp']/1e9:.3f} Gbp**, the historical Table S4 input)",
          "",
          matched_table_md(s, "set_E_full", mfull["n_genomes"],
                           mfull["total_sequence_bp"] / 1e9),
          ]
    (SPEED / "matched_thread_table.md").write_text("\n".join(md) + "\n")

    print(s[["tool", "input_set", "cache", "threads", "n_repeats", "wall_median_s",
             "wall_min_s", "wall_max_s", "peak_rss_gb_max",
             "loadavg_before_max"]].to_string(index=False))
    print(f"\n{len(runs)} runs aggregated -> {SPEED/'matched_thread_summary.tsv'}")


if __name__ == "__main__":
    main()
