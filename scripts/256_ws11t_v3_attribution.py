#!/usr/bin/env python3
"""
WS11.T (T3) — attribute the archived V3 (74.4 s) vs measured V5 (49.0 s)
single-thread gap on the 1,000-genome Set E.

Design: a five-rung ladder in which each rung differs from the next by exactly
ONE thing, all at 1 thread on the identical 1,000 genomes (4,806,980,767 bp),
all on the same host in the same time window, n = 3 each.

  rung 1  magicc                 production console script, --input-list,  V5
  rung 2  magicc_dir             production console script, --input DIR,   V5
  rung 3  magicc_v5code          code-root launcher + current code tree,   V5
  rung 4  magicc_v3code_nostats  launcher + V3 tree, 19 assembly stats OFF, V3
  rung 5  magicc_v3code          launcher + V3 tree, UNMODIFIED,            V3

  rung1 -> rung2   isolates input-discovery mode (list vs directory scan)
  rung2 -> rung3   isolates the launcher itself (must be ~0)
  rung3 -> rung4   isolates the MODEL and the 26-vs-7-wide assembly input
                   (V5 code with 7 features vs V3 code with the 19 real
                   assembly statistics short-circuited, i.e. the same work)
  rung4 -> rung5   isolates THE ASSEMBLY STATISTICS THEMSELVES

Writes results/revision/speed_v3/v3_vs_v5_attribution.{tsv,md}
       results/revision/speed_v3/v3_vs_v5_phase_runs.tsv
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROJECT = Path("/path/to/magicc")
OLD = PROJECT / "results" / "revision" / "speed"
NEW = PROJECT / "results" / "revision" / "speed_v3"

PATTERNS = {
    "feature_extraction_s": r"Feature extraction:\s+([0-9.]+)s",
    "normalization_s": r"Normalization:\s+([0-9.]+)s",
    "onnx_inference_s": r"ONNX inference:\s+([0-9.]+)s",
    "internal_total_s": r"Total time:\s+([0-9.]+)s",
}

LADDER = [
    ("magicc", "1. production CLI, --input-list, V5",
     "models/magicc_v5.onnx", "current (HEAD)", "list"),
    ("magicc_dir", "2. production CLI, --input DIR, V5",
     "models/magicc_v5.onnx", "current (HEAD)", "directory"),
    ("magicc_v5code", "3. launcher + current code tree, V5",
     "models/magicc_v5.onnx", "current (HEAD)", "directory"),
    ("magicc_v3code_nostats", "4. launcher + V3 tree, assembly stats OFF, V3",
     "models/magicc_v3.onnx", "471eb28 patched", "directory"),
    ("magicc_v3code", "5. launcher + V3 tree UNMODIFIED, V3",
     "models/magicc_v3.onnx", "471eb28", "directory"),
]

# archived V3 breakdown, results/revision/speed/reconciliation_40s_vs_97.5s.json
ARCHIVED_V3_97_5 = dict(wall_clock_s=97.52, feature_extraction_s=91.7,
                        normalization_s=0.251, onnx_inference_s=1.81,
                        process_overhead_s=2.92)
ARCHIVED_V3_74_4 = 74.364
ARCHIVED_V5_49_0 = 49.015


def collect(dirpath: Path, tools: set[str]) -> pd.DataFrame:
    rows = []
    for js in sorted((dirpath / "runs").glob("*.json")):
        rec = json.loads(js.read_text())
        if "wall_clock_s" not in rec or rec["tool"] not in tools:
            continue
        if rec["threads"] != 1:
            continue
        so = dirpath / "runs" / (js.name[:-5] + ".stdout.txt")
        row = {k: rec.get(k) for k in
               ["tool", "threads", "repeat", "input_set", "cache", "n_genomes",
                "n_output_rows", "wall_clock_s", "peak_rss_gb", "pct_cpu",
                "user_s", "sys_s", "fs_inputs", "return_code", "model",
                "code_root", "t_start_utc"]}
        row["loadavg_1min_before"] = rec["loadavg_before"][0]
        if so.exists():
            txt = so.read_text()
            for name, pat in PATTERNS.items():
                m = re.search(pat, txt)
                row[name] = float(m.group(1)) if m else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    tools = {t for t, *_ in LADDER}
    df = pd.concat([collect(OLD, tools), collect(NEW, tools)], ignore_index=True)
    if df.empty:
        print("no ladder runs yet")
        return
    df["process_overhead_s"] = df.wall_clock_s - df.internal_total_s
    df = df.sort_values(["tool", "repeat"])
    df.round(4).to_csv(NEW / "v3_vs_v5_phase_runs.tsv", sep="\t", index=False)

    g = df.groupby("tool", as_index=False).agg(
        n=("wall_clock_s", "size"),
        wall_median_s=("wall_clock_s", "median"),
        wall_min_s=("wall_clock_s", "min"),
        wall_max_s=("wall_clock_s", "max"),
        feature_extraction_s=("feature_extraction_s", "median"),
        normalization_s=("normalization_s", "median"),
        onnx_inference_s=("onnx_inference_s", "median"),
        internal_total_s=("internal_total_s", "median"),
        process_overhead_s=("process_overhead_s", "median"),
        peak_rss_gb=("peak_rss_gb", "median"),
        pct_cpu=("pct_cpu", "median"),
        load1_before_max=("loadavg_1min_before", "max"),
        rc_max=("return_code", "max"),
    )
    order = {t: i for i, (t, *_) in enumerate(LADDER)}
    lab = {t: l for t, l, *_ in LADDER}
    mdl = {t: m for t, _, m, *_ in LADDER}
    tree = {t: c for t, _, _, c, _ in LADDER}
    imode = {t: i for t, _, _, _, i in LADDER}
    g["rung"] = g.tool.map(order)
    g["arm"] = g.tool.map(lab)
    g["model"] = g.tool.map(mdl)
    g["code_tree"] = g.tool.map(tree)
    g["input_mode"] = g.tool.map(imode)
    g = g.sort_values("rung").reset_index(drop=True)
    g["delta_vs_previous_rung_s"] = g.wall_median_s.diff().round(3)
    cols = ["rung", "arm", "tool", "code_tree", "model", "input_mode", "n",
            "wall_median_s", "wall_min_s", "wall_max_s",
            "delta_vs_previous_rung_s",
            "feature_extraction_s", "normalization_s", "onnx_inference_s",
            "internal_total_s", "process_overhead_s",
            "peak_rss_gb", "pct_cpu", "load1_before_max", "rc_max"]
    g[cols].round(3).to_csv(NEW / "v3_vs_v5_attribution.tsv", sep="\t", index=False)
    print(g[["rung", "tool", "n", "wall_median_s", "delta_vs_previous_rung_s",
             "feature_extraction_s", "onnx_inference_s",
             "process_overhead_s"]].to_string(index=False))
    print(f"\nwrote {NEW/'v3_vs_v5_attribution.tsv'}")


if __name__ == "__main__":
    main()
