#!/usr/bin/env python3
"""
WS8.3 support — quantify, for MAGICC V5, exactly how much of the end-to-end wall
clock the *internal* compute-phase timer (the one the 1,451 figure was built on)
never saw.

Parses the MAGICC CLI's own log lines out of every captured stdout file and pairs
them with the `/usr/bin/time -v` wall clock for the same run.

Writes results/revision/speed/magicc_phase_breakdown.tsv
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

SPEED = Path("/path/to/magicc/results/revision/speed")
RUNS = SPEED / "runs"

PATTERNS = {
    "feature_extraction_s": r"Feature extraction:\s+([0-9.]+)s",
    "normalization_s": r"Normalization:\s+([0-9.]+)s",
    "onnx_inference_s": r"ONNX inference:\s+([0-9.]+)s",
    "internal_total_s": r"Total time:\s+([0-9.]+)s",
}


def main() -> None:
    rows = []
    for js in sorted(RUNS.glob("magicc*__*.json")):
        rec = json.loads(js.read_text())
        if "wall_clock_s" not in rec:
            continue
        so = js.with_suffix("").with_suffix(".stdout.txt")
        so = RUNS / (js.name[:-5] + ".stdout.txt")
        if not so.exists():
            continue
        txt = so.read_text()
        row = {k: rec[k] for k in ["tool", "threads", "repeat", "input_set",
                                   "cache", "n_genomes", "wall_clock_s", "peak_rss_gb"]}
        for name, pat in PATTERNS.items():
            m = re.search(pat, txt)
            row[name] = float(m.group(1)) if m else None
        rows.append(row)

    if not rows:
        print("no MAGICC runs with stdout yet")
        return

    df = pd.DataFrame(rows)
    df["compute_phase_s"] = df.feature_extraction_s.fillna(0) + df.onnx_inference_s.fillna(0)
    df["overhead_outside_internal_timer_s"] = df.wall_clock_s - df.internal_total_s
    df["overhead_outside_compute_phase_s"] = df.wall_clock_s - df.compute_phase_s
    df["pct_wall_not_seen_by_1451_timer"] = (
        df.overhead_outside_compute_phase_s / df.wall_clock_s * 100)
    df = df.sort_values(["input_set", "cache", "tool", "threads", "repeat"])
    df.round(3).to_csv(SPEED / "magicc_phase_breakdown.tsv", sep="\t", index=False)

    g = (df[df.cache == "warm"]
         .groupby(["tool", "input_set", "threads"], as_index=False)
         .agg(n=("wall_clock_s", "size"),
              wall_s=("wall_clock_s", "median"),
              compute_phase_s=("compute_phase_s", "median"),
              overhead_s=("overhead_outside_compute_phase_s", "median"),
              pct_hidden=("pct_wall_not_seen_by_1451_timer", "median")))
    print(g.round(2).to_string(index=False))
    print(f"\nwrote {SPEED/'magicc_phase_breakdown.tsv'}")
    print("\n'compute_phase_s' is feature extraction + ONNX inference, i.e. exactly what")
    print("scripts/26_benchmark_run_magicc.py stored as `wall_clock_s` and what the")
    print("published 1,451 genomes/min/thread figure was computed from.")


if __name__ == "__main__":
    main()
