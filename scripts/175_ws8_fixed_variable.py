#!/usr/bin/env python3
"""
WS8.1 support — decompose each tool's wall clock into a fixed start-up cost and a
per-genome marginal cost, from the two measured workload sizes (100 and 1,000
genomes) at the same thread count.

Why this matters for honesty: the complete 4-tool x 4-thread factorial had to be
run on 100 genomes, and a 100-genome workload amortises a tool's fixed database/
model-load cost over 10x fewer genomes than the 1,000-genome Set E does. That
DISADVANTAGES the tools with large databases relative to the historical
comparison. Quantifying the fixed component makes the size of that bias explicit
instead of leaving it as an unstated caveat.

    wall(n) = a + b*n     solved exactly from the two measured points

Writes results/revision/speed/fixed_variable_costs.tsv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SPEED = Path("/path/to/magicc/results/revision/speed")


def main() -> None:
    s = pd.read_csv(SPEED / "matched_thread_summary.tsv", sep="\t")
    warm = s[s.cache == "warm"]

    rows = []
    for tool in warm.tool.unique():
        for th in sorted(warm[warm.tool == tool].threads.unique()):
            a = warm[(warm.tool == tool) & (warm.threads == th)
                     & (warm.input_set == "set_E_100")]
            b = warm[(warm.tool == tool) & (warm.threads == th)
                     & (warm.input_set == "set_E_full")]
            if a.empty or b.empty:
                continue
            n1, t1 = float(a.iloc[0].n_genomes), float(a.iloc[0].wall_median_s)
            n2, t2 = float(b.iloc[0].n_genomes), float(b.iloc[0].wall_median_s)
            slope = (t2 - t1) / (n2 - n1)
            fixed = t1 - slope * n1
            rows.append({
                "tool": tool, "threads": int(th),
                "wall_100_s": round(t1, 2), "wall_1000_s": round(t2, 2),
                "fixed_startup_s": round(fixed, 2),
                "marginal_s_per_genome": round(slope, 4),
                "fixed_share_of_100genome_run_pct": round(100 * fixed / t1, 1),
                "fixed_share_of_1000genome_run_pct": round(100 * fixed / t2, 1),
            })

    if not rows:
        print("need both set_E_100 and set_E_full at a common thread count; "
              "not available yet")
        return

    df = pd.DataFrame(rows).sort_values(["tool", "threads"])
    df.to_csv(SPEED / "fixed_variable_costs.tsv", sep="\t", index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {SPEED/'fixed_variable_costs.tsv'}")
    print("\nNegative fixed_startup_s would indicate super-linear scaling (e.g. memory "
          "pressure at the larger size) and must be reported as such, not clipped to 0.")


if __name__ == "__main__":
    main()
