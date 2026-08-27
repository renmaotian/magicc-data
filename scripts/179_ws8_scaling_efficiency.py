#!/usr/bin/env python3
"""
WS8.1 — parallel scaling efficiency, stated honestly.

speedup(T)     = wall(1 thread) / wall(T threads)
efficiency(T)  = speedup(T) / T

Both are computed from the SAME end-to-end wall clock used everywhere else
(process start -> results written, including model/database load), so the fixed
start-up cost is inside the numerator and the denominator. That is deliberate:
it is what a user experiences. It also means efficiency falls with thread count
for every tool, which is the honest result and is exactly why a "genomes per
minute per thread" figure cannot be quoted without its thread count (R1-m20).

Writes results/revision/speed/scaling_efficiency.tsv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SPEED = Path("/path/to/magicc/results/revision/speed")


def main() -> None:
    s = pd.read_csv(SPEED / "matched_thread_summary.tsv", sep="\t")
    warm = s[(s.cache == "warm") & (s.tool != "checkm2_vectors")]

    rows = []
    for (tool, iset), g in warm.groupby(["tool", "input_set"]):
        base = g[g.threads == 1]
        if base.empty:
            continue
        t1 = float(base.iloc[0].wall_median_s)
        for _, r in g.sort_values("threads").iterrows():
            sp = t1 / r.wall_median_s
            rows.append({
                "tool": tool,
                "tool_label": r.tool_label,
                "input_set": iset,
                "n_genomes": int(r.n_genomes),
                "threads": int(r.threads),
                "wall_median_s": round(float(r.wall_median_s), 2),
                "n_repeats": int(r.n_repeats),
                "speedup_vs_1_thread": round(float(sp), 2),
                "parallel_efficiency_pct": round(float(sp / r.threads * 100), 1),
                "genomes_per_min": round(float(r.genomes_per_min), 1),
                "genomes_per_min_per_thread": round(float(r.genomes_per_min_per_thread), 1),
            })
    df = pd.DataFrame(rows).sort_values(["input_set", "tool", "threads"])
    df.to_csv(SPEED / "scaling_efficiency.tsv", sep="\t", index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {SPEED/'scaling_efficiency.tsv'}")


if __name__ == "__main__":
    main()
