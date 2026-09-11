#!/usr/bin/env python3
"""Sensitivity of the pooled five-set confidence intervals to the clustering unit.

The WS5 pooled bootstrap resamples 2,586 clusters, which are
benchmark-set x reference-genome units. Sets A_v2, B_v2 and E all draw their
dominant genomes from the same 1,810-genome finished test pool, so those 2,586
units come from only 1,648 DISTINCT reference genomes. A genome contributing to
more than one benchmark set therefore forms a separate cluster in each, which
treats correlated observations as independent and can narrow the interval.

This script recomputes the pooled MAE confidence intervals under both
definitions so the manuscript can state the clustering unit precisely and report
the stricter variant as a sensitivity analysis (reviewer point R1-m16).

Point estimates are unaffected by the clustering unit; only the intervals move.

Output: results/revision/metrics/ws5.10_pooled_cluster_definition_sensitivity.tsv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = "/media/Data_1/tianrm/projects/magicc2"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PER_GENOME = os.path.join(
    ROOT, "results/revision/metrics/ws5.3_signed_errors_per_genome.tsv.gz")
OUT = os.path.join(
    ROOT, "results/revision/metrics/ws5.10_pooled_cluster_definition_sensitivity.tsv")

SETS = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]
TOOLS = ["magicc_v5", "checkm2", "cocopye", "deepcheck"]
METRICS = ["completeness", "contamination"]
N_BOOT = 2000


def stable_hash(text: str) -> int:
    """CRC-32 seed, matching scripts/101_metrics_framework.py (defect D1 fix)."""
    import zlib
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def cluster_bootstrap_mae(df: pd.DataFrame, by: str, seed: int) -> tuple:
    """95 % percentile cluster bootstrap of the MAE, resampling whole clusters."""
    g = df.groupby(by)["abs_error"].agg(["sum", "count"])
    sums = g["sum"].to_numpy()
    counts = g["count"].to_numpy()
    n = len(sums)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws[b] = sums[idx].sum() / counts[idx].sum()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), n


def main() -> int:
    if not os.path.isfile(PER_GENOME):
        raise FileNotFoundError(PER_GENOME)
    d = pd.read_csv(PER_GENOME, sep="\t")
    d = d[d["set"].isin(SETS)].copy()
    d["set_x_ref"] = d["set"] + "|" + d["dominant_accession"]

    n_distinct = d["dominant_accession"].nunique()
    n_setref = d["set_x_ref"].nunique()
    print(f"distinct reference genomes: {n_distinct} | set x reference clusters: {n_setref}")

    rows = []
    for tool in TOOLS:
        for metric in METRICS:
            sub = d[(d["tool"] == tool) & (d["metric"] == metric)].copy()
            if sub.empty:
                print(f"  WARNING: no rows for {tool}/{metric}")
                continue
            sub["abs_error"] = sub["signed_error"].abs()
            mae = float(sub["abs_error"].mean())
            seed = stable_hash(f"pooled_cluster_sensitivity|{tool}|{metric}")
            lo_a, hi_a, n_a = cluster_bootstrap_mae(sub, "set_x_ref", seed)
            lo_b, hi_b, n_b = cluster_bootstrap_mae(sub, "dominant_accession", seed)
            rows.append(dict(
                tool=tool, metric=metric, n=len(sub), mae=round(mae, 4),
                n_clusters_set_x_ref=n_a,
                ci_lo_set_x_ref=round(lo_a, 4), ci_hi_set_x_ref=round(hi_a, 4),
                ci_width_set_x_ref=round(hi_a - lo_a, 4),
                n_clusters_distinct_ref=n_b,
                ci_lo_distinct_ref=round(lo_b, 4), ci_hi_distinct_ref=round(hi_b, 4),
                ci_width_distinct_ref=round(hi_b - lo_b, 4),
                ci_width_ratio=round((hi_b - lo_b) / (hi_a - lo_a), 4),
                seed=seed,
            ))
            print(f"  {tool:10s} {metric:13s} MAE={mae:7.3f}  "
                  f"set x ref [{lo_a:.3f}, {hi_a:.3f}] w={hi_a-lo_a:.3f}  "
                  f"distinct ref [{lo_b:.3f}, {hi_b:.3f}] w={hi_b-lo_b:.3f}  "
                  f"ratio={(hi_b-lo_b)/(hi_a-lo_a):.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"\nwrote {OUT} ({len(out)} rows)")
    print("Point estimates are identical under both definitions; only the "
          "intervals differ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
