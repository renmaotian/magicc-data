#!/usr/bin/env python3
"""
Script 36b: Parse CheckM2 results for the 5 new finished-genome benchmark sets.

Merges CheckM2 quality_report.tsv with metadata.tsv to create checkm2_predictions.tsv
with columns: genome_id, true_completeness, true_contamination, dominant_accession,
dominant_phylum, sample_type, n_contigs, total_length, pred_completeness,
pred_contamination, wall_clock_s, n_threads
"""

import pandas as pd
import os
import numpy as np

PROJECT_DIR = "/home/tianrm/projects/magicc2"
BENCHMARK_DIR = os.path.join(PROJECT_DIR, "data", "benchmarks")
N_THREADS = 32

# Define the 5 sets with their paths
sets_info = [
    {"label": "motivating_v2/set_A", "path": os.path.join(BENCHMARK_DIR, "motivating_v2", "set_A"), "n_genomes": 1000},
    {"label": "motivating_v2/set_B", "path": os.path.join(BENCHMARK_DIR, "motivating_v2", "set_B"), "n_genomes": 1000},
    {"label": "set_A_v2", "path": os.path.join(BENCHMARK_DIR, "set_A_v2"), "n_genomes": 1000},
    {"label": "set_B_v2", "path": os.path.join(BENCHMARK_DIR, "set_B_v2"), "n_genomes": 1000},
    {"label": "set_E", "path": os.path.join(BENCHMARK_DIR, "set_E"), "n_genomes": 1000},
]

all_results = []

for info in sets_info:
    label = info["label"]
    set_dir = info["path"]
    metadata_path = os.path.join(set_dir, "metadata.tsv")
    checkm2_output_dir = os.path.join(set_dir, "checkm2_output")
    quality_report_path = os.path.join(checkm2_output_dir, "quality_report.tsv")
    wallclock_path = os.path.join(set_dir, "checkm2_wallclock.txt")
    output_path = os.path.join(set_dir, "checkm2_predictions.tsv")

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Read metadata
    if not os.path.exists(metadata_path):
        print(f"  WARNING: metadata.tsv not found at {metadata_path}, skipping")
        continue
    metadata = pd.read_csv(metadata_path, sep="\t")
    print(f"  Metadata: {len(metadata)} genomes")
    print(f"  Metadata columns: {list(metadata.columns)}")

    # Read CheckM2 quality report
    if not os.path.exists(quality_report_path):
        print(f"  WARNING: quality_report.tsv not found at {quality_report_path}, skipping")
        continue
    checkm2 = pd.read_csv(quality_report_path, sep="\t")
    print(f"  CheckM2 results: {len(checkm2)} genomes")

    # Read wall-clock time
    if os.path.exists(wallclock_path):
        with open(wallclock_path) as f:
            wall_clock_s = float(f.read().strip())
        print(f"  Wall-clock time: {wall_clock_s:.0f}s ({wall_clock_s/60:.1f} min)")
    else:
        wall_clock_s = 0.0
        print(f"  WARNING: checkm2_wallclock.txt not found, using 0s")

    # CheckM2 uses 'Name' column (without .fasta extension)
    # Our metadata uses 'genome_id' column
    checkm2_slim = checkm2[["Name", "Completeness", "Contamination"]].copy()
    checkm2_slim.rename(columns={
        "Name": "genome_id",
        "Completeness": "pred_completeness",
        "Contamination": "pred_contamination"
    }, inplace=True)

    # Merge on genome_id
    merged = metadata.merge(checkm2_slim, on="genome_id", how="left")

    # Add timing info
    merged["wall_clock_s"] = wall_clock_s
    merged["n_threads"] = N_THREADS

    # Select output columns (keep only the required ones)
    output_cols = [
        "genome_id", "true_completeness", "true_contamination",
        "dominant_accession", "dominant_phylum", "sample_type",
        "n_contigs", "total_length",
        "pred_completeness", "pred_contamination",
        "wall_clock_s", "n_threads"
    ]
    # Only keep columns that exist
    available_cols = [c for c in output_cols if c in merged.columns]
    merged_out = merged[available_cols]

    # Check for missing predictions
    n_missing = merged_out["pred_completeness"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} genomes missing CheckM2 predictions")

    # Save
    merged_out.to_csv(output_path, sep="\t", index=False)
    print(f"  Saved: {output_path}")
    print(f"  Output columns: {list(merged_out.columns)}")

    # Compute accuracy
    valid = merged_out.dropna(subset=["pred_completeness", "pred_contamination"])
    n_valid = len(valid)
    comp_mae = np.mean(np.abs(valid["true_completeness"] - valid["pred_completeness"]))
    cont_mae = np.mean(np.abs(valid["true_contamination"] - valid["pred_contamination"]))
    genomes_per_min = n_valid / (wall_clock_s / 60) if wall_clock_s > 0 else 0
    genomes_per_min_per_thread = genomes_per_min / N_THREADS

    print(f"\n  --- {label} Summary ---")
    print(f"  Genomes: {n_valid}")
    print(f"  Wall-clock: {wall_clock_s:.0f}s ({wall_clock_s/60:.1f} min)")
    print(f"  Completeness MAE: {comp_mae:.2f}%")
    print(f"  Contamination MAE: {cont_mae:.2f}%")
    print(f"  Speed: {genomes_per_min:.1f} genomes/min ({N_THREADS} threads)")
    print(f"  Speed: {genomes_per_min_per_thread:.2f} genomes/min/thread")

    all_results.append({
        "set": label,
        "n_genomes": n_valid,
        "wall_clock_s": wall_clock_s,
        "genomes_per_min": genomes_per_min,
        "genomes_per_min_per_thread": genomes_per_min_per_thread,
        "comp_mae": comp_mae,
        "cont_mae": cont_mae,
    })


# Overall summary
print(f"\n\n{'='*60}")
print(f"  OVERALL SUMMARY (5 sets)")
print(f"{'='*60}")

total_genomes = sum(r["n_genomes"] for r in all_results)
total_time = sum(r["wall_clock_s"] for r in all_results)

all_comp_errors = []
all_cont_errors = []

for info in sets_info:
    pred_path = os.path.join(info["path"], "checkm2_predictions.tsv")
    if not os.path.exists(pred_path):
        continue
    df = pd.read_csv(pred_path, sep="\t")
    valid = df.dropna(subset=["pred_completeness", "pred_contamination"])
    comp_errors = np.abs(valid["true_completeness"] - valid["pred_completeness"])
    cont_errors = np.abs(valid["true_contamination"] - valid["pred_contamination"])
    all_comp_errors.extend(comp_errors.tolist())
    all_cont_errors.extend(cont_errors.tolist())

print(f"\n{'Set':<25s} {'N':>6s} {'Wall(s)':>8s} {'G/min':>8s} {'G/min/t':>9s} {'CompMAE':>8s} {'ContMAE':>8s}")
print("-" * 75)
for r in all_results:
    print(f"{r['set']:<25s} {r['n_genomes']:>6d} {r['wall_clock_s']:>8.0f} "
          f"{r['genomes_per_min']:>8.1f} {r['genomes_per_min_per_thread']:>9.2f} "
          f"{r['comp_mae']:>8.2f} {r['cont_mae']:>8.2f}")

overall_speed = total_genomes / (total_time / 60) if total_time > 0 else 0
overall_speed_per_thread = overall_speed / N_THREADS
print("-" * 75)
print(f"{'TOTAL':<25s} {total_genomes:>6d} {total_time:>8.0f} "
      f"{overall_speed:>8.1f} {overall_speed_per_thread:>9.2f} "
      f"{np.mean(all_comp_errors):>8.2f} {np.mean(all_cont_errors):>8.2f}")

print(f"\nTotal wall-clock: {total_time:.0f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hours)")
print(f"Overall completeness MAE: {np.mean(all_comp_errors):.2f}%")
print(f"Overall contamination MAE: {np.mean(all_cont_errors):.2f}%")
print(f"Overall speed: {overall_speed:.1f} genomes/min ({N_THREADS} threads)")
print(f"Overall speed: {overall_speed_per_thread:.2f} genomes/min/thread")
