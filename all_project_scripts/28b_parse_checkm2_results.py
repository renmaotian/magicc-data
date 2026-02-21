#!/usr/bin/env python3
"""
Script 28b: Parse CheckM2 results and merge with benchmark metadata.

Creates checkm2_predictions.tsv for each set with columns:
genome_id, true_completeness, true_contamination, dominant_accession,
dominant_phylum, sample_type, n_contigs, total_length,
pred_completeness, pred_contamination, wall_clock_s, n_threads
"""

import pandas as pd
import os
import sys
import numpy as np

PROJECT_DIR = "/home/tianrm/projects/magicc2"
BENCHMARK_DIR = os.path.join(PROJECT_DIR, "data", "benchmarks")
N_THREADS = 32

sets_info = {
    "A": {"n_genomes": 600},
    "B": {"n_genomes": 600},
    "C": {"n_genomes": 1000},
    "D": {"n_genomes": 1000},
}

for set_name, info in sets_info.items():
    set_dir = os.path.join(BENCHMARK_DIR, f"set_{set_name}")
    metadata_path = os.path.join(set_dir, "metadata.tsv")
    checkm2_output_dir = os.path.join(set_dir, "checkm2_output")
    quality_report_path = os.path.join(checkm2_output_dir, "quality_report.tsv")
    wallclock_path = os.path.join(set_dir, "checkm2_wallclock.txt")
    output_path = os.path.join(set_dir, "checkm2_predictions.tsv")

    print(f"\n=== Set {set_name} ===")

    # Read metadata
    metadata = pd.read_csv(metadata_path, sep="\t")
    print(f"  Metadata: {len(metadata)} genomes")

    # Read CheckM2 quality report
    if not os.path.exists(quality_report_path):
        print(f"  WARNING: quality_report.tsv not found, skipping")
        continue

    checkm2 = pd.read_csv(quality_report_path, sep="\t")
    print(f"  CheckM2 results: {len(checkm2)} genomes")
    print(f"  CheckM2 columns: {list(checkm2.columns)}")

    # Read wall-clock time
    if os.path.exists(wallclock_path):
        with open(wallclock_path) as f:
            wall_clock_s = float(f.read().strip())
        print(f"  Wall-clock time: {wall_clock_s:.0f}s ({wall_clock_s/60:.1f} min)")
    else:
        wall_clock_s = 0.0
        print(f"  WARNING: wallclock file not found")

    # CheckM2 uses 'Name' column (without .fasta extension)
    # Our metadata uses 'genome_id' column
    # Merge on genome_id = Name
    checkm2_slim = checkm2[["Name", "Completeness", "Contamination"]].copy()
    checkm2_slim.rename(columns={
        "Name": "genome_id",
        "Completeness": "pred_completeness",
        "Contamination": "pred_contamination"
    }, inplace=True)

    # Merge
    merged = metadata.merge(checkm2_slim, on="genome_id", how="left")

    # Add timing info
    merged["wall_clock_s"] = wall_clock_s
    merged["n_threads"] = N_THREADS

    # Check for missing predictions
    n_missing = merged["pred_completeness"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} genomes missing CheckM2 predictions")

    # Save
    merged.to_csv(output_path, sep="\t", index=False)
    print(f"  Saved: {output_path}")
    print(f"  Columns: {list(merged.columns)}")

    # Quick accuracy summary
    valid = merged.dropna(subset=["pred_completeness", "pred_contamination"])
    comp_mae = np.mean(np.abs(valid["true_completeness"] - valid["pred_completeness"]))
    cont_mae = np.mean(np.abs(valid["true_contamination"] - valid["pred_contamination"]))
    genomes_per_min = len(valid) / (wall_clock_s / 60) if wall_clock_s > 0 else 0
    genomes_per_min_per_thread = genomes_per_min / N_THREADS

    print(f"\n  --- Set {set_name} Summary ---")
    print(f"  Genomes: {len(valid)}")
    print(f"  Wall-clock: {wall_clock_s:.0f}s ({wall_clock_s/60:.1f} min)")
    print(f"  Completeness MAE: {comp_mae:.2f}%")
    print(f"  Contamination MAE: {cont_mae:.2f}%")
    print(f"  Speed: {genomes_per_min:.1f} genomes/min (32 threads)")
    print(f"  Speed: {genomes_per_min_per_thread:.2f} genomes/min/thread")

print("\n\n=== OVERALL SUMMARY ===")
total_genomes = 0
total_time = 0
all_comp_errors = []
all_cont_errors = []

for set_name in ["A", "B", "C", "D"]:
    set_dir = os.path.join(BENCHMARK_DIR, f"set_{set_name}")
    pred_path = os.path.join(set_dir, "checkm2_predictions.tsv")
    if not os.path.exists(pred_path):
        continue
    df = pd.read_csv(pred_path, sep="\t")
    valid = df.dropna(subset=["pred_completeness", "pred_contamination"])
    wall_s = valid["wall_clock_s"].iloc[0] if len(valid) > 0 else 0

    comp_errors = np.abs(valid["true_completeness"] - valid["pred_completeness"])
    cont_errors = np.abs(valid["true_contamination"] - valid["pred_contamination"])

    total_genomes += len(valid)
    total_time += wall_s
    all_comp_errors.extend(comp_errors.tolist())
    all_cont_errors.extend(cont_errors.tolist())

    genomes_per_min = len(valid) / (wall_s / 60) if wall_s > 0 else 0
    print(f"  Set {set_name}: {len(valid)} genomes, {wall_s:.0f}s, "
          f"comp MAE={np.mean(comp_errors):.2f}%, "
          f"cont MAE={np.mean(cont_errors):.2f}%, "
          f"{genomes_per_min:.1f} genomes/min")

overall_speed = total_genomes / (total_time / 60) if total_time > 0 else 0
overall_speed_per_thread = overall_speed / N_THREADS

print(f"\n  Total genomes: {total_genomes}")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Overall comp MAE: {np.mean(all_comp_errors):.2f}%")
print(f"  Overall cont MAE: {np.mean(all_cont_errors):.2f}%")
print(f"  Overall speed: {overall_speed:.1f} genomes/min (32 threads)")
print(f"  Overall speed: {overall_speed_per_thread:.2f} genomes/min/thread")
