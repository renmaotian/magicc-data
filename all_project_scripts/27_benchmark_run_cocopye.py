#!/usr/bin/env python3
"""
Phase 6 - Run CoCoPyE on Benchmark Sets A-D

Runs CoCoPyE (v0.5.0) on each benchmark set's FASTA directory.
CoCoPyE processes all genomes in a folder at once via its -i flag.

For each set:
  - Runs `cocopye run -i <fasta_dir> -o <output.csv> -t <threads>`
  - Records wall-clock time and number of threads
  - Parses CoCoPyE CSV output to extract completeness and contamination
  - Merges with benchmark metadata (true labels)
  - Saves results to data/benchmarks/set_{A,B,C,D}/cocopye_predictions.tsv

CoCoPyE output format:
  - Completeness/contamination reported as 0-1 fractions
  - Column '3_completeness' = final ML completeness estimate
  - Column '3_contamination' = final ML contamination estimate
  - We convert to 0-100% for consistency with MAGICC predictions

Usage:
    conda run -n magicc2 python scripts/27_benchmark_run_cocopye.py [--threads N] [--sets A B C D]
"""

import sys
import os
import time
import subprocess
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'

SETS = {
    'A': {'dir': 'set_A', 'n_expected': 600},
    'B': {'dir': 'set_B', 'n_expected': 600},
    'C': {'dir': 'set_C', 'n_expected': 1000},
    'D': {'dir': 'set_D', 'n_expected': 1000},
}

# Default number of threads — CoCoPyE supports -t flag
DEFAULT_THREADS = 48


def run_cocopye_on_set(set_name, set_info, n_threads):
    """Run CoCoPyE on a single benchmark set and return results."""

    set_dir = BENCHMARK_DIR / set_info['dir']
    fasta_dir = set_dir / 'fasta'
    metadata_path = set_dir / 'metadata.tsv'
    cocopye_raw_output = set_dir / 'cocopye_raw_output.csv'
    output_path = set_dir / 'cocopye_predictions.tsv'

    print(f"\n{'='*70}")
    print(f"Running CoCoPyE on Set {set_name}")
    print(f"  FASTA dir: {fasta_dir}")
    print(f"  Expected genomes: {set_info['n_expected']}")
    print(f"  Threads: {n_threads}")
    print(f"{'='*70}")

    # Verify FASTA directory exists
    if not fasta_dir.exists():
        print(f"  ERROR: FASTA directory not found: {fasta_dir}")
        return None

    # Count FASTA files
    fasta_files = list(fasta_dir.glob('*.fasta'))
    n_fasta = len(fasta_files)
    print(f"  Found {n_fasta} FASTA files")

    if n_fasta != set_info['n_expected']:
        print(f"  WARNING: Expected {set_info['n_expected']}, found {n_fasta}")

    # Build CoCoPyE command
    cmd = [
        'conda', 'run', '-n', 'magicc2',
        'cocopye', 'run',
        '-i', str(fasta_dir),
        '-o', str(cocopye_raw_output),
        '-t', str(n_threads),
        '-v', 'full',
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run CoCoPyE and measure wall-clock time
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per set
        )
        wall_clock_s = time.time() - start_time

        if result.returncode != 0:
            print(f"  ERROR: CoCoPyE returned exit code {result.returncode}")
            print(f"  STDERR: {result.stderr[:2000]}")
            print(f"  STDOUT: {result.stdout[:2000]}")
            return None

    except subprocess.TimeoutExpired:
        wall_clock_s = time.time() - start_time
        print(f"  ERROR: CoCoPyE timed out after {wall_clock_s:.1f}s")
        return None

    print(f"  Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Wall-clock time: {wall_clock_s:.2f}s")

    # Parse CoCoPyE output
    if not cocopye_raw_output.exists():
        print(f"  ERROR: CoCoPyE output file not found: {cocopye_raw_output}")
        return None

    cocopye_df = pd.read_csv(cocopye_raw_output)
    print(f"  CoCoPyE output rows: {len(cocopye_df)}")
    print(f"  CoCoPyE output columns: {list(cocopye_df.columns)}")

    if len(cocopye_df) == 0:
        print(f"  ERROR: CoCoPyE output is empty")
        return None

    # Show a sample
    print(f"\n  Sample CoCoPyE output (first 3 rows):")
    print(cocopye_df.head(3).to_string(index=False))

    # Extract predictions
    # CoCoPyE reports: 3_completeness and 3_contamination as 0-1 fractions
    # Convert to 0-100% for consistency
    cocopye_df['genome_id'] = cocopye_df['bin'].astype(str)

    # Use stage 3 (ML) estimates as primary; fall back to stage 2 if stage 3 is NaN
    if '3_completeness' in cocopye_df.columns:
        cocopye_df['pred_completeness'] = cocopye_df['3_completeness'].fillna(
            cocopye_df.get('2_completeness', 0)
        ) * 100.0
        cocopye_df['pred_contamination'] = cocopye_df['3_contamination'].fillna(
            cocopye_df.get('2_contamination', 0)
        ) * 100.0
    elif '2_completeness' in cocopye_df.columns:
        cocopye_df['pred_completeness'] = cocopye_df['2_completeness'] * 100.0
        cocopye_df['pred_contamination'] = cocopye_df['2_contamination'] * 100.0
    else:
        print(f"  ERROR: Cannot find completeness columns in CoCoPyE output")
        return None

    # Load metadata to get true labels
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    metadata_df['genome_id'] = metadata_df['genome_id'].astype(str)

    # Merge predictions with metadata
    merged = metadata_df.merge(
        cocopye_df[['genome_id', 'pred_completeness', 'pred_contamination']],
        on='genome_id',
        how='left',
    )

    # Add timing info
    merged['wall_clock_s'] = wall_clock_s
    merged['n_threads'] = n_threads

    # Check for missing predictions
    n_missing = merged['pred_completeness'].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} genomes have no CoCoPyE predictions")
        # Show which ones are missing
        missing = merged[merged['pred_completeness'].isna()]['genome_id'].tolist()
        print(f"  Missing: {missing[:10]}...")

    # Save results
    merged.to_csv(output_path, sep='\t', index=False)
    print(f"\n  Saved predictions to: {output_path}")

    # Compute summary statistics
    valid = merged.dropna(subset=['pred_completeness', 'pred_contamination'])
    n_valid = len(valid)

    comp_mae = np.abs(valid['true_completeness'] - valid['pred_completeness']).mean()
    cont_mae = np.abs(valid['true_contamination'] - valid['pred_contamination']).mean()

    genomes_per_min = n_valid / (wall_clock_s / 60.0) if wall_clock_s > 0 else 0
    genomes_per_min_per_thread = genomes_per_min / n_threads

    print(f"\n  === Set {set_name} Summary ===")
    print(f"  Genomes processed:       {n_valid} / {len(merged)}")
    print(f"  Wall-clock time:         {wall_clock_s:.2f}s ({wall_clock_s/60:.2f} min)")
    print(f"  Threads:                 {n_threads}")
    print(f"  Genomes/min:             {genomes_per_min:.2f}")
    print(f"  Genomes/min/thread:      {genomes_per_min_per_thread:.2f}")
    print(f"  Completeness MAE:        {comp_mae:.2f}%")
    print(f"  Contamination MAE:       {cont_mae:.2f}%")
    print(f"  Pred completeness range: [{valid['pred_completeness'].min():.2f}, {valid['pred_completeness'].max():.2f}]%")
    print(f"  Pred contamination range:[{valid['pred_contamination'].min():.2f}, {valid['pred_contamination'].max():.2f}]%")

    return {
        'set': set_name,
        'n_genomes': n_valid,
        'n_expected': set_info['n_expected'],
        'wall_clock_s': wall_clock_s,
        'n_threads': n_threads,
        'genomes_per_min': genomes_per_min,
        'genomes_per_min_per_thread': genomes_per_min_per_thread,
        'comp_mae': comp_mae,
        'cont_mae': cont_mae,
        'output_path': str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Run CoCoPyE on benchmark sets')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS,
                        help=f'Number of threads for CoCoPyE (default: {DEFAULT_THREADS})')
    parser.add_argument('--sets', nargs='+', default=['A', 'B', 'C', 'D'],
                        help='Which sets to run (default: A B C D)')
    args = parser.parse_args()

    print("=" * 70)
    print("CoCoPyE Benchmark Runner")
    print("=" * 70)
    print(f"CoCoPyE version: 0.5.0")
    print(f"Threads: {args.threads}")
    print(f"Sets to run: {args.sets}")
    print(f"Benchmark dir: {BENCHMARK_DIR}")

    results = []
    total_start = time.time()

    for set_name in args.sets:
        if set_name not in SETS:
            print(f"\nWARNING: Unknown set '{set_name}', skipping")
            continue

        result = run_cocopye_on_set(set_name, SETS[set_name], args.threads)
        if result is not None:
            results.append(result)

    total_time = time.time() - total_start

    # Print overall summary
    print(f"\n\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total wall-clock time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Threads: {args.threads}")
    print()

    if results:
        print(f"{'Set':>4} | {'N':>6} | {'Time(s)':>8} | {'G/min':>8} | {'G/min/thr':>10} | {'Comp MAE':>9} | {'Cont MAE':>9}")
        print(f"{'-'*4:>4}-+-{'-'*6:>6}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*10:>10}-+-{'-'*9:>9}-+-{'-'*9:>9}")
        for r in results:
            print(f"{r['set']:>4} | {r['n_genomes']:>6} | {r['wall_clock_s']:>8.1f} | {r['genomes_per_min']:>8.1f} | {r['genomes_per_min_per_thread']:>10.2f} | {r['comp_mae']:>8.2f}% | {r['cont_mae']:>8.2f}%")

        total_genomes = sum(r['n_genomes'] for r in results)
        total_set_time = sum(r['wall_clock_s'] for r in results)
        overall_rate = total_genomes / (total_set_time / 60) if total_set_time > 0 else 0
        overall_rate_per_thread = overall_rate / args.threads

        print(f"\nTotal genomes processed: {total_genomes}")
        print(f"Total processing time:   {total_set_time:.1f}s ({total_set_time/60:.1f} min)")
        print(f"Overall rate:            {overall_rate:.1f} genomes/min")
        print(f"Overall rate/thread:     {overall_rate_per_thread:.2f} genomes/min/thread")
    else:
        print("No sets completed successfully.")

    print(f"\nOutput files:")
    for r in results:
        print(f"  {r['output_path']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
