#!/usr/bin/env python3
"""
Run CoCoPyE on motivating benchmark sets A and B.
CoCoPyE v0.5.0, 48 threads, magicc2 conda env.
"""

import sys
import os
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
BENCHMARK_DIR = PROJECT_DIR / 'data' / 'benchmarks' / 'motivating'
N_THREADS = 48

SETS = {
    'A': {'dir': 'set_A', 'n_expected': 600},
    'B': {'dir': 'set_B', 'n_expected': 1100},
}


def run_cocopye_on_set(set_name, set_info):
    set_dir = BENCHMARK_DIR / set_info['dir']
    fasta_dir = set_dir / 'fasta'
    metadata_path = set_dir / 'metadata.tsv'
    cocopye_raw_output = set_dir / 'cocopye_raw_output.csv'
    output_path = set_dir / 'cocopye_predictions.tsv'

    print(f"\n{'='*70}")
    print(f"Running CoCoPyE on Motivating Set {set_name}")
    print(f"  FASTA dir: {fasta_dir}")
    print(f"  Threads: {N_THREADS}")
    print(f"{'='*70}")

    # Check if already done
    if output_path.exists():
        existing = pd.read_csv(output_path, sep='\t')
        metadata = pd.read_csv(metadata_path, sep='\t')
        if len(existing) >= len(metadata):
            print(f"  SKIPPING: Already completed ({len(existing)} results)")
            # Still return stats
            valid = existing.dropna(subset=['pred_completeness', 'pred_contamination'])
            wall_s = valid['wall_clock_s'].iloc[0] if 'wall_clock_s' in valid.columns else 0
            comp_mae = np.mean(np.abs(valid['true_completeness'] - valid['pred_completeness']))
            cont_mae = np.mean(np.abs(valid['true_contamination'] - valid['pred_contamination']))
            return {'set': set_name, 'n_genomes': len(valid), 'wall_clock_s': wall_s,
                    'comp_mae': comp_mae, 'cont_mae': cont_mae}

    fasta_files = list(fasta_dir.glob('*.fasta'))
    print(f"  Found {len(fasta_files)} FASTA files")

    cmd = [
        'conda', 'run', '-n', 'magicc2',
        'cocopye', 'run',
        '-i', str(fasta_dir),
        '-o', str(cocopye_raw_output),
        '-t', str(N_THREADS),
        '-v', 'full',
    ]

    print(f"  Command: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        wall_clock_s = time.time() - start_time

        if result.returncode != 0:
            print(f"  ERROR: CoCoPyE returned exit code {result.returncode}")
            print(f"  STDERR: {result.stderr[:2000]}")
            return None
    except subprocess.TimeoutExpired:
        wall_clock_s = time.time() - start_time
        print(f"  ERROR: Timed out after {wall_clock_s:.1f}s")
        return None

    print(f"  Wall-clock time: {wall_clock_s:.2f}s ({wall_clock_s/60:.1f} min)")

    # Parse CoCoPyE output
    cocopye_df = pd.read_csv(cocopye_raw_output)
    print(f"  CoCoPyE output rows: {len(cocopye_df)}")

    cocopye_df['genome_id'] = cocopye_df['bin'].astype(str)

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
        print(f"  ERROR: Cannot find completeness columns")
        return None

    metadata_df = pd.read_csv(metadata_path, sep='\t')
    metadata_df['genome_id'] = metadata_df['genome_id'].astype(str)

    merged = metadata_df.merge(
        cocopye_df[['genome_id', 'pred_completeness', 'pred_contamination']],
        on='genome_id', how='left',
    )
    merged['wall_clock_s'] = wall_clock_s
    merged['n_threads'] = N_THREADS

    n_missing = merged['pred_completeness'].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} genomes missing predictions")

    merged.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved: {output_path}")

    valid = merged.dropna(subset=['pred_completeness', 'pred_contamination'])
    comp_mae = np.mean(np.abs(valid['true_completeness'] - valid['pred_completeness']))
    cont_mae = np.mean(np.abs(valid['true_contamination'] - valid['pred_contamination']))
    genomes_per_min = len(valid) / (wall_clock_s / 60) if wall_clock_s > 0 else 0

    print(f"\n  === Set {set_name} Summary ===")
    print(f"  Genomes: {len(valid)}")
    print(f"  Completeness MAE: {comp_mae:.2f}%")
    print(f"  Contamination MAE: {cont_mae:.2f}%")
    print(f"  Speed: {genomes_per_min:.1f} genomes/min ({genomes_per_min/N_THREADS:.2f} genomes/min/thread)")

    return {'set': set_name, 'n_genomes': len(valid), 'wall_clock_s': wall_clock_s,
            'comp_mae': comp_mae, 'cont_mae': cont_mae}


def main():
    print("CoCoPyE Motivating Benchmark Runner")
    print(f"Threads: {N_THREADS}")

    total_start = time.time()
    results = []

    for set_name in ['A', 'B']:
        r = run_cocopye_on_set(set_name, SETS[set_name])
        if r:
            results.append(r)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    for r in results:
        print(f"  Set {r['set']}: {r['n_genomes']} genomes, {r['wall_clock_s']:.0f}s, "
              f"comp MAE={r['comp_mae']:.2f}%, cont MAE={r['cont_mae']:.2f}%")


if __name__ == '__main__':
    main()
