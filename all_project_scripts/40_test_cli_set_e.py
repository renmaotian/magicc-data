#!/usr/bin/env python3
"""
Phase 7 - Test MAGICC CLI End-to-End on Benchmark Set E

Runs the MAGICC CLI on the 1,000 genomes in data/benchmarks/set_E/fasta/
using raw genome FASTA files, CPU-only (no GPU).

Tests:
1. Single-thread speed (--threads 1)
2. Multi-thread throughput (--threads 43)
3. Accuracy comparison against ground truth
4. Verification against previous MAGICC Set E results

Output: results/phase7_set_e_test/
  - predictions_1thread.tsv     -- CLI output (1 thread)
  - predictions_multithread.tsv -- CLI output (multi-thread)
  - accuracy_report.tsv         -- accuracy metrics
  - speed_report.tsv            -- timing results
  - summary.json                -- full machine-readable summary
"""

import sys
import os
import time
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
SET_E_DIR = PROJECT_DIR / 'data' / 'benchmarks' / 'set_E'
FASTA_DIR = SET_E_DIR / 'fasta'
METADATA_PATH = SET_E_DIR / 'metadata.tsv'
OUTPUT_DIR = PROJECT_DIR / 'results' / 'phase7_set_e_test'

# Thread configurations to test
SINGLE_THREAD = 1
MULTI_THREAD = 43  # 90% of 48 cores

# Previous MAGICC results on Set E (updated after new composition: 200 pure + 200 complete + 600 other)
EXPECTED_COMP_MAE = 3.92
EXPECTED_CONT_MAE = 4.32
EXPECTED_COMP_RMSE = 6.92
EXPECTED_CONT_RMSE = 7.47

# Tolerance for matching previous results (percentage points)
TOLERANCE = 0.05  # Allow tiny floating-point differences


def run_cli(threads: int, output_tsv: str, label: str) -> dict:
    """
    Run the MAGICC CLI and measure wall-clock time.

    Returns dict with timing info.
    """
    print(f"\n{'='*60}")
    print(f"Running MAGICC CLI: {label}")
    print(f"  Threads: {threads}")
    print(f"  Output:  {output_tsv}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, '-m', 'magicc', 'predict',
        '--input', str(FASTA_DIR),
        '--output', output_tsv,
        '--threads', str(threads),
        '--batch-size', '64',
        '--extension', '.fasta',
    ]

    t_start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
    )
    t_end = time.time()
    wall_clock = t_end - t_start

    # Print CLI output
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            print(f"  {line}")

    if result.returncode != 0:
        print(f"\nERROR: CLI exited with code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)

    # Verify output exists
    if not os.path.isfile(output_tsv):
        print(f"\nERROR: Output file not created: {output_tsv}")
        sys.exit(1)

    # Count predictions
    pred_df = pd.read_csv(output_tsv, sep='\t')
    n_predictions = len(pred_df)

    speed = n_predictions / wall_clock * 60 if wall_clock > 0 else 0
    speed_per_thread = speed / max(threads, 1)

    print(f"\n  Wall-clock:     {wall_clock:.1f}s")
    print(f"  Predictions:    {n_predictions}")
    print(f"  Speed:          {speed:.0f} genomes/min")
    print(f"  Speed/thread:   {speed_per_thread:.0f} genomes/min/thread")

    return {
        'label': label,
        'threads': threads,
        'wall_clock_s': wall_clock,
        'n_predictions': n_predictions,
        'genomes_per_min': speed,
        'genomes_per_min_per_thread': speed_per_thread,
        'ms_per_genome': wall_clock / max(n_predictions, 1) * 1000,
    }


def compute_accuracy(pred_path: str, metadata_path: str) -> dict:
    """
    Compare predictions against ground truth and compute accuracy metrics.

    Returns dict with per-metric accuracy values.
    """
    pred_df = pd.read_csv(pred_path, sep='\t')
    meta_df = pd.read_csv(metadata_path, sep='\t')

    # Merge on genome name/id
    pred_df = pred_df.rename(columns={'genome_name': 'genome_id'})
    merged = meta_df.merge(pred_df, on='genome_id', how='inner')

    n = len(merged)
    if n == 0:
        raise RuntimeError("No matching genomes between predictions and metadata")

    true_comp = merged['true_completeness'].values
    true_cont = merged['true_contamination'].values
    pred_comp = merged['pred_completeness'].values
    pred_cont = merged['pred_contamination'].values

    # MAE
    mae_comp = float(np.mean(np.abs(true_comp - pred_comp)))
    mae_cont = float(np.mean(np.abs(true_cont - pred_cont)))

    # RMSE
    rmse_comp = float(np.sqrt(np.mean((true_comp - pred_comp) ** 2)))
    rmse_cont = float(np.sqrt(np.mean((true_cont - pred_cont) ** 2)))

    # R-squared (Pearson r^2)
    if np.std(true_comp) > 0 and np.std(pred_comp) > 0:
        r2_comp = float(np.corrcoef(true_comp, pred_comp)[0, 1] ** 2)
    else:
        r2_comp = float('nan')
    if np.std(true_cont) > 0 and np.std(pred_cont) > 0:
        r2_cont = float(np.corrcoef(true_cont, pred_cont)[0, 1] ** 2)
    else:
        r2_cont = float('nan')

    # Outlier rate (predictions with > 20% absolute error)
    outlier_comp = float(np.mean(np.abs(true_comp - pred_comp) > 20) * 100)
    outlier_cont = float(np.mean(np.abs(true_cont - pred_cont) > 20) * 100)

    # Median absolute error
    medae_comp = float(np.median(np.abs(true_comp - pred_comp)))
    medae_cont = float(np.median(np.abs(true_cont - pred_cont)))

    return {
        'n_genomes': n,
        'completeness_mae': mae_comp,
        'contamination_mae': mae_cont,
        'completeness_rmse': rmse_comp,
        'contamination_rmse': rmse_cont,
        'completeness_r2': r2_comp,
        'contamination_r2': r2_cont,
        'completeness_medae': medae_comp,
        'contamination_medae': medae_cont,
        'completeness_outlier_pct': outlier_comp,
        'contamination_outlier_pct': outlier_cont,
    }


def verify_against_previous(accuracy: dict) -> dict:
    """
    Verify accuracy matches previous Set E MAGICC results.

    Returns dict with pass/fail status and differences.
    """
    checks = {}

    diff_comp_mae = abs(accuracy['completeness_mae'] - EXPECTED_COMP_MAE)
    diff_cont_mae = abs(accuracy['contamination_mae'] - EXPECTED_CONT_MAE)
    diff_comp_rmse = abs(accuracy['completeness_rmse'] - EXPECTED_COMP_RMSE)
    diff_cont_rmse = abs(accuracy['contamination_rmse'] - EXPECTED_CONT_RMSE)

    checks['comp_mae'] = {
        'expected': EXPECTED_COMP_MAE,
        'actual': accuracy['completeness_mae'],
        'diff': diff_comp_mae,
        'pass': diff_comp_mae <= TOLERANCE,
    }
    checks['cont_mae'] = {
        'expected': EXPECTED_CONT_MAE,
        'actual': accuracy['contamination_mae'],
        'diff': diff_cont_mae,
        'pass': diff_cont_mae <= TOLERANCE,
    }
    checks['comp_rmse'] = {
        'expected': EXPECTED_COMP_RMSE,
        'actual': accuracy['completeness_rmse'],
        'diff': diff_comp_rmse,
        'pass': diff_comp_rmse <= TOLERANCE,
    }
    checks['cont_rmse'] = {
        'expected': EXPECTED_CONT_RMSE,
        'actual': accuracy['contamination_rmse'],
        'diff': diff_cont_rmse,
        'pass': diff_cont_rmse <= TOLERANCE,
    }

    all_pass = all(c['pass'] for c in checks.values())
    checks['overall_pass'] = all_pass

    return checks


def main():
    t0 = time.time()
    print("MAGICC Phase 7: CLI End-to-End Test on Set E")
    print(f"Set E: {FASTA_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify input data
    fasta_files = list(FASTA_DIR.glob('*.fasta'))
    print(f"\nInput: {len(fasta_files)} FASTA files")
    assert len(fasta_files) == 1000, f"Expected 1000 files, got {len(fasta_files)}"
    assert METADATA_PATH.exists(), f"Metadata not found: {METADATA_PATH}"

    # ====================================================================
    # Test 1: Single-thread run
    # ====================================================================
    speed_1t = run_cli(
        threads=SINGLE_THREAD,
        output_tsv=str(OUTPUT_DIR / 'predictions_1thread.tsv'),
        label='Single-thread (1 CPU)',
    )

    # ====================================================================
    # Test 2: Multi-thread run
    # ====================================================================
    speed_mt = run_cli(
        threads=MULTI_THREAD,
        output_tsv=str(OUTPUT_DIR / 'predictions_multithread.tsv'),
        label=f'Multi-thread ({MULTI_THREAD} CPUs)',
    )

    # ====================================================================
    # Accuracy analysis (using single-thread predictions -- should be identical)
    # ====================================================================
    print(f"\n{'='*60}")
    print("Accuracy Analysis")
    print(f"{'='*60}")

    accuracy = compute_accuracy(
        str(OUTPUT_DIR / 'predictions_1thread.tsv'),
        str(METADATA_PATH),
    )

    print(f"\n  Genomes:               {accuracy['n_genomes']}")
    print(f"  Completeness MAE:      {accuracy['completeness_mae']:.2f}%")
    print(f"  Contamination MAE:     {accuracy['contamination_mae']:.2f}%")
    print(f"  Completeness RMSE:     {accuracy['completeness_rmse']:.2f}%")
    print(f"  Contamination RMSE:    {accuracy['contamination_rmse']:.2f}%")
    print(f"  Completeness R2:       {accuracy['completeness_r2']:.4f}")
    print(f"  Contamination R2:      {accuracy['contamination_r2']:.4f}")
    print(f"  Completeness MedAE:    {accuracy['completeness_medae']:.2f}%")
    print(f"  Contamination MedAE:   {accuracy['contamination_medae']:.2f}%")
    print(f"  Completeness outliers: {accuracy['completeness_outlier_pct']:.1f}%")
    print(f"  Contamination outliers:{accuracy['contamination_outlier_pct']:.1f}%")

    # ====================================================================
    # Verify multi-thread predictions match single-thread
    # ====================================================================
    print(f"\n{'='*60}")
    print("Multi-Thread Consistency Check")
    print(f"{'='*60}")

    pred_1t = pd.read_csv(OUTPUT_DIR / 'predictions_1thread.tsv', sep='\t')
    pred_mt = pd.read_csv(OUTPUT_DIR / 'predictions_multithread.tsv', sep='\t')

    pred_1t = pred_1t.sort_values('genome_name').reset_index(drop=True)
    pred_mt = pred_mt.sort_values('genome_name').reset_index(drop=True)

    comp_diff = np.abs(pred_1t['pred_completeness'].values - pred_mt['pred_completeness'].values)
    cont_diff = np.abs(pred_1t['pred_contamination'].values - pred_mt['pred_contamination'].values)

    print(f"  Max completeness diff:  {comp_diff.max():.6f}")
    print(f"  Max contamination diff: {cont_diff.max():.6f}")
    consistent = comp_diff.max() < 0.01 and cont_diff.max() < 0.01
    print(f"  Consistent (within 0.01): {'PASS' if consistent else 'FAIL'}")

    # ====================================================================
    # Verify against previous results
    # ====================================================================
    print(f"\n{'='*60}")
    print("Verification Against Previous Set E Results")
    print(f"{'='*60}")

    checks = verify_against_previous(accuracy)

    print(f"\n  {'Metric':<15} {'Expected':>10} {'Actual':>10} {'Diff':>10} {'Status':>8}")
    print(f"  {'-'*55}")
    for metric in ['comp_mae', 'cont_mae', 'comp_rmse', 'cont_rmse']:
        c = checks[metric]
        status = 'PASS' if c['pass'] else 'FAIL'
        print(f"  {metric:<15} {c['expected']:>9.2f}% {c['actual']:>9.2f}% {c['diff']:>9.4f} {status:>8}")

    overall = 'PASS' if checks['overall_pass'] else 'FAIL'
    print(f"\n  Overall verification: {overall}")

    # ====================================================================
    # Write output files
    # ====================================================================
    print(f"\n{'='*60}")
    print("Writing Output Files")
    print(f"{'='*60}")

    # 1. Accuracy report
    accuracy_df = pd.DataFrame([
        {'metric': 'completeness_mae', 'value': accuracy['completeness_mae'], 'unit': '%'},
        {'metric': 'contamination_mae', 'value': accuracy['contamination_mae'], 'unit': '%'},
        {'metric': 'completeness_rmse', 'value': accuracy['completeness_rmse'], 'unit': '%'},
        {'metric': 'contamination_rmse', 'value': accuracy['contamination_rmse'], 'unit': '%'},
        {'metric': 'completeness_r2', 'value': accuracy['completeness_r2'], 'unit': ''},
        {'metric': 'contamination_r2', 'value': accuracy['contamination_r2'], 'unit': ''},
        {'metric': 'completeness_medae', 'value': accuracy['completeness_medae'], 'unit': '%'},
        {'metric': 'contamination_medae', 'value': accuracy['contamination_medae'], 'unit': '%'},
        {'metric': 'completeness_outlier_pct', 'value': accuracy['completeness_outlier_pct'], 'unit': '%'},
        {'metric': 'contamination_outlier_pct', 'value': accuracy['contamination_outlier_pct'], 'unit': '%'},
        {'metric': 'n_genomes', 'value': accuracy['n_genomes'], 'unit': ''},
    ])
    accuracy_path = OUTPUT_DIR / 'accuracy_report.tsv'
    accuracy_df.to_csv(accuracy_path, sep='\t', index=False)
    print(f"  {accuracy_path}")

    # 2. Speed report
    speed_df = pd.DataFrame([
        {
            'test': speed_1t['label'],
            'threads': speed_1t['threads'],
            'wall_clock_s': round(speed_1t['wall_clock_s'], 2),
            'genomes_per_min': round(speed_1t['genomes_per_min'], 1),
            'genomes_per_min_per_thread': round(speed_1t['genomes_per_min_per_thread'], 1),
            'ms_per_genome': round(speed_1t['ms_per_genome'], 1),
        },
        {
            'test': speed_mt['label'],
            'threads': speed_mt['threads'],
            'wall_clock_s': round(speed_mt['wall_clock_s'], 2),
            'genomes_per_min': round(speed_mt['genomes_per_min'], 1),
            'genomes_per_min_per_thread': round(speed_mt['genomes_per_min_per_thread'], 1),
            'ms_per_genome': round(speed_mt['ms_per_genome'], 1),
        },
    ])
    speed_path = OUTPUT_DIR / 'speed_report.tsv'
    speed_df.to_csv(speed_path, sep='\t', index=False)
    print(f"  {speed_path}")

    # 3. Copy predictions as canonical predictions.tsv
    pred_1t_canonical = OUTPUT_DIR / 'predictions.tsv'
    pred_1t.to_csv(pred_1t_canonical, sep='\t', index=False)
    print(f"  {pred_1t_canonical}")

    # 4. Full summary JSON
    summary = {
        'phase': 'Phase 7: CLI End-to-End Test',
        'dataset': 'Set E (1,000 genomes)',
        'model': 'magicc_v3.onnx',
        'accuracy': accuracy,
        'speed_single_thread': speed_1t,
        'speed_multi_thread': speed_mt,
        'verification': {
            k: v for k, v in checks.items() if k != 'overall_pass'
        },
        'verification_overall': checks['overall_pass'],
        'multithread_consistency': bool(consistent),
        'total_test_time_s': time.time() - t0,
    }
    summary_path = OUTPUT_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  {summary_path}")

    # ====================================================================
    # Final summary
    # ====================================================================
    total_test_time = time.time() - t0
    print(f"\n{'='*60}")
    print("PHASE 7 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset:         Set E ({accuracy['n_genomes']} genomes)")
    print(f"  Model:           magicc_v3.onnx (V3, SE attention + cross-attention)")
    print(f"")
    print(f"  Accuracy:")
    print(f"    Comp MAE:      {accuracy['completeness_mae']:.2f}%")
    print(f"    Cont MAE:      {accuracy['contamination_mae']:.2f}%")
    print(f"    Comp RMSE:     {accuracy['completeness_rmse']:.2f}%")
    print(f"    Cont RMSE:     {accuracy['contamination_rmse']:.2f}%")
    print(f"    Comp R2:       {accuracy['completeness_r2']:.4f}")
    print(f"    Cont R2:       {accuracy['contamination_r2']:.4f}")
    print(f"")
    print(f"  Single-thread speed (1 CPU):")
    print(f"    Wall-clock:    {speed_1t['wall_clock_s']:.1f}s")
    print(f"    Speed:         {speed_1t['genomes_per_min']:.0f} genomes/min")
    print(f"    Per-genome:    {speed_1t['ms_per_genome']:.1f} ms")
    print(f"")
    print(f"  Multi-thread speed ({MULTI_THREAD} CPUs):")
    print(f"    Wall-clock:    {speed_mt['wall_clock_s']:.1f}s")
    print(f"    Speed:         {speed_mt['genomes_per_min']:.0f} genomes/min")
    print(f"    Per-genome:    {speed_mt['ms_per_genome']:.1f} ms")
    print(f"    Speedup:       {speed_1t['wall_clock_s']/max(speed_mt['wall_clock_s'],0.1):.1f}x")
    print(f"    Efficiency:    {speed_mt['genomes_per_min_per_thread']/max(speed_1t['genomes_per_min_per_thread'],0.1)*100:.0f}%")
    print(f"")
    print(f"  Verification vs previous results: {overall}")
    print(f"  Multi-thread consistency:          {'PASS' if consistent else 'FAIL'}")
    print(f"  Total test time:                   {total_test_time:.0f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
