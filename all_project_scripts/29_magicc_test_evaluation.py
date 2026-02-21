#!/usr/bin/env python3
"""
Phase 6.1 - MAGICC V3 ONNX Inference on 100K Test Set

Loads pre-normalized test features from HDF5, runs ONNX inference,
computes comprehensive accuracy metrics, and saves per-sample predictions.

Output: data/benchmarks/test_100k/magicc_predictions.tsv
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/home/tianrm/projects/magicc2')

import h5py
import onnxruntime as ort

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'

HDF5_PATH = str(DATA_DIR / 'features' / 'magicc_features.h5')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v3.onnx')
OUTPUT_DIR = DATA_DIR / 'benchmarks' / 'test_100k'

N_THREADS = 1
BATCH_SIZE = 2048  # Larger batch for efficiency on 100K samples


def setup_onnx_session():
    """Create ONNX Runtime session with 1 thread."""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = N_THREADS
    sess_options.inter_op_num_threads = N_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    return session


def compute_metrics(true_vals, pred_vals):
    """Compute MAE, RMSE, R2 (Pearson), outlier rate."""
    errors = np.abs(true_vals - pred_vals)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))

    if np.std(true_vals) > 0 and np.std(pred_vals) > 0:
        r2 = np.corrcoef(true_vals, pred_vals)[0, 1] ** 2
    else:
        r2 = float('nan')

    outlier_rate = np.mean(errors > 20.0) * 100  # % with >20% absolute error

    return mae, rmse, r2, outlier_rate


def main():
    t0 = time.time()
    print("=" * 70)
    print("MAGICC V3 - 100K Test Set Evaluation")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ONNX model
    print("\nLoading ONNX model...")
    session = setup_onnx_session()
    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name
    print(f"  Input names: {input_names}")
    print(f"  Output: {output_name}")

    # Load test data from HDF5
    print("\nLoading test data from HDF5...")
    t_load_start = time.time()
    with h5py.File(HDF5_PATH, 'r') as f:
        test_grp = f['test']
        kmer_features = test_grp['kmer_features'][:].astype(np.float32)
        assembly_features = test_grp['assembly_features'][:].astype(np.float32)
        labels = test_grp['labels'][:]
        metadata = test_grp['metadata'][:]
    t_load_end = time.time()

    n_samples = kmer_features.shape[0]
    print(f"  Loaded {n_samples} test samples in {t_load_end - t_load_start:.1f}s")
    print(f"  K-mer features shape: {kmer_features.shape}")
    print(f"  Assembly features shape: {assembly_features.shape}")
    print(f"  Labels shape: {labels.shape}")

    true_comp = labels[:, 0]
    true_cont = labels[:, 1]

    # Decode metadata
    sample_types = np.array([s.decode() for s in metadata['sample_type']])
    phyla = np.array([s.decode() for s in metadata['dominant_phylum']])
    accessions = np.array([s.decode() for s in metadata['dominant_accession']])
    quality_tiers = np.array([s.decode() for s in metadata['quality_tier']])

    # NOTE: Features in HDF5 are already normalized (normalization was applied during batch synthesis)
    # No additional normalization needed

    # Run ONNX inference
    print(f"\nRunning ONNX inference ({n_samples} samples, batch_size={BATCH_SIZE}, 1 thread)...")
    t_infer_start = time.time()

    predictions = np.zeros((n_samples, 2), dtype=np.float32)
    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_samples)
        batch_kmer = kmer_features[batch_start:batch_end]
        batch_assembly = assembly_features[batch_start:batch_end]

        feed = {
            input_names[0]: batch_kmer,
            input_names[1]: batch_assembly,
        }
        result = session.run([output_name], feed)
        predictions[batch_start:batch_end] = result[0]

        if (batch_start // BATCH_SIZE) % 10 == 0:
            pct = batch_end / n_samples * 100
            print(f"  Progress: {batch_end}/{n_samples} ({pct:.1f}%)")

    t_infer_end = time.time()
    infer_time = t_infer_end - t_infer_start
    print(f"  Inference completed: {infer_time:.2f}s ({infer_time / n_samples * 1000:.3f} ms/sample)")
    print(f"  Speed: {n_samples / infer_time * 60:.0f} genomes/min (inference only, 1 thread)")

    pred_comp = predictions[:, 0]
    pred_cont = predictions[:, 1]

    # ========================================================================
    # Save per-sample predictions as TSV
    # ========================================================================
    print("\nSaving per-sample predictions...")
    df = pd.DataFrame({
        'genome_id': [f'test_{i}' for i in range(n_samples)],
        'true_completeness': true_comp,
        'true_contamination': true_cont,
        'pred_completeness': pred_comp,
        'pred_contamination': pred_cont,
        'dominant_accession': accessions,
        'dominant_phylum': phyla,
        'sample_type': sample_types,
        'quality_tier': quality_tiers,
        'n_contaminants': metadata['n_contaminants'],
        'genome_full_length': metadata['genome_full_length'],
    })
    output_path = OUTPUT_DIR / 'magicc_predictions.tsv'
    df.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved: {output_path} ({len(df)} rows)")

    # ========================================================================
    # Overall Metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("OVERALL METRICS (100,000 test samples)")
    print("=" * 70)

    mae_comp, rmse_comp, r2_comp, outlier_comp = compute_metrics(true_comp, pred_comp)
    mae_cont, rmse_cont, r2_cont, outlier_cont = compute_metrics(true_cont, pred_cont)

    print(f"  Completeness:  MAE={mae_comp:.3f}%, RMSE={rmse_comp:.3f}%, R2={r2_comp:.4f}, Outlier(>20%)={outlier_comp:.2f}%")
    print(f"  Contamination: MAE={mae_cont:.3f}%, RMSE={rmse_cont:.3f}%, R2={r2_cont:.4f}, Outlier(>20%)={outlier_cont:.2f}%")

    # ========================================================================
    # Per-Sample-Type Breakdown
    # ========================================================================
    print("\n" + "=" * 70)
    print("PER-SAMPLE-TYPE BREAKDOWN")
    print("=" * 70)
    print(f"{'Sample Type':<20} {'N':>7} {'Comp MAE':>10} {'Cont MAE':>10} {'Comp R2':>9} {'Cont R2':>9} {'Comp Out%':>10} {'Cont Out%':>10}")
    print("-" * 95)

    type_results = {}
    for st in sorted(np.unique(sample_types)):
        mask = sample_types == st
        n = np.sum(mask)
        m_comp, r_comp, r2_c, o_comp = compute_metrics(true_comp[mask], pred_comp[mask])
        m_cont, r_cont, r2_t, o_cont = compute_metrics(true_cont[mask], pred_cont[mask])
        print(f"  {st:<18} {n:>7} {m_comp:>9.3f}% {m_cont:>9.3f}% {r2_c:>9.4f} {r2_t:>9.4f} {o_comp:>9.2f}% {o_cont:>9.2f}%")
        type_results[st] = {
            'n': int(n), 'comp_mae': float(m_comp), 'cont_mae': float(m_cont),
            'comp_r2': float(r2_c), 'cont_r2': float(r2_t),
            'comp_rmse': float(r_comp), 'cont_rmse': float(r_cont),
            'comp_outlier': float(o_comp), 'cont_outlier': float(o_cont),
        }

    # ========================================================================
    # Per-Quality-Tier Breakdown
    # ========================================================================
    print("\n" + "=" * 70)
    print("PER-QUALITY-TIER BREAKDOWN")
    print("=" * 70)
    print(f"{'Quality Tier':<22} {'N':>7} {'Comp MAE':>10} {'Cont MAE':>10} {'Comp R2':>9} {'Cont R2':>9}")
    print("-" * 70)

    for qt in ['high', 'medium', 'low', 'highly_fragmented']:
        mask = quality_tiers == qt
        n = np.sum(mask)
        if n == 0:
            continue
        m_comp, r_comp, r2_c, o_comp = compute_metrics(true_comp[mask], pred_comp[mask])
        m_cont, r_cont, r2_t, o_cont = compute_metrics(true_cont[mask], pred_cont[mask])
        print(f"  {qt:<20} {n:>7} {m_comp:>9.3f}% {m_cont:>9.3f}% {r2_c:>9.4f} {r2_t:>9.4f}")

    # ========================================================================
    # Per-Phylum Top-10 Breakdown
    # ========================================================================
    print("\n" + "=" * 70)
    print("PER-PHYLUM BREAKDOWN (Top 10 by count)")
    print("=" * 70)

    phylum_counts = Counter(phyla)
    top_phyla = [p for p, _ in phylum_counts.most_common(10)]

    print(f"{'Phylum':<25} {'N':>7} {'Comp MAE':>10} {'Cont MAE':>10} {'Comp R2':>9} {'Cont R2':>9}")
    print("-" * 75)

    phylum_results = {}
    for phylum in top_phyla:
        mask = phyla == phylum
        n = np.sum(mask)
        m_comp, r_comp, r2_c, o_comp = compute_metrics(true_comp[mask], pred_comp[mask])
        m_cont, r_cont, r2_t, o_cont = compute_metrics(true_cont[mask], pred_cont[mask])
        print(f"  {phylum:<23} {n:>7} {m_comp:>9.3f}% {m_cont:>9.3f}% {r2_c:>9.4f} {r2_t:>9.4f}")
        phylum_results[phylum] = {
            'n': int(n), 'comp_mae': float(m_comp), 'cont_mae': float(m_cont),
            'comp_r2': float(r2_c), 'cont_r2': float(r2_t),
        }

    # ========================================================================
    # Speed Summary
    # ========================================================================
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"  Data loading:     {t_load_end - t_load_start:.2f}s")
    print(f"  ONNX inference:   {infer_time:.2f}s")
    print(f"  Total wall-clock: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"  Inference speed:  {n_samples / infer_time * 60:.0f} genomes/min (1 thread, inference only)")
    print(f"  Throughput:       {infer_time / n_samples * 1000:.3f} ms/sample")

    # ========================================================================
    # Save summary JSON
    # ========================================================================
    summary = {
        'n_samples': n_samples,
        'model': 'magicc_v3.onnx',
        'n_threads': N_THREADS,
        'batch_size': BATCH_SIZE,
        'overall': {
            'comp_mae': float(mae_comp),
            'comp_rmse': float(rmse_comp),
            'comp_r2': float(r2_comp),
            'comp_outlier_rate': float(outlier_comp),
            'cont_mae': float(mae_cont),
            'cont_rmse': float(rmse_cont),
            'cont_r2': float(r2_cont),
            'cont_outlier_rate': float(outlier_cont),
        },
        'per_sample_type': type_results,
        'per_phylum_top10': phylum_results,
        'timing': {
            'data_loading_s': float(t_load_end - t_load_start),
            'inference_s': float(infer_time),
            'total_s': float(total_time),
            'genomes_per_min': float(n_samples / infer_time * 60),
            'ms_per_sample': float(infer_time / n_samples * 1000),
        },
    }

    summary_path = OUTPUT_DIR / 'test_100k_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
