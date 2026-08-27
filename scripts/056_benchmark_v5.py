#!/usr/bin/env python3
"""
Script 56: Run MAGICC V5 Inference on Benchmark Sets A_v2, B_v2, C, D, E

Model: models/magicc_v5.onnx (V2 arch, 9249 k-mer + 7 assembly features)
Pipeline: FASTA -> k-mer counting -> assembly stats (7 features) -> normalization -> ONNX inference

Computes MAE, RMSE, R2 for completeness and contamination.
Saves per-set predictions and overall comparison table.
"""

import sys
import os
import time
import signal
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

# --- Collapse-safe: ignore SIGHUP ---
signal.signal(signal.SIGHUP, signal.SIG_IGN)

sys.path.insert(0, '/mnt/5c77b453-f7e1-48c8-afa3-5641857a41c7/tianrm/projects/magicc2')

import onnxruntime as ort
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats
from magicc.normalization import FeatureNormalizer

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/mnt/5c77b453-f7e1-48c8-afa3-5641857a41c7/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'
RESULTS_DIR = PROJECT_DIR / 'results'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')

N_WORKERS = 43
BATCH_SIZE = 64
ONNX_THREADS = 1

# Set definitions
SETS = {
    'A_v2': {'subdir': 'set_A_v2', 'desc': 'Completeness gradient (50-100%), 0% cont'},
    'B_v2': {'subdir': 'set_B_v2', 'desc': '100% comp, contamination gradient (0-80%)'},
    'C':    {'subdir': 'set_C',    'desc': 'Patescibacteria, uniform comp/cont'},
    'D':    {'subdir': 'set_D',    'desc': 'Archaea, uniform comp/cont'},
    'E':    {'subdir': 'set_E',    'desc': 'Mixed genomes'},
}

# V3 and V4 results for comparison (from progress file)
V3_RESULTS = {
    'A_v2': {'comp_mae': 1.99, 'cont_mae': 0.46},
    'B_v2': {'comp_mae': 0.77, 'cont_mae': 2.78},
    'C':    {'comp_mae': 2.96, 'cont_mae': 5.05},
    'D':    {'comp_mae': 4.06, 'cont_mae': 5.40},
    'E':    {'comp_mae': 3.92, 'cont_mae': 4.32},
}

V4_RESULTS = {
    'A_v2': {'comp_mae': 2.59, 'cont_mae': 1.48},
    'B_v2': {'comp_mae': 3.12, 'cont_mae': 5.11},
    'C':    {'comp_mae': 3.36, 'cont_mae': 7.44},
    'D':    {'comp_mae': 5.20, 'cont_mae': 7.77},
    'E':    {'comp_mae': 5.71, 'cont_mae': 5.88},
}

CHECKM2_RESULTS = {
    'A_v2': {'comp_mae': 2.54, 'cont_mae': 0.27},
    'B_v2': {'comp_mae': 0.45, 'cont_mae': 17.66},
    'C':    {'comp_mae': 7.99, 'cont_mae': 42.37},
    'D':    {'comp_mae': 9.89, 'cont_mae': 36.92},
    'E':    {'comp_mae': 9.14, 'cont_mae': 18.47},
}

# ============================================================================
# FASTA I/O
# ============================================================================
def read_fasta_contigs(fasta_path: str) -> List[str]:
    """Read FASTA file and return list of contig sequences."""
    contigs = []
    current_parts = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_parts:
                    contigs.append(''.join(current_parts).upper())
                    current_parts = []
            else:
                current_parts.append(line.strip())
    if current_parts:
        contigs.append(''.join(current_parts).upper())
    return [c for c in contigs if len(c) > 0]


# ============================================================================
# Worker for parallel feature extraction
# ============================================================================
_worker_kmer_counter = None

def _init_worker(kmers_path):
    """Initialize worker with KmerCounter (Numba JIT warmup)."""
    global _worker_kmer_counter
    _worker_kmer_counter = KmerCounter(kmers_path)
    # Warm up Numba
    dummy = ["ACGTACGTACGTACGTACGT" * 50]
    _worker_kmer_counter.count_contigs(dummy)


def _extract_features_worker(args):
    """Worker function: extract features for a single genome."""
    idx, fasta_path = args
    global _worker_kmer_counter

    try:
        contigs = read_fasta_contigs(fasta_path)
        if len(contigs) == 0:
            return idx, None, None, "no_contigs"

        kmer_counts = _worker_kmer_counter.count_contigs(contigs)
        log10_total = _worker_kmer_counter.total_kmer_count(kmer_counts)
        assembly_feats = compute_assembly_stats(log10_total, kmer_counts)
        return idx, kmer_counts.astype(np.float32), assembly_feats.astype(np.float32), None
    except Exception as e:
        return idx, None, None, str(e)


# ============================================================================
# Main inference function
# ============================================================================
def run_v5_on_set(set_name: str, set_info: dict, session, normalizer: FeatureNormalizer):
    """Run V5 inference on a single benchmark set."""
    set_dir = BENCHMARK_DIR / set_info['subdir']
    metadata_path = set_dir / 'metadata.tsv'
    fasta_dir = set_dir / 'fasta'
    output_path = set_dir / 'magicc_v5_predictions.tsv'

    if not metadata_path.exists():
        print(f"  {set_name}: metadata not found at {metadata_path}, skipping")
        return None

    df = pd.read_csv(metadata_path, sep='\t')
    n_genomes = len(df)
    print(f"\n{'='*70}")
    print(f"  Set {set_name}: {n_genomes} genomes — {set_info['desc']}")
    print(f"{'='*70}")

    # Build work items
    work_items = []
    for i, row in df.iterrows():
        genome_id = row['genome_id']
        fasta_path = fasta_dir / f"{genome_id}.fasta"
        if fasta_path.exists():
            work_items.append((i, str(fasta_path)))
        else:
            print(f"    WARNING: {fasta_path} not found")

    # Parallel feature extraction
    print(f"  Extracting features ({N_WORKERS} workers, {len(work_items)} genomes)...")
    t_feat_start = time.time()

    all_results = {}
    with Pool(processes=N_WORKERS, initializer=_init_worker,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        for result in pool.imap_unordered(_extract_features_worker, work_items, chunksize=10):
            idx, kmer, asm, err = result
            if err:
                print(f"    WARNING: genome at index {idx}: {err}")
            else:
                all_results[idx] = (kmer, asm)

    t_feat_end = time.time()
    feat_time = t_feat_end - t_feat_start

    valid_indices = sorted(all_results.keys())
    n_valid = len(valid_indices)
    print(f"  Feature extraction: {feat_time:.1f}s ({n_valid} genomes, "
          f"{feat_time/max(1,n_valid)*1000:.1f} ms/genome)")

    if n_valid == 0:
        print(f"  No valid genomes found!")
        return None

    # Stack arrays
    kmer_array = np.stack([all_results[i][0] for i in valid_indices])
    assembly_array = np.stack([all_results[i][1] for i in valid_indices])

    # Normalize
    print(f"  Normalizing features...")
    kmer_norm = normalizer.normalize_kmer(kmer_array).astype(np.float32)
    assembly_norm = normalizer.normalize_assembly(assembly_array).astype(np.float32)

    # ONNX inference
    print(f"  Running ONNX inference ({ONNX_THREADS} thread)...")
    t_infer_start = time.time()

    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    predictions = np.zeros((n_valid, 2), dtype=np.float32)
    for batch_start in range(0, n_valid, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_valid)
        feed = {
            input_names[0]: kmer_norm[batch_start:batch_end],
            input_names[1]: assembly_norm[batch_start:batch_end],
        }
        result = session.run([output_name], feed)
        predictions[batch_start:batch_end] = result[0]

    t_infer_end = time.time()
    infer_time = t_infer_end - t_infer_start
    total_time = feat_time + infer_time
    print(f"  ONNX inference: {infer_time:.2f}s ({infer_time/max(1,n_valid)*1000:.2f} ms/genome)")

    # Build results
    results = df.iloc[valid_indices].copy().reset_index(drop=True)
    results['pred_completeness'] = predictions[:, 0]
    results['pred_contamination'] = predictions[:, 1]
    results['wall_clock_s'] = total_time
    results['n_threads'] = N_WORKERS

    # Save predictions
    results.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved: {output_path}")

    # Compute metrics
    true_comp = results['true_completeness'].values
    true_cont = results['true_contamination'].values
    pred_comp = results['pred_completeness'].values
    pred_cont = results['pred_contamination'].values

    mae_comp = np.mean(np.abs(true_comp - pred_comp))
    mae_cont = np.mean(np.abs(true_cont - pred_cont))
    rmse_comp = np.sqrt(np.mean((true_comp - pred_comp)**2))
    rmse_cont = np.sqrt(np.mean((true_cont - pred_cont)**2))

    if np.std(true_comp) > 0 and np.std(pred_comp) > 0:
        r2_comp = np.corrcoef(true_comp, pred_comp)[0, 1]**2
    else:
        r2_comp = float('nan')
    if np.std(true_cont) > 0 and np.std(pred_cont) > 0:
        r2_cont = np.corrcoef(true_cont, pred_cont)[0, 1]**2
    else:
        r2_cont = float('nan')

    speed = n_valid / total_time * 60

    print(f"\n  Results for Set {set_name}:")
    print(f"    Completeness  MAE: {mae_comp:.2f}%, RMSE: {rmse_comp:.2f}%, R2: {r2_comp:.4f}")
    print(f"    Contamination MAE: {mae_cont:.2f}%, RMSE: {rmse_cont:.2f}%, R2: {r2_cont:.4f}")
    print(f"    Total time: {total_time:.1f}s, Speed: {speed:.0f} genomes/min ({N_WORKERS} threads)")

    return {
        'set_name': set_name,
        'n_genomes': n_valid,
        'description': set_info['desc'],
        'comp_mae': mae_comp,
        'cont_mae': mae_cont,
        'comp_rmse': rmse_comp,
        'cont_rmse': rmse_cont,
        'comp_r2': r2_comp,
        'cont_r2': r2_cont,
        'total_time': total_time,
        'feat_time': feat_time,
        'infer_time': infer_time,
        'speed_genomes_per_min': speed,
    }


def main():
    t0 = time.time()
    print("MAGICC V5 Benchmark on Sets A_v2, B_v2, C, D, E")
    print(f"Model: {ONNX_MODEL_PATH}")
    print(f"Workers: {N_WORKERS}")
    print()

    # Ensure results dir exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading ONNX model...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ONNX_THREADS
    sess_options.inter_op_num_threads = ONNX_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    print(f"  Inputs: {[(i.name, i.shape) for i in session.get_inputs()]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in session.get_outputs()]}")

    # Load normalizer
    print("Loading normalizer...")
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)
    print(f"  K-mer features: {normalizer.n_kmer_features}")
    print(f"  Assembly features: {normalizer.n_assembly_features}")

    # Run on each set sequentially
    all_results = []
    for set_name in ['A_v2', 'B_v2', 'C', 'D', 'E']:
        result = run_v5_on_set(set_name, SETS[set_name], session, normalizer)
        all_results.append(result)

    # ========================================================================
    # Save results summary
    # ========================================================================
    valid_results = [r for r in all_results if r is not None]

    # Per-set results table
    rows = []
    for r in valid_results:
        rows.append({
            'Set': r['set_name'],
            'N': r['n_genomes'],
            'Description': r['description'],
            'Comp_MAE': round(r['comp_mae'], 4),
            'Comp_RMSE': round(r['comp_rmse'], 4),
            'Comp_R2': round(r['comp_r2'], 4) if not np.isnan(r['comp_r2']) else '',
            'Cont_MAE': round(r['cont_mae'], 4),
            'Cont_RMSE': round(r['cont_rmse'], 4),
            'Cont_R2': round(r['cont_r2'], 4) if not np.isnan(r['cont_r2']) else '',
            'Speed_genomes_per_min': round(r['speed_genomes_per_min'], 1),
            'Wall_clock_s': round(r['total_time'], 1),
        })

    # Overall row
    total_n = sum(r['n_genomes'] for r in valid_results)
    total_time = sum(r['total_time'] for r in valid_results)

    # Compute overall MAE/RMSE from per-set predictions
    all_comp_ae = []
    all_cont_ae = []
    all_comp_se = []
    all_cont_se = []
    all_true_comp = []
    all_true_cont = []
    all_pred_comp = []
    all_pred_cont = []

    for r in valid_results:
        set_dir = BENCHMARK_DIR / SETS[r['set_name']]['subdir']
        pred_df = pd.read_csv(set_dir / 'magicc_v5_predictions.tsv', sep='\t')
        tc = pred_df['true_completeness'].values
        tx = pred_df['true_contamination'].values
        pc = pred_df['pred_completeness'].values
        px = pred_df['pred_contamination'].values
        all_comp_ae.extend(np.abs(tc - pc))
        all_cont_ae.extend(np.abs(tx - px))
        all_comp_se.extend((tc - pc)**2)
        all_cont_se.extend((tx - px)**2)
        all_true_comp.extend(tc)
        all_true_cont.extend(tx)
        all_pred_comp.extend(pc)
        all_pred_cont.extend(px)

    overall_comp_mae = np.mean(all_comp_ae)
    overall_cont_mae = np.mean(all_cont_ae)
    overall_comp_rmse = np.sqrt(np.mean(all_comp_se))
    overall_cont_rmse = np.sqrt(np.mean(all_cont_se))
    overall_comp_r2 = np.corrcoef(all_true_comp, all_pred_comp)[0, 1]**2
    overall_cont_r2 = np.corrcoef(all_true_cont, all_pred_cont)[0, 1]**2

    rows.append({
        'Set': 'Overall',
        'N': total_n,
        'Description': 'All 5 sets combined',
        'Comp_MAE': round(overall_comp_mae, 4),
        'Comp_RMSE': round(overall_comp_rmse, 4),
        'Comp_R2': round(overall_comp_r2, 4),
        'Cont_MAE': round(overall_cont_mae, 4),
        'Cont_RMSE': round(overall_cont_rmse, 4),
        'Cont_R2': round(overall_cont_r2, 4),
        'Speed_genomes_per_min': round(total_n / total_time * 60, 1),
        'Wall_clock_s': round(total_time, 1),
    })

    results_df = pd.DataFrame(rows)
    results_path = RESULTS_DIR / 'benchmark_v5_results.tsv'
    results_df.to_csv(results_path, sep='\t', index=False)
    print(f"\nSaved V5 results: {results_path}")

    # ========================================================================
    # Print comparison table
    # ========================================================================
    print(f"\n{'='*110}")
    print("V5 BENCHMARK RESULTS — COMPARISON WITH V3, V4, CheckM2")
    print(f"{'='*110}")
    print(f"{'Set':<8} {'N':>5}  |  {'V5 Comp':>8} {'V5 Cont':>8}  |  "
          f"{'V4 Comp':>8} {'V4 Cont':>8}  |  {'V3 Comp':>8} {'V3 Cont':>8}  |  "
          f"{'CkM2 Comp':>9} {'CkM2 Cont':>9}")
    print("-" * 110)

    for r in valid_results:
        sn = r['set_name']
        v4c = V4_RESULTS.get(sn, {}).get('comp_mae', 0)
        v4x = V4_RESULTS.get(sn, {}).get('cont_mae', 0)
        v3c = V3_RESULTS.get(sn, {}).get('comp_mae', 0)
        v3x = V3_RESULTS.get(sn, {}).get('cont_mae', 0)
        ckc = CHECKM2_RESULTS.get(sn, {}).get('comp_mae', 0)
        ckx = CHECKM2_RESULTS.get(sn, {}).get('cont_mae', 0)
        print(f"  {sn:<6} {r['n_genomes']:>5}  |  {r['comp_mae']:>7.2f}% {r['cont_mae']:>7.2f}%  |  "
              f"{v4c:>7.2f}% {v4x:>7.2f}%  |  {v3c:>7.2f}% {v3x:>7.2f}%  |  "
              f"{ckc:>8.2f}% {ckx:>8.2f}%")

    # Overall
    print("-" * 110)
    print(f"  {'ALL':<6} {total_n:>5}  |  {overall_comp_mae:>7.2f}% {overall_cont_mae:>7.2f}%  |  "
          f"{'4.00':>7}% {'5.53':>7}%  |  {'2.74':>7}% {'3.60':>7}%  |  "
          f"{'6.00':>8}% {'23.14':>8}%")
    print(f"{'='*110}")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
