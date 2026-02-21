#!/usr/bin/env python3
"""
Run MAGICC Inference on Finished-Genome Benchmark Sets A_v2, B_v2, E

Model: models/magicc_v3.onnx (V3 with SE attention + cross-attention)
Pipeline: FASTA -> k-mer counting (9,249 canonical 9-mers) -> assembly stats (26 features)
          -> normalization -> ONNX inference

Output: magicc_predictions.tsv per set
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, '/home/tianrm/projects/magicc2')

import onnxruntime as ort
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats
from magicc.normalization import FeatureNormalizer

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v3.onnx')

N_THREADS = 1
BATCH_SIZE = 64

# Set definitions: (name, subdirectory)
SETS = {
    'A_v2': 'set_A_v2',
    'B_v2': 'set_B_v2',
    'E':    'set_E',
    'motiv_A': 'motivating_v2/set_A',
    'motiv_B': 'motivating_v2/set_B',
}


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


def extract_features_single(contigs: List[str], kmer_counter: KmerCounter) -> Tuple[np.ndarray, np.ndarray]:
    """Extract k-mer and assembly features for a single genome."""
    kmer_counts = kmer_counter.count_contigs(contigs)
    log10_total = kmer_counter.total_kmer_count(kmer_counts)
    assembly_feats = compute_assembly_stats(contigs, log10_total, kmer_counts)
    return kmer_counts.astype(np.float32), assembly_feats.astype(np.float32)


def run_magicc_on_set(set_name: str, set_subdir: str, session, kmer_counter: KmerCounter,
                       normalizer: FeatureNormalizer):
    """Run MAGICC inference on a single benchmark set."""
    set_dir = BENCHMARK_DIR / set_subdir
    metadata_path = set_dir / 'metadata.tsv'
    fasta_dir = set_dir / 'fasta'
    output_path = set_dir / 'magicc_predictions.tsv'

    if not metadata_path.exists():
        print(f"  {set_name}: metadata not found at {metadata_path}, skipping")
        return None

    df = pd.read_csv(metadata_path, sep='\t')
    n_genomes = len(df)
    print(f"\n  {set_name}: {n_genomes} genomes")

    # Extract features for all genomes
    print(f"  Extracting features...")
    t_feat_start = time.time()

    all_kmer = []
    all_assembly = []
    valid_indices = []

    for i, row in df.iterrows():
        genome_id = row['genome_id']
        fasta_path = fasta_dir / f"{genome_id}.fasta"

        if not fasta_path.exists():
            print(f"    WARNING: {fasta_path} not found")
            continue

        contigs = read_fasta_contigs(str(fasta_path))
        if len(contigs) == 0:
            print(f"    WARNING: {genome_id} has no contigs")
            continue

        kmer_counts, assembly_feats = extract_features_single(contigs, kmer_counter)
        all_kmer.append(kmer_counts)
        all_assembly.append(assembly_feats)
        valid_indices.append(i)

    t_feat_end = time.time()
    feat_time = t_feat_end - t_feat_start
    print(f"  Feature extraction: {feat_time:.1f}s ({len(valid_indices)} genomes, "
          f"{feat_time/max(1,len(valid_indices))*1000:.1f} ms/genome)")

    if len(valid_indices) == 0:
        print(f"  No valid genomes found!")
        return None

    # Stack into arrays
    kmer_array = np.stack(all_kmer)
    assembly_array = np.stack(all_assembly)

    # Normalize features
    print(f"  Normalizing features...")
    kmer_norm = normalizer.normalize_kmer(kmer_array).astype(np.float32)
    assembly_norm = normalizer.normalize_assembly(assembly_array).astype(np.float32)

    # Run ONNX inference in batches
    print(f"  Running ONNX inference (1 thread)...")
    t_infer_start = time.time()

    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    n_total = len(valid_indices)
    predictions = np.zeros((n_total, 2), dtype=np.float32)

    for batch_start in range(0, n_total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_total)
        batch_kmer = kmer_norm[batch_start:batch_end]
        batch_assembly = assembly_norm[batch_start:batch_end]

        feed = {
            input_names[0]: batch_kmer,
            input_names[1]: batch_assembly,
        }
        result = session.run([output_name], feed)
        predictions[batch_start:batch_end] = result[0]

    t_infer_end = time.time()
    infer_time = t_infer_end - t_infer_start
    print(f"  ONNX inference: {infer_time:.2f}s ({infer_time/max(1,n_total)*1000:.2f} ms/genome)")

    total_time = feat_time + infer_time

    # Build results DataFrame
    results = df.iloc[valid_indices].copy().reset_index(drop=True)
    results['pred_completeness'] = predictions[:, 0]
    results['pred_contamination'] = predictions[:, 1]
    results['wall_clock_s'] = total_time
    results['n_threads'] = N_THREADS

    # Save predictions
    results.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved: {output_path}")

    # Compute accuracy metrics
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

    speed = n_total / total_time * 60

    print(f"\n  Results for {set_name}:")
    print(f"    Completeness  MAE: {mae_comp:.2f}%, RMSE: {rmse_comp:.2f}%, R2: {r2_comp:.4f}")
    print(f"    Contamination MAE: {mae_cont:.2f}%, RMSE: {rmse_cont:.2f}%, R2: {r2_cont:.4f}")
    print(f"    Total time: {total_time:.1f}s, Speed: {speed:.0f} genomes/min (1 thread)")
    print(f"    Feature extraction: {feat_time:.1f}s, Inference: {infer_time:.2f}s")

    return {
        'set_name': set_name,
        'n_genomes': n_total,
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
        'results_df': results,
    }


def print_summary(all_results):
    """Print summary of all results."""
    print(f"\n{'='*90}")
    print("MAGICC FINISHED-GENOME BENCHMARK SUMMARY")
    print(f"{'='*90}")
    print(f"{'Set':<12} {'N':>6} {'Comp MAE':>10} {'Cont MAE':>10} {'Comp RMSE':>11} {'Cont RMSE':>11} "
          f"{'Comp R2':>8} {'Cont R2':>8} {'Speed':>12}")
    print("-"*90)

    for r in all_results:
        if r is None:
            continue
        print(f"  {r['set_name']:<10} {r['n_genomes']:>6} {r['comp_mae']:>9.2f}% {r['cont_mae']:>9.2f}% "
              f"{r['comp_rmse']:>10.2f}% {r['cont_rmse']:>10.2f}% "
              f"{r['comp_r2']:>8.4f} {r['cont_r2']:>8.4f} "
              f"{r['speed_genomes_per_min']:>8.0f}/min")

    print("="*90)


def main():
    t0 = time.time()
    print("MAGICC Inference on Finished-Genome Benchmark Sets")
    print(f"Model: {ONNX_MODEL_PATH}")
    print(f"Threads: {N_THREADS}")

    # Parse sets to run
    sets_to_run = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in SETS:
                sets_to_run.append(arg)
    if not sets_to_run:
        sets_to_run = ['A_v2', 'B_v2', 'E']

    # Load model and normalizer
    print("\nLoading ONNX model...")
    session = setup_onnx_session()
    print(f"  Input names: {[inp.name for inp in session.get_inputs()]}")
    print(f"  Output names: {[out.name for out in session.get_outputs()]}")

    print("Loading k-mer counter...")
    kmer_counter = KmerCounter(SELECTED_KMERS_PATH)
    print(f"  {kmer_counter.n_features} k-mers loaded")

    print("Loading normalizer...")
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)
    print(f"  K-mer features: {normalizer.n_kmer_features}")
    print(f"  Assembly features: {normalizer.n_assembly_features}")

    # Warm up Numba JIT
    print("Warming up Numba JIT...")
    dummy_contigs = ["ACGTACGTACGTACGTACGT" * 100]
    extract_features_single(dummy_contigs, kmer_counter)

    # Run on each set
    all_results = []
    for set_name in sets_to_run:
        result = run_magicc_on_set(
            set_name, SETS[set_name], session, kmer_counter, normalizer
        )
        all_results.append(result)

    # Print summary
    print_summary(all_results)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
