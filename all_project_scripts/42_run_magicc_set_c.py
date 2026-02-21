#!/usr/bin/env python3
"""
Run MAGICC Inference on Motivating Set C

Model: models/magicc_v3.onnx (V3 with SE attention + cross-attention)
Pipeline: FASTA -> k-mer counting (9,249 canonical 9-mers) -> assembly stats (26 features)
          -> normalization -> ONNX inference

Output: data/benchmarks/motivating_v2/set_C/magicc_predictions.tsv
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

SET_DIR = BENCHMARK_DIR / 'motivating_v2' / 'set_C'

N_THREADS = 1
BATCH_SIZE = 64


def setup_onnx_session():
    """Create ONNX Runtime session with 1 thread."""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = N_THREADS
    sess_options.inter_op_num_threads = N_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider']
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


def main():
    t0 = time.time()
    print("MAGICC Inference on Motivating Set C")
    print(f"Model: {ONNX_MODEL_PATH}")
    print(f"Threads: {N_THREADS}")

    metadata_path = SET_DIR / 'metadata.tsv'
    fasta_dir = SET_DIR / 'fasta'
    output_path = SET_DIR / 'magicc_predictions.tsv'

    # Load model
    print("\nLoading ONNX model...")
    session = setup_onnx_session()
    print(f"  Input names: {[inp.name for inp in session.get_inputs()]}")

    print("Loading k-mer counter...")
    kmer_counter = KmerCounter(SELECTED_KMERS_PATH)
    print(f"  {kmer_counter.n_features} k-mers loaded")

    print("Loading normalizer...")
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)

    # Warm up Numba JIT
    print("Warming up Numba JIT...")
    dummy_contigs = ["ACGTACGTACGTACGTACGT" * 100]
    extract_features_single(dummy_contigs, kmer_counter)

    # Load metadata
    df = pd.read_csv(metadata_path, sep='\t')
    n_genomes = len(df)
    print(f"\nSet C: {n_genomes} genomes")

    # Extract features
    print("Extracting features...")
    t_feat_start = time.time()

    all_kmer = []
    all_assembly = []
    valid_indices = []

    for i, row in df.iterrows():
        genome_id = row['genome_id']
        fasta_path = fasta_dir / f"{genome_id}.fasta"

        if not fasta_path.exists():
            print(f"  WARNING: {fasta_path} not found")
            continue

        contigs = read_fasta_contigs(str(fasta_path))
        if len(contigs) == 0:
            print(f"  WARNING: {genome_id} has no contigs")
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
        print("  No valid genomes found!")
        return

    # Stack and normalize
    kmer_array = np.stack(all_kmer)
    assembly_array = np.stack(all_assembly)
    kmer_norm = normalizer.normalize_kmer(kmer_array).astype(np.float32)
    assembly_norm = normalizer.normalize_assembly(assembly_array).astype(np.float32)

    # ONNX inference
    print("Running ONNX inference...")
    t_infer_start = time.time()

    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    n_total = len(valid_indices)
    predictions = np.zeros((n_total, 2), dtype=np.float32)

    for batch_start in range(0, n_total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_total)
        feed = {
            input_names[0]: kmer_norm[batch_start:batch_end],
            input_names[1]: assembly_norm[batch_start:batch_end],
        }
        result = session.run([output_name], feed)
        predictions[batch_start:batch_end] = result[0]

    t_infer_end = time.time()
    infer_time = t_infer_end - t_infer_start
    total_time = feat_time + infer_time

    print(f"  ONNX inference: {infer_time:.2f}s ({infer_time/max(1,n_total)*1000:.2f} ms/genome)")

    # Build results
    results = df.iloc[valid_indices].copy().reset_index(drop=True)
    results['pred_completeness'] = predictions[:, 0]
    results['pred_contamination'] = predictions[:, 1]
    results['wall_clock_s'] = total_time
    results['n_threads'] = N_THREADS

    results.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved: {output_path}")

    # Compute accuracy
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

    print(f"\n{'='*70}")
    print("MAGICC RESULTS ON SET C")
    print(f"{'='*70}")
    print(f"  N genomes:        {n_total}")
    print(f"  Completeness MAE: {mae_comp:.2f}%")
    print(f"  Contamination MAE: {mae_cont:.2f}%")
    print(f"  Completeness RMSE: {rmse_comp:.2f}%")
    print(f"  Contamination RMSE: {rmse_cont:.2f}%")
    print(f"  Completeness R2:  {r2_comp:.4f}")
    print(f"  Contamination R2: {r2_cont:.4f}")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Speed:            {speed:.0f} genomes/min (1 thread)")
    print(f"  Feature time:     {feat_time:.1f}s")
    print(f"  Inference time:   {infer_time:.2f}s")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
