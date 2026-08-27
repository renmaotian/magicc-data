#!/usr/bin/env python3
"""
Script 57: Pathogen Analysis with MAGICC V5

Part B1: Synthetic S.enterica + L.monocytogenes (10 replicates per configuration)
Part B2: GTDB 1000+1000 genomes — V5 vs CheckM2 metadata

Model: models/magicc_v5.onnx (V2 arch, 9249 k-mer + 7 assembly features)
"""

import sys
import os
import time
import signal
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool

# --- Collapse-safe: ignore SIGHUP ---
signal.signal(signal.SIGHUP, signal.SIG_IGN)

sys.path.insert(0, '/path/to/magicc-legacy')

import onnxruntime as ort
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats
from magicc.normalization import FeatureNormalizer
from magicc.fragmentation import simulate_fragmentation, read_fasta

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/path/to/magicc-legacy')
DATA_DIR = PROJECT_DIR / 'data'
GENOME_DIR = DATA_DIR / 'genomes'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')

SE_GENOME = DATA_DIR / 'genomes' / 'GCF_001302605.1' / 'GCF_001302605.1_ASM130260v1_genomic.fna'
LM_GENOME = DATA_DIR / 'genomes' / 'GCF_000021185.1' / 'GCF_000021185.1_ASM2118v1_genomic.fna'

# Pathogen V2 data (for GTDB comparison reuse)
PATHOGEN_V2_DIR = DATA_DIR / 'benchmarks' / 'pathogen_analysis_v2'
GTDB_V2_DIR = PATHOGEN_V2_DIR / 'gtdb_comparison'
GTDB_V2_GENOMES_DIR = GTDB_V2_DIR / 'magicc_input'

# Output directory
OUTPUT_DIR = DATA_DIR / 'benchmarks' / 'pathogen_analysis_v5'

N_WORKERS = 43
BATCH_SIZE = 64
ONNX_THREADS = 1

# Replicate seeds
N_REPLICATES = 10
REPLICATE_SEEDS = list(range(1000, 1010))

# 5 genome configurations: (name, se_comp_frac, lm_cont_frac_of_se)
GENOME_SPECS = [
    ("100se_5lm",   1.00, 0.05),
    ("100se_20lm",  1.00, 0.20),
    ("100se_50lm",  1.00, 0.50),
    ("100se_100lm", 1.00, 1.00),
    ("60se_40lm",   0.60, 0.40),
]

# CheckM2 results from V2 pathogen analysis (whole-contig, for comparison)
CHECKM2_V2_RESULTS = {
    "100se_5lm":   {"comp": 100.0,  "cont": 6.71},
    "100se_20lm":  {"comp": 100.0,  "cont": 19.84},
    "100se_50lm":  {"comp": 100.0,  "cont": 52.96},
    "100se_100lm": {"comp": 100.0,  "cont": 76.59},
    "60se_40lm":   {"comp": 91.85,  "cont": 20.30},
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


def read_full_sequence(fasta_path: str) -> str:
    """Read FASTA and return concatenated sequence."""
    contigs = read_fasta_contigs(fasta_path)
    return ''.join(contigs)


# ============================================================================
# Feature extraction + inference
# ============================================================================
def extract_and_predict(contigs: List[str], kmer_counter: KmerCounter,
                        normalizer: FeatureNormalizer, session) -> Tuple[float, float]:
    """Extract features from contigs and run ONNX inference. Returns (comp, cont)."""
    kmer_counts = kmer_counter.count_contigs(contigs)
    log10_total = kmer_counter.total_kmer_count(kmer_counts)
    assembly_feats = compute_assembly_stats(log10_total, kmer_counts)

    kmer_norm = normalizer.normalize_kmer(kmer_counts.astype(np.float32).reshape(1, -1)).astype(np.float32)
    asm_norm = normalizer.normalize_assembly(assembly_feats.astype(np.float32).reshape(1, -1)).astype(np.float32)

    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name
    feed = {input_names[0]: kmer_norm, input_names[1]: asm_norm}
    result = session.run([output_name], feed)
    pred = result[0][0]
    return float(pred[0]), float(pred[1])


# ============================================================================
# Part B1: Synthetic genomes with 10 replicates
# ============================================================================
def generate_synthetic_genome(se_seq: str, lm_seq: str, se_comp_frac: float,
                               lm_cont_frac: float, seed: int) -> Tuple[List[str], float, float]:
    """
    Generate a synthetic genome with fragmentation for stochasticity.

    Uses simulate_fragmentation with 'medium' quality tier to create
    realistic contig structures, then adds contamination contigs.

    Returns: (contigs, true_completeness_pct, true_contamination_pct)
    """
    rng = np.random.default_rng(seed)
    se_size = len(se_seq)

    # Fragment the dominant genome (S. enterica)
    frag_result = simulate_fragmentation(
        se_seq, target_completeness=se_comp_frac,
        quality_tier='medium', rng=rng
    )
    se_contigs = frag_result['contigs']
    se_actual_bp = sum(len(c) for c in se_contigs)
    true_comp = se_actual_bp / se_size * 100.0

    # Contamination: fragment the contaminant genome (L. monocytogenes)
    target_lm_bp = int(se_size * lm_cont_frac)
    lm_contigs = []

    if target_lm_bp > 0:
        # Tile contaminant if needed
        lm_size = len(lm_seq)
        if target_lm_bp <= lm_size:
            lm_substr = lm_seq[:target_lm_bp]
        else:
            repeats = (target_lm_bp // lm_size) + 1
            lm_substr = (lm_seq * repeats)[:target_lm_bp]

        # Fragment the contaminant with different seed
        lm_frag = simulate_fragmentation(
            lm_substr, target_completeness=1.0,
            quality_tier='medium', rng=rng
        )
        lm_contigs = lm_frag['contigs']

    lm_actual_bp = sum(len(c) for c in lm_contigs)
    true_cont = lm_actual_bp / se_size * 100.0

    # Merge contigs (interleave for realism)
    all_contigs = list(se_contigs) + list(lm_contigs)
    rng.shuffle(all_contigs)

    return all_contigs, true_comp, true_cont


def run_part_b1(kmer_counter, normalizer, session):
    """Part B1: 5 configs x 10 replicates of synthetic genomes."""
    print("\n" + "=" * 80)
    print("PART B1: Synthetic S.enterica + L.monocytogenes (10 replicates)")
    print("=" * 80)

    synth_dir = OUTPUT_DIR / 'synthetic'
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Read reference genomes
    print("Reading reference genomes...")
    se_seq = read_full_sequence(str(SE_GENOME))
    lm_seq = read_full_sequence(str(LM_GENOME))
    se_size = len(se_seq)
    lm_size = len(lm_seq)
    print(f"  S. enterica: {se_size:,} bp ({se_size/1e6:.3f} Mbp)")
    print(f"  L. monocytogenes: {lm_size:,} bp ({lm_size/1e6:.3f} Mbp)")

    # Results storage
    all_replicate_results = []

    for spec_name, se_comp, lm_cont in GENOME_SPECS:
        print(f"\n--- Configuration: {spec_name} (S.e. {se_comp*100:.0f}% + L.m. {lm_cont*100:.0f}%) ---")

        true_comps = []
        true_conts = []
        pred_comps = []
        pred_conts = []

        for rep_idx, seed in enumerate(REPLICATE_SEEDS):
            # Generate synthetic genome
            contigs, true_comp, true_cont = generate_synthetic_genome(
                se_seq, lm_seq, se_comp, lm_cont, seed
            )

            # Run V5 prediction
            pred_comp, pred_cont = extract_and_predict(
                contigs, kmer_counter, normalizer, session
            )

            true_comps.append(true_comp)
            true_conts.append(true_cont)
            pred_comps.append(pred_comp)
            pred_conts.append(pred_cont)

            # Save FASTA for first replicate only (for reference)
            if rep_idx == 0:
                fasta_path = synth_dir / f"{spec_name}_rep0.fasta"
                with open(fasta_path, 'w') as f:
                    for ci, contig in enumerate(contigs):
                        f.write(f">contig_{ci}\n{contig}\n")

        true_comps = np.array(true_comps)
        true_conts = np.array(true_conts)
        pred_comps = np.array(pred_comps)
        pred_conts = np.array(pred_conts)

        # Report
        ckm2 = CHECKM2_V2_RESULTS.get(spec_name, {})
        print(f"  True completeness: {true_comps.mean():.2f}% +/- {true_comps.std():.2f}%")
        print(f"  True contamination: {true_conts.mean():.2f}% +/- {true_conts.std():.2f}%")
        print(f"  V5 pred comp: {pred_comps.mean():.2f}% +/- {pred_comps.std():.2f}%")
        print(f"  V5 pred cont: {pred_conts.mean():.2f}% +/- {pred_conts.std():.2f}%")
        print(f"  CheckM2 (whole-contig): comp={ckm2.get('comp','N/A')}, cont={ckm2.get('cont','N/A')}")

        result = {
            'config': spec_name,
            'target_comp': se_comp * 100,
            'target_cont': lm_cont * 100,
            'true_comp_mean': round(true_comps.mean(), 2),
            'true_comp_std': round(true_comps.std(), 2),
            'true_cont_mean': round(true_conts.mean(), 2),
            'true_cont_std': round(true_conts.std(), 2),
            'v5_comp_mean': round(pred_comps.mean(), 2),
            'v5_comp_std': round(pred_comps.std(), 2),
            'v5_cont_mean': round(pred_conts.mean(), 2),
            'v5_cont_std': round(pred_conts.std(), 2),
            'checkm2_comp': ckm2.get('comp', np.nan),
            'checkm2_cont': ckm2.get('cont', np.nan),
        }
        # Store per-replicate data
        result['replicate_true_comps'] = true_comps.tolist()
        result['replicate_true_conts'] = true_conts.tolist()
        result['replicate_pred_comps'] = pred_comps.tolist()
        result['replicate_pred_conts'] = pred_conts.tolist()

        all_replicate_results.append(result)

    # Save summary table
    summary_rows = []
    for r in all_replicate_results:
        summary_rows.append({
            'configuration': r['config'],
            'target_comp': r['target_comp'],
            'target_cont': r['target_cont'],
            'true_comp_mean': r['true_comp_mean'],
            'true_comp_std': r['true_comp_std'],
            'true_cont_mean': r['true_cont_mean'],
            'true_cont_std': r['true_cont_std'],
            'v5_comp_mean': r['v5_comp_mean'],
            'v5_comp_std': r['v5_comp_std'],
            'v5_cont_mean': r['v5_cont_mean'],
            'v5_cont_std': r['v5_cont_std'],
            'checkm2_comp': r['checkm2_comp'],
            'checkm2_cont': r['checkm2_cont'],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'synthetic_replicate_summary.tsv'
    summary_df.to_csv(summary_path, sep='\t', index=False)
    print(f"\nSaved synthetic summary: {summary_path}")

    # Save full replicate data as JSON
    json_path = OUTPUT_DIR / 'synthetic_replicate_details.json'
    with open(json_path, 'w') as f:
        json.dump(all_replicate_results, f, indent=2)
    print(f"Saved replicate details: {json_path}")

    return all_replicate_results


# ============================================================================
# Part B2: GTDB 1000+1000 genomes
# ============================================================================
_worker_kmer_counter_b2 = None

def _init_worker_b2(kmers_path):
    """Initialize worker."""
    global _worker_kmer_counter_b2
    _worker_kmer_counter_b2 = KmerCounter(kmers_path)
    dummy = ["ACGTACGTACGTACGTACGT" * 50]
    _worker_kmer_counter_b2.count_contigs(dummy)


def _extract_features_worker_b2(args):
    """Worker: extract features for a single genome."""
    idx, fasta_path = args
    global _worker_kmer_counter_b2

    try:
        contigs = read_fasta_contigs(fasta_path)
        if len(contigs) == 0:
            return idx, None, None, "no_contigs"
        kmer_counts = _worker_kmer_counter_b2.count_contigs(contigs)
        log10_total = _worker_kmer_counter_b2.total_kmer_count(kmer_counts)
        assembly_feats = compute_assembly_stats(log10_total, kmer_counts)
        return idx, kmer_counts.astype(np.float32), assembly_feats.astype(np.float32), None
    except Exception as e:
        return idx, None, None, str(e)


def run_part_b2(session, normalizer):
    """Part B2: GTDB 1000+1000 genomes with V5."""
    print("\n" + "=" * 80)
    print("PART B2: GTDB 1000+1000 Genomes — V5 vs CheckM2")
    print("=" * 80)

    gtdb_out_dir = OUTPUT_DIR / 'gtdb_comparison'
    gtdb_out_dir.mkdir(parents=True, exist_ok=True)

    # Load selected genomes from V2 pathogen analysis
    selected_path = GTDB_V2_DIR / 'selected_genomes.tsv'
    if not selected_path.exists():
        print(f"  ERROR: {selected_path} not found")
        return None
    selected_df = pd.read_csv(str(selected_path), sep='\t')
    print(f"  Selected genomes: {len(selected_df)}")

    # Build work items from FASTA files
    work_items = []
    acc_to_idx = {}
    for i, row in selected_df.iterrows():
        acc = row['accession']
        fasta_path = GTDB_V2_GENOMES_DIR / f"{acc}.fasta"
        if fasta_path.exists() or fasta_path.is_symlink():
            work_items.append((i, str(fasta_path)))
            acc_to_idx[acc] = i
        else:
            # Try .fna
            fna_path = GENOME_DIR / acc
            if fna_path.exists():
                fna_files = list(fna_path.glob("*.fna"))
                if fna_files:
                    work_items.append((i, str(fna_files[0])))
                    acc_to_idx[acc] = i

    print(f"  FASTA files found: {len(work_items)}")

    # Parallel feature extraction
    print(f"  Extracting features ({N_WORKERS} workers)...")
    t0 = time.time()

    all_results = {}
    with Pool(processes=N_WORKERS, initializer=_init_worker_b2,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        for result in pool.imap_unordered(_extract_features_worker_b2, work_items, chunksize=10):
            idx, kmer, asm, err = result
            if err:
                pass  # skip silently
            else:
                all_results[idx] = (kmer, asm)

    feat_time = time.time() - t0
    valid_indices = sorted(all_results.keys())
    n_valid = len(valid_indices)
    print(f"  Feature extraction: {feat_time:.1f}s ({n_valid} genomes)")

    if n_valid == 0:
        print("  No valid genomes!")
        return None

    # Stack and normalize
    kmer_array = np.stack([all_results[i][0] for i in valid_indices])
    assembly_array = np.stack([all_results[i][1] for i in valid_indices])

    kmer_norm = normalizer.normalize_kmer(kmer_array).astype(np.float32)
    asm_norm = normalizer.normalize_assembly(assembly_array).astype(np.float32)

    # ONNX inference
    print(f"  Running ONNX inference...")
    t_infer = time.time()
    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    predictions = np.zeros((n_valid, 2), dtype=np.float32)
    for batch_start in range(0, n_valid, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_valid)
        feed = {
            input_names[0]: kmer_norm[batch_start:batch_end],
            input_names[1]: asm_norm[batch_start:batch_end],
        }
        result = session.run([output_name], feed)
        predictions[batch_start:batch_end] = result[0]

    infer_time = time.time() - t_infer
    print(f"  ONNX inference: {infer_time:.2f}s")

    # Build results
    results_df = selected_df.iloc[valid_indices].copy().reset_index(drop=True)
    results_df['v5_completeness'] = predictions[:, 0]
    results_df['v5_contamination'] = predictions[:, 1]

    # Save raw predictions
    pred_path = gtdb_out_dir / 'v5_predictions.tsv'
    results_df.to_csv(pred_path, sep='\t', index=False)
    print(f"  Saved predictions: {pred_path}")

    # ================================================================
    # Compare V5 vs CheckM2 metadata
    # ================================================================
    print("\n  Comparing V5 vs CheckM2 metadata values...")

    comparison_rows = []
    for _, row in results_df.iterrows():
        comparison_rows.append({
            'accession': row['accession'],
            'species': row['species'],
            'checkm2_completeness': float(row['checkm2_completeness']),
            'checkm2_contamination': float(row['checkm2_contamination']),
            'v5_completeness': round(float(row['v5_completeness']), 4),
            'v5_contamination': round(float(row['v5_contamination']), 4),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = gtdb_out_dir / 'comparison_table.tsv'
    comparison_df.to_csv(comparison_path, sep='\t', index=False)

    # Per-species accuracy
    print(f"\n  V5 vs CheckM2 Accuracy (MAE against CheckM2 metadata):")
    accuracy_stats = {}
    for species in ["Salmonella_enterica", "Listeria_monocytogenes"]:
        sp_df = comparison_df[comparison_df['species'] == species]
        if len(sp_df) == 0:
            continue

        comp_diff = sp_df['v5_completeness'] - sp_df['checkm2_completeness']
        cont_diff = sp_df['v5_contamination'] - sp_df['checkm2_contamination']

        stats = {
            'n_genomes': int(len(sp_df)),
            'comp_mae': round(float(comp_diff.abs().mean()), 3),
            'comp_mean_diff': round(float(comp_diff.mean()), 3),
            'comp_median_ae': round(float(comp_diff.abs().median()), 3),
            'cont_mae': round(float(cont_diff.abs().mean()), 3),
            'cont_mean_diff': round(float(cont_diff.mean()), 3),
            'cont_median_ae': round(float(cont_diff.abs().median()), 3),
        }
        accuracy_stats[species] = stats

        sp_short = "S. enterica" if "Salmonella" in species else "L. monocytogenes"
        print(f"\n    {sp_short} ({stats['n_genomes']} genomes):")
        print(f"      Completeness:  MAE={stats['comp_mae']:.3f}%, "
              f"mean_diff={stats['comp_mean_diff']:.3f}%")
        print(f"      Contamination: MAE={stats['cont_mae']:.3f}%, "
              f"mean_diff={stats['cont_mean_diff']:.3f}%")

    # Overall
    all_comp_diff = comparison_df['v5_completeness'] - comparison_df['checkm2_completeness']
    all_cont_diff = comparison_df['v5_contamination'] - comparison_df['checkm2_contamination']
    overall_comp_mae = float(all_comp_diff.abs().mean())
    overall_cont_mae = float(all_cont_diff.abs().mean())
    print(f"\n    Overall ({len(comparison_df)} genomes):")
    print(f"      Completeness MAE: {overall_comp_mae:.3f}%")
    print(f"      Contamination MAE: {overall_cont_mae:.3f}%")

    # ================================================================
    # MIMAG Misclassification Analysis
    # ================================================================
    print(f"\n  MIMAG Misclassification Analysis (HQ = comp>=90% AND cont<=5%):")

    comparison_df['hq_checkm2'] = (
        (comparison_df['checkm2_completeness'] >= 90) &
        (comparison_df['checkm2_contamination'] <= 5)
    )
    comparison_df['hq_v5'] = (
        (comparison_df['v5_completeness'] >= 90) &
        (comparison_df['v5_contamination'] <= 5)
    )

    both_hq = comparison_df[comparison_df['hq_checkm2'] & comparison_df['hq_v5']]
    checkm2_only_hq = comparison_df[comparison_df['hq_checkm2'] & ~comparison_df['hq_v5']]
    v5_only_hq = comparison_df[comparison_df['hq_v5'] & ~comparison_df['hq_checkm2']]
    neither_hq = comparison_df[~comparison_df['hq_checkm2'] & ~comparison_df['hq_v5']]

    print(f"\n    Overall ({len(comparison_df)} genomes):")
    print(f"      Both HQ:          {len(both_hq)}")
    print(f"      CheckM2-only HQ:  {len(checkm2_only_hq)}")
    print(f"      V5-only HQ:       {len(v5_only_hq)}")
    print(f"      Neither HQ:       {len(neither_hq)}")

    mimag_rows = []
    for species in ["Salmonella_enterica", "Listeria_monocytogenes", "Total"]:
        if species == "Total":
            sp_df = comparison_df
        else:
            sp_df = comparison_df[comparison_df['species'] == species]

        n_both = len(sp_df[sp_df['hq_checkm2'] & sp_df['hq_v5']])
        n_ckm2_only = len(sp_df[sp_df['hq_checkm2'] & ~sp_df['hq_v5']])
        n_v5_only = len(sp_df[sp_df['hq_v5'] & ~sp_df['hq_checkm2']])
        n_neither = len(sp_df[~sp_df['hq_checkm2'] & ~sp_df['hq_v5']])

        mimag_rows.append({
            'species': species,
            'n_genomes': len(sp_df),
            'both_hq': n_both,
            'checkm2_only_hq': n_ckm2_only,
            'v5_only_hq': n_v5_only,
            'neither_hq': n_neither,
            'checkm2_hq_total': n_both + n_ckm2_only,
            'v5_hq_total': n_both + n_v5_only,
        })

        if species != "Total":
            sp_short = "S. enterica" if "Salmonella" in species else "L. monocytogenes"
            print(f"\n    {sp_short} ({len(sp_df)} genomes):")
            print(f"      Both HQ: {n_both}, CheckM2-only: {n_ckm2_only}, "
                  f"V5-only: {n_v5_only}, Neither: {n_neither}")

    mimag_df = pd.DataFrame(mimag_rows)
    mimag_path = gtdb_out_dir / 'mimag_analysis.tsv'
    mimag_df.to_csv(mimag_path, sep='\t', index=False)
    print(f"\n  Saved MIMAG analysis: {mimag_path}")

    # Save summary JSON
    summary = {
        'total_genomes_analyzed': int(len(comparison_df)),
        'species_breakdown': {
            'Salmonella_enterica': int(len(comparison_df[comparison_df['species'] == 'Salmonella_enterica'])),
            'Listeria_monocytogenes': int(len(comparison_df[comparison_df['species'] == 'Listeria_monocytogenes'])),
        },
        'v5_vs_checkm2': {
            'overall_comp_mae': round(overall_comp_mae, 3),
            'overall_cont_mae': round(overall_cont_mae, 3),
        },
        'accuracy_stats': accuracy_stats,
        'mimag_hq_analysis': {
            'both_hq': int(len(both_hq)),
            'checkm2_only_hq': int(len(checkm2_only_hq)),
            'v5_only_hq': int(len(v5_only_hq)),
            'neither_hq': int(len(neither_hq)),
        },
    }
    summary_path = gtdb_out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    return comparison_df, mimag_df, summary


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("Script 57: Pathogen Analysis with MAGICC V5")
    print(f"Model: {ONNX_MODEL_PATH}")
    print(f"Workers: {N_WORKERS}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading ONNX model...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ONNX_THREADS
    sess_options.inter_op_num_threads = ONNX_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    print(f"  Inputs: {[(i.name, i.shape) for i in session.get_inputs()]}")

    # Load normalizer and kmer counter
    print("Loading normalizer...")
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)
    print(f"  K-mer features: {normalizer.n_kmer_features}, Assembly features: {normalizer.n_assembly_features}")

    print("Loading k-mer counter...")
    kmer_counter = KmerCounter(SELECTED_KMERS_PATH)
    print(f"  {kmer_counter.n_features} k-mers loaded")

    # Warm up Numba
    print("Warming up Numba JIT...")
    dummy = ["ACGTACGTACGTACGTACGT" * 100]
    kmer_counter.count_contigs(dummy)

    # ================================================================
    # Part B1: Synthetic genomes with replicates
    # ================================================================
    synth_results = run_part_b1(kmer_counter, normalizer, session)

    # ================================================================
    # Part B2: GTDB 1000+1000 genomes
    # ================================================================
    gtdb_results = run_part_b2(session, normalizer)

    # ================================================================
    # Final summary
    # ================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print("PATHOGEN ANALYSIS V5 COMPLETE")
    print(f"{'='*80}")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if synth_results:
        print(f"\nPart B1 — Synthetic Genome Results (10 replicates each):")
        print(f"{'Config':<16} {'True Comp':>12} {'True Cont':>12} "
              f"{'V5 Comp':>18} {'V5 Cont':>18} "
              f"{'CkM2 Comp':>10} {'CkM2 Cont':>10}")
        print("-" * 100)
        for r in synth_results:
            print(f"  {r['config']:<14} "
                  f"{r['true_comp_mean']:>5.1f}+/-{r['true_comp_std']:>4.1f} "
                  f"{r['true_cont_mean']:>5.1f}+/-{r['true_cont_std']:>4.1f}  "
                  f"{r['v5_comp_mean']:>5.1f}+/-{r['v5_comp_std']:>4.1f}  "
                  f"{r['v5_cont_mean']:>5.1f}+/-{r['v5_cont_std']:>4.1f}  "
                  f"{r['checkm2_comp']:>9.1f} {r['checkm2_cont']:>9.1f}")

    if gtdb_results:
        comparison_df, mimag_df, summary = gtdb_results
        print(f"\nPart B2 — GTDB Comparison:")
        print(f"  Overall comp MAE vs CheckM2: {summary['v5_vs_checkm2']['overall_comp_mae']:.3f}%")
        print(f"  Overall cont MAE vs CheckM2: {summary['v5_vs_checkm2']['overall_cont_mae']:.3f}%")
        print(f"  MIMAG: Both HQ={summary['mimag_hq_analysis']['both_hq']}, "
              f"CheckM2-only={summary['mimag_hq_analysis']['checkm2_only_hq']}, "
              f"V5-only={summary['mimag_hq_analysis']['v5_only_hq']}")


if __name__ == '__main__':
    main()
