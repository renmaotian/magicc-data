#!/usr/bin/env python3
"""
Script 60: GTDB Pure Culture vs MAG/SAG Comparison with MAGICC V5

Samples 1000 pure culture genomes and 1000 MAG/SAG genomes from GTDB metadata,
runs MAGICC V5 on them, compares with CheckM2 metadata values, and analyzes
MIMAG misclassification.

Uses only genomes already downloaded in data/genomes/ to avoid lengthy downloads.

Steps:
  1. Load GTDB metadata (bacteria + archaea), identify pure culture vs MAG/SAG
  2. Filter to genomes available locally
  3. Sample 1000 pure culture + 1000 MAG/SAG (seed=42)
  4. Run MAGICC V5 inference (ONNX) with parallel feature extraction
  5. Compare MAGICC V5 vs CheckM2 (from GTDB metadata)
  6. MIMAG classification analysis

Output: data/benchmarks/gtdb_pure_vs_mag/
"""

import sys
import os
import time
import signal
import json
import glob as glob_mod
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from multiprocessing import Pool

# --- Collapse-safe: ignore SIGHUP ---
signal.signal(signal.SIGHUP, signal.SIG_IGN)

sys.path.insert(0, '/path/to/magicc-legacy')

import onnxruntime as ort
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats
from magicc.normalization import FeatureNormalizer

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/path/to/magicc-legacy')
DATA_DIR = PROJECT_DIR / 'data'
GENOMES_DIR = DATA_DIR / 'genomes'
OUTPUT_DIR = DATA_DIR / 'benchmarks' / 'gtdb_pure_vs_mag'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')

N_WORKERS = 43
BATCH_SIZE = 64
ONNX_THREADS = 1
SEED = 42

CHECKPOINT_PATH = OUTPUT_DIR / 'checkpoint.json'

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


def find_fasta(genome_dir: str) -> Optional[str]:
    """Find the FASTA file in a genome directory."""
    patterns = [
        os.path.join(genome_dir, '*_genomic.fna'),
        os.path.join(genome_dir, '*.fna'),
        os.path.join(genome_dir, '*.fasta'),
        os.path.join(genome_dir, '*.fa'),
    ]
    for pattern in patterns:
        matches = glob_mod.glob(pattern)
        if matches:
            return matches[0]
    return None


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
# Step 1: Load and sample GTDB metadata
# ============================================================================
def load_and_sample_metadata():
    """Load GTDB metadata, filter to available genomes, sample 1000+1000."""

    print("=" * 70)
    print("STEP 1: Loading GTDB metadata and sampling genomes")
    print("=" * 70)

    # Columns we need
    usecols = [
        'accession', 'ncbi_genbank_assembly_accession',
        'ncbi_genome_category', 'ncbi_assembly_level',
        'checkm2_completeness', 'checkm2_contamination',
        'genome_size', 'contig_count', 'n50_contigs',
        'gc_percentage', 'gtdb_taxonomy',
    ]

    # Load bacteria
    print("  Loading bacterial metadata...")
    bac = pd.read_csv(
        DATA_DIR / 'gtdb' / 'bac120_metadata.tsv.gz',
        sep='\t', usecols=usecols,
    )
    bac['domain'] = 'Bacteria'
    print(f"    Bacterial genomes: {len(bac):,}")

    # Load archaea
    print("  Loading archaeal metadata...")
    arc = pd.read_csv(
        DATA_DIR / 'gtdb' / 'ar53_metadata.tsv.gz',
        sep='\t', usecols=usecols,
    )
    arc['domain'] = 'Archaea'
    print(f"    Archaeal genomes: {len(arc):,}")

    # Combine
    df = pd.concat([bac, arc], ignore_index=True)
    print(f"    Total genomes: {len(df):,}")

    # Extract GCA accession from ncbi_genbank_assembly_accession
    df['gca_accession'] = df['ncbi_genbank_assembly_accession']

    # Get available genome directories
    print("\n  Checking available genomes...")
    available_dirs = set(os.listdir(str(GENOMES_DIR)))
    print(f"    Available genome directories: {len(available_dirs):,}")

    # Filter to available genomes
    df['available'] = df['gca_accession'].isin(available_dirs)
    n_available = df['available'].sum()
    print(f"    GTDB genomes with local download: {n_available:,}")

    # Also check that FASTA files actually exist
    df_available = df[df['available']].copy()

    # Classify genome type
    pure_categories = {'none'}  # "none" means pure culture
    mag_sag_categories = {
        'derived from metagenome',
        'derived from single cell',
        'derived from environmental sample',
    }

    df_available['genome_type'] = 'unknown'
    df_available.loc[
        df_available['ncbi_genome_category'].isin(pure_categories),
        'genome_type'
    ] = 'pure_culture'
    df_available.loc[
        df_available['ncbi_genome_category'].isin(mag_sag_categories),
        'genome_type'
    ] = 'mag_sag'

    print(f"\n  Genome type distribution (available genomes):")
    print(f"    Pure culture: {(df_available['genome_type'] == 'pure_culture').sum():,}")
    print(f"    MAG/SAG:      {(df_available['genome_type'] == 'mag_sag').sum():,}")
    print(f"    Unknown:      {(df_available['genome_type'] == 'unknown').sum():,}")

    # Sample 1000 pure culture
    pure_pool = df_available[df_available['genome_type'] == 'pure_culture']
    mag_pool = df_available[df_available['genome_type'] == 'mag_sag']

    rng = np.random.RandomState(SEED)

    n_pure_target = min(1000, len(pure_pool))
    n_mag_target = min(1000, len(mag_pool))

    pure_sample = pure_pool.sample(n=n_pure_target, random_state=rng).copy()
    mag_sample = mag_pool.sample(n=n_mag_target, random_state=rng).copy()

    print(f"\n  Sampled:")
    print(f"    Pure culture: {len(pure_sample)}")
    print(f"    MAG/SAG:      {len(mag_sample)}")

    # Domain breakdown
    for label, sample in [('Pure culture', pure_sample), ('MAG/SAG', mag_sample)]:
        bac_n = (sample['domain'] == 'Bacteria').sum()
        arc_n = (sample['domain'] == 'Archaea').sum()
        print(f"    {label}: {bac_n} bacteria, {arc_n} archaea")

    # CheckM2 quality distribution
    for label, sample in [('Pure culture', pure_sample), ('MAG/SAG', mag_sample)]:
        print(f"\n    {label} CheckM2 quality:")
        print(f"      Completeness: mean={sample['checkm2_completeness'].mean():.1f}%, "
              f"median={sample['checkm2_completeness'].median():.1f}%, "
              f"range=[{sample['checkm2_completeness'].min():.1f}%, "
              f"{sample['checkm2_completeness'].max():.1f}%]")
        print(f"      Contamination: mean={sample['checkm2_contamination'].mean():.1f}%, "
              f"median={sample['checkm2_contamination'].median():.1f}%, "
              f"range=[{sample['checkm2_contamination'].min():.1f}%, "
              f"{sample['checkm2_contamination'].max():.1f}%]")

    return pure_sample, mag_sample


# ============================================================================
# Step 2: Locate FASTA files
# ============================================================================
def locate_fasta_files(sample_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Find FASTA file paths for sampled genomes."""
    print(f"\n  Locating FASTA files for {label}...")

    fasta_paths = []
    missing = []
    for _, row in sample_df.iterrows():
        gca = row['gca_accession']
        genome_dir = str(GENOMES_DIR / gca)
        fasta = find_fasta(genome_dir)
        if fasta:
            fasta_paths.append(fasta)
        else:
            fasta_paths.append(None)
            missing.append(gca)

    sample_df = sample_df.copy()
    sample_df['fasta_path'] = fasta_paths
    n_found = sample_df['fasta_path'].notna().sum()
    print(f"    Found: {n_found}, Missing: {len(missing)}")

    if missing and len(missing) <= 10:
        for m in missing:
            print(f"      Missing: {m}")

    # Filter to only those with FASTA
    sample_df = sample_df[sample_df['fasta_path'].notna()].reset_index(drop=True)
    return sample_df


# ============================================================================
# Step 3: Run MAGICC V5 inference
# ============================================================================
def run_magicc_v5(sample_df: pd.DataFrame, label: str,
                  session, normalizer: FeatureNormalizer,
                  checkpoint_key: str) -> pd.DataFrame:
    """Run MAGICC V5 on a set of genomes."""

    print(f"\n{'='*70}")
    print(f"STEP 3: Running MAGICC V5 on {label} ({len(sample_df)} genomes)")
    print(f"{'='*70}")

    # Check checkpoint
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)

    output_path = OUTPUT_DIR / f'{checkpoint_key}_predictions.tsv'

    if checkpoint.get(checkpoint_key) == 'done' and output_path.exists():
        print(f"  Checkpoint found: {checkpoint_key} already done. Loading results...")
        result_df = pd.read_csv(output_path, sep='\t')
        print(f"  Loaded {len(result_df)} predictions from checkpoint.")
        return result_df

    # Build work items
    work_items = []
    for i, row in sample_df.iterrows():
        work_items.append((i, row['fasta_path']))

    # Parallel feature extraction
    print(f"  Extracting features ({N_WORKERS} workers, {len(work_items)} genomes)...")
    t_feat_start = time.time()

    all_results = {}
    errors = []
    with Pool(processes=N_WORKERS, initializer=_init_worker,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        for result in pool.imap_unordered(_extract_features_worker, work_items, chunksize=10):
            idx, kmer, asm, err = result
            if err:
                errors.append((idx, err))
            else:
                all_results[idx] = (kmer, asm)

    t_feat_end = time.time()
    feat_time = t_feat_end - t_feat_start

    valid_indices = sorted(all_results.keys())
    n_valid = len(valid_indices)
    print(f"  Feature extraction: {feat_time:.1f}s ({n_valid} genomes, "
          f"{feat_time/max(1,n_valid)*1000:.1f} ms/genome)")

    if errors:
        print(f"  Errors: {len(errors)}")
        for idx, err in errors[:5]:
            print(f"    Index {idx}: {err}")

    if n_valid == 0:
        print("  ERROR: No valid genomes found!")
        return pd.DataFrame()

    # Stack arrays
    kmer_array = np.stack([all_results[i][0] for i in valid_indices])
    assembly_array = np.stack([all_results[i][1] for i in valid_indices])

    # Normalize
    print(f"  Normalizing features...")
    kmer_norm = normalizer.normalize_kmer(kmer_array).astype(np.float32)
    assembly_norm = normalizer.normalize_assembly(assembly_array).astype(np.float32)

    # ONNX inference
    print(f"  Running ONNX inference...")
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
    print(f"  Total: {total_time:.1f}s, {n_valid/total_time*60:.0f} genomes/min")

    # Build results DataFrame
    result_df = sample_df.iloc[valid_indices].copy().reset_index(drop=True)
    result_df['magicc_completeness'] = predictions[:, 0]
    result_df['magicc_contamination'] = predictions[:, 1]
    result_df['wall_clock_s'] = total_time
    result_df['n_workers'] = N_WORKERS

    # Save
    result_df.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved: {output_path}")

    # Update checkpoint
    checkpoint[checkpoint_key] = 'done'
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f)

    return result_df


# ============================================================================
# Step 4: Compare MAGICC V5 vs CheckM2
# ============================================================================
def compute_metrics(true_vals, pred_vals):
    """Compute MAE, RMSE, R2, mean difference."""
    ae = np.abs(true_vals - pred_vals)
    se = (true_vals - pred_vals) ** 2
    mae = np.mean(ae)
    rmse = np.sqrt(np.mean(se))
    diff = np.mean(pred_vals - true_vals)  # positive = MAGICC predicts higher

    ss_res = np.sum(se)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_diff': diff,
        'median_ae': np.median(ae),
        'p95_ae': np.percentile(ae, 95),
    }


def compare_predictions(pure_df: pd.DataFrame, mag_df: pd.DataFrame):
    """Compare MAGICC V5 predictions vs CheckM2 metadata values."""

    print(f"\n{'='*70}")
    print("STEP 4: Comparing MAGICC V5 vs CheckM2 metadata values")
    print(f"{'='*70}")

    summary_rows = []

    for label, df in [('Pure Culture', pure_df), ('MAG/SAG', mag_df)]:
        print(f"\n  --- {label} ({len(df)} genomes) ---")

        checkm2_comp = df['checkm2_completeness'].values.astype(float)
        checkm2_cont = df['checkm2_contamination'].values.astype(float)
        magicc_comp = df['magicc_completeness'].values.astype(float)
        magicc_cont = df['magicc_contamination'].values.astype(float)

        comp_metrics = compute_metrics(checkm2_comp, magicc_comp)
        cont_metrics = compute_metrics(checkm2_cont, magicc_cont)

        print(f"    Completeness:   MAE={comp_metrics['mae']:.2f}%, "
              f"RMSE={comp_metrics['rmse']:.2f}%, "
              f"R2={comp_metrics['r2']:.4f}, "
              f"MeanDiff={comp_metrics['mean_diff']:+.2f}%")
        print(f"    Contamination:  MAE={cont_metrics['mae']:.2f}%, "
              f"RMSE={cont_metrics['rmse']:.2f}%, "
              f"R2={cont_metrics['r2']:.4f}, "
              f"MeanDiff={cont_metrics['mean_diff']:+.2f}%")
        print(f"    Comp median AE: {comp_metrics['median_ae']:.2f}%, "
              f"95th: {comp_metrics['p95_ae']:.2f}%")
        print(f"    Cont median AE: {cont_metrics['median_ae']:.2f}%, "
              f"95th: {cont_metrics['p95_ae']:.2f}%")

        summary_rows.append({
            'genome_type': label,
            'n_genomes': len(df),
            'comp_mae': round(comp_metrics['mae'], 4),
            'comp_rmse': round(comp_metrics['rmse'], 4),
            'comp_r2': round(comp_metrics['r2'], 4),
            'comp_mean_diff': round(comp_metrics['mean_diff'], 4),
            'comp_median_ae': round(comp_metrics['median_ae'], 4),
            'comp_p95_ae': round(comp_metrics['p95_ae'], 4),
            'cont_mae': round(cont_metrics['mae'], 4),
            'cont_rmse': round(cont_metrics['rmse'], 4),
            'cont_r2': round(cont_metrics['r2'], 4),
            'cont_mean_diff': round(cont_metrics['mean_diff'], 4),
            'cont_median_ae': round(cont_metrics['median_ae'], 4),
            'cont_p95_ae': round(cont_metrics['p95_ae'], 4),
        })

    # Combined stats
    all_df = pd.concat([pure_df, mag_df], ignore_index=True)
    checkm2_comp = all_df['checkm2_completeness'].values.astype(float)
    checkm2_cont = all_df['checkm2_contamination'].values.astype(float)
    magicc_comp = all_df['magicc_completeness'].values.astype(float)
    magicc_cont = all_df['magicc_contamination'].values.astype(float)

    comp_metrics = compute_metrics(checkm2_comp, magicc_comp)
    cont_metrics = compute_metrics(checkm2_cont, magicc_cont)

    print(f"\n  --- Combined ({len(all_df)} genomes) ---")
    print(f"    Completeness:   MAE={comp_metrics['mae']:.2f}%, "
          f"RMSE={comp_metrics['rmse']:.2f}%, "
          f"R2={comp_metrics['r2']:.4f}, "
          f"MeanDiff={comp_metrics['mean_diff']:+.2f}%")
    print(f"    Contamination:  MAE={cont_metrics['mae']:.2f}%, "
          f"RMSE={cont_metrics['rmse']:.2f}%, "
          f"R2={cont_metrics['r2']:.4f}, "
          f"MeanDiff={cont_metrics['mean_diff']:+.2f}%")

    summary_rows.append({
        'genome_type': 'Combined',
        'n_genomes': len(all_df),
        'comp_mae': round(comp_metrics['mae'], 4),
        'comp_rmse': round(comp_metrics['rmse'], 4),
        'comp_r2': round(comp_metrics['r2'], 4),
        'comp_mean_diff': round(comp_metrics['mean_diff'], 4),
        'comp_median_ae': round(comp_metrics['median_ae'], 4),
        'comp_p95_ae': round(comp_metrics['p95_ae'], 4),
        'cont_mae': round(cont_metrics['mae'], 4),
        'cont_rmse': round(cont_metrics['rmse'], 4),
        'cont_r2': round(cont_metrics['r2'], 4),
        'cont_mean_diff': round(cont_metrics['mean_diff'], 4),
        'cont_median_ae': round(cont_metrics['median_ae'], 4),
        'cont_p95_ae': round(cont_metrics['p95_ae'], 4),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'comparison_summary.tsv'
    summary_df.to_csv(summary_path, sep='\t', index=False)
    print(f"\n  Saved: {summary_path}")

    return summary_df


# ============================================================================
# Step 5: MIMAG Classification Analysis
# ============================================================================
def mimag_analysis(pure_df: pd.DataFrame, mag_df: pd.DataFrame):
    """Analyze MIMAG classification agreement between MAGICC V5 and CheckM2."""

    print(f"\n{'='*70}")
    print("STEP 5: MIMAG Classification Analysis")
    print(f"{'='*70}")

    mimag_rows = []

    for label, df in [('Pure Culture', pure_df), ('MAG/SAG', mag_df)]:
        print(f"\n  --- {label} ({len(df)} genomes) ---")

        checkm2_comp = df['checkm2_completeness'].values.astype(float)
        checkm2_cont = df['checkm2_contamination'].values.astype(float)
        magicc_comp = df['magicc_completeness'].values.astype(float)
        magicc_cont = df['magicc_contamination'].values.astype(float)

        # ---- MIMAG High Quality ----
        # HQ: completeness >= 90% AND contamination <= 5%
        checkm2_hq = (checkm2_comp >= 90) & (checkm2_cont <= 5)
        magicc_hq = (magicc_comp >= 90) & (magicc_cont <= 5)

        both_hq = checkm2_hq & magicc_hq
        checkm2_only_hq = checkm2_hq & ~magicc_hq
        magicc_only_hq = ~checkm2_hq & magicc_hq
        neither_hq = ~checkm2_hq & ~magicc_hq

        print(f"\n    MIMAG High Quality (comp>=90%, cont<=5%):")
        print(f"      CheckM2 HQ: {checkm2_hq.sum()}")
        print(f"      MAGICC  HQ: {magicc_hq.sum()}")
        print(f"      Both agree HQ:        {both_hq.sum()}")
        print(f"      CheckM2-only HQ:      {checkm2_only_hq.sum()}  (potential over-classification)")
        print(f"      MAGICC-only HQ:       {magicc_only_hq.sum()}  (potential under-classification by CheckM2)")
        print(f"      Neither HQ:           {neither_hq.sum()}")

        # Breakdown of CheckM2-only HQ disagreement reasons
        if checkm2_only_hq.sum() > 0:
            # MAGICC says not HQ - why?
            magicc_low_comp = (magicc_comp[checkm2_only_hq] < 90)
            magicc_high_cont = (magicc_cont[checkm2_only_hq] > 5)
            magicc_both = magicc_low_comp & magicc_high_cont

            print(f"      CheckM2-only HQ breakdown (MAGICC disagrees because):")
            print(f"        MAGICC comp < 90%:                 {(magicc_low_comp & ~magicc_high_cont).sum()}")
            print(f"        MAGICC cont > 5%:                  {(magicc_high_cont & ~magicc_low_comp).sum()}")
            print(f"        Both comp < 90% AND cont > 5%:     {magicc_both.sum()}")

        # ---- MIMAG Medium Quality ----
        # MQ: completeness >= 50% AND contamination <= 10%
        checkm2_mq = (checkm2_comp >= 50) & (checkm2_cont <= 10)
        magicc_mq = (magicc_comp >= 50) & (magicc_cont <= 10)

        both_mq = checkm2_mq & magicc_mq
        checkm2_only_mq = checkm2_mq & ~magicc_mq
        magicc_only_mq = ~checkm2_mq & magicc_mq
        neither_mq = ~checkm2_mq & ~magicc_mq

        print(f"\n    MIMAG Medium Quality (comp>=50%, cont<=10%):")
        print(f"      CheckM2 MQ: {checkm2_mq.sum()}")
        print(f"      MAGICC  MQ: {magicc_mq.sum()}")
        print(f"      Both agree MQ:        {both_mq.sum()}")
        print(f"      CheckM2-only MQ:      {checkm2_only_mq.sum()}")
        print(f"      MAGICC-only MQ:       {magicc_only_mq.sum()}")
        print(f"      Neither MQ:           {neither_mq.sum()}")

        # Breakdown of CheckM2-only MQ disagreement reasons
        if checkm2_only_mq.sum() > 0:
            magicc_low_comp = (magicc_comp[checkm2_only_mq] < 50)
            magicc_high_cont = (magicc_cont[checkm2_only_mq] > 10)
            magicc_both = magicc_low_comp & magicc_high_cont

            print(f"      CheckM2-only MQ breakdown (MAGICC disagrees because):")
            print(f"        MAGICC comp < 50%:                 {(magicc_low_comp & ~magicc_high_cont).sum()}")
            print(f"        MAGICC cont > 10%:                 {(magicc_high_cont & ~magicc_low_comp).sum()}")
            print(f"        Both comp < 50% AND cont > 10%:    {magicc_both.sum()}")

        # Store results
        mimag_rows.append({
            'genome_type': label,
            'n_genomes': len(df),
            # HQ
            'checkm2_hq': int(checkm2_hq.sum()),
            'magicc_hq': int(magicc_hq.sum()),
            'both_hq': int(both_hq.sum()),
            'checkm2_only_hq': int(checkm2_only_hq.sum()),
            'magicc_only_hq': int(magicc_only_hq.sum()),
            'neither_hq': int(neither_hq.sum()),
            'hq_agreement_rate': round(
                (both_hq.sum() + neither_hq.sum()) / len(df) * 100, 2
            ),
            # MQ
            'checkm2_mq': int(checkm2_mq.sum()),
            'magicc_mq': int(magicc_mq.sum()),
            'both_mq': int(both_mq.sum()),
            'checkm2_only_mq': int(checkm2_only_mq.sum()),
            'magicc_only_mq': int(magicc_only_mq.sum()),
            'neither_mq': int(neither_mq.sum()),
            'mq_agreement_rate': round(
                (both_mq.sum() + neither_mq.sum()) / len(df) * 100, 2
            ),
        })

    # Combined
    all_df = pd.concat([pure_df, mag_df], ignore_index=True)
    checkm2_comp = all_df['checkm2_completeness'].values.astype(float)
    checkm2_cont = all_df['checkm2_contamination'].values.astype(float)
    magicc_comp = all_df['magicc_completeness'].values.astype(float)
    magicc_cont = all_df['magicc_contamination'].values.astype(float)

    checkm2_hq = (checkm2_comp >= 90) & (checkm2_cont <= 5)
    magicc_hq = (magicc_comp >= 90) & (magicc_cont <= 5)
    both_hq = checkm2_hq & magicc_hq
    checkm2_only_hq = checkm2_hq & ~magicc_hq
    magicc_only_hq = ~checkm2_hq & magicc_hq
    neither_hq = ~checkm2_hq & ~magicc_hq

    checkm2_mq = (checkm2_comp >= 50) & (checkm2_cont <= 10)
    magicc_mq = (magicc_comp >= 50) & (magicc_cont <= 10)
    both_mq = checkm2_mq & magicc_mq
    checkm2_only_mq = checkm2_mq & ~magicc_mq
    magicc_only_mq = ~checkm2_mq & magicc_mq
    neither_mq = ~checkm2_mq & ~magicc_mq

    mimag_rows.append({
        'genome_type': 'Combined',
        'n_genomes': len(all_df),
        'checkm2_hq': int(checkm2_hq.sum()),
        'magicc_hq': int(magicc_hq.sum()),
        'both_hq': int(both_hq.sum()),
        'checkm2_only_hq': int(checkm2_only_hq.sum()),
        'magicc_only_hq': int(magicc_only_hq.sum()),
        'neither_hq': int(neither_hq.sum()),
        'hq_agreement_rate': round(
            (both_hq.sum() + neither_hq.sum()) / len(all_df) * 100, 2
        ),
        'checkm2_mq': int(checkm2_mq.sum()),
        'magicc_mq': int(magicc_mq.sum()),
        'both_mq': int(both_mq.sum()),
        'checkm2_only_mq': int(checkm2_only_mq.sum()),
        'magicc_only_mq': int(magicc_only_mq.sum()),
        'neither_mq': int(neither_mq.sum()),
        'mq_agreement_rate': round(
            (both_mq.sum() + neither_mq.sum()) / len(all_df) * 100, 2
        ),
    })

    mimag_df = pd.DataFrame(mimag_rows)
    mimag_path = OUTPUT_DIR / 'mimag_analysis.tsv'
    mimag_df.to_csv(mimag_path, sep='\t', index=False)
    print(f"\n  Saved: {mimag_path}")

    # Print summary table
    print(f"\n  {'='*90}")
    print(f"  MIMAG CLASSIFICATION SUMMARY")
    print(f"  {'='*90}")
    print(f"  {'Category':<15} {'N':>5}  |  {'CkM2 HQ':>7} {'MAG HQ':>7} {'Agree':>7} {'CkM2only':>8} {'MAGonly':>7}  |  "
          f"{'CkM2 MQ':>7} {'MAG MQ':>7} {'Agree':>7} {'CkM2only':>8} {'MAGonly':>7}")
    print(f"  {'-'*90}")
    for _, row in mimag_df.iterrows():
        print(f"  {row['genome_type']:<15} {row['n_genomes']:>5}  |  "
              f"{row['checkm2_hq']:>7} {row['magicc_hq']:>7} {row['both_hq']:>7} "
              f"{row['checkm2_only_hq']:>8} {row['magicc_only_hq']:>7}  |  "
              f"{row['checkm2_mq']:>7} {row['magicc_mq']:>7} {row['both_mq']:>7} "
              f"{row['checkm2_only_mq']:>8} {row['magicc_only_mq']:>7}")
    print(f"  {'='*90}")

    return mimag_df


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("MAGICC V5 vs CheckM2: GTDB Pure Culture vs MAG/SAG Comparison")
    print(f"Model: {ONNX_MODEL_PATH}")
    print(f"Workers: {N_WORKERS}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and sample metadata
    pure_sample, mag_sample = load_and_sample_metadata()

    # Step 2: Locate FASTA files
    pure_sample = locate_fasta_files(pure_sample, 'pure culture')
    mag_sample = locate_fasta_files(mag_sample, 'MAG/SAG')

    print(f"\n  Final sample sizes after FASTA validation:")
    print(f"    Pure culture: {len(pure_sample)}")
    print(f"    MAG/SAG:      {len(mag_sample)}")

    # Load ONNX model and normalizer
    print(f"\n  Loading ONNX model...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ONNX_THREADS
    sess_options.inter_op_num_threads = ONNX_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options,
                                   providers=['CPUExecutionProvider'])
    print(f"    Inputs: {[(i.name, i.shape) for i in session.get_inputs()]}")
    print(f"    Outputs: {[(o.name, o.shape) for o in session.get_outputs()]}")

    print(f"\n  Loading normalizer...")
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)
    print(f"    K-mer features: {normalizer.n_kmer_features}")
    print(f"    Assembly features: {normalizer.n_assembly_features}")

    # Step 3: Run MAGICC V5
    pure_results = run_magicc_v5(pure_sample, 'Pure Culture', session, normalizer,
                                 'pure_culture')
    mag_results = run_magicc_v5(mag_sample, 'MAG/SAG', session, normalizer,
                                'mag_sag')

    # Step 4: Compare predictions
    summary_df = compare_predictions(pure_results, mag_results)

    # Step 5: MIMAG analysis
    mimag_df = mimag_analysis(pure_results, mag_results)

    # ========================================================================
    # Final Summary
    # ========================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Sample sizes:")
    print(f"    Pure culture: {len(pure_results)}")
    print(f"    MAG/SAG:      {len(mag_results)}")
    print(f"    Total:        {len(pure_results) + len(mag_results)}")

    print(f"\n  MAE Comparison (MAGICC V5 vs CheckM2):")
    print(f"  {'Category':<15} {'Comp MAE':>10} {'Cont MAE':>10} {'Comp R2':>10} {'Cont R2':>10}")
    print(f"  {'-'*55}")
    for _, row in summary_df.iterrows():
        r2c = f"{row['comp_r2']:.4f}" if not pd.isna(row['comp_r2']) else 'N/A'
        r2x = f"{row['cont_r2']:.4f}" if not pd.isna(row['cont_r2']) else 'N/A'
        print(f"  {row['genome_type']:<15} {row['comp_mae']:>9.2f}% {row['cont_mae']:>9.2f}% "
              f"{r2c:>10} {r2x:>10}")

    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n  Output files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {f}")


if __name__ == '__main__':
    main()
