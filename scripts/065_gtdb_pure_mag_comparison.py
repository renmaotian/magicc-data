#!/usr/bin/env python3
"""
Script 60: GTDB Pure Culture vs MAG/SAG Comparison — MAGICC V5 vs CheckM2

Samples 1000 pure culture (isolate) genomes and 1000 MAG/SAG genomes from the
FULL GTDB metadata (bac120 + ar53), downloads any missing genomes, runs V5
inference, and compares predictions against CheckM2 metadata values.

Key analyses:
  - V5 vs CheckM2 agreement (MAE, median AE, Pearson r)
  - MIMAG classification concordance (HQ/MQ/LQ)
  - Misclassification: CheckM2 says HQ but V5 disagrees (and vice versa)
  - Breakdown by genome category (pure culture vs MAG/SAG)

Model: models/magicc_v5.onnx
"""

import sys
import os
import signal
import json
import time
import subprocess
import tempfile
import zipfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from multiprocessing import Pool
from scipy import stats

# Collapse-safe: ignore SIGHUP
signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path('/path/to/magicc-legacy')
sys.path.insert(0, str(PROJECT_DIR))

import onnxruntime as ort
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats
from magicc.normalization import FeatureNormalizer

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = PROJECT_DIR / 'data'
GENOME_DIR = DATA_DIR / 'genomes'
OUTPUT_DIR = DATA_DIR / 'benchmarks' / 'gtdb_pure_mag_v5'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')

BAC_METADATA = DATA_DIR / 'gtdb' / 'bac120_metadata.tsv.gz'
AR_METADATA = DATA_DIR / 'gtdb' / 'ar53_metadata.tsv.gz'

N_PURE = 1000
N_MAG_SAG = 1000
SEED = 42
N_WORKERS = 43
BATCH_SIZE = 64
ONNX_THREADS = 1
DOWNLOAD_BATCH_SIZE = 200


# ============================================================================
# Utility: FASTA reading
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


def find_fasta_for_accession(accession: str) -> Optional[str]:
    """Find the FASTA file for a given NCBI accession in the local genomes directory."""
    acc_dir = GENOME_DIR / accession
    if acc_dir.is_dir():
        # Look for .fna files
        fna_files = list(acc_dir.glob('*.fna'))
        if fna_files:
            return str(fna_files[0])
        # Also check for .fasta files
        fasta_files = list(acc_dir.glob('*.fasta'))
        if fasta_files:
            return str(fasta_files[0])
    return None


# ============================================================================
# Step 1: Load and sample genomes from GTDB metadata
# ============================================================================
def load_and_sample_genomes():
    """Load GTDB metadata and sample pure culture and MAG/SAG genomes."""
    checkpoint_pure = OUTPUT_DIR / 'pure_culture_genomes.tsv'
    checkpoint_mag = OUTPUT_DIR / 'mag_sag_genomes.tsv'

    if checkpoint_pure.exists() and checkpoint_mag.exists():
        print("Loading cached genome selections...")
        df_pure = pd.read_csv(checkpoint_pure, sep='\t')
        df_mag = pd.read_csv(checkpoint_mag, sep='\t')
        print(f"  Pure culture: {len(df_pure)} genomes")
        print(f"  MAG/SAG: {len(df_mag)} genomes")
        return df_pure, df_mag

    print("Loading GTDB metadata...")
    t0 = time.time()

    # Load both bacteria and archaea
    cols_needed = [
        'accession', 'checkm2_completeness', 'checkm2_contamination',
        'ncbi_genome_category', 'ncbi_assembly_level', 'gtdb_taxonomy',
        'genome_size', 'contig_count', 'gc_percentage',
    ]

    print(f"  Reading bacterial metadata: {BAC_METADATA}")
    df_bac = pd.read_csv(BAC_METADATA, sep='\t', usecols=cols_needed,
                         dtype={'checkm2_completeness': float, 'checkm2_contamination': float})
    df_bac['domain'] = 'Bacteria'
    print(f"    {len(df_bac)} bacterial genomes")

    print(f"  Reading archaeal metadata: {AR_METADATA}")
    df_ar = pd.read_csv(AR_METADATA, sep='\t', usecols=cols_needed,
                        dtype={'checkm2_completeness': float, 'checkm2_contamination': float})
    df_ar['domain'] = 'Archaea'
    print(f"    {len(df_ar)} archaeal genomes")

    df = pd.concat([df_bac, df_ar], ignore_index=True)
    print(f"  Total: {len(df)} genomes")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Drop rows with missing CheckM2 values
    before = len(df)
    df = df.dropna(subset=['checkm2_completeness', 'checkm2_contamination'])
    print(f"  After dropping missing CheckM2: {len(df)} ({before - len(df)} dropped)")

    # Strip GB_/RS_ prefix to get NCBI accession
    df['ncbi_accession'] = df['accession'].str.replace(r'^(RS_|GB_)', '', regex=True)

    # Categorize genomes
    # Pure culture / isolate: ncbi_genome_category == 'none'
    # MAG/SAG: ncbi_genome_category in ('derived from metagenome', 'derived from single cell')
    df['genome_source'] = 'other'
    df.loc[df['ncbi_genome_category'] == 'none', 'genome_source'] = 'pure_culture'
    df.loc[df['ncbi_genome_category'] == 'derived from metagenome', 'genome_source'] = 'MAG'
    df.loc[df['ncbi_genome_category'] == 'derived from single cell', 'genome_source'] = 'SAG'

    # Count by category
    print("\n  Genome source distribution:")
    for src, count in df['genome_source'].value_counts().items():
        print(f"    {src}: {count:,}")

    # Sample pure culture genomes
    pure_pool = df[df['genome_source'] == 'pure_culture']
    mag_sag_pool = df[df['genome_source'].isin(['MAG', 'SAG'])]

    print(f"\n  Pure culture pool: {len(pure_pool):,}")
    print(f"  MAG/SAG pool: {len(mag_sag_pool):,}")

    rng = np.random.RandomState(SEED)

    # Sample
    df_pure = pure_pool.sample(n=N_PURE, random_state=rng).reset_index(drop=True)
    df_mag = mag_sag_pool.sample(n=N_MAG_SAG, random_state=rng).reset_index(drop=True)

    print(f"\n  Sampled {len(df_pure)} pure culture genomes")
    print(f"  Sampled {len(df_mag)} MAG/SAG genomes")

    # Print summary stats
    for name, sdf in [('Pure Culture', df_pure), ('MAG/SAG', df_mag)]:
        print(f"\n  {name} CheckM2 summary:")
        print(f"    Completeness: mean={sdf['checkm2_completeness'].mean():.1f}, "
              f"median={sdf['checkm2_completeness'].median():.1f}, "
              f"min={sdf['checkm2_completeness'].min():.1f}, "
              f"max={sdf['checkm2_completeness'].max():.1f}")
        print(f"    Contamination: mean={sdf['checkm2_contamination'].mean():.1f}, "
              f"median={sdf['checkm2_contamination'].median():.1f}, "
              f"min={sdf['checkm2_contamination'].min():.1f}, "
              f"max={sdf['checkm2_contamination'].max():.1f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_pure.to_csv(checkpoint_pure, sep='\t', index=False)
    df_mag.to_csv(checkpoint_mag, sep='\t', index=False)
    print(f"\n  Saved: {checkpoint_pure}")
    print(f"  Saved: {checkpoint_mag}")

    return df_pure, df_mag


# ============================================================================
# Step 2: Download missing genomes
# ============================================================================
def download_missing_genomes(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Check which genomes are locally available, download missing ones in batches."""
    print(f"\n{'='*70}")
    print(f"  Checking local availability for {label} ({len(df)} genomes)")
    print(f"{'='*70}")

    # Check which are already available
    df = df.copy()
    df['fasta_path'] = None
    df['download_status'] = 'missing'

    for i, row in df.iterrows():
        fasta = find_fasta_for_accession(row['ncbi_accession'])
        if fasta:
            df.at[i, 'fasta_path'] = fasta
            df.at[i, 'download_status'] = 'local'

    n_local = (df['download_status'] == 'local').sum()
    n_missing = (df['download_status'] == 'missing').sum()
    print(f"  Already local: {n_local}")
    print(f"  Need to download: {n_missing}")

    if n_missing == 0:
        return df

    # Download missing genomes in batches
    missing_accessions = df.loc[df['download_status'] == 'missing', 'ncbi_accession'].tolist()

    # Checkpoint file for download progress
    dl_checkpoint = OUTPUT_DIR / f'{label}_download_progress.json'
    if dl_checkpoint.exists():
        with open(dl_checkpoint) as f:
            dl_progress = json.load(f)
        downloaded_set = set(dl_progress.get('downloaded', []))
        failed_set = set(dl_progress.get('failed', []))
        print(f"  Resuming: {len(downloaded_set)} already downloaded, {len(failed_set)} previously failed")
        # Re-check downloaded ones
        for acc in list(downloaded_set):
            fasta = find_fasta_for_accession(acc)
            if fasta:
                mask = df['ncbi_accession'] == acc
                df.loc[mask, 'fasta_path'] = fasta
                df.loc[mask, 'download_status'] = 'local'
        # Only download truly missing ones
        missing_accessions = [a for a in missing_accessions
                              if a not in downloaded_set and a not in failed_set]
        missing_accessions += [a for a in failed_set
                               if find_fasta_for_accession(a) is None]
    else:
        dl_progress = {'downloaded': [], 'failed': []}
        downloaded_set = set()
        failed_set = set()

    if not missing_accessions:
        print("  All missing genomes already handled.")
        return df

    print(f"  Downloading {len(missing_accessions)} genomes in batches of {DOWNLOAD_BATCH_SIZE}...")

    n_batches = (len(missing_accessions) + DOWNLOAD_BATCH_SIZE - 1) // DOWNLOAD_BATCH_SIZE
    total_downloaded = 0
    total_failed = 0

    for batch_idx in range(n_batches):
        start = batch_idx * DOWNLOAD_BATCH_SIZE
        end = min(start + DOWNLOAD_BATCH_SIZE, len(missing_accessions))
        batch_accessions = missing_accessions[start:end]

        print(f"\n  Batch {batch_idx+1}/{n_batches}: {len(batch_accessions)} accessions")

        # Write accession list to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            for acc in batch_accessions:
                tf.write(acc + '\n')
            acc_file = tf.name

        tmpzip = tempfile.mktemp(suffix='.zip')
        tmpextract = tempfile.mkdtemp()

        try:
            # Download using NCBI datasets
            cmd = [
                'datasets', 'download', 'genome', 'accession',
                '--inputfile', acc_file,
                '--include', 'genome',
                '--filename', tmpzip,
                '--no-progressbar',
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0 and os.path.exists(tmpzip):
                # Extract
                try:
                    with zipfile.ZipFile(tmpzip, 'r') as zf:
                        zf.extractall(tmpextract)

                    # Move genome directories
                    ncbi_data_dir = Path(tmpextract) / 'ncbi_dataset' / 'data'
                    if ncbi_data_dir.exists():
                        batch_downloaded = 0
                        for acc_dir in ncbi_data_dir.iterdir():
                            if acc_dir.is_dir() and acc_dir.name.startswith('GC'):
                                target_dir = GENOME_DIR / acc_dir.name
                                if not target_dir.exists():
                                    shutil.move(str(acc_dir), str(target_dir))
                                batch_downloaded += 1
                                downloaded_set.add(acc_dir.name)

                        total_downloaded += batch_downloaded
                        print(f"    Downloaded: {batch_downloaded} genomes")

                        # Check which ones from this batch now have FASTAs
                        for acc in batch_accessions:
                            fasta = find_fasta_for_accession(acc)
                            if fasta:
                                mask = df['ncbi_accession'] == acc
                                df.loc[mask, 'fasta_path'] = fasta
                                df.loc[mask, 'download_status'] = 'downloaded'
                            else:
                                failed_set.add(acc)
                                total_failed += 1
                    else:
                        print(f"    WARNING: No ncbi_dataset/data directory in zip")
                        for acc in batch_accessions:
                            failed_set.add(acc)
                        total_failed += len(batch_accessions)

                except zipfile.BadZipFile:
                    print(f"    WARNING: Bad zip file")
                    for acc in batch_accessions:
                        failed_set.add(acc)
                    total_failed += len(batch_accessions)
            else:
                stderr_preview = result.stderr[:500] if result.stderr else 'no stderr'
                print(f"    WARNING: Download failed (rc={result.returncode}): {stderr_preview}")
                for acc in batch_accessions:
                    failed_set.add(acc)
                total_failed += len(batch_accessions)

        except subprocess.TimeoutExpired:
            print(f"    WARNING: Download timed out")
            for acc in batch_accessions:
                failed_set.add(acc)
            total_failed += len(batch_accessions)

        finally:
            os.unlink(acc_file)
            if os.path.exists(tmpzip):
                os.unlink(tmpzip)
            if os.path.exists(tmpextract):
                shutil.rmtree(tmpextract)

        # Save checkpoint after each batch
        dl_progress['downloaded'] = list(downloaded_set)
        dl_progress['failed'] = list(failed_set)
        with open(dl_checkpoint, 'w') as f:
            json.dump(dl_progress, f)

    print(f"\n  Download summary for {label}:")
    print(f"    Total downloaded: {total_downloaded}")
    print(f"    Total failed: {total_failed}")
    print(f"    Total available: {(df['download_status'] != 'missing').sum()}")

    return df


# ============================================================================
# Step 3: Feature extraction workers
# ============================================================================
_worker_kmer_counter = None


def _init_worker(kmers_path):
    """Initialize worker with KmerCounter (Numba JIT warmup)."""
    global _worker_kmer_counter
    _worker_kmer_counter = KmerCounter(kmers_path)
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
# Step 3b: Run V5 inference
# ============================================================================
def run_v5_inference(df: pd.DataFrame, session, normalizer):
    """Run V5 inference on genomes with available FASTA paths."""
    valid_df = df[df['fasta_path'].notna()].copy()
    n_valid = len(valid_df)

    if n_valid == 0:
        print("  No valid genomes to process!")
        return df

    print(f"\n  Running V5 inference on {n_valid} genomes...")

    # Build work items
    work_items = []
    for i, row in valid_df.iterrows():
        work_items.append((i, row['fasta_path']))

    # Parallel feature extraction
    print(f"  Extracting features ({N_WORKERS} workers)...")
    t_feat_start = time.time()

    all_results = {}
    with Pool(processes=N_WORKERS, initializer=_init_worker,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        done = 0
        for result in pool.imap_unordered(_extract_features_worker, work_items, chunksize=10):
            idx, kmer, asm, err = result
            if err:
                pass  # silently skip errors
            else:
                all_results[idx] = (kmer, asm)
            done += 1
            if done % 200 == 0:
                print(f"    Processed {done}/{n_valid}...")

    t_feat_end = time.time()
    feat_time = t_feat_end - t_feat_start

    valid_indices = sorted(all_results.keys())
    n_features_ok = len(valid_indices)
    print(f"  Feature extraction: {feat_time:.1f}s ({n_features_ok} genomes, "
          f"{feat_time/max(1,n_features_ok)*1000:.1f} ms/genome)")

    if n_features_ok == 0:
        print("  No valid features extracted!")
        return df

    # Stack arrays
    kmer_array = np.stack([all_results[i][0] for i in valid_indices])
    assembly_array = np.stack([all_results[i][1] for i in valid_indices])

    # Normalize
    kmer_norm = normalizer.normalize_kmer(kmer_array).astype(np.float32)
    assembly_norm = normalizer.normalize_assembly(assembly_array).astype(np.float32)

    # ONNX inference
    print(f"  Running ONNX inference...")
    t_infer_start = time.time()

    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    predictions = np.zeros((n_features_ok, 2), dtype=np.float32)
    for batch_start in range(0, n_features_ok, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_features_ok)
        feed = {
            input_names[0]: kmer_norm[batch_start:batch_end],
            input_names[1]: assembly_norm[batch_start:batch_end],
        }
        result = session.run([output_name], feed)
        predictions[batch_start:batch_end] = result[0]

    t_infer_end = time.time()
    infer_time = t_infer_end - t_infer_start
    total_time = feat_time + infer_time
    print(f"  ONNX inference: {infer_time:.2f}s ({infer_time/max(1,n_features_ok)*1000:.2f} ms/genome)")
    print(f"  Total: {total_time:.1f}s ({n_features_ok / total_time * 60:.0f} genomes/min)")

    # Add predictions to dataframe
    df['v5_completeness'] = np.nan
    df['v5_contamination'] = np.nan
    for pos, idx in enumerate(valid_indices):
        df.at[idx, 'v5_completeness'] = float(predictions[pos, 0])
        df.at[idx, 'v5_contamination'] = float(predictions[pos, 1])

    return df


# ============================================================================
# MIMAG classification
# ============================================================================
def classify_mimag(completeness, contamination):
    """Classify a genome according to MIMAG standards."""
    if completeness >= 90 and contamination <= 5:
        return 'HQ'
    elif completeness >= 50 and contamination <= 10:
        return 'MQ'
    else:
        return 'LQ'


# ============================================================================
# Analysis
# ============================================================================
def analyze_results(df_pure, df_mag):
    """Compare V5 predictions with CheckM2 metadata values."""
    print(f"\n{'='*80}")
    print("ANALYSIS: MAGICC V5 vs CheckM2 (from GTDB metadata)")
    print(f"{'='*80}")

    all_results = {}

    for label, df in [('Pure Culture', df_pure), ('MAG/SAG', df_mag)]:
        # Only keep genomes with both V5 and CheckM2 values
        valid = df.dropna(subset=['v5_completeness', 'v5_contamination']).copy()
        n_valid = len(valid)

        print(f"\n{'─'*70}")
        print(f"  {label}: {n_valid} genomes with V5 predictions")
        print(f"{'─'*70}")

        if n_valid == 0:
            continue

        # Compute metrics
        comp_diff = valid['v5_completeness'].values - valid['checkm2_completeness'].values
        cont_diff = valid['v5_contamination'].values - valid['checkm2_contamination'].values

        comp_ae = np.abs(comp_diff)
        cont_ae = np.abs(cont_diff)

        # Agreement stats
        comp_mae = np.mean(comp_ae)
        comp_median_ae = np.median(comp_ae)
        cont_mae = np.mean(cont_ae)
        cont_median_ae = np.median(cont_ae)

        comp_r, comp_p = stats.pearsonr(valid['checkm2_completeness'], valid['v5_completeness'])
        cont_r, cont_p = stats.pearsonr(valid['checkm2_contamination'], valid['v5_contamination'])

        comp_rmse = np.sqrt(np.mean(comp_diff**2))
        cont_rmse = np.sqrt(np.mean(cont_diff**2))

        print(f"\n  COMPLETENESS:")
        print(f"    MAE:          {comp_mae:.2f}%")
        print(f"    Median AE:    {comp_median_ae:.2f}%")
        print(f"    RMSE:         {comp_rmse:.2f}%")
        print(f"    Pearson r:    {comp_r:.4f} (p={comp_p:.2e})")
        print(f"    Mean bias:    {np.mean(comp_diff):.2f}% (V5 - CheckM2)")

        print(f"\n  CONTAMINATION:")
        print(f"    MAE:          {cont_mae:.2f}%")
        print(f"    Median AE:    {cont_median_ae:.2f}%")
        print(f"    RMSE:         {cont_rmse:.2f}%")
        print(f"    Pearson r:    {cont_r:.4f} (p={cont_p:.2e})")
        print(f"    Mean bias:    {np.mean(cont_diff):.2f}% (V5 - CheckM2)")

        # MIMAG classification
        valid['mimag_checkm2'] = valid.apply(
            lambda r: classify_mimag(r['checkm2_completeness'], r['checkm2_contamination']), axis=1)
        valid['mimag_v5'] = valid.apply(
            lambda r: classify_mimag(r['v5_completeness'], r['v5_contamination']), axis=1)

        # Concordance matrix
        cats = ['HQ', 'MQ', 'LQ']
        print(f"\n  MIMAG CONCORDANCE MATRIX (rows=CheckM2, cols=V5):")
        print(f"  {'':>12} {'V5_HQ':>8} {'V5_MQ':>8} {'V5_LQ':>8} {'Total':>8}")
        print(f"  {'-'*48}")

        concordance_matrix = {}
        for ckm2_cat in cats:
            row_counts = {}
            for v5_cat in cats:
                count = ((valid['mimag_checkm2'] == ckm2_cat) & (valid['mimag_v5'] == v5_cat)).sum()
                row_counts[v5_cat] = int(count)
            concordance_matrix[ckm2_cat] = row_counts
            total = sum(row_counts.values())
            print(f"  {'CkM2_'+ckm2_cat:>12} {row_counts['HQ']:>8} {row_counts['MQ']:>8} "
                  f"{row_counts['LQ']:>8} {total:>8}")

        v5_totals = {cat: sum(concordance_matrix[r][cat] for r in cats) for cat in cats}
        print(f"  {'-'*48}")
        print(f"  {'Total':>12} {v5_totals['HQ']:>8} {v5_totals['MQ']:>8} "
              f"{v5_totals['LQ']:>8} {n_valid:>8}")

        # Agreement
        agree = (valid['mimag_checkm2'] == valid['mimag_v5']).sum()
        agree_pct = agree / n_valid * 100
        print(f"\n  MIMAG Agreement: {agree}/{n_valid} ({agree_pct:.1f}%)")

        # Misclassification analysis
        # CheckM2 says HQ but V5 does not
        ckm2_hq_v5_not = ((valid['mimag_checkm2'] == 'HQ') & (valid['mimag_v5'] != 'HQ'))
        n_ckm2_hq_v5_not = ckm2_hq_v5_not.sum()
        n_ckm2_hq = (valid['mimag_checkm2'] == 'HQ').sum()

        # V5 says HQ but CheckM2 does not
        v5_hq_ckm2_not = ((valid['mimag_v5'] == 'HQ') & (valid['mimag_checkm2'] != 'HQ'))
        n_v5_hq_ckm2_not = v5_hq_ckm2_not.sum()
        n_v5_hq = (valid['mimag_v5'] == 'HQ').sum()

        print(f"\n  MISCLASSIFICATION:")
        print(f"    CheckM2 HQ total: {n_ckm2_hq}")
        print(f"    V5 HQ total:      {n_v5_hq}")
        print(f"    CheckM2 HQ but V5 NOT HQ: {n_ckm2_hq_v5_not} "
              f"({n_ckm2_hq_v5_not/max(1,n_ckm2_hq)*100:.1f}% of CheckM2 HQ)")
        print(f"    V5 HQ but CheckM2 NOT HQ: {n_v5_hq_ckm2_not} "
              f"({n_v5_hq_ckm2_not/max(1,n_v5_hq)*100:.1f}% of V5 HQ)")

        # Detailed breakdown of CheckM2 HQ but V5 not HQ
        if n_ckm2_hq_v5_not > 0:
            misclass_df = valid[ckm2_hq_v5_not]
            print(f"\n    Breakdown of {n_ckm2_hq_v5_not} 'CheckM2 HQ, V5 not HQ':")
            for v5_cat in ['MQ', 'LQ']:
                n = (misclass_df['mimag_v5'] == v5_cat).sum()
                if n > 0:
                    subset = misclass_df[misclass_df['mimag_v5'] == v5_cat]
                    print(f"      V5={v5_cat}: {n}")
                    print(f"        V5 comp:  mean={subset['v5_completeness'].mean():.1f}, "
                          f"min={subset['v5_completeness'].min():.1f}, max={subset['v5_completeness'].max():.1f}")
                    print(f"        V5 cont:  mean={subset['v5_contamination'].mean():.1f}, "
                          f"min={subset['v5_contamination'].min():.1f}, max={subset['v5_contamination'].max():.1f}")
                    print(f"        CkM2 comp: mean={subset['checkm2_completeness'].mean():.1f}, "
                          f"min={subset['checkm2_completeness'].min():.1f}, max={subset['checkm2_completeness'].max():.1f}")
                    print(f"        CkM2 cont: mean={subset['checkm2_contamination'].mean():.1f}, "
                          f"min={subset['checkm2_contamination'].min():.1f}, max={subset['checkm2_contamination'].max():.1f}")

        # Save per-genome MIMAG analysis
        valid_out = valid[['accession', 'ncbi_accession', 'genome_source',
                           'checkm2_completeness', 'checkm2_contamination',
                           'v5_completeness', 'v5_contamination',
                           'mimag_checkm2', 'mimag_v5']].copy()
        valid_out['mimag_agree'] = valid_out['mimag_checkm2'] == valid_out['mimag_v5']

        label_key = label.lower().replace(' ', '_').replace('/', '_')
        all_results[label_key] = {
            'n_genomes': n_valid,
            'completeness': {
                'mae': round(comp_mae, 4),
                'median_ae': round(comp_median_ae, 4),
                'rmse': round(comp_rmse, 4),
                'pearson_r': round(comp_r, 4),
                'mean_bias': round(float(np.mean(comp_diff)), 4),
            },
            'contamination': {
                'mae': round(cont_mae, 4),
                'median_ae': round(cont_median_ae, 4),
                'rmse': round(cont_rmse, 4),
                'pearson_r': round(cont_r, 4),
                'mean_bias': round(float(np.mean(cont_diff)), 4),
            },
            'mimag': {
                'concordance_matrix': concordance_matrix,
                'agreement_rate': round(agree_pct, 2),
                'checkm2_hq_total': int(n_ckm2_hq),
                'v5_hq_total': int(n_v5_hq),
                'checkm2_hq_but_v5_not_hq': int(n_ckm2_hq_v5_not),
                'v5_hq_but_checkm2_not_hq': int(n_v5_hq_ckm2_not),
            },
        }

    return all_results


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("Script 60: GTDB Pure Culture vs MAG/SAG — MAGICC V5 vs CheckM2")
    print("=" * 80)
    print(f"Model: {ONNX_MODEL_PATH}")
    print(f"Workers: {N_WORKERS}")
    print(f"Samples: {N_PURE} pure culture + {N_MAG_SAG} MAG/SAG (seed={SEED})")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Sample genomes from GTDB metadata
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Sample genomes from GTDB metadata")
    print("=" * 70)
    df_pure, df_mag = load_and_sample_genomes()

    # ----------------------------------------------------------------
    # Step 2: Download missing genomes
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Download missing genomes")
    print("=" * 70)
    df_pure = download_missing_genomes(df_pure, 'pure_culture')
    df_mag = download_missing_genomes(df_mag, 'mag_sag')

    # Summary of availability
    n_pure_avail = df_pure['fasta_path'].notna().sum()
    n_mag_avail = df_mag['fasta_path'].notna().sum()
    print(f"\n  Available genomes:")
    print(f"    Pure culture: {n_pure_avail}/{len(df_pure)}")
    print(f"    MAG/SAG:      {n_mag_avail}/{len(df_mag)}")

    # ----------------------------------------------------------------
    # Step 3: Run V5 inference
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Run MAGICC V5 inference")
    print("=" * 70)

    # Load model
    print("Loading ONNX model...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ONNX_THREADS
    sess_options.inter_op_num_threads = ONNX_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    print(f"  Inputs: {[(i.name, i.shape) for i in session.get_inputs()]}")

    print("Loading normalizer...")
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)
    print(f"  K-mer features: {normalizer.n_kmer_features}")
    print(f"  Assembly features: {normalizer.n_assembly_features}")

    # Check for cached predictions
    pred_checkpoint = OUTPUT_DIR / 'v5_predictions.tsv'
    if pred_checkpoint.exists():
        print(f"\n  Loading cached V5 predictions from {pred_checkpoint}")
        pred_df = pd.read_csv(pred_checkpoint, sep='\t')

        # Merge predictions back
        pred_map = pred_df.set_index('ncbi_accession')[['v5_completeness', 'v5_contamination']].to_dict('index')

        df_pure['v5_completeness'] = np.nan
        df_pure['v5_contamination'] = np.nan
        df_mag['v5_completeness'] = np.nan
        df_mag['v5_contamination'] = np.nan

        for idx, row in df_pure.iterrows():
            if row['ncbi_accession'] in pred_map:
                df_pure.at[idx, 'v5_completeness'] = pred_map[row['ncbi_accession']]['v5_completeness']
                df_pure.at[idx, 'v5_contamination'] = pred_map[row['ncbi_accession']]['v5_contamination']

        for idx, row in df_mag.iterrows():
            if row['ncbi_accession'] in pred_map:
                df_mag.at[idx, 'v5_completeness'] = pred_map[row['ncbi_accession']]['v5_completeness']
                df_mag.at[idx, 'v5_contamination'] = pred_map[row['ncbi_accession']]['v5_contamination']

        n_pure_pred = df_pure['v5_completeness'].notna().sum()
        n_mag_pred = df_mag['v5_completeness'].notna().sum()
        print(f"  Pure culture with predictions: {n_pure_pred}")
        print(f"  MAG/SAG with predictions: {n_mag_pred}")

        # Check if any genomes with fasta but no prediction yet
        pure_need_pred = df_pure[df_pure['fasta_path'].notna() & df_pure['v5_completeness'].isna()]
        mag_need_pred = df_mag[df_mag['fasta_path'].notna() & df_mag['v5_completeness'].isna()]

        if len(pure_need_pred) > 0 or len(mag_need_pred) > 0:
            print(f"  Still need predictions: {len(pure_need_pred)} pure, {len(mag_need_pred)} MAG/SAG")
            if len(pure_need_pred) > 0:
                df_pure = run_v5_inference(df_pure, session, normalizer)
            if len(mag_need_pred) > 0:
                df_mag = run_v5_inference(df_mag, session, normalizer)
    else:
        # Run V5 inference
        print("\n  Processing pure culture genomes...")
        df_pure = run_v5_inference(df_pure, session, normalizer)

        print("\n  Processing MAG/SAG genomes...")
        df_mag = run_v5_inference(df_mag, session, normalizer)

    # Save predictions
    pred_rows = []
    for label, df in [('pure_culture', df_pure), ('mag_sag', df_mag)]:
        for _, row in df.iterrows():
            if pd.notna(row.get('v5_completeness')):
                pred_rows.append({
                    'accession': row['accession'],
                    'ncbi_accession': row['ncbi_accession'],
                    'genome_source': row['genome_source'],
                    'checkm2_completeness': row['checkm2_completeness'],
                    'checkm2_contamination': row['checkm2_contamination'],
                    'v5_completeness': row['v5_completeness'],
                    'v5_contamination': row['v5_contamination'],
                    'category': label,
                })

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(pred_checkpoint, sep='\t', index=False)
    print(f"\n  Saved V5 predictions: {pred_checkpoint} ({len(pred_df)} genomes)")

    # ----------------------------------------------------------------
    # Step 4: Analyze results
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Analysis")
    print("=" * 70)

    results = analyze_results(df_pure, df_mag)

    # Save per-genome MIMAG analysis
    mimag_rows = []
    for label, df in [('pure_culture', df_pure), ('mag_sag', df_mag)]:
        valid = df.dropna(subset=['v5_completeness', 'v5_contamination']).copy()
        if len(valid) == 0:
            continue
        valid['mimag_checkm2'] = valid.apply(
            lambda r: classify_mimag(r['checkm2_completeness'], r['checkm2_contamination']), axis=1)
        valid['mimag_v5'] = valid.apply(
            lambda r: classify_mimag(r['v5_completeness'], r['v5_contamination']), axis=1)
        valid['mimag_agree'] = valid['mimag_checkm2'] == valid['mimag_v5']
        valid['category'] = label

        for _, row in valid.iterrows():
            mimag_rows.append({
                'accession': row['accession'],
                'ncbi_accession': row['ncbi_accession'],
                'category': label,
                'genome_source': row['genome_source'],
                'checkm2_completeness': row['checkm2_completeness'],
                'checkm2_contamination': row['checkm2_contamination'],
                'v5_completeness': row['v5_completeness'],
                'v5_contamination': row['v5_contamination'],
                'mimag_checkm2': row['mimag_checkm2'],
                'mimag_v5': row['mimag_v5'],
                'mimag_agree': row['mimag_agree'],
            })

    mimag_df = pd.DataFrame(mimag_rows)
    mimag_path = OUTPUT_DIR / 'mimag_analysis.tsv'
    mimag_df.to_csv(mimag_path, sep='\t', index=False)
    print(f"\n  Saved MIMAG analysis: {mimag_path}")

    # Save summary JSON
    summary = {
        'config': {
            'n_pure_sampled': N_PURE,
            'n_mag_sag_sampled': N_MAG_SAG,
            'seed': SEED,
            'model': ONNX_MODEL_PATH,
        },
        'availability': {
            'pure_culture_available': int(n_pure_avail),
            'mag_sag_available': int(n_mag_avail),
            'pure_culture_with_predictions': int(df_pure['v5_completeness'].notna().sum()),
            'mag_sag_with_predictions': int(df_mag['v5_completeness'].notna().sum()),
        },
        'results': results,
    }
    summary_path = OUTPUT_DIR / 'comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")

    print(f"\nOutput files:")
    print(f"  {OUTPUT_DIR / 'pure_culture_genomes.tsv'}")
    print(f"  {OUTPUT_DIR / 'mag_sag_genomes.tsv'}")
    print(f"  {OUTPUT_DIR / 'v5_predictions.tsv'}")
    print(f"  {OUTPUT_DIR / 'mimag_analysis.tsv'}")
    print(f"  {OUTPUT_DIR / 'comparison_summary.json'}")


if __name__ == '__main__':
    main()
