#!/usr/bin/env python3
"""
Script 60: Sample 1000 bacterial pure culture genomes + 1000 bacterial MAG genomes
from NCBI GenBank, run both CheckM2 and MAGICC V5, and compare results including
MIMAG misclassification analysis.

Steps:
  1. Parse GenBank assembly_summary_genbank.txt
  2. Filter and sample 1000 pure cultures (Complete Genome, Full, non-metagenome)
     and 1000 MAGs (derived from metagenome)
  3. Download genomes via FTP (parallel, resumable)
  4. Run MAGICC V5 inference
  5. Run CheckM2 (via subprocess, conda env checkm2_py39)
  6. Compare results and MIMAG analysis

Output: data/ncbi/ and results/ncbi_comparison/
"""

import sys
import os
import time
import signal
import json
import gzip
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

# --- Collapse-safe: ignore SIGHUP ---
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
NCBI_DIR = DATA_DIR / 'ncbi'
PURE_CULTURE_DIR = NCBI_DIR / 'pure_culture'
MAG_DIR = NCBI_DIR / 'mags'
RESULTS_DIR = PROJECT_DIR / 'results' / 'ncbi_comparison'

ASSEMBLY_SUMMARY = NCBI_DIR / 'assembly_summary_genbank.txt'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')

CHECKM2_DB = '/path/to/magicc/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd'
CHECKM2_ENV = 'checkm2_py39'

N_WORKERS = 43
BATCH_SIZE = 64
ONNX_THREADS = 1
DOWNLOAD_WORKERS = 20
SEED = 42
N_PURE_CULTURE = 1000
N_MAG = 1000


# ============================================================================
# Step 1-2: Parse and Sample
# ============================================================================
def parse_and_sample():
    """Parse assembly summary and sample 1000 pure cultures + 1000 MAGs."""
    print("=" * 70)
    print("STEP 1-2: Parse GenBank assembly summary and sample genomes")
    print("=" * 70)

    # Check for cached samples
    pure_cache = NCBI_DIR / 'sampled_pure_culture.tsv'
    mag_cache = NCBI_DIR / 'sampled_mags.tsv'
    if pure_cache.exists() and mag_cache.exists():
        pure_df = pd.read_csv(pure_cache, sep='\t')
        mag_df = pd.read_csv(mag_cache, sep='\t')
        print(f"  Loaded cached samples: {len(pure_df)} pure culture, {len(mag_df)} MAGs")
        return pure_df, mag_df

    # Parse header
    col_names = None
    with open(ASSEMBLY_SUMMARY, 'r') as f:
        for line in f:
            if line.startswith('#assembly_accession') or line.startswith('# assembly_accession'):
                col_names = line.lstrip('#').strip().split('\t')
                break

    if col_names is None:
        # Fallback - use known column order
        col_names = [
            'assembly_accession', 'bioproject', 'biosample', 'wgs_master',
            'refseq_category', 'taxid', 'species_taxid', 'organism_name',
            'infraspecific_name', 'isolate', 'version_status', 'assembly_level',
            'release_type', 'genome_rep', 'seq_rel_date', 'asm_name',
            'asm_submitter', 'gbrs_paired_asm', 'paired_asm_comp', 'ftp_path',
            'excluded_from_refseq', 'relation_to_type_material', 'asm_not_live_date',
            'assembly_type', 'group', 'genome_size', 'genome_size_ungapped',
            'gc_percent', 'replicon_count', 'scaffold_count', 'contig_count',
            'annotation_provider', 'annotation_name', 'annotation_date',
            'total_gene_count', 'protein_coding_gene_count', 'non_coding_gene_count',
            'pubmed_id',
        ]

    print(f"  Columns found: {len(col_names)}")

    # Parse rows
    pure_culture_rows = []
    mag_rows = []
    total_rows = 0

    with open(ASSEMBLY_SUMMARY, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < len(col_names):
                # Pad with empty strings
                cols.extend([''] * (len(col_names) - len(cols)))

            row = dict(zip(col_names, cols))
            total_rows += 1

            group = row.get('group', '')
            if group != 'bacteria':
                continue

            assembly_level = row.get('assembly_level', '')
            genome_rep = row.get('genome_rep', '')
            ftp_path = row.get('ftp_path', '')
            excluded = row.get('excluded_from_refseq', '')

            # Pure culture: Complete Genome, Full, valid FTP, not from metagenome
            if (assembly_level == 'Complete Genome' and
                genome_rep == 'Full' and
                ftp_path != 'na' and ftp_path != '' and
                'metagenome' not in excluded.lower()):
                pure_culture_rows.append(row)

            # MAG: excluded_from_refseq contains "derived from metagenome"
            if 'derived from metagenome' in excluded.lower():
                # Also need valid FTP path
                if ftp_path != 'na' and ftp_path != '':
                    mag_rows.append(row)

    print(f"  Total rows parsed: {total_rows:,}")
    print(f"  Pure culture pool (Complete Genome, Full, non-metagenome, bacteria): {len(pure_culture_rows):,}")
    print(f"  MAG pool (derived from metagenome, bacteria): {len(mag_rows):,}")

    # Convert to DataFrames
    pure_pool = pd.DataFrame(pure_culture_rows)
    mag_pool = pd.DataFrame(mag_rows)

    # Sample
    rng = np.random.RandomState(SEED)
    pure_sample = pure_pool.sample(n=N_PURE_CULTURE, random_state=rng).reset_index(drop=True)
    mag_sample = mag_pool.sample(n=N_MAG, random_state=rng).reset_index(drop=True)

    print(f"\n  Sampled {len(pure_sample)} pure culture genomes")
    print(f"  Sampled {len(mag_sample)} MAG genomes")

    # Build FTP download URLs
    for df in [pure_sample, mag_sample]:
        # ftp_path looks like: https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/005/825/GCA_000005825.2_ASM582v2/
        # Genome FASTA: <ftp_path>/<basename>_genomic.fna.gz
        df['fna_url'] = df.apply(
            lambda r: r['ftp_path'].rstrip('/') + '/' + r['ftp_path'].rstrip('/').split('/')[-1] + '_genomic.fna.gz',
            axis=1
        )

    # Save
    NCBI_DIR.mkdir(parents=True, exist_ok=True)
    pure_sample.to_csv(pure_cache, sep='\t', index=False)
    mag_sample.to_csv(mag_cache, sep='\t', index=False)
    print(f"  Saved: {pure_cache}")
    print(f"  Saved: {mag_cache}")

    return pure_sample, mag_sample


# ============================================================================
# Step 3: Download genomes
# ============================================================================
def download_one_genome(args):
    """Download a single genome FNA file."""
    accession, url, output_path = args
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return accession, 'cached', None

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            tmp_gz = output_path + '.gz.tmp'
            urllib.request.urlretrieve(url, tmp_gz)

            # Verify it's a valid gzip
            with gzip.open(tmp_gz, 'rb') as gz:
                data = gz.read()

            if len(data) < 100:
                os.remove(tmp_gz)
                return accession, 'error', 'file too small after decompression'

            # Write decompressed
            with open(output_path, 'wb') as f:
                f.write(data)
            os.remove(tmp_gz)
            return accession, 'ok', None

        except Exception as e:
            if os.path.exists(tmp_gz):
                try:
                    os.remove(tmp_gz)
                except:
                    pass
            if attempt < max_retries:
                time.sleep(attempt * 2)
                continue
            return accession, 'error', str(e)[:200]


def download_genomes(pure_df, mag_df):
    """Download all genomes with parallel workers."""
    print("\n" + "=" * 70)
    print("STEP 3: Downloading genomes from NCBI FTP")
    print("=" * 70)

    PURE_CULTURE_DIR.mkdir(parents=True, exist_ok=True)
    MAG_DIR.mkdir(parents=True, exist_ok=True)

    # Build work items
    work_items = []
    for _, row in pure_df.iterrows():
        acc = row['assembly_accession']
        url = row['fna_url']
        path = str(PURE_CULTURE_DIR / f'{acc}.fna')
        work_items.append((acc, url, path))

    for _, row in mag_df.iterrows():
        acc = row['assembly_accession']
        url = row['fna_url']
        path = str(MAG_DIR / f'{acc}.fna')
        work_items.append((acc, url, path))

    # Count already cached
    cached = sum(1 for _, _, p in work_items if os.path.exists(p) and os.path.getsize(p) > 0)
    print(f"  Total to download: {len(work_items)}")
    print(f"  Already cached: {cached}")
    print(f"  Remaining: {len(work_items) - cached}")
    print(f"  Workers: {DOWNLOAD_WORKERS}")

    t0 = time.time()
    results = {'ok': 0, 'cached': 0, 'error': 0}
    errors = []
    done = 0

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_one_genome, item): item[0] for item in work_items}
        for future in as_completed(futures):
            acc, status, err = future.result()
            results[status] = results.get(status, 0) + 1
            if err:
                errors.append((acc, err))
            done += 1
            if done % 100 == 0 or done == len(work_items):
                elapsed = time.time() - t0
                print(f"  {done}/{len(work_items)} ({elapsed:.0f}s) - ok:{results['ok']} cached:{results['cached']} error:{results['error']}")

    elapsed = time.time() - t0
    print(f"\n  Download complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {results}")
    if errors:
        print(f"  First 10 errors:")
        for acc, err in errors[:10]:
            print(f"    {acc}: {err}")

    # Verify downloads
    pure_downloaded = sum(1 for _, row in pure_df.iterrows()
                          if os.path.exists(str(PURE_CULTURE_DIR / f"{row['assembly_accession']}.fna"))
                          and os.path.getsize(str(PURE_CULTURE_DIR / f"{row['assembly_accession']}.fna")) > 0)
    mag_downloaded = sum(1 for _, row in mag_df.iterrows()
                         if os.path.exists(str(MAG_DIR / f"{row['assembly_accession']}.fna"))
                         and os.path.getsize(str(MAG_DIR / f"{row['assembly_accession']}.fna")) > 0)
    print(f"\n  Pure culture downloaded: {pure_downloaded}/{N_PURE_CULTURE}")
    print(f"  MAG downloaded: {mag_downloaded}/{N_MAG}")

    return pure_downloaded, mag_downloaded, errors


# ============================================================================
# Step 4: Run MAGICC V5
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


_worker_kmer_counter = None

def _init_worker(kmers_path):
    """Initialize worker with KmerCounter."""
    global _worker_kmer_counter
    _worker_kmer_counter = KmerCounter(kmers_path)
    dummy = ["ACGTACGTACGTACGTACGT" * 50]
    _worker_kmer_counter.count_contigs(dummy)


def _extract_features_worker(args):
    """Worker: extract features for a single genome."""
    acc, fasta_path = args
    global _worker_kmer_counter

    try:
        contigs = read_fasta_contigs(fasta_path)
        if len(contigs) == 0:
            return acc, None, None, "no_contigs"

        kmer_counts = _worker_kmer_counter.count_contigs(contigs)
        log10_total = _worker_kmer_counter.total_kmer_count(kmer_counts)
        assembly_feats = compute_assembly_stats(log10_total, kmer_counts)
        return acc, kmer_counts.astype(np.float32), assembly_feats.astype(np.float32), None
    except Exception as e:
        return acc, None, None, str(e)


def run_magicc_v5(genome_dir: Path, output_path: Path, label: str):
    """Run MAGICC V5 on all .fna files in genome_dir."""
    print(f"\n  Running MAGICC V5 on {label}...")

    # Check for checkpoint
    if output_path.exists():
        existing = pd.read_csv(output_path, sep='\t')
        print(f"    Found existing predictions: {len(existing)} genomes, reusing.")
        return existing

    # Collect FASTA files
    fasta_files = sorted(genome_dir.glob('*.fna'))
    if len(fasta_files) == 0:
        print(f"    No .fna files found in {genome_dir}")
        return pd.DataFrame()

    print(f"    Found {len(fasta_files)} FASTA files")

    # Load model
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = ONNX_THREADS
    sess_options.inter_op_num_threads = ONNX_THREADS
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    # Load normalizer
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)

    # Build work items
    work_items = [(f.stem, str(f)) for f in fasta_files]

    # Parallel feature extraction
    t0 = time.time()
    all_results = {}
    errors = []

    with Pool(processes=N_WORKERS, initializer=_init_worker,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        done_count = 0
        for result in pool.imap_unordered(_extract_features_worker, work_items, chunksize=10):
            acc, kmer, asm, err = result
            done_count += 1
            if err:
                errors.append((acc, err))
            else:
                all_results[acc] = (kmer, asm)
            if done_count % 200 == 0 or done_count == len(work_items):
                elapsed = time.time() - t0
                print(f"      {done_count}/{len(work_items)} features extracted ({elapsed:.0f}s)")

    feat_time = time.time() - t0
    print(f"    Feature extraction: {feat_time:.1f}s ({len(all_results)} genomes, {len(errors)} errors)")

    if len(all_results) == 0:
        print(f"    No valid genomes!")
        return pd.DataFrame()

    # Stack and normalize
    valid_accs = sorted(all_results.keys())
    kmer_array = np.stack([all_results[a][0] for a in valid_accs])
    assembly_array = np.stack([all_results[a][1] for a in valid_accs])

    kmer_norm = normalizer.normalize_kmer(kmer_array).astype(np.float32)
    assembly_norm = normalizer.normalize_assembly(assembly_array).astype(np.float32)

    # ONNX inference
    t_infer = time.time()
    n_valid = len(valid_accs)
    predictions = np.zeros((n_valid, 2), dtype=np.float32)
    for batch_start in range(0, n_valid, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_valid)
        feed = {
            input_names[0]: kmer_norm[batch_start:batch_end],
            input_names[1]: assembly_norm[batch_start:batch_end],
        }
        result = session.run([output_name], feed)
        predictions[batch_start:batch_end] = result[0]

    infer_time = time.time() - t_infer
    total_time = feat_time + infer_time
    print(f"    Inference: {infer_time:.2f}s")
    print(f"    Total V5 time: {total_time:.1f}s ({n_valid / total_time * 60:.0f} genomes/min)")

    # Build predictions DataFrame
    pred_df = pd.DataFrame({
        'accession': valid_accs,
        'v5_completeness': predictions[:, 0],
        'v5_contamination': predictions[:, 1],
    })

    pred_df.to_csv(output_path, sep='\t', index=False)
    print(f"    Saved: {output_path}")

    return pred_df


def run_magicc_v5_all():
    """Run MAGICC V5 on both pure culture and MAG genomes."""
    print("\n" + "=" * 70)
    print("STEP 4: Running MAGICC V5 inference")
    print("=" * 70)

    pure_pred = run_magicc_v5(PURE_CULTURE_DIR,
                               NCBI_DIR / 'magicc_v5_pure_culture.tsv',
                               'pure culture')
    mag_pred = run_magicc_v5(MAG_DIR,
                              NCBI_DIR / 'magicc_v5_mags.tsv',
                              'MAGs')

    return pure_pred, mag_pred


# ============================================================================
# Step 5: Run CheckM2
# ============================================================================
def run_checkm2(genome_dir: Path, output_dir: Path, label: str):
    """Run CheckM2 on all .fna files in genome_dir."""
    print(f"\n  Running CheckM2 on {label}...")

    quality_report = output_dir / 'quality_report.tsv'
    if quality_report.exists():
        df = pd.read_csv(quality_report, sep='\t')
        print(f"    Found existing CheckM2 results: {len(df)} genomes")
        return df

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count input files
    fna_files = list(genome_dir.glob('*.fna'))
    print(f"    Input: {len(fna_files)} .fna files in {genome_dir}")

    if len(fna_files) == 0:
        print(f"    No .fna files found!")
        return pd.DataFrame()

    cmd = [
        'conda', 'run', '-n', CHECKM2_ENV,
        'env', f'CHECKM2DB={CHECKM2_DB}',
        'checkm2', 'predict',
        '--threads', str(N_WORKERS),
        '-x', '.fna',
        '--input', str(genome_dir),
        '--output-directory', str(output_dir),
        '--force',
    ]

    print(f"    Command: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)

    elapsed = time.time() - t0
    print(f"    CheckM2 completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        print(f"    WARNING: CheckM2 returned code {result.returncode}")
        print(f"    stderr: {result.stderr[:500]}")

    # Parse results
    if quality_report.exists():
        df = pd.read_csv(quality_report, sep='\t')
        print(f"    Results: {len(df)} genomes assessed")
        return df
    else:
        print(f"    ERROR: quality_report.tsv not found!")
        return pd.DataFrame()


def run_checkm2_all():
    """Run CheckM2 on both pure culture and MAG genomes."""
    print("\n" + "=" * 70)
    print("STEP 5: Running CheckM2")
    print("=" * 70)

    pure_checkm2 = run_checkm2(PURE_CULTURE_DIR,
                                NCBI_DIR / 'checkm2_pure_culture',
                                'pure culture')
    mag_checkm2 = run_checkm2(MAG_DIR,
                               NCBI_DIR / 'checkm2_mags',
                               'MAGs')

    return pure_checkm2, mag_checkm2


# ============================================================================
# Step 6: Compare results and MIMAG analysis
# ============================================================================
def classify_mimag(completeness: float, contamination: float) -> str:
    """Classify genome quality according to MIMAG tiers."""
    if completeness >= 90 and contamination <= 5:
        return 'HQ'
    elif completeness >= 50 and contamination <= 10:
        return 'MQ'
    else:
        return 'LQ'


def run_comparison(pure_df, mag_df, pure_v5, mag_v5, pure_checkm2, mag_checkm2):
    """Compare MAGICC V5 vs CheckM2 results and perform MIMAG analysis."""
    print("\n" + "=" * 70)
    print("STEP 6: Comparison and MIMAG Analysis")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_all = {}

    for gtype, sample_df, v5_df, checkm2_df in [
        ('pure_culture', pure_df, pure_v5, pure_checkm2),
        ('MAG', mag_df, mag_v5, mag_checkm2),
    ]:
        print(f"\n{'='*60}")
        print(f"  {gtype.upper()} Analysis")
        print(f"{'='*60}")

        if v5_df is None or len(v5_df) == 0:
            print(f"    No V5 predictions available")
            continue
        if checkm2_df is None or len(checkm2_df) == 0:
            print(f"    No CheckM2 predictions available")
            continue

        # Merge V5 predictions with sample metadata
        # V5 accession = file stem = assembly_accession (e.g., GCA_000005825.2)
        # CheckM2 'Name' column = file stem without extension
        v5_df = v5_df.copy()
        checkm2_df = checkm2_df.copy()

        # CheckM2 Name column
        checkm2_df['accession'] = checkm2_df['Name'].astype(str)
        checkm2_renamed = checkm2_df[['accession', 'Completeness', 'Contamination']].copy()
        checkm2_renamed.columns = ['accession', 'checkm2_completeness', 'checkm2_contamination']

        # Merge
        merged = v5_df.merge(checkm2_renamed, on='accession', how='inner')
        print(f"    Merged: {len(merged)} genomes (V5={len(v5_df)}, CheckM2={len(checkm2_df)})")

        if len(merged) == 0:
            print(f"    No overlap between V5 and CheckM2 results!")
            # Debug
            print(f"    V5 accessions sample: {v5_df['accession'].head().tolist()}")
            print(f"    CheckM2 accessions sample: {checkm2_renamed['accession'].head().tolist()}")
            continue

        # ---- Distribution stats ----
        print(f"\n    MAGICC V5 predictions:")
        print(f"      Completeness:  mean={merged['v5_completeness'].mean():.2f}, "
              f"median={merged['v5_completeness'].median():.2f}, "
              f"std={merged['v5_completeness'].std():.2f}, "
              f"min={merged['v5_completeness'].min():.2f}, "
              f"max={merged['v5_completeness'].max():.2f}")
        print(f"      Contamination: mean={merged['v5_contamination'].mean():.2f}, "
              f"median={merged['v5_contamination'].median():.2f}, "
              f"std={merged['v5_contamination'].std():.2f}, "
              f"min={merged['v5_contamination'].min():.2f}, "
              f"max={merged['v5_contamination'].max():.2f}")

        print(f"\n    CheckM2 predictions:")
        print(f"      Completeness:  mean={merged['checkm2_completeness'].mean():.2f}, "
              f"median={merged['checkm2_completeness'].median():.2f}, "
              f"std={merged['checkm2_completeness'].std():.2f}, "
              f"min={merged['checkm2_completeness'].min():.2f}, "
              f"max={merged['checkm2_completeness'].max():.2f}")
        print(f"      Contamination: mean={merged['checkm2_contamination'].mean():.2f}, "
              f"median={merged['checkm2_contamination'].median():.2f}, "
              f"std={merged['checkm2_contamination'].std():.2f}, "
              f"min={merged['checkm2_contamination'].min():.2f}, "
              f"max={merged['checkm2_contamination'].max():.2f}")

        # ---- Agreement metrics (V5 vs CheckM2) ----
        comp_ae = np.abs(merged['v5_completeness'].values - merged['checkm2_completeness'].values)
        cont_ae = np.abs(merged['v5_contamination'].values - merged['checkm2_contamination'].values)
        comp_mae = np.mean(comp_ae)
        cont_mae = np.mean(cont_ae)
        comp_rmse = np.sqrt(np.mean(comp_ae ** 2))
        cont_rmse = np.sqrt(np.mean(cont_ae ** 2))

        v5c = merged['v5_completeness'].values
        ckc = merged['checkm2_completeness'].values
        v5x = merged['v5_contamination'].values
        ckx = merged['checkm2_contamination'].values

        if np.std(v5c) > 0 and np.std(ckc) > 0:
            comp_r = np.corrcoef(v5c, ckc)[0, 1]
        else:
            comp_r = float('nan')
        if np.std(v5x) > 0 and np.std(ckx) > 0:
            cont_r = np.corrcoef(v5x, ckx)[0, 1]
        else:
            cont_r = float('nan')

        print(f"\n    Agreement (V5 vs CheckM2):")
        print(f"      Completeness:  MAE={comp_mae:.2f}%, RMSE={comp_rmse:.2f}%, r={comp_r:.4f}")
        print(f"      Contamination: MAE={cont_mae:.2f}%, RMSE={cont_rmse:.2f}%, r={cont_r:.4f}")

        # ---- MIMAG classification ----
        merged['v5_mimag'] = merged.apply(
            lambda r: classify_mimag(r['v5_completeness'], r['v5_contamination']), axis=1)
        merged['checkm2_mimag'] = merged.apply(
            lambda r: classify_mimag(r['checkm2_completeness'], r['checkm2_contamination']), axis=1)

        v5_counts = merged['v5_mimag'].value_counts()
        ckm2_counts = merged['checkm2_mimag'].value_counts()

        print(f"\n    MIMAG Tier Distribution:")
        print(f"      {'Tier':<6} {'CheckM2':>10} {'MAGICC V5':>10}")
        for tier in ['HQ', 'MQ', 'LQ']:
            print(f"      {tier:<6} {ckm2_counts.get(tier, 0):>10} {v5_counts.get(tier, 0):>10}")

        # Confusion matrix
        tiers = ['HQ', 'MQ', 'LQ']
        confusion = pd.crosstab(merged['checkm2_mimag'], merged['v5_mimag'],
                                rownames=['CheckM2'], colnames=['V5'],
                                dropna=False)
        for t in tiers:
            if t not in confusion.columns:
                confusion[t] = 0
            if t not in confusion.index:
                confusion.loc[t] = 0
        confusion = confusion.reindex(index=tiers, columns=tiers, fill_value=0)

        print(f"\n    Confusion Matrix (rows=CheckM2, cols=V5):")
        print(f"      {'':>10} {'V5-HQ':>8} {'V5-MQ':>8} {'V5-LQ':>8}")
        for row_tier in tiers:
            vals = [confusion.loc[row_tier, col_tier] for col_tier in tiers]
            print(f"      CkM2-{row_tier:<4} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

        # Agreement rate
        agree = (merged['checkm2_mimag'] == merged['v5_mimag']).sum()
        agreement_pct = agree / len(merged) * 100
        print(f"\n    MIMAG Agreement: {agree}/{len(merged)} ({agreement_pct:.1f}%)")

        # HQ-specific analysis
        checkm2_hq = merged['checkm2_mimag'] == 'HQ'
        v5_hq = merged['v5_mimag'] == 'HQ'

        hq_both = (checkm2_hq & v5_hq).sum()
        hq_checkm2_only = (checkm2_hq & ~v5_hq).sum()
        hq_v5_only = (~checkm2_hq & v5_hq).sum()
        hq_neither = (~checkm2_hq & ~v5_hq).sum()

        print(f"\n    HQ Classification Analysis:")
        print(f"      Both HQ:         {hq_both}")
        print(f"      CheckM2 HQ only: {hq_checkm2_only} (CheckM2 says HQ, V5 does not)")
        print(f"      V5 HQ only:      {hq_v5_only} (V5 says HQ, CheckM2 does not)")
        print(f"      Neither HQ:      {hq_neither}")

        # Detail downgrades
        if hq_checkm2_only > 0:
            downgraded = merged[checkm2_hq & ~v5_hq]
            v5_tier_of_downgraded = downgraded['v5_mimag'].value_counts()
            print(f"      CheckM2-HQ downgraded by V5:")
            for tier, cnt in v5_tier_of_downgraded.items():
                print(f"        -> {tier}: {cnt}")
            comp_low = (downgraded['v5_completeness'] < 90).sum()
            cont_high = (downgraded['v5_contamination'] > 5).sum()
            print(f"      Reasons: V5 completeness<90: {comp_low}, V5 contamination>5: {cont_high}")

        if hq_v5_only > 0:
            upgraded = merged[~checkm2_hq & v5_hq]
            ckm2_tier_of_upgraded = upgraded['checkm2_mimag'].value_counts()
            print(f"      V5-HQ but CheckM2 disagrees:")
            for tier, cnt in ckm2_tier_of_upgraded.items():
                print(f"        CheckM2 {tier}: {cnt}")

        # Save per-genome results
        merged_out_path = RESULTS_DIR / f'{gtype}_comparison.tsv'
        merged.to_csv(merged_out_path, sep='\t', index=False)
        print(f"\n    Saved: {merged_out_path}")

        results_all[gtype] = {
            'n': len(merged),
            'v5_comp_mean': float(merged['v5_completeness'].mean()),
            'v5_cont_mean': float(merged['v5_contamination'].mean()),
            'checkm2_comp_mean': float(merged['checkm2_completeness'].mean()),
            'checkm2_cont_mean': float(merged['checkm2_contamination'].mean()),
            'comp_mae': float(comp_mae),
            'comp_rmse': float(comp_rmse),
            'comp_r': float(comp_r),
            'cont_mae': float(cont_mae),
            'cont_rmse': float(cont_rmse),
            'cont_r': float(cont_r),
            'agreement_n': int(agree),
            'agreement_pct': float(agreement_pct),
            'checkm2_hq': int(ckm2_counts.get('HQ', 0)),
            'checkm2_mq': int(ckm2_counts.get('MQ', 0)),
            'checkm2_lq': int(ckm2_counts.get('LQ', 0)),
            'v5_hq': int(v5_counts.get('HQ', 0)),
            'v5_mq': int(v5_counts.get('MQ', 0)),
            'v5_lq': int(v5_counts.get('LQ', 0)),
            'hq_both': int(hq_both),
            'hq_checkm2_only': int(hq_checkm2_only),
            'hq_v5_only': int(hq_v5_only),
            'hq_neither': int(hq_neither),
        }

    # ---- Combined summary table ----
    print(f"\n{'='*70}")
    print("COMBINED COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Pure Culture':>15} {'MAGs':>15}")
    print("-" * 67)
    for gtype_label, gtype_key in [('Pure Culture', 'pure_culture'), ('MAGs', 'MAG')]:
        if gtype_key not in results_all:
            continue
        r = results_all[gtype_key]
        print(f"  N genomes                         {r['n']:>15}")
        print(f"  CheckM2 Comp mean                 {r['checkm2_comp_mean']:>14.2f}%")
        print(f"  V5 Comp mean                      {r['v5_comp_mean']:>14.2f}%")
        print(f"  CheckM2 Cont mean                 {r['checkm2_cont_mean']:>14.2f}%")
        print(f"  V5 Cont mean                      {r['v5_cont_mean']:>14.2f}%")
        print(f"  Comp MAE (V5 vs CheckM2)          {r['comp_mae']:>14.2f}%")
        print(f"  Cont MAE (V5 vs CheckM2)          {r['cont_mae']:>14.2f}%")
        print(f"  Comp r                            {r['comp_r']:>14.4f}")
        print(f"  Cont r                            {r['cont_r']:>14.4f}")
        print(f"  MIMAG Agreement                   {r['agreement_pct']:>13.1f}%")
        print(f"  CheckM2 HQ / V5 HQ               {r['checkm2_hq']:>6} / {r['v5_hq']:>6}")
        print(f"  HQ both / CkM2-only / V5-only     {r['hq_both']:>5} / {r['hq_checkm2_only']:>5} / {r['hq_v5_only']:>5}")
        print()

    # Actually print the combined table properly
    if 'pure_culture' in results_all and 'MAG' in results_all:
        rp = results_all['pure_culture']
        rm = results_all['MAG']
        print(f"\n{'Metric':<40} {'Pure Culture':>14} {'MAGs':>14}")
        print("-" * 70)
        print(f"{'N genomes':<40} {rp['n']:>14} {rm['n']:>14}")
        print(f"{'CheckM2 mean completeness':<40} {rp['checkm2_comp_mean']:>13.2f}% {rm['checkm2_comp_mean']:>13.2f}%")
        print(f"{'V5 mean completeness':<40} {rp['v5_comp_mean']:>13.2f}% {rm['v5_comp_mean']:>13.2f}%")
        print(f"{'CheckM2 mean contamination':<40} {rp['checkm2_cont_mean']:>13.2f}% {rm['checkm2_cont_mean']:>13.2f}%")
        print(f"{'V5 mean contamination':<40} {rp['v5_cont_mean']:>13.2f}% {rm['v5_cont_mean']:>13.2f}%")
        print(f"{'Completeness MAE (V5 vs CheckM2)':<40} {rp['comp_mae']:>13.2f}% {rm['comp_mae']:>13.2f}%")
        print(f"{'Contamination MAE (V5 vs CheckM2)':<40} {rp['cont_mae']:>13.2f}% {rm['cont_mae']:>13.2f}%")
        print(f"{'Completeness r':<40} {rp['comp_r']:>14.4f} {rm['comp_r']:>14.4f}")
        print(f"{'Contamination r':<40} {rp['cont_r']:>14.4f} {rm['cont_r']:>14.4f}")
        print(f"{'MIMAG Agreement':<40} {rp['agreement_pct']:>13.1f}% {rm['agreement_pct']:>13.1f}%")
        print(f"{'CheckM2 HQ count':<40} {rp['checkm2_hq']:>14} {rm['checkm2_hq']:>14}")
        print(f"{'V5 HQ count':<40} {rp['v5_hq']:>14} {rm['v5_hq']:>14}")
        print(f"{'Both HQ':<40} {rp['hq_both']:>14} {rm['hq_both']:>14}")
        print(f"{'CheckM2 HQ only':<40} {rp['hq_checkm2_only']:>14} {rm['hq_checkm2_only']:>14}")
        print(f"{'V5 HQ only':<40} {rp['hq_v5_only']:>14} {rm['hq_v5_only']:>14}")

    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_pure_culture_pool': 73556,
        'n_mag_pool': 635598,
        'n_pure_sampled': N_PURE_CULTURE,
        'n_mag_sampled': N_MAG,
        'results': results_all,
    }
    summary_path = RESULTS_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved summary: {summary_path}")

    return results_all


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("NCBI GenBank Genome Comparison: MAGICC V5 vs CheckM2")
    print("1000 Pure Culture + 1000 MAG bacterial genomes")
    print("=" * 70)
    print()

    # Step 1-2: Parse and sample
    pure_df, mag_df = parse_and_sample()

    # Step 3: Download
    pure_dl, mag_dl, errors = download_genomes(pure_df, mag_df)

    # Step 4: MAGICC V5
    pure_v5, mag_v5 = run_magicc_v5_all()

    # Step 5: CheckM2
    pure_checkm2, mag_checkm2 = run_checkm2_all()

    # Step 6: Compare
    results = run_comparison(pure_df, mag_df, pure_v5, mag_v5, pure_checkm2, mag_checkm2)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL DONE in {elapsed:.0f}s ({elapsed/60:.1f} min, {elapsed/3600:.2f} hours)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
