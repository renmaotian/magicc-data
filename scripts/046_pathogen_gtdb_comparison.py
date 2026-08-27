#!/usr/bin/env python3
"""
Script 46: Pathogen GTDB Comparison - Part B
Compare CheckM2 vs MAGICC on real Salmonella enterica and Listeria monocytogenes
genomes from GTDB.

Steps:
  1. Select 1000 S. enterica + 1000 L. monocytogenes from GTDB metadata
  2. Download missing genomes from NCBI
  3. Run MAGICC on all 2000 genomes
  4. Compare MAGICC vs CheckM2 (GTDB metadata values)
  5. MIMAG misclassification analysis
"""

import os
import sys
import json
import gzip
import csv
import random
import subprocess
import shutil
import glob
import tempfile
import time
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = "/mnt/5c77b453-f7e1-48c8-afa3-5641857a41c7/tianrm/projects/magicc2"
GTDB_METADATA = os.path.join(PROJECT_DIR, "data/gtdb/bac120_metadata.tsv.gz")
GENOME_DIR = os.path.join(PROJECT_DIR, "data/genomes")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data/benchmarks/pathogen_analysis/gtdb_comparison")
MAGICC_INPUT_DIR = os.path.join(OUTPUT_DIR, "magicc_input")

SELECTED_GENOMES_TSV = os.path.join(OUTPUT_DIR, "selected_genomes.tsv")
MAGICC_PREDICTIONS_TSV = os.path.join(OUTPUT_DIR, "magicc_predictions.tsv")
COMPARISON_TABLE_TSV = os.path.join(OUTPUT_DIR, "comparison_table.tsv")
MIMAG_ANALYSIS_TSV = os.path.join(OUTPUT_DIR, "mimag_analysis.tsv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")

MAGICC_MODEL = os.path.join(PROJECT_DIR, "models/magicc_v3.onnx")
MAGICC_NORM = os.path.join(PROJECT_DIR, "data/features/normalization_params.json")
MAGICC_KMERS = os.path.join(PROJECT_DIR, "data/kmer_selection/selected_kmers.txt")

SEED = 42
N_PER_SPECIES = 1000
NUM_THREADS = 43
DOWNLOAD_WORKERS = 10
DOWNLOAD_BATCH_SIZE = 100

# MIMAG HQ thresholds
MIMAG_COMP_THRESHOLD = 90.0
MIMAG_CONT_THRESHOLD = 5.0

# Species patterns in GTDB taxonomy
SPECIES_PATTERNS = {
    "Salmonella_enterica": "s__Salmonella enterica",
    "Listeria_monocytogenes": "s__Listeria monocytogenes",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MAGICC_INPUT_DIR, exist_ok=True)


# ============================================================
# Step 1: Select genomes from GTDB metadata
# ============================================================
def step1_select_genomes():
    """Parse GTDB metadata, find pathogens, randomly select 1000 of each."""
    print("=" * 60)
    print("STEP 1: Selecting genomes from GTDB metadata")
    print("=" * 60)

    if os.path.exists(SELECTED_GENOMES_TSV):
        print(f"  [SKIP] {SELECTED_GENOMES_TSV} already exists")
        # Load and return
        selected = []
        with open(SELECTED_GENOMES_TSV) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                selected.append(row)
        species_counts = {}
        for row in selected:
            sp = row['species']
            species_counts[sp] = species_counts.get(sp, 0) + 1
        print(f"  Loaded {len(selected)} genomes: {species_counts}")
        return selected

    # Parse GTDB metadata
    species_genomes = {sp: [] for sp in SPECIES_PATTERNS}

    print(f"  Parsing {GTDB_METADATA}...")
    with gzip.open(GTDB_METADATA, 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            taxonomy = row.get('gtdb_taxonomy', '')
            for sp_key, sp_pattern in SPECIES_PATTERNS.items():
                if sp_pattern in taxonomy:
                    # Extract accession (strip RS_ or GB_ prefix)
                    acc = row['accession']
                    if acc.startswith('RS_') or acc.startswith('GB_'):
                        acc = acc[3:]

                    species_genomes[sp_key].append({
                        'accession': acc,
                        'gtdb_accession': row['accession'],
                        'species': sp_key,
                        'checkm2_completeness': float(row['checkm2_completeness']),
                        'checkm2_contamination': float(row['checkm2_contamination']),
                        'genome_size': int(row['genome_size']),
                        'contig_count': int(row['contig_count']),
                        'n50_contigs': int(row['n50_contigs']),
                        'gtdb_taxonomy': taxonomy,
                    })
                    break

    for sp_key, genomes in species_genomes.items():
        print(f"  {sp_key}: {len(genomes)} total genomes found")

    # Random selection
    rng = random.Random(SEED)
    selected = []
    for sp_key, genomes in species_genomes.items():
        n_select = min(N_PER_SPECIES, len(genomes))
        chosen = rng.sample(genomes, n_select)
        selected.extend(chosen)
        print(f"  Selected {n_select} {sp_key} genomes")

    # Save
    fieldnames = ['accession', 'gtdb_accession', 'species',
                  'checkm2_completeness', 'checkm2_contamination',
                  'genome_size', 'contig_count', 'n50_contigs', 'gtdb_taxonomy']
    with open(SELECTED_GENOMES_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(selected)

    print(f"  Saved {len(selected)} genomes to {SELECTED_GENOMES_TSV}")
    return selected


# ============================================================
# Step 2: Download missing genomes
# ============================================================
def step2_download_genomes(selected):
    """Download missing genomes using NCBI datasets CLI."""
    print("\n" + "=" * 60)
    print("STEP 2: Downloading missing genomes")
    print("=" * 60)

    accessions = [row['accession'] for row in selected]

    # Check which already exist
    existing = set()
    missing = []
    for acc in accessions:
        acc_dir = os.path.join(GENOME_DIR, acc)
        if os.path.isdir(acc_dir):
            # Check if there's a .fna file inside
            fna_files = glob.glob(os.path.join(acc_dir, "*.fna"))
            if fna_files:
                existing.add(acc)
            else:
                missing.append(acc)
        else:
            missing.append(acc)

    print(f"  Already downloaded: {len(existing)}")
    print(f"  Need to download:  {len(missing)}")

    if not missing:
        print("  [SKIP] All genomes already downloaded")
        return

    # Track download progress
    download_log = os.path.join(OUTPUT_DIR, "download_log.txt")
    done_file = os.path.join(OUTPUT_DIR, "download_done_accessions.txt")

    # Load previously completed downloads for this run
    done_accessions = set()
    if os.path.exists(done_file):
        with open(done_file) as f:
            done_accessions = set(line.strip() for line in f if line.strip())

    # Filter out already-done from missing
    still_missing = [acc for acc in missing if acc not in done_accessions]
    print(f"  Previously downloaded in this run: {len(done_accessions)}")
    print(f"  Still need to download: {len(still_missing)}")

    if not still_missing:
        print("  [SKIP] All downloads completed")
        return

    # Batch download
    batches = []
    for i in range(0, len(still_missing), DOWNLOAD_BATCH_SIZE):
        batches.append(still_missing[i:i + DOWNLOAD_BATCH_SIZE])

    print(f"  {len(batches)} batches of up to {DOWNLOAD_BATCH_SIZE}")

    failed_accessions = []
    total_downloaded = 0

    def download_batch(batch_idx, batch_accs):
        """Download a batch of genomes."""
        batch_name = f"pathogen_batch_{batch_idx:04d}"
        tmpzip = f"/tmp/ncbi_dl_{batch_name}_{os.getpid()}.zip"
        tmpextract = f"/tmp/ncbi_extract_{batch_name}_{os.getpid()}"

        # Write accession file
        acc_file = f"/tmp/{batch_name}_acc.txt"
        with open(acc_file, 'w') as f:
            f.write('\n'.join(batch_accs) + '\n')

        success = False
        moved = 0

        for retry in range(3):
            try:
                # Clean up any previous attempt
                if os.path.exists(tmpzip):
                    os.remove(tmpzip)
                if os.path.exists(tmpextract):
                    shutil.rmtree(tmpextract)

                result = subprocess.run(
                    ['conda', 'run', '-n', 'magicc2', 'datasets', 'download', 'genome', 'accession',
                     '--inputfile', acc_file,
                     '--include', 'genome',
                     '--filename', tmpzip,
                     '--no-progressbar'],
                    capture_output=True, text=True, timeout=600
                )

                if result.returncode != 0:
                    if retry < 2:
                        time.sleep((retry + 1) * 5)
                        continue
                    return batch_idx, [], batch_accs

                if not os.path.exists(tmpzip):
                    if retry < 2:
                        time.sleep((retry + 1) * 5)
                        continue
                    return batch_idx, [], batch_accs

                # Extract
                os.makedirs(tmpextract, exist_ok=True)
                subprocess.run(
                    ['unzip', '-q', '-o', tmpzip, '-d', tmpextract],
                    capture_output=True, timeout=120
                )

                data_dir = os.path.join(tmpextract, 'ncbi_dataset', 'data')
                downloaded_accs = []
                if os.path.isdir(data_dir):
                    for item in os.listdir(data_dir):
                        if item.startswith('GC'):
                            src = os.path.join(data_dir, item)
                            dst = os.path.join(GENOME_DIR, item)
                            if os.path.isdir(src):
                                if not os.path.exists(dst):
                                    shutil.move(src, dst)
                                downloaded_accs.append(item)
                                moved += 1

                success = True
                failed = [a for a in batch_accs if a not in downloaded_accs]
                return batch_idx, downloaded_accs, failed

            except subprocess.TimeoutExpired:
                if retry < 2:
                    time.sleep((retry + 1) * 5)
                    continue
                return batch_idx, [], batch_accs
            except Exception as e:
                if retry < 2:
                    time.sleep((retry + 1) * 5)
                    continue
                return batch_idx, [], batch_accs
            finally:
                # Cleanup
                if os.path.exists(tmpzip):
                    os.remove(tmpzip)
                if os.path.exists(tmpextract):
                    shutil.rmtree(tmpextract, ignore_errors=True)
                if os.path.exists(acc_file):
                    os.remove(acc_file)

        return batch_idx, [], batch_accs

    # Run downloads with thread pool
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {}
        for idx, batch in enumerate(batches):
            fut = executor.submit(download_batch, idx, batch)
            futures[fut] = idx

        for fut in as_completed(futures):
            batch_idx, downloaded, failed = fut.result()
            total_downloaded += len(downloaded)

            # Record done accessions
            with open(done_file, 'a') as f:
                for acc in downloaded:
                    f.write(acc + '\n')

            if failed:
                failed_accessions.extend(failed)

            completed = sum(1 for f in futures if f.done())
            print(f"  Batch {batch_idx}: downloaded {len(downloaded)}, "
                  f"failed {len(failed)} | "
                  f"Progress: {completed}/{len(batches)} batches, "
                  f"{total_downloaded} genomes total")

    print(f"\n  Download complete: {total_downloaded} new genomes")
    if failed_accessions:
        print(f"  Failed downloads: {len(failed_accessions)}")
        fail_file = os.path.join(OUTPUT_DIR, "failed_downloads.txt")
        with open(fail_file, 'w') as f:
            f.write('\n'.join(failed_accessions) + '\n')
        print(f"  Failed accessions saved to {fail_file}")

    return failed_accessions


# ============================================================
# Step 3: Prepare FASTA files and run MAGICC
# ============================================================
def step3_run_magicc(selected):
    """Symlink/copy genome FASTA files and run MAGICC."""
    print("\n" + "=" * 60)
    print("STEP 3: Running MAGICC on all genomes")
    print("=" * 60)

    if os.path.exists(MAGICC_PREDICTIONS_TSV):
        # Check if predictions cover all (or most) selected genomes
        existing_preds = set()
        with open(MAGICC_PREDICTIONS_TSV) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                existing_preds.add(row.get('genome_name', row.get('genome', row.get('file', ''))))
        print(f"  [CHECK] Existing predictions: {len(existing_preds)}")
        if len(existing_preds) >= len(selected) * 0.9:
            print(f"  [SKIP] Already have predictions for most genomes")
            return

    # Prepare input directory: symlink .fna files as .fasta
    print("  Preparing MAGICC input directory...")

    # Clean existing symlinks
    for f in glob.glob(os.path.join(MAGICC_INPUT_DIR, "*.fasta")):
        os.remove(f)

    prepared = 0
    skipped = 0
    missing_accs = []
    for row in selected:
        acc = row['accession']
        acc_dir = os.path.join(GENOME_DIR, acc)
        if not os.path.isdir(acc_dir):
            skipped += 1
            missing_accs.append(acc)
            continue

        # Find .fna file
        fna_files = glob.glob(os.path.join(acc_dir, "*.fna"))
        if not fna_files:
            skipped += 1
            missing_accs.append(acc)
            continue

        # Use the first (usually only) .fna file
        src = fna_files[0]
        dst = os.path.join(MAGICC_INPUT_DIR, f"{acc}.fasta")

        # Create symlink
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        prepared += 1

    print(f"  Prepared: {prepared} genomes")
    print(f"  Missing:  {skipped} genomes")

    if missing_accs:
        missing_file = os.path.join(OUTPUT_DIR, "missing_for_magicc.txt")
        with open(missing_file, 'w') as f:
            f.write('\n'.join(missing_accs) + '\n')
        print(f"  Missing accessions saved to {missing_file}")

    if prepared == 0:
        print("  ERROR: No genomes prepared for MAGICC")
        return

    # Run MAGICC
    print(f"\n  Running MAGICC on {prepared} genomes with {NUM_THREADS} threads...")
    cmd = [
        'conda', 'run', '-n', 'magicc2',
        'python', '-m', 'magicc', 'predict',
        '--input', MAGICC_INPUT_DIR,
        '--output', MAGICC_PREDICTIONS_TSV,
        '--threads', str(NUM_THREADS),
        '--model', MAGICC_MODEL,
        '--normalization', MAGICC_NORM,
        '--kmers', MAGICC_KMERS,
        '--extension', '.fasta',
    ]

    print(f"  Command: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  MAGICC STDERR:\n{result.stderr}")
        print(f"  MAGICC STDOUT:\n{result.stdout}")
        raise RuntimeError(f"MAGICC failed with return code {result.returncode}")

    print(f"  MAGICC completed in {elapsed:.1f}s")

    # Count predictions
    with open(MAGICC_PREDICTIONS_TSV) as f:
        n_preds = sum(1 for _ in f) - 1  # minus header
    print(f"  Predictions: {n_preds}")


# ============================================================
# Step 4: Compare results
# ============================================================
def step4_compare(selected):
    """Create comparison table."""
    print("\n" + "=" * 60)
    print("STEP 4: Comparing MAGICC vs CheckM2")
    print("=" * 60)

    # Load MAGICC predictions
    magicc_results = {}
    with open(MAGICC_PREDICTIONS_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        magicc_cols = reader.fieldnames
        print(f"  MAGICC columns: {magicc_cols}")
        for row in reader:
            # Extract accession from genome_name/genome/file column
            genome_name = row.get('genome_name', row.get('genome', row.get('file', '')))
            # Remove .fasta extension if present
            acc = genome_name.replace('.fasta', '').replace('.fna', '')
            # Also try just the basename
            acc = os.path.basename(acc)
            magicc_results[acc] = row

    print(f"  Loaded {len(magicc_results)} MAGICC predictions")

    # Build comparison table
    comparison = []
    matched = 0
    unmatched = 0

    for row in selected:
        acc = row['accession']
        magicc = magicc_results.get(acc, None)

        if magicc is None:
            unmatched += 1
            continue

        matched += 1

        # Get MAGICC completeness and contamination
        # Column names may vary - check common patterns
        magicc_comp = None
        magicc_cont = None
        for key in ['pred_completeness', 'completeness', 'predicted_completeness', 'Completeness']:
            if key in magicc:
                magicc_comp = float(magicc[key])
                break
        for key in ['pred_contamination', 'contamination', 'predicted_contamination', 'Contamination']:
            if key in magicc:
                magicc_cont = float(magicc[key])
                break

        if magicc_comp is None or magicc_cont is None:
            print(f"  WARNING: Could not find completeness/contamination for {acc}")
            print(f"  Available keys: {list(magicc.keys())}")
            unmatched += 1
            continue

        comparison.append({
            'accession': acc,
            'species': row['species'],
            'checkm2_completeness': float(row['checkm2_completeness']),
            'checkm2_contamination': float(row['checkm2_contamination']),
            'magicc_completeness': magicc_comp,
            'magicc_contamination': magicc_cont,
            'genome_size': row['genome_size'],
            'contig_count': row['contig_count'],
        })

    print(f"  Matched: {matched}, Unmatched: {unmatched}")

    # Save comparison table
    fieldnames = ['accession', 'species', 'checkm2_completeness', 'checkm2_contamination',
                  'magicc_completeness', 'magicc_contamination', 'genome_size', 'contig_count']
    with open(COMPARISON_TABLE_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(comparison)

    print(f"  Saved comparison table with {len(comparison)} rows to {COMPARISON_TABLE_TSV}")
    return comparison


# ============================================================
# Step 5: MIMAG misclassification analysis
# ============================================================
def step5_mimag_analysis(comparison):
    """MIMAG HQ classification analysis."""
    print("\n" + "=" * 60)
    print("STEP 5: MIMAG Misclassification Analysis")
    print("=" * 60)

    def is_mimag_hq(comp, cont):
        return comp >= MIMAG_COMP_THRESHOLD and cont <= MIMAG_CONT_THRESHOLD

    # Classify each genome
    mimag_data = []
    for row in comparison:
        checkm2_hq = is_mimag_hq(row['checkm2_completeness'], row['checkm2_contamination'])
        magicc_hq = is_mimag_hq(row['magicc_completeness'], row['magicc_contamination'])

        # Determine disagreement reason
        disagree_reason = ""
        if checkm2_hq != magicc_hq:
            comp_diff = abs(row['magicc_completeness'] - row['checkm2_completeness'])
            cont_diff = abs(row['magicc_contamination'] - row['checkm2_contamination'])

            # Check which metric caused the disagreement
            checkm2_comp_pass = row['checkm2_completeness'] >= MIMAG_COMP_THRESHOLD
            checkm2_cont_pass = row['checkm2_contamination'] <= MIMAG_CONT_THRESHOLD
            magicc_comp_pass = row['magicc_completeness'] >= MIMAG_COMP_THRESHOLD
            magicc_cont_pass = row['magicc_contamination'] <= MIMAG_CONT_THRESHOLD

            reasons = []
            if checkm2_comp_pass != magicc_comp_pass:
                reasons.append("completeness")
            if checkm2_cont_pass != magicc_cont_pass:
                reasons.append("contamination")
            disagree_reason = "+".join(reasons) if reasons else "threshold_edge"

        mimag_data.append({
            'accession': row['accession'],
            'species': row['species'],
            'checkm2_completeness': row['checkm2_completeness'],
            'checkm2_contamination': row['checkm2_contamination'],
            'magicc_completeness': row['magicc_completeness'],
            'magicc_contamination': row['magicc_contamination'],
            'checkm2_mimag_hq': checkm2_hq,
            'magicc_mimag_hq': magicc_hq,
            'agreement': checkm2_hq == magicc_hq,
            'disagree_reason': disagree_reason,
        })

    # Save MIMAG analysis
    fieldnames = ['accession', 'species', 'checkm2_completeness', 'checkm2_contamination',
                  'magicc_completeness', 'magicc_contamination',
                  'checkm2_mimag_hq', 'magicc_mimag_hq', 'agreement', 'disagree_reason']
    with open(MIMAG_ANALYSIS_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(mimag_data)

    print(f"  Saved MIMAG analysis to {MIMAG_ANALYSIS_TSV}")

    # ---- Compute statistics ----
    species_list = sorted(set(row['species'] for row in comparison))

    summary = {
        'total_genomes': len(comparison),
        'species_counts': {},
        'mimag_analysis': {},
        'accuracy_metrics': {},
    }

    for species in species_list:
        sp_data = [r for r in mimag_data if r['species'] == species]
        sp_comp = [r for r in comparison if r['species'] == species]
        n = len(sp_data)

        checkm2_hq_count = sum(1 for r in sp_data if r['checkm2_mimag_hq'])
        magicc_hq_count = sum(1 for r in sp_data if r['magicc_mimag_hq'])

        # Misclassification counts
        checkm2_hq_not_magicc = sum(1 for r in sp_data
                                      if r['checkm2_mimag_hq'] and not r['magicc_mimag_hq'])
        magicc_hq_not_checkm2 = sum(1 for r in sp_data
                                      if r['magicc_mimag_hq'] and not r['checkm2_mimag_hq'])
        agreement_count = sum(1 for r in sp_data if r['agreement'])

        # Disagreement reasons
        disagree_reasons = {}
        for r in sp_data:
            if not r['agreement'] and r['disagree_reason']:
                reason = r['disagree_reason']
                disagree_reasons[reason] = disagree_reasons.get(reason, 0) + 1

        # Accuracy metrics
        comp_checkm2 = np.array([r['checkm2_completeness'] for r in sp_comp])
        comp_magicc = np.array([r['magicc_completeness'] for r in sp_comp])
        cont_checkm2 = np.array([r['checkm2_contamination'] for r in sp_comp])
        cont_magicc = np.array([r['magicc_contamination'] for r in sp_comp])

        def calc_metrics(true_vals, pred_vals):
            diff = pred_vals - true_vals
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            ss_res = float(np.sum(diff ** 2))
            ss_tot = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff))
            return {
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'R2': round(r2, 4),
                'mean_diff': round(mean_diff, 4),
                'std_diff': round(std_diff, 4),
                'median_abs_diff': round(float(np.median(np.abs(diff))), 4),
            }

        comp_metrics = calc_metrics(comp_checkm2, comp_magicc)
        cont_metrics = calc_metrics(cont_checkm2, cont_magicc)

        summary['species_counts'][species] = n
        summary['mimag_analysis'][species] = {
            'checkm2_hq': checkm2_hq_count,
            'magicc_hq': magicc_hq_count,
            'checkm2_hq_not_magicc': checkm2_hq_not_magicc,
            'magicc_hq_not_checkm2': magicc_hq_not_checkm2,
            'agreement': agreement_count,
            'agreement_pct': round(100.0 * agreement_count / n, 2) if n > 0 else 0,
            'disagree_reasons': disagree_reasons,
        }
        summary['accuracy_metrics'][species] = {
            'completeness': comp_metrics,
            'contamination': cont_metrics,
        }

        # Print per-species results
        print(f"\n  --- {species} ({n} genomes) ---")
        print(f"  CheckM2 MIMAG HQ: {checkm2_hq_count} ({100*checkm2_hq_count/n:.1f}%)")
        print(f"  MAGICC  MIMAG HQ: {magicc_hq_count} ({100*magicc_hq_count/n:.1f}%)")
        print(f"  Agreement: {agreement_count}/{n} ({100*agreement_count/n:.1f}%)")
        print(f"  CheckM2 HQ but NOT MAGICC: {checkm2_hq_not_magicc}")
        print(f"  MAGICC HQ but NOT CheckM2: {magicc_hq_not_checkm2}")
        if disagree_reasons:
            print(f"  Disagreement reasons: {disagree_reasons}")
        print(f"  Completeness - MAE: {comp_metrics['MAE']}, RMSE: {comp_metrics['RMSE']}, R2: {comp_metrics['R2']}")
        print(f"  Contamination - MAE: {cont_metrics['MAE']}, RMSE: {cont_metrics['RMSE']}, R2: {cont_metrics['R2']}")

    # Overall stats
    all_checkm2_hq = sum(1 for r in mimag_data if r['checkm2_mimag_hq'])
    all_magicc_hq = sum(1 for r in mimag_data if r['magicc_mimag_hq'])
    all_checkm2_not_magicc = sum(1 for r in mimag_data
                                   if r['checkm2_mimag_hq'] and not r['magicc_mimag_hq'])
    all_magicc_not_checkm2 = sum(1 for r in mimag_data
                                   if r['magicc_mimag_hq'] and not r['checkm2_mimag_hq'])
    all_agreement = sum(1 for r in mimag_data if r['agreement'])

    all_comp_checkm2 = np.array([r['checkm2_completeness'] for r in comparison])
    all_comp_magicc = np.array([r['magicc_completeness'] for r in comparison])
    all_cont_checkm2 = np.array([r['checkm2_contamination'] for r in comparison])
    all_cont_magicc = np.array([r['magicc_contamination'] for r in comparison])

    def calc_metrics(true_vals, pred_vals):
        diff = pred_vals - true_vals
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))
        return {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'mean_diff': round(mean_diff, 4),
            'std_diff': round(std_diff, 4),
            'median_abs_diff': round(float(np.median(np.abs(diff))), 4),
        }

    summary['overall'] = {
        'total_genomes': len(comparison),
        'checkm2_hq': all_checkm2_hq,
        'magicc_hq': all_magicc_hq,
        'checkm2_hq_not_magicc': all_checkm2_not_magicc,
        'magicc_hq_not_checkm2': all_magicc_not_checkm2,
        'agreement': all_agreement,
        'agreement_pct': round(100.0 * all_agreement / len(comparison), 2),
        'completeness': calc_metrics(all_comp_checkm2, all_comp_magicc),
        'contamination': calc_metrics(all_cont_checkm2, all_cont_magicc),
    }

    print(f"\n  --- OVERALL ({len(comparison)} genomes) ---")
    print(f"  CheckM2 MIMAG HQ: {all_checkm2_hq}")
    print(f"  MAGICC  MIMAG HQ: {all_magicc_hq}")
    print(f"  Agreement: {all_agreement}/{len(comparison)} ({summary['overall']['agreement_pct']}%)")
    print(f"  CheckM2 HQ but NOT MAGICC: {all_checkm2_not_magicc}")
    print(f"  MAGICC HQ but NOT CheckM2: {all_magicc_not_checkm2}")
    print(f"  Completeness - MAE: {summary['overall']['completeness']['MAE']}, "
          f"RMSE: {summary['overall']['completeness']['RMSE']}, "
          f"R2: {summary['overall']['completeness']['R2']}")
    print(f"  Contamination - MAE: {summary['overall']['contamination']['MAE']}, "
          f"RMSE: {summary['overall']['contamination']['RMSE']}, "
          f"R2: {summary['overall']['contamination']['R2']}")

    # Save summary
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary to {SUMMARY_JSON}")

    return summary


# ============================================================
# Main
# ============================================================
def main():
    print("Pathogen GTDB Comparison: CheckM2 vs MAGICC")
    print(f"Seed: {SEED}, N per species: {N_PER_SPECIES}")
    print()

    # Step 1
    selected = step1_select_genomes()

    # Step 2
    failed = step2_download_genomes(selected)

    # Step 3
    step3_run_magicc(selected)

    # Step 4
    comparison = step4_compare(selected)

    # Step 5
    summary = step5_mimag_analysis(comparison)

    print("\n" + "=" * 60)
    print("DONE - All steps completed")
    print("=" * 60)


if __name__ == '__main__':
    main()
