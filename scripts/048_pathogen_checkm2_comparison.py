#!/usr/bin/env python3
"""
Script 45: Pathogen CheckM2 vs MAGICC Comparison Analysis

Evaluates whether CheckM2 misestimates contamination and completeness for three
important pathogen species (E. coli, Salmonella enterica, Listeria monocytogenes)
using MAGICC as an independent quality assessor.

Steps:
1. Extract pathogen metadata from GTDB bac120_metadata.tsv.gz
2. Define 4 CheckM2 contamination intervals and assign genomes
3. For each interval batch: download genomes -> run MAGICC -> save results -> cleanup
4. Aggregate results and compare MAGICC vs CheckM2

Usage:
    conda run -n magicc2 python scripts/45_pathogen_checkm2_comparison.py
"""

import gzip
import json
import os
import subprocess
import sys
import shutil
import glob
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/mnt/5c77b453-f7e1-48c8-afa3-5641857a41c7/tianrm/projects/magicc2")
GTDB_METADATA = PROJECT_ROOT / "data" / "gtdb" / "bac120_metadata.tsv.gz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pathogen_analysis"
MAGICC_MODEL = PROJECT_ROOT / "models" / "magicc_v3.onnx"
NORM_PARAMS = PROJECT_ROOT / "data" / "features" / "normalization_params.json"
KMERS_FILE = PROJECT_ROOT / "data" / "kmer_selection" / "selected_kmers.txt"

TARGET_SPECIES = [
    "s__Escherichia coli",
    "s__Salmonella enterica",
    "s__Listeria monocytogenes",
]

RANDOM_SEED = 42
INTERVAL_1_SAMPLE_SIZE = 3000
MAGICC_THREADS = 40
DOWNLOAD_BATCH_SIZE = 500  # sub-batch for NCBI downloads

# Contamination intervals
INTERVALS = {
    1: {"min": 0.0, "max": 1.0, "label": "0-1%", "sample": INTERVAL_1_SAMPLE_SIZE},
    2: {"min": 1.0, "max": 3.0, "label": "1-3%", "sample": None},  # ALL
    3: {"min": 3.0, "max": 5.0, "label": "3-5%", "sample": None},  # ALL
    4: {"min": 5.0, "max": float("inf"), "label": ">=5%", "sample": None},  # ALL
}


def log(msg):
    """Print timestamped log message."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Step 1: Extract pathogen metadata ───────────────────────────────────────

def step1_extract_metadata():
    """Extract metadata for target pathogen species from GTDB."""
    output_file = OUTPUT_DIR / "pathogen_metadata.tsv"

    if output_file.exists():
        log(f"Step 1: Metadata already exists at {output_file}, loading...")
        df = pd.read_csv(output_file, sep="\t")
        log(f"  Loaded {len(df)} genomes")
        for sp in TARGET_SPECIES:
            sp_short = sp.split("__")[1]
            count = (df["species"] == sp).sum()
            log(f"  {sp_short}: {count} genomes")
        return df

    log("Step 1: Extracting pathogen metadata from GTDB...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with gzip.open(GTDB_METADATA, "rt") as f:
        header = f.readline().strip().split("\t")
        # Verify column indices
        assert header[0] == "accession", f"Expected 'accession' at col 0, got '{header[0]}'"
        assert header[2] == "checkm2_completeness", f"Expected 'checkm2_completeness' at col 2, got '{header[2]}'"
        assert header[3] == "checkm2_contamination", f"Expected 'checkm2_contamination' at col 3, got '{header[3]}'"
        assert header[19] == "gtdb_taxonomy", f"Expected 'gtdb_taxonomy' at col 19, got '{header[19]}'"

        for line in f:
            fields = line.strip().split("\t")
            taxonomy = fields[19]

            matched_species = None
            for sp in TARGET_SPECIES:
                if sp in taxonomy:
                    matched_species = sp
                    break

            if matched_species is None:
                continue

            accession_raw = fields[0]
            # Strip RS_/GB_ prefix for NCBI download
            if accession_raw.startswith("RS_"):
                ncbi_accession = accession_raw[3:]
            elif accession_raw.startswith("GB_"):
                ncbi_accession = accession_raw[3:]
            else:
                ncbi_accession = accession_raw

            rows.append({
                "accession": ncbi_accession,
                "gtdb_accession": accession_raw,
                "species": matched_species,
                "checkm2_completeness": float(fields[2]),
                "checkm2_contamination": float(fields[3]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, sep="\t", index=False)
    log(f"  Saved {len(df)} genomes to {output_file}")

    for sp in TARGET_SPECIES:
        sp_short = sp.split("__")[1]
        count = (df["species"] == sp).sum()
        log(f"  {sp_short}: {count} genomes")

    return df


# ─── Step 2: Define contamination intervals ──────────────────────────────────

def step2_define_intervals(df):
    """Assign genomes to contamination intervals and save accession lists."""
    log("Step 2: Defining contamination intervals...")

    interval_dfs = {}
    total_to_process = 0

    for interval_id, params in INTERVALS.items():
        acc_file = OUTPUT_DIR / f"interval_{interval_id}_accessions.txt"

        # Filter by contamination range
        if params["max"] == float("inf"):
            mask = df["checkm2_contamination"] >= params["min"]
        else:
            mask = (df["checkm2_contamination"] >= params["min"]) & \
                   (df["checkm2_contamination"] < params["max"])

        interval_df = df[mask].copy()
        log(f"  Interval {interval_id} ({params['label']}): {len(interval_df)} genomes total")

        # Apply sampling if needed (stratified by species)
        if params["sample"] is not None and len(interval_df) > params["sample"]:
            # Stratified sampling by species, proportional
            species_counts = interval_df["species"].value_counts()
            species_fractions = species_counts / species_counts.sum()

            sampled_parts = []
            remaining = params["sample"]
            species_list = list(species_fractions.index)

            for i, sp in enumerate(species_list):
                sp_df = interval_df[interval_df["species"] == sp]
                if i == len(species_list) - 1:
                    n_sample = remaining
                else:
                    n_sample = int(round(species_fractions[sp] * params["sample"]))
                    n_sample = min(n_sample, len(sp_df))

                sampled = sp_df.sample(n=min(n_sample, len(sp_df)), random_state=RANDOM_SEED)
                sampled_parts.append(sampled)
                remaining -= len(sampled)

            interval_df = pd.concat(sampled_parts)
            log(f"    Sampled {len(interval_df)} genomes (stratified by species)")

        for sp in TARGET_SPECIES:
            sp_count = (interval_df["species"] == sp).sum()
            sp_short = sp.split("__")[1]
            log(f"    {sp_short}: {sp_count}")

        # Save accession list
        interval_df["accession"].to_csv(acc_file, index=False, header=False)
        log(f"    Saved accessions to {acc_file}")

        interval_dfs[interval_id] = interval_df
        total_to_process += len(interval_df)

    log(f"  Total genomes to process: {total_to_process}")
    return interval_dfs


# ─── Step 3: Download + MAGICC for each interval ─────────────────────────────

def download_sub_batch(accessions, batch_dir, sub_batch_idx):
    """Download a sub-batch of genomes using ncbi-datasets CLI."""
    tmp_acc_file = batch_dir / f"sub_batch_{sub_batch_idx}_accessions.txt"
    with open(tmp_acc_file, "w") as f:
        for acc in accessions:
            f.write(acc + "\n")

    zip_file = batch_dir / f"sub_batch_{sub_batch_idx}.zip"
    extract_dir = batch_dir / f"sub_batch_{sub_batch_idx}_extract"

    # Download
    cmd = [
        "datasets", "download", "genome", "accession",
        "--inputfile", str(tmp_acc_file),
        "--include", "genome",
        "--filename", str(zip_file),
    ]

    log(f"    Downloading sub-batch {sub_batch_idx} ({len(accessions)} accessions)...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            log(f"    WARNING: Download failed for sub-batch {sub_batch_idx}: {result.stderr[:500]}")
            return []
    except subprocess.TimeoutExpired:
        log(f"    WARNING: Download timed out for sub-batch {sub_batch_idx}")
        return []

    if not zip_file.exists():
        log(f"    WARNING: Zip file not created for sub-batch {sub_batch_idx}")
        return []

    # Unzip
    try:
        subprocess.run(
            ["unzip", "-o", str(zip_file), "-d", str(extract_dir)],
            capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        log(f"    WARNING: Unzip timed out for sub-batch {sub_batch_idx}")
        return []

    # Find all .fna files
    fna_files = list(extract_dir.rglob("*.fna"))
    log(f"    Found {len(fna_files)} .fna files in sub-batch {sub_batch_idx}")

    # Clean up zip
    zip_file.unlink(missing_ok=True)
    tmp_acc_file.unlink(missing_ok=True)

    return fna_files


def step3_process_interval(interval_id, interval_df):
    """Download genomes, run MAGICC, save results, cleanup for one interval."""
    pred_file = OUTPUT_DIR / f"interval_{interval_id}_magicc_predictions.tsv"

    if pred_file.exists():
        existing_preds = pd.read_csv(pred_file, sep="\t")
        if len(existing_preds) > 0:
            log(f"Step 3 (interval {interval_id}): Predictions already exist ({len(existing_preds)} genomes), skipping...")
            return existing_preds

    log(f"Step 3 (interval {interval_id}): Processing {len(interval_df)} genomes...")

    accessions = interval_df["accession"].tolist()
    batch_dir = OUTPUT_DIR / f"interval_{interval_id}_batch"
    fasta_dir = OUTPUT_DIR / f"interval_{interval_id}_fasta"
    batch_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir.mkdir(parents=True, exist_ok=True)

    # Download in sub-batches
    all_fna_files = []
    n_sub_batches = (len(accessions) + DOWNLOAD_BATCH_SIZE - 1) // DOWNLOAD_BATCH_SIZE

    for i in range(n_sub_batches):
        start = i * DOWNLOAD_BATCH_SIZE
        end = min(start + DOWNLOAD_BATCH_SIZE, len(accessions))
        sub_accessions = accessions[start:end]

        fna_files = download_sub_batch(sub_accessions, batch_dir, i)
        all_fna_files.extend(fna_files)

    if not all_fna_files:
        log(f"  WARNING: No genomes downloaded for interval {interval_id}")
        return pd.DataFrame()

    # Move all .fna files to a single flat directory for MAGICC
    # Use the accession from the filename as the key
    for fna_file in all_fna_files:
        dest = fasta_dir / fna_file.name
        if not dest.exists():
            shutil.copy2(fna_file, dest)

    n_fasta = len(list(fasta_dir.glob("*.fna")))
    log(f"  Total .fna files for MAGICC: {n_fasta}")

    # Clean up batch download directories
    shutil.rmtree(batch_dir, ignore_errors=True)

    # Run MAGICC
    magicc_output = OUTPUT_DIR / f"interval_{interval_id}_magicc_raw.tsv"
    cmd = [
        sys.executable, "-m", "magicc", "predict",
        "--input", str(fasta_dir),
        "--output", str(magicc_output),
        "--threads", str(MAGICC_THREADS),
        "--extension", ".fna",
        "--model", str(MAGICC_MODEL),
        "--normalization", str(NORM_PARAMS),
        "--kmers", str(KMERS_FILE),
    ]

    log(f"  Running MAGICC on {n_fasta} genomes with {MAGICC_THREADS} threads...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        log(f"  ERROR: MAGICC failed: {result.stderr[:1000]}")
        log(f"  STDOUT: {result.stdout[:1000]}")
        # Still try to continue if partial results exist

    # Read MAGICC predictions
    if magicc_output.exists():
        preds = pd.read_csv(magicc_output, sep="\t")
        log(f"  MAGICC produced {len(preds)} predictions")

        # Save final predictions
        preds.to_csv(pred_file, sep="\t", index=False)
        log(f"  Saved predictions to {pred_file}")

        # Clean up raw output
        magicc_output.unlink(missing_ok=True)
    else:
        log(f"  ERROR: No MAGICC output produced")
        preds = pd.DataFrame()

    # Delete downloaded genomes to free disk space
    log(f"  Cleaning up downloaded genomes...")
    shutil.rmtree(fasta_dir, ignore_errors=True)
    shutil.rmtree(batch_dir, ignore_errors=True)

    return preds


# ─── Step 4: Aggregate and compare ───────────────────────────────────────────

def extract_accession_from_genome_id(genome_id):
    """Extract the GCF/GCA accession from MAGICC genome_id (which is the filename stem).

    NCBI datasets downloads produce filenames like:
    GCF_034719275.1_ASM3471927v1_genomic.fna
    We need to extract GCF_034719275.1
    """
    # Pattern: GC[AF]_XXXXXXXXX.N_*
    parts = str(genome_id).split("_")
    if len(parts) >= 3 and parts[0] in ("GCF", "GCA"):
        # Reconstruct GCF_XXXXXXXXX.N
        return f"{parts[0]}_{parts[1]}"
    return genome_id


def step4_aggregate_and_compare(metadata_df, interval_dfs):
    """Merge MAGICC predictions with CheckM2 metadata and compute comparisons."""
    log("Step 4: Aggregating results and comparing MAGICC vs CheckM2...")

    comparison_file = OUTPUT_DIR / "comparison_results.tsv"
    summary_file = OUTPUT_DIR / "analysis_summary.json"

    # Load all MAGICC predictions
    all_preds = []
    for interval_id in INTERVALS:
        pred_file = OUTPUT_DIR / f"interval_{interval_id}_magicc_predictions.tsv"
        if pred_file.exists():
            preds = pd.read_csv(pred_file, sep="\t")
            preds["interval"] = interval_id
            all_preds.append(preds)
            log(f"  Loaded {len(preds)} predictions from interval {interval_id}")
        else:
            log(f"  WARNING: No predictions file for interval {interval_id}")

    if not all_preds:
        log("  ERROR: No predictions found at all!")
        return

    all_preds_df = pd.concat(all_preds, ignore_index=True)
    log(f"  Total MAGICC predictions: {len(all_preds_df)}")

    # Extract accession from genome_id
    # MAGICC output column is 'genome_name' (not 'genome_id')
    genome_col = "genome_name" if "genome_name" in all_preds_df.columns else "genome_id"
    all_preds_df["accession_extracted"] = all_preds_df[genome_col].apply(extract_accession_from_genome_id)

    # Merge with metadata
    merged = all_preds_df.merge(
        metadata_df,
        left_on="accession_extracted",
        right_on="accession",
        how="inner",
    )
    log(f"  Merged: {len(merged)} genomes (from {len(all_preds_df)} predictions)")

    if len(merged) < len(all_preds_df) * 0.5:
        log("  WARNING: Less than 50% of predictions could be matched to metadata!")
        # Debug: show sample of unmatched
        matched_acc = set(merged["accession_extracted"])
        unmatched = all_preds_df[~all_preds_df["accession_extracted"].isin(matched_acc)]
        log(f"  Sample unmatched genome_ids: {unmatched['genome_id'].head(5).tolist()}")
        log(f"  Sample unmatched extracted: {unmatched['accession_extracted'].head(5).tolist()}")
        log(f"  Sample metadata accessions: {metadata_df['accession'].head(5).tolist()}")

    # Compute differences
    merged["completeness_diff"] = merged["pred_completeness"] - merged["checkm2_completeness"]
    merged["contamination_diff"] = merged["pred_contamination"] - merged["checkm2_contamination"]

    # MIMAG HQ classification
    merged["checkm2_hq"] = (merged["checkm2_completeness"] >= 90) & (merged["checkm2_contamination"] <= 5)
    merged["magicc_hq"] = (merged["pred_completeness"] >= 90) & (merged["pred_contamination"] <= 5)

    # Save comparison results
    save_cols = [
        "accession_extracted", "species", "interval",
        "checkm2_completeness", "checkm2_contamination",
        "pred_completeness", "pred_contamination",
        "completeness_diff", "contamination_diff",
        "checkm2_hq", "magicc_hq",
    ]
    # Only keep columns that exist
    save_cols = [c for c in save_cols if c in merged.columns]
    merged[save_cols].to_csv(comparison_file, sep="\t", index=False)
    log(f"  Saved comparison results to {comparison_file}")

    # ─── Compute summary statistics ─────────────────────────────────────
    summary = {
        "total_genomes_analyzed": int(len(merged)),
        "species_counts": {},
        "interval_counts": {},
        "by_interval": {},
        "by_species": {},
        "by_interval_and_species": {},
        "mimag_hq_discordance": {},
    }

    # Species counts
    for sp in TARGET_SPECIES:
        sp_short = sp.split("__")[1]
        summary["species_counts"][sp_short] = int((merged["species"] == sp).sum())

    # Interval counts
    for interval_id in INTERVALS:
        summary["interval_counts"][f"interval_{interval_id}"] = int((merged["interval"] == interval_id).sum())

    def compute_stats(sub_df, prefix=""):
        """Compute summary statistics for a subset."""
        stats = {}
        for metric, pred_col, checkm2_col, diff_col in [
            ("completeness", "pred_completeness", "checkm2_completeness", "completeness_diff"),
            ("contamination", "pred_contamination", "checkm2_contamination", "contamination_diff"),
        ]:
            if len(sub_df) == 0:
                continue
            stats[metric] = {
                "n": int(len(sub_df)),
                "magicc_mean": round(float(sub_df[pred_col].mean()), 4),
                "magicc_median": round(float(sub_df[pred_col].median()), 4),
                "magicc_std": round(float(sub_df[pred_col].std()), 4),
                "checkm2_mean": round(float(sub_df[checkm2_col].mean()), 4),
                "checkm2_median": round(float(sub_df[checkm2_col].median()), 4),
                "checkm2_std": round(float(sub_df[checkm2_col].std()), 4),
                "diff_mean": round(float(sub_df[diff_col].mean()), 4),
                "diff_median": round(float(sub_df[diff_col].median()), 4),
                "diff_std": round(float(sub_df[diff_col].std()), 4),
            }
        return stats

    # By interval
    for interval_id in INTERVALS:
        interval_data = merged[merged["interval"] == interval_id]
        summary["by_interval"][f"interval_{interval_id}"] = compute_stats(interval_data)

    # By species
    for sp in TARGET_SPECIES:
        sp_short = sp.split("__")[1]
        sp_data = merged[merged["species"] == sp]
        summary["by_species"][sp_short] = compute_stats(sp_data)

    # By interval AND species
    for interval_id in INTERVALS:
        interval_key = f"interval_{interval_id}"
        summary["by_interval_and_species"][interval_key] = {}
        for sp in TARGET_SPECIES:
            sp_short = sp.split("__")[1]
            sub = merged[(merged["interval"] == interval_id) & (merged["species"] == sp)]
            if len(sub) > 0:
                summary["by_interval_and_species"][interval_key][sp_short] = compute_stats(sub)

    # MIMAG HQ discordance analysis
    # CheckM2 says HQ but MAGICC says not HQ
    checkm2_hq_magicc_not = merged[merged["checkm2_hq"] & ~merged["magicc_hq"]]
    # MAGICC says HQ but CheckM2 doesn't
    magicc_hq_checkm2_not = merged[~merged["checkm2_hq"] & merged["magicc_hq"]]
    # Both agree HQ
    both_hq = merged[merged["checkm2_hq"] & merged["magicc_hq"]]
    # Both agree not HQ
    both_not_hq = merged[~merged["checkm2_hq"] & ~merged["magicc_hq"]]

    summary["mimag_hq_discordance"] = {
        "both_hq": int(len(both_hq)),
        "both_not_hq": int(len(both_not_hq)),
        "checkm2_hq_magicc_not_hq": {
            "total": int(len(checkm2_hq_magicc_not)),
            "by_species": {},
            "mean_magicc_completeness": round(float(checkm2_hq_magicc_not["pred_completeness"].mean()), 4) if len(checkm2_hq_magicc_not) > 0 else None,
            "mean_magicc_contamination": round(float(checkm2_hq_magicc_not["pred_contamination"].mean()), 4) if len(checkm2_hq_magicc_not) > 0 else None,
            "reason_low_completeness": int(((checkm2_hq_magicc_not["pred_completeness"] < 90)).sum()) if len(checkm2_hq_magicc_not) > 0 else 0,
            "reason_high_contamination": int(((checkm2_hq_magicc_not["pred_contamination"] > 5)).sum()) if len(checkm2_hq_magicc_not) > 0 else 0,
            "reason_both": int(((checkm2_hq_magicc_not["pred_completeness"] < 90) & (checkm2_hq_magicc_not["pred_contamination"] > 5)).sum()) if len(checkm2_hq_magicc_not) > 0 else 0,
        },
        "magicc_hq_checkm2_not_hq": {
            "total": int(len(magicc_hq_checkm2_not)),
            "by_species": {},
            "mean_checkm2_completeness": round(float(magicc_hq_checkm2_not["checkm2_completeness"].mean()), 4) if len(magicc_hq_checkm2_not) > 0 else None,
            "mean_checkm2_contamination": round(float(magicc_hq_checkm2_not["checkm2_contamination"].mean()), 4) if len(magicc_hq_checkm2_not) > 0 else None,
        },
    }

    for sp in TARGET_SPECIES:
        sp_short = sp.split("__")[1]
        summary["mimag_hq_discordance"]["checkm2_hq_magicc_not_hq"]["by_species"][sp_short] = \
            int((checkm2_hq_magicc_not["species"] == sp).sum())
        summary["mimag_hq_discordance"]["magicc_hq_checkm2_not_hq"]["by_species"][sp_short] = \
            int((magicc_hq_checkm2_not["species"] == sp).sum())

    # Identify highly discordant genomes (|diff| > 10% for either metric)
    highly_discordant = merged[
        (merged["completeness_diff"].abs() > 10) |
        (merged["contamination_diff"].abs() > 10)
    ]
    summary["highly_discordant_genomes"] = {
        "total": int(len(highly_discordant)),
        "by_species": {},
        "by_interval": {},
    }
    for sp in TARGET_SPECIES:
        sp_short = sp.split("__")[1]
        summary["highly_discordant_genomes"]["by_species"][sp_short] = \
            int((highly_discordant["species"] == sp).sum())
    for interval_id in INTERVALS:
        summary["highly_discordant_genomes"]["by_interval"][f"interval_{interval_id}"] = \
            int((highly_discordant["interval"] == interval_id).sum())

    # Save summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved summary to {summary_file}")

    # Print key findings
    log("\n" + "=" * 80)
    log("KEY FINDINGS")
    log("=" * 80)
    log(f"Total genomes analyzed: {len(merged)}")
    log(f"\nMIMAG HQ Agreement:")
    log(f"  Both agree HQ:           {len(both_hq)}")
    log(f"  Both agree not HQ:       {len(both_not_hq)}")
    log(f"  CheckM2 HQ, MAGICC not:  {len(checkm2_hq_magicc_not)}")
    log(f"  MAGICC HQ, CheckM2 not:  {len(magicc_hq_checkm2_not)}")

    if len(checkm2_hq_magicc_not) > 0:
        log(f"\n  Wrong MIMAG HQ by CheckM2 (MAGICC disagrees):")
        for sp in TARGET_SPECIES:
            sp_short = sp.split("__")[1]
            count = (checkm2_hq_magicc_not["species"] == sp).sum()
            log(f"    {sp_short}: {count}")

    log(f"\nOverall statistics (all genomes):")
    for metric in ["completeness", "contamination"]:
        if metric in summary["by_interval"].get("interval_1", {}):
            all_stats = compute_stats(merged)
            if metric in all_stats:
                s = all_stats[metric]
                log(f"  {metric.capitalize()}:")
                log(f"    MAGICC  mean={s['magicc_mean']:.2f}, median={s['magicc_median']:.2f}, std={s['magicc_std']:.2f}")
                log(f"    CheckM2 mean={s['checkm2_mean']:.2f}, median={s['checkm2_median']:.2f}, std={s['checkm2_std']:.2f}")
                log(f"    Diff    mean={s['diff_mean']:.2f}, median={s['diff_median']:.2f}, std={s['diff_std']:.2f}")

    return merged


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log("=" * 80)
    log("Pathogen CheckM2 vs MAGICC Comparison Analysis")
    log("=" * 80)

    # Step 1: Extract metadata
    metadata_df = step1_extract_metadata()

    # Step 2: Define intervals
    interval_dfs = step2_define_intervals(metadata_df)

    # Step 3: Process each interval
    for interval_id in sorted(INTERVALS.keys()):
        step3_process_interval(interval_id, interval_dfs[interval_id])

    # Step 4: Aggregate and compare
    step4_aggregate_and_compare(metadata_df, interval_dfs)

    log("\nDone!")


if __name__ == "__main__":
    main()
