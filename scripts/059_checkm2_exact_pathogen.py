#!/usr/bin/env python3
"""
Script 59: Run CheckM2 on ALL 50 exact synthetic pathogen genomes (5 configs x 10 replicates)
and compute mean +/- sd per configuration.

Steps:
  1. Regenerate all 50 FASTA files (matching script 58's exact logic)
  2. Run CheckM2 batch on all 50 genomes
  3. Parse quality_report.tsv
  4. Compute mean +/- sd per configuration
  5. Save results
"""

import sys
import os
import signal
import json
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# Collapse-safe: ignore SIGHUP
signal.signal(signal.SIGHUP, signal.SIG_IGN)

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/path/to/magicc-legacy')
DATA_DIR = PROJECT_DIR / 'data'

SE_GENOME = DATA_DIR / 'genomes' / 'GCF_001302605.1' / 'GCF_001302605.1_ASM130260v1_genomic.fna'
LM_GENOME = DATA_DIR / 'genomes' / 'GCF_000021185.1' / 'GCF_000021185.1_ASM2118v1_genomic.fna'

OUTPUT_DIR = DATA_DIR / 'benchmarks' / 'pathogen_analysis_v5' / 'exact_synthetic'
FASTA_DIR = OUTPUT_DIR / 'checkm2_fasta_all'
CHECKM2_OUTPUT_DIR = OUTPUT_DIR / 'checkm2_output'

CHECKM2DB = '/path/to/magicc/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd'
CHECKM2_CONDA_ENV = 'checkm2_py39'
CHECKM2_THREADS = 32

N_REPLICATES = 10
REPLICATE_SEEDS = list(range(10))

# 5 genome configurations: (name, dominant_comp_frac, contaminant_frac_of_dominant)
GENOME_CONFIGS = [
    ("100se_5lm",   1.00, 0.05),
    ("100se_20lm",  1.00, 0.20),
    ("100se_50lm",  1.00, 0.50),
    ("100se_100lm", 1.00, 1.00),
    ("60se_40lm",   0.60, 0.40),
]


# ============================================================================
# FASTA I/O (identical to script 58)
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
    """Read FASTA and return all contigs concatenated into one sequence."""
    contigs = read_fasta_contigs(fasta_path)
    return ''.join(contigs)


def write_fasta(path: str, contigs: List[Tuple[str, str]]):
    """Write contigs as FASTA. contigs = list of (header, sequence)."""
    with open(path, 'w') as f:
        for header, seq in contigs:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


# ============================================================================
# Synthetic genome generation (identical to script 58)
# ============================================================================
def generate_exact_synthetic(
    dominant_seq: str,
    contaminant_seq: str,
    dominant_full_size: int,
    comp_frac: float,
    cont_frac: float,
    seed: int,
) -> Tuple[List[str], int, int]:
    """
    Generate synthetic genome with EXACT completeness and contamination.
    No fragmentation -- output has 1 or 2 contigs.
    """
    rng = np.random.default_rng(seed)

    # --- Dominant portion ---
    if comp_frac >= 1.0:
        dominant_contig = dominant_seq
    else:
        target_bp = int(round(dominant_full_size * comp_frac))
        max_start = len(dominant_seq) - target_bp
        assert max_start >= 0
        start = rng.integers(0, max_start + 1)
        dominant_contig = dominant_seq[start:start + target_bp]

    dominant_bp = len(dominant_contig)

    # --- Contaminant portion ---
    contaminant_bp = 0
    contaminant_contig = None

    if cont_frac > 0:
        target_cont_bp = int(round(dominant_full_size * cont_frac))
        cont_pool = contaminant_seq
        if len(cont_pool) < target_cont_bp:
            repeats = (target_cont_bp // len(contaminant_seq)) + 1
            cont_pool = contaminant_seq * repeats
        max_start = len(cont_pool) - target_cont_bp
        start = rng.integers(0, max_start + 1)
        contaminant_contig = cont_pool[start:start + target_cont_bp]
        contaminant_bp = len(contaminant_contig)

    contigs = [dominant_contig]
    if contaminant_contig is not None:
        contigs.append(contaminant_contig)

    return contigs, dominant_bp, contaminant_bp


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("Script 59: CheckM2 on ALL 50 Exact Synthetic Pathogen Genomes")
    print("=" * 80)
    print()

    # ----------------------------------------------------------------
    # Step 1: Generate all 50 FASTA files
    # ----------------------------------------------------------------
    FASTA_DIR.mkdir(parents=True, exist_ok=True)

    # Check how many already exist
    existing = list(FASTA_DIR.glob("*.fasta"))
    if len(existing) == 50:
        print(f"All 50 FASTA files already exist in {FASTA_DIR}, skipping generation.")
    else:
        print(f"Found {len(existing)} existing FASTA files, generating all 50...")
        print("\nReading reference genomes...")
        se_seq = read_full_sequence(str(SE_GENOME))
        lm_seq = read_full_sequence(str(LM_GENOME))
        se_size = len(se_seq)
        lm_size = len(lm_seq)
        print(f"  S. enterica:       {se_size:>10,} bp")
        print(f"  L. monocytogenes:  {lm_size:>10,} bp")

        generation_log = []

        for config_name, comp_frac, cont_frac in GENOME_CONFIGS:
            print(f"\n  Config: {config_name}")
            for rep_idx, seed in enumerate(REPLICATE_SEEDS):
                fname = f"{config_name}_rep{rep_idx}.fasta"
                fpath = FASTA_DIR / fname

                contigs, dom_bp, cont_bp = generate_exact_synthetic(
                    se_seq, lm_seq, se_size, comp_frac, cont_frac, seed
                )

                fasta_entries = [("dominant", contigs[0])]
                if len(contigs) > 1:
                    fasta_entries.append(("contaminant", contigs[1]))
                write_fasta(str(fpath), fasta_entries)

                total_bp = sum(len(c) for c in contigs)
                generation_log.append({
                    'filename': fname,
                    'config': config_name,
                    'replicate': rep_idx,
                    'seed': seed,
                    'dominant_bp': dom_bp,
                    'contaminant_bp': cont_bp,
                    'total_bp': total_bp,
                    'n_contigs': len(contigs),
                    'true_completeness': dom_bp / se_size * 100.0,
                    'true_contamination': cont_bp / se_size * 100.0,
                })

                print(f"    rep{rep_idx}: dom={dom_bp:,}bp cont={cont_bp:,}bp total={total_bp:,}bp")

        # Save generation log
        gen_log_path = FASTA_DIR / 'generation_log.json'
        with open(gen_log_path, 'w') as f:
            json.dump(generation_log, f, indent=2)
        print(f"\n  Saved generation log: {gen_log_path}")

    # Verify 50 FASTA files
    fasta_files = sorted(FASTA_DIR.glob("*.fasta"))
    print(f"\n  Total FASTA files: {len(fasta_files)}")
    assert len(fasta_files) == 50, f"Expected 50 FASTA files, found {len(fasta_files)}"

    # ----------------------------------------------------------------
    # Step 2: Run CheckM2
    # ----------------------------------------------------------------
    quality_report = CHECKM2_OUTPUT_DIR / 'quality_report.tsv'

    if quality_report.exists():
        print(f"\nCheckM2 output already exists: {quality_report}")
        print("  Skipping CheckM2 run (delete output dir to re-run).")
    else:
        print(f"\nRunning CheckM2 on {len(fasta_files)} genomes...")
        print(f"  Input dir:  {FASTA_DIR}")
        print(f"  Output dir: {CHECKM2_OUTPUT_DIR}")
        print(f"  Threads:    {CHECKM2_THREADS}")
        print(f"  DB:         {CHECKM2DB}")

        cmd = [
            'conda', 'run', '-n', CHECKM2_CONDA_ENV,
            'env', f'CHECKM2DB={CHECKM2DB}',
            'checkm2', 'predict',
            '--threads', str(CHECKM2_THREADS),
            '-x', '.fasta',
            '--input', str(FASTA_DIR),
            '--output-directory', str(CHECKM2_OUTPUT_DIR),
            '--force',
        ]
        print(f"  Command: {' '.join(cmd)}")
        print()

        t_ckm2 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        elapsed_ckm2 = time.time() - t_ckm2
        print(f"  CheckM2 finished in {elapsed_ckm2:.1f}s ({elapsed_ckm2/60:.1f} min)")

        if result.returncode != 0:
            print(f"  STDERR:\n{result.stderr}")
            print(f"  STDOUT:\n{result.stdout}")
            sys.exit(1)
        else:
            print(f"  CheckM2 completed successfully.")
            if result.stdout:
                # Print last few lines of stdout
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(f"    {line}")

    # ----------------------------------------------------------------
    # Step 3: Parse quality_report.tsv
    # ----------------------------------------------------------------
    print(f"\nParsing CheckM2 results from {quality_report}...")
    assert quality_report.exists(), f"quality_report.tsv not found at {quality_report}"

    df_ckm2 = pd.read_csv(quality_report, sep='\t')
    print(f"  Loaded {len(df_ckm2)} rows from quality_report.tsv")
    print(f"  Columns: {list(df_ckm2.columns)}")

    # Parse config and replicate from the Name column
    # Name format: 100se_5lm_rep0 (without .fasta extension)
    def parse_name(name):
        parts = name.rsplit('_rep', 1)
        config = parts[0]
        rep = int(parts[1])
        return config, rep

    df_ckm2['config'] = df_ckm2['Name'].apply(lambda x: parse_name(x)[0])
    df_ckm2['replicate'] = df_ckm2['Name'].apply(lambda x: parse_name(x)[1])

    print(f"\n  Per-genome CheckM2 results:")
    print(f"  {'Name':<25} {'Completeness':>14} {'Contamination':>14}")
    print(f"  {'-'*55}")
    for _, row in df_ckm2.sort_values(['config', 'replicate']).iterrows():
        print(f"  {row['Name']:<25} {row['Completeness']:>14.2f} {row['Contamination']:>14.2f}")

    # ----------------------------------------------------------------
    # Step 4: Compute mean +/- sd per configuration
    # ----------------------------------------------------------------
    print(f"\n{'='*100}")
    print("SUMMARY: CheckM2 Results (mean +/- sd) per Configuration")
    print(f"{'='*100}")

    # Load V5 results for comparison
    v5_tsv = OUTPUT_DIR / 'replicate_summary.tsv'
    df_v5 = None
    if v5_tsv.exists():
        df_v5 = pd.read_csv(v5_tsv, sep='\t')
        print("  (Loaded V5 results for comparison)")

    summary_rows = []

    config_order = [c[0] for c in GENOME_CONFIGS]
    true_values = {c[0]: (c[1], c[2]) for c in GENOME_CONFIGS}

    # Read SE genome size for true value computation
    se_size = 4793553  # Known from the replicate_summary.tsv

    print(f"\n  {'Config':<16} {'True Comp':>10} {'True Cont':>10} "
          f"{'CkM2 Comp':>22} {'CkM2 Cont':>22} "
          f"{'V5 Comp':>22} {'V5 Cont':>22}")
    print(f"  {'-'*126}")

    for config_name in config_order:
        comp_frac, cont_frac = true_values[config_name]

        # True values
        if comp_frac >= 1.0:
            true_comp = 100.0
        else:
            true_comp = comp_frac * 100.0
        true_cont = cont_frac * 100.0

        # CheckM2 stats
        mask = df_ckm2['config'] == config_name
        ckm2_sub = df_ckm2[mask]
        assert len(ckm2_sub) == 10, f"Expected 10 replicates for {config_name}, got {len(ckm2_sub)}"

        ckm2_comp_mean = ckm2_sub['Completeness'].mean()
        ckm2_comp_std = ckm2_sub['Completeness'].std(ddof=1)
        ckm2_cont_mean = ckm2_sub['Contamination'].mean()
        ckm2_cont_std = ckm2_sub['Contamination'].std(ddof=1)

        # V5 stats
        v5_comp_str = "N/A"
        v5_cont_str = "N/A"
        v5_comp_mean = v5_comp_std = v5_cont_mean = v5_cont_std = None
        if df_v5 is not None:
            v5_mask = df_v5['config'] == config_name
            v5_sub = df_v5[v5_mask]
            if len(v5_sub) == 10:
                v5_comp_mean = v5_sub['pred_completeness'].mean()
                v5_comp_std = v5_sub['pred_completeness'].std(ddof=1)
                v5_cont_mean = v5_sub['pred_contamination'].mean()
                v5_cont_std = v5_sub['pred_contamination'].std(ddof=1)
                v5_comp_str = f"{v5_comp_mean:.2f}+/-{v5_comp_std:.2f}"
                v5_cont_str = f"{v5_cont_mean:.2f}+/-{v5_cont_std:.2f}"

        ckm2_comp_str = f"{ckm2_comp_mean:.2f}+/-{ckm2_comp_std:.2f}"
        ckm2_cont_str = f"{ckm2_cont_mean:.2f}+/-{ckm2_cont_std:.2f}"

        print(f"  {config_name:<16} {true_comp:>10.2f} {true_cont:>10.2f} "
              f"{ckm2_comp_str:>22} {ckm2_cont_str:>22} "
              f"{v5_comp_str:>22} {v5_cont_str:>22}")

        row = {
            'config': config_name,
            'true_completeness': true_comp,
            'true_contamination': true_cont,
            'checkm2_comp_mean': round(ckm2_comp_mean, 4),
            'checkm2_comp_std': round(ckm2_comp_std, 4),
            'checkm2_cont_mean': round(ckm2_cont_mean, 4),
            'checkm2_cont_std': round(ckm2_cont_std, 4),
        }
        if v5_comp_mean is not None:
            row['v5_comp_mean'] = round(v5_comp_mean, 4)
            row['v5_comp_std'] = round(v5_comp_std, 4)
            row['v5_cont_mean'] = round(v5_cont_mean, 4)
            row['v5_cont_std'] = round(v5_cont_std, 4)
        summary_rows.append(row)

    # ----------------------------------------------------------------
    # Step 5: Save results
    # ----------------------------------------------------------------
    # Save summary
    results_path = OUTPUT_DIR / 'checkm2_results.tsv'
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(results_path, sep='\t', index=False)
    print(f"\n  Saved summary: {results_path}")

    # Save per-replicate CheckM2 results
    per_rep_path = OUTPUT_DIR / 'checkm2_per_replicate.tsv'
    df_ckm2.sort_values(['config', 'replicate']).to_csv(per_rep_path, sep='\t', index=False)
    print(f"  Saved per-replicate: {per_rep_path}")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("  Done.")


if __name__ == '__main__':
    main()
