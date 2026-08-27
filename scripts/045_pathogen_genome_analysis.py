#!/usr/bin/env python3
"""
Script 45: Pathogen Genome Analysis - Part A
Synthesize 4 draft genomes from S. enterica + L. monocytogenes
and evaluate with MAGICC and CheckM2.

Uses magicc.fragmentation and magicc.contamination modules for genome synthesis.

Reference genomes:
  - Salmonella enterica: GCF_001302605.1 (4,793,553 bp, 1 contig)
  - Listeria monocytogenes: GCF_000021185.1 (2,976,212 bp, 1 contig)

Synthetic genomes (quality_tier='low' -> 200-500 contigs):
  Genome 1: 100% S.e. + 5% contamination from L.m.
  Genome 2: 100% S.e. + 50% contamination from L.m.
  Genome 3: 100% S.e. + 100% contamination from L.m.
  Genome 4: 60% S.e. + 40% contamination from L.m.

All contamination percentages are based on S. enterica full genome size.
"""

import os
import sys
import json
import numpy as np
import subprocess
import csv
from pathlib import Path

# Add project root to path so magicc modules can be imported
PROJECT_DIR = "/mnt/5c77b453-f7e1-48c8-afa3-5641857a41c7/tianrm/projects/magicc2"
sys.path.insert(0, PROJECT_DIR)

from magicc.fragmentation import read_fasta
from magicc.contamination import generate_contaminated_sample

# ============================================================
# Configuration
# ============================================================
SE_FASTA = os.path.join(PROJECT_DIR, "data/genomes/GCF_001302605.1/GCF_001302605.1_ASM130260v1_genomic.fna")
LM_FASTA = os.path.join(PROJECT_DIR, "data/genomes/GCF_000021185.1/GCF_000021185.1_ASM2118v1_genomic.fna")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data/benchmarks/pathogen_analysis/synthetic")
MAGICC_OUTPUT = os.path.join(PROJECT_DIR, "data/benchmarks/pathogen_analysis/magicc_predictions.tsv")
CHECKM2_OUTPUT_DIR = os.path.join(PROJECT_DIR, "data/benchmarks/pathogen_analysis/checkm2_output")
CHECKM2_DB = os.path.join(PROJECT_DIR, "tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd")

SEED = 42
QUALITY_TIER = "low"  # 200-500 contigs per the quality tier definition

# Genome definitions: (name, target_completeness, target_contamination)
GENOMES = [
    ("genome_1_100se_5lm",   1.0,  5.0),
    ("genome_2_100se_50lm",  1.0, 50.0),
    ("genome_3_100se_100lm", 1.0, 100.0),
    ("genome_4_60se_40lm",   0.6, 40.0),
]


# ============================================================
# Helper: write contigs to FASTA
# ============================================================
def write_fasta(contigs, output_path):
    """Write a list of contig sequences to a FASTA file with numbered headers."""
    with open(output_path, 'w') as f:
        for i, seq in enumerate(contigs, 1):
            f.write(f">contig_{i:03d}\n")
            # Write in 80-char lines
            for j in range(0, len(seq), 80):
                f.write(seq[j:j+80] + "\n")


# ============================================================
# Main synthesis
# ============================================================
def main():
    print("=" * 60)
    print("Pathogen Genome Synthesis and Analysis")
    print("(Using magicc.fragmentation + magicc.contamination modules)")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read reference genomes using magicc.fragmentation.read_fasta
    print("\nReading reference genomes...")
    se_seq = read_fasta(SE_FASTA)
    lm_seq = read_fasta(LM_FASTA)

    se_full_size = len(se_seq)
    lm_full_size = len(lm_seq)

    print(f"  S. enterica: {se_full_size:,} bp")
    print(f"  L. monocytogenes: {lm_full_size:,} bp")

    # Initialize RNG
    rng = np.random.default_rng(SEED)

    # Synthesize genomes
    print("\nSynthesizing draft genomes...")
    all_stats = []

    for name, target_completeness, target_contamination in GENOMES:
        print(f"\n  {name}:")
        print(f"    Target completeness: {target_completeness*100:.0f}%")
        print(f"    Target contamination: {target_contamination:.0f}%")

        result = generate_contaminated_sample(
            dominant_sequence=se_seq,
            contaminant_sequences=[lm_seq],
            target_completeness=target_completeness,
            target_contamination=target_contamination,
            rng=rng,
            dominant_quality_tier=QUALITY_TIER,
            contaminant_quality_tier=QUALITY_TIER,
        )

        contigs = result['contigs']
        actual_completeness = result['completeness'] * 100  # convert to percentage
        actual_contamination = result['contamination']  # already in percentage

        # Independent verification from dominant/contaminant contig bp counts
        dominant_bp = sum(len(c) for c in result['dominant_contigs'])
        contaminant_bp = sum(len(c) for c in result['contaminant_contigs'])
        total_bp = sum(len(c) for c in contigs)

        verify_completeness = dominant_bp / se_full_size * 100
        verify_contamination = contaminant_bp / se_full_size * 100

        print(f"    Result: {len(contigs)} contigs, {total_bp:,} total bp")
        print(f"    Dominant (S.e.): {result['n_contigs_dominant']} contigs, {dominant_bp:,} bp")
        print(f"    Contaminant (L.m.): {result['n_contigs_contaminant']} contigs, {contaminant_bp:,} bp")
        print(f"    Module completeness: {actual_completeness:.2f}%")
        print(f"    Module contamination: {actual_contamination:.2f}%")
        print(f"    Verified completeness: {verify_completeness:.2f}%")
        print(f"    Verified contamination: {verify_contamination:.2f}%")

        # Write FASTA
        fasta_path = os.path.join(OUTPUT_DIR, f"{name}.fasta")
        write_fasta(contigs, fasta_path)
        print(f"    Written to: {fasta_path}")

        stats = {
            "name": name,
            "target_completeness": target_completeness * 100,
            "target_contamination": target_contamination,
            "actual_se_bp": dominant_bp,
            "actual_lm_bp": contaminant_bp,
            "total_bp": total_bp,
            "n_contigs": len(contigs),
            "n_contigs_dominant": result['n_contigs_dominant'],
            "n_contigs_contaminant": result['n_contigs_contaminant'],
            "actual_completeness": actual_completeness,
            "actual_contamination": actual_contamination,
            "verified_completeness": verify_completeness,
            "verified_contamination": verify_contamination,
            "quality_tier": result['dominant_quality_tier'],
            "fasta_path": fasta_path,
        }
        all_stats.append(stats)

    # ============================================================
    # Verification table
    # ============================================================
    print("\n" + "=" * 60)
    print("Verification Table")
    print("=" * 60)
    print(f"{'Genome':<30} {'Total bp':>12} {'S.e. bp':>12} {'L.m. bp':>12} "
          f"{'Comp%':>8} {'Cont%':>8} {'Contigs':>8}")
    print("-" * 96)

    for stats in all_stats:
        print(f"{stats['name']:<30} {stats['total_bp']:>12,} {stats['actual_se_bp']:>12,} "
              f"{stats['actual_lm_bp']:>12,} {stats['actual_completeness']:>8.2f} "
              f"{stats['actual_contamination']:>8.2f} {stats['n_contigs']:>8}")

    # Save verification stats
    verification_path = os.path.join(PROJECT_DIR, "data/benchmarks/pathogen_analysis/verification_stats.json")
    with open(verification_path, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nVerification stats saved to: {verification_path}")

    # ============================================================
    # Run MAGICC
    # ============================================================
    print("\n" + "=" * 60)
    print("Running MAGICC on synthetic genomes...")
    print("=" * 60)

    magicc_cmd = [
        "conda", "run", "-n", "magicc2",
        "python", "-m", "magicc", "predict",
        "--input", OUTPUT_DIR,
        "--output", MAGICC_OUTPUT,
        "--threads", "1"
    ]
    print(f"  Command: {' '.join(magicc_cmd)}")
    result = subprocess.run(magicc_cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
    print(result.stdout)
    if result.returncode != 0:
        print(f"MAGICC ERROR: {result.stderr}")
        sys.exit(1)

    # Parse MAGICC results
    magicc_results = {}
    with open(MAGICC_OUTPUT) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            magicc_results[row['genome_name']] = {
                'completeness': float(row['pred_completeness']),
                'contamination': float(row['pred_contamination'])
            }

    print("\nMAGICC Results:")
    for name, res in sorted(magicc_results.items()):
        print(f"  {name}: comp={res['completeness']:.2f}%, cont={res['contamination']:.2f}%")

    # ============================================================
    # Run CheckM2
    # ============================================================
    print("\n" + "=" * 60)
    print("Running CheckM2 on synthetic genomes...")
    print("=" * 60)

    os.makedirs(CHECKM2_OUTPUT_DIR, exist_ok=True)

    checkm2_cmd = [
        "conda", "run", "-n", "checkm2_py39",
        "checkm2", "predict",
        "--threads", "32",
        "-x", ".fasta",
        "--input", OUTPUT_DIR,
        "--output-directory", CHECKM2_OUTPUT_DIR,
        "--force",
        "--database_path", CHECKM2_DB
    ]
    print(f"  Command: {' '.join(checkm2_cmd)}")
    result = subprocess.run(checkm2_cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
    print(result.stdout)
    if result.returncode != 0:
        print(f"CheckM2 STDERR: {result.stderr[-2000:]}")
        # Don't exit - CheckM2 prints warnings to stderr but may still succeed

    # Parse CheckM2 results
    checkm2_report = os.path.join(CHECKM2_OUTPUT_DIR, "quality_report.tsv")
    checkm2_results = {}
    if os.path.exists(checkm2_report):
        with open(checkm2_report) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                checkm2_results[row['Name']] = {
                    'completeness': float(row['Completeness']),
                    'contamination': float(row['Contamination'])
                }

        print("\nCheckM2 Results:")
        for name, res in sorted(checkm2_results.items()):
            print(f"  {name}: comp={res['completeness']:.2f}%, cont={res['contamination']:.2f}%")
    else:
        print("WARNING: CheckM2 quality_report.tsv not found!")

    # ============================================================
    # Comparison table
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    header = (f"{'Genome':<30} {'True':>6} {'True':>6} {'MAGICC':>8} {'MAGICC':>8} "
              f"{'CheckM2':>8} {'CheckM2':>8}")
    subheader = (f"{'':30} {'Comp%':>6} {'Cont%':>6} {'Comp%':>8} {'Cont%':>8} "
                 f"{'Comp%':>8} {'Cont%':>8}")
    print(header)
    print(subheader)
    print("-" * 80)

    comparison_rows = []
    for stats in all_stats:
        name = stats['name']
        true_comp = stats['actual_completeness']
        true_cont = stats['actual_contamination']

        m_comp = magicc_results.get(name, {}).get('completeness', float('nan'))
        m_cont = magicc_results.get(name, {}).get('contamination', float('nan'))
        c_comp = checkm2_results.get(name, {}).get('completeness', float('nan'))
        c_cont = checkm2_results.get(name, {}).get('contamination', float('nan'))

        print(f"{name:<30} {true_comp:>6.2f} {true_cont:>6.2f} "
              f"{m_comp:>8.2f} {m_cont:>8.2f} {c_comp:>8.2f} {c_cont:>8.2f}")

        comparison_rows.append({
            'genome_id': name,
            'true_completeness': round(true_comp, 2),
            'true_contamination': round(true_cont, 2),
            'magicc_completeness': round(m_comp, 2),
            'magicc_contamination': round(m_cont, 2),
            'checkm2_completeness': round(c_comp, 2),
            'checkm2_contamination': round(c_cont, 2),
        })

    # Save comparison table
    comparison_path = os.path.join(PROJECT_DIR, "data/benchmarks/pathogen_analysis/comparison_table.tsv")
    with open(comparison_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\nComparison table saved to: {comparison_path}")

    # Save full summary
    summary = {
        "reference_genomes": {
            "salmonella_enterica": {
                "accession": "GCF_001302605.1",
                "size_bp": se_full_size,
                "fasta": SE_FASTA
            },
            "listeria_monocytogenes": {
                "accession": "GCF_000021185.1",
                "size_bp": lm_full_size,
                "fasta": LM_FASTA
            }
        },
        "synthesis_params": {
            "seed": SEED,
            "quality_tier": QUALITY_TIER,
            "method": "magicc.contamination.generate_contaminated_sample"
        },
        "genome_stats": all_stats,
        "magicc_results": magicc_results,
        "checkm2_results": checkm2_results,
        "comparison": comparison_rows
    }

    summary_path = os.path.join(PROJECT_DIR, "data/benchmarks/pathogen_analysis/synthesis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Full summary saved to: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
