#!/usr/bin/env python3
"""
10_count_9mers.py
Count 9-mers using KMC3 on extracted core gene DNA sequences.

For each genome's core gene FASTA:
1. Run KMC3 to count canonical 9-mers (k=9)
2. Dump the k-mer list
3. Record which k-mers are present in each genome (for prevalence counting)

Produces per-genome k-mer presence files and aggregated prevalence counts.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import time
import logging
import pickle

PROJ = Path("/home/tianrm/projects/magicc2")
OUT_DIR = PROJ / "data/kmer_selection"

N_CPUS = max(1, int(os.cpu_count() * 0.9))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUT_DIR / "kmer_counting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def count_kmers_for_genome(args):
    """
    Run KMC3 on a single genome's core gene FASTA.
    Returns: (accession, set_of_kmers_present, error_or_None)
    """
    accession, core_gene_fasta = args

    if not os.path.exists(core_gene_fasta) or os.path.getsize(core_gene_fasta) == 0:
        return (accession, set(), f"Core gene FASTA missing or empty: {core_gene_fasta}")

    tmpdir = tempfile.mkdtemp(prefix=f"kmc_{accession}_")
    try:
        kmc_db = os.path.join(tmpdir, "kmc_out")
        kmc_dump_file = os.path.join(tmpdir, "kmers.txt")

        # Run KMC3: k=9, canonical form, min count=1, multi-FASTA input
        proc = subprocess.run(
            ["kmc", "-k9", "-ci1", "-cs65535", "-fm",
             core_gene_fasta, kmc_db, tmpdir],
            capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            return (accession, set(), f"KMC failed: {proc.stderr[:200]}")

        # Dump k-mers (just presence, not counts needed for prevalence)
        proc = subprocess.run(
            ["kmc_dump", kmc_db, kmc_dump_file],
            capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            return (accession, set(), f"kmc_dump failed: {proc.stderr[:200]}")

        # Read k-mers (format: kmer\tcount per line)
        kmers = set()
        with open(kmc_dump_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    kmers.add(parts[0])

        return (accession, kmers, None)

    except subprocess.TimeoutExpired:
        return (accession, set(), "Timeout")
    except Exception as e:
        return (accession, set(), str(e)[:200])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def process_domain(domain_name, core_gene_dir, output_prefix, n_workers):
    """
    Count 9-mers for all genomes in a domain and compute prevalence.
    """
    prevalence_file = OUT_DIR / f"{output_prefix}_kmer_prevalence.tsv"
    stats_file = OUT_DIR / f"{output_prefix}_kmer_stats.json"

    # Check if already done
    if prevalence_file.exists():
        logger.info(f"{domain_name}: Prevalence file already exists: {prevalence_file}")
        df = pd.read_csv(prevalence_file, sep='\t')
        logger.info(f"  {len(df)} unique k-mers, max prevalence: {df['prevalence'].max()}")
        return

    # Get list of core gene FASTA files
    results_tsv = core_gene_dir / "core_gene_results.tsv"
    results_df = pd.read_csv(results_tsv, sep='\t')

    # Filter to successful genomes with core genes
    valid = results_df[results_df['core_gene_path'].notna() & (results_df['n_core_genes'] > 0)]
    logger.info(f"{domain_name}: {len(valid)} genomes with core genes")

    tasks = [(row['accession'], row['core_gene_path']) for _, row in valid.iterrows()]

    # Count k-mers in parallel
    logger.info(f"{domain_name}: Counting 9-mers with {n_workers} workers...")
    prevalence = Counter()  # kmer -> number of genomes containing it
    n_done = 0
    n_total = len(tasks)
    n_errors = 0
    kmers_per_genome = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(count_kmers_for_genome, task): task[0] for task in tasks}

        for future in as_completed(futures):
            acc = futures[future]
            try:
                accession, kmers, error = future.result()
                if error:
                    n_errors += 1
                    if n_errors <= 5:
                        logger.warning(f"  {accession}: {error}")
                else:
                    kmers_per_genome.append(len(kmers))
                    for kmer in kmers:
                        prevalence[kmer] += 1
            except Exception as e:
                n_errors += 1
                logger.warning(f"  {acc}: Exception: {str(e)[:200]}")

            n_done += 1
            if n_done % 200 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_total - n_done) / rate if rate > 0 else 0
                logger.info(f"  {domain_name}: {n_done}/{n_total} ({n_done/n_total*100:.1f}%) "
                           f"- {rate:.1f} genomes/sec - ETA: {eta:.0f}s "
                           f"- {len(prevalence)} unique k-mers so far")

    elapsed = time.time() - t0
    logger.info(f"{domain_name}: Completed in {elapsed:.1f}s")

    # Save prevalence as TSV (sorted by prevalence descending)
    prev_list = [(kmer, count) for kmer, count in prevalence.items()]
    prev_list.sort(key=lambda x: (-x[1], x[0]))
    prev_df = pd.DataFrame(prev_list, columns=['kmer', 'prevalence'])
    prev_df.to_csv(prevalence_file, sep='\t', index=False)
    logger.info(f"  Saved prevalence to: {prevalence_file}")

    # Stats
    n_genomes_used = n_total - n_errors
    stats = {
        'domain': domain_name,
        'n_genomes_total': n_total,
        'n_genomes_used': n_genomes_used,
        'n_errors': n_errors,
        'total_unique_kmers': len(prevalence),
        'kmers_per_genome_mean': float(np.mean(kmers_per_genome)) if kmers_per_genome else 0,
        'kmers_per_genome_median': float(np.median(kmers_per_genome)) if kmers_per_genome else 0,
        'kmers_per_genome_min': int(np.min(kmers_per_genome)) if kmers_per_genome else 0,
        'kmers_per_genome_max': int(np.max(kmers_per_genome)) if kmers_per_genome else 0,
        'max_prevalence': max(prevalence.values()) if prevalence else 0,
        'prevalence_p50': int(np.percentile(list(prevalence.values()), 50)) if prevalence else 0,
        'prevalence_p90': int(np.percentile(list(prevalence.values()), 90)) if prevalence else 0,
        'prevalence_p99': int(np.percentile(list(prevalence.values()), 99)) if prevalence else 0,
        'kmers_in_all_genomes': sum(1 for v in prevalence.values() if v == n_genomes_used),
        'kmers_in_90pct_genomes': sum(1 for v in prevalence.values() if v >= 0.9 * n_genomes_used),
        'kmers_in_50pct_genomes': sum(1 for v in prevalence.values() if v >= 0.5 * n_genomes_used),
        'elapsed_seconds': elapsed,
    }
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Stats: {json.dumps(stats, indent=2)}")

    return stats


def main():
    logger.info("=== 9-mer Counting on Core Gene Sequences ===")
    logger.info(f"CPUs: {N_CPUS}")

    bact_dir = OUT_DIR / "bacterial_core_genes"
    arch_dir = OUT_DIR / "archaeal_core_genes"

    # Process bacteria
    logger.info("\n--- Bacterial 9-mer Counting ---")
    process_domain("Bacteria", bact_dir, "bacterial", N_CPUS)

    # Process archaea
    logger.info("\n--- Archaeal 9-mer Counting ---")
    process_domain("Archaea", arch_dir, "archaeal", N_CPUS)

    logger.info("\n=== 9-mer Counting Complete ===")


if __name__ == "__main__":
    main()
