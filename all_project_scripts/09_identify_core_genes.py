#!/usr/bin/env python3
"""
09_identify_core_genes.py
Identify core genes in selected bacterial and archaeal genomes using Prodigal + HMMER.

For each genome:
1. Run Prodigal to predict genes (protein + nucleotide)
2. Run hmmsearch with appropriate HMM profile (--cut_tc)
3. Extract DNA sequences of identified core genes
4. Concatenate core gene DNA into a single FASTA per genome

Uses multiprocessing for parallel execution.
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
from collections import defaultdict
import time
import logging

PROJ = Path("/home/tianrm/projects/magicc2")
OUT_DIR = PROJ / "data/kmer_selection"
BCG_HMM = PROJ / "85_bcg.hmm"
UACG_HMM = PROJ / "uacg.hmm"

# Use 90% of CPUs
N_CPUS = max(1, int(os.cpu_count() * 0.9))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUT_DIR / "core_gene_identification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_prodigal_ids(faa_path, fna_path):
    """
    Parse prodigal output to build a mapping from protein ID to nucleotide ID.
    Prodigal uses the same ID scheme for both: >contig_N where N is the gene number.
    Returns dict of protein_id -> nucleotide_sequence.
    """
    # Read nucleotide sequences
    nuc_seqs = {}
    current_id = None
    current_seq = []
    with open(fna_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    nuc_seqs[current_id] = ''.join(current_seq)
                # Prodigal header: >contig_genenum # start # end # strand # info
                current_id = line.split()[0][1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        nuc_seqs[current_id] = ''.join(current_seq)

    return nuc_seqs


def process_genome(args):
    """
    Process a single genome: Prodigal + hmmsearch + extract core gene DNA.
    Returns: (accession, n_hits, core_gene_dna_path or None, error_msg or None)
    """
    accession, fasta_path, hmm_path, out_dir, domain = args

    core_dna_path = out_dir / f"{accession}_core_genes.fna"

    # Check if already processed
    if core_dna_path.exists() and os.path.getsize(core_dna_path) > 0:
        # Count sequences in existing file
        n_seqs = 0
        with open(core_dna_path) as f:
            for line in f:
                if line.startswith('>'):
                    n_seqs += 1
        return (accession, n_seqs, str(core_dna_path), None)

    if not os.path.exists(fasta_path):
        return (accession, 0, None, f"FASTA not found: {fasta_path}")

    tmpdir = tempfile.mkdtemp(prefix=f"magicc_{accession}_")

    try:
        prot_path = os.path.join(tmpdir, "proteins.faa")
        nuc_path = os.path.join(tmpdir, "genes.fna")
        gff_path = os.path.join(tmpdir, "genes.gff")
        tbl_path = os.path.join(tmpdir, "hmm_hits.tbl")

        # Step 1: Run Prodigal
        proc = subprocess.run(
            ["prodigal", "-i", fasta_path, "-a", prot_path, "-d", nuc_path,
             "-o", gff_path, "-f", "gff", "-p", "single", "-q"],
            capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0:
            return (accession, 0, None, f"Prodigal failed: {proc.stderr[:200]}")

        if not os.path.exists(prot_path) or os.path.getsize(prot_path) == 0:
            return (accession, 0, None, "Prodigal produced no proteins")

        # Step 2: Run hmmsearch with --cut_tc
        proc = subprocess.run(
            ["hmmsearch", "--cut_tc", "--tblout", tbl_path, "--noali",
             "--cpu", "1", str(hmm_path), prot_path],
            capture_output=True, text=True, timeout=600
        )
        if proc.returncode != 0:
            return (accession, 0, None, f"hmmsearch failed: {proc.stderr[:200]}")

        # Step 3: Parse hmmsearch results
        hit_protein_ids = set()
        if os.path.exists(tbl_path):
            with open(tbl_path) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 1:
                        hit_protein_ids.add(parts[0])  # target name (protein ID)

        if not hit_protein_ids:
            # Write empty file to mark as processed
            with open(core_dna_path, 'w') as f:
                pass
            return (accession, 0, str(core_dna_path), None)

        # Step 4: Extract corresponding DNA sequences
        nuc_seqs = parse_prodigal_ids(prot_path, nuc_path)

        n_extracted = 0
        with open(core_dna_path, 'w') as out_f:
            for prot_id in sorted(hit_protein_ids):
                if prot_id in nuc_seqs:
                    out_f.write(f">{accession}_{prot_id}\n")
                    seq = nuc_seqs[prot_id]
                    # Write in 80-char lines
                    for i in range(0, len(seq), 80):
                        out_f.write(seq[i:i+80] + '\n')
                    n_extracted += 1

        return (accession, n_extracted, str(core_dna_path), None)

    except subprocess.TimeoutExpired:
        return (accession, 0, None, "Timeout")
    except Exception as e:
        return (accession, 0, None, str(e)[:200])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_domain(domain_name, genome_tsv, hmm_path, out_subdir, n_workers):
    """Run core gene identification for a set of genomes."""
    out_dir = OUT_DIR / out_subdir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(genome_tsv, sep='\t')
    logger.info(f"Processing {len(df)} {domain_name} genomes with {n_workers} workers")
    logger.info(f"HMM: {hmm_path}")
    logger.info(f"Output: {out_dir}")

    # Prepare args
    tasks = []
    for _, row in df.iterrows():
        acc = row['ncbi_accession']
        fasta = row['fasta_path']
        tasks.append((acc, fasta, hmm_path, out_dir, domain_name))

    # Process in parallel
    results = []
    n_done = 0
    n_total = len(tasks)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_genome, task): task[0] for task in tasks}

        for future in as_completed(futures):
            acc = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append((acc, 0, None, str(e)[:200]))

            n_done += 1
            if n_done % 100 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n_total - n_done) / rate if rate > 0 else 0
                logger.info(f"  {domain_name}: {n_done}/{n_total} ({n_done/n_total*100:.1f}%) "
                           f"- {rate:.1f} genomes/sec - ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"  {domain_name}: Completed in {elapsed:.1f}s ({len(tasks)/elapsed:.1f} genomes/sec)")

    # Summarize results
    n_success = sum(1 for r in results if r[2] is not None)
    n_error = sum(1 for r in results if r[3] is not None)
    hits_per_genome = [r[1] for r in results if r[2] is not None]
    errors = [(r[0], r[3]) for r in results if r[3] is not None]

    stats = {
        'domain': domain_name,
        'total_genomes': len(tasks),
        'successful': n_success,
        'errors': n_error,
        'hits_per_genome_mean': float(np.mean(hits_per_genome)) if hits_per_genome else 0,
        'hits_per_genome_median': float(np.median(hits_per_genome)) if hits_per_genome else 0,
        'hits_per_genome_min': int(np.min(hits_per_genome)) if hits_per_genome else 0,
        'hits_per_genome_max': int(np.max(hits_per_genome)) if hits_per_genome else 0,
        'hits_per_genome_std': float(np.std(hits_per_genome)) if hits_per_genome else 0,
        'genomes_with_zero_hits': sum(1 for h in hits_per_genome if h == 0),
        'elapsed_seconds': elapsed,
    }

    if errors:
        logger.warning(f"  {domain_name}: {n_error} errors. First 5:")
        for acc, err in errors[:5]:
            logger.warning(f"    {acc}: {err}")

    # Save stats
    stats_path = out_dir / "core_gene_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Save per-genome results
    results_df = pd.DataFrame(results, columns=['accession', 'n_core_genes', 'core_gene_path', 'error'])
    results_df.to_csv(out_dir / "core_gene_results.tsv", sep='\t', index=False)

    return stats, results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info(f"=== Core Gene Identification ===")
    logger.info(f"CPUs: {N_CPUS}")
    logger.info(f"Bacterial HMM: {BCG_HMM} (85 models)")
    logger.info(f"Archaeal HMM: {UACG_HMM} (128 models)")

    bact_tsv = OUT_DIR / "selected_bacterial_1000.tsv"
    arch_tsv = OUT_DIR / "selected_archaeal_1000.tsv"

    # Process bacteria
    logger.info("\n--- Bacterial Core Gene Identification ---")
    bact_stats_path = OUT_DIR / "bacterial_core_genes/core_gene_stats.json"
    if bact_stats_path.exists():
        with open(bact_stats_path) as f:
            bact_stats = json.load(f)
        logger.info(f"Bacterial core genes already identified: {bact_stats['successful']} genomes")
        # Check if all are done
        bact_results_path = OUT_DIR / "bacterial_core_genes/core_gene_results.tsv"
        bact_results_df = pd.read_csv(bact_results_path, sep='\t')
        n_missing = bact_results_df['core_gene_path'].isna().sum()
        if n_missing > 0:
            logger.info(f"  Re-running {n_missing} failed genomes...")
            bact_stats, bact_results = run_domain("Bacteria", bact_tsv, BCG_HMM,
                                                   "bacterial_core_genes", N_CPUS)
        else:
            bact_results = None
    else:
        bact_stats, bact_results = run_domain("Bacteria", bact_tsv, BCG_HMM,
                                               "bacterial_core_genes", N_CPUS)

    # Process archaea
    logger.info("\n--- Archaeal Core Gene Identification ---")
    arch_stats_path = OUT_DIR / "archaeal_core_genes/core_gene_stats.json"
    if arch_stats_path.exists():
        with open(arch_stats_path) as f:
            arch_stats = json.load(f)
        logger.info(f"Archaeal core genes already identified: {arch_stats['successful']} genomes")
        arch_results_path = OUT_DIR / "archaeal_core_genes/core_gene_results.tsv"
        arch_results_df = pd.read_csv(arch_results_path, sep='\t')
        n_missing = arch_results_df['core_gene_path'].isna().sum()
        if n_missing > 0:
            logger.info(f"  Re-running {n_missing} failed genomes...")
            arch_stats, arch_results = run_domain("Archaea", arch_tsv, UACG_HMM,
                                                   "archaeal_core_genes", N_CPUS)
        else:
            arch_results = None
    else:
        arch_stats, arch_results = run_domain("Archaea", arch_tsv, UACG_HMM,
                                               "archaeal_core_genes", N_CPUS)

    # Print summary
    logger.info("\n=== Summary ===")
    for label, stats in [("Bacteria", bact_stats), ("Archaea", arch_stats)]:
        logger.info(f"\n{label}:")
        logger.info(f"  Genomes processed: {stats['successful']}/{stats['total_genomes']}")
        logger.info(f"  Core genes per genome: mean={stats['hits_per_genome_mean']:.1f}, "
                    f"median={stats['hits_per_genome_median']:.1f}, "
                    f"min={stats['hits_per_genome_min']}, max={stats['hits_per_genome_max']}")
        logger.info(f"  Genomes with zero hits: {stats['genomes_with_zero_hits']}")
        logger.info(f"  Elapsed: {stats['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
