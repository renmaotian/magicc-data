#!/usr/bin/env python3
"""
11_select_kmers.py
Select the ~10,000 most prevalent canonical 9-mers from bacterial and archaeal core genes.

- 9,000 most prevalent from bacterial core genes (by genome breadth)
- 1,000 most prevalent from archaeal core genes (by genome breadth)
- Merge and deduplicate
- Save the final k-mer list for downstream use
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

PROJ = Path("/home/tianrm/projects/magicc2")
OUT_DIR = PROJ / "data/kmer_selection"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUT_DIR / "kmer_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def canonical_kmer(kmer):
    """Return the canonical form of a k-mer (lexicographically smaller of k-mer and reverse complement)."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    rc = ''.join(complement.get(b, 'N') for b in reversed(kmer.upper()))
    return min(kmer.upper(), rc)


def main():
    output_file = OUT_DIR / "selected_kmers.txt"

    logger.info("=== K-mer Feature Selection ===")

    # Load prevalence data
    bact_prev = pd.read_csv(OUT_DIR / "bacterial_kmer_prevalence.tsv", sep='\t')
    arch_prev = pd.read_csv(OUT_DIR / "archaeal_kmer_prevalence.tsv", sep='\t')

    logger.info(f"Bacterial k-mers: {len(bact_prev)} total, max prevalence: {bact_prev['prevalence'].max()}")
    logger.info(f"Archaeal k-mers: {len(arch_prev)} total, max prevalence: {arch_prev['prevalence'].max()}")

    # Already sorted by prevalence descending in the TSV files
    # Verify sorting
    assert bact_prev['prevalence'].is_monotonic_decreasing or \
           (bact_prev['prevalence'].iloc[0] >= bact_prev['prevalence'].iloc[-1]), \
           "Bacterial prevalence not sorted"
    assert arch_prev['prevalence'].is_monotonic_decreasing or \
           (arch_prev['prevalence'].iloc[0] >= arch_prev['prevalence'].iloc[-1]), \
           "Archaeal prevalence not sorted"

    # Select top 9,000 bacterial k-mers
    N_BACT = 9000
    N_ARCH = 1000

    bact_top = bact_prev.head(N_BACT)
    arch_top = arch_prev.head(N_ARCH)

    logger.info(f"\nBacterial top {N_BACT} k-mers:")
    logger.info(f"  Prevalence range: {bact_top['prevalence'].min()} - {bact_top['prevalence'].max()} (out of 1000 genomes)")
    logger.info(f"  Prevalence at cutoff (rank {N_BACT}): {bact_top['prevalence'].iloc[-1]}")

    logger.info(f"\nArchaeal top {N_ARCH} k-mers:")
    logger.info(f"  Prevalence range: {arch_top['prevalence'].min()} - {arch_top['prevalence'].max()} (out of 1000 genomes)")
    logger.info(f"  Prevalence at cutoff (rank {N_ARCH}): {arch_top['prevalence'].iloc[-1]}")

    # Get k-mer sets
    bact_kmers = set(bact_top['kmer'].str.upper())
    arch_kmers = set(arch_top['kmer'].str.upper())

    # Verify all are canonical
    n_non_canonical_bact = sum(1 for k in bact_kmers if canonical_kmer(k) != k)
    n_non_canonical_arch = sum(1 for k in arch_kmers if canonical_kmer(k) != k)
    logger.info(f"\nCanonical check:")
    logger.info(f"  Bacterial: {n_non_canonical_bact} non-canonical (should be 0)")
    logger.info(f"  Archaeal: {n_non_canonical_arch} non-canonical (should be 0)")

    # Merge and deduplicate
    overlap = bact_kmers & arch_kmers
    merged = sorted(bact_kmers | arch_kmers)

    logger.info(f"\n=== Merge Results ===")
    logger.info(f"  Bacterial k-mers: {len(bact_kmers)}")
    logger.info(f"  Archaeal k-mers: {len(arch_kmers)}")
    logger.info(f"  Overlap: {len(overlap)} k-mers shared between bacterial and archaeal")
    logger.info(f"  Final merged set: {len(merged)} unique canonical 9-mers")

    # Save final k-mer list
    with open(output_file, 'w') as f:
        for kmer in merged:
            f.write(kmer + '\n')
    logger.info(f"\nSaved to: {output_file}")

    # Prevalence distribution analysis
    logger.info(f"\n=== Prevalence Distribution Analysis ===")

    # For the merged set, show prevalence in both domains
    bact_prev_dict = dict(zip(bact_prev['kmer'].str.upper(), bact_prev['prevalence']))
    arch_prev_dict = dict(zip(arch_prev['kmer'].str.upper(), arch_prev['prevalence']))

    bact_only = bact_kmers - arch_kmers
    arch_only = arch_kmers - bact_kmers

    logger.info(f"  K-mers from bacteria only: {len(bact_only)}")
    logger.info(f"  K-mers from archaea only: {len(arch_only)}")
    logger.info(f"  K-mers from both (overlap): {len(overlap)}")

    # Prevalence stats for overlap k-mers
    if overlap:
        overlap_bact_prev = [bact_prev_dict.get(k, 0) for k in overlap]
        overlap_arch_prev = [arch_prev_dict.get(k, 0) for k in overlap]
        logger.info(f"\n  Overlap k-mers bacterial prevalence: mean={np.mean(overlap_bact_prev):.1f}, "
                    f"median={np.median(overlap_bact_prev):.0f}")
        logger.info(f"  Overlap k-mers archaeal prevalence: mean={np.mean(overlap_arch_prev):.1f}, "
                    f"median={np.median(overlap_arch_prev):.0f}")

    # K-mer length verification
    kmer_lengths = [len(k) for k in merged]
    assert all(l == 9 for l in kmer_lengths), f"Not all k-mers are 9-mers: {set(kmer_lengths)}"
    logger.info(f"\n  All k-mers verified as 9-mers: YES")

    # Save detailed stats
    stats = {
        'n_bacterial_selected': len(bact_kmers),
        'n_archaeal_selected': len(arch_kmers),
        'n_overlap': len(overlap),
        'n_bact_only': len(bact_only),
        'n_arch_only': len(arch_only),
        'n_final_merged': len(merged),
        'bacterial_prevalence_cutoff': int(bact_top['prevalence'].iloc[-1]),
        'archaeal_prevalence_cutoff': int(arch_top['prevalence'].iloc[-1]),
        'bacterial_max_prevalence': int(bact_top['prevalence'].max()),
        'archaeal_max_prevalence': int(arch_top['prevalence'].max()),
        'output_file': str(output_file),
    }
    stats_file = OUT_DIR / "kmer_selection_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nStats saved to: {stats_file}")

    # Save annotated k-mer list with source and prevalence info
    annotated = []
    for kmer in merged:
        in_bact = kmer in bact_kmers
        in_arch = kmer in arch_kmers
        source = []
        if in_bact:
            source.append('bacterial')
        if in_arch:
            source.append('archaeal')
        annotated.append({
            'kmer': kmer,
            'source': '+'.join(source),
            'bacterial_prevalence': bact_prev_dict.get(kmer, 0),
            'archaeal_prevalence': arch_prev_dict.get(kmer, 0),
        })

    annotated_df = pd.DataFrame(annotated)
    annotated_file = OUT_DIR / "selected_kmers_annotated.tsv"
    annotated_df.to_csv(annotated_file, sep='\t', index=False)
    logger.info(f"Annotated k-mer list saved to: {annotated_file}")

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {len(merged)} unique canonical 9-mers selected")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
