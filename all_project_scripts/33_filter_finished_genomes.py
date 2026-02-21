#!/usr/bin/env python3
"""
Step 0: Filter Test Genomes to Finished Only

Loads test genome metadata and cross-references with GTDB metadata to
identify genomes with NCBI assembly level "Complete Genome" or "Chromosome".

Output: data/splits/test_finished_genomes.tsv (same format as test_genomes.tsv)
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'

def main():
    print("="*70)
    print("Filter Test Genomes to Finished Only (Complete Genome + Chromosome)")
    print("="*70)

    # Load test genomes
    test_path = SPLITS_DIR / 'test_genomes.tsv'
    test_df = pd.read_csv(test_path, sep='\t')
    print(f"\nTotal test genomes: {len(test_df)}")

    # Load GTDB metadata for assembly level
    print("\nLoading GTDB metadata...")
    bac = pd.read_csv(DATA_DIR / 'gtdb' / 'bac120_metadata.tsv.gz', sep='\t',
                       usecols=['accession', 'ncbi_assembly_level'])
    ar = pd.read_csv(DATA_DIR / 'gtdb' / 'ar53_metadata.tsv.gz', sep='\t',
                      usecols=['accession', 'ncbi_assembly_level'])
    gtdb = pd.concat([bac, ar], ignore_index=True)
    print(f"GTDB metadata loaded: {len(gtdb)} genomes")

    # Merge assembly level onto test genomes
    test_df = test_df.merge(
        gtdb.rename(columns={'accession': 'gtdb_accession'}),
        on='gtdb_accession',
        how='left'
    )

    # Check for missing assembly level info
    n_missing = test_df['ncbi_assembly_level'].isna().sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} test genomes have no assembly level info")

    # Assembly level breakdown
    print(f"\nAssembly level breakdown (all test genomes):")
    for level, count in test_df['ncbi_assembly_level'].value_counts().items():
        pct = 100 * count / len(test_df)
        print(f"  {level:<20s}: {count:>5d} ({pct:.1f}%)")

    # Filter to finished genomes only
    finished_levels = ['Complete Genome', 'Chromosome']
    finished_df = test_df[test_df['ncbi_assembly_level'].isin(finished_levels)].copy()
    print(f"\nFinished genomes (Complete Genome + Chromosome): {len(finished_df)}")

    # Phylum distribution
    print(f"\nPhylum distribution of finished genomes ({finished_df['phylum'].nunique()} phyla):")
    phylum_counts = finished_df['phylum'].value_counts()
    for phylum, count in phylum_counts.head(20).items():
        pct = 100 * count / len(finished_df)
        print(f"  {phylum:<30s}: {count:>4d} ({pct:.1f}%)")
    if len(phylum_counts) > 20:
        remaining = phylum_counts.iloc[20:].sum()
        print(f"  {'(other phyla)':<30s}: {remaining:>4d}")

    # Domain distribution
    print(f"\nDomain distribution:")
    for domain, count in finished_df['domain'].value_counts().items():
        print(f"  {domain}: {count}")

    # Assembly level within finished
    print(f"\nAssembly level within finished:")
    for level, count in finished_df['ncbi_assembly_level'].value_counts().items():
        print(f"  {level}: {count}")

    # Drop the ncbi_assembly_level column to match original format
    # (keep it for reference, but also save a version without it)
    output_path = SPLITS_DIR / 'test_finished_genomes.tsv'
    # Save WITH assembly level for reference
    finished_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved: {output_path} ({len(finished_df)} genomes)")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total test genomes:     {len(test_df)}")
    print(f"Finished genomes:       {len(finished_df)} ({100*len(finished_df)/len(test_df):.1f}%)")
    print(f"  Complete Genome:      {(finished_df['ncbi_assembly_level']=='Complete Genome').sum()}")
    print(f"  Chromosome:           {(finished_df['ncbi_assembly_level']=='Chromosome').sum()}")
    print(f"Non-finished:           {len(test_df) - len(finished_df)}")
    print(f"Unique phyla:           {finished_df['phylum'].nunique()}")
    print(f"Output file:            {output_path}")


if __name__ == '__main__':
    main()
