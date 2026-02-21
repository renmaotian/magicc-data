#!/usr/bin/env python3
"""
08_select_representative_genomes.py
Select 1,000 representative bacterial and 1,000 representative archaeal genomes
(stratified by phylum) from the training set for k-mer feature selection.

Uses train_genomes.tsv to ensure we only use training data for feature selection.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path("/home/tianrm/projects/magicc2")
TRAIN_TSV = PROJ / "data/splits/train_genomes.tsv"
OUT_DIR = PROJ / "data/kmer_selection"
SEED = 42

def stratified_sample(df, n_target, domain_name, seed=42):
    """
    Stratified sampling by phylum.
    Uses square-root proportional sampling:
    - Each phylum gets at least 1 genome
    - Remaining slots are allocated proportional to sqrt(phylum_size)
    If total available < n_target, use all.
    """
    rng = np.random.RandomState(seed)

    phylum_counts = df['phylum'].value_counts()
    n_phyla = len(phylum_counts)
    n_available = len(df)

    if n_available <= n_target:
        print(f"  {domain_name}: Only {n_available} genomes available (target {n_target}), using all")
        return df.copy()

    # Each phylum gets at least 1
    allocation = {p: 1 for p in phylum_counts.index}
    remaining = n_target - n_phyla

    if remaining > 0:
        # Allocate remaining proportional to sqrt(count)
        sqrt_counts = {p: np.sqrt(c) for p, c in phylum_counts.items()}
        total_sqrt = sum(sqrt_counts.values())

        for p in phylum_counts.index:
            extra = int(remaining * sqrt_counts[p] / total_sqrt)
            allocation[p] += extra

        # Distribute any leftover due to rounding (largest phyla first)
        current_total = sum(allocation.values())
        deficit = n_target - current_total
        for p in phylum_counts.index:
            if deficit <= 0:
                break
            allocation[p] += 1
            deficit -= 1

    # Cap each phylum allocation at its available count
    for p in phylum_counts.index:
        allocation[p] = min(allocation[p], phylum_counts[p])

    # If capping caused deficit, redistribute to uncapped phyla
    current_total = sum(allocation.values())
    if current_total < n_target:
        deficit = n_target - current_total
        for p in phylum_counts.index:
            if deficit <= 0:
                break
            can_add = phylum_counts[p] - allocation[p]
            add = min(can_add, deficit)
            allocation[p] += add
            deficit -= add

    # Sample from each phylum
    selected = []
    for p, n in allocation.items():
        phylum_df = df[df['phylum'] == p]
        sampled = phylum_df.sample(n=n, random_state=rng)
        selected.append(sampled)

    result = pd.concat(selected, ignore_index=True)
    print(f"  {domain_name}: Selected {len(result)} genomes from {n_phyla} phyla (target {n_target})")
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bact_out = OUT_DIR / "selected_bacterial_1000.tsv"
    arch_out = OUT_DIR / "selected_archaeal_1000.tsv"

    # Check if already done
    if bact_out.exists() and arch_out.exists():
        bact_df = pd.read_csv(bact_out, sep='\t')
        arch_df = pd.read_csv(arch_out, sep='\t')
        print(f"Already completed: {len(bact_df)} bacterial, {len(arch_df)} archaeal genomes selected")
        print("Delete output files to re-run.")
        return

    print("Loading training genomes...")
    df = pd.read_csv(TRAIN_TSV, sep='\t')
    print(f"  Total: {len(df)} genomes ({df['domain'].value_counts().to_dict()})")

    # Filter to genomes with valid fasta_path
    df = df[df['fasta_path'].notna()].copy()
    # Verify fasta files exist
    df['fasta_exists'] = df['fasta_path'].apply(lambda x: os.path.exists(x))
    n_missing = (~df['fasta_exists']).sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} genomes have missing FASTA files, excluding them")
        df = df[df['fasta_exists']].copy()
    df.drop(columns=['fasta_exists'], inplace=True)

    bacteria = df[df['domain'] == 'Bacteria'].copy()
    archaea = df[df['domain'] == 'Archaea'].copy()

    print(f"\nSelecting representative genomes (seed={SEED}):")
    bact_selected = stratified_sample(bacteria, 1000, "Bacteria", seed=SEED)
    arch_selected = stratified_sample(archaea, 1000, "Archaea", seed=SEED)

    # Save
    bact_selected.to_csv(bact_out, sep='\t', index=False)
    arch_selected.to_csv(arch_out, sep='\t', index=False)

    # Print summary statistics
    print(f"\n=== Bacterial Selection Summary ===")
    print(f"  Total: {len(bact_selected)} genomes from {bact_selected['phylum'].nunique()} phyla")
    print(f"  Top 5 phyla:")
    for p, c in bact_selected['phylum'].value_counts().head(5).items():
        print(f"    {p}: {c}")
    print(f"  Bottom 5 phyla:")
    for p, c in bact_selected['phylum'].value_counts().tail(5).items():
        print(f"    {p}: {c}")

    print(f"\n=== Archaeal Selection Summary ===")
    print(f"  Total: {len(arch_selected)} genomes from {arch_selected['phylum'].nunique()} phyla")
    print(f"  Phylum distribution:")
    for p, c in arch_selected['phylum'].value_counts().items():
        print(f"    {p}: {c}")

    print(f"\nOutputs:")
    print(f"  Bacterial: {bact_out}")
    print(f"  Archaeal: {arch_out}")


if __name__ == "__main__":
    main()
