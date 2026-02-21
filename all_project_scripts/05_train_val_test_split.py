#!/usr/bin/env python3
"""
Step 6: Split 100,000 selected reference genomes into:
  - 80,000 training
  - 10,000 validation
  - 10,000 test

Using stratified sampling by phylum to ensure:
  - Proportional representation of each phylum in all splits
  - No shared genomes across splits
  - Reproducible results (fixed random seed)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

np.random.seed(42)

# Paths
INPUT_FILE = "/home/tianrm/projects/magicc2/data/gtdb/selected_100k_genomes.tsv"
OUTPUT_DIR = "/home/tianrm/projects/magicc2/data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load selected genomes
    print(f"Loading selected genomes from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"  Total genomes: {len(df):,}")
    print(f"  Unique phyla: {df['phylum'].nunique()}")

    # For stratification, we need to handle phyla with very few genomes
    # sklearn's train_test_split requires at least 2 members per class for stratification
    # For 80/10/10 split done in two steps (80/20 then 50/50), we need at least:
    # - 2 in valtest group after first split => need >=10 total (20% of 10 = 2)
    # Group phyla with <10 genomes for stratification purposes
    phylum_counts = df["phylum"].value_counts()

    # Create stratification column
    # Need enough members so that after 80/20 split, each group in the 20% portion
    # still has >=2 members for the second 50/50 split
    min_for_stratify = 10
    rare_phyla = phylum_counts[phylum_counts < min_for_stratify].index.tolist()
    df["strat_group"] = df["phylum"].copy()
    df.loc[df["phylum"].isin(rare_phyla), "strat_group"] = "_RARE_PHYLA_"

    rare_count = df[df["strat_group"] == "_RARE_PHYLA_"].shape[0]
    print(f"  Phyla with <{min_for_stratify} genomes (grouped for stratification): {len(rare_phyla)} phyla, {rare_count} genomes")

    # Manual assignment approach for robust 3-way split:
    # For each phylum, assign genomes to train/val/test maintaining 80/10/10 ratio
    print("\nPerforming manual stratified 3-way split...")
    rng = np.random.RandomState(42)

    train_indices = []
    val_indices = []
    test_indices = []

    for phylum in df["phylum"].unique():
        phylum_idx = df[df["phylum"] == phylum].index.values.copy()
        rng.shuffle(phylum_idx)
        n = len(phylum_idx)

        if n == 1:
            # Single genome: assign to train (largest split)
            train_indices.extend(phylum_idx)
        elif n == 2:
            # Two genomes: one to train, one to val
            train_indices.append(phylum_idx[0])
            val_indices.append(phylum_idx[1])
        elif n <= 4:
            # 3-4 genomes: distribute with at least 1 each to val and test
            n_test = 1
            n_val = 1
            n_train = n - n_test - n_val
            train_indices.extend(phylum_idx[:n_train])
            val_indices.append(phylum_idx[n_train])
            test_indices.append(phylum_idx[n_train + 1])
        else:
            # Standard 80/10/10 split
            n_test = max(1, round(n * 0.1))
            n_val = max(1, round(n * 0.1))
            n_train = n - n_val - n_test
            train_indices.extend(phylum_idx[:n_train])
            val_indices.extend(phylum_idx[n_train:n_train + n_val])
            test_indices.extend(phylum_idx[n_train + n_val:])

    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()

    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    # Verify no overlap
    train_accs = set(train_df["gtdb_accession"])
    val_accs = set(val_df["gtdb_accession"])
    test_accs = set(test_df["gtdb_accession"])
    assert len(train_accs & val_accs) == 0, "Train/Val overlap!"
    assert len(train_accs & test_accs) == 0, "Train/Test overlap!"
    assert len(val_accs & test_accs) == 0, "Val/Test overlap!"
    print("  No overlap between splits (verified)")

    # Drop helper column
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["strat_group"], inplace=True)

    # Save splits
    train_file = os.path.join(OUTPUT_DIR, "train_genomes.tsv")
    val_file = os.path.join(OUTPUT_DIR, "val_genomes.tsv")
    test_file = os.path.join(OUTPUT_DIR, "test_genomes.tsv")

    train_df.to_csv(train_file, sep="\t", index=False)
    val_df.to_csv(val_file, sep="\t", index=False)
    test_df.to_csv(test_file, sep="\t", index=False)

    print(f"\nSaved splits to:")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Test:  {test_file}")

    # Also save accession-only lists
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        acc_file = os.path.join(OUTPUT_DIR, f"{split_name}_accessions.txt")
        split_df["ncbi_accession"].to_csv(acc_file, index=False, header=False)

    # Detailed statistics per split
    print(f"\n{'='*80}")
    print("SPLIT STATISTICS")
    print(f"{'='*80}")

    stats = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n--- {split_name.upper()} ({len(split_df):,} genomes) ---")
        phylum_dist = split_df["phylum"].value_counts()

        split_stats = {
            "total": len(split_df),
            "phyla": len(phylum_dist),
            "domains": split_df["domain"].value_counts().to_dict(),
            "phylum_distribution": phylum_dist.to_dict(),
        }
        stats[split_name] = split_stats

        print(f"  Phyla represented: {len(phylum_dist)}")
        print(f"  Domain distribution:")
        for domain, count in split_df["domain"].value_counts().items():
            print(f"    {domain}: {count:,}")

    # Compare phylum proportions across splits to verify stratification
    print(f"\n{'='*80}")
    print("PHYLUM REPRESENTATION ACROSS SPLITS (top 20)")
    print(f"{'='*80}")
    print(f"{'Phylum':<35} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 80)

    all_phyla = df["phylum"].value_counts()
    for phylum in all_phyla.index[:20]:
        n_train = train_df[train_df["phylum"] == phylum].shape[0]
        n_val = val_df[val_df["phylum"] == phylum].shape[0]
        n_test = test_df[test_df["phylum"] == phylum].shape[0]
        total = n_train + n_val + n_test
        print(f"  {phylum:<33} {n_train:>8,} {n_val:>8,} {n_test:>8,} {total:>8,}")

    print("-" * 80)
    print(f"  {'TOTAL':<33} {len(train_df):>8,} {len(val_df):>8,} {len(test_df):>8,} {len(df):>8,}")

    # Show all phyla
    print(f"\n{'='*80}")
    print("ALL PHYLA ACROSS SPLITS")
    print(f"{'='*80}")
    print(f"{'Phylum':<35} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 80)
    for phylum in all_phyla.index:
        n_train = train_df[train_df["phylum"] == phylum].shape[0]
        n_val = val_df[val_df["phylum"] == phylum].shape[0]
        n_test = test_df[test_df["phylum"] == phylum].shape[0]
        total = n_train + n_val + n_test
        print(f"  {phylum:<33} {n_train:>8,} {n_val:>8,} {n_test:>8,} {total:>8,}")

    # Save statistics as JSON
    stats_file = os.path.join(OUTPUT_DIR, "split_statistics.json")
    # Convert numpy types to Python native for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj

    stats = convert_types(stats)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to: {stats_file}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Train: {len(train_df):>8,} genomes ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):>8,} genomes ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):>8,} genomes ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total: {len(df):>8,} genomes")
    print(f"  Phyla in train: {train_df['phylum'].nunique()}")
    print(f"  Phyla in val:   {val_df['phylum'].nunique()}")
    print(f"  Phyla in test:  {test_df['phylum'].nunique()}")
    print(f"  No overlapping genomes between splits: VERIFIED")


if __name__ == "__main__":
    main()
