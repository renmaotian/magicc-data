#!/usr/bin/env python3
"""
Steps 4 & 5: Stratified sampling across phyla to select exactly 100,000 reference genomes.
Over-sample underrepresented lineages per protocol:
  - >= 2000 Patescibacteria (GTDB: Patescibacteriota) if available
  - >= 300 DPANN archaea if available
  - >= 1000 other candidate phyla if available

Strategy:
1. First, reserve all genomes from underrepresented lineages (take all available)
2. Set minimum targets for special lineages
3. Distribute remaining slots proportionally across phyla using square-root proportional
   sampling to ensure balanced representation (pure proportional would over-represent
   Pseudomonadota and Bacillota)
4. Cap at available genomes per phylum
5. Fill remaining slots proportionally
"""

import pandas as pd
import numpy as np
import os
import sys

np.random.seed(42)

# Config
INPUT_FILE = "/home/tianrm/projects/magicc2/data/gtdb/filtered_genomes.tsv"
OUTPUT_DIR = "/home/tianrm/projects/magicc2/data/gtdb"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "selected_100k_genomes.tsv")
TARGET_TOTAL = 100_000

# DPANN archaea phyla in GTDB taxonomy
DPANN_PHYLA = [
    "Nanoarchaeota", "Aenigmarchaeota", "Diapherotrites",
    "Micrarchaeota", "Woesearchaeota", "Altiarchaeota",
    "Huberarchaeota", "Undinarchaeota", "Iainarchaeota",
    "Nanohaloarchaeota", "Nanohalarchaeota", "Asgardarchaeota",
    "Hydrothermarchaeota", "Korarchaeota",
]

# Patescibacteria phylum in GTDB
PATESCIBACTERIA_PHYLA = ["Patescibacteriota"]

# Other candidate phyla (small/rare phyla that need over-sampling)
# These are phyla with generally small representation
CANDIDATE_PHYLA = [
    "Elusimicrobiota", "Omnitrophota", "Margulisbacteria",
    "Bdellovibrionota", "Bdellovibrionota_B", "Bdellovibrionota_G",
    "Zixibacteria", "Edwardsbacteria", "Babelota",
    "Cloacimonadota", "WOR-3", "Vulcanimicrobiota",
    "Zhuqueibacterota", "Marinisomatota", "Fermentibacterota",
    "Calditrichota", "Methylomirabilota", "Caldisericota",
    "Atribacterota", "Goldbacteria", "Electryoneota",
    "Schekmanbacteria", "Thermodesulfobiota", "Bipolaricaulota",
    "Firestonebacteria", "Joyebacterota", "Muiribacteriota",
    "Hydrogenedentota", "Eisenbacteria", "Mcinerneyibacteriota",
    "Coprothermobacterota", "Thermosulfidibacterota",
]


def main():
    # Load filtered genomes
    print(f"Loading filtered genomes from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"  Total available: {len(df):,}")

    # Parse phylum counts
    phylum_counts = df["phylum"].value_counts()
    print(f"  Unique phyla: {len(phylum_counts)}")

    # Identify special lineage genomes
    is_patescibacteria = df["phylum"].isin(PATESCIBACTERIA_PHYLA)
    is_dpann = df["phylum"].isin(DPANN_PHYLA)
    is_candidate = df["phylum"].isin(CANDIDATE_PHYLA)

    patescibacteria_available = is_patescibacteria.sum()
    dpann_available = is_dpann.sum()
    candidate_available = is_candidate.sum()

    print(f"\n  Special lineages available:")
    print(f"    Patescibacteria: {patescibacteria_available:,} (target: >=2000)")
    print(f"    DPANN archaea:   {dpann_available:,} (target: >=300)")
    print(f"    Candidate phyla: {candidate_available:,} (target: >=1000)")

    # Step 1: Determine minimum guarantees for special lineages
    # Take ALL available since they're underrepresented
    patescibacteria_target = min(patescibacteria_available, max(2000, patescibacteria_available))
    dpann_target = min(dpann_available, max(300, dpann_available))
    candidate_target = min(candidate_available, max(1000, candidate_available))

    print(f"\n  Guaranteed allocation (all available):")
    print(f"    Patescibacteria: {patescibacteria_target:,}")
    print(f"    DPANN archaea:   {dpann_target:,}")
    print(f"    Candidate phyla: {candidate_target:,}")

    # Step 2: Compute allocation for all phyla using square-root proportional sampling
    # This balances between equal representation and proportional representation
    # Small phyla get relatively more, large phyla get relatively less

    # First, guarantee at least minimum count for each phylum
    MIN_PER_PHYLUM = 1  # Take at least 1 from every phylum that has genomes

    # For overrepresented lineages (special ones), ensure all are included
    # Build the allocation
    allocation = {}

    # Start with special lineages getting all available
    special_phyla = set()
    for phylum in df["phylum"].unique():
        if phylum in PATESCIBACTERIA_PHYLA or phylum in DPANN_PHYLA or phylum in CANDIDATE_PHYLA:
            special_phyla.add(phylum)
            allocation[phylum] = phylum_counts[phylum]  # Take all

    special_total = sum(allocation.values())
    remaining_target = TARGET_TOTAL - special_total

    print(f"\n  Special lineages reserved: {special_total:,}")
    print(f"  Remaining target for other phyla: {remaining_target:,}")

    # Non-special phyla
    non_special_phyla = [p for p in phylum_counts.index if p not in special_phyla]
    non_special_counts = {p: phylum_counts[p] for p in non_special_phyla}
    non_special_total = sum(non_special_counts.values())

    # Square-root proportional allocation
    sqrt_props = {p: np.sqrt(c) for p, c in non_special_counts.items()}
    sqrt_total = sum(sqrt_props.values())

    for phylum in non_special_phyla:
        raw_alloc = int(remaining_target * sqrt_props[phylum] / sqrt_total)
        # Cap at available
        allocation[phylum] = min(raw_alloc, non_special_counts[phylum])
        # Ensure at least MIN_PER_PHYLUM
        allocation[phylum] = max(allocation[phylum], min(MIN_PER_PHYLUM, non_special_counts[phylum]))

    current_total = sum(allocation.values())
    deficit = TARGET_TOTAL - current_total

    print(f"  After sqrt-proportional allocation: {current_total:,}")
    print(f"  Deficit to fill: {deficit:,}")

    # Fill deficit by adding more from phyla that have headroom, proportional to remaining availability
    if deficit > 0:
        headroom = {p: non_special_counts[p] - allocation[p]
                    for p in non_special_phyla if allocation[p] < non_special_counts[p]}
        headroom_total = sum(headroom.values())

        if headroom_total > 0:
            for phylum in headroom:
                extra = int(deficit * headroom[phylum] / headroom_total)
                extra = min(extra, non_special_counts[phylum] - allocation[phylum])
                allocation[phylum] += extra

    current_total = sum(allocation.values())
    deficit = TARGET_TOTAL - current_total

    # Fine-tune: add 1 genome at a time from largest headroom phyla
    if deficit > 0:
        headroom_list = [(non_special_counts[p] - allocation[p], p)
                         for p in non_special_phyla if allocation[p] < non_special_counts[p]]
        headroom_list.sort(reverse=True)
        for i in range(min(deficit, len(headroom_list))):
            allocation[headroom_list[i][1]] += 1

    current_total = sum(allocation.values())
    print(f"  Final allocation total: {current_total:,}")

    # Step 3: Sample genomes according to allocation
    selected_indices = []
    for phylum, target_n in allocation.items():
        phylum_df = df[df["phylum"] == phylum]
        if len(phylum_df) <= target_n:
            selected_indices.extend(phylum_df.index.tolist())
        else:
            sampled = phylum_df.sample(n=target_n, random_state=42)
            selected_indices.extend(sampled.index.tolist())

    selected = df.loc[selected_indices].copy()
    print(f"\n  Selected genomes: {len(selected):,}")

    # Final phylum distribution
    print(f"\n{'='*70}")
    print("FINAL SELECTION - Phylum distribution:")
    print(f"{'='*70}")
    sel_phylum_counts = selected["phylum"].value_counts()
    print(f"{'Phylum':<40} {'Selected':>10} {'Available':>10} {'Pct':>8}")
    print("-" * 70)
    for phylum in sel_phylum_counts.index:
        avail = phylum_counts.get(phylum, 0)
        pct = sel_phylum_counts[phylum] / avail * 100 if avail > 0 else 0
        marker = ""
        if phylum in PATESCIBACTERIA_PHYLA:
            marker = " [PATESCIBACTERIA]"
        elif phylum in DPANN_PHYLA:
            marker = " [DPANN]"
        elif phylum in CANDIDATE_PHYLA:
            marker = " [CANDIDATE]"
        print(f"  {phylum:<38} {sel_phylum_counts[phylum]:>10,} {avail:>10,} {pct:>7.1f}%{marker}")
    print("-" * 70)
    print(f"  {'TOTAL':<38} {len(selected):>10,} {len(df):>10,}")

    # Report on special lineages
    print(f"\n  Special lineage summary:")
    n_pates = selected[selected["phylum"].isin(PATESCIBACTERIA_PHYLA)].shape[0]
    n_dpann = selected[selected["phylum"].isin(DPANN_PHYLA)].shape[0]
    n_cand = selected[selected["phylum"].isin(CANDIDATE_PHYLA)].shape[0]
    print(f"    Patescibacteria: {n_pates:,} (target >=2000, available {patescibacteria_available:,})")
    print(f"    DPANN archaea:   {n_dpann:,} (target >=300, available {dpann_available:,})")
    print(f"    Candidate phyla: {n_cand:,} (target >=1000, available {candidate_available:,})")

    # Domain distribution
    print(f"\n  Domain distribution:")
    for domain, count in selected["domain"].value_counts().items():
        print(f"    {domain}: {count:,}")

    # Save
    selected.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nSaved selected 100K genomes to: {OUTPUT_FILE}")

    # Also save just the accession list for downloading
    acc_file = os.path.join(OUTPUT_DIR, "selected_100k_accessions.txt")
    # Use ncbi_accession for download; prefer GCF where available
    download_accessions = selected["ncbi_accession"].copy()
    # Where GCF is available, use it (RefSeq preferred over GenBank)
    gcf_mask = selected["gcf_accession"].notna()
    download_accessions[gcf_mask] = selected.loc[gcf_mask, "gcf_accession"]
    download_accessions.to_csv(acc_file, index=False, header=False)
    print(f"Saved accession list for download: {acc_file}")
    print(f"  GCF accessions: {gcf_mask.sum():,}")
    print(f"  GCA accessions: {(~gcf_mask).sum():,}")


if __name__ == "__main__":
    main()
