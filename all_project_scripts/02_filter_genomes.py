#!/usr/bin/env python3
"""
Step 2: Filter GTDB metadata for high-quality reference genomes.

Criteria (all must be met):
  - CheckM2 completeness >= 98%
  - CheckM2 contamination <= 2%
  - Contig count < 100
  - N50 > 20,000 bp
  - Longest contig > 100,000 bp

Extracts GCA/GCF accessions and taxonomy info.
Reports statistics on passing genomes.
"""

import pandas as pd
import numpy as np
import os
import sys

# Paths
DATA_DIR = "/home/tianrm/projects/magicc2/data/gtdb"
BAC_META = os.path.join(DATA_DIR, "bac120_metadata.tsv.gz")
AR_META = os.path.join(DATA_DIR, "ar53_metadata.tsv.gz")
OUTPUT_DIR = "/home/tianrm/projects/magicc2/data/gtdb"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "filtered_genomes.tsv")

# Columns to load (to save memory)
USECOLS = [
    "accession",
    "checkm2_completeness",
    "checkm2_contamination",
    "contig_count",
    "n50_contigs",
    "longest_contig",
    "gtdb_taxonomy",
    "ncbi_genbank_assembly_accession",
    "genome_size",
]

def parse_phylum(taxonomy_str):
    """Extract phylum from GTDB taxonomy string."""
    if pd.isna(taxonomy_str):
        return "Unknown"
    parts = taxonomy_str.split(";")
    for part in parts:
        if part.startswith("p__"):
            phylum = part[3:].strip()
            return phylum if phylum else "Unknown"
    return "Unknown"

def parse_domain(taxonomy_str):
    """Extract domain from GTDB taxonomy string."""
    if pd.isna(taxonomy_str):
        return "Unknown"
    parts = taxonomy_str.split(";")
    for part in parts:
        if part.startswith("d__"):
            return part[3:].strip()
    return "Unknown"

def load_metadata(filepath, label):
    """Load GTDB metadata TSV file."""
    print(f"Loading {label} metadata from {filepath}...")
    df = pd.read_csv(filepath, sep="\t", usecols=USECOLS, low_memory=False)
    print(f"  Loaded {len(df):,} genomes")
    return df

def filter_genomes(df, label):
    """Apply quality filters and report statistics."""
    total = len(df)
    print(f"\n{'='*60}")
    print(f"Filtering {label} genomes (n={total:,})")
    print(f"{'='*60}")

    # Convert to numeric, coerce errors to NaN
    for col in ["checkm2_completeness", "checkm2_contamination", "contig_count", "n50_contigs", "longest_contig"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Report NaN counts
    for col in ["checkm2_completeness", "checkm2_contamination", "contig_count", "n50_contigs", "longest_contig"]:
        na_count = df[col].isna().sum()
        if na_count > 0:
            print(f"  WARNING: {col} has {na_count:,} NaN values")

    # Apply filters sequentially and report
    mask_comp = df["checkm2_completeness"] >= 98
    print(f"  CheckM2 completeness >= 98%: {mask_comp.sum():,} pass ({mask_comp.sum()/total*100:.1f}%)")

    mask_cont = df["checkm2_contamination"] <= 2
    print(f"  CheckM2 contamination <= 2%: {mask_cont.sum():,} pass ({mask_cont.sum()/total*100:.1f}%)")

    mask_contigs = df["contig_count"] < 100
    print(f"  Contig count < 100:          {mask_contigs.sum():,} pass ({mask_contigs.sum()/total*100:.1f}%)")

    mask_n50 = df["n50_contigs"] > 20000
    print(f"  N50 > 20 kbp:               {mask_n50.sum():,} pass ({mask_n50.sum()/total*100:.1f}%)")

    mask_maxcontig = df["longest_contig"] > 100000
    print(f"  Max contig > 100 kbp:        {mask_maxcontig.sum():,} pass ({mask_maxcontig.sum()/total*100:.1f}%)")

    # Combined filter
    mask_all = mask_comp & mask_cont & mask_contigs & mask_n50 & mask_maxcontig
    print(f"\n  ALL filters combined:        {mask_all.sum():,} pass ({mask_all.sum()/total*100:.1f}%)")

    return df[mask_all].copy()

def main():
    # Load both metadata files
    bac_df = load_metadata(BAC_META, "Bacterial")
    ar_df = load_metadata(AR_META, "Archaeal")

    # Filter
    bac_filtered = filter_genomes(bac_df, "Bacterial")
    ar_filtered = filter_genomes(ar_df, "Archaeal")

    # Combine
    combined = pd.concat([bac_filtered, ar_filtered], ignore_index=True)
    print(f"\n{'='*60}")
    print(f"COMBINED: {len(combined):,} genomes pass all filters")
    print(f"  Bacterial: {len(bac_filtered):,}")
    print(f"  Archaeal:  {len(ar_filtered):,}")
    print(f"{'='*60}")

    # Extract GCA/GCF accessions
    # The accession column has format like "RS_GCF_034719275.1" or "GB_GCA_034719275.1"
    # The ncbi_genbank_assembly_accession has format "GCA_034719275.1"
    # For NCBI datasets, we need GCF or GCA accessions
    # Prefer ncbi_genbank_assembly_accession, but also extract from accession column
    combined["ncbi_accession"] = combined["ncbi_genbank_assembly_accession"].copy()

    # For rows where ncbi_genbank_assembly_accession is missing, extract from accession
    missing_mask = combined["ncbi_accession"].isna() | (combined["ncbi_accession"] == "none") | (combined["ncbi_accession"] == "")
    if missing_mask.sum() > 0:
        # Extract GCA/GCF from the accession column (strip RS_/GB_ prefix)
        combined.loc[missing_mask, "ncbi_accession"] = combined.loc[missing_mask, "accession"].str.replace(r"^(RS_|GB_)", "", regex=True)
        print(f"  Filled {missing_mask.sum():,} missing NCBI accessions from GTDB accession column")

    # Also try to get GCF accessions where available (from the accession column)
    combined["gtdb_accession"] = combined["accession"]
    combined["gcf_accession"] = combined["accession"].str.replace(r"^RS_", "", regex=True)
    combined["gcf_accession"] = combined["gcf_accession"].where(combined["accession"].str.startswith("RS_"), other=np.nan)

    # Parse taxonomy
    combined["domain"] = combined["gtdb_taxonomy"].apply(parse_domain)
    combined["phylum"] = combined["gtdb_taxonomy"].apply(parse_phylum)

    # Report phylum distribution
    print(f"\n{'='*60}")
    print("Phylum distribution of filtered genomes:")
    print(f"{'='*60}")
    phylum_counts = combined["phylum"].value_counts()
    for phylum, count in phylum_counts.items():
        print(f"  {phylum}: {count:,}")
    print(f"\n  Total phyla: {len(phylum_counts)}")

    # Report domain distribution
    print(f"\n  Domain distribution:")
    domain_counts = combined["domain"].value_counts()
    for domain, count in domain_counts.items():
        print(f"    {domain}: {count:,}")

    # Check for special lineages mentioned in protocol
    print(f"\n  Special lineages:")
    patescibacteria = combined[combined["phylum"] == "Patescibacteria"]
    print(f"    Patescibacteria: {len(patescibacteria):,}")

    # DPANN archaea phyla
    dpann_phyla = ["Nanoarchaeota", "Aenigmarchaeota", "Diapherotrites",
                   "Micrarchaeota", "Woesearchaeota", "Altiarchaeota",
                   "Huberarchaeota", "Undinarchaeota", "Iainarchaeota",
                   "Nanohaloarchaeota", "Asgardarchaeota"]
    dpann = combined[combined["phylum"].isin(dpann_phyla)]
    print(f"    DPANN archaea: {len(dpann):,} (phyla: {dpann['phylum'].unique().tolist()})")

    # Other candidate phyla (typically have 'Candidatus' in GTDB or are small)
    # Look for uncommon phyla
    small_phyla = phylum_counts[phylum_counts < 100]
    print(f"    Phyla with <100 genomes: {len(small_phyla)}")
    for p, c in small_phyla.items():
        print(f"      {p}: {c}")

    # Save output
    output_cols = [
        "gtdb_accession", "ncbi_accession", "gcf_accession",
        "checkm2_completeness", "checkm2_contamination",
        "contig_count", "n50_contigs", "longest_contig", "genome_size",
        "domain", "phylum", "gtdb_taxonomy"
    ]
    combined[output_cols].to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nSaved filtered genome list to: {OUTPUT_FILE}")
    print(f"Total genomes: {len(combined):,}")

    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"  Total passing genomes: {len(combined):,}")
    print(f"  Genome size: median={combined['genome_size'].median()/1e6:.1f} Mbp, "
          f"min={combined['genome_size'].min()/1e6:.1f} Mbp, "
          f"max={combined['genome_size'].max()/1e6:.1f} Mbp")
    print(f"  N50: median={combined['n50_contigs'].median()/1e3:.0f} kbp")
    print(f"  Contig count: median={combined['contig_count'].median():.0f}")
    print(f"  Unique phyla: {combined['phylum'].nunique()}")

if __name__ == "__main__":
    main()
