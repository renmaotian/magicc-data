#!/usr/bin/env python3
"""
Verify downloaded genomes and create a final genome manifest.
- Check which accessions have downloaded FASTA files
- Report missing genomes
- Verify FASTA files are non-empty
- Update split files to exclude missing genomes
"""

import os
import glob
import pandas as pd

GENOME_DIR = "/home/tianrm/projects/magicc2/data/genomes"
SELECTED_FILE = "/home/tianrm/projects/magicc2/data/gtdb/selected_100k_genomes.tsv"
SPLITS_DIR = "/home/tianrm/projects/magicc2/data/splits"

def main():
    # Load selected genomes
    df = pd.read_csv(SELECTED_FILE, sep="\t")
    print(f"Selected genomes: {len(df):,}")

    # Get list of downloaded genome directories
    downloaded = set()
    for d in os.listdir(GENOME_DIR):
        if d.startswith("GC") and os.path.isdir(os.path.join(GENOME_DIR, d)):
            downloaded.add(d)
    print(f"Downloaded genome directories: {len(downloaded):,}")

    # Check which accessions have downloads
    # The ncbi_accession in our file could be GCA_* or GCF_*
    # Downloaded dirs are based on what NCBI returns (usually GCA_*)
    df["has_download"] = False
    df["download_dir"] = ""
    df["fasta_path"] = ""

    for idx, row in df.iterrows():
        ncbi_acc = str(row["ncbi_accession"])
        gcf_acc = str(row.get("gcf_accession", ""))

        # Check ncbi_accession
        if ncbi_acc in downloaded:
            df.at[idx, "has_download"] = True
            df.at[idx, "download_dir"] = ncbi_acc
        # Check GCF accession
        elif gcf_acc in downloaded and gcf_acc != "nan":
            df.at[idx, "has_download"] = True
            df.at[idx, "download_dir"] = gcf_acc
        else:
            # Also check GCA version of GCF
            if ncbi_acc.startswith("GCF_"):
                gca_version = "GCA_" + ncbi_acc[4:]
                if gca_version in downloaded:
                    df.at[idx, "has_download"] = True
                    df.at[idx, "download_dir"] = gca_version
            # Check GCF version of GCA
            elif ncbi_acc.startswith("GCA_"):
                gcf_version = "GCF_" + ncbi_acc[4:]
                if gcf_version in downloaded:
                    df.at[idx, "has_download"] = True
                    df.at[idx, "download_dir"] = gcf_version

    # Find FASTA files
    for idx, row in df[df["has_download"]].iterrows():
        dir_path = os.path.join(GENOME_DIR, row["download_dir"])
        fasta_files = glob.glob(os.path.join(dir_path, "*.fna")) + \
                     glob.glob(os.path.join(dir_path, "*.fasta")) + \
                     glob.glob(os.path.join(dir_path, "*.fa"))
        if fasta_files:
            # Use the largest FASTA file (genomic, not protein)
            fasta_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            df.at[idx, "fasta_path"] = fasta_files[0]

    # Report
    n_downloaded = df["has_download"].sum()
    n_missing = (~df["has_download"]).sum()
    n_with_fasta = (df["fasta_path"] != "").sum()

    print(f"\nAccessions with downloads: {n_downloaded:,}")
    print(f"Accessions without downloads: {n_missing:,}")
    print(f"Accessions with FASTA files: {n_with_fasta:,}")

    # Report missing
    if n_missing > 0:
        missing = df[~df["has_download"]]
        print(f"\nMissing accessions ({n_missing}):")
        for _, row in missing.iterrows():
            print(f"  {row['ncbi_accession']} ({row['phylum']})")

        # Save missing list
        missing_file = os.path.join(SPLITS_DIR, "missing_accessions.txt")
        missing["ncbi_accession"].to_csv(missing_file, index=False, header=False)
        print(f"  Saved to: {missing_file}")

    # Filter to only successfully downloaded genomes
    available = df[df["fasta_path"] != ""].copy()
    print(f"\nFinal available genomes: {len(available):,}")

    # Save manifest
    manifest_file = os.path.join(SPLITS_DIR, "genome_manifest.tsv")
    available.to_csv(manifest_file, sep="\t", index=False)
    print(f"Saved manifest: {manifest_file}")

    # Update splits to exclude missing genomes
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(SPLITS_DIR, f"{split_name}_genomes.tsv")
        split_df = pd.read_csv(split_file, sep="\t")

        # Merge with available genomes to get fasta paths
        available_accs = set(available["gtdb_accession"])
        split_available = split_df[split_df["gtdb_accession"].isin(available_accs)].copy()

        # Add fasta_path and download_dir
        path_lookup = available.set_index("gtdb_accession")[["fasta_path", "download_dir"]]
        split_available = split_available.merge(
            path_lookup, left_on="gtdb_accession", right_index=True, how="left"
        )

        n_original = len(split_df)
        n_available = len(split_available)
        n_dropped = n_original - n_available

        # Save updated split
        updated_file = os.path.join(SPLITS_DIR, f"{split_name}_genomes.tsv")
        split_available.to_csv(updated_file, sep="\t", index=False)

        print(f"  {split_name}: {n_original} -> {n_available} ({n_dropped} dropped)")

    # Final statistics
    print(f"\n{'='*60}")
    print("FINAL DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Genomes selected: 100,000")
    print(f"  Genomes downloaded: {len(available):,}")
    print(f"  Missing/unavailable: {100000 - len(available):,}")
    print(f"  Success rate: {len(available)/100000*100:.2f}%")

    # Split sizes
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(SPLITS_DIR, f"{split_name}_genomes.tsv")
        split_df = pd.read_csv(split_file, sep="\t")
        print(f"  {split_name}: {len(split_df):,} genomes")


if __name__ == "__main__":
    main()
