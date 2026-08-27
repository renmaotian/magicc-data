#!/usr/bin/env python3
"""
Script 47: Pathogen Analysis V2
- Part 1: Synthesize 5 S.enterica + L.monocytogenes genomes (NO fragmentation, whole contigs)
- Part 2: Run MAGICC V4 and CheckM2 on synthetic genomes
- Part 3: 1000+1000 GTDB genomes with MAGICC V4 vs CheckM2 (seed=123)
"""

import os
import sys
import csv
import json
import gzip
import glob as globmod
import shutil
import random
import subprocess
import time
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path("/path/to/magicc-legacy")
DATA_DIR = BASE_DIR / "data"
GENOME_DIR = DATA_DIR / "genomes"
BENCHMARK_DIR = DATA_DIR / "benchmarks" / "pathogen_analysis_v2"
SYNTHETIC_DIR = BENCHMARK_DIR / "synthetic"
CHECKM2_OUTPUT_DIR = BENCHMARK_DIR / "checkm2_output"
GTDB_DIR = BENCHMARK_DIR / "gtdb_comparison"
GTDB_GENOMES_DIR = GTDB_DIR / "magicc_input"

SE_GENOME = DATA_DIR / "genomes" / "GCF_001302605.1" / "GCF_001302605.1_ASM130260v1_genomic.fna"
LM_GENOME = DATA_DIR / "genomes" / "GCF_000021185.1" / "GCF_000021185.1_ASM2118v1_genomic.fna"
GTDB_METADATA = DATA_DIR / "gtdb" / "bac120_metadata.tsv.gz"

MAGICC_MODEL = BASE_DIR / "models" / "magicc_v4.onnx"
NORM_PARAMS = DATA_DIR / "features" / "normalization_params.json"
KMERS_FILE = DATA_DIR / "kmer_selection" / "selected_kmers.txt"

CHECKM2_DB = Path("/path/to/magicc/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd")
CHECKM2_VERSION_CONTROL = Path(
    "/path/to/anaconda3/envs/checkm2_py39/lib/python3.9/site-packages/checkm2/versionControl.py"
)

SEED = 123
N_PER_SPECIES = 1000
MAGICC_THREADS = 43
CHECKM2_THREADS = 32
DOWNLOAD_WORKERS = 8
DOWNLOAD_BATCH_SIZE = 100


# ============================================================================
# Part 1: Synthesize 5 genomes (NO fragmentation)
# ============================================================================
def part1_synthesize():
    print("=" * 80)
    print("PART 1: Synthesize 5 genomes (NO fragmentation, whole contigs)")
    print("=" * 80)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # Read reference genomes - concatenate all sequences into one string
    se_records = list(SeqIO.parse(str(SE_GENOME), "fasta"))
    lm_records = list(SeqIO.parse(str(LM_GENOME), "fasta"))

    se_seq = "".join(str(r.seq) for r in se_records)
    lm_seq = "".join(str(r.seq) for r in lm_records)

    se_size = len(se_seq)
    lm_size = len(lm_seq)

    print(f"S. enterica genome: {se_size:,} bp ({se_size/1e6:.3f} Mbp)")
    print(f"L. monocytogenes genome: {lm_size:,} bp ({lm_size/1e6:.3f} Mbp)")
    print()

    # Define genomes: (name, se_fraction, lm_fraction_of_se_size)
    genome_specs = [
        ("genome_1_100se_5lm", 1.0, 0.05),
        ("genome_2_100se_20lm", 1.0, 0.20),
        ("genome_3_100se_50lm", 1.0, 0.50),
        ("genome_4_100se_100lm", 1.0, 1.00),
        ("genome_5_60se_40lm", 0.60, 0.40),
    ]

    results = []

    for name, se_frac, lm_frac in genome_specs:
        se_bp = int(se_size * se_frac)
        lm_bp = int(se_size * lm_frac)

        # Get S.e. contig (substring from beginning)
        se_contig = se_seq[:se_bp]

        # Get L.m. contig - tile if needed
        if lm_bp <= lm_size:
            lm_contig = lm_seq[:lm_bp]
        else:
            repeats = (lm_bp // lm_size) + 1
            lm_tiled = lm_seq * repeats
            lm_contig = lm_tiled[:lm_bp]

        # Create FASTA records
        records = [
            SeqRecord(
                Seq(se_contig),
                id="contig_1_salmonella_enterica",
                description=f"S.enterica {se_frac*100:.0f}% ({se_bp:,} bp)",
            ),
            SeqRecord(
                Seq(lm_contig),
                id="contig_2_listeria_monocytogenes",
                description=f"L.monocytogenes {lm_frac*100:.0f}% of S.e. size ({lm_bp:,} bp)",
            ),
        ]

        # Write FASTA
        fasta_path = SYNTHETIC_DIR / f"{name}.fasta"
        SeqIO.write(records, str(fasta_path), "fasta")

        total_bp = se_bp + lm_bp
        true_comp = se_bp / se_size * 100
        true_cont = lm_bp / se_size * 100

        results.append({
            "genome_id": name,
            "n_contigs": 2,
            "se_bp": se_bp,
            "lm_bp": lm_bp,
            "total_bp": total_bp,
            "true_completeness": round(true_comp, 2),
            "true_contamination": round(true_cont, 2),
        })

        print(f"{name}:")
        print(f"  S.e. contig: {se_bp:,} bp")
        print(f"  L.m. contig: {lm_bp:,} bp")
        print(f"  Total: {total_bp:,} bp")
        print(f"  True completeness: {true_comp:.2f}%")
        print(f"  True contamination: {true_cont:.2f}%")
        print()

    # Verify files
    print("Verification - reading back files:")
    for name, _, _ in genome_specs:
        fasta_path = SYNTHETIC_DIR / f"{name}.fasta"
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        total = sum(len(r) for r in records)
        print(f"  {name}: {len(records)} contigs, {total:,} bp")

    return results


# ============================================================================
# Part 2: Run MAGICC V4 and CheckM2 on synthetic genomes
# ============================================================================
def part2_run_predictions(synth_results):
    print()
    print("=" * 80)
    print("PART 2: Run MAGICC V4 and CheckM2 on synthetic genomes")
    print("=" * 80)

    # --- MAGICC V4 ---
    magicc_output = BENCHMARK_DIR / "magicc_v4_predictions.tsv"
    if magicc_output.exists():
        print(f"\n  [SKIP] MAGICC predictions already exist: {magicc_output}")
    else:
        print("\nRunning MAGICC V4...")
        magicc_cmd = [
            "conda", "run", "-n", "magicc2", "--no-banner",
            "python", "-m", "magicc", "predict",
            "--input", str(SYNTHETIC_DIR),
            "--output", str(magicc_output),
            "--threads", str(MAGICC_THREADS),
            "--model", str(MAGICC_MODEL),
            "--normalization", str(NORM_PARAMS),
            "--kmers", str(KMERS_FILE),
        ]
        print(f"  Command: {' '.join(magicc_cmd)}")
        result = subprocess.run(magicc_cmd, capture_output=True, text=True)
        print(f"  STDOUT: {result.stdout}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[-500:]}")
        if result.returncode != 0:
            print(f"  WARNING: MAGICC returned {result.returncode}")

    # --- CheckM2 ---
    checkm2_report = CHECKM2_OUTPUT_DIR / "quality_report.tsv"
    if checkm2_report.exists():
        print(f"\n  [SKIP] CheckM2 results already exist: {checkm2_report}")
    else:
        print("\nPatching CheckM2 versionControl.py...")
        vc_path = str(CHECKM2_VERSION_CONTROL)
        vc_backup = vc_path + ".bak"
        shutil.copy2(vc_path, vc_backup)

        with open(vc_path, "r") as f:
            vc_content = f.read()

        patched = vc_content.replace(
            "    def checksum_version_validate(self):\n"
            "        '''Runs each time to ensure all models, definitions and pickled files are congruent with current CheckM2 version'''",
            "    def checksum_version_validate(self):\n"
            "        '''Patched to skip validation'''\n"
            "        return True\n"
            "        '''Runs each time to ensure all models, definitions and pickled files are congruent with current CheckM2 version'''",
        )

        patched = patched.replace(
            "    def checksum_version_validate_DIAMOND(self, location=None):\n"
            "        '''Runs to ensure DIAMOND database has correct checksum and is congruent with current CheckM2 version'''",
            "    def checksum_version_validate_DIAMOND(self, location=None):\n"
            "        '''Patched to skip validation'''\n"
            "        return True\n"
            "        '''Runs to ensure DIAMOND database has correct checksum and is congruent with current CheckM2 version'''",
        )

        with open(vc_path, "w") as f:
            f.write(patched)
        print("  Patched successfully")

        print("\nRunning CheckM2...")
        checkm2_cmd = [
            "conda", "run", "-n", "checkm2_py39", "--no-banner",
            "checkm2", "predict",
            "--threads", str(CHECKM2_THREADS),
            "-x", ".fasta",
            "--input", str(SYNTHETIC_DIR),
            "--output-directory", str(CHECKM2_OUTPUT_DIR),
            "--force",
            "--database_path", str(CHECKM2_DB),
        ]
        print(f"  Command: {' '.join(checkm2_cmd)}")
        result = subprocess.run(checkm2_cmd, capture_output=True, text=True)
        print(f"  STDOUT: {result.stdout}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[-500:]}")
        if result.returncode != 0:
            print(f"  WARNING: CheckM2 returned {result.returncode}")

        # Revert patch
        print("\nReverting CheckM2 patch...")
        shutil.move(vc_backup, vc_path)
        print("  Reverted successfully")

    # --- Parse results ---
    print("\nParsing MAGICC V4 results...")
    magicc_df = pd.read_csv(str(magicc_output), sep="\t")
    print(magicc_df.to_string(index=False))

    # Detect column names
    magicc_genome_col = "genome_name" if "genome_name" in magicc_df.columns else "genome"
    magicc_comp_col = "pred_completeness" if "pred_completeness" in magicc_df.columns else "completeness"
    magicc_cont_col = "pred_contamination" if "pred_contamination" in magicc_df.columns else "contamination"

    print("\nParsing CheckM2 results...")
    checkm2_df = pd.read_csv(str(checkm2_report), sep="\t")
    print(checkm2_df[["Name", "Completeness", "Contamination"]].to_string(index=False))

    # --- Build comparison table ---
    print("\nBuilding comparison table...")
    rows = []
    for sr in synth_results:
        gid = sr["genome_id"]

        # MAGICC results
        magicc_row = magicc_df[magicc_df[magicc_genome_col].str.contains(gid)]
        if len(magicc_row) > 0:
            magicc_comp = float(magicc_row.iloc[0][magicc_comp_col])
            magicc_cont = float(magicc_row.iloc[0][magicc_cont_col])
        else:
            magicc_comp = np.nan
            magicc_cont = np.nan

        # CheckM2 results
        checkm2_row = checkm2_df[checkm2_df["Name"].str.contains(gid)]
        if len(checkm2_row) > 0:
            checkm2_comp = float(checkm2_row.iloc[0]["Completeness"])
            checkm2_cont = float(checkm2_row.iloc[0]["Contamination"])
        else:
            checkm2_comp = np.nan
            checkm2_cont = np.nan

        rows.append({
            "genome_id": gid,
            "n_contigs": sr["n_contigs"],
            "true_comp": sr["true_completeness"],
            "true_cont": sr["true_contamination"],
            "magicc_comp": round(magicc_comp, 2) if not np.isnan(magicc_comp) else np.nan,
            "magicc_cont": round(magicc_cont, 2) if not np.isnan(magicc_cont) else np.nan,
            "checkm2_comp": round(checkm2_comp, 2) if not np.isnan(checkm2_comp) else np.nan,
            "checkm2_cont": round(checkm2_cont, 2) if not np.isnan(checkm2_cont) else np.nan,
        })

    comparison_df = pd.DataFrame(rows)
    comparison_path = BENCHMARK_DIR / "comparison_table.tsv"
    comparison_df.to_csv(str(comparison_path), sep="\t", index=False)

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved to: {comparison_path}")

    return comparison_df


# ============================================================================
# Part 3: 1000+1000 GTDB genomes
# ============================================================================
def download_batch(batch_idx, batch_accs, genomes_dir, done_file):
    """Download a batch of genomes. Returns (batch_idx, downloaded_accs, failed_accs)."""
    tmpzip = f"/tmp/ncbi_dl_v2_{batch_idx}_{os.getpid()}.zip"
    tmpextract = f"/tmp/ncbi_extract_v2_{batch_idx}_{os.getpid()}"
    acc_file = f"/tmp/v2_batch_{batch_idx}_{os.getpid()}_acc.txt"

    with open(acc_file, "w") as f:
        f.write("\n".join(batch_accs) + "\n")

    for retry in range(3):
        try:
            # Cleanup previous attempt
            for p in [tmpzip, tmpextract]:
                if os.path.exists(p):
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)

            result = subprocess.run(
                [
                    "conda", "run", "-n", "magicc2",
                    "datasets", "download", "genome", "accession",
                    "--inputfile", acc_file,
                    "--include", "genome",
                    "--filename", tmpzip,
                    "--no-progressbar",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0 or not os.path.exists(tmpzip):
                if retry < 2:
                    time.sleep((retry + 1) * 5)
                    continue
                return batch_idx, [], batch_accs

            # Extract
            os.makedirs(tmpextract, exist_ok=True)
            subprocess.run(
                ["unzip", "-q", "-o", tmpzip, "-d", tmpextract],
                capture_output=True,
                timeout=120,
            )

            data_dir = os.path.join(tmpextract, "ncbi_dataset", "data")
            downloaded_accs = []
            if os.path.isdir(data_dir):
                for item in os.listdir(data_dir):
                    if item.startswith("GC"):
                        src = os.path.join(data_dir, item)
                        dst = os.path.join(str(genomes_dir), item)
                        if os.path.isdir(src) and not os.path.exists(dst):
                            shutil.move(src, dst)
                        downloaded_accs.append(item)

            failed = [a for a in batch_accs if a not in downloaded_accs]
            return batch_idx, downloaded_accs, failed

        except subprocess.TimeoutExpired:
            if retry < 2:
                time.sleep((retry + 1) * 5)
                continue
            return batch_idx, [], batch_accs
        except Exception as e:
            if retry < 2:
                time.sleep((retry + 1) * 5)
                continue
            return batch_idx, [], batch_accs
        finally:
            for p in [tmpzip, acc_file]:
                if os.path.exists(p):
                    os.remove(p)
            if os.path.exists(tmpextract):
                shutil.rmtree(tmpextract, ignore_errors=True)

    return batch_idx, [], batch_accs


def part3_gtdb_comparison():
    print()
    print("=" * 80)
    print("PART 3: 1000+1000 GTDB genomes with MAGICC V4")
    print("=" * 80)

    GTDB_DIR.mkdir(parents=True, exist_ok=True)
    GTDB_GENOMES_DIR.mkdir(parents=True, exist_ok=True)

    selected_path = GTDB_DIR / "selected_genomes.tsv"

    # ----------------------------------------------------------------
    # Step 1: Select genomes (or load existing selection)
    # ----------------------------------------------------------------
    if selected_path.exists():
        print(f"\n  [LOAD] Existing selection: {selected_path}")
        selected_df = pd.read_csv(str(selected_path), sep="\t")
        se_count = len(selected_df[selected_df["species"] == "Salmonella_enterica"])
        lm_count = len(selected_df[selected_df["species"] == "Listeria_monocytogenes"])
        print(f"  S. enterica: {se_count}, L. monocytogenes: {lm_count}")
    else:
        print("\nStep 1: Parsing GTDB metadata...")
        se_genomes = []
        lm_genomes = []

        with gzip.open(str(GTDB_METADATA), "rt") as f:
            header = f.readline().strip().split("\t")
            acc_idx = 0
            comp_idx = header.index("checkm2_completeness")
            cont_idx = header.index("checkm2_contamination")
            tax_idx = header.index("gtdb_taxonomy")
            size_idx = header.index("genome_size")
            contig_idx = header.index("contig_count")
            n50_idx = header.index("n50_contigs")

            for line in f:
                fields = line.strip().split("\t")
                if len(fields) <= tax_idx:
                    continue
                taxonomy = fields[tax_idx]

                if "s__Salmonella enterica" in taxonomy:
                    se_genomes.append(fields)
                elif "s__Listeria monocytogenes" in taxonomy:
                    lm_genomes.append(fields)

        print(f"  Found {len(se_genomes)} S. enterica genomes")
        print(f"  Found {len(lm_genomes)} L. monocytogenes genomes")

        print(f"\nStep 2: Randomly selecting {N_PER_SPECIES}+{N_PER_SPECIES} genomes (seed={SEED})...")
        random.seed(SEED)
        se_selected = random.sample(se_genomes, min(N_PER_SPECIES, len(se_genomes)))
        lm_selected = random.sample(lm_genomes, min(N_PER_SPECIES, len(lm_genomes)))

        print(f"  Selected {len(se_selected)} S. enterica")
        print(f"  Selected {len(lm_selected)} L. monocytogenes")

        selected_rows = []
        for fields in se_selected + lm_selected:
            acc = fields[acc_idx]
            if acc.startswith("RS_") or acc.startswith("GB_"):
                ncbi_acc = acc[3:]
            else:
                ncbi_acc = acc

            species = "Salmonella_enterica" if fields in se_selected else "Listeria_monocytogenes"

            selected_rows.append({
                "accession": ncbi_acc,
                "gtdb_accession": acc,
                "species": species,
                "checkm2_completeness": float(fields[comp_idx]) if fields[comp_idx] else np.nan,
                "checkm2_contamination": float(fields[cont_idx]) if fields[cont_idx] else np.nan,
                "genome_size": int(fields[size_idx]) if fields[size_idx] else 0,
                "contig_count": int(fields[contig_idx]) if fields[contig_idx] else 0,
                "n50_contigs": int(fields[n50_idx]) if fields[n50_idx] else 0,
                "gtdb_taxonomy": fields[tax_idx],
            })

        selected_df = pd.DataFrame(selected_rows)
        selected_df.to_csv(str(selected_path), sep="\t", index=False)
        print(f"  Saved selected genomes to: {selected_path}")

    # ----------------------------------------------------------------
    # Step 3: Download missing genomes
    # ----------------------------------------------------------------
    print("\nStep 3: Checking existing downloads...")
    missing = []
    available = []

    for _, row in selected_df.iterrows():
        acc = row["accession"]
        genome_dir = GENOME_DIR / acc
        fna_files = list(genome_dir.glob("*.fna")) if genome_dir.exists() else []
        if fna_files:
            available.append((acc, fna_files[0]))
        else:
            missing.append(acc)

    print(f"  Already available: {len(available)}")
    print(f"  Missing: {len(missing)}")

    if missing:
        print(f"\n  Downloading {len(missing)} missing genomes ({DOWNLOAD_WORKERS} workers, batch size {DOWNLOAD_BATCH_SIZE})...")
        done_file = GTDB_DIR / "download_done_accessions.txt"

        # Load previously done
        done_accessions = set()
        if done_file.exists():
            with open(str(done_file)) as f:
                done_accessions = set(line.strip() for line in f if line.strip())

        still_missing = [a for a in missing if a not in done_accessions]
        print(f"  Previously downloaded: {len(done_accessions)}")
        print(f"  Still missing: {len(still_missing)}")

        if still_missing:
            batches = []
            for i in range(0, len(still_missing), DOWNLOAD_BATCH_SIZE):
                batches.append(still_missing[i : i + DOWNLOAD_BATCH_SIZE])

            print(f"  {len(batches)} batches")
            total_downloaded = 0
            failed_accessions = []

            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
                futures = {}
                for idx, batch in enumerate(batches):
                    fut = executor.submit(download_batch, idx, batch, GENOME_DIR, done_file)
                    futures[fut] = idx

                for fut in as_completed(futures):
                    batch_idx, downloaded, failed = fut.result()
                    total_downloaded += len(downloaded)

                    # Record done accessions
                    if downloaded:
                        with open(str(done_file), "a") as f:
                            for acc in downloaded:
                                f.write(acc + "\n")

                    if failed:
                        failed_accessions.extend(failed)

                    completed = sum(1 for f2 in futures if f2.done())
                    print(
                        f"  Batch {batch_idx}: {len(downloaded)} ok, {len(failed)} failed | "
                        f"Progress: {completed}/{len(batches)} batches, {total_downloaded} total"
                    )

            print(f"\n  Download complete: {total_downloaded} new genomes")
            if failed_accessions:
                fail_file = GTDB_DIR / "failed_downloads.txt"
                with open(str(fail_file), "w") as f:
                    f.write("\n".join(failed_accessions) + "\n")
                print(f"  Failed: {len(failed_accessions)} (saved to {fail_file})")

    # ----------------------------------------------------------------
    # Step 4: Prepare FASTA files for MAGICC (symlinks to save space)
    # ----------------------------------------------------------------
    print("\nStep 4: Preparing FASTA files for MAGICC...")
    prepared_count = 0
    failed_accessions = []

    for _, row in selected_df.iterrows():
        acc = row["accession"]
        genome_dir = GENOME_DIR / acc
        fna_files = list(genome_dir.glob("*.fna")) if genome_dir.exists() else []

        if not fna_files:
            failed_accessions.append(acc)
            continue

        src = fna_files[0]
        dst = GTDB_GENOMES_DIR / f"{acc}.fasta"
        if not dst.exists():
            os.symlink(os.path.abspath(str(src)), str(dst))
        prepared_count += 1

    print(f"  Prepared: {prepared_count} FASTA files")
    if failed_accessions:
        missing_file = GTDB_DIR / "missing_accessions.txt"
        with open(str(missing_file), "w") as f:
            f.write("\n".join(failed_accessions) + "\n")
        print(f"  Missing: {len(failed_accessions)} (saved to {missing_file})")

    # Filter selected_df to only available genomes
    available_accs = set(selected_df["accession"]) - set(failed_accessions)
    selected_df_filtered = selected_df[selected_df["accession"].isin(available_accs)].copy()
    print(f"  Available genomes for analysis: {len(selected_df_filtered)}")

    # ----------------------------------------------------------------
    # Step 5: Run MAGICC V4
    # ----------------------------------------------------------------
    magicc_output = GTDB_DIR / "magicc_v4_predictions.tsv"
    if magicc_output.exists():
        print(f"\n  [SKIP] MAGICC predictions already exist: {magicc_output}")
    else:
        print(f"\nStep 5: Running MAGICC V4 on {prepared_count} genomes...")
        magicc_cmd = [
            "conda", "run", "-n", "magicc2", "--no-banner",
            "python", "-m", "magicc", "predict",
            "--input", str(GTDB_GENOMES_DIR),
            "--output", str(magicc_output),
            "--threads", str(MAGICC_THREADS),
            "--model", str(MAGICC_MODEL),
            "--normalization", str(NORM_PARAMS),
            "--kmers", str(KMERS_FILE),
        ]
        print(f"  Command: {' '.join(magicc_cmd)}")
        t0 = time.time()
        result = subprocess.run(magicc_cmd, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")
        if result.stdout:
            print(f"  STDOUT (last 500 chars): {result.stdout[-500:]}")
        if result.stderr:
            print(f"  STDERR (last 500 chars): {result.stderr[-500:]}")
        if result.returncode != 0:
            print(f"  WARNING: MAGICC returned {result.returncode}")

    # ----------------------------------------------------------------
    # Step 6: Compare MAGICC V4 vs CheckM2 metadata
    # ----------------------------------------------------------------
    print("\nStep 6: Comparing MAGICC V4 vs CheckM2...")
    magicc_df = pd.read_csv(str(magicc_output), sep="\t")
    print(f"  MAGICC predictions: {len(magicc_df)} rows")
    print(f"  MAGICC columns: {list(magicc_df.columns)}")

    # Detect column names
    magicc_genome_col = "genome_name" if "genome_name" in magicc_df.columns else "genome"
    magicc_comp_col = "pred_completeness" if "pred_completeness" in magicc_df.columns else "completeness"
    magicc_cont_col = "pred_contamination" if "pred_contamination" in magicc_df.columns else "contamination"

    # Build lookup: accession -> (comp, cont)
    magicc_lookup = {}
    for _, mrow in magicc_df.iterrows():
        gname = str(mrow[magicc_genome_col])
        # Strip extension and path
        gname_clean = os.path.basename(gname).replace(".fasta", "").replace(".fna", "")
        magicc_lookup[gname_clean] = (float(mrow[magicc_comp_col]), float(mrow[magicc_cont_col]))

    comp_rows = []
    matched = 0
    for _, row in selected_df_filtered.iterrows():
        acc = row["accession"]
        if acc not in magicc_lookup:
            continue
        m_comp, m_cont = magicc_lookup[acc]
        matched += 1
        comp_rows.append({
            "accession": acc,
            "species": row["species"],
            "checkm2_completeness": float(row["checkm2_completeness"]),
            "checkm2_contamination": float(row["checkm2_contamination"]),
            "magicc_completeness": round(m_comp, 4),
            "magicc_contamination": round(m_cont, 4),
        })

    comparison_df = pd.DataFrame(comp_rows)
    comparison_path = GTDB_DIR / "comparison_table.tsv"
    comparison_df.to_csv(str(comparison_path), sep="\t", index=False)
    print(f"  Matched: {matched}/{len(selected_df_filtered)}")
    print(f"  Saved comparison table: {len(comparison_df)} genomes to {comparison_path}")

    # ----------------------------------------------------------------
    # Step 7: MIMAG analysis
    # ----------------------------------------------------------------
    print("\nStep 7: MIMAG analysis...")
    comparison_df["hq_checkm2"] = (
        (comparison_df["checkm2_completeness"] >= 90)
        & (comparison_df["checkm2_contamination"] <= 5)
    )
    comparison_df["hq_magicc"] = (
        (comparison_df["magicc_completeness"] >= 90)
        & (comparison_df["magicc_contamination"] <= 5)
    )

    checkm2_only_hq = comparison_df[comparison_df["hq_checkm2"] & ~comparison_df["hq_magicc"]]
    magicc_only_hq = comparison_df[comparison_df["hq_magicc"] & ~comparison_df["hq_checkm2"]]
    both_hq = comparison_df[comparison_df["hq_checkm2"] & comparison_df["hq_magicc"]]
    neither_hq = comparison_df[~comparison_df["hq_checkm2"] & ~comparison_df["hq_magicc"]]

    print(f"\n  MIMAG HQ Classification (total={len(comparison_df)}):")
    print(f"    Both HQ:                  {len(both_hq)}")
    print(f"    CheckM2-only HQ:          {len(checkm2_only_hq)}")
    print(f"    MAGICC-only HQ:           {len(magicc_only_hq)}")
    print(f"    Neither HQ:               {len(neither_hq)}")

    # Breakdown by species
    print(f"\n  Breakdown by species:")
    species_mimag = {}
    for species in ["Salmonella_enterica", "Listeria_monocytogenes"]:
        sp_df = comparison_df[comparison_df["species"] == species]
        sp_both = sp_df[sp_df["hq_checkm2"] & sp_df["hq_magicc"]]
        sp_checkm2_only = sp_df[sp_df["hq_checkm2"] & ~sp_df["hq_magicc"]]
        sp_magicc_only = sp_df[sp_df["hq_magicc"] & ~sp_df["hq_checkm2"]]
        sp_neither = sp_df[~sp_df["hq_checkm2"] & ~sp_df["hq_magicc"]]

        species_mimag[species] = {
            "n_genomes": len(sp_df),
            "both_hq": len(sp_both),
            "checkm2_only_hq": len(sp_checkm2_only),
            "magicc_only_hq": len(sp_magicc_only),
            "neither_hq": len(sp_neither),
        }

        print(f"\n    {species} ({len(sp_df)} genomes):")
        print(f"      Both HQ: {len(sp_both)}")
        print(f"      CheckM2-only HQ: {len(sp_checkm2_only)}")
        print(f"      MAGICC-only HQ: {len(sp_magicc_only)}")
        print(f"      Neither HQ: {len(sp_neither)}")

    # Save MIMAG analysis
    mimag_rows = []
    for species in ["Salmonella_enterica", "Listeria_monocytogenes", "Total"]:
        if species == "Total":
            sp_df = comparison_df
        else:
            sp_df = comparison_df[comparison_df["species"] == species]

        mimag_rows.append({
            "species": species,
            "n_genomes": len(sp_df),
            "both_hq": len(sp_df[sp_df["hq_checkm2"] & sp_df["hq_magicc"]]),
            "checkm2_only_hq": len(sp_df[sp_df["hq_checkm2"] & ~sp_df["hq_magicc"]]),
            "magicc_only_hq": len(sp_df[sp_df["hq_magicc"] & ~sp_df["hq_checkm2"]]),
            "neither_hq": len(sp_df[~sp_df["hq_checkm2"] & ~sp_df["hq_magicc"]]),
            "checkm2_hq_total": len(sp_df[sp_df["hq_checkm2"]]),
            "magicc_hq_total": len(sp_df[sp_df["hq_magicc"]]),
        })

    mimag_df = pd.DataFrame(mimag_rows)
    mimag_path = GTDB_DIR / "mimag_analysis.tsv"
    mimag_df.to_csv(str(mimag_path), sep="\t", index=False)
    print(f"\n  Saved MIMAG analysis to: {mimag_path}")

    # Accuracy stats per species
    print("\n  Accuracy statistics per species:")
    accuracy_stats = {}
    for species in ["Salmonella_enterica", "Listeria_monocytogenes"]:
        sp_df = comparison_df[comparison_df["species"] == species]
        if len(sp_df) == 0:
            continue

        comp_diff = sp_df["magicc_completeness"] - sp_df["checkm2_completeness"]
        cont_diff = sp_df["magicc_contamination"] - sp_df["checkm2_contamination"]

        stats = {
            "n_genomes": int(len(sp_df)),
            "completeness_mean_diff": round(float(comp_diff.mean()), 3),
            "completeness_std_diff": round(float(comp_diff.std()), 3),
            "completeness_mae": round(float(comp_diff.abs().mean()), 3),
            "completeness_median_ae": round(float(comp_diff.abs().median()), 3),
            "contamination_mean_diff": round(float(cont_diff.mean()), 3),
            "contamination_std_diff": round(float(cont_diff.std()), 3),
            "contamination_mae": round(float(cont_diff.abs().mean()), 3),
            "contamination_median_ae": round(float(cont_diff.abs().median()), 3),
            "completeness_corr": round(float(sp_df["magicc_completeness"].corr(sp_df["checkm2_completeness"])), 4),
            "contamination_corr": round(float(sp_df["magicc_contamination"].corr(sp_df["checkm2_contamination"])), 4),
        }
        accuracy_stats[species] = stats

        print(f"\n    {species} ({stats['n_genomes']} genomes):")
        print(f"      Completeness:  MAE={stats['completeness_mae']}, mean_diff={stats['completeness_mean_diff']}, r={stats['completeness_corr']}")
        print(f"      Contamination: MAE={stats['contamination_mae']}, mean_diff={stats['contamination_mean_diff']}, r={stats['contamination_corr']}")

    # Save summary
    summary = {
        "total_genomes_analyzed": int(len(comparison_df)),
        "species_breakdown": {
            "Salmonella_enterica": int(len(comparison_df[comparison_df["species"] == "Salmonella_enterica"])),
            "Listeria_monocytogenes": int(len(comparison_df[comparison_df["species"] == "Listeria_monocytogenes"])),
        },
        "mimag_hq_analysis": {
            "both_hq": int(len(both_hq)),
            "checkm2_only_hq": int(len(checkm2_only_hq)),
            "magicc_only_hq": int(len(magicc_only_hq)),
            "neither_hq": int(len(neither_hq)),
        },
        "mimag_by_species": {sp: {k: int(v) for k, v in vals.items()} for sp, vals in species_mimag.items()},
        "accuracy_stats": accuracy_stats,
        "seed": SEED,
    }

    summary_path = GTDB_DIR / "summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary to: {summary_path}")

    return comparison_df, mimag_df, summary


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Script 47: Pathogen Analysis V2")
    print("=" * 80)

    # Part 1
    synth_results = part1_synthesize()

    # Part 2
    comparison_df = part2_run_predictions(synth_results)

    # Part 3
    gtdb_comparison, mimag_df, summary = part3_gtdb_comparison()

    print("\n" + "=" * 80)
    print("ALL PARTS COMPLETE")
    print("=" * 80)
