#!/usr/bin/env python3
"""
Step 46: Alignment-based ground-truth validation of MAGICC vs CheckM2
for pathogen genomes (E. coli, S. enterica, L. monocytogenes).

Methodology:
1. Select 10 reference genomes per species where BOTH tools agree on near-perfect quality
   (CheckM2 comp>=99.5%, cont<=0.5%; MAGICC comp>=99%, cont<=1%).
   Prefer NCBI "Complete Genome" assembly level.
2. Subsample 10 query genomes per species x interval (4 intervals x 3 species = ~120 genomes).
3. Download all genomes from NCBI using `datasets` CLI.
4. For each species: concatenate 10 references into one reference FASTA, build minimap2 index.
   Align each query genome to the species reference using `minimap2 -c -x asm5`.
5. Parse PAF output: for each contig, merge all high-identity (>=95%) alignment intervals
   and compute total coverage. A contig is "local" if merged coverage >= 50% of contig length.
6. Ground-truth completeness = sum(local contig bases) / median_reference_genome_size * 100
   Ground-truth contamination = sum(non-local contig bases) / median_reference_genome_size * 100
7. Compare MAGICC and CheckM2 predictions to ground truth via MAE.

Usage:
    python scripts/46_alignment_validation.py
"""

import os
import sys
import subprocess
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time
import glob as glob_module

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path("/mnt/5c77b453-f7e1-48c8-afa3-5641857a41c7/tianrm/projects/magicc2")
DATA_DIR = PROJECT_DIR / "data" / "pathogen_analysis"
GTDB_FILE = PROJECT_DIR / "data" / "gtdb" / "bac120_metadata.tsv.gz"
COMPARISON_FILE = DATA_DIR / "comparison_results.tsv"

THREADS = 43  # 90% of 48 CPUs
SEED = 42
N_REFS_PER_SPECIES = 10
N_QUERIES_PER_GROUP = 10

SPECIES_LIST = [
    's__Escherichia coli',
    's__Salmonella enterica',
    's__Listeria monocytogenes',
]
SPECIES_SHORTNAMES = {
    's__Escherichia coli': 'ecoli',
    's__Salmonella enterica': 'senterica',
    's__Listeria monocytogenes': 'lmonocytogenes',
}

# Alignment parameters
IDENTITY_THRESHOLD = 0.95
COVERAGE_THRESHOLD = 0.50
MINIMAP2_PRESET = 'asm5'

# Output files
REF_ACCESSIONS_FILE = DATA_DIR / "reference_accessions.tsv"
QUERY_GENOMES_FILE = DATA_DIR / "validation_query_genomes.tsv"
VALIDATION_RESULTS_FILE = DATA_DIR / "validation_results.tsv"
VALIDATION_SUMMARY_FILE = DATA_DIR / "validation_summary.json"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / "alignment_validation_v2.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ============================================================================
# Helper functions
# ============================================================================

def merge_intervals(intervals):
    """Merge overlapping intervals and return total covered length."""
    if not intervals:
        return 0
    sorted_ivs = sorted(intervals)
    merged = [list(sorted_ivs[0])]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def read_contig_sizes(fasta_path):
    """Read contig names and sizes from a FASTA file."""
    contigs = {}
    current = None
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                current = line[1:].strip().split()[0]
                contigs[current] = 0
            else:
                contigs[current] += len(line.strip())
    return contigs


def compute_genome_size(fasta_path):
    """Compute total bases in a FASTA file."""
    total = 0
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith('>'):
                total += len(line.strip())
    return total


# ============================================================================
# Step 1: Select reference genomes
# ============================================================================

def select_references(comp_df, gtdb_df):
    """Select 10 reference genomes per species meeting quality criteria."""
    log.info("=" * 80)
    log.info("STEP 1: Select reference genomes")
    log.info("=" * 80)

    # Quality criteria
    mask = (
        (comp_df['checkm2_completeness'] >= 99.5) &
        (comp_df['checkm2_contamination'] <= 0.5) &
        (comp_df['pred_completeness'] >= 99) &
        (comp_df['pred_contamination'] <= 1)
    )
    candidates = comp_df[mask].copy()
    log.info(f"Candidates meeting quality criteria: {len(candidates)}")

    # Merge with GTDB to get assembly level
    gtdb_sub = gtdb_df[['accession_extracted', 'ncbi_assembly_level', 'genome_size']].copy()
    candidates = candidates.merge(gtdb_sub, on='accession_extracted', how='left')

    np.random.seed(SEED)
    all_refs = []
    for species in SPECIES_LIST:
        sp_cands = candidates[candidates['species'] == species].copy()
        sp_cands['is_complete'] = sp_cands['ncbi_assembly_level'] == 'Complete Genome'
        sp_cands = sp_cands.sort_values('is_complete', ascending=False)

        complete = sp_cands[sp_cands['is_complete']]
        other = sp_cands[~sp_cands['is_complete']]

        if len(complete) >= N_REFS_PER_SPECIES:
            selected = complete.sample(n=N_REFS_PER_SPECIES, random_state=SEED)
        else:
            needed = N_REFS_PER_SPECIES - len(complete)
            selected = pd.concat([complete, other.sample(n=needed, random_state=SEED)])

        log.info(f"  {species}: {len(sp_cands)} candidates, "
                 f"{len(complete)} complete genomes, selected {len(selected)}")
        log.info(f"    Assembly levels: {selected['ncbi_assembly_level'].value_counts().to_dict()}")
        all_refs.append(selected)

    ref_df = pd.concat(all_refs)
    ref_df = ref_df[[
        'accession_extracted', 'species', 'checkm2_completeness',
        'checkm2_contamination', 'pred_completeness', 'pred_contamination',
        'ncbi_assembly_level', 'genome_size'
    ]].copy()
    ref_df.columns = [
        'accession', 'species', 'checkm2_comp', 'checkm2_cont',
        'magicc_comp', 'magicc_cont', 'assembly_level', 'genome_size'
    ]

    ref_df.to_csv(REF_ACCESSIONS_FILE, sep='\t', index=False)
    log.info(f"Saved {len(ref_df)} reference accessions to {REF_ACCESSIONS_FILE}")
    return ref_df


# ============================================================================
# Step 2: Subsample query genomes
# ============================================================================

def subsample_queries(comp_df, ref_df):
    """Subsample 10 query genomes per species x interval."""
    log.info("=" * 80)
    log.info("STEP 2: Subsample query genomes")
    log.info("=" * 80)

    ref_accessions = set(ref_df['accession'].values)
    queries = comp_df[~comp_df['accession_extracted'].isin(ref_accessions)].copy()

    np.random.seed(SEED)
    selected = []
    for species in SPECIES_LIST:
        for interval in [1, 2, 3, 4]:
            group = queries[(queries['species'] == species) & (queries['interval'] == interval)]
            n_sample = min(N_QUERIES_PER_GROUP, len(group))
            if n_sample == 0:
                log.warning(f"No genomes for {species} interval {interval}")
                continue
            sampled = group.sample(n=n_sample, random_state=SEED)
            selected.append(sampled)
            log.info(f"  {species} interval {interval}: {len(group)} available, selected {n_sample}")

    query_df = pd.concat(selected)
    query_df.to_csv(QUERY_GENOMES_FILE, sep='\t', index=False)
    log.info(f"Saved {len(query_df)} query genomes to {QUERY_GENOMES_FILE}")
    return query_df


# ============================================================================
# Step 3: Download genomes
# ============================================================================

def download_genomes(ref_df, query_df):
    """Download all reference and query genomes from NCBI."""
    log.info("=" * 80)
    log.info("STEP 3: Download genomes")
    log.info("=" * 80)

    all_accs = list(set(
        ref_df['accession'].tolist() +
        query_df['accession_extracted'].tolist()
    ))
    log.info(f"Total unique accessions to download: {len(all_accs)}")

    # Write accessions to file
    acc_file = DATA_DIR / "all_accessions_download.txt"
    with open(acc_file, 'w') as f:
        for acc in all_accs:
            f.write(acc + '\n')

    # Download with datasets CLI
    zip_file = DATA_DIR / "all_genomes.zip"
    cmd = [
        "datasets", "download", "genome", "accession",
        "--inputfile", str(acc_file),
        "--include", "genome",
        "--filename", str(zip_file),
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log.error(f"Download failed: {result.stderr}")
        sys.exit(1)
    log.info("Download complete")

    # Unzip
    extract_dir = DATA_DIR / "all_genomes_unzipped"
    subprocess.run(
        ["unzip", "-q", "-o", str(zip_file), "-d", str(extract_dir)],
        capture_output=True, text=True, timeout=300,
    )

    # Build manifest: accession -> fasta path
    data_dir = extract_dir / "ncbi_dataset" / "data"
    manifest = {}
    for fna in glob_module.glob(str(data_dir / "*" / "*.fna")):
        acc = os.path.basename(os.path.dirname(fna))
        manifest[acc] = os.path.abspath(fna)

    log.info(f"Found {len(manifest)} FASTA files")

    # Verify references
    ref_missing = [a for a in ref_df['accession'] if a not in manifest]
    query_missing = [a for a in query_df['accession_extracted'] if a not in manifest]
    if ref_missing:
        log.error(f"Missing reference genomes: {ref_missing}")
        sys.exit(1)
    if query_missing:
        log.warning(f"Missing query genomes ({len(query_missing)}): {query_missing}")

    return manifest


# ============================================================================
# Step 4: Build reference databases and align
# ============================================================================

def build_reference_databases(ref_df, manifest):
    """Build concatenated reference FASTAs and minimap2 indices for each species."""
    log.info("=" * 80)
    log.info("STEP 4a: Build reference databases")
    log.info("=" * 80)

    ref_info = {}  # species -> {ref_fasta, median_size, genome_sizes}
    for species in SPECIES_LIST:
        shortname = SPECIES_SHORTNAMES[species]
        outdir = DATA_DIR / "references" / shortname
        outdir.mkdir(parents=True, exist_ok=True)

        sp_refs = ref_df[ref_df['species'] == species]['accession'].tolist()
        ref_fasta = outdir / "reference.fasta"

        # Concatenate references, renaming sequences
        genome_sizes = []
        with open(ref_fasta, 'w') as out:
            for acc in sp_refs:
                fasta_path = manifest[acc]
                acc_bases = 0
                with open(fasta_path) as f:
                    for line in f:
                        if line.startswith('>'):
                            orig_id = line[1:].strip().split()[0]
                            out.write(f'>{acc}_{orig_id}\n')
                        else:
                            acc_bases += len(line.strip())
                            out.write(line)
                genome_sizes.append(acc_bases)

        median_size = np.median(genome_sizes)
        log.info(f"  {species}: {len(sp_refs)} refs, "
                 f"median genome = {median_size:,.0f} bp, "
                 f"range = [{min(genome_sizes):,}, {max(genome_sizes):,}]")

        # Build minimap2 index
        idx_path = outdir / "reference.mmi"
        result = subprocess.run(
            ['minimap2', '-d', str(idx_path), str(ref_fasta)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            log.error(f"Index build failed for {species}: {result.stderr[:300]}")
            sys.exit(1)

        ref_info[species] = {
            'ref_fasta': str(ref_fasta),
            'median_size': median_size,
            'genome_sizes': genome_sizes,
        }

    return ref_info


def process_single_genome(args):
    """Align a single query genome to its species reference and compute ground truth.

    This function is designed to run in a ProcessPoolExecutor worker.
    """
    acc, species, interval, ck_comp, ck_cont, mg_comp, mg_cont, ref_fasta, median_ref, query_fasta = args

    if not os.path.exists(query_fasta):
        return None

    # Read query contigs
    contig_sizes = read_contig_sizes(query_fasta)

    # Align with minimap2 (PAF output, 1 thread per worker)
    try:
        result = subprocess.run(
            ['minimap2', '-c', '-x', MINIMAP2_PRESET, '-t', '1', ref_fasta, query_fasta],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return None
    except subprocess.TimeoutExpired:
        return None

    # Parse PAF: collect high-identity alignment intervals per contig
    contig_alns = defaultdict(list)
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) < 12:
            continue
        qname = fields[0]
        qstart = int(fields[2])
        qend = int(fields[3])
        matching = int(fields[9])
        block_len = int(fields[10])
        identity = matching / block_len if block_len > 0 else 0

        if identity >= IDENTITY_THRESHOLD:
            contig_alns[qname].append((qstart, qend))

    # Determine local contigs: merged coverage >= threshold
    local_contigs = set()
    for ctg, alns in contig_alns.items():
        covered = merge_intervals(alns)
        qlen = contig_sizes.get(ctg, 0)
        if qlen > 0 and covered / qlen >= COVERAGE_THRESHOLD:
            local_contigs.add(ctg)

    # Compute ground truth
    local_bp = sum(sz for ctg, sz in contig_sizes.items() if ctg in local_contigs)
    nonlocal_bp = sum(sz for ctg, sz in contig_sizes.items() if ctg not in local_contigs)

    gt_comp = min(100.0, local_bp / median_ref * 100)
    gt_cont = nonlocal_bp / median_ref * 100

    return {
        'accession': acc,
        'species': species,
        'interval': interval,
        'n_contigs': len(contig_sizes),
        'total_bp': sum(contig_sizes.values()),
        'n_local': len(local_contigs),
        'local_bp': local_bp,
        'nonlocal_bp': nonlocal_bp,
        'ref_size': median_ref,
        'gt_completeness': round(gt_comp, 4),
        'gt_contamination': round(gt_cont, 4),
        'checkm2_completeness': ck_comp,
        'checkm2_contamination': ck_cont,
        'magicc_completeness': mg_comp,
        'magicc_contamination': mg_cont,
    }


def run_alignments(query_df, manifest, ref_info):
    """Run alignments for all query genomes in parallel."""
    log.info("=" * 80)
    log.info("STEP 4b: Align query genomes and compute ground truth")
    log.info("=" * 80)

    # Prepare tasks
    tasks = []
    skipped = 0
    for _, row in query_df.iterrows():
        acc = row['accession_extracted']
        species = row['species']
        if acc not in manifest:
            skipped += 1
            continue
        tasks.append((
            acc,
            species,
            row['interval'],
            row['checkm2_completeness'],
            row['checkm2_contamination'],
            row['pred_completeness'],
            row['pred_contamination'],
            ref_info[species]['ref_fasta'],
            ref_info[species]['median_size'],
            manifest[acc],
        ))

    if skipped > 0:
        log.warning(f"Skipped {skipped} genomes (not downloaded)")

    log.info(f"Processing {len(tasks)} query genomes with {THREADS} parallel workers...")
    start_time = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(process_single_genome, task): task[0] for task in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                results.append(result)
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.time() - start_time
                log.info(f"  {completed}/{len(tasks)} done ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    log.info(f"Completed {len(results)}/{len(tasks)} genomes in {elapsed:.1f}s")

    return pd.DataFrame(results)


# ============================================================================
# Step 6: Compute MAE comparison
# ============================================================================

def compute_maes(results_df):
    """Compute MAE tables comparing tools vs ground truth."""
    log.info("=" * 80)
    log.info("STEP 6: Compute MAE comparison")
    log.info("=" * 80)

    def group_maes(group):
        n = len(group)
        return pd.Series({
            'N': n,
            'GT_Comp_mean': round(group['gt_completeness'].mean(), 2),
            'GT_Cont_mean': round(group['gt_contamination'].mean(), 2),
            'MAGICC_Comp_MAE': round(
                np.abs(group['magicc_completeness'] - group['gt_completeness']).mean(), 2),
            'MAGICC_Cont_MAE': round(
                np.abs(group['magicc_contamination'] - group['gt_contamination']).mean(), 2),
            'CheckM2_Comp_MAE': round(
                np.abs(group['checkm2_completeness'] - group['gt_completeness']).mean(), 2),
            'CheckM2_Cont_MAE': round(
                np.abs(group['checkm2_contamination'] - group['gt_contamination']).mean(), 2),
        })

    # Per species x interval
    summary = results_df.groupby(['species', 'interval'], group_keys=False).apply(
        group_maes
    ).reset_index()

    log.info("\nPer Species x Interval:")
    log.info(summary.to_string(index=False))

    # Per species overall
    sp_summary = results_df.groupby('species', group_keys=False).apply(
        group_maes
    ).reset_index()
    sp_summary['interval'] = 'ALL'

    log.info("\nPer Species (overall):")
    log.info(sp_summary.to_string(index=False))

    # Overall
    overall = group_maes(results_df)
    overall['species'] = 'OVERALL'
    overall['interval'] = 'ALL'

    log.info(f"\nOverall:")
    log.info(f"  MAGICC  - Comp MAE: {overall['MAGICC_Comp_MAE']}, "
             f"Cont MAE: {overall['MAGICC_Cont_MAE']}")
    log.info(f"  CheckM2 - Comp MAE: {overall['CheckM2_Comp_MAE']}, "
             f"Cont MAE: {overall['CheckM2_Cont_MAE']}")

    # Per-genome winner analysis
    magicc_comp_closer = (
        np.abs(results_df['magicc_completeness'] - results_df['gt_completeness']) <
        np.abs(results_df['checkm2_completeness'] - results_df['gt_completeness'])
    ).sum()
    magicc_cont_closer = (
        np.abs(results_df['magicc_contamination'] - results_df['gt_contamination']) <
        np.abs(results_df['checkm2_contamination'] - results_df['gt_contamination'])
    ).sum()
    n = len(results_df)
    log.info(f"\nPer-genome: MAGICC closer on completeness: "
             f"{magicc_comp_closer}/{n} ({magicc_comp_closer/n*100:.1f}%)")
    log.info(f"Per-genome: MAGICC closer on contamination: "
             f"{magicc_cont_closer}/{n} ({magicc_cont_closer/n*100:.1f}%)")

    # Build summary dict
    summary_dict = {
        'per_species_interval': summary.to_dict(orient='records'),
        'per_species': sp_summary.to_dict(orient='records'),
        'overall': overall.to_dict(),
        'n_genomes': n,
        'magicc_closer_completeness': int(magicc_comp_closer),
        'magicc_closer_contamination': int(magicc_cont_closer),
        'parameters': {
            'identity_threshold': IDENTITY_THRESHOLD,
            'coverage_threshold': COVERAGE_THRESHOLD,
            'minimap2_preset': MINIMAP2_PRESET,
            'n_references_per_species': N_REFS_PER_SPECIES,
            'n_queries_per_group': N_QUERIES_PER_GROUP,
            'seed': SEED,
        },
    }

    return summary_dict


# ============================================================================
# Cleanup
# ============================================================================

def cleanup():
    """Remove downloaded genomes and temporary files to conserve disk space."""
    log.info("=" * 80)
    log.info("Cleanup: Removing downloaded genomes and temporary files")
    log.info("=" * 80)

    for path in [
        DATA_DIR / "all_genomes_unzipped",
        DATA_DIR / "all_genomes.zip",
        DATA_DIR / "references",
        DATA_DIR / "all_accessions_download.txt",
        DATA_DIR / "genome_manifest.tsv",
    ]:
        if path.is_dir():
            size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e6
            shutil.rmtree(path)
            log.info(f"  Removed {path} ({size_mb:.1f} MB)")
        elif path.is_file():
            path.unlink()
            log.info(f"  Removed {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    log.info("=" * 80)
    log.info("Alignment-based ground-truth validation: MAGICC vs CheckM2 on pathogen genomes")
    log.info("=" * 80)
    log.info(f"Parameters: identity>={IDENTITY_THRESHOLD}, coverage>={COVERAGE_THRESHOLD}, "
             f"preset={MINIMAP2_PRESET}, refs/species={N_REFS_PER_SPECIES}, seed={SEED}")

    # Load data
    comp_df = pd.read_csv(COMPARISON_FILE, sep='\t')
    log.info(f"Loaded comparison results: {len(comp_df)} genomes")

    gtdb_cols = ['accession', 'gtdb_taxonomy', 'ncbi_assembly_level',
                 'checkm2_completeness', 'checkm2_contamination', 'genome_size']
    gtdb_df = pd.read_csv(GTDB_FILE, sep='\t', usecols=gtdb_cols, compression='gzip')
    gtdb_df['accession_extracted'] = gtdb_df['accession'].str.replace(r'^[A-Z]{2}_', '', regex=True)
    log.info(f"Loaded GTDB metadata: {len(gtdb_df)} genomes")

    # Step 1: Select references
    ref_df = select_references(comp_df, gtdb_df)

    # Step 2: Subsample queries
    query_df = subsample_queries(comp_df, ref_df)

    # Step 3: Download genomes
    manifest = download_genomes(ref_df, query_df)

    # Step 4a: Build reference databases
    ref_info = build_reference_databases(ref_df, manifest)

    # Step 4b-5: Align and compute ground truth
    results_df = run_alignments(query_df, manifest, ref_info)

    # Save per-genome results
    results_df.to_csv(VALIDATION_RESULTS_FILE, sep='\t', index=False)
    log.info(f"Saved per-genome results to {VALIDATION_RESULTS_FILE}")

    # Step 6: Compute MAE comparison
    summary_dict = compute_maes(results_df)

    with open(VALIDATION_SUMMARY_FILE, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    log.info(f"Saved validation summary to {VALIDATION_SUMMARY_FILE}")

    # Cleanup
    cleanup()

    log.info("\nDone!")


if __name__ == "__main__":
    main()
