#!/usr/bin/env python3
"""
Motivating Analysis - Benchmark Dataset Generation (Sets A_motiv and B_motiv)

Generates NEW benchmark datasets using different genomes from the existing Sets A/B
to demonstrate limitations of existing tools (CheckM2, DeepCheck, CoCoPyE).

Set A_motiv (Completeness gradient):
  - 0% contamination (pure genomes)
  - 6 completeness levels: 50%, 60%, 70%, 80%, 90%, 100%
  - 100 genomes per level = 600 total
  - For 100% completeness: use load_original_contigs() (no fragmentation)
  - For other levels: medium quality tier fragmentation
  - Random reference genomes from test split (avoiding overlap with existing Sets A/B)

Set B_motiv (Contamination gradient):
  - 100% completeness (original contigs via load_original_contigs(), NO fragmentation)
  - 11 contamination levels: 0%, 5%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 80%, 100%
  - 100 genomes per level = 1,100 total
  - Cross-phylum contamination
  - Contamination = contaminant_bp / dominant_genome_FULL_size x 100

Output:
  data/benchmarks/motivating/set_A/fasta/genome_{i}.fasta
  data/benchmarks/motivating/set_B/fasta/genome_{i}.fasta
  data/benchmarks/motivating/set_A/metadata.tsv
  data/benchmarks/motivating/set_B/metadata.tsv
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.fragmentation import (
    simulate_fragmentation, read_fasta, load_original_contigs,
    _warm_numba_fragmentation
)
from magicc.contamination import (
    generate_contaminated_sample, generate_pure_sample
)

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'
BENCHMARK_DIR = DATA_DIR / 'benchmarks' / 'motivating'
SPLITS_DIR = DATA_DIR / 'splits'
EXISTING_BENCHMARK_DIR = DATA_DIR / 'benchmarks'

N_WORKERS = max(1, int(cpu_count() * 0.90))  # ~43 workers
SEED = 12345  # Different seed from existing sets (which use 42)


# ============================================================================
# Helper functions
# ============================================================================

def write_fasta(contigs: List[str], fasta_path: str, genome_id: str = "genome"):
    """Write contigs to FASTA file."""
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as f:
        for i, contig in enumerate(contigs):
            f.write(f">{genome_id}_contig_{i} len={len(contig)}\n")
            for j in range(0, len(contig), 80):
                f.write(contig[j:j+80])
                f.write('\n')


def get_excluded_accessions():
    """Get accessions used in existing benchmark Sets A and B to avoid overlap."""
    excluded = set()
    for set_name in ['set_A', 'set_B']:
        meta_path = EXISTING_BENCHMARK_DIR / set_name / 'metadata.tsv'
        if meta_path.exists():
            df = pd.read_csv(meta_path, sep='\t')
            excluded.update(df['dominant_accession'].unique())
    return excluded


def get_cross_phylum_contaminants(dominant_phylum: str, test_df: pd.DataFrame,
                                   rng: np.random.Generator, n_contaminants: int = 1) -> List[str]:
    """Select random cross-phylum contaminant genome sequences from test set."""
    candidates = test_df[test_df['phylum'] != dominant_phylum]
    if len(candidates) == 0:
        candidates = test_df  # fallback

    selected = candidates.sample(n=min(n_contaminants, len(candidates)),
                                  random_state=int(rng.integers(0, 2**31)))
    sequences = []
    for _, row in selected.iterrows():
        fasta_path = row['fasta_path']
        if os.path.exists(fasta_path):
            seq = read_fasta(fasta_path)
            if len(seq) > 0:
                sequences.append(seq)
    return sequences


# ============================================================================
# Worker functions
# ============================================================================

def generate_set_a_genome(args):
    """Generate a single Set A_motiv genome (completeness gradient, 0% contamination)."""
    idx, row_dict, target_completeness, fasta_dir = args
    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    # Resumable
    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None

    rng = np.random.default_rng(SEED + idx + 500000)
    accession = row_dict['gtdb_accession']
    phylum = row_dict['phylum']
    src_fasta = row_dict['fasta_path']

    try:
        if target_completeness >= 1.0:
            # 100% completeness: use original contigs (no fragmentation)
            contigs = load_original_contigs(src_fasta)
            actual_completeness = 1.0
        else:
            sequence = read_fasta(src_fasta)
            if len(sequence) == 0:
                return None
            result = simulate_fragmentation(
                sequence,
                target_completeness=target_completeness,
                quality_tier='medium',  # medium quality tier as specified
                rng=rng
            )
            contigs = result['contigs']
            actual_completeness = result['completeness']

        if len(contigs) == 0:
            return None

        total_length = sum(len(c) for c in contigs)
        write_fasta(contigs, fasta_path, genome_id)

        return {
            'genome_id': genome_id,
            'true_completeness': actual_completeness * 100.0,
            'true_contamination': 0.0,
            'dominant_accession': accession,
            'dominant_phylum': phylum,
            'sample_type': f'set_a_comp{int(target_completeness*100)}',
            'n_contigs': len(contigs),
            'total_length': total_length,
        }
    except Exception as e:
        print(f"  Error generating {genome_id}: {e}", flush=True)
        return None


def generate_set_b_genome(args):
    """Generate a single Set B_motiv genome (contamination gradient, 100% completeness)."""
    idx, row_dict, target_contamination, fasta_dir, test_df_path = args
    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None

    rng = np.random.default_rng(SEED + idx + 600000)
    accession = row_dict['gtdb_accession']
    phylum = row_dict['phylum']
    src_fasta = row_dict['fasta_path']

    try:
        test_df = pd.read_csv(test_df_path, sep='\t')

        # 100% completeness: use original contigs (no fragmentation)
        dominant_contigs = load_original_contigs(src_fasta)
        if len(dominant_contigs) == 0:
            return None

        dominant_sequence = read_fasta(src_fasta)
        dominant_full_length = len(dominant_sequence)

        if target_contamination <= 0:
            contigs = dominant_contigs
            actual_contamination = 0.0
        else:
            # Cross-phylum contaminants: 1-5 genomes
            n_cont = rng.integers(1, 6)
            contaminant_sequences = get_cross_phylum_contaminants(
                phylum, test_df, rng, n_cont
            )

            if len(contaminant_sequences) == 0:
                contigs = dominant_contigs
                actual_contamination = 0.0
            else:
                # Generate contaminated sample using the contamination module
                result = generate_contaminated_sample(
                    dominant_sequence=dominant_sequence,
                    contaminant_sequences=contaminant_sequences,
                    target_completeness=1.0,
                    target_contamination=target_contamination,
                    rng=rng,
                )
                # Replace dominant contigs with original (unfragmented) contigs
                # but keep the contaminant contigs
                contaminant_contigs = result['contaminant_contigs']
                contigs = dominant_contigs + contaminant_contigs

                # Shuffle
                indices = list(range(len(contigs)))
                rng.shuffle(indices)
                contigs = [contigs[i] for i in indices]

                contaminant_bp = sum(len(c) for c in contaminant_contigs)
                actual_contamination = 100.0 * contaminant_bp / dominant_full_length

        total_length = sum(len(c) for c in contigs)
        write_fasta(contigs, fasta_path, genome_id)

        return {
            'genome_id': genome_id,
            'true_completeness': 100.0,
            'true_contamination': actual_contamination,
            'dominant_accession': accession,
            'dominant_phylum': phylum,
            'sample_type': f'set_b_cont{int(target_contamination)}',
            'n_contigs': len(contigs),
            'total_length': total_length,
        }
    except Exception as e:
        print(f"  Error generating {genome_id}: {e}", flush=True)
        return None


# ============================================================================
# Set generation functions
# ============================================================================

def generate_set_a(test_df: pd.DataFrame, excluded_accessions: set):
    """Generate Set A_motiv: Completeness gradient."""
    print("\n" + "="*70)
    print("Generating Set A_motiv: Completeness Gradient")
    print("="*70)

    set_dir = BENCHMARK_DIR / 'set_A'
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    completeness_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    n_per_level = 100

    # Filter out excluded accessions
    available = test_df[~test_df['gtdb_accession'].isin(excluded_accessions)]
    print(f"  Available test genomes (excluding overlap): {len(available)} / {len(test_df)}")

    all_results = []
    idx = 0

    for comp in completeness_levels:
        print(f"\n  Completeness = {comp*100:.0f}%")

        # Sample 100 random test genomes with different seed per level
        rng_sample = np.random.default_rng(SEED + int(comp * 1000) + 7777)
        sampled = available.sample(n=n_per_level,
                                    random_state=int(rng_sample.integers(0, 2**31)))

        tasks = []
        for _, row in sampled.iterrows():
            tasks.append((idx, row.to_dict(), comp, str(fasta_dir)))
            idx += 1

        with Pool(N_WORKERS) as pool:
            results = pool.map(generate_set_a_genome, tasks)

        valid = [r for r in results if r is not None]
        all_results.extend(valid)
        print(f"    Generated {len(valid)}/{n_per_level} genomes")

    # Reconcile with existing files
    all_results = _reconcile_results(all_results, fasta_dir, idx)
    _save_set_outputs(set_dir, all_results)
    print(f"\n  Set A_motiv total: {len(all_results)} genomes")
    return all_results


def generate_set_b(test_df: pd.DataFrame, excluded_accessions: set):
    """Generate Set B_motiv: Contamination gradient."""
    print("\n" + "="*70)
    print("Generating Set B_motiv: Contamination Gradient")
    print("="*70)

    set_dir = BENCHMARK_DIR / 'set_B'
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    contamination_levels = [0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    n_per_level = 100
    test_df_path = str(SPLITS_DIR / 'test_genomes.tsv')

    # Filter out excluded accessions
    available = test_df[~test_df['gtdb_accession'].isin(excluded_accessions)]
    print(f"  Available test genomes (excluding overlap): {len(available)} / {len(test_df)}")

    all_results = []
    idx = 0

    for cont in contamination_levels:
        print(f"\n  Contamination = {cont}%")

        rng_sample = np.random.default_rng(SEED + cont + 9999)
        sampled = available.sample(n=n_per_level,
                                    random_state=int(rng_sample.integers(0, 2**31)))

        tasks = []
        for _, row in sampled.iterrows():
            tasks.append((idx, row.to_dict(), float(cont), str(fasta_dir), test_df_path))
            idx += 1

        with Pool(N_WORKERS) as pool:
            results = pool.map(generate_set_b_genome, tasks)

        valid = [r for r in results if r is not None]
        all_results.extend(valid)
        print(f"    Generated {len(valid)}/{n_per_level} genomes")

    all_results = _reconcile_results(all_results, fasta_dir, idx)
    _save_set_outputs(set_dir, all_results)
    print(f"\n  Set B_motiv total: {len(all_results)} genomes")
    return all_results


# ============================================================================
# Utility functions
# ============================================================================

def _reconcile_results(results: List[Dict], fasta_dir, n_total):
    """Reconcile results with existing files for resumability."""
    existing_ids = {r['genome_id'] for r in results}
    fasta_dir_path = Path(fasta_dir)

    if fasta_dir_path.exists():
        for fasta_file in fasta_dir_path.glob("genome_*.fasta"):
            gid = fasta_file.stem
            if gid not in existing_ids and fasta_file.stat().st_size > 0:
                n_contigs = 0
                total_length = 0
                with open(fasta_file) as f:
                    for line in f:
                        if line.startswith('>'):
                            n_contigs += 1
                        else:
                            total_length += len(line.strip())
                results.append({
                    'genome_id': gid,
                    'true_completeness': -1.0,
                    'true_contamination': -1.0,
                    'dominant_accession': 'unknown',
                    'dominant_phylum': 'unknown',
                    'sample_type': 'resumed',
                    'n_contigs': n_contigs,
                    'total_length': total_length,
                })

    return results


def _save_set_outputs(set_dir: Path, results: List[Dict]):
    """Save metadata.tsv and labels.npy for a benchmark set."""
    if len(results) == 0:
        print("  WARNING: No genomes generated!")
        return

    valid_results = [r for r in results if r['true_completeness'] >= 0]
    df = pd.DataFrame(valid_results)
    df = df.sort_values('genome_id').reset_index(drop=True)

    metadata_path = set_dir / 'metadata.tsv'
    df.to_csv(metadata_path, sep='\t', index=False)
    print(f"  Saved metadata: {metadata_path} ({len(df)} rows)")

    labels = df[['true_completeness', 'true_contamination']].values.astype(np.float32)
    labels_path = set_dir / 'labels.npy'
    np.save(labels_path, labels)
    print(f"  Saved labels: {labels_path} ({labels.shape})")


# ============================================================================
# Validation
# ============================================================================

def validate_motivating_sets():
    """Validate both motivating benchmark sets."""
    print("\n" + "="*70)
    print("VALIDATION OF MOTIVATING BENCHMARK SETS")
    print("="*70)

    for set_name in ['A', 'B']:
        set_dir = BENCHMARK_DIR / f'set_{set_name}'
        metadata_path = set_dir / 'metadata.tsv'
        labels_path = set_dir / 'labels.npy'
        fasta_dir = set_dir / 'fasta'

        if not metadata_path.exists():
            print(f"\n  Set {set_name}: MISSING")
            continue

        print(f"\n  {'='*60}")
        print(f"  Set {set_name}_motiv")
        print(f"  {'='*60}")

        df = pd.read_csv(metadata_path, sep='\t')
        labels = np.load(labels_path)

        comp = labels[:, 0]
        cont = labels[:, 1]

        print(f"  Total genomes: {len(df)}")
        print(f"  Completeness:  mean={comp.mean():.1f}%, std={comp.std():.1f}%, "
              f"min={comp.min():.1f}%, max={comp.max():.1f}%")
        print(f"  Contamination: mean={cont.mean():.1f}%, std={cont.std():.1f}%, "
              f"min={cont.min():.1f}%, max={cont.max():.1f}%")

        # FASTA integrity check
        n_missing = 0
        n_empty = 0
        total_size_bytes = 0
        for _, row in df.iterrows():
            fasta_path = fasta_dir / f"{row['genome_id']}.fasta"
            if not fasta_path.exists():
                n_missing += 1
            elif fasta_path.stat().st_size == 0:
                n_empty += 1
            else:
                total_size_bytes += fasta_path.stat().st_size

        total_size_gb = total_size_bytes / (1024**3)
        print(f"  FASTA files: {len(df)} expected, {n_missing} missing, {n_empty} empty")
        print(f"  Total FASTA size: {total_size_gb:.2f} GB")

        # Per-level validation
        if set_name == 'A':
            print("\n  Per-level completeness validation:")
            for stype in sorted(df['sample_type'].unique()):
                mask = df['sample_type'] == stype
                level_comp = comp[mask]
                level_cont = cont[mask]
                target = int(stype.split('comp')[-1])

                within_tol = np.abs(level_comp - target) <= 10
                pct_pass = 100 * within_tol.sum() / len(level_comp)

                print(f"    {stype}: n={len(level_comp)}, "
                      f"comp mean={level_comp.mean():.1f}% (target={target}%, "
                      f"within +/-10%: {pct_pass:.0f}%), "
                      f"cont mean={level_cont.mean():.1f}%")

        elif set_name == 'B':
            print("\n  Per-level contamination validation:")
            for stype in sorted(df['sample_type'].unique(), key=lambda x: int(x.split('cont')[-1])):
                mask = df['sample_type'] == stype
                level_comp = comp[mask]
                level_cont = cont[mask]
                target = int(stype.split('cont')[-1])

                within_tol = np.abs(level_cont - target) <= 15
                pct_pass = 100 * within_tol.sum() / len(level_cont)

                mean_err = np.mean(level_cont - target)

                print(f"    {stype}: n={len(level_cont)}, "
                      f"cont mean={level_cont.mean():.1f}% (target={target}%, "
                      f"mean error={mean_err:+.1f}%, within +/-15%: {pct_pass:.0f}%), "
                      f"comp mean={level_comp.mean():.1f}%")

            # Verify all 100% completeness
            comp_ok = np.abs(comp - 100) <= 1
            print(f"\n    Completeness == 100%: {comp_ok.sum()}/{len(comp)} "
                  f"({100*comp_ok.sum()/len(comp):.0f}%)")

        print(f"  FASTA integrity: {'PASS' if (n_missing == 0 and n_empty == 0) else 'FAIL'}")


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print(f"Motivating Benchmark Generation Script")
    print(f"Workers: {N_WORKERS}")
    print(f"Seed: {SEED}")

    # Parse command-line arguments
    sets_to_generate = set()
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.upper() in ('A', 'B'):
                sets_to_generate.add(arg.upper())
            elif arg.lower() == 'validate':
                sets_to_generate.add('VALIDATE')
    if not sets_to_generate:
        sets_to_generate = {'A', 'B', 'VALIDATE'}

    # Load test genomes
    print("\nLoading genome data...")
    test_df = pd.read_csv(SPLITS_DIR / 'test_genomes.tsv', sep='\t')
    print(f"  Test genomes: {len(test_df)}")

    # Get excluded accessions (already used in existing Sets A/B)
    excluded = get_excluded_accessions()
    print(f"  Excluded accessions (from existing Sets A/B): {len(excluded)}")
    available = test_df[~test_df['gtdb_accession'].isin(excluded)]
    print(f"  Available test genomes: {len(available)}")

    # Warm up Numba
    print("Warming up Numba JIT...")
    _warm_numba_fragmentation()

    # Generate sets
    if 'A' in sets_to_generate:
        generate_set_a(test_df, excluded)

    if 'B' in sets_to_generate:
        generate_set_b(test_df, excluded)

    # Validate
    if 'VALIDATE' in sets_to_generate or sets_to_generate.intersection({'A', 'B'}):
        validate_motivating_sets()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
