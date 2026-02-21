#!/usr/bin/env python3
"""
Generate Motivating Set C: Realistic mixed set (1,000 genomes)

- Composition:
  - 200 pure genomes: 0% contamination, 50-100% completeness (uniform)
  - 200 complete genomes: 100% completeness (original contigs), 0-100% contamination (uniform)
  - 600 others: mixed contaminated genomes, 50-100% completeness, 0-100% contamination
    - 70% cross-phylum, 30% within-phylum contamination
- Fragmentation: 20-200 contigs, >1kbp min contig (for non-complete genomes)
- Dominant genomes: ONLY finished/complete genomes (from test_finished_genomes.tsv)
- Contaminant genomes: ALL test reference genomes
- Seed: 300

Output: data/benchmarks/motivating_v2/set_C/
  - fasta/genome_{i}.fasta (1,000 FASTA files)
  - metadata.tsv (ground truth)
  - labels.npy

Modeled after generate_set_e() in scripts/34_generate_finished_benchmarks.py
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
    generate_contaminated_sample, fragment_contaminant
)

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'
SPLITS_DIR = DATA_DIR / 'splits'

FINISHED_GENOMES_PATH = SPLITS_DIR / 'test_finished_genomes.tsv'
ALL_TEST_GENOMES_PATH = SPLITS_DIR / 'test_genomes.tsv'

OUTPUT_DIR = BENCHMARK_DIR / 'motivating_v2' / 'set_C'

SEED = 300
N_TOTAL = 1000
N_PURE = 200
N_COMPLETE = 200
N_OTHER = N_TOTAL - N_PURE - N_COMPLETE  # 600

N_WORKERS = max(1, int(cpu_count() * 0.90))


# ============================================================================
# Helper functions (same as 34_generate_finished_benchmarks.py)
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


def get_cross_phylum_contaminants(dominant_phylum: str, test_df: pd.DataFrame,
                                   rng: np.random.Generator, n_contaminants: int = 1) -> List[str]:
    """Select random cross-phylum contaminant genome sequences from ALL test genomes."""
    candidates = test_df[test_df['phylum'] != dominant_phylum]
    if len(candidates) == 0:
        candidates = test_df
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


def get_within_phylum_contaminants(dominant_phylum: str, dominant_accession: str,
                                    test_df: pd.DataFrame, rng: np.random.Generator,
                                    n_contaminants: int = 1) -> List[str]:
    """Select random within-phylum contaminant genome sequences."""
    candidates = test_df[(test_df['phylum'] == dominant_phylum) &
                         (test_df['gtdb_accession'] != dominant_accession)]
    if len(candidates) == 0:
        candidates = test_df[test_df['gtdb_accession'] != dominant_accession]
    if len(candidates) == 0:
        candidates = test_df
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
# Worker function (same pattern as generate_set_e_genome from script 34)
# ============================================================================

def generate_set_c_genome(args):
    """Generate a single genome for Set C (realistic mixed)."""
    (idx, row_dict, target_completeness, target_contamination,
     n_target_contigs, min_contig_bp, contamination_type,
     fasta_dir, all_test_df_path, seed) = args

    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None  # Skip existing

    rng = np.random.default_rng(seed + idx)
    accession = row_dict['gtdb_accession']
    phylum = row_dict['phylum']
    src_fasta = row_dict['fasta_path']

    try:
        test_df = pd.read_csv(all_test_df_path, sep='\t')

        dominant_sequence = read_fasta(src_fasta)
        if len(dominant_sequence) == 0:
            return None
        dominant_full_length = len(dominant_sequence)

        # Fragment dominant genome
        if target_completeness >= 1.0:
            dominant_contigs = load_original_contigs(src_fasta)
            actual_completeness = 1.0
        else:
            frag_result = simulate_fragmentation(
                dominant_sequence,
                target_completeness=target_completeness,
                quality_tier='medium',
                rng=rng,
            )
            dominant_contigs = frag_result['contigs']
            actual_completeness = frag_result['completeness']
            if len(dominant_contigs) == 0:
                return None

        # Handle contamination
        if target_contamination <= 0:
            contigs = dominant_contigs
            actual_contamination = 0.0
        else:
            if contamination_type in ('cross_phylum', 'complete_cross_phylum'):
                n_cont = rng.integers(1, 6)
                contaminant_sequences = get_cross_phylum_contaminants(
                    phylum, test_df, rng, n_cont
                )
            else:  # within_phylum or complete_within_phylum
                n_cont = rng.integers(1, 4)
                contaminant_sequences = get_within_phylum_contaminants(
                    phylum, accession, test_df, rng, n_cont
                )

            if len(contaminant_sequences) == 0:
                contigs = dominant_contigs
                actual_contamination = 0.0
            elif contamination_type.startswith('complete'):
                # For "complete" genomes: use original contigs + add contaminant contigs on top
                # Similar to generate_set_b_genome in script 34
                result = generate_contaminated_sample(
                    dominant_sequence=dominant_sequence,
                    contaminant_sequences=contaminant_sequences,
                    target_completeness=1.0,
                    target_contamination=target_contamination,
                    rng=rng,
                )
                contaminant_contigs = result['contaminant_contigs']
                contigs = dominant_contigs + contaminant_contigs

                indices = list(range(len(contigs)))
                rng.shuffle(indices)
                contigs = [contigs[i] for i in indices]

                contaminant_bp = sum(len(c) for c in contaminant_contigs)
                actual_contamination = 100.0 * contaminant_bp / dominant_full_length
                actual_completeness = 1.0

                # Cap contamination if overshooting
                if actual_contamination > target_contamination * 1.2 + 5:
                    max_cont_bp = int(target_contamination / 100.0 * dominant_full_length)
                    kept_cont = []
                    kept_bp = 0
                    for c in contaminant_contigs:
                        if kept_bp + len(c) <= max_cont_bp:
                            kept_cont.append(c)
                            kept_bp += len(c)
                        elif max_cont_bp - kept_bp >= 500:
                            kept_cont.append(c[:max_cont_bp - kept_bp])
                            kept_bp = max_cont_bp
                            break
                        else:
                            break
                    contigs = dominant_contigs + kept_cont
                    indices = list(range(len(contigs)))
                    rng.shuffle(indices)
                    contigs = [contigs[i] for i in indices]
                    actual_contamination = 100.0 * kept_bp / dominant_full_length
            else:
                result = generate_contaminated_sample(
                    dominant_sequence=dominant_sequence,
                    contaminant_sequences=contaminant_sequences,
                    target_completeness=target_completeness,
                    target_contamination=target_contamination,
                    rng=rng,
                )
                contigs = result['contigs']
                actual_completeness_raw = result['completeness']
                actual_contamination = result['contamination']

                # Use actual completeness from the contaminated sample
                if actual_completeness_raw <= 1.0:
                    actual_completeness = actual_completeness_raw
                else:
                    actual_completeness = actual_completeness_raw / 100.0

                # Cap contamination if overshooting
                if actual_contamination > target_contamination * 1.2 + 5:
                    max_cont_bp = int(target_contamination / 100.0 * dominant_full_length)
                    dom_contigs = result['dominant_contigs']
                    cont_contigs = result['contaminant_contigs']
                    kept_cont = []
                    kept_bp = 0
                    for c in cont_contigs:
                        if kept_bp + len(c) <= max_cont_bp:
                            kept_cont.append(c)
                            kept_bp += len(c)
                        elif max_cont_bp - kept_bp >= 500:
                            kept_cont.append(c[:max_cont_bp - kept_bp])
                            kept_bp = max_cont_bp
                            break
                        else:
                            break
                    contigs = dom_contigs + kept_cont
                    indices = list(range(len(contigs)))
                    rng.shuffle(indices)
                    contigs = [contigs[i] for i in indices]
                    actual_contamination = 100.0 * kept_bp / dominant_full_length

        if len(contigs) == 0:
            return None

        total_length = sum(len(c) for c in contigs)
        write_fasta(contigs, fasta_path, genome_id)

        return {
            'genome_id': genome_id,
            'true_completeness': actual_completeness * 100.0 if actual_completeness <= 1.0 else actual_completeness,
            'true_contamination': min(actual_contamination, 100.0),
            'dominant_accession': accession,
            'dominant_phylum': phylum,
            'sample_type': f'set_c_{contamination_type}',
            'n_contigs': len(contigs),
            'total_length': total_length,
            'target_completeness': target_completeness * 100.0,
            'target_contamination': target_contamination,
            'category': 'complete' if contamination_type.startswith('complete') else ('pure' if contamination_type == 'pure' else 'other'),
        }
    except Exception as e:
        print(f"  Error generating {genome_id}: {e}", flush=True)
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("Motivating Set C: Realistic Mixed Set (1,000 genomes)")
    print("=" * 70)
    print(f"  Seed: {SEED}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Output: {OUTPUT_DIR}")

    # Load genome data
    print("\nLoading genome data...")
    finished_df = pd.read_csv(FINISHED_GENOMES_PATH, sep='\t')
    print(f"  Finished test genomes: {len(finished_df)} (from {finished_df['phylum'].nunique()} phyla)")
    all_test_df_path = str(ALL_TEST_GENOMES_PATH)

    # Warm up Numba
    print("Warming up Numba JIT...")
    _warm_numba_fragmentation()

    # Create output directory
    fasta_dir = OUTPUT_DIR / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    # Prepare parameters
    rng = np.random.default_rng(SEED)

    # Build per-genome parameters for 3 categories:
    # Category 1: Pure (200) - 50-100% completeness, 0% contamination
    # Category 2: Complete (200) - 100% completeness, 0-100% contamination
    # Category 3: Other (600) - 50-100% completeness, 0-100% contamination

    completeness_values = np.zeros(N_TOTAL)
    contamination_values = np.zeros(N_TOTAL)
    contamination_types = [''] * N_TOTAL

    # Pure genomes (indices 0..199)
    completeness_values[:N_PURE] = rng.uniform(0.50, 1.0, size=N_PURE)
    contamination_values[:N_PURE] = 0.0
    for i in range(N_PURE):
        contamination_types[i] = 'pure'

    # Complete genomes (indices 200..399)
    completeness_values[N_PURE:N_PURE+N_COMPLETE] = 1.0  # 100% completeness
    contamination_values[N_PURE:N_PURE+N_COMPLETE] = rng.uniform(0.0, 100.0, size=N_COMPLETE)
    for i in range(N_PURE, N_PURE + N_COMPLETE):
        if contamination_values[i] <= 0:
            contamination_types[i] = 'complete_pure'
        elif rng.random() < 0.7:
            contamination_types[i] = 'complete_cross_phylum'
        else:
            contamination_types[i] = 'complete_within_phylum'

    # Other genomes (indices 400..999)
    completeness_values[N_PURE+N_COMPLETE:] = rng.uniform(0.50, 1.0, size=N_OTHER)
    contamination_values[N_PURE+N_COMPLETE:] = rng.uniform(0.0, 100.0, size=N_OTHER)
    for i in range(N_PURE + N_COMPLETE, N_TOTAL):
        if rng.random() < 0.7:
            contamination_types[i] = 'cross_phylum'
        else:
            contamination_types[i] = 'within_phylum'

    # Shuffle so categories are mixed in
    shuffle_idx = rng.permutation(N_TOTAL)
    completeness_values = completeness_values[shuffle_idx]
    contamination_values = contamination_values[shuffle_idx]
    contamination_types = [contamination_types[i] for i in shuffle_idx]

    # Fragmentation: 20-200 contigs, >1kbp min contig (only used for non-complete genomes)
    n_target_contigs_values = rng.integers(20, 201, size=N_TOTAL)
    min_contig_bp = 1000  # >1kbp

    # Sample finished genomes (with replacement if needed)
    genome_indices = rng.choice(len(finished_df), size=N_TOTAL, replace=True)

    tasks = []
    for idx in range(N_TOTAL):
        row = finished_df.iloc[genome_indices[idx]]
        tasks.append((
            idx, row.to_dict(),
            float(completeness_values[idx]),
            float(contamination_values[idx]),
            int(n_target_contigs_values[idx]),
            min_contig_bp,
            contamination_types[idx],
            str(fasta_dir),
            all_test_df_path,
            SEED,
        ))

    # Count types
    n_pure_actual = sum(1 for ct in contamination_types if ct == 'pure')
    n_complete_actual = sum(1 for ct in contamination_types if ct.startswith('complete'))
    n_cross = sum(1 for ct in contamination_types if ct == 'cross_phylum')
    n_within = sum(1 for ct in contamination_types if ct == 'within_phylum')

    print(f"\n  Generating {N_TOTAL} genomes with {N_WORKERS} workers...")
    print(f"  Pure (0% contamination, 50-100% comp): {n_pure_actual}")
    print(f"  Complete (100% completeness, 0-100% cont): {n_complete_actual}")
    print(f"  Cross-phylum contaminated: {n_cross}")
    print(f"  Within-phylum contaminated: {n_within}")

    with Pool(N_WORKERS) as pool:
        results = pool.map(generate_set_c_genome, tasks)

    valid_results = [r for r in results if r is not None]
    print(f"\n  Generated {len(valid_results)}/{N_TOTAL} genomes")

    # Save metadata
    if len(valid_results) == 0:
        print("  ERROR: No genomes generated!")
        return

    df = pd.DataFrame(valid_results)
    df = df.sort_values('genome_id').reset_index(drop=True)

    metadata_path = OUTPUT_DIR / 'metadata.tsv'
    df.to_csv(metadata_path, sep='\t', index=False)
    print(f"  Saved metadata: {metadata_path} ({len(df)} rows)")

    labels = df[['true_completeness', 'true_contamination']].values.astype(np.float32)
    labels_path = OUTPUT_DIR / 'labels.npy'
    np.save(labels_path, labels)
    print(f"  Saved labels: {labels_path} ({labels.shape})")

    # Validation
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")
    comp = labels[:, 0]
    cont = labels[:, 1]

    print(f"  Total genomes: {len(df)}")
    print(f"  Completeness:  mean={comp.mean():.1f}%, std={comp.std():.1f}%, "
          f"min={comp.min():.1f}%, max={comp.max():.1f}%")
    print(f"  Contamination: mean={cont.mean():.1f}%, std={cont.std():.1f}%, "
          f"min={cont.min():.1f}%, max={cont.max():.1f}%")

    # Check FASTA integrity
    n_missing = 0
    n_empty = 0
    total_bytes = 0
    for _, row in df.iterrows():
        fpath = fasta_dir / f"{row['genome_id']}.fasta"
        if not fpath.exists():
            n_missing += 1
        elif fpath.stat().st_size == 0:
            n_empty += 1
        else:
            total_bytes += fpath.stat().st_size

    print(f"  FASTA files: {len(df)} expected, {n_missing} missing, {n_empty} empty")
    print(f"  Total FASTA size: {total_bytes / (1024**3):.2f} GB")
    print(f"  FASTA integrity: {'PASS' if (n_missing == 0 and n_empty == 0) else 'FAIL'}")

    # Sample type distribution
    print(f"\n  Sample type distribution:")
    for st, count in df['sample_type'].value_counts().items():
        print(f"    {st}: {count}")

    # Phylum distribution
    print(f"\n  Unique dominant phyla: {df['dominant_phylum'].nunique()}")
    top5 = df['dominant_phylum'].value_counts().head(5)
    for phylum, count in top5.items():
        print(f"    {phylum}: {count}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Disk usage
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file())
    print(f"  Set C disk usage: {total_size / (1024**3):.2f} GB")


if __name__ == '__main__':
    main()
