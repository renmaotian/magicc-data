#!/usr/bin/env python3
"""
Generate 5 Benchmark Datasets Using Finished/Complete Genomes as Dominant References

Generates:
  Motivating Set A (seed=100): 1000 genomes, 0% contamination, 6 completeness levels
  Motivating Set B (seed=200): 1000 genomes, 100% completeness, 5 contamination levels
  Benchmark Set A  (seed=300): same structure as Motivating Set A, different genomes
  Benchmark Set B  (seed=400): same structure as Motivating Set B, different genomes
  Benchmark Set E  (seed=500): 1000 genomes with uniform completeness/contamination

KEY: Dominant genomes from finished genomes only (Complete Genome / Chromosome).
     Contaminant genomes from ALL test reference genomes.

Output locations:
  data/benchmarks/motivating_v2/set_A/
  data/benchmarks/motivating_v2/set_B/
  data/benchmarks/set_A_v2/
  data/benchmarks/set_B_v2/
  data/benchmarks/set_E/
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
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

N_WORKERS = max(1, int(cpu_count() * 0.90))  # ~43 workers

# Paths to genome lists
FINISHED_GENOMES_PATH = SPLITS_DIR / 'test_finished_genomes.tsv'
ALL_TEST_GENOMES_PATH = SPLITS_DIR / 'test_genomes.tsv'


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
    """Select random within-phylum contaminant genome sequences (different genome, same phylum)."""
    candidates = test_df[(test_df['phylum'] == dominant_phylum) &
                         (test_df['gtdb_accession'] != dominant_accession)]
    if len(candidates) == 0:
        # Fallback to cross-phylum if no same-phylum available
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


def _save_set_outputs(set_dir: Path, results: List[Dict], extra_columns: List[str] = None):
    """Save metadata.tsv and labels.npy for a benchmark set."""
    if len(results) == 0:
        print("  WARNING: No genomes generated!")
        return

    valid_results = [r for r in results if r is not None and r.get('true_completeness', -1) >= 0]
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
# Worker functions for Set A-type (completeness gradient, 0% contamination)
# ============================================================================

def generate_set_a_genome(args):
    """Generate a single genome for Set A-type (completeness gradient, 0% contamination)."""
    idx, row_dict, target_completeness, fasta_dir, seed = args
    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None  # Skip existing

    rng = np.random.default_rng(seed + idx)
    accession = row_dict['gtdb_accession']
    phylum = row_dict['phylum']
    src_fasta = row_dict['fasta_path']

    try:
        if target_completeness >= 1.0:
            contigs = load_original_contigs(src_fasta)
            actual_completeness = 1.0
        else:
            sequence = read_fasta(src_fasta)
            if len(sequence) == 0:
                return None
            result = simulate_fragmentation(
                sequence,
                target_completeness=target_completeness,
                quality_tier='medium',
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
            'target_completeness': target_completeness * 100.0,
        }
    except Exception as e:
        print(f"  Error generating {genome_id}: {e}", flush=True)
        return None


# ============================================================================
# Worker functions for Set B-type (contamination gradient, 100% completeness)
# ============================================================================

def generate_set_b_genome(args):
    """Generate a single genome for Set B-type (contamination gradient, 100% completeness)."""
    idx, row_dict, target_contamination, fasta_dir, all_test_df_path, seed = args
    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None

    rng = np.random.default_rng(seed + idx)
    accession = row_dict['gtdb_accession']
    phylum = row_dict['phylum']
    src_fasta = row_dict['fasta_path']

    try:
        test_df = pd.read_csv(all_test_df_path, sep='\t')

        dominant_contigs = load_original_contigs(src_fasta)
        if len(dominant_contigs) == 0:
            return None

        dominant_sequence = read_fasta(src_fasta)
        dominant_full_length = len(dominant_sequence)

        if target_contamination <= 0:
            contigs = dominant_contigs
            actual_contamination = 0.0
        else:
            n_cont = rng.integers(1, 6)  # 1-5 contaminant genomes
            contaminant_sequences = get_cross_phylum_contaminants(
                phylum, test_df, rng, n_cont
            )

            if len(contaminant_sequences) == 0:
                contigs = dominant_contigs
                actual_contamination = 0.0
            else:
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
            'target_contamination': target_contamination,
        }
    except Exception as e:
        print(f"  Error generating {genome_id}: {e}", flush=True)
        return None


# ============================================================================
# Worker function for Set E (uniform completeness + contamination)
# ============================================================================

def generate_set_e_genome(args):
    """Generate a single genome for Set E (uniform completeness & contamination)."""
    (idx, row_dict, target_completeness, target_contamination,
     n_target_contigs, min_contig_bp, contamination_type,
     fasta_dir, all_test_df_path, seed) = args
    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None

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
            # Use medium quality tier for fragmentation
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

                # Cap contamination
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
            'sample_type': f'set_e_{contamination_type}',
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
# Set generation functions
# ============================================================================

def generate_completeness_set(finished_df: pd.DataFrame, set_name: str, output_dir: Path, seed: int):
    """Generate a completeness gradient set (Set A type).

    1000 genomes, 0% contamination, 6 completeness levels:
    50%, 60%, 70%, 80%, 90%, 100%
    Allocations: 167, 167, 167, 167, 167, 165 = 1000
    """
    print(f"\n{'='*70}")
    print(f"Generating {set_name}: Completeness Gradient (seed={seed})")
    print(f"{'='*70}")

    set_dir = output_dir
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    completeness_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    n_per_level = [167, 167, 167, 167, 167, 165]  # = 1000

    all_results = []
    idx = 0

    for comp, n_genomes in zip(completeness_levels, n_per_level):
        print(f"\n  Completeness = {comp*100:.0f}% ({n_genomes} genomes)")

        rng_sample = np.random.default_rng(seed + int(comp * 1000))
        sampled = finished_df.sample(n=n_genomes,
                                      random_state=int(rng_sample.integers(0, 2**31)),
                                      replace=(n_genomes > len(finished_df)))

        tasks = []
        for _, row in sampled.iterrows():
            tasks.append((idx, row.to_dict(), comp, str(fasta_dir), seed))
            idx += 1

        with Pool(N_WORKERS) as pool:
            results = pool.map(generate_set_a_genome, tasks)

        valid = [r for r in results if r is not None]
        all_results.extend(valid)
        print(f"    Generated {len(valid)}/{n_genomes} genomes")

    _save_set_outputs(set_dir, all_results)
    print(f"\n  {set_name} total: {len(all_results)} genomes")
    return all_results


def generate_contamination_set(finished_df: pd.DataFrame, all_test_df_path: str,
                                set_name: str, output_dir: Path, seed: int):
    """Generate a contamination gradient set (Set B type).

    1000 genomes, 100% completeness (original contigs), 5 contamination levels:
    0%, 20%, 40%, 60%, 80%
    200 genomes per level = 1000 total
    Cross-phylum contamination
    """
    print(f"\n{'='*70}")
    print(f"Generating {set_name}: Contamination Gradient (seed={seed})")
    print(f"{'='*70}")

    set_dir = output_dir
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    contamination_levels = [0, 20, 40, 60, 80]
    n_per_level = 200

    all_results = []
    idx = 0

    for cont in contamination_levels:
        print(f"\n  Contamination = {cont}% ({n_per_level} genomes)")

        rng_sample = np.random.default_rng(seed + cont + 5000)
        sampled = finished_df.sample(n=n_per_level,
                                      random_state=int(rng_sample.integers(0, 2**31)),
                                      replace=(n_per_level > len(finished_df)))

        tasks = []
        for _, row in sampled.iterrows():
            tasks.append((idx, row.to_dict(), float(cont), str(fasta_dir),
                          all_test_df_path, seed))
            idx += 1

        with Pool(N_WORKERS) as pool:
            results = pool.map(generate_set_b_genome, tasks)

        valid = [r for r in results if r is not None]
        all_results.extend(valid)
        print(f"    Generated {len(valid)}/{n_per_level} genomes")

    _save_set_outputs(set_dir, all_results)
    print(f"\n  {set_name} total: {len(all_results)} genomes")
    return all_results


def generate_set_e(finished_df: pd.DataFrame, all_test_df_path: str,
                    output_dir: Path, seed: int):
    """Generate Set E: 1000 genomes with uniform completeness and contamination.

    Composition:
    - 200 pure genomes: 0% contamination, 50-100% completeness (uniform)
    - 200 complete genomes: 100% completeness (original contigs), 0-100% contamination (uniform)
    - 600 others: mixed contaminated genomes, 50-100% completeness, 0-100% contamination
      - 70% cross-phylum, 30% within-phylum contamination
    - Fragmentation: 20-200 contigs, >1kbp min contig (for non-complete genomes)
    """
    print(f"\n{'='*70}")
    print(f"Generating Set E: Mixed Completeness & Contamination (seed={seed})")
    print(f"{'='*70}")

    set_dir = output_dir
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    n_total = 1000
    n_pure = 200
    n_complete = 200
    n_other = n_total - n_pure - n_complete  # 600

    rng = np.random.default_rng(seed)

    # Build per-genome parameters for 3 categories
    completeness_values = np.zeros(n_total)
    contamination_values = np.zeros(n_total)
    contamination_types = [''] * n_total

    # Category 1: Pure (200) - 50-100% completeness, 0% contamination
    completeness_values[:n_pure] = rng.uniform(0.50, 1.0, size=n_pure)
    contamination_values[:n_pure] = 0.0
    for i in range(n_pure):
        contamination_types[i] = 'pure'

    # Category 2: Complete (200) - 100% completeness, 0-100% contamination
    completeness_values[n_pure:n_pure+n_complete] = 1.0
    contamination_values[n_pure:n_pure+n_complete] = rng.uniform(0.0, 100.0, size=n_complete)
    for i in range(n_pure, n_pure + n_complete):
        if contamination_values[i] <= 0:
            contamination_types[i] = 'complete_pure'
        elif rng.random() < 0.7:
            contamination_types[i] = 'complete_cross_phylum'
        else:
            contamination_types[i] = 'complete_within_phylum'

    # Category 3: Other (600) - 50-100% completeness, 0-100% contamination
    completeness_values[n_pure+n_complete:] = rng.uniform(0.50, 1.0, size=n_other)
    contamination_values[n_pure+n_complete:] = rng.uniform(0.0, 100.0, size=n_other)
    for i in range(n_pure + n_complete, n_total):
        if rng.random() < 0.7:
            contamination_types[i] = 'cross_phylum'
        else:
            contamination_types[i] = 'within_phylum'

    # Shuffle so categories are mixed in
    shuffle_idx = rng.permutation(n_total)
    completeness_values = completeness_values[shuffle_idx]
    contamination_values = contamination_values[shuffle_idx]
    contamination_types = [contamination_types[i] for i in shuffle_idx]

    # Fragmentation: 20-200 contigs, >1kbp min contig (only used for non-complete genomes)
    n_target_contigs_values = rng.integers(20, 201, size=n_total)
    min_contig_bp = 1000  # >1kbp

    # Sample finished genomes (with replacement if needed)
    genome_indices = rng.choice(len(finished_df), size=n_total, replace=True)

    tasks = []
    for idx in range(n_total):
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
            seed,
        ))

    n_pure_actual = sum(1 for ct in contamination_types if ct == 'pure')
    n_complete_actual = sum(1 for ct in contamination_types if ct.startswith('complete'))
    n_cross = sum(1 for ct in contamination_types if ct == 'cross_phylum')
    n_within = sum(1 for ct in contamination_types if ct == 'within_phylum')

    print(f"  Generating {n_total} genomes with {N_WORKERS} workers...")
    print(f"  Pure (0% contamination, 50-100% comp): {n_pure_actual}")
    print(f"  Complete (100% completeness, 0-100% cont): {n_complete_actual}")
    print(f"  Cross-phylum contaminated: {n_cross}")
    print(f"  Within-phylum contaminated: {n_within}")

    with Pool(N_WORKERS) as pool:
        results = pool.map(generate_set_e_genome, tasks)

    valid = [r for r in results if r is not None]
    _save_set_outputs(set_dir, valid)
    print(f"\n  Set E total: {len(valid)} genomes")
    return valid


# ============================================================================
# Validation
# ============================================================================

def validate_set(set_dir: Path, set_name: str, expect_zero_cont: bool = False,
                  expect_100_comp: bool = False):
    """Validate a benchmark set."""
    metadata_path = set_dir / 'metadata.tsv'
    labels_path = set_dir / 'labels.npy'
    fasta_dir = set_dir / 'fasta'

    if not metadata_path.exists():
        print(f"  {set_name}: MISSING")
        return

    print(f"\n  {'='*60}")
    print(f"  {set_name}")
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

    # Check FASTA integrity
    n_missing = 0
    n_empty = 0
    total_bytes = 0
    for _, row in df.iterrows():
        fasta_path = fasta_dir / f"{row['genome_id']}.fasta"
        if not fasta_path.exists():
            n_missing += 1
        elif fasta_path.stat().st_size == 0:
            n_empty += 1
        else:
            total_bytes += fasta_path.stat().st_size

    print(f"  FASTA files: {len(df)} expected, {n_missing} missing, {n_empty} empty")
    print(f"  Total FASTA size: {total_bytes / (1024**3):.2f} GB")

    # Set-specific checks
    if expect_zero_cont:
        all_zero = np.all(cont == 0.0)
        print(f"  All contamination = 0%: {'PASS' if all_zero else 'FAIL'} "
              f"(max={cont.max():.2f}%)")

    if expect_100_comp:
        all_100 = np.all(np.abs(comp - 100.0) <= 0.5)
        print(f"  All completeness = 100%: {'PASS' if all_100 else 'FAIL'} "
              f"(min={comp.min():.1f}%, max={comp.max():.1f}%)")

    # Per-level stats (if applicable)
    if 'target_completeness' in df.columns:
        print("\n  Per-level completeness:")
        for target in sorted(df['target_completeness'].unique()):
            mask = df['target_completeness'] == target
            level_comp = comp[mask]
            level_cont = cont[mask]
            print(f"    Target {target:.0f}%: n={mask.sum()}, "
                  f"actual mean={level_comp.mean():.1f}%, std={level_comp.std():.1f}%, "
                  f"min={level_comp.min():.1f}%, max={level_comp.max():.1f}%")

    if 'target_contamination' in df.columns:
        print("\n  Per-level contamination:")
        for target in sorted(df['target_contamination'].unique()):
            mask = df['target_contamination'] == target
            level_comp = comp[mask]
            level_cont = cont[mask]
            print(f"    Target {target:.0f}%: n={mask.sum()}, "
                  f"actual mean={level_cont.mean():.1f}%, std={level_cont.std():.1f}%, "
                  f"min={level_cont.min():.1f}%, max={level_cont.max():.1f}%")

    # Phylum distribution
    print(f"\n  Unique dominant phyla: {df['dominant_phylum'].nunique()}")
    top5 = df['dominant_phylum'].value_counts().head(5)
    for phylum, count in top5.items():
        print(f"    {phylum}: {count}")

    all_ok = (n_missing == 0 and n_empty == 0)
    print(f"\n  FASTA integrity: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print("Finished Genome Benchmark Generation Script")
    print(f"Workers: {N_WORKERS}")

    # Parse command-line arguments
    sets_to_generate = set()
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            arg_lower = arg.lower()
            if arg_lower in ('motiv_a', 'motiv_b', 'bench_a', 'bench_b', 'set_e', 'validate'):
                sets_to_generate.add(arg_lower)
    if not sets_to_generate:
        sets_to_generate = {'motiv_a', 'motiv_b', 'bench_a', 'bench_b', 'set_e', 'validate'}

    # Load genome data
    print("\nLoading genome data...")
    finished_df = pd.read_csv(FINISHED_GENOMES_PATH, sep='\t')
    print(f"  Finished test genomes: {len(finished_df)} (from {finished_df['phylum'].nunique()} phyla)")
    all_test_df_path = str(ALL_TEST_GENOMES_PATH)

    # Warm up Numba
    print("Warming up Numba JIT...")
    _warm_numba_fragmentation()

    # Set output directories
    motiv_a_dir = BENCHMARK_DIR / 'motivating_v2' / 'set_A'
    motiv_b_dir = BENCHMARK_DIR / 'motivating_v2' / 'set_B'
    bench_a_dir = BENCHMARK_DIR / 'set_A_v2'
    bench_b_dir = BENCHMARK_DIR / 'set_B_v2'
    set_e_dir   = BENCHMARK_DIR / 'set_E'

    # Generate sets
    if 'motiv_a' in sets_to_generate:
        generate_completeness_set(finished_df, "Motivating Set A", motiv_a_dir, seed=100)

    if 'motiv_b' in sets_to_generate:
        generate_contamination_set(finished_df, all_test_df_path,
                                    "Motivating Set B", motiv_b_dir, seed=200)

    if 'bench_a' in sets_to_generate:
        generate_completeness_set(finished_df, "Benchmark Set A_v2", bench_a_dir, seed=300)

    if 'bench_b' in sets_to_generate:
        generate_contamination_set(finished_df, all_test_df_path,
                                    "Benchmark Set B_v2", bench_b_dir, seed=400)

    if 'set_e' in sets_to_generate:
        generate_set_e(finished_df, all_test_df_path, set_e_dir, seed=500)

    # Validate
    if 'validate' in sets_to_generate or len(sets_to_generate - {'validate'}) > 0:
        print(f"\n{'='*70}")
        print("VALIDATION")
        print(f"{'='*70}")

        validate_set(motiv_a_dir, "Motivating Set A (v2)", expect_zero_cont=True)
        validate_set(motiv_b_dir, "Motivating Set B (v2)", expect_100_comp=True)
        validate_set(bench_a_dir, "Benchmark Set A (v2)", expect_zero_cont=True)
        validate_set(bench_b_dir, "Benchmark Set B (v2)", expect_100_comp=True)
        validate_set(set_e_dir, "Set E")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Disk usage
    for name, d in [("motivating_v2/set_A", motiv_a_dir),
                     ("motivating_v2/set_B", motiv_b_dir),
                     ("set_A_v2", bench_a_dir),
                     ("set_B_v2", bench_b_dir),
                     ("set_E", set_e_dir)]:
        if d.exists():
            total = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            print(f"  {name}: {total / (1024**3):.2f} GB")


if __name__ == '__main__':
    main()
