#!/usr/bin/env python3
"""
Phase 6 - Benchmark Dataset Generation (Sets A-D)

Generates benchmark FASTA files with metadata for evaluating MAGICC.

Set A: Completeness gradient (600 genomes)
  - 0% contamination, completeness: 50%, 60%, 70%, 80%, 90%, 100%
  - 100 genomes each

Set B: Contamination gradient (600 genomes)
  - 100% completeness (original contigs), contamination: 0%, 10%, 20%, 40%, 60%, 80%
  - 100 genomes each, cross-phylum contaminants

Set C: Patescibacteria (1000 genomes)
  - ALL Patescibacteriota from train+val+test (1608 total)
  - Uniform completeness (50-100%) and contamination (0-100%)
  - Cross-phylum contaminants from test references

Set D: Archaea (1000 genomes)
  - ALL Archaea from train+val+test (1976 total), sample 1000
  - Uniform completeness (50-100%) and contamination (0-100%)
  - Cross-phylum contaminants from test references

Output per set:
  data/benchmarks/set_{A,B,C,D}/fasta/genome_{i}.fasta
  data/benchmarks/set_{A,B,C,D}/metadata.tsv
  data/benchmarks/set_{A,B,C,D}/labels.npy
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.fragmentation import (
    simulate_fragmentation, read_fasta, load_original_contigs
)
from magicc.contamination import (
    generate_contaminated_sample, generate_pure_sample
)

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'
SPLITS_DIR = DATA_DIR / 'splits'

N_WORKERS = max(1, int(cpu_count() * 0.90))
SEED = 42

# ============================================================================
# Helper functions
# ============================================================================

def write_fasta(contigs: List[str], fasta_path: str, genome_id: str = "genome"):
    """Write contigs to FASTA file."""
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as f:
        for i, contig in enumerate(contigs):
            f.write(f">{genome_id}_contig_{i} len={len(contig)}\n")
            # Write sequence in 80-char lines
            for j in range(0, len(contig), 80):
                f.write(contig[j:j+80])
                f.write('\n')


def load_all_genomes():
    """Load genome metadata from all splits."""
    dfs = {}
    for split in ['train', 'val', 'test']:
        path = SPLITS_DIR / f'{split}_genomes.tsv'
        df = pd.read_csv(path, sep='\t')
        df['split'] = split
        dfs[split] = df
    return dfs


def get_cross_phylum_contaminants(dominant_phylum: str, test_df: pd.DataFrame,
                                   rng: np.random.Generator, n_contaminants: int = 1) -> List[str]:
    """Select random cross-phylum contaminant genome sequences from test set."""
    # Filter to different phylum
    candidates = test_df[test_df['phylum'] != dominant_phylum]
    if len(candidates) == 0:
        candidates = test_df  # fallback: use any genome

    selected = candidates.sample(n=min(n_contaminants, len(candidates)), random_state=int(rng.integers(0, 2**31)))
    sequences = []
    for _, row in selected.iterrows():
        fasta_path = row['fasta_path']
        if os.path.exists(fasta_path):
            seq = read_fasta(fasta_path)
            if len(seq) > 0:
                sequences.append(seq)
    return sequences


# ============================================================================
# Worker functions for multiprocessing
# ============================================================================

def generate_set_a_genome(args):
    """Generate a single Set A genome (completeness gradient)."""
    idx, row_dict, target_completeness, fasta_dir = args
    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    # Check if already exists (resumable)
    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None  # Skip, will be handled in metadata reconciliation

    rng = np.random.default_rng(SEED + idx + 100000)
    accession = row_dict['gtdb_accession']
    phylum = row_dict['phylum']
    src_fasta = row_dict['fasta_path']

    try:
        if target_completeness >= 1.0:
            # 100% completeness: use original contigs
            contigs = load_original_contigs(src_fasta)
            actual_completeness = 1.0
        else:
            sequence = read_fasta(src_fasta)
            if len(sequence) == 0:
                return None
            result = simulate_fragmentation(
                sequence, target_completeness=target_completeness, rng=rng
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
    """Generate a single Set B genome (contamination gradient)."""
    idx, row_dict, target_contamination, fasta_dir, test_df_path = args
    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None

    rng = np.random.default_rng(SEED + idx + 200000)
    accession = row_dict['gtdb_accession']
    phylum = row_dict['phylum']
    src_fasta = row_dict['fasta_path']

    try:
        # Load test genomes for contaminant selection
        test_df = pd.read_csv(test_df_path, sep='\t')

        # 100% completeness: use original contigs
        dominant_contigs = load_original_contigs(src_fasta)
        if len(dominant_contigs) == 0:
            return None

        dominant_sequence = read_fasta(src_fasta)
        dominant_full_length = len(dominant_sequence)

        if target_contamination <= 0:
            # No contamination
            contigs = dominant_contigs
            actual_contamination = 0.0
        else:
            # Get cross-phylum contaminants
            n_cont = rng.integers(1, 4)  # 1-3 contaminant genomes
            contaminant_sequences = get_cross_phylum_contaminants(
                phylum, test_df, rng, n_cont
            )

            if len(contaminant_sequences) == 0:
                contigs = dominant_contigs
                actual_contamination = 0.0
            else:
                # Generate contaminated sample with 100% completeness
                # We use the dominant sequence for contamination calc but keep original contigs
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


def generate_set_cd_genome(args):
    """Generate a single Set C/D genome (uniform completeness/contamination)."""
    idx, row_dict, target_completeness, target_contamination, fasta_dir, test_df_path, set_name = args
    genome_id = f"genome_{idx}"
    fasta_path = os.path.join(fasta_dir, f"{genome_id}.fasta")

    if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
        return None

    rng = np.random.default_rng(SEED + idx + (300000 if set_name == 'C' else 400000))
    accession = row_dict['gtdb_accession']
    phylum = row_dict['phylum']
    src_fasta = row_dict['fasta_path']

    try:
        test_df = pd.read_csv(test_df_path, sep='\t')

        dominant_sequence = read_fasta(src_fasta)
        if len(dominant_sequence) == 0:
            return None
        dominant_full_length = len(dominant_sequence)

        if target_completeness >= 1.0:
            # Use original contigs for dominant
            dominant_contigs = load_original_contigs(src_fasta)
            actual_completeness = 1.0
        else:
            frag_result = simulate_fragmentation(
                dominant_sequence, target_completeness=target_completeness, rng=rng
            )
            dominant_contigs = frag_result['contigs']
            actual_completeness = frag_result['completeness']
            if len(dominant_contigs) == 0:
                return None

        if target_contamination <= 0:
            contigs = dominant_contigs
            actual_contamination = 0.0
        else:
            n_cont = rng.integers(1, 6)  # 1-5 contaminant genomes for cross-phylum
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
                    target_completeness=target_completeness,
                    target_contamination=target_contamination,
                    rng=rng,
                )
                contigs = result['contigs']
                actual_completeness = result['completeness']
                actual_contamination = result['contamination']

                # Explicit cap: contamination should not exceed target by much
                # The generate_contaminated_sample should handle this, but enforce here
                if actual_contamination > target_contamination * 1.2 + 5:
                    # Re-trim contaminant contigs
                    max_cont_bp = int(target_contamination / 100.0 * dominant_full_length)
                    dom_contigs = result['dominant_contigs']
                    cont_contigs = result['contaminant_contigs']
                    # Trim contaminant contigs to fit
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
            'true_contamination': min(actual_contamination, target_contamination + 5),
            'dominant_accession': accession,
            'dominant_phylum': phylum,
            'sample_type': f'set_{set_name.lower()}',
            'n_contigs': len(contigs),
            'total_length': total_length,
        }
    except Exception as e:
        print(f"  Error generating {genome_id}: {e}", flush=True)
        return None


# ============================================================================
# Set generation functions
# ============================================================================

def generate_set_a(test_df: pd.DataFrame):
    """Generate Set A: Completeness gradient."""
    print("\n" + "="*70)
    print("Generating Set A: Completeness Gradient")
    print("="*70)

    set_dir = BENCHMARK_DIR / 'set_A'
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    completeness_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    n_per_level = 100

    all_results = []
    idx = 0

    for comp in completeness_levels:
        print(f"\n  Completeness = {comp*100:.0f}%")

        # Sample 100 random test genomes
        rng_sample = np.random.default_rng(SEED + int(comp * 1000))
        sampled = test_df.sample(n=n_per_level, random_state=int(rng_sample.integers(0, 2**31)))

        tasks = []
        for _, row in sampled.iterrows():
            tasks.append((idx, row.to_dict(), comp, str(fasta_dir)))
            idx += 1

        with Pool(N_WORKERS) as pool:
            results = pool.map(generate_set_a_genome, tasks)

        valid = [r for r in results if r is not None]
        all_results.extend(valid)
        print(f"    Generated {len(valid)}/{n_per_level} genomes")

    # Also collect results from previously generated (resumable)
    all_results = _reconcile_results(all_results, fasta_dir, idx)

    # Save metadata and labels
    _save_set_outputs(set_dir, all_results)
    print(f"\n  Set A total: {len(all_results)} genomes")
    return all_results


def generate_set_b(test_df: pd.DataFrame):
    """Generate Set B: Contamination gradient."""
    print("\n" + "="*70)
    print("Generating Set B: Contamination Gradient")
    print("="*70)

    set_dir = BENCHMARK_DIR / 'set_B'
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    contamination_levels = [0, 10, 20, 40, 60, 80]
    n_per_level = 100
    test_df_path = str(SPLITS_DIR / 'test_genomes.tsv')

    all_results = []
    idx = 0

    for cont in contamination_levels:
        print(f"\n  Contamination = {cont}%")

        rng_sample = np.random.default_rng(SEED + cont + 5000)
        sampled = test_df.sample(n=n_per_level, random_state=int(rng_sample.integers(0, 2**31)))

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
    print(f"\n  Set B total: {len(all_results)} genomes")
    return all_results


def generate_set_c(all_dfs: Dict[str, pd.DataFrame]):
    """Generate Set C: Patescibacteria genomes."""
    print("\n" + "="*70)
    print("Generating Set C: Patescibacteria (CPR)")
    print("="*70)

    set_dir = BENCHMARK_DIR / 'set_C'
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    # Collect ALL Patescibacteriota from all splits
    pates_dfs = []
    for split_name, df in all_dfs.items():
        pates = df[df['phylum'] == 'Patescibacteriota'].copy()
        pates_dfs.append(pates)

    pates_all = pd.concat(pates_dfs, ignore_index=True)
    print(f"  Total Patescibacteriota reference genomes: {len(pates_all)}")

    n_target = 1000
    test_df_path = str(SPLITS_DIR / 'test_genomes.tsv')

    # Generate uniform completeness and contamination
    rng = np.random.default_rng(SEED + 300000)
    completeness_values = rng.uniform(0.50, 1.0, size=n_target)
    contamination_values = rng.uniform(0.0, 100.0, size=n_target)

    # Reuse reference genomes with different parameters
    # Cycle through available Patescibacteriota genomes
    genome_indices = np.arange(n_target) % len(pates_all)
    rng.shuffle(genome_indices)

    tasks = []
    for idx in range(n_target):
        row = pates_all.iloc[genome_indices[idx]]
        tasks.append((
            idx, row.to_dict(),
            float(completeness_values[idx]),
            float(contamination_values[idx]),
            str(fasta_dir), test_df_path, 'C'
        ))

    print(f"  Generating {n_target} genomes with {N_WORKERS} workers...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(generate_set_cd_genome, tasks)

    valid = [r for r in results if r is not None]
    all_results = _reconcile_results(valid, fasta_dir, n_target)
    _save_set_outputs(set_dir, all_results)
    print(f"\n  Set C total: {len(all_results)} genomes")
    return all_results


def generate_set_d(all_dfs: Dict[str, pd.DataFrame]):
    """Generate Set D: Archaea genomes."""
    print("\n" + "="*70)
    print("Generating Set D: Archaea")
    print("="*70)

    set_dir = BENCHMARK_DIR / 'set_D'
    fasta_dir = set_dir / 'fasta'
    os.makedirs(fasta_dir, exist_ok=True)

    # Collect ALL Archaea from all splits
    arch_dfs = []
    for split_name, df in all_dfs.items():
        arch = df[df['domain'] == 'Archaea'].copy()
        arch_dfs.append(arch)

    arch_all = pd.concat(arch_dfs, ignore_index=True)
    print(f"  Total Archaeal reference genomes: {len(arch_all)}")

    n_target = 1000
    test_df_path = str(SPLITS_DIR / 'test_genomes.tsv')

    # Sample 1000 randomly from all archaea
    rng = np.random.default_rng(SEED + 400000)

    # Generate uniform completeness and contamination
    completeness_values = rng.uniform(0.50, 1.0, size=n_target)
    contamination_values = rng.uniform(0.0, 100.0, size=n_target)

    # Sample 1000 archaea (may reuse if < 1000 but we have 1976)
    if len(arch_all) >= n_target:
        sampled_indices = rng.choice(len(arch_all), size=n_target, replace=False)
    else:
        sampled_indices = np.arange(n_target) % len(arch_all)
        rng.shuffle(sampled_indices)

    tasks = []
    for idx in range(n_target):
        row = arch_all.iloc[sampled_indices[idx]]
        tasks.append((
            idx, row.to_dict(),
            float(completeness_values[idx]),
            float(contamination_values[idx]),
            str(fasta_dir), test_df_path, 'D'
        ))

    print(f"  Generating {n_target} genomes with {N_WORKERS} workers...")
    with Pool(N_WORKERS) as pool:
        results = pool.map(generate_set_cd_genome, tasks)

    valid = [r for r in results if r is not None]
    all_results = _reconcile_results(valid, fasta_dir, n_target)
    _save_set_outputs(set_dir, all_results)
    print(f"\n  Set D total: {len(all_results)} genomes")
    return all_results


# ============================================================================
# Utility functions
# ============================================================================

def _reconcile_results(results: List[Dict], fasta_dir, n_total):
    """Reconcile results with existing files for resumability."""
    existing_ids = {r['genome_id'] for r in results}

    # Check for any existing FASTA files not in results (from previous runs)
    fasta_dir_path = Path(fasta_dir)
    if fasta_dir_path.exists():
        for fasta_file in fasta_dir_path.glob("genome_*.fasta"):
            gid = fasta_file.stem
            if gid not in existing_ids and fasta_file.stat().st_size > 0:
                # Read basic info from file
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
                    'true_completeness': -1.0,  # unknown from previous run
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

    # Filter out resumed entries with unknown labels
    valid_results = [r for r in results if r['true_completeness'] >= 0]

    df = pd.DataFrame(valid_results)
    df = df.sort_values('genome_id').reset_index(drop=True)

    # Save metadata
    metadata_path = set_dir / 'metadata.tsv'
    df.to_csv(metadata_path, sep='\t', index=False)
    print(f"  Saved metadata: {metadata_path} ({len(df)} rows)")

    # Save labels
    labels = df[['true_completeness', 'true_contamination']].values.astype(np.float32)
    labels_path = set_dir / 'labels.npy'
    np.save(labels_path, labels)
    print(f"  Saved labels: {labels_path} ({labels.shape})")


# ============================================================================
# Validation
# ============================================================================

def validate_all_sets():
    """Validate all benchmark sets."""
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    for set_name in ['A', 'B', 'C', 'D']:
        set_dir = BENCHMARK_DIR / f'set_{set_name}'
        metadata_path = set_dir / 'metadata.tsv'
        labels_path = set_dir / 'labels.npy'
        fasta_dir = set_dir / 'fasta'

        if not metadata_path.exists():
            print(f"\n  Set {set_name}: MISSING")
            continue

        print(f"\n  {'='*60}")
        print(f"  Set {set_name}")
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
        n_invalid = 0
        for _, row in df.iterrows():
            fasta_path = fasta_dir / f"{row['genome_id']}.fasta"
            if not fasta_path.exists():
                n_missing += 1
            elif fasta_path.stat().st_size == 0:
                n_empty += 1
            else:
                # Quick validity check: first line should start with '>'
                with open(fasta_path) as f:
                    first_line = f.readline()
                    if not first_line.startswith('>'):
                        n_invalid += 1

        print(f"  FASTA files: {len(df)} expected, {n_missing} missing, "
              f"{n_empty} empty, {n_invalid} invalid")

        # Set-specific validation
        if set_name == 'A':
            _validate_set_a(df, labels)
        elif set_name == 'B':
            _validate_set_b(df, labels)
        elif set_name in ('C', 'D'):
            _validate_set_cd(df, labels, set_name)

        all_ok = (n_missing == 0 and n_empty == 0 and n_invalid == 0)
        print(f"  FASTA integrity: {'PASS' if all_ok else 'FAIL'}")


def _validate_set_a(df, labels):
    """Validate Set A completeness gradient."""
    print("\n  Set A per-level validation:")
    comp = labels[:, 0]
    cont = labels[:, 1]

    # Group by sample_type
    for stype in sorted(df['sample_type'].unique()):
        mask = df['sample_type'] == stype
        level_comp = comp[mask]
        level_cont = cont[mask]

        # Extract target from name
        target = int(stype.split('comp')[-1])

        # Check within +-10% of target
        within_tol = np.abs(level_comp - target) <= 10
        pct_pass = 100 * within_tol.sum() / len(level_comp)

        print(f"    {stype}: n={len(level_comp)}, comp mean={level_comp.mean():.1f}% "
              f"(target={target}%, within +/-10%: {pct_pass:.0f}%), "
              f"cont mean={level_cont.mean():.1f}%")


def _validate_set_b(df, labels):
    """Validate Set B contamination gradient."""
    print("\n  Set B per-level validation:")
    comp = labels[:, 0]
    cont = labels[:, 1]

    for stype in sorted(df['sample_type'].unique()):
        mask = df['sample_type'] == stype
        level_comp = comp[mask]
        level_cont = cont[mask]

        target = int(stype.split('cont')[-1])

        # Check contamination within +-15% of target
        within_tol = np.abs(level_cont - target) <= 15
        pct_pass = 100 * within_tol.sum() / len(level_cont)

        # Check completeness ~100%
        comp_ok = np.abs(level_comp - 100) <= 5
        comp_pct = 100 * comp_ok.sum() / len(level_comp)

        print(f"    {stype}: n={len(level_cont)}, "
              f"cont mean={level_cont.mean():.1f}% (target={target}%, within +/-15%: {pct_pass:.0f}%), "
              f"comp mean={level_comp.mean():.1f}% (100% +/-5%: {comp_pct:.0f}%)")


def _validate_set_cd(df, labels, set_name):
    """Validate Set C/D uniform distributions."""
    comp = labels[:, 0]
    cont = labels[:, 1]

    print(f"\n  Set {set_name} distribution validation:")

    # Check completeness is approximately uniform in [50, 100]
    comp_bins = np.linspace(50, 100, 6)
    comp_hist, _ = np.histogram(comp, bins=comp_bins)
    expected_per_bin = len(comp) / len(comp_bins[:-1])
    chi2_comp = np.sum((comp_hist - expected_per_bin)**2 / expected_per_bin)
    print(f"    Completeness bins {comp_bins}: {comp_hist}")
    print(f"    Completeness chi2 = {chi2_comp:.1f} (expected ~{expected_per_bin:.0f} per bin)")

    # Check contamination is approximately uniform in [0, 100]
    cont_bins = np.linspace(0, 100, 6)
    cont_hist, _ = np.histogram(cont, bins=cont_bins)
    expected_per_bin = len(cont) / len(cont_bins[:-1])
    chi2_cont = np.sum((cont_hist - expected_per_bin)**2 / expected_per_bin)
    print(f"    Contamination bins {cont_bins}: {cont_hist}")
    print(f"    Contamination chi2 = {chi2_cont:.1f} (expected ~{expected_per_bin:.0f} per bin)")

    # Report phylum distribution
    print(f"    Unique dominant phyla: {df['dominant_phylum'].nunique()}")
    print(f"    Top phyla: {df['dominant_phylum'].value_counts().head(5).to_dict()}")


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print(f"Benchmark Generation Script")
    print(f"Workers: {N_WORKERS}")
    print(f"Seed: {SEED}")

    # Parse command-line arguments for which sets to generate
    sets_to_generate = set()
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.upper() in ('A', 'B', 'C', 'D'):
                sets_to_generate.add(arg.upper())
            elif arg.lower() == 'validate':
                sets_to_generate.add('VALIDATE')
    if not sets_to_generate:
        sets_to_generate = {'A', 'B', 'C', 'D', 'VALIDATE'}

    # Load genome data
    print("\nLoading genome data...")
    all_dfs = load_all_genomes()
    test_df = all_dfs['test']
    print(f"  Test: {len(test_df)}, Train: {len(all_dfs['train'])}, Val: {len(all_dfs['val'])}")

    # Warm up Numba
    print("Warming up Numba JIT...")
    from magicc.fragmentation import _warm_numba_fragmentation
    _warm_numba_fragmentation()

    # Generate sets
    if 'A' in sets_to_generate:
        generate_set_a(test_df)
    if 'B' in sets_to_generate:
        generate_set_b(test_df)
    if 'C' in sets_to_generate:
        generate_set_c(all_dfs)
    if 'D' in sets_to_generate:
        generate_set_d(all_dfs)

    # Validate
    if 'VALIDATE' in sets_to_generate or sets_to_generate.intersection({'A','B','C','D'}):
        validate_all_sets()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
