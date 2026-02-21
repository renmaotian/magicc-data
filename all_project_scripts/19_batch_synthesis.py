#!/usr/bin/env python3
"""
Phase 4: Batch Genome Synthesis and Feature Extraction

Generates 1,000,000 synthetic genomes in 100 batches of 10,000:
  - Training:   80 batches (800,000 samples) from train_genomes.tsv
  - Validation: 10 batches (100,000 samples) from val_genomes.tsv
  - Test:       10 batches (100,000 samples) from test_genomes.tsv

Sample type ratios per batch:
  - 15% pure genomes (50-100% completeness, 0% contamination)
  - 15% complete genomes (100% completeness, 0-100% contamination)
  - 30% within-phylum contamination (50-100% comp, 0-100% contam, 1-3 genomes same phylum)
  - 30% cross-phylum contamination (50-100% comp, 0-100% contam, 1-5 genomes diff phyla)
  -  5% reduced genome organisms (Patescibacteria, DPANN, symbionts)
  -  5% archaeal genomes

Features: checkpoint-based resumability, multiprocessing (fork), memory-efficient genome loading.

Usage:
    conda activate magicc2
    python scripts/19_batch_synthesis.py [--workers N] [--start-batch B] [--end-batch E]
"""

import sys
import os

# Set NUMBA_NUM_THREADS=1 before importing numba to avoid thread contention
# in forked workers. Each worker process runs single-threaded for Numba.
os.environ['NUMBA_NUM_THREADS'] = '1'

import json
import time
import logging
import argparse
import traceback
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import h5py

# Project root
PROJECT_ROOT = '/home/tianrm/projects/magicc2'
sys.path.insert(0, PROJECT_ROOT)

from magicc.fragmentation import read_fasta, simulate_fragmentation
from magicc.contamination import generate_contaminated_sample, generate_pure_sample
from magicc.kmer_counter import (KmerCounter, load_selected_kmers, build_kmer_index,
                                  _count_kmers_single, K)
from magicc.assembly_stats import compute_assembly_stats, N_FEATURES as N_ASSEMBLY_FEATURES
from magicc.normalization import FeatureNormalizer
from magicc.storage import FeatureStore, METADATA_DTYPE

# ============================================================================
# Constants and Paths
# ============================================================================
KMER_PATH = os.path.join(PROJECT_ROOT, 'data/kmer_selection/selected_kmers.txt')
HDF5_PATH = os.path.join(PROJECT_ROOT, 'data/features/magicc_features.h5')
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'data/features/batch_checkpoint.json')
NORMALIZER_STATS_PATH = os.path.join(PROJECT_ROOT, 'data/features/normalizer_running_stats.json')
NORM_PARAMS_PATH = os.path.join(PROJECT_ROOT, 'data/features/normalization_params.json')

TRAIN_GENOMES_PATH = os.path.join(PROJECT_ROOT, 'data/splits/train_genomes.tsv')
VAL_GENOMES_PATH = os.path.join(PROJECT_ROOT, 'data/splits/val_genomes.tsv')
TEST_GENOMES_PATH = os.path.join(PROJECT_ROOT, 'data/splits/test_genomes.tsv')

BATCH_SIZE = 10_000
N_KMER_FEATURES = 9249
N_ASSEMBLY_FEATURES_CONST = 20

# Batch assignments: (batch_id, split_name, offset_in_split)
# Batches 0-79: train (80 batches x 10,000 = 800,000)
# Batches 80-89: val (10 batches x 10,000 = 100,000)
# Batches 90-99: test (10 batches x 10,000 = 100,000)
BATCH_ASSIGNMENTS = []
for i in range(80):
    BATCH_ASSIGNMENTS.append((i, 'train', i * BATCH_SIZE))
for i in range(10):
    BATCH_ASSIGNMENTS.append((80 + i, 'val', i * BATCH_SIZE))
for i in range(10):
    BATCH_ASSIGNMENTS.append((90 + i, 'test', i * BATCH_SIZE))

# Sample type counts per batch of 10,000
SAMPLE_TYPES = {
    'pure':              1500,   # 15%
    'complete':          1500,   # 15%
    'within_phylum':     3000,   # 30%
    'cross_phylum':      3000,   # 30%
    'reduced_genome':     500,   #  5%
    'archaeal':           500,   #  5%
}
assert sum(SAMPLE_TYPES.values()) == BATCH_SIZE

# Quality tier weights: more weight toward medium/low for realism
QUALITY_TIER_WEIGHTS = {
    'high': 0.15,
    'medium': 0.35,
    'low': 0.35,
    'highly_fragmented': 0.15,
}
QUALITY_TIERS = list(QUALITY_TIER_WEIGHTS.keys())
QUALITY_WEIGHTS = np.array([QUALITY_TIER_WEIGHTS[t] for t in QUALITY_TIERS])
QUALITY_WEIGHTS /= QUALITY_WEIGHTS.sum()

# Reduced genome phyla (Patescibacteria + DPANN archaea + candidate phyla)
REDUCED_GENOME_PHYLA = {
    'Patescibacteriota',
    # DPANN archaea
    'Aenigmatarchaeota', 'Altiarchaeota', 'Diapherotrites',
    'Huberarchaeota', 'Iainarchaeota', 'Micrarchaeota',
    'Nanoarchaeota', 'Nanohaloarchaeota', 'Nanohalarchaeota',
    'Undinarchaeota', 'Woesearchaeota', 'Nanobdellota',
    # Other candidate phyla / symbionts
    'Dependentiae', 'Bdellovibrionota',
}

# ============================================================================
# Logging
# ============================================================================
def setup_logging():
    """Configure logging to both file and stdout."""
    log_path = os.path.join(PROJECT_ROOT, 'data/features/batch_synthesis.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger('batch_synthesis')


# ============================================================================
# Genome Index (loaded once, shared read-only across forks)
# ============================================================================
class GenomeIndex:
    """
    Index of reference genomes organized by phylum for efficient lookup.

    Stores only metadata (accession, phylum, domain, fasta_path, genome_size)
    in memory. Genome sequences are loaded on demand.
    """

    def __init__(self, tsv_path: str):
        self.all_genomes = []       # List of dicts
        self.by_phylum = {}         # phylum -> list of indices
        self.archaea_indices = []   # Indices of archaeal genomes
        self.reduced_indices = []   # Indices of reduced genome organisms
        self.general_indices = []   # All indices (for dominant genome selection)
        self.phylum_list = []       # List of unique phyla

        self._load(tsv_path)

    def _load(self, tsv_path: str):
        """Load genome metadata from TSV."""
        with open(tsv_path) as f:
            header = f.readline().strip().split('\t')
            col_idx = {name: i for i, name in enumerate(header)}

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < len(header):
                    continue

                acc = parts[col_idx['ncbi_accession']]
                fasta = parts[col_idx['fasta_path']]
                phylum = parts[col_idx['phylum']]
                domain = parts[col_idx['domain']]
                genome_size = int(parts[col_idx['genome_size']])

                # Skip if fasta doesn't exist
                if not os.path.exists(fasta):
                    continue

                idx = len(self.all_genomes)
                self.all_genomes.append({
                    'accession': acc,
                    'fasta_path': fasta,
                    'phylum': phylum,
                    'domain': domain,
                    'genome_size': genome_size,
                })

                if phylum not in self.by_phylum:
                    self.by_phylum[phylum] = []
                self.by_phylum[phylum].append(idx)

                if domain == 'Archaea':
                    self.archaea_indices.append(idx)

                if phylum in REDUCED_GENOME_PHYLA:
                    self.reduced_indices.append(idx)

                self.general_indices.append(idx)

        self.phylum_list = list(self.by_phylum.keys())

    def get_genome(self, idx: int) -> Dict:
        """Get genome metadata by index."""
        return self.all_genomes[idx]


# ============================================================================
# Worker function for multiprocessing (fork mode)
# ============================================================================

# Module-level k-mer data, loaded once in parent, inherited by forked children
_kmer_codes = None
_kmer_index = None
_n_kmer_features = None


def init_kmer_globals():
    """Load k-mer data and warm up Numba JIT in the parent process."""
    global _kmer_codes, _kmer_index, _n_kmer_features

    _kmer_codes = load_selected_kmers(KMER_PATH)
    _kmer_index = build_kmer_index(_kmer_codes)
    _n_kmer_features = len(_kmer_codes)

    # Warm up Numba JIT compilation (runs once in parent, inherited by forks)
    dummy_seq = np.frombuffer(b'ACGTACGTACGTACGTACGT', dtype=np.uint8)
    _count_kmers_single(dummy_seq, _kmer_index, _n_kmer_features, K)

    # Also warm up assembly stats Numba functions
    from magicc.assembly_stats import _compute_gc_from_bytes, _compute_nx_lx, _compute_bimodality
    _compute_gc_from_bytes(dummy_seq)
    test_lengths = np.array([100, 200, 300], dtype=np.int64)
    _compute_nx_lx(test_lengths, 600, 0.5)
    test_gc = np.array([0.4, 0.5, 0.6, 0.3], dtype=np.float64)
    _compute_bimodality(test_gc)


def process_single_sample(args: Tuple) -> Optional[Dict]:
    """
    Process a single synthetic genome sample in a worker process.

    Args is a tuple of:
        (sample_id, sample_type, dominant_info, contaminant_infos,
         target_completeness, target_contamination, quality_tier, seed)

    dominant_info: dict with 'fasta_path', 'accession', 'phylum', 'genome_size'
    contaminant_infos: list of dict (same format) or None

    Returns dict with features and metadata, or None on failure.
    """
    global _kmer_index, _n_kmer_features

    (sample_id, sample_type, dominant_info, contaminant_infos,
     target_completeness, target_contamination, quality_tier, seed) = args

    rng = np.random.default_rng(seed)

    try:
        # Load dominant genome sequence
        dominant_seq = read_fasta(dominant_info['fasta_path'])
        if dominant_seq is None or len(dominant_seq) < 500:
            return None

        # Load contaminant sequences if needed
        contaminant_seqs = []
        if contaminant_infos:
            for ci in contaminant_infos:
                try:
                    cseq = read_fasta(ci['fasta_path'])
                    if cseq and len(cseq) >= 500:
                        contaminant_seqs.append(cseq)
                except Exception:
                    pass

        # Generate synthetic sample
        if sample_type == 'pure':
            sample = generate_pure_sample(
                dominant_sequence=dominant_seq,
                target_completeness=target_completeness,
                rng=rng,
                quality_tier=quality_tier,
            )
        else:
            sample = generate_contaminated_sample(
                dominant_sequence=dominant_seq,
                contaminant_sequences=contaminant_seqs if contaminant_seqs else [],
                target_completeness=target_completeness,
                target_contamination=target_contamination,
                rng=rng,
                dominant_quality_tier=quality_tier,
            )

        contigs = sample['contigs']
        if not contigs or len(contigs) == 0:
            return None

        actual_completeness = sample['completeness'] * 100.0  # to percentage
        actual_contamination = sample['contamination']
        dominant_full_length = sample['dominant_full_length']
        used_quality_tier = sample.get('dominant_quality_tier', quality_tier)

        # Extract k-mer features
        total_counts = np.zeros(_n_kmer_features, dtype=np.int64)
        for contig in contigs:
            if len(contig) >= K:
                seq_bytes = np.frombuffer(contig.encode('ascii'), dtype=np.uint8)
                total_counts += _count_kmers_single(
                    seq_bytes, _kmer_index, _n_kmer_features, K
                )

        # Log10 total kmer count
        total_sum = total_counts.sum()
        log10_total = np.log10(float(total_sum)) if total_sum > 0 else 0.0

        # Assembly statistics
        assembly_features = compute_assembly_stats(contigs, log10_total)

        n_contaminants = len(contaminant_seqs) if contaminant_seqs else 0

        return {
            'sample_id': sample_id,
            'kmer_counts': total_counts,
            'assembly_features': assembly_features,
            'completeness': actual_completeness,
            'contamination': actual_contamination,
            'dominant_phylum': dominant_info['phylum'],
            'sample_type': sample_type,
            'quality_tier': used_quality_tier,
            'dominant_accession': dominant_info['accession'],
            'dominant_full_length': dominant_full_length,
            'n_contaminants': n_contaminants,
        }

    except Exception:
        return None


# ============================================================================
# Batch Planner
# ============================================================================
class BatchPlanner:
    """
    Plans which genomes to use for each sample in a batch.

    Handles sample type ratios, genome selection, and contaminant pairing.
    """

    def __init__(self, genome_index: GenomeIndex, batch_id: int):
        self.gi = genome_index
        self.batch_id = batch_id
        self.rng = np.random.default_rng(batch_id * 10000)

    def plan_batch(self) -> List[Tuple]:
        """
        Plan all 10,000 samples for this batch.

        Returns list of tuples for process_single_sample.
        """
        samples = []
        sample_id = 0

        for sample_type, count in SAMPLE_TYPES.items():
            for i in range(count):
                plan = self._plan_single(sample_id, sample_type)
                if plan is not None:
                    samples.append(plan)
                sample_id += 1

        # Shuffle to mix sample types
        self.rng.shuffle(samples)

        # Reassign sample IDs after shuffle
        for i, s in enumerate(samples):
            samples[i] = (i,) + s[1:]

        return samples

    def _plan_single(self, sample_id: int, sample_type: str) -> Optional[Tuple]:
        """Plan a single sample."""
        seed = self.batch_id * 10000 + sample_id

        # Select quality tier
        quality_tier = self.rng.choice(QUALITY_TIERS, p=QUALITY_WEIGHTS)

        if sample_type == 'pure':
            return self._plan_pure(sample_id, seed, quality_tier)
        elif sample_type == 'complete':
            return self._plan_complete(sample_id, seed, quality_tier)
        elif sample_type == 'within_phylum':
            return self._plan_within_phylum(sample_id, seed, quality_tier)
        elif sample_type == 'cross_phylum':
            return self._plan_cross_phylum(sample_id, seed, quality_tier)
        elif sample_type == 'reduced_genome':
            return self._plan_reduced(sample_id, seed, quality_tier)
        elif sample_type == 'archaeal':
            return self._plan_archaeal(sample_id, seed, quality_tier)
        return None

    def _select_dominant(self, pool_indices: List[int]) -> Optional[Dict]:
        """Select a dominant genome from given pool."""
        if not pool_indices:
            return None
        idx = self.rng.choice(pool_indices)
        return self.gi.get_genome(idx)

    def _plan_pure(self, sample_id, seed, quality_tier):
        """Pure genome: 50-100% completeness, 0% contamination."""
        dom = self._select_dominant(self.gi.general_indices)
        if dom is None:
            return None
        completeness = float(self.rng.uniform(0.5, 1.0))
        return (sample_id, 'pure', dom, None, completeness, 0.0, quality_tier, seed)

    def _plan_complete(self, sample_id, seed, quality_tier):
        """Complete genome: 100% completeness, 0-100% contamination."""
        dom = self._select_dominant(self.gi.general_indices)
        if dom is None:
            return None
        contamination = float(self.rng.uniform(0.0, 100.0))

        # Need contaminant genomes (use cross-phylum for variety)
        contaminant_infos = self._select_contaminants_cross_phylum(
            dom['phylum'], int(self.rng.integers(1, 6)))
        return (sample_id, 'complete', dom, contaminant_infos,
                1.0, contamination, quality_tier, seed)

    def _plan_within_phylum(self, sample_id, seed, quality_tier):
        """Within-phylum contamination: 50-100% comp, 0-100% contam, 1-3 genomes same phylum."""
        dom = self._select_dominant(self.gi.general_indices)
        if dom is None:
            return None
        completeness = float(self.rng.uniform(0.5, 1.0))
        contamination = float(self.rng.uniform(0.0, 100.0))

        n_contaminants = int(self.rng.integers(1, 4))  # 1-3
        contaminant_infos = self._select_contaminants_within_phylum(
            dom['phylum'], n_contaminants, dom['accession'])

        # If not enough within-phylum, fall back to any available
        if not contaminant_infos:
            contaminant_infos = self._select_contaminants_cross_phylum(dom['phylum'], 1)

        return (sample_id, 'within_phylum', dom, contaminant_infos,
                completeness, contamination, quality_tier, seed)

    def _plan_cross_phylum(self, sample_id, seed, quality_tier):
        """Cross-phylum contamination: 50-100% comp, 0-100% contam, 1-5 genomes diff phyla."""
        dom = self._select_dominant(self.gi.general_indices)
        if dom is None:
            return None
        completeness = float(self.rng.uniform(0.5, 1.0))
        contamination = float(self.rng.uniform(0.0, 100.0))

        n_contaminants = int(self.rng.integers(1, 6))  # 1-5
        contaminant_infos = self._select_contaminants_cross_phylum(
            dom['phylum'], n_contaminants)

        return (sample_id, 'cross_phylum', dom, contaminant_infos,
                completeness, contamination, quality_tier, seed)

    def _plan_reduced(self, sample_id, seed, quality_tier):
        """Reduced genome: from Patescibacteria/DPANN/symbionts, 50-100% comp, 0-100% contam."""
        pool = self.gi.reduced_indices if self.gi.reduced_indices else self.gi.general_indices
        dom = self._select_dominant(pool)
        if dom is None:
            return None
        completeness = float(self.rng.uniform(0.5, 1.0))
        contamination = float(self.rng.uniform(0.0, 100.0))

        n_contaminants = int(self.rng.integers(1, 4))
        contaminant_infos = self._select_contaminants_cross_phylum(dom['phylum'], n_contaminants)

        return (sample_id, 'reduced_genome', dom, contaminant_infos,
                completeness, contamination, quality_tier, seed)

    def _plan_archaeal(self, sample_id, seed, quality_tier):
        """Archaeal genome: from archaea, 50-100% comp, 0-100% contam."""
        pool = self.gi.archaea_indices if self.gi.archaea_indices else self.gi.general_indices
        dom = self._select_dominant(pool)
        if dom is None:
            return None
        completeness = float(self.rng.uniform(0.5, 1.0))
        contamination = float(self.rng.uniform(0.0, 100.0))

        n_contaminants = int(self.rng.integers(1, 4))
        contaminant_infos = self._select_contaminants_cross_phylum(dom['phylum'], n_contaminants)

        return (sample_id, 'archaeal', dom, contaminant_infos,
                completeness, contamination, quality_tier, seed)

    def _select_contaminants_within_phylum(self, phylum: str, n: int,
                                            exclude_acc: str = '') -> List[Dict]:
        """Select n contaminant genomes from the same phylum."""
        pool = self.gi.by_phylum.get(phylum, [])
        # Filter out the dominant genome
        if exclude_acc:
            pool = [idx for idx in pool
                    if self.gi.all_genomes[idx]['accession'] != exclude_acc]
        if len(pool) < 1:
            return []
        n = min(n, len(pool))
        chosen = self.rng.choice(pool, size=n, replace=False)
        return [self.gi.get_genome(int(idx)) for idx in chosen]

    def _select_contaminants_cross_phylum(self, dominant_phylum: str, n: int) -> List[Dict]:
        """Select n contaminant genomes from different phyla than the dominant."""
        other_phyla = [p for p in self.gi.phylum_list if p != dominant_phylum]
        if not other_phyla:
            other_phyla = self.gi.phylum_list

        n_phyla = min(n, len(other_phyla))
        chosen_phyla = self.rng.choice(other_phyla, size=n_phyla, replace=False)

        contaminants = []
        for phylum in chosen_phyla:
            pool = self.gi.by_phylum.get(phylum, [])
            if pool:
                idx = self.rng.choice(pool)
                contaminants.append(self.gi.get_genome(int(idx)))

        return contaminants


# ============================================================================
# Checkpoint Management
# ============================================================================
class CheckpointManager:
    """Manages batch completion tracking for resumability."""

    def __init__(self, path: str):
        self.path = path
        self.completed_batches = set()
        self.batch_times = {}
        self.batch_errors = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.completed_batches = set(data.get('completed_batches', []))
            self.batch_times = data.get('batch_times', {})
            self.batch_errors = data.get('batch_errors', {})

    def _save(self):
        data = {
            'completed_batches': sorted(list(self.completed_batches)),
            'batch_times': self.batch_times,
            'batch_errors': self.batch_errors,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        tmp_path = self.path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def is_completed(self, batch_id: int) -> bool:
        return batch_id in self.completed_batches

    def mark_completed(self, batch_id: int, elapsed: float):
        self.completed_batches.add(batch_id)
        self.batch_times[str(batch_id)] = round(elapsed, 1)
        self._save()

    def mark_error(self, batch_id: int, error: str):
        self.batch_errors[str(batch_id)] = error
        self._save()


# ============================================================================
# Main Processing
# ============================================================================
def load_genome_index(split: str) -> GenomeIndex:
    """Load genome index for a given split."""
    paths = {
        'train': TRAIN_GENOMES_PATH,
        'val': VAL_GENOMES_PATH,
        'test': TEST_GENOMES_PATH,
    }
    return GenomeIndex(paths[split])


def process_batch(batch_id: int, split: str, offset: int,
                  genome_index: GenomeIndex, n_workers: int,
                  logger: logging.Logger) -> Dict[str, Any]:
    """
    Process a single batch of 10,000 synthetic genomes.

    Returns dict with batch results or raises on failure.
    """
    batch_start = time.perf_counter()

    # Plan the batch
    planner = BatchPlanner(genome_index, batch_id)
    sample_plans = planner.plan_batch()

    n_planned = len(sample_plans)
    logger.info(f"  Batch {batch_id}: Planned {n_planned} samples, "
                f"split={split}, offset={offset}")

    # Process samples in parallel using fork-based pool
    # The parent has already loaded k-mer globals and warmed up Numba,
    # so forked children inherit everything — no per-worker init needed.
    results = []
    n_failures = 0

    with mp.Pool(processes=n_workers) as pool:
        # Submit all work at once, using imap_unordered for better load balancing
        chunk_results = pool.imap_unordered(
            process_single_sample, sample_plans, chunksize=50
        )

        for r in chunk_results:
            if r is not None:
                results.append(r)
            else:
                n_failures += 1

            # Log progress every 1000 samples
            total_done = len(results) + n_failures
            if total_done % 2000 == 0:
                elapsed = time.perf_counter() - batch_start
                rate = total_done / elapsed if elapsed > 0 else 0
                logger.info(f"    Batch {batch_id}: {total_done}/{n_planned} processed, "
                            f"{len(results)} ok, {n_failures} fail, "
                            f"{rate:.0f} samples/sec")

    logger.info(f"    Batch {batch_id}: Main pass done: {len(results)} ok, "
                f"{n_failures} fail out of {n_planned}")

    # Handle failures: retry with replacement genomes
    max_retries = 3
    retry_count = 0
    while len(results) < BATCH_SIZE and retry_count < max_retries:
        retry_count += 1
        n_needed = BATCH_SIZE - len(results)
        logger.info(f"    Batch {batch_id}: Retry {retry_count}, need {n_needed} more samples")

        # Generate replacement plans (pure genomes for simplicity)
        replacement_rng = np.random.default_rng(batch_id * 10000 + 90000 + retry_count * 1000)
        replacement_plans = []
        for i in range(n_needed):
            sid = BATCH_SIZE + retry_count * n_needed + i
            seed = batch_id * 10000 + sid
            quality_tier = replacement_rng.choice(QUALITY_TIERS, p=QUALITY_WEIGHTS)
            dom_idx = replacement_rng.choice(genome_index.general_indices)
            dom = genome_index.get_genome(dom_idx)
            comp = float(replacement_rng.uniform(0.5, 1.0))
            replacement_plans.append(
                (sid, 'pure', dom, None, comp, 0.0, quality_tier, seed)
            )

        with mp.Pool(processes=n_workers) as pool:
            retry_results = pool.map(process_single_sample, replacement_plans, chunksize=50)

        for r in retry_results:
            if r is not None and len(results) < BATCH_SIZE:
                results.append(r)

    # Truncate to exactly BATCH_SIZE
    results = results[:BATCH_SIZE]

    if len(results) < BATCH_SIZE:
        logger.warning(f"    Batch {batch_id}: Only {len(results)}/{BATCH_SIZE} samples "
                       f"(shortfall of {BATCH_SIZE - len(results)})")

    # Assemble into arrays
    n_samples = len(results)
    kmer_counts = np.zeros((n_samples, N_KMER_FEATURES), dtype=np.int64)
    assembly_features = np.zeros((n_samples, N_ASSEMBLY_FEATURES_CONST), dtype=np.float64)
    labels = np.zeros((n_samples, 2), dtype=np.float32)
    metadata = np.zeros(n_samples, dtype=METADATA_DTYPE)

    for i, r in enumerate(results):
        kmer_counts[i] = r['kmer_counts']
        assembly_features[i] = r['assembly_features']
        labels[i, 0] = r['completeness']
        labels[i, 1] = r['contamination']
        metadata[i]['completeness'] = r['completeness']
        metadata[i]['contamination'] = r['contamination']
        metadata[i]['dominant_phylum'] = r['dominant_phylum'].encode('utf-8')[:64]
        metadata[i]['sample_type'] = r['sample_type'].encode('utf-8')[:32]
        metadata[i]['quality_tier'] = r['quality_tier'].encode('utf-8')[:20]
        metadata[i]['dominant_accession'] = r['dominant_accession'].encode('utf-8')[:30]
        metadata[i]['genome_full_length'] = r['dominant_full_length']
        metadata[i]['n_contaminants'] = r['n_contaminants']
        metadata[i]['batch_id'] = batch_id

    elapsed = time.perf_counter() - batch_start

    # Count sample types
    type_counts = {}
    for r in results:
        st = r['sample_type']
        type_counts[st] = type_counts.get(st, 0) + 1

    return {
        'batch_id': batch_id,
        'split': split,
        'offset': offset,
        'n_samples': n_samples,
        'n_failures': n_failures,
        'kmer_counts': kmer_counts,
        'assembly_features': assembly_features,
        'labels': labels,
        'metadata': metadata,
        'elapsed': elapsed,
        'type_counts': type_counts,
    }


def write_batch_to_hdf5(batch_result: Dict, logger: logging.Logger):
    """Write a batch of results to HDF5 storage (atomic write)."""
    split = batch_result['split']
    offset = batch_result['offset']

    with FeatureStore(HDF5_PATH, mode='a') as store:
        n_written = store.write_batch(
            split=split,
            kmer_features=batch_result['kmer_counts'].astype(np.float32),
            assembly_features=batch_result['assembly_features'].astype(np.float32),
            labels=batch_result['labels'],
            metadata=batch_result['metadata'],
            batch_offset=offset,
        )
        store._file.flush()

    logger.info(f"    Wrote {n_written} samples to {split} at offset {offset}")


def update_normalizer(normalizer: FeatureNormalizer, batch_result: Dict):
    """Update running normalization statistics with batch results."""
    normalizer.update_kmer_batch(batch_result['kmer_counts'].astype(np.float64))
    normalizer.update_assembly_batch(batch_result['assembly_features'])


def save_normalizer_checkpoint(normalizer: FeatureNormalizer):
    """Save normalizer running stats to disk for resumability."""
    stats = {
        'kmer_stats': {
            'count': int(normalizer.kmer_stats.count),
            'mean': normalizer.kmer_stats.mean.tolist(),
            'm2': normalizer.kmer_stats.m2.tolist(),
            'min_vals': normalizer.kmer_stats.min_vals.tolist(),
            'max_vals': normalizer.kmer_stats.max_vals.tolist(),
            'reservoir_count': int(normalizer.kmer_stats.reservoir_count),
        },
        'assembly_stats': {
            'count': int(normalizer.assembly_stats.count),
            'mean': normalizer.assembly_stats.mean.tolist(),
            'm2': normalizer.assembly_stats.m2.tolist(),
            'min_vals': normalizer.assembly_stats.min_vals.tolist(),
            'max_vals': normalizer.assembly_stats.max_vals.tolist(),
            'reservoir_count': int(normalizer.assembly_stats.reservoir_count),
        },
        'kmer_reservoir': normalizer.kmer_stats.reservoir[:normalizer.kmer_stats.reservoir_count].tolist()
            if normalizer.kmer_stats.reservoir_count <= 5000 else 'too_large',
        'assembly_reservoir': normalizer.assembly_stats.reservoir[:normalizer.assembly_stats.reservoir_count].tolist()
            if normalizer.assembly_stats.reservoir_count <= 5000 else 'too_large',
    }
    tmp_path = NORMALIZER_STATS_PATH + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(stats, f)
    os.replace(tmp_path, NORMALIZER_STATS_PATH)


def load_normalizer_checkpoint(normalizer: FeatureNormalizer) -> bool:
    """Load normalizer running stats from checkpoint. Returns True if loaded."""
    if not os.path.exists(NORMALIZER_STATS_PATH):
        return False

    try:
        with open(NORMALIZER_STATS_PATH) as f:
            stats = json.load(f)

        ks = stats['kmer_stats']
        normalizer.kmer_stats.count = ks['count']
        normalizer.kmer_stats.mean = np.array(ks['mean'], dtype=np.float64)
        normalizer.kmer_stats.m2 = np.array(ks['m2'], dtype=np.float64)
        normalizer.kmer_stats.min_vals = np.array(ks['min_vals'], dtype=np.float64)
        normalizer.kmer_stats.max_vals = np.array(ks['max_vals'], dtype=np.float64)
        normalizer.kmer_stats.reservoir_count = ks['reservoir_count']

        asm = stats['assembly_stats']
        normalizer.assembly_stats.count = asm['count']
        normalizer.assembly_stats.mean = np.array(asm['mean'], dtype=np.float64)
        normalizer.assembly_stats.m2 = np.array(asm['m2'], dtype=np.float64)
        normalizer.assembly_stats.min_vals = np.array(asm['min_vals'], dtype=np.float64)
        normalizer.assembly_stats.max_vals = np.array(asm['max_vals'], dtype=np.float64)
        normalizer.assembly_stats.reservoir_count = asm['reservoir_count']

        # Restore reservoirs if available
        if stats.get('kmer_reservoir') != 'too_large' and stats.get('kmer_reservoir'):
            reservoir_data = np.array(stats['kmer_reservoir'], dtype=np.float64)
            n = min(len(reservoir_data), normalizer.kmer_stats.reservoir_size)
            normalizer.kmer_stats.reservoir[:n] = reservoir_data[:n]
            normalizer.kmer_stats.reservoir_count = n

        if stats.get('assembly_reservoir') != 'too_large' and stats.get('assembly_reservoir'):
            reservoir_data = np.array(stats['assembly_reservoir'], dtype=np.float64)
            n = min(len(reservoir_data), normalizer.assembly_stats.reservoir_size)
            normalizer.assembly_stats.reservoir[:n] = reservoir_data[:n]
            normalizer.assembly_stats.reservoir_count = n

        return True
    except Exception:
        return False


def finalize_normalization(normalizer: FeatureNormalizer, logger: logging.Logger):
    """Finalize normalization and apply to all stored features."""
    logger.info("Finalizing normalization parameters...")
    normalizer.finalize()
    normalizer.save(NORM_PARAMS_PATH)
    logger.info(f"  Saved normalization params to {NORM_PARAMS_PATH}")

    # Apply normalization to all stored data in HDF5
    logger.info("Applying normalization to all stored features...")

    for split, total_samples in [('train', 800_000), ('val', 100_000), ('test', 100_000)]:
        logger.info(f"  Normalizing {split} ({total_samples:,} samples)...")

        # Process in chunks to limit memory
        chunk_size = 10_000
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)

            with FeatureStore(HDF5_PATH, mode='a') as store:
                kmer, asm, labels, meta = store.read_batch(split, start, end)

                # Normalize k-mer features: raw counts -> log(count+1) -> z-score
                kmer_norm = normalizer.normalize_kmer(kmer).astype(np.float32)

                # Normalize assembly features
                asm_norm = normalizer.normalize_assembly(asm).astype(np.float32)

                # Write back normalized features
                store._file[split]['kmer_features'][start:end] = kmer_norm
                store._file[split]['assembly_features'][start:end] = asm_norm
                store._file.flush()

        logger.info(f"    {split} normalization complete")

    logger.info("All normalization applied successfully")


def print_summary(checkpoint: CheckpointManager, normalizer: FeatureNormalizer,
                  total_start_time: float, logger: logging.Logger):
    """Print a comprehensive summary of the batch processing."""
    total_elapsed = time.perf_counter() - total_start_time

    n_completed = len(checkpoint.completed_batches)
    n_errors = len(checkpoint.batch_errors)

    # Compute timing stats
    batch_times = [v for v in checkpoint.batch_times.values()]
    if batch_times:
        avg_time = sum(batch_times) / len(batch_times)
        min_time = min(batch_times)
        max_time = max(batch_times)
    else:
        avg_time = min_time = max_time = 0

    logger.info("=" * 70)
    logger.info("BATCH SYNTHESIS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Completed batches: {n_completed}/100")
    logger.info(f"  Total samples:     {n_completed * BATCH_SIZE:,}")
    logger.info(f"  Error batches:     {n_errors}")
    logger.info(f"  Total time:        {total_elapsed/3600:.1f} hours")
    logger.info(f"  Avg batch time:    {avg_time:.1f} seconds")
    logger.info(f"  Min/Max batch:     {min_time:.1f}s / {max_time:.1f}s")

    if normalizer.kmer_stats.count > 0:
        logger.info(f"  Normalizer stats:")
        logger.info(f"    K-mer samples:     {normalizer.kmer_stats.count:,}")
        logger.info(f"    Assembly samples:  {normalizer.assembly_stats.count:,}")
        kmer_mean_of_means = normalizer.kmer_stats.mean.mean()
        kmer_std_of_stds = np.sqrt(normalizer.kmer_stats.variance).mean()
        logger.info(f"    K-mer log mean (avg across features): {kmer_mean_of_means:.4f}")
        logger.info(f"    K-mer log std (avg across features):  {kmer_std_of_stds:.4f}")

    if n_errors > 0:
        logger.info(f"  Errors:")
        for bid, err in checkpoint.batch_errors.items():
            logger.info(f"    Batch {bid}: {err}")

    # Verify HDF5 contents
    try:
        with FeatureStore(HDF5_PATH, mode='r') as store:
            for split in ['train', 'val', 'test']:
                info = store.get_split_info(split)
                logger.info(f"  HDF5 {split}: {info['n_written']:,}/{info['n_total']:,} written")
    except Exception as e:
        logger.warning(f"  Could not read HDF5 info: {e}")

    logger.info("=" * 70)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Phase 4: Batch Genome Synthesis')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: 90%% of CPUs)')
    parser.add_argument('--start-batch', type=int, default=0,
                        help='First batch to process (default: 0, uses checkpoint)')
    parser.add_argument('--end-batch', type=int, default=100,
                        help='Last batch (exclusive) to process (default: 100)')
    parser.add_argument('--finalize-only', action='store_true',
                        help='Skip batch processing, only finalize normalization')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("Phase 4: Batch Genome Synthesis and Feature Extraction")
    logger.info("=" * 70)

    # Determine worker count
    n_cpus = mp.cpu_count()
    n_workers = args.workers if args.workers else max(1, int(n_cpus * 0.90))
    logger.info(f"System: {n_cpus} CPUs, using {n_workers} workers")

    # Load k-mer data and warm up Numba JIT in parent process BEFORE forking
    logger.info("Loading k-mer index and warming up Numba JIT...")
    jit_start = time.perf_counter()
    init_kmer_globals()
    jit_time = time.perf_counter() - jit_start
    logger.info(f"  K-mer index loaded ({_n_kmer_features} features), "
                f"Numba JIT warmed up in {jit_time:.1f}s")

    # Initialize checkpoint
    checkpoint = CheckpointManager(CHECKPOINT_PATH)
    logger.info(f"Checkpoint: {len(checkpoint.completed_batches)} batches already completed")

    # Initialize normalizer
    normalizer = FeatureNormalizer(n_kmer_features=N_KMER_FEATURES, reservoir_size=50000)
    if load_normalizer_checkpoint(normalizer):
        logger.info(f"Loaded normalizer checkpoint: {normalizer.kmer_stats.count:,} samples")

    total_start = time.perf_counter()

    if not args.finalize_only:
        # Load genome indices for each split (only load when needed)
        genome_indices = {}

        # Process batches
        for batch_id, split, offset in BATCH_ASSIGNMENTS:
            if batch_id < args.start_batch or batch_id >= args.end_batch:
                continue

            if checkpoint.is_completed(batch_id):
                logger.info(f"Batch {batch_id} ({split}): Already completed, skipping")
                continue

            # Load genome index for this split if not already loaded
            if split not in genome_indices:
                logger.info(f"Loading genome index for {split}...")
                gi_start = time.perf_counter()
                genome_indices[split] = load_genome_index(split)
                gi_time = time.perf_counter() - gi_start
                gi = genome_indices[split]
                logger.info(f"  Loaded {len(gi.all_genomes):,} genomes, "
                            f"{len(gi.phylum_list)} phyla, "
                            f"{len(gi.archaea_indices)} archaea, "
                            f"{len(gi.reduced_indices)} reduced, "
                            f"in {gi_time:.1f}s")

            genome_index = genome_indices[split]

            # Estimate remaining time
            n_remaining = sum(1 for bid, _, _ in BATCH_ASSIGNMENTS
                             if bid >= batch_id and bid < args.end_batch
                             and not checkpoint.is_completed(bid))
            avg_time = (sum(float(v) for v in checkpoint.batch_times.values()) /
                       len(checkpoint.batch_times)) if checkpoint.batch_times else 0
            eta_sec = avg_time * n_remaining

            logger.info(f"\nBatch {batch_id}/99 ({split}, offset={offset}) "
                        f"[{n_remaining} remaining, ETA ~{eta_sec/60:.0f} min]")

            try:
                # Process the batch
                batch_result = process_batch(
                    batch_id=batch_id,
                    split=split,
                    offset=offset,
                    genome_index=genome_index,
                    n_workers=n_workers,
                    logger=logger,
                )

                # Write to HDF5
                write_batch_to_hdf5(batch_result, logger)

                # Update normalizer statistics
                update_normalizer(normalizer, batch_result)

                # Mark as completed
                checkpoint.mark_completed(batch_id, batch_result['elapsed'])

                # Save normalizer checkpoint every 5 batches
                if (batch_id + 1) % 5 == 0:
                    save_normalizer_checkpoint(normalizer)
                    logger.info(f"    Saved normalizer checkpoint")

                # Log batch summary
                elapsed = batch_result['elapsed']
                n_samples = batch_result['n_samples']
                rate = n_samples / elapsed if elapsed > 0 else 0
                logger.info(f"    Batch {batch_id} COMPLETE: {n_samples} samples, "
                            f"{elapsed:.1f}s ({rate:.0f} samples/sec)")
                logger.info(f"    Type counts: {batch_result['type_counts']}")
                logger.info(f"    Comp range: [{batch_result['labels'][:,0].min():.1f}, "
                            f"{batch_result['labels'][:,0].max():.1f}]%, "
                            f"mean={batch_result['labels'][:,0].mean():.1f}%")
                logger.info(f"    Contam range: [{batch_result['labels'][:,1].min():.1f}, "
                            f"{batch_result['labels'][:,1].max():.1f}]%, "
                            f"mean={batch_result['labels'][:,1].mean():.1f}%")

                # Free memory
                del batch_result

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Batch {batch_id} FAILED: {e}\n{tb}")
                checkpoint.mark_error(batch_id, str(e))
                continue

        # Save final normalizer checkpoint
        save_normalizer_checkpoint(normalizer)
        logger.info("Saved final normalizer checkpoint")

    # Finalize normalization if all batches are complete
    all_complete = all(checkpoint.is_completed(bid) for bid, _, _ in BATCH_ASSIGNMENTS)
    if all_complete or args.finalize_only:
        if args.finalize_only and normalizer.kmer_stats.count == 0:
            # Need to rebuild normalizer from stored raw features
            logger.info("Rebuilding normalizer from stored raw features...")
            normalizer = FeatureNormalizer(n_kmer_features=N_KMER_FEATURES, reservoir_size=50000)
            for split, total_samples in [('train', 800_000), ('val', 100_000), ('test', 100_000)]:
                logger.info(f"  Reading {split} raw features...")
                chunk_size = 10_000
                for start in range(0, total_samples, chunk_size):
                    end = min(start + chunk_size, total_samples)
                    with FeatureStore(HDF5_PATH, mode='r') as store:
                        kmer, asm, _, _ = store.read_batch(split, start, end)
                    normalizer.update_kmer_batch(kmer.astype(np.float64))
                    normalizer.update_assembly_batch(asm.astype(np.float64))

        finalize_normalization(normalizer, logger)
    else:
        n_done = len(checkpoint.completed_batches)
        logger.info(f"Not all batches complete ({n_done}/100), "
                    f"skipping normalization finalization")

    print_summary(checkpoint, normalizer, total_start, logger)


if __name__ == '__main__':
    # Use fork mode (Linux default) for fast worker startup.
    # Parent process loads k-mer data and compiles Numba, children inherit it.
    mp.set_start_method('fork', force=True)
    main()
