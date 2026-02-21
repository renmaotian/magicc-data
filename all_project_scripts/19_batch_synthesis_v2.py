#!/usr/bin/env python3
"""
Phase 4: Batch Genome Synthesis and Feature Extraction (OPTIMIZED v2)

Optimizations:
1. Per-batch page cache warming: Pre-read needed genomes in parallel threads
   to ensure they're in OS page cache, then workers read them instantly.
2. Persistent mp.Pool across all batches in the same split (no fork overhead
   per batch).
3. Larger chunksize for imap_unordered to reduce IPC overhead.
4. Workers do their own read_fasta from page-cached files (fast) to avoid
   pickling large genome strings.
5. Numba JIT warmed up once in parent, inherited by forks.
6. Plan batches using indices, not dicts.

Usage:
    conda activate magicc2
    python scripts/19_batch_synthesis_v2.py [--workers N] [--start-batch B] [--end-batch E]
    python scripts/19_batch_synthesis_v2.py --test-batch 4   # test single batch
"""

import sys
import os

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import gc
import json
import time
import logging
import argparse
import traceback
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import h5py

PROJECT_ROOT = '/home/tianrm/projects/magicc2'
sys.path.insert(0, PROJECT_ROOT)

from magicc.fragmentation import (read_fasta, simulate_fragmentation,
                                   _warm_numba_fragmentation, load_original_contigs)
from magicc.contamination import generate_contaminated_sample, generate_pure_sample
from magicc.kmer_counter import (load_selected_kmers, build_kmer_index,
                                  _count_kmers_single, K)
from magicc.assembly_stats import compute_assembly_stats, N_FEATURES as N_ASSEMBLY_FEATURES
from magicc.normalization import FeatureNormalizer
from magicc.storage import FeatureStore, METADATA_DTYPE

# ============================================================================
# Constants
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
N_ASSEMBLY_FEATURES_CONST = 26

BATCH_ASSIGNMENTS = []
for i in range(80):
    BATCH_ASSIGNMENTS.append((i, 'train', i * BATCH_SIZE))
for i in range(10):
    BATCH_ASSIGNMENTS.append((80 + i, 'val', i * BATCH_SIZE))
for i in range(10):
    BATCH_ASSIGNMENTS.append((90 + i, 'test', i * BATCH_SIZE))

SAMPLE_TYPES = {
    'pure':              1500,
    'complete':          1500,
    'within_phylum':     3000,
    'cross_phylum':      3000,
    'reduced_genome':     500,
    'archaeal':           500,
}
assert sum(SAMPLE_TYPES.values()) == BATCH_SIZE

QUALITY_TIER_WEIGHTS = {
    'high': 0.15,
    'medium': 0.35,
    'low': 0.35,
    'highly_fragmented': 0.15,
}
QUALITY_TIERS = list(QUALITY_TIER_WEIGHTS.keys())
QUALITY_WEIGHTS = np.array([QUALITY_TIER_WEIGHTS[t] for t in QUALITY_TIERS])
QUALITY_WEIGHTS /= QUALITY_WEIGHTS.sum()

REDUCED_GENOME_PHYLA = {
    'Patescibacteriota',
    'Aenigmatarchaeota', 'Altiarchaeota', 'Diapherotrites',
    'Huberarchaeota', 'Iainarchaeota', 'Micrarchaeota',
    'Nanoarchaeota', 'Nanohaloarchaeota', 'Nanohalarchaeota',
    'Undinarchaeota', 'Woesearchaeota', 'Nanobdellota',
    'Dependentiae', 'Bdellovibrionota',
}

# ============================================================================
# Logging
# ============================================================================
def setup_logging(log_path=None):
    if log_path is None:
        log_path = os.path.join(PROJECT_ROOT, 'data/features/batch_synthesis_v3.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger('batch_synthesis_v3')


# ============================================================================
# Genome Index
# ============================================================================
class GenomeIndex:
    """Index of reference genomes. Metadata only (paths, accessions, phyla)."""

    def __init__(self, tsv_path: str):
        self.all_genomes = []
        self.by_phylum = {}
        self.archaea_indices = []
        self.reduced_indices = []
        self.general_indices = []
        self.phylum_list = []
        self._load(tsv_path)

    def _load(self, tsv_path):
        _by_phylum_lists = {}
        _archaea = []
        _reduced = []
        _general = []
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
                if phylum not in _by_phylum_lists:
                    _by_phylum_lists[phylum] = []
                _by_phylum_lists[phylum].append(idx)
                if domain == 'Archaea':
                    _archaea.append(idx)
                if phylum in REDUCED_GENOME_PHYLA:
                    _reduced.append(idx)
                _general.append(idx)
        # Convert to numpy arrays for fast rng.choice
        self.by_phylum = {k: np.array(v, dtype=np.int64) for k, v in _by_phylum_lists.items()}
        self.archaea_indices = np.array(_archaea, dtype=np.int64)
        self.reduced_indices = np.array(_reduced, dtype=np.int64)
        self.general_indices = np.array(_general, dtype=np.int64)
        self.phylum_list = list(self.by_phylum.keys())

    def get_genome(self, idx):
        return self.all_genomes[idx]


# ============================================================================
# Module-level globals (inherited by fork)
# ============================================================================
_kmer_index = None
_n_kmer_features = None
_genome_index_ref = None   # GenomeIndex reference (metadata)


def init_kmer_globals():
    global _kmer_index, _n_kmer_features
    kmer_codes = load_selected_kmers(KMER_PATH)
    _kmer_index = build_kmer_index(kmer_codes)
    _n_kmer_features = len(kmer_codes)

    # Warm up Numba JIT - k-mer and assembly
    dummy_seq = np.frombuffer(b'ACGTACGTACGTACGTACGT', dtype=np.uint8)
    _count_kmers_single(dummy_seq, _kmer_index, _n_kmer_features, K)

    from magicc.assembly_stats import _compute_gc_from_bytes, _compute_nx_lx, _compute_bimodality
    _compute_gc_from_bytes(dummy_seq)
    test_lengths = np.array([100, 200, 300], dtype=np.int64)
    _compute_nx_lx(test_lengths, 600, 0.5)
    test_gc = np.array([0.4, 0.5, 0.6, 0.3], dtype=np.float64)
    _compute_bimodality(test_gc)

    # Warm up fragmentation Numba JIT
    _warm_numba_fragmentation()


# ============================================================================
# Page Cache Warming
# ============================================================================
def warm_page_cache(fasta_paths, n_threads=44, logger=None):
    """Read genome files to warm the OS page cache, using parallel threads."""
    start = time.perf_counter()

    def read_file(path):
        try:
            with open(path, 'rb') as f:
                f.read()
            return True
        except Exception:
            return False

    n_ok = 0
    n_fail = 0
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for result in executor.map(read_file, fasta_paths):
            if result:
                n_ok += 1
            else:
                n_fail += 1

    elapsed = time.perf_counter() - start
    if logger:
        logger.info(f"    Page cache warmed: {n_ok} files in {elapsed:.1f}s "
                    f"({n_fail} failed)")
    return elapsed


# ============================================================================
# Worker function
# ============================================================================
def process_single_sample(args):
    """Process a single synthetic genome sample."""
    global _kmer_index, _n_kmer_features, _genome_index_ref

    (sample_id, sample_type, dominant_idx, contaminant_indices,
     target_completeness, target_contamination, quality_tier, seed) = args

    rng = np.random.default_rng(seed)

    try:
        dominant_info = _genome_index_ref.all_genomes[dominant_idx]

        # Read contaminant sequences
        contaminant_seqs = []
        if contaminant_indices:
            for ci in contaminant_indices:
                ci_info = _genome_index_ref.all_genomes[ci]
                try:
                    cseq = read_fasta(ci_info['fasta_path'])
                    if cseq and len(cseq) >= 500:
                        contaminant_seqs.append(cseq)
                except Exception:
                    pass

        # Generate synthetic sample
        if sample_type == 'complete':
            # FIX B: For "complete" type, use original contigs from the FASTA
            # without any fragmentation - the reference genome's natural
            # contig structure. Then add contamination on top.
            original_contigs = load_original_contigs(dominant_info['fasta_path'])
            if not original_contigs:
                return None
            dominant_full_length = sum(len(c) for c in original_contigs)
            if dominant_full_length < 500:
                return None

            # Add contamination if any
            if contaminant_seqs and target_contamination > 0:
                from magicc.contamination import (
                    select_contaminant_target_bp, fragment_contaminant,
                    compute_contamination_rate
                )
                total_target_bp = select_contaminant_target_bp(
                    target_contamination, dominant_full_length
                )
                n_contaminants = len(contaminant_seqs)
                if n_contaminants == 1:
                    bp_per_contaminant = [total_target_bp]
                else:
                    proportions = rng.dirichlet(np.ones(n_contaminants))
                    bp_per_contaminant = (proportions * total_target_bp).astype(int)
                    bp_per_contaminant[-1] = total_target_bp - bp_per_contaminant[:-1].sum()

                contaminant_contigs = []
                for seq, tbp in zip(contaminant_seqs, bp_per_contaminant):
                    if tbp <= 0 or len(seq) == 0:
                        continue
                    cr = fragment_contaminant(seq, tbp, rng)
                    contaminant_contigs.extend(cr['contigs'])

                # Cap contaminant bp
                contaminant_total_bp = sum(len(c) for c in contaminant_contigs)
                max_allowed_bp = int(target_contamination / 100.0 * dominant_full_length)
                if contaminant_total_bp > max_allowed_bp and len(contaminant_contigs) > 0:
                    cc_indices = np.arange(len(contaminant_contigs))
                    rng.shuffle(cc_indices)
                    kept_contigs = []
                    kept_bp = 0
                    for idx in cc_indices:
                        c = contaminant_contigs[idx]
                        if kept_bp + len(c) <= max_allowed_bp:
                            kept_contigs.append(c)
                            kept_bp += len(c)
                        else:
                            remaining = max_allowed_bp - kept_bp
                            if remaining >= 500:
                                kept_contigs.append(c[:remaining])
                                kept_bp += remaining
                            break
                    if len(kept_contigs) > 0:
                        contaminant_contigs = kept_contigs

                contaminant_total_bp = sum(len(c) for c in contaminant_contigs)
                actual_contamination = compute_contamination_rate(
                    contaminant_total_bp, dominant_full_length
                )

                # Merge and shuffle
                contigs = original_contigs + contaminant_contigs
                indices = np.arange(len(contigs))
                rng.shuffle(indices)
                contigs = [contigs[i] for i in indices]
            else:
                contigs = original_contigs
                actual_contamination = 0.0

            actual_completeness = 100.0  # complete genome = 100%
            used_quality_tier = 'high'  # original contigs are high quality

        else:
            # Non-complete types: read concatenated sequence, apply fragmentation
            dominant_seq = read_fasta(dominant_info['fasta_path'])
            if dominant_seq is None or len(dominant_seq) < 500:
                return None
            dominant_full_length = len(dominant_seq)

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

            actual_completeness = sample['completeness'] * 100.0
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

        total_sum = total_counts.sum()
        log10_total = np.log10(float(total_sum)) if total_sum > 0 else 0.0

        # Pass kmer_counts to compute_assembly_stats for new k-mer summary features
        assembly_features = compute_assembly_stats(contigs, log10_total, total_counts)

        n_contaminants = len(contaminant_seqs) if contaminant_seqs else 0

        return (
            sample_id,
            total_counts,
            assembly_features,
            actual_completeness,
            actual_contamination,
            dominant_info['phylum'],
            sample_type,
            used_quality_tier,
            dominant_info['accession'],
            dominant_full_length,
            n_contaminants,
        )

    except Exception:
        return None


# ============================================================================
# Batch Planning
# ============================================================================
class BatchPlanner:
    def __init__(self, genome_index, batch_id):
        self.gi = genome_index
        self.batch_id = batch_id
        self.rng = np.random.default_rng(batch_id * 10000)

    def plan_batch(self):
        samples = []
        needed_indices = set()
        sample_id = 0
        for sample_type, count in SAMPLE_TYPES.items():
            for i in range(count):
                plan = self._plan_single(sample_id, sample_type)
                if plan is not None:
                    samples.append(plan)
                    needed_indices.add(plan[2])
                    if plan[3]:
                        needed_indices.update(plan[3])
                sample_id += 1
        self.rng.shuffle(samples)
        for i, s in enumerate(samples):
            samples[i] = (i,) + s[1:]
        return samples, needed_indices

    def _plan_single(self, sample_id, sample_type):
        seed = self.batch_id * 10000 + sample_id
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

    def _select_dominant_idx(self, pool):
        if pool is None or len(pool) == 0:
            return None
        return int(pool[self.rng.integers(0, len(pool))])

    def _plan_pure(self, sid, seed, qt):
        di = self._select_dominant_idx(self.gi.general_indices)
        if di is None: return None
        return (sid, 'pure', di, None, float(self.rng.uniform(0.5, 1.0)), 0.0, qt, seed)

    def _plan_complete(self, sid, seed, qt):
        di = self._select_dominant_idx(self.gi.general_indices)
        if di is None: return None
        contam = float(self.rng.uniform(0.0, 100.0))
        ci = self._cross_phylum(self.gi.all_genomes[di]['phylum'], int(self.rng.integers(1, 6)))
        return (sid, 'complete', di, ci, 1.0, contam, qt, seed)

    def _plan_within_phylum(self, sid, seed, qt):
        di = self._select_dominant_idx(self.gi.general_indices)
        if di is None: return None
        comp = float(self.rng.uniform(0.5, 1.0))
        contam = float(self.rng.uniform(0.0, 100.0))
        meta = self.gi.all_genomes[di]
        n = int(self.rng.integers(1, 4))
        ci = self._within_phylum(meta['phylum'], n, meta['accession'])
        if not ci:
            ci = self._cross_phylum(meta['phylum'], 1)
        return (sid, 'within_phylum', di, ci, comp, contam, qt, seed)

    def _plan_cross_phylum(self, sid, seed, qt):
        di = self._select_dominant_idx(self.gi.general_indices)
        if di is None: return None
        comp = float(self.rng.uniform(0.5, 1.0))
        contam = float(self.rng.uniform(0.0, 100.0))
        ci = self._cross_phylum(self.gi.all_genomes[di]['phylum'], int(self.rng.integers(1, 6)))
        return (sid, 'cross_phylum', di, ci, comp, contam, qt, seed)

    def _plan_reduced(self, sid, seed, qt):
        pool = self.gi.reduced_indices if len(self.gi.reduced_indices) > 0 else self.gi.general_indices
        di = self._select_dominant_idx(pool)
        if di is None: return None
        comp = float(self.rng.uniform(0.5, 1.0))
        contam = float(self.rng.uniform(0.0, 100.0))
        ci = self._cross_phylum(self.gi.all_genomes[di]['phylum'], int(self.rng.integers(1, 4)))
        return (sid, 'reduced_genome', di, ci, comp, contam, qt, seed)

    def _plan_archaeal(self, sid, seed, qt):
        pool = self.gi.archaea_indices if len(self.gi.archaea_indices) > 0 else self.gi.general_indices
        di = self._select_dominant_idx(pool)
        if di is None: return None
        comp = float(self.rng.uniform(0.5, 1.0))
        contam = float(self.rng.uniform(0.0, 100.0))
        ci = self._cross_phylum(self.gi.all_genomes[di]['phylum'], int(self.rng.integers(1, 4)))
        return (sid, 'archaeal', di, ci, comp, contam, qt, seed)

    def _within_phylum(self, phylum, n, exclude_acc=''):
        pool = self.gi.by_phylum.get(phylum, None)
        if pool is None or len(pool) == 0:
            return []
        if exclude_acc:
            # Filter out excluded accession (rare operation, OK to be slow)
            mask = np.array([self.gi.all_genomes[i]['accession'] != exclude_acc for i in pool])
            pool = pool[mask]
        if len(pool) == 0:
            return []
        n = min(n, len(pool))
        return [int(i) for i in self.rng.choice(pool, size=n, replace=False)]

    def _cross_phylum(self, dominant_phylum, n):
        other = [p for p in self.gi.phylum_list if p != dominant_phylum]
        if not other:
            other = self.gi.phylum_list
        n_p = min(n, len(other))
        chosen_indices = self.rng.choice(len(other), size=n_p, replace=False)
        result = []
        for idx in chosen_indices:
            p = other[idx]
            pool = self.gi.by_phylum.get(p, None)
            if pool is not None and len(pool) > 0:
                result.append(int(pool[self.rng.integers(0, len(pool))]))
        return result


# ============================================================================
# Checkpoint
# ============================================================================
class CheckpointManager:
    def __init__(self, path):
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
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def is_completed(self, bid): return bid in self.completed_batches

    def mark_completed(self, bid, elapsed):
        self.completed_batches.add(bid)
        self.batch_times[str(bid)] = round(elapsed, 1)
        self._save()

    def mark_error(self, bid, error):
        self.batch_errors[str(bid)] = error
        self._save()


# ============================================================================
# Processing
# ============================================================================
def process_batch(batch_id, split, offset, genome_index, pool, n_workers, logger):
    """Process a single batch using a persistent worker pool."""
    global _genome_index_ref
    batch_start = time.perf_counter()

    # Plan batch
    planner = BatchPlanner(genome_index, batch_id)
    sample_plans, needed_indices = planner.plan_batch()
    n_planned = len(sample_plans)
    plan_time = time.perf_counter() - batch_start

    # Warm page cache for needed genomes
    fasta_paths = list(set(
        genome_index.all_genomes[idx]['fasta_path']
        for idx in needed_indices
        if idx < len(genome_index.all_genomes)
    ))
    logger.info(f"  Batch {batch_id}: {n_planned} samples, {len(fasta_paths)} files, "
                f"plan={plan_time:.1f}s. Warming page cache...")
    cache_elapsed = warm_page_cache(fasta_paths, n_threads=n_workers, logger=logger)

    # Set genome index for workers
    _genome_index_ref = genome_index

    setup_time = time.perf_counter() - batch_start
    logger.info(f"  Batch {batch_id}: Setup done in {setup_time:.1f}s, starting processing...")

    # Process in parallel
    results = []
    n_failures = 0
    chunksize = max(4, n_planned // (n_workers * 8))

    for r in pool.imap_unordered(process_single_sample, sample_plans, chunksize=chunksize):
        if r is not None:
            results.append(r)
        else:
            n_failures += 1

        total_done = len(results) + n_failures
        if total_done % 2000 == 0:
            elapsed = time.perf_counter() - batch_start
            proc_elapsed = elapsed - setup_time
            rate = total_done / proc_elapsed if proc_elapsed > 0 else 0
            logger.info(f"    Batch {batch_id}: {total_done}/{n_planned}, "
                        f"{len(results)} ok, {n_failures} fail, "
                        f"{rate:.0f} samples/sec (processing only)")

    logger.info(f"    Batch {batch_id}: Done: {len(results)} ok, {n_failures} fail")

    # Retries
    max_retries = 3
    retry_count = 0
    while len(results) < BATCH_SIZE and retry_count < max_retries:
        retry_count += 1
        n_needed = BATCH_SIZE - len(results)
        logger.info(f"    Retry {retry_count}, need {n_needed} more")
        rrng = np.random.default_rng(batch_id * 10000 + 90000 + retry_count * 1000)
        rplans = []
        for i in range(n_needed):
            sid = BATCH_SIZE + retry_count * n_needed + i
            seed = batch_id * 10000 + sid
            qt = rrng.choice(QUALITY_TIERS, p=QUALITY_WEIGHTS)
            di = int(rrng.choice(genome_index.general_indices))
            comp = float(rrng.uniform(0.5, 1.0))
            rplans.append((sid, 'pure', di, None, comp, 0.0, qt, seed))

        for r in pool.imap_unordered(process_single_sample, rplans, chunksize=50):
            if r is not None and len(results) < BATCH_SIZE:
                results.append(r)

    results = results[:BATCH_SIZE]

    if len(results) < BATCH_SIZE:
        logger.warning(f"    Batch {batch_id}: Only {len(results)}/{BATCH_SIZE}")

    # Assemble arrays
    n_samples = len(results)
    kmer_counts = np.zeros((n_samples, N_KMER_FEATURES), dtype=np.int64)
    assembly_features = np.zeros((n_samples, N_ASSEMBLY_FEATURES_CONST), dtype=np.float64)
    labels = np.zeros((n_samples, 2), dtype=np.float32)
    metadata = np.zeros(n_samples, dtype=METADATA_DTYPE)

    type_counts = {}
    for i, r in enumerate(results):
        (sid, kc, af, comp, contam, phylum, stype, qt, acc, fl, nc) = r
        kmer_counts[i] = kc
        assembly_features[i] = af
        # Belt-and-suspenders clamp: ensure labels are strictly within protocol ranges
        # Completeness: [0, 100], Contamination: [0, 100]
        comp = float(np.clip(comp, 0.0, 100.0))
        contam = float(np.clip(contam, 0.0, 100.0))
        labels[i, 0] = comp
        labels[i, 1] = contam
        metadata[i]['completeness'] = comp
        metadata[i]['contamination'] = contam
        metadata[i]['dominant_phylum'] = phylum.encode('utf-8')[:64]
        metadata[i]['sample_type'] = stype.encode('utf-8')[:32]
        metadata[i]['quality_tier'] = qt.encode('utf-8')[:20]
        metadata[i]['dominant_accession'] = acc.encode('utf-8')[:30]
        metadata[i]['genome_full_length'] = fl
        metadata[i]['n_contaminants'] = nc
        metadata[i]['batch_id'] = batch_id
        type_counts[stype] = type_counts.get(stype, 0) + 1

    elapsed = time.perf_counter() - batch_start

    return {
        'batch_id': batch_id, 'split': split, 'offset': offset,
        'n_samples': n_samples, 'n_failures': n_failures,
        'kmer_counts': kmer_counts, 'assembly_features': assembly_features,
        'labels': labels, 'metadata': metadata,
        'elapsed': elapsed, 'type_counts': type_counts,
        'setup_time': setup_time,
    }


def write_batch_to_hdf5(batch_result, logger):
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


def update_normalizer(normalizer, batch_result):
    normalizer.update_kmer_batch(batch_result['kmer_counts'].astype(np.float64))
    normalizer.update_assembly_batch(batch_result['assembly_features'])


def save_normalizer_checkpoint(normalizer):
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
    tmp = NORMALIZER_STATS_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(stats, f)
    os.replace(tmp, NORMALIZER_STATS_PATH)


def load_normalizer_checkpoint(normalizer):
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
        if stats.get('kmer_reservoir') != 'too_large' and stats.get('kmer_reservoir'):
            rd = np.array(stats['kmer_reservoir'], dtype=np.float64)
            n = min(len(rd), normalizer.kmer_stats.reservoir_size)
            normalizer.kmer_stats.reservoir[:n] = rd[:n]
            normalizer.kmer_stats.reservoir_count = n
        if stats.get('assembly_reservoir') != 'too_large' and stats.get('assembly_reservoir'):
            rd = np.array(stats['assembly_reservoir'], dtype=np.float64)
            n = min(len(rd), normalizer.assembly_stats.reservoir_size)
            normalizer.assembly_stats.reservoir[:n] = rd[:n]
            normalizer.assembly_stats.reservoir_count = n
        return True
    except Exception:
        return False


def finalize_normalization(normalizer, logger):
    logger.info("Finalizing normalization parameters...")
    normalizer.finalize()
    normalizer.save(NORM_PARAMS_PATH)
    logger.info(f"  Saved normalization params to {NORM_PARAMS_PATH}")
    logger.info("Applying normalization to all stored features...")
    for split, total in [('train', 800_000), ('val', 100_000), ('test', 100_000)]:
        logger.info(f"  Normalizing {split} ({total:,} samples)...")
        for start in range(0, total, 10_000):
            end = min(start + 10_000, total)
            with FeatureStore(HDF5_PATH, mode='a') as store:
                kmer, asm, _, _ = store.read_batch(split, start, end)
                store._file[split]['kmer_features'][start:end] = normalizer.normalize_kmer(kmer).astype(np.float32)
                store._file[split]['assembly_features'][start:end] = normalizer.normalize_assembly(asm).astype(np.float32)
                store._file.flush()
        logger.info(f"    {split} done")
    logger.info("All normalization applied")


def print_summary(checkpoint, normalizer, total_start, logger):
    total_elapsed = time.perf_counter() - total_start
    n_completed = len(checkpoint.completed_batches)
    bt = list(checkpoint.batch_times.values())
    avg_t = sum(bt)/len(bt) if bt else 0
    min_t = min(bt) if bt else 0
    max_t = max(bt) if bt else 0
    logger.info("=" * 70)
    logger.info("BATCH SYNTHESIS v2 SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Completed: {n_completed}/100, Total: {n_completed * BATCH_SIZE:,}")
    logger.info(f"  Errors: {len(checkpoint.batch_errors)}")
    logger.info(f"  Total time: {total_elapsed/3600:.1f}h, Avg batch: {avg_t:.1f}s")
    logger.info(f"  Min/Max: {min_t:.1f}s / {max_t:.1f}s")
    try:
        with FeatureStore(HDF5_PATH, mode='r') as store:
            for s in ['train', 'val', 'test']:
                info = store.get_split_info(s)
                logger.info(f"  HDF5 {s}: {info['n_written']:,}/{info['n_total']:,}")
    except Exception as e:
        logger.warning(f"  HDF5 info error: {e}")
    logger.info("=" * 70)


# ============================================================================
# Main
# ============================================================================
def main():
    global _genome_index_ref

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--start-batch', type=int, default=0)
    parser.add_argument('--end-batch', type=int, default=100)
    parser.add_argument('--test-batch', type=int, default=None)
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("Phase 4: Batch Genome Synthesis v2 (OPTIMIZED)")
    logger.info("=" * 70)

    n_cpus = mp.cpu_count()
    n_workers = args.workers if args.workers else min(44, max(1, n_cpus - 4))
    logger.info(f"System: {n_cpus} CPUs, using {n_workers} workers")

    logger.info("Loading k-mer index and warming up Numba JIT...")
    t0 = time.perf_counter()
    init_kmer_globals()
    logger.info(f"  Done in {time.perf_counter()-t0:.1f}s")

    checkpoint = CheckpointManager(CHECKPOINT_PATH)
    logger.info(f"Checkpoint: {len(checkpoint.completed_batches)} batches done")

    normalizer = FeatureNormalizer(n_kmer_features=N_KMER_FEATURES, reservoir_size=50000)
    if load_normalizer_checkpoint(normalizer):
        logger.info(f"Loaded normalizer: {normalizer.kmer_stats.count:,} samples")

    total_start = time.perf_counter()

    if args.test_batch is not None:
        args.start_batch = args.test_batch
        args.end_batch = args.test_batch + 1

    if not args.finalize_only:
        split_paths = {
            'train': TRAIN_GENOMES_PATH,
            'val': VAL_GENOMES_PATH,
            'test': TEST_GENOMES_PATH,
        }
        genome_indices = {}
        current_pool = None
        current_split = None

        for batch_id, split, offset in BATCH_ASSIGNMENTS:
            if batch_id < args.start_batch or batch_id >= args.end_batch:
                continue
            if checkpoint.is_completed(batch_id):
                logger.info(f"Batch {batch_id} ({split}): Already done, skipping")
                continue

            # Load genome index if needed
            if split not in genome_indices:
                logger.info(f"Loading genome index for {split}...")
                t0 = time.perf_counter()
                genome_indices[split] = GenomeIndex(split_paths[split])
                gi = genome_indices[split]
                logger.info(f"  {len(gi.all_genomes):,} genomes, {len(gi.phylum_list)} phyla "
                            f"in {time.perf_counter()-t0:.1f}s")

            # Create/replace pool if split changed
            if split != current_split:
                if current_pool is not None:
                    current_pool.terminate()
                    current_pool.join()
                _genome_index_ref = genome_indices[split]
                current_pool = mp.Pool(processes=n_workers)
                current_split = split
                logger.info(f"Created worker pool for split '{split}'")

            genome_index = genome_indices[split]

            # ETA
            n_rem = sum(1 for b, _, _ in BATCH_ASSIGNMENTS
                       if b >= batch_id and b < args.end_batch
                       and not checkpoint.is_completed(b))
            v2t = [float(v) for k, v in checkpoint.batch_times.items()
                   if int(k) >= args.start_batch]
            avg_t = (sum(v2t) / len(v2t)) if v2t else 0
            logger.info(f"\nBatch {batch_id}/99 ({split}, offset={offset}) "
                        f"[{n_rem} remaining, ETA ~{avg_t*n_rem/60:.0f} min]")

            try:
                batch_result = process_batch(
                    batch_id, split, offset, genome_index,
                    current_pool, n_workers, logger
                )

                write_batch_to_hdf5(batch_result, logger)
                update_normalizer(normalizer, batch_result)
                checkpoint.mark_completed(batch_id, batch_result['elapsed'])

                if (batch_id + 1) % 5 == 0:
                    save_normalizer_checkpoint(normalizer)

                el = batch_result['elapsed']
                ns = batch_result['n_samples']
                st = batch_result.get('setup_time', 0)
                proc_rate = ns / (el - st) if (el - st) > 0 else 0
                logger.info(f"    Batch {batch_id} COMPLETE: {ns} samples, "
                            f"{el:.1f}s total ({st:.1f}s setup + {el-st:.1f}s proc), "
                            f"{proc_rate:.0f} samples/sec (proc)")
                logger.info(f"    Types: {batch_result['type_counts']}")
                logger.info(f"    Comp: [{batch_result['labels'][:,0].min():.1f}, "
                            f"{batch_result['labels'][:,0].max():.1f}]%, "
                            f"mean={batch_result['labels'][:,0].mean():.1f}%")
                logger.info(f"    Contam: [{batch_result['labels'][:,1].min():.1f}, "
                            f"{batch_result['labels'][:,1].max():.1f}]%, "
                            f"mean={batch_result['labels'][:,1].mean():.1f}%")

                del batch_result
                gc.collect()

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Batch {batch_id} FAILED: {e}\n{tb}")
                checkpoint.mark_error(batch_id, str(e))

        if current_pool is not None:
            current_pool.terminate()
            current_pool.join()

        save_normalizer_checkpoint(normalizer)
        logger.info("Saved final normalizer checkpoint")

    # Finalize
    all_done = all(checkpoint.is_completed(b) for b, _, _ in BATCH_ASSIGNMENTS)
    if all_done or args.finalize_only:
        if args.finalize_only and normalizer.kmer_stats.count == 0:
            logger.info("Rebuilding normalizer...")
            normalizer = FeatureNormalizer(n_kmer_features=N_KMER_FEATURES, reservoir_size=50000)
            for s, n in [('train', 800_000), ('val', 100_000), ('test', 100_000)]:
                for start in range(0, n, 10_000):
                    end = min(start + 10_000, n)
                    with FeatureStore(HDF5_PATH, mode='r') as store:
                        kmer, asm, _, _ = store.read_batch(s, start, end)
                    normalizer.update_kmer_batch(kmer.astype(np.float64))
                    normalizer.update_assembly_batch(asm.astype(np.float64))
        finalize_normalization(normalizer, logger)
    else:
        logger.info(f"Not all done ({len(checkpoint.completed_batches)}/100)")

    print_summary(checkpoint, normalizer, total_start, logger)


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
