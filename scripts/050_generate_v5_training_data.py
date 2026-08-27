#!/usr/bin/env python3
"""
Generate 200K new training samples for MAGICC V5 model and merge with V4 data.

Part A: 100K samples with 100% completeness and 0% contamination
  - load_original_contigs() to get full reference genome
  - No contamination

Part B: 100K samples with 100% completeness and 0-10% contamination
  - load_original_contigs() for dominant genome
  - Add 0-10% contamination (uniform)
  - Both within-phylum and cross-phylum contamination

Pipeline per sample:
  1. Load dominant genome with load_original_contigs()
  2. For contaminated samples, add contaminant sequences
  3. Count k-mers (9,249 selected k-mers)
  4. Compute assembly stats (7 features)
  5. Store in HDF5

Disk space strategy:
  - Remove old V3 26-feat HDF5 to free ~46 GB
  - Generate 200K new samples into temp HDF5 (~4 GB)
  - Create V5 by streaming copy from V4 + appending new samples (~56 GB)
  - V4 remains untouched

Usage:
    conda activate magicc2
    python scripts/50_generate_v5_training_data.py [--workers N] [--phase PHASE]
    python scripts/50_generate_v5_training_data.py --phase generate   # generate 200K
    python scripts/50_generate_v5_training_data.py --phase merge      # merge into V5
    python scripts/50_generate_v5_training_data.py --phase verify     # verify V5
    python scripts/50_generate_v5_training_data.py --phase all        # all steps
"""

import sys
import os

os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import gc
import json
import time
import shutil
import logging
import argparse
import traceback
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import h5py

PROJECT_ROOT = '/path/to/magicc'
sys.path.insert(0, PROJECT_ROOT)

from magicc.fragmentation import (read_fasta, load_original_contigs,
                                   _warm_numba_fragmentation)
from magicc.contamination import (
    select_contaminant_target_bp, fragment_contaminant,
    compute_contamination_rate
)
from magicc.kmer_counter import (load_selected_kmers, build_kmer_index,
                                  _count_kmers_single, K)
from magicc.assembly_stats import compute_assembly_stats, N_FEATURES as N_ASSEMBLY_FEATURES
from magicc.normalization import FeatureNormalizer
from magicc.storage import METADATA_DTYPE

# ============================================================================
# Constants
# ============================================================================
KMER_PATH = os.path.join(PROJECT_ROOT, 'data/kmer_selection/selected_kmers.txt')
V4_HDF5_PATH = os.path.join(PROJECT_ROOT, 'data/features/magicc_features.h5')
V5_HDF5_PATH = os.path.join(PROJECT_ROOT, 'data/features/magicc_v5_features.h5')
TEMP_HDF5_PATH = os.path.join(PROJECT_ROOT, 'data/features/v5_new_200k_temp.h5')
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'data/features/v5_generation_checkpoint.json')
NORM_PARAMS_PATH = os.path.join(PROJECT_ROOT, 'data/features/normalization_params.json')
LOG_PATH = os.path.join(PROJECT_ROOT, 'data/features/v5_generation.log')

TRAIN_GENOMES_PATH = os.path.join(PROJECT_ROOT, 'data/splits/train_genomes.tsv')

# Old V3 files to clean up for disk space
V3_CLEANUP_FILES = [
    os.path.join(PROJECT_ROOT, 'data/features/magicc_features_v3_26feat.h5'),
    os.path.join(PROJECT_ROOT, 'data/features/batch_checkpoint_v3_26feat.json'),
    os.path.join(PROJECT_ROOT, 'data/features/normalization_params_v3_26feat.json'),
    os.path.join(PROJECT_ROOT, 'data/features/normalizer_running_stats_v3_26feat.json'),
]

BATCH_SIZE = 10_000
N_KMER_FEATURES = 9249
N_ASSEMBLY_FEATURES_CONST = 7

# Part A: 100K pure complete, Part B: 100K complete with 0-10% contamination
N_PART_A = 100_000  # pure, 100% completeness, 0% contamination
N_PART_B = 100_000  # 100% completeness, 0-10% contamination
N_TOTAL_NEW = N_PART_A + N_PART_B

# Batch structure: 10 Part-A batches + 10 Part-B batches = 20 batches of 10K
N_BATCHES_A = N_PART_A // BATCH_SIZE  # 10
N_BATCHES_B = N_PART_B // BATCH_SIZE  # 10
N_TOTAL_BATCHES = N_BATCHES_A + N_BATCHES_B  # 20

# V4 and V5 split sizes
V4_TRAIN = 800_000
V4_VAL = 100_000
V4_TEST = 100_000
V5_TRAIN = V4_TRAIN + N_TOTAL_NEW  # 1,000,000
V5_VAL = V4_VAL     # 100,000
V5_TEST = V4_TEST   # 100,000


# ============================================================================
# Logging
# ============================================================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH, mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger('v5_generation')


# ============================================================================
# Genome Index (same as V4 pipeline)
# ============================================================================
REDUCED_GENOME_PHYLA = {
    'Patescibacteriota',
    'Aenigmatarchaeota', 'Altiarchaeota', 'Diapherotrites',
    'Huberarchaeota', 'Iainarchaeota', 'Micrarchaeota',
    'Nanoarchaeota', 'Nanohaloarchaeota', 'Nanohalarchaeota',
    'Undinarchaeota', 'Woesearchaeota', 'Nanobdellota',
    'Dependentiae', 'Bdellovibrionota',
}


class GenomeIndex:
    """Index of reference genomes."""

    def __init__(self, tsv_path: str):
        self.all_genomes = []
        self.by_phylum = {}
        self.general_indices = None
        self.phylum_list = []
        self._load(tsv_path)

    def _load(self, tsv_path):
        _by_phylum_lists = {}
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
                _general.append(idx)
        self.by_phylum = {k: np.array(v, dtype=np.int64) for k, v in _by_phylum_lists.items()}
        self.general_indices = np.array(_general, dtype=np.int64)
        self.phylum_list = list(self.by_phylum.keys())


# ============================================================================
# Module-level globals (inherited by fork)
# ============================================================================
_kmer_index = None
_n_kmer_features = None
_genome_index_ref = None


def init_kmer_globals():
    global _kmer_index, _n_kmer_features
    kmer_codes = load_selected_kmers(KMER_PATH)
    _kmer_index = build_kmer_index(kmer_codes)
    _n_kmer_features = len(kmer_codes)
    # Warm up Numba JIT
    dummy_seq = np.frombuffer(b'ACGTACGTACGTACGTACGT', dtype=np.uint8)
    _count_kmers_single(dummy_seq, _kmer_index, _n_kmer_features, K)
    _warm_numba_fragmentation()


# ============================================================================
# Page Cache Warming
# ============================================================================
def warm_page_cache(fasta_paths, n_threads=44, logger=None):
    """Read genome files to warm the OS page cache."""
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
# Worker: Process single sample
# ============================================================================
def process_sample_part_a(args):
    """Part A: 100% completeness, 0% contamination."""
    global _kmer_index, _n_kmer_features, _genome_index_ref

    sample_id, dominant_idx, seed = args
    rng = np.random.default_rng(seed)

    try:
        info = _genome_index_ref.all_genomes[dominant_idx]
        contigs = load_original_contigs(info['fasta_path'])
        if not contigs:
            return None

        dominant_full_length = sum(len(c) for c in contigs)
        if dominant_full_length < 500:
            return None

        # K-mer counting
        total_counts = np.zeros(_n_kmer_features, dtype=np.int64)
        for contig in contigs:
            if len(contig) >= K:
                seq_bytes = np.frombuffer(contig.encode('ascii'), dtype=np.uint8)
                total_counts += _count_kmers_single(
                    seq_bytes, _kmer_index, _n_kmer_features, K
                )

        total_sum = total_counts.sum()
        log10_total = np.log10(float(total_sum)) if total_sum > 0 else 0.0
        assembly_features = compute_assembly_stats(log10_total, total_counts)

        return (
            sample_id,
            total_counts,
            assembly_features,
            100.0,   # completeness
            0.0,     # contamination
            info['phylum'],
            'v5_pure_complete',
            'high',
            info['accession'],
            dominant_full_length,
            0,       # n_contaminants
        )
    except Exception:
        return None


def process_sample_part_b(args):
    """Part B: 100% completeness, 0-10% contamination."""
    global _kmer_index, _n_kmer_features, _genome_index_ref

    (sample_id, dominant_idx, contaminant_indices,
     target_contamination, contamination_type, seed) = args
    rng = np.random.default_rng(seed)

    try:
        info = _genome_index_ref.all_genomes[dominant_idx]
        # Load dominant genome as original contigs (100% completeness)
        dominant_contigs = load_original_contigs(info['fasta_path'])
        if not dominant_contigs:
            return None

        dominant_full_length = sum(len(c) for c in dominant_contigs)
        if dominant_full_length < 500:
            return None

        # Add contamination
        contaminant_contigs_all = []
        if target_contamination > 0 and contaminant_indices:
            total_target_bp = select_contaminant_target_bp(
                target_contamination, dominant_full_length
            )
            if total_target_bp <= 0:
                # Zero contamination effectively
                pass
            else:
                # Read contaminant genomes
                contaminant_seqs = []
                for ci in contaminant_indices:
                    ci_info = _genome_index_ref.all_genomes[ci]
                    try:
                        cseq = read_fasta(ci_info['fasta_path'])
                        if cseq and len(cseq) >= 500:
                            contaminant_seqs.append(cseq)
                    except Exception:
                        pass

                if contaminant_seqs:
                    n_cont = len(contaminant_seqs)
                    if n_cont == 1:
                        bp_per = [total_target_bp]
                    else:
                        proportions = rng.dirichlet(np.ones(n_cont))
                        bp_per = (proportions * total_target_bp).astype(int)
                        bp_per[-1] = total_target_bp - bp_per[:-1].sum()

                    for seq, tbp in zip(contaminant_seqs, bp_per):
                        if tbp <= 0 or len(seq) == 0:
                            continue
                        cr = fragment_contaminant(seq, tbp, rng)
                        contaminant_contigs_all.extend(cr['contigs'])

                    # Cap contaminant bp to target
                    contaminant_total_bp = sum(len(c) for c in contaminant_contigs_all)
                    max_allowed_bp = int(target_contamination / 100.0 * dominant_full_length)
                    if contaminant_total_bp > max_allowed_bp and contaminant_contigs_all:
                        cc_idx = np.arange(len(contaminant_contigs_all))
                        rng.shuffle(cc_idx)
                        kept = []
                        kept_bp = 0
                        for idx in cc_idx:
                            c = contaminant_contigs_all[idx]
                            if kept_bp + len(c) <= max_allowed_bp:
                                kept.append(c)
                                kept_bp += len(c)
                            else:
                                remaining = max_allowed_bp - kept_bp
                                if remaining >= 500:
                                    kept.append(c[:remaining])
                                    kept_bp += remaining
                                break
                        contaminant_contigs_all = kept if kept else []

        # Merge contigs
        all_contigs = dominant_contigs + contaminant_contigs_all
        if len(all_contigs) > 1:
            indices = np.arange(len(all_contigs))
            rng.shuffle(indices)
            all_contigs = [all_contigs[i] for i in indices]

        # Actual contamination
        contaminant_total_bp = sum(len(c) for c in contaminant_contigs_all)
        actual_contamination = compute_contamination_rate(
            contaminant_total_bp, dominant_full_length
        )

        # K-mer counting
        total_counts = np.zeros(_n_kmer_features, dtype=np.int64)
        for contig in all_contigs:
            if len(contig) >= K:
                seq_bytes = np.frombuffer(contig.encode('ascii'), dtype=np.uint8)
                total_counts += _count_kmers_single(
                    seq_bytes, _kmer_index, _n_kmer_features, K
                )

        total_sum = total_counts.sum()
        log10_total = np.log10(float(total_sum)) if total_sum > 0 else 0.0
        assembly_features = compute_assembly_stats(log10_total, total_counts)

        n_contaminants = len(contaminant_indices) if contaminant_indices else 0

        return (
            sample_id,
            total_counts,
            assembly_features,
            100.0,    # completeness (always 100% for Part B)
            actual_contamination,
            info['phylum'],
            f'v5_{contamination_type}',
            'high',
            info['accession'],
            dominant_full_length,
            n_contaminants,
        )
    except Exception:
        return None


# ============================================================================
# Batch Planning
# ============================================================================
class V5BatchPlanner:
    """Plan batches for V5 new samples."""

    def __init__(self, genome_index: GenomeIndex, batch_id: int, part: str):
        self.gi = genome_index
        self.batch_id = batch_id
        self.part = part  # 'A' or 'B'
        self.rng = np.random.default_rng(batch_id * 100000 + 42)

    def plan_batch(self):
        """Plan a batch of 10K samples."""
        samples = []
        needed_indices = set()

        if self.part == 'A':
            samples, needed_indices = self._plan_part_a()
        elif self.part == 'B':
            samples, needed_indices = self._plan_part_b()

        return samples, needed_indices

    def _plan_part_a(self):
        """Plan 10K pure complete samples."""
        samples = []
        needed = set()
        n_genomes = len(self.gi.general_indices)

        for i in range(BATCH_SIZE):
            seed = self.batch_id * 100000 + i + 1000000
            dom_idx = int(self.gi.general_indices[self.rng.integers(0, n_genomes)])
            samples.append((i, dom_idx, seed))
            needed.add(dom_idx)

        return samples, needed

    def _plan_part_b(self):
        """Plan 10K complete samples with 0-10% contamination."""
        samples = []
        needed = set()
        n_genomes = len(self.gi.general_indices)

        for i in range(BATCH_SIZE):
            seed = self.batch_id * 100000 + i + 2000000
            dom_idx = int(self.gi.general_indices[self.rng.integers(0, n_genomes)])
            needed.add(dom_idx)

            # Contamination rate: uniform 0-10%
            target_contam = float(self.rng.uniform(0.0, 10.0))

            # Choose contamination type (50/50 within/cross phylum)
            dom_phylum = self.gi.all_genomes[dom_idx]['phylum']
            if self.rng.random() < 0.5:
                contam_type = 'within_phylum_low_contam'
                contam_indices = self._within_phylum(dom_phylum, dom_idx)
            else:
                contam_type = 'cross_phylum_low_contam'
                contam_indices = self._cross_phylum(dom_phylum)

            for ci in contam_indices:
                needed.add(ci)

            samples.append((i, dom_idx, contam_indices, target_contam, contam_type, seed))

        return samples, needed

    def _within_phylum(self, phylum, exclude_idx):
        """Select 1-3 contaminant genomes from same phylum."""
        pool = self.gi.by_phylum.get(phylum, None)
        if pool is None or len(pool) == 0:
            return self._cross_phylum(phylum)

        # Filter out dominant genome
        mask = pool != exclude_idx
        pool = pool[mask]
        if len(pool) == 0:
            return self._cross_phylum(phylum)

        n = min(int(self.rng.integers(1, 4)), len(pool))
        return [int(i) for i in self.rng.choice(pool, size=n, replace=False)]

    def _cross_phylum(self, dominant_phylum):
        """Select 1-3 contaminant genomes from different phyla."""
        other = [p for p in self.gi.phylum_list if p != dominant_phylum]
        if not other:
            other = self.gi.phylum_list
        n = min(int(self.rng.integers(1, 4)), len(other))
        chosen_phyla_idx = self.rng.choice(len(other), size=n, replace=False)
        result = []
        for pi in chosen_phyla_idx:
            p = other[pi]
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
        self.phase_complete = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.completed_batches = set(data.get('completed_batches', []))
                self.batch_times = data.get('batch_times', {})
                self.batch_errors = data.get('batch_errors', {})
                self.phase_complete = data.get('phase_complete', {})
            except Exception:
                pass

    def _save(self):
        data = {
            'completed_batches': sorted(list(self.completed_batches)),
            'batch_times': self.batch_times,
            'batch_errors': self.batch_errors,
            'phase_complete': self.phase_complete,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def is_completed(self, bid):
        return bid in self.completed_batches

    def mark_completed(self, bid, elapsed):
        self.completed_batches.add(bid)
        self.batch_times[str(bid)] = round(elapsed, 1)
        self._save()

    def mark_error(self, bid, error):
        self.batch_errors[str(bid)] = error
        self._save()

    def mark_phase(self, phase, status='complete'):
        self.phase_complete[phase] = status
        self._save()

    def is_phase_complete(self, phase):
        return self.phase_complete.get(phase) == 'complete'


# ============================================================================
# Phase 0: Disk Space Cleanup
# ============================================================================
def cleanup_disk_space(logger):
    """Remove old V3 26-feat files to free disk space."""
    total_freed = 0
    for fpath in V3_CLEANUP_FILES:
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            logger.info(f"  Removing {os.path.basename(fpath)} ({size/1e9:.1f} GB)")
            os.remove(fpath)
            total_freed += size

    if total_freed > 0:
        logger.info(f"  Freed {total_freed/1e9:.1f} GB total")
    else:
        logger.info("  No old V3 files to remove")

    # Check available disk space
    stat = os.statvfs(os.path.dirname(V5_HDF5_PATH))
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    logger.info(f"  Available disk space: {free_gb:.1f} GB")
    if free_gb < 70:
        logger.warning(f"  Low disk space! Need ~60 GB for V5 + temp. Available: {free_gb:.1f} GB")
    return free_gb


# ============================================================================
# Phase 1: Generate 200K new samples into temp HDF5
# ============================================================================
def create_temp_hdf5(logger):
    """Create temporary HDF5 for 200K new samples."""
    if os.path.exists(TEMP_HDF5_PATH):
        logger.info(f"  Temp HDF5 already exists, will append")
        return

    logger.info(f"  Creating temp HDF5 at {TEMP_HDF5_PATH}")
    with h5py.File(TEMP_HDF5_PATH, 'w') as f:
        # Part A: 100K pure complete
        ga = f.create_group('part_a')
        ga.create_dataset('kmer_features', shape=(N_PART_A, N_KMER_FEATURES),
                          dtype=np.float32, chunks=(BATCH_SIZE, N_KMER_FEATURES),
                          compression='gzip', compression_opts=1, fillvalue=0.0)
        ga.create_dataset('assembly_features', shape=(N_PART_A, N_ASSEMBLY_FEATURES_CONST),
                          dtype=np.float32, chunks=(BATCH_SIZE, N_ASSEMBLY_FEATURES_CONST),
                          compression='gzip', compression_opts=1, fillvalue=0.0)
        ga.create_dataset('labels', shape=(N_PART_A, 2), dtype=np.float32,
                          chunks=(BATCH_SIZE, 2), compression='gzip', compression_opts=1)
        ga.create_dataset('metadata', shape=(N_PART_A,), dtype=METADATA_DTYPE,
                          chunks=(BATCH_SIZE,), compression='gzip', compression_opts=1)
        ga.attrs['n_written'] = 0
        ga.attrs['n_total'] = N_PART_A

        # Part B: 100K complete with contamination
        gb = f.create_group('part_b')
        gb.create_dataset('kmer_features', shape=(N_PART_B, N_KMER_FEATURES),
                          dtype=np.float32, chunks=(BATCH_SIZE, N_KMER_FEATURES),
                          compression='gzip', compression_opts=1, fillvalue=0.0)
        gb.create_dataset('assembly_features', shape=(N_PART_B, N_ASSEMBLY_FEATURES_CONST),
                          dtype=np.float32, chunks=(BATCH_SIZE, N_ASSEMBLY_FEATURES_CONST),
                          compression='gzip', compression_opts=1, fillvalue=0.0)
        gb.create_dataset('labels', shape=(N_PART_B, 2), dtype=np.float32,
                          chunks=(BATCH_SIZE, 2), compression='gzip', compression_opts=1)
        gb.create_dataset('metadata', shape=(N_PART_B,), dtype=METADATA_DTYPE,
                          chunks=(BATCH_SIZE,), compression='gzip', compression_opts=1)
        gb.attrs['n_written'] = 0
        gb.attrs['n_total'] = N_PART_B

        f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"  Temp HDF5 created")


def process_batch_part_a(batch_id, genome_index, pool, n_workers, logger):
    """Process a Part A batch (pure complete, no contamination)."""
    global _genome_index_ref
    batch_start = time.perf_counter()

    planner = V5BatchPlanner(genome_index, batch_id, 'A')
    sample_plans, needed_indices = planner.plan_batch()
    n_planned = len(sample_plans)

    # Warm page cache
    fasta_paths = list(set(
        genome_index.all_genomes[idx]['fasta_path']
        for idx in needed_indices
        if idx < len(genome_index.all_genomes)
    ))
    logger.info(f"  Part A Batch {batch_id}: {n_planned} samples, "
                f"{len(fasta_paths)} files. Warming cache...")
    warm_page_cache(fasta_paths, n_threads=n_workers, logger=logger)

    _genome_index_ref = genome_index
    setup_time = time.perf_counter() - batch_start

    # Process
    results = []
    n_failures = 0
    chunksize = max(4, n_planned // (n_workers * 8))

    for r in pool.imap_unordered(process_sample_part_a, sample_plans, chunksize=chunksize):
        if r is not None:
            results.append(r)
        else:
            n_failures += 1
        total_done = len(results) + n_failures
        if total_done % 2000 == 0:
            elapsed = time.perf_counter() - batch_start
            rate = total_done / (elapsed - setup_time) if (elapsed - setup_time) > 0 else 0
            logger.info(f"    Part A Batch {batch_id}: {total_done}/{n_planned}, "
                        f"{len(results)} ok, {n_failures} fail, {rate:.0f}/sec")

    # Verify: all must have completeness=100% and contamination=0%
    verified = []
    for r in results:
        comp = r[3]
        contam = r[4]
        if comp == 100.0 and contam == 0.0:
            verified.append(r)
    n_rejected = len(results) - len(verified)
    if n_rejected > 0:
        logger.warning(f"    Part A Batch {batch_id}: Rejected {n_rejected} samples "
                       f"(not 100%/0%)")

    # Retry if needed
    max_retries = 5
    retry = 0
    while len(verified) < BATCH_SIZE and retry < max_retries:
        retry += 1
        n_need = BATCH_SIZE - len(verified)
        logger.info(f"    Part A retry {retry}: need {n_need} more")
        rrng = np.random.default_rng(batch_id * 100000 + 90000 + retry * 10000)
        rplans = []
        n_gen = len(genome_index.general_indices)
        for i in range(n_need * 2):  # over-generate to account for failures
            sid = BATCH_SIZE + retry * n_need * 2 + i
            seed = batch_id * 100000 + sid + 5000000
            di = int(genome_index.general_indices[rrng.integers(0, n_gen)])
            rplans.append((sid, di, seed))

        for r in pool.imap_unordered(process_sample_part_a, rplans, chunksize=50):
            if r is not None and r[3] == 100.0 and r[4] == 0.0:
                verified.append(r)
                if len(verified) >= BATCH_SIZE:
                    break

    verified = verified[:BATCH_SIZE]
    elapsed = time.perf_counter() - batch_start

    if len(verified) < BATCH_SIZE:
        logger.warning(f"    Part A Batch {batch_id}: Only {len(verified)}/{BATCH_SIZE}")

    return _assemble_results(verified, batch_id, elapsed, setup_time, logger)


def process_batch_part_b(batch_id, genome_index, pool, n_workers, logger):
    """Process a Part B batch (complete with 0-10% contamination)."""
    global _genome_index_ref
    batch_start = time.perf_counter()

    planner = V5BatchPlanner(genome_index, batch_id, 'B')
    sample_plans, needed_indices = planner.plan_batch()
    n_planned = len(sample_plans)

    # Warm page cache
    fasta_paths = list(set(
        genome_index.all_genomes[idx]['fasta_path']
        for idx in needed_indices
        if idx < len(genome_index.all_genomes)
    ))
    logger.info(f"  Part B Batch {batch_id}: {n_planned} samples, "
                f"{len(fasta_paths)} files. Warming cache...")
    warm_page_cache(fasta_paths, n_threads=n_workers, logger=logger)

    _genome_index_ref = genome_index
    setup_time = time.perf_counter() - batch_start

    # Process
    results = []
    n_failures = 0
    chunksize = max(4, n_planned // (n_workers * 8))

    for r in pool.imap_unordered(process_sample_part_b, sample_plans, chunksize=chunksize):
        if r is not None:
            results.append(r)
        else:
            n_failures += 1
        total_done = len(results) + n_failures
        if total_done % 2000 == 0:
            elapsed = time.perf_counter() - batch_start
            rate = total_done / (elapsed - setup_time) if (elapsed - setup_time) > 0 else 0
            logger.info(f"    Part B Batch {batch_id}: {total_done}/{n_planned}, "
                        f"{len(results)} ok, {n_failures} fail, {rate:.0f}/sec")

    # Verify: completeness must be 100%, contamination in [0%, 10%]
    verified = []
    for r in results:
        comp = r[3]
        contam = r[4]
        if comp == 100.0 and 0.0 <= contam <= 10.0:
            verified.append(r)
    n_rejected = len(results) - len(verified)
    if n_rejected > 0:
        logger.warning(f"    Part B Batch {batch_id}: Rejected {n_rejected} samples "
                       f"(outside 100%/[0-10%])")

    # Retry if needed
    max_retries = 5
    retry = 0
    while len(verified) < BATCH_SIZE and retry < max_retries:
        retry += 1
        n_need = BATCH_SIZE - len(verified)
        logger.info(f"    Part B retry {retry}: need {n_need} more")
        rrng = np.random.default_rng(batch_id * 100000 + 90000 + retry * 10000)
        rplans = []
        n_gen = len(genome_index.general_indices)
        for i in range(n_need * 2):
            sid = BATCH_SIZE + retry * n_need * 2 + i
            seed = batch_id * 100000 + sid + 6000000
            di = int(genome_index.general_indices[rrng.integers(0, n_gen)])
            target_contam = float(rrng.uniform(0.0, 10.0))
            dom_phylum = genome_index.all_genomes[di]['phylum']
            # Simple cross-phylum for retries
            other_phyla = [p for p in genome_index.phylum_list if p != dom_phylum]
            if other_phyla:
                cp = other_phyla[int(rrng.integers(0, len(other_phyla)))]
                cp_pool = genome_index.by_phylum.get(cp, genome_index.general_indices)
                ci = [int(cp_pool[rrng.integers(0, len(cp_pool))])]
            else:
                ci = []
            rplans.append((sid, di, ci, target_contam, 'cross_phylum_low_contam', seed))

        for r in pool.imap_unordered(process_sample_part_b, rplans, chunksize=50):
            if r is not None and r[3] == 100.0 and 0.0 <= r[4] <= 10.0:
                verified.append(r)
                if len(verified) >= BATCH_SIZE:
                    break

    verified = verified[:BATCH_SIZE]
    elapsed = time.perf_counter() - batch_start

    if len(verified) < BATCH_SIZE:
        logger.warning(f"    Part B Batch {batch_id}: Only {len(verified)}/{BATCH_SIZE}")

    return _assemble_results(verified, batch_id, elapsed, setup_time, logger)


def _assemble_results(results, batch_id, elapsed, setup_time, logger):
    """Assemble worker results into arrays."""
    n_samples = len(results)
    kmer_counts = np.zeros((n_samples, N_KMER_FEATURES), dtype=np.int64)
    assembly_features = np.zeros((n_samples, N_ASSEMBLY_FEATURES_CONST), dtype=np.float64)
    labels = np.zeros((n_samples, 2), dtype=np.float32)
    metadata = np.zeros(n_samples, dtype=METADATA_DTYPE)

    for i, r in enumerate(results):
        (sid, kc, af, comp, contam, phylum, stype, qt, acc, fl, nc) = r
        kmer_counts[i] = kc
        assembly_features[i] = af
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

    proc_rate = n_samples / (elapsed - setup_time) if (elapsed - setup_time) > 0 else 0
    logger.info(f"    Batch {batch_id} DONE: {n_samples} samples, "
                f"{elapsed:.1f}s total, {proc_rate:.0f}/sec")

    return {
        'batch_id': batch_id,
        'n_samples': n_samples,
        'kmer_counts': kmer_counts,
        'assembly_features': assembly_features,
        'labels': labels,
        'metadata': metadata,
        'elapsed': elapsed,
    }


def write_batch_to_temp(batch_result, part, logger):
    """Write batch results to temp HDF5."""
    group_name = f'part_{part.lower()}'
    with h5py.File(TEMP_HDF5_PATH, 'a') as f:
        grp = f[group_name]
        offset = int(grp.attrs['n_written'])
        n = batch_result['n_samples']
        end = offset + n

        grp['kmer_features'][offset:end] = batch_result['kmer_counts'].astype(np.float32)
        grp['assembly_features'][offset:end] = batch_result['assembly_features'].astype(np.float32)
        grp['labels'][offset:end] = batch_result['labels']
        grp['metadata'][offset:end] = batch_result['metadata']
        grp.attrs['n_written'] = end
        f.flush()

    logger.info(f"    Wrote {n} samples to temp {group_name} at offset {offset}")


def generate_all_samples(n_workers, logger, checkpoint):
    """Generate all 200K new samples."""
    global _genome_index_ref

    # Load genome index
    logger.info("Loading genome index for training genomes...")
    t0 = time.perf_counter()
    genome_index = GenomeIndex(TRAIN_GENOMES_PATH)
    logger.info(f"  {len(genome_index.all_genomes):,} genomes, "
                f"{len(genome_index.phylum_list)} phyla in {time.perf_counter()-t0:.1f}s")

    _genome_index_ref = genome_index

    # Create temp HDF5
    create_temp_hdf5(logger)

    # Create worker pool
    pool = mp.Pool(processes=n_workers)

    try:
        # Part A: batches 0-9
        logger.info("\n" + "=" * 70)
        logger.info("PART A: 100K pure complete samples (100% completeness, 0% contamination)")
        logger.info("=" * 70)

        for batch_idx in range(N_BATCHES_A):
            batch_id = batch_idx  # 0-9
            if checkpoint.is_completed(batch_id):
                logger.info(f"  Part A Batch {batch_id}: already done, skipping")
                continue

            n_rem = sum(1 for b in range(batch_idx, N_TOTAL_BATCHES)
                        if not checkpoint.is_completed(b))
            bt = [float(v) for v in checkpoint.batch_times.values()]
            avg_t = sum(bt) / len(bt) if bt else 0
            logger.info(f"\nPart A Batch {batch_id}/{N_BATCHES_A-1} "
                        f"[{n_rem} total remaining, ETA ~{avg_t*n_rem/60:.0f} min]")

            try:
                result = process_batch_part_a(batch_id, genome_index, pool, n_workers, logger)
                write_batch_to_temp(result, 'A', logger)
                checkpoint.mark_completed(batch_id, result['elapsed'])

                # Stats
                comp = result['labels'][:, 0]
                contam = result['labels'][:, 1]
                logger.info(f"    Comp: [{comp.min():.1f}, {comp.max():.1f}]%, "
                            f"mean={comp.mean():.1f}%")
                logger.info(f"    Contam: [{contam.min():.1f}, {contam.max():.1f}]%, "
                            f"mean={contam.mean():.1f}%")

                del result
                gc.collect()
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Part A Batch {batch_id} FAILED: {e}\n{tb}")
                checkpoint.mark_error(batch_id, str(e))

        # Part B: batches 10-19
        logger.info("\n" + "=" * 70)
        logger.info("PART B: 100K complete samples with 0-10% contamination")
        logger.info("=" * 70)

        for batch_idx in range(N_BATCHES_B):
            batch_id = N_BATCHES_A + batch_idx  # 10-19
            if checkpoint.is_completed(batch_id):
                logger.info(f"  Part B Batch {batch_id}: already done, skipping")
                continue

            n_rem = sum(1 for b in range(batch_id, N_TOTAL_BATCHES)
                        if not checkpoint.is_completed(b))
            bt = [float(v) for v in checkpoint.batch_times.values()]
            avg_t = sum(bt) / len(bt) if bt else 0
            logger.info(f"\nPart B Batch {batch_id-N_BATCHES_A}/{N_BATCHES_B-1} "
                        f"(global {batch_id}) "
                        f"[{n_rem} remaining, ETA ~{avg_t*n_rem/60:.0f} min]")

            try:
                result = process_batch_part_b(batch_id, genome_index, pool, n_workers, logger)
                write_batch_to_temp(result, 'B', logger)
                checkpoint.mark_completed(batch_id, result['elapsed'])

                # Stats
                comp = result['labels'][:, 0]
                contam = result['labels'][:, 1]
                logger.info(f"    Comp: [{comp.min():.1f}, {comp.max():.1f}]%, "
                            f"mean={comp.mean():.1f}%")
                logger.info(f"    Contam: [{contam.min():.1f}, {contam.max():.1f}]%, "
                            f"mean={contam.mean():.1f}%")

                del result
                gc.collect()
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Part B Batch {batch_id} FAILED: {e}\n{tb}")
                checkpoint.mark_error(batch_id, str(e))

    finally:
        pool.terminate()
        pool.join()

    # Verify temp HDF5
    all_gen_done = all(checkpoint.is_completed(b) for b in range(N_TOTAL_BATCHES))
    if all_gen_done:
        checkpoint.mark_phase('generate', 'complete')
        logger.info("\nAll 200K samples generated successfully!")
    else:
        missing = [b for b in range(N_TOTAL_BATCHES) if not checkpoint.is_completed(b)]
        logger.warning(f"\nGeneration incomplete! Missing batches: {missing}")

    return all_gen_done


# ============================================================================
# Phase 2: Merge into V5 HDF5
# ============================================================================
def merge_into_v5(logger, checkpoint):
    """Create V5 HDF5 by copying V4 data + appending new 200K samples."""
    logger.info("\n" + "=" * 70)
    logger.info("MERGING: Creating V5 HDF5")
    logger.info("=" * 70)

    # Check disk space
    stat = os.statvfs(os.path.dirname(V5_HDF5_PATH))
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    logger.info(f"  Available disk space: {free_gb:.1f} GB")
    if free_gb < 60:
        logger.error(f"  Insufficient disk space! Need ~60 GB, have {free_gb:.1f} GB")
        logger.error("  Consider removing old files to free space.")
        return False

    # Load V4 normalization params
    logger.info("  Loading V4 normalization parameters...")
    normalizer = FeatureNormalizer.load(NORM_PARAMS_PATH)
    assert normalizer.finalized, "Normalization params not finalized!"
    logger.info(f"  Normalization loaded: {normalizer.n_kmer_features} k-mer, "
                f"{normalizer.n_assembly_features} assembly features")

    # Create V5 HDF5 with preallocated structure
    if os.path.exists(V5_HDF5_PATH):
        # Check if partial merge exists
        with h5py.File(V5_HDF5_PATH, 'r') as f:
            if 'train' in f and int(f['train'].attrs.get('n_written', 0)) >= V5_TRAIN:
                logger.info("  V5 HDF5 already complete!")
                checkpoint.mark_phase('merge', 'complete')
                return True
        logger.info("  Removing incomplete V5 HDF5...")
        os.remove(V5_HDF5_PATH)

    logger.info(f"  Creating V5 HDF5 at {V5_HDF5_PATH}")
    logger.info(f"  Structure: train={V5_TRAIN:,}, val={V5_VAL:,}, test={V5_TEST:,}")

    chunk_size = 10_000
    with h5py.File(V5_HDF5_PATH, 'w') as f:
        for split_name, n_samples in [('train', V5_TRAIN), ('val', V5_VAL), ('test', V5_TEST)]:
            grp = f.create_group(split_name)
            grp.create_dataset('kmer_features', shape=(n_samples, N_KMER_FEATURES),
                               dtype=np.float32, chunks=(chunk_size, N_KMER_FEATURES),
                               compression='gzip', compression_opts=1, fillvalue=0.0)
            grp.create_dataset('assembly_features', shape=(n_samples, N_ASSEMBLY_FEATURES_CONST),
                               dtype=np.float32, chunks=(chunk_size, N_ASSEMBLY_FEATURES_CONST),
                               compression='gzip', compression_opts=1, fillvalue=0.0)
            grp.create_dataset('labels', shape=(n_samples, 2), dtype=np.float32,
                               chunks=(chunk_size, 2), compression='gzip', compression_opts=1)
            grp.create_dataset('metadata', shape=(n_samples,), dtype=METADATA_DTYPE,
                               chunks=(chunk_size,), compression='gzip', compression_opts=1)
            grp.attrs['n_written'] = 0
            grp.attrs['n_total'] = n_samples
            logger.info(f"    {split_name}: {n_samples:,} samples preallocated")

        f.attrs['n_kmer_features'] = N_KMER_FEATURES
        f.attrs['n_assembly_features'] = N_ASSEMBLY_FEATURES_CONST
        f.attrs['n_total_features'] = N_KMER_FEATURES + N_ASSEMBLY_FEATURES_CONST
        f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['version'] = 'v5'
        f.attrs['description'] = 'V5: V4 data (800K train) + 200K new complete/low-contam samples'

    # Step 1: Copy val and test from V4 (unchanged)
    for split_name, n_total in [('val', V4_VAL), ('test', V4_TEST)]:
        logger.info(f"  Copying {split_name} from V4 ({n_total:,} samples)...")
        t0 = time.perf_counter()
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            with h5py.File(V4_HDF5_PATH, 'r') as v4:
                kmer = v4[split_name]['kmer_features'][start:end]
                asm = v4[split_name]['assembly_features'][start:end]
                labels = v4[split_name]['labels'][start:end]
                meta = v4[split_name]['metadata'][start:end]
            with h5py.File(V5_HDF5_PATH, 'a') as v5:
                v5[split_name]['kmer_features'][start:end] = kmer
                v5[split_name]['assembly_features'][start:end] = asm
                v5[split_name]['labels'][start:end] = labels
                v5[split_name]['metadata'][start:end] = meta
                v5[split_name].attrs['n_written'] = end
                v5.flush()
        elapsed = time.perf_counter() - t0
        logger.info(f"    {split_name} copied in {elapsed:.1f}s")

    # Step 2: Copy V4 train data (first 800K)
    logger.info(f"  Copying V4 train data ({V4_TRAIN:,} samples)...")
    t0 = time.perf_counter()
    for start in range(0, V4_TRAIN, chunk_size):
        end = min(start + chunk_size, V4_TRAIN)
        with h5py.File(V4_HDF5_PATH, 'r') as v4:
            kmer = v4['train']['kmer_features'][start:end]
            asm = v4['train']['assembly_features'][start:end]
            labels = v4['train']['labels'][start:end]
            meta = v4['train']['metadata'][start:end]
        with h5py.File(V5_HDF5_PATH, 'a') as v5:
            v5['train']['kmer_features'][start:end] = kmer
            v5['train']['assembly_features'][start:end] = asm
            v5['train']['labels'][start:end] = labels
            v5['train']['metadata'][start:end] = meta
            v5['train'].attrs['n_written'] = end
            v5.flush()
        if (start // chunk_size) % 20 == 0:
            logger.info(f"    V4 train: {end:,}/{V4_TRAIN:,}")
    elapsed = time.perf_counter() - t0
    logger.info(f"    V4 train copied in {elapsed:.1f}s")

    # Step 3: Append new 200K samples (normalize + write)
    logger.info(f"  Appending 200K new samples to train...")
    t0 = time.perf_counter()
    write_offset = V4_TRAIN  # Start writing at 800K

    for part_name, n_part in [('part_a', N_PART_A), ('part_b', N_PART_B)]:
        logger.info(f"    Processing {part_name} ({n_part:,} samples)...")
        for start in range(0, n_part, chunk_size):
            end = min(start + chunk_size, n_part)
            with h5py.File(TEMP_HDF5_PATH, 'r') as tmp:
                kmer_raw = tmp[part_name]['kmer_features'][start:end]
                asm_raw = tmp[part_name]['assembly_features'][start:end]
                labels = tmp[part_name]['labels'][start:end]
                meta = tmp[part_name]['metadata'][start:end]

            # Normalize new samples using V4 normalization params
            kmer_norm = normalizer.normalize_kmer(kmer_raw).astype(np.float32)
            asm_norm = normalizer.normalize_assembly(asm_raw).astype(np.float32)

            dest_start = write_offset
            dest_end = write_offset + (end - start)

            with h5py.File(V5_HDF5_PATH, 'a') as v5:
                v5['train']['kmer_features'][dest_start:dest_end] = kmer_norm
                v5['train']['assembly_features'][dest_start:dest_end] = asm_norm
                v5['train']['labels'][dest_start:dest_end] = labels
                v5['train']['metadata'][dest_start:dest_end] = meta
                v5['train'].attrs['n_written'] = dest_end
                v5.flush()

            write_offset = dest_end

        logger.info(f"    {part_name} appended, write_offset = {write_offset:,}")

    elapsed = time.perf_counter() - t0
    logger.info(f"  New 200K appended in {elapsed:.1f}s")
    logger.info(f"  Total train samples written: {write_offset:,}")

    checkpoint.mark_phase('merge', 'complete')
    logger.info("  V5 merge complete!")
    return True


# ============================================================================
# Phase 3: Verification
# ============================================================================
def verify_v5(logger):
    """Verify V5 HDF5 integrity and sample distributions."""
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)

    # Check file exists and sizes
    if not os.path.exists(V5_HDF5_PATH):
        logger.error("V5 HDF5 not found!")
        return False

    v5_size_gb = os.path.getsize(V5_HDF5_PATH) / 1e9
    logger.info(f"  V5 file size: {v5_size_gb:.1f} GB")

    with h5py.File(V5_HDF5_PATH, 'r') as f:
        for split in ['train', 'val', 'test']:
            grp = f[split]
            n_written = int(grp.attrs['n_written'])
            n_total = int(grp.attrs['n_total'])
            kmer_shape = grp['kmer_features'].shape
            asm_shape = grp['assembly_features'].shape
            labels_shape = grp['labels'].shape
            logger.info(f"\n  {split}:")
            logger.info(f"    n_written={n_written:,}, n_total={n_total:,}")
            logger.info(f"    kmer_features shape: {kmer_shape}")
            logger.info(f"    assembly_features shape: {asm_shape}")
            logger.info(f"    labels shape: {labels_shape}")

            if split == 'train':
                assert kmer_shape == (V5_TRAIN, N_KMER_FEATURES), \
                    f"Train kmer shape mismatch: {kmer_shape}"
                assert asm_shape == (V5_TRAIN, N_ASSEMBLY_FEATURES_CONST), \
                    f"Train assembly shape mismatch: {asm_shape}"
                assert n_written == V5_TRAIN, \
                    f"Train not fully written: {n_written}/{V5_TRAIN}"
            elif split == 'val':
                assert kmer_shape == (V5_VAL, N_KMER_FEATURES)
                assert asm_shape == (V5_VAL, N_ASSEMBLY_FEATURES_CONST)
            elif split == 'test':
                assert kmer_shape == (V5_TEST, N_KMER_FEATURES)
                assert asm_shape == (V5_TEST, N_ASSEMBLY_FEATURES_CONST)

    # Verify the new 200K samples specifically
    logger.info("\n  --- Verifying new 200K samples (train[800000:1000000]) ---")

    with h5py.File(V5_HDF5_PATH, 'r') as f:
        # Part A: indices 800000-899999
        part_a_start = V4_TRAIN
        part_a_end = V4_TRAIN + N_PART_A
        logger.info(f"\n  Part A (train[{part_a_start}:{part_a_end}]):")

        # Read in chunks to verify
        a_comp_all = []
        a_contam_all = []
        a_kmer_nonzero = 0
        a_asm_nonzero = 0
        for s in range(part_a_start, part_a_end, 10000):
            e = min(s + 10000, part_a_end)
            labels = f['train']['labels'][s:e]
            kmer = f['train']['kmer_features'][s:e]
            asm = f['train']['assembly_features'][s:e]
            a_comp_all.append(labels[:, 0])
            a_contam_all.append(labels[:, 1])
            a_kmer_nonzero += np.count_nonzero(np.any(kmer != 0, axis=1))
            a_asm_nonzero += np.count_nonzero(np.any(asm != 0, axis=1))

        a_comp = np.concatenate(a_comp_all)
        a_contam = np.concatenate(a_contam_all)

        logger.info(f"    Completeness: min={a_comp.min():.2f}, max={a_comp.max():.2f}, "
                    f"mean={a_comp.mean():.2f}")
        logger.info(f"    Contamination: min={a_contam.min():.2f}, max={a_contam.max():.2f}, "
                    f"mean={a_contam.mean():.2f}")
        logger.info(f"    Samples with non-zero kmer features: {a_kmer_nonzero:,}/{N_PART_A:,}")
        logger.info(f"    Samples with non-zero assembly features: {a_asm_nonzero:,}/{N_PART_A:,}")

        # Verify Part A requirements
        n_comp_100 = np.sum(a_comp == 100.0)
        n_contam_0 = np.sum(a_contam == 0.0)
        logger.info(f"    Part A CHECK - completeness==100%: {n_comp_100:,}/{N_PART_A:,} "
                    f"{'PASS' if n_comp_100 == N_PART_A else 'FAIL'}")
        logger.info(f"    Part A CHECK - contamination==0%: {n_contam_0:,}/{N_PART_A:,} "
                    f"{'PASS' if n_contam_0 == N_PART_A else 'FAIL'}")

        # Part B: indices 900000-999999
        part_b_start = V4_TRAIN + N_PART_A
        part_b_end = V4_TRAIN + N_PART_A + N_PART_B
        logger.info(f"\n  Part B (train[{part_b_start}:{part_b_end}]):")

        b_comp_all = []
        b_contam_all = []
        b_kmer_nonzero = 0
        b_asm_nonzero = 0
        for s in range(part_b_start, part_b_end, 10000):
            e = min(s + 10000, part_b_end)
            labels = f['train']['labels'][s:e]
            kmer = f['train']['kmer_features'][s:e]
            asm = f['train']['assembly_features'][s:e]
            b_comp_all.append(labels[:, 0])
            b_contam_all.append(labels[:, 1])
            b_kmer_nonzero += np.count_nonzero(np.any(kmer != 0, axis=1))
            b_asm_nonzero += np.count_nonzero(np.any(asm != 0, axis=1))

        b_comp = np.concatenate(b_comp_all)
        b_contam = np.concatenate(b_contam_all)

        logger.info(f"    Completeness: min={b_comp.min():.2f}, max={b_comp.max():.2f}, "
                    f"mean={b_comp.mean():.2f}")
        logger.info(f"    Contamination: min={b_contam.min():.2f}, max={b_contam.max():.2f}, "
                    f"mean={b_contam.mean():.2f}")
        logger.info(f"    Samples with non-zero kmer features: {b_kmer_nonzero:,}/{N_PART_B:,}")
        logger.info(f"    Samples with non-zero assembly features: {b_asm_nonzero:,}/{N_PART_B:,}")

        # Verify Part B requirements
        n_comp_100_b = np.sum(b_comp == 100.0)
        n_contam_le10 = np.sum(b_contam <= 10.0)
        n_contam_ge0 = np.sum(b_contam >= 0.0)
        logger.info(f"    Part B CHECK - completeness==100%: {n_comp_100_b:,}/{N_PART_B:,} "
                    f"{'PASS' if n_comp_100_b == N_PART_B else 'FAIL'}")
        logger.info(f"    Part B CHECK - contamination in [0%,10%]: "
                    f"{min(n_contam_le10, n_contam_ge0):,}/{N_PART_B:,} "
                    f"{'PASS' if n_contam_le10 == N_PART_B and n_contam_ge0 == N_PART_B else 'FAIL'}")

        # Contamination distribution for Part B
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        hist, _ = np.histogram(b_contam, bins=bins)
        logger.info(f"    Part B contamination distribution:")
        for i in range(len(bins) - 1):
            logger.info(f"      [{bins[i]}-{bins[i+1]}%): {hist[i]:,}")

    # Verify overall train split
    logger.info(f"\n  --- Overall train split statistics ---")
    with h5py.File(V5_HDF5_PATH, 'r') as f:
        # Sample from V4 portion and new portion
        for region, start, end in [
            ("V4 original (0-800K)", 0, V4_TRAIN),
            ("V5 new Part A (800K-900K)", V4_TRAIN, V4_TRAIN + N_PART_A),
            ("V5 new Part B (900K-1M)", V4_TRAIN + N_PART_A, V5_TRAIN),
        ]:
            labels = f['train']['labels'][start:end]
            comp = labels[:, 0]
            contam = labels[:, 1]
            logger.info(f"  {region}:")
            logger.info(f"    N={end-start:,}")
            logger.info(f"    Completeness: min={comp.min():.2f}, max={comp.max():.2f}, "
                        f"mean={comp.mean():.2f}, std={comp.std():.2f}")
            logger.info(f"    Contamination: min={contam.min():.2f}, max={contam.max():.2f}, "
                        f"mean={contam.mean():.2f}, std={contam.std():.2f}")

    # Feature shape verification
    logger.info(f"\n  --- Feature shape verification ---")
    with h5py.File(V5_HDF5_PATH, 'r') as f:
        kmer_shape = f['train']['kmer_features'].shape
        asm_shape = f['train']['assembly_features'].shape
        logger.info(f"  Train kmer_features: {kmer_shape} "
                    f"{'PASS' if kmer_shape == (V5_TRAIN, N_KMER_FEATURES) else 'FAIL'}")
        logger.info(f"  Train assembly_features: {asm_shape} "
                    f"{'PASS' if asm_shape == (V5_TRAIN, N_ASSEMBLY_FEATURES_CONST) else 'FAIL'}")

    # Check NaN/Inf
    logger.info(f"\n  --- NaN/Inf check ---")
    n_nan = 0
    n_inf = 0
    with h5py.File(V5_HDF5_PATH, 'r') as f:
        for s in range(V4_TRAIN, V5_TRAIN, 10000):
            e = min(s + 10000, V5_TRAIN)
            kmer = f['train']['kmer_features'][s:e]
            asm = f['train']['assembly_features'][s:e]
            n_nan += np.isnan(kmer).sum() + np.isnan(asm).sum()
            n_inf += np.isinf(kmer).sum() + np.isinf(asm).sum()
    logger.info(f"  NaN count in new 200K: {n_nan} {'PASS' if n_nan == 0 else 'FAIL'}")
    logger.info(f"  Inf count in new 200K: {n_inf} {'PASS' if n_inf == 0 else 'FAIL'}")

    # Final disk usage
    v5_size = os.path.getsize(V5_HDF5_PATH) / 1e9
    stat = os.statvfs(os.path.dirname(V5_HDF5_PATH))
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    logger.info(f"\n  V5 file size: {v5_size:.1f} GB")
    logger.info(f"  Available disk space: {free_gb:.1f} GB")

    logger.info("\n  VERIFICATION COMPLETE")
    return True


# ============================================================================
# Also verify temp HDF5 before merge
# ============================================================================
def verify_temp_hdf5(logger):
    """Verify the temp HDF5 before merging."""
    logger.info("\n  --- Verifying temp HDF5 ---")
    if not os.path.exists(TEMP_HDF5_PATH):
        logger.error("  Temp HDF5 not found!")
        return False

    with h5py.File(TEMP_HDF5_PATH, 'r') as f:
        for part in ['part_a', 'part_b']:
            grp = f[part]
            n_written = int(grp.attrs['n_written'])
            n_total = int(grp.attrs['n_total'])
            logger.info(f"  {part}: {n_written:,}/{n_total:,} written")

            if n_written < n_total:
                logger.error(f"  {part} not fully written!")
                return False

            # Read labels
            labels = grp['labels'][:]
            comp = labels[:, 0]
            contam = labels[:, 1]
            logger.info(f"    Completeness: [{comp.min():.2f}, {comp.max():.2f}], "
                        f"mean={comp.mean():.2f}")
            logger.info(f"    Contamination: [{contam.min():.2f}, {contam.max():.2f}], "
                        f"mean={contam.mean():.2f}")

            if part == 'part_a':
                assert np.all(comp == 100.0), f"Part A completeness not all 100%!"
                assert np.all(contam == 0.0), f"Part A contamination not all 0%!"
                logger.info(f"    Part A: ALL completeness=100%, ALL contamination=0% PASS")
            elif part == 'part_b':
                assert np.all(comp == 100.0), f"Part B completeness not all 100%!"
                assert np.all(contam >= 0.0), f"Part B contamination has negatives!"
                assert np.all(contam <= 10.0), f"Part B contamination exceeds 10%!"
                logger.info(f"    Part B: ALL completeness=100%, ALL contamination in [0,10]% PASS")

    logger.info("  Temp HDF5 verification PASSED")
    return True


# ============================================================================
# Main
# ============================================================================
def main():
    global _genome_index_ref

    parser = argparse.ArgumentParser(description='Generate V5 training data')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: auto)')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['cleanup', 'generate', 'merge', 'verify', 'all'],
                        help='Which phase to run')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("MAGICC V5 Training Data Generation")
    logger.info("=" * 70)

    n_cpus = mp.cpu_count()
    n_workers = args.workers if args.workers else min(43, max(1, n_cpus - 5))
    logger.info(f"System: {n_cpus} CPUs, using {n_workers} workers")

    # Initialize k-mer globals
    logger.info("Loading k-mer index and warming up Numba JIT...")
    t0 = time.perf_counter()
    init_kmer_globals()
    logger.info(f"  Done in {time.perf_counter()-t0:.1f}s")

    checkpoint = CheckpointManager(CHECKPOINT_PATH)
    logger.info(f"Checkpoint: {len(checkpoint.completed_batches)} batches done, "
                f"phases: {checkpoint.phase_complete}")

    total_start = time.perf_counter()

    # Phase 0: Cleanup disk space
    if args.phase in ('cleanup', 'all'):
        logger.info("\n--- PHASE 0: Disk Space Cleanup ---")
        cleanup_disk_space(logger)

    # Phase 1: Generate 200K new samples
    if args.phase in ('generate', 'all'):
        if checkpoint.is_phase_complete('generate'):
            logger.info("\nGeneration phase already complete, skipping")
        else:
            logger.info("\n--- PHASE 1: Generate 200K New Samples ---")
            success = generate_all_samples(n_workers, logger, checkpoint)
            if not success:
                logger.error("Generation incomplete! Re-run to resume.")
                if args.phase == 'all':
                    logger.info("Continuing to check if enough data for merge...")

    # Phase 2: Merge into V5
    if args.phase in ('merge', 'all'):
        if checkpoint.is_phase_complete('merge'):
            logger.info("\nMerge phase already complete, skipping")
        else:
            # Verify temp first
            if not checkpoint.is_phase_complete('generate'):
                # Check if temp HDF5 is actually complete even if checkpoint doesn't say so
                try:
                    with h5py.File(TEMP_HDF5_PATH, 'r') as f:
                        pa_ok = int(f['part_a'].attrs.get('n_written', 0)) == N_PART_A
                        pb_ok = int(f['part_b'].attrs.get('n_written', 0)) == N_PART_B
                    if pa_ok and pb_ok:
                        logger.info("Temp HDF5 is complete (checkpoint may be stale)")
                        checkpoint.mark_phase('generate', 'complete')
                    else:
                        logger.error("Generation not complete! Cannot merge.")
                        return
                except Exception:
                    logger.error("Cannot read temp HDF5! Generation must complete first.")
                    return

            logger.info("\n--- PHASE 2: Verify Temp and Merge into V5 ---")
            if not verify_temp_hdf5(logger):
                logger.error("Temp HDF5 verification failed! Cannot merge.")
                return

            success = merge_into_v5(logger, checkpoint)
            if not success:
                logger.error("Merge failed!")
                return

    # Phase 3: Verification
    if args.phase in ('verify', 'all'):
        logger.info("\n--- PHASE 3: Full Verification ---")
        verify_v5(logger)

    # Summary
    total_elapsed = time.perf_counter() - total_start
    logger.info("\n" + "=" * 70)
    logger.info("V5 GENERATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total time: {total_elapsed/3600:.2f} hours ({total_elapsed:.0f}s)")
    logger.info(f"  Phases complete: {checkpoint.phase_complete}")

    bt = [float(v) for v in checkpoint.batch_times.values()]
    if bt:
        logger.info(f"  Batch times: avg={sum(bt)/len(bt):.1f}s, "
                    f"min={min(bt):.1f}s, max={max(bt):.1f}s")

    if os.path.exists(V5_HDF5_PATH):
        logger.info(f"  V5 file: {V5_HDF5_PATH}")
        logger.info(f"  V5 size: {os.path.getsize(V5_HDF5_PATH)/1e9:.1f} GB")

    if os.path.exists(TEMP_HDF5_PATH):
        temp_size = os.path.getsize(TEMP_HDF5_PATH) / 1e9
        logger.info(f"  Temp file ({temp_size:.1f} GB) can be removed: {TEMP_HDF5_PATH}")

    stat = os.statvfs(os.path.dirname(V5_HDF5_PATH))
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    logger.info(f"  Available disk space: {free_gb:.1f} GB")

    if checkpoint.batch_errors:
        logger.warning(f"  Errors: {checkpoint.batch_errors}")

    logger.info("=" * 70)


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
