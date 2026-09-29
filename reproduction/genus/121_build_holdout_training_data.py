#!/usr/bin/env python3
"""
WS1.6 / WS1.9 step 2 - Build leave-TAXON-out synthetic training data.

ONE code path serves both experiments; the taxonomic level is selected by the
environment variable MAGICC_HOLDOUT_LEVEL (phylum | family, default phylum) and
every artifact path is level-suffixed - see scripts/holdout_lib/config.py.

  MAGICC_HOLDOUT_LEVEL=phylum  WS1.6 leave-phylum-out  (completed 2026-07-26)
  MAGICC_HOLDOUT_LEVEL=family  WS1.9 leave-family-out

WHY REGENERATION IS REQUIRED (verified by scripts 120 / 127)
  magicc/storage.py METADATA_DTYPE records only dominant_phylum /
  dominant_accession / n_contaminants. Contaminant taxonomy was NEVER stored, so
  data/features/magicc_v5_features.h5 cannot be filtered into a valid holdout:
  panel-phylum genomes appear as unlabelled CONTAMINANTS in a large fraction of
  samples (23.46% of V5 train samples have a panel-phylum DOMINANT; the
  contaminant-role fraction is unknown but necessarily much larger, since 60% of
  samples carry 1-5 contaminants drawn from the full training pool).
  The panel is therefore excluded from BOTH roles by regenerating from scratch.

RECIPE - matched to production V5 exactly except for the genome pool
  train  1,000,000 = 800,000 V4-recipe + 100,000 part A + 100,000 part B
  val      100,000   V4-recipe, drawn from the panel-free VAL genome pool
  test     100,000   V4-recipe, drawn from the panel-free TEST genome pool
  V4-recipe composition per 10,000-sample batch:
      pure 1500 / complete 1500 / within_phylum 3000 / cross_phylum 3000 /
      reduced_genome 500 / archaeal 500          (15/15/30/30/5/5 %)
  part A: 100% completeness, 0% contamination, original contigs
  part B: 100% completeness, 0-10% contamination (50% within- / 50% cross-phylum)
  completeness U[50,100]%, contamination U[0, completeness]% (V4/V5 constraint
  contaminant_bp <= dominant_actual_bp), quality tiers 15/35/35/15,
  7 k-mer summary features, log1p + z-score normalization.
  Sample count is IDENTICAL to V5 (1.2 M) so there is no dataset-size confound.

DOCUMENTED ADAPTATION (reduced-genome category) - PHYLUM LEVEL ONLY
  The PHYLUM panel removes Patescibacteriota and every DPANN phylum, collapsing
  the V5 reduced-genome pool from 1,453 to 103 genomes (Bdellovibrionota only).
  Using that pool would confound "never saw Patescibacteriota" with "never saw
  ANY small genome". The pool was therefore redefined as
      surviving V5 reduced-genome phyla  U  non-panel genomes < 1.5 Mbp
  = 1,622 genomes (median 1.17 Mbp vs V5's 1,453 genomes / 0.91 Mbp). The model
  still learns from small genomes; only the LINEAGES are withheld.

  AT FAMILY LEVEL NO ADAPTATION IS APPLIED. The family panel removes only 530 of
  1,286 Patescibacteriota training genomes, so the pool survives at 923/1,453 =
  63.5% (median 0.950 Mbp vs V5's 0.912) under V5's own definition, which is
  used verbatim. The surviving fraction is re-verified at run time against
  C.REDUCED_POOL_ADAPT_THRESHOLD and the job aborts rather than proceed silently
  if it has collapsed.

Normalization is estimated exactly as for V5: streaming Welford over the
V4-recipe portion only (800K train + 100K val + 100K test), then applied to all
splits including parts A and B.

Resumable: per-batch checkpoint; a killed/rebooted job resumes at the next batch.

Usage
  python scripts/121_build_holdout_training_data.py --workers 24
  python scripts/121_build_holdout_training_data.py --smoke        # 5k-sample E2E test
  python scripts/121_build_holdout_training_data.py --finalize-only
"""

import os
import sys

os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import gc
import json
import logging
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_lib import config as C
from holdout_lib.fragmentation import (read_fasta, load_original_contigs,
                                       _warm_numba_fragmentation)
from holdout_lib.contamination import (generate_contaminated_sample, generate_pure_sample,
                                       select_contaminant_target_bp, fragment_contaminant,
                                       compute_contamination_rate)
from holdout_lib.kmer_counter import (load_selected_kmers, build_kmer_index,
                                      _count_kmers_single, K)
from holdout_lib.assembly_stats import compute_assembly_stats
from holdout_lib.normalization import FeatureNormalizer
from holdout_lib.storage import METADATA_DTYPE

NK = C.N_KMER_FEATURES
NA = C.N_SUMMARY_FEATURES
BS = C.BATCH_SIZE

CKPT = C.HOLDOUT_DIR / 'build_checkpoint.json'
NORMSTATS = C.HOLDOUT_DIR / 'normalizer_running_stats.json'
LOG = C.LOGS_DIR / f'{C.WS}_build_holdout_data.log'

QUALITY_TIERS = list(C.QUALITY_TIER_WEIGHTS.keys())
QUALITY_WEIGHTS = np.array([C.QUALITY_TIER_WEIGHTS[t] for t in QUALITY_TIERS])
QUALITY_WEIGHTS = QUALITY_WEIGHTS / QUALITY_WEIGHTS.sum()

# Batch plan.  ('v4', i) matches V5's BATCH_ASSIGNMENTS 1:1 (0-79 train,
# 80-89 val, 90-99 test).  ('A', i) / ('B', i) match scripts/50 batches 0-9.
V4_ASSIGN = ([(i, 'train', i * BS) for i in range(80)] +
             [(80 + i, 'val', i * BS) for i in range(10)] +
             [(90 + i, 'test', i * BS) for i in range(10)])
A_ASSIGN = [(i, 'train', C.N_TRAIN_V4RECIPE + i * BS) for i in range(10)]
B_ASSIGN = [(i, 'train', C.N_TRAIN_V4RECIPE + C.N_TRAIN_PART_A + i * BS) for i in range(10)]


def setup_logging(smoke=False):
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = str(LOG).replace('.log', '_smoke.log') if smoke else str(LOG)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(path, mode='a'), logging.StreamHandler(sys.stdout)],
        force=True)
    return logging.getLogger('build_holdout')


# ===========================================================================
# Genome index over the PANEL-FREE pool
# ===========================================================================
class GenomeIndex:
    """Index of panel-free reference genomes. Panel taxa are absent entirely, so
    every draw (dominant, within-phylum contaminant, cross-phylum contaminant)
    is automatically panel-free.

    NOTE on within/cross_phylum contamination at FAMILY level: those two sample
    types are defined by PHYLUM (the V5 recipe) and stay that way. A
    within_phylum contaminant for a surviving Bacteroidota dominant is drawn
    from another surviving Bacteroidota family, which is correct - the panel
    families are absent from the pool entirely, in both roles."""

    def __init__(self, tsv_path, logger=None):
        self.all_genomes = []
        self.by_phylum = {}
        self.archaea_indices = None
        self.reduced_indices = None
        self.general_indices = None
        self.phylum_list = []
        # Cache the FASTA-existence resolution: 64k cold stat() calls on the HDD
        # take ~10 min, and a resumable job re-does it on every restart.
        cache = Path(str(tsv_path).replace('.tsv', '.fastacache.json'))
        self._cache = cache
        self._load(tsv_path, logger)

    def _load(self, tsv_path, logger):
        if self._cache.exists():
            d = json.loads(self._cache.read_text())
            self.all_genomes = d['all_genomes']
            self.by_phylum = {k: np.array(v, dtype=np.int64)
                              for k, v in d['by_phylum'].items()}
            self.archaea_indices = np.array(d['archaea'], dtype=np.int64)
            self.reduced_indices = np.array(d['reduced'], dtype=np.int64)
            self.general_indices = np.array(d['general'], dtype=np.int64)
            self.phylum_list = list(self.by_phylum.keys())
            if logger:
                logger.info(f'  {Path(tsv_path).name}: {len(self.all_genomes):,} genomes '
                            f'(from cache), {len(self.phylum_list)} phyla, '
                            f'{len(self.archaea_indices)} archaeal, '
                            f'{len(self.reduced_indices)} reduced/small')
            return
        self._build(tsv_path, logger)
        self._cache.write_text(json.dumps({
            'all_genomes': self.all_genomes,
            'by_phylum': {k: v.tolist() for k, v in self.by_phylum.items()},
            'archaea': self.archaea_indices.tolist(),
            'reduced': self.reduced_indices.tolist(),
            'general': self.general_indices.tolist()}))

    def _build(self, tsv_path, logger):
        t0 = time.perf_counter()
        surviving_reduced = C.V5_REDUCED_GENOME_PHYLA - set(C.PANEL_PHYLA)
        panel = set(C.PANEL_TAXA)
        rows = []
        with open(tsv_path) as f:
            hdr = f.readline().rstrip('\n').split('\t')
            ci = {n: i for i, n in enumerate(hdr)}
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) < len(hdr):
                    continue
                phylum = p[ci['phylum']]
                # hard guarantee: no panel taxon may be present in ANY role
                taxon = (p[ci[C.TAXON_COL]] if C.TAXON_COL in ci else
                         C.rank_from_taxonomy(p[ci['gtdb_taxonomy']], C.TAXON_COL))
                if taxon in panel:
                    raise RuntimeError(f'panel {C.PANEL_LEVEL} {taxon} present '
                                       f'in {tsv_path}')
                rows.append({'accession': p[ci['ncbi_accession']],
                             'fasta_path': C.remap_fasta_path(p[ci['fasta_path']]),
                             'phylum': phylum, 'domain': p[ci['domain']],
                             'genome_size': int(p[ci['genome_size']])})
        # Parallel existence check (cold HDD stat() is the bottleneck)
        with ThreadPoolExecutor(max_workers=32) as ex:
            exists = list(ex.map(os.path.exists, [r['fasta_path'] for r in rows]))
        byp, arch, red, gen = {}, [], [], []
        n_missing = 0
        for r, ok in zip(rows, exists):
            if not ok:
                n_missing += 1
                continue
            idx = len(self.all_genomes)
            self.all_genomes.append(r)
            byp.setdefault(r['phylum'], []).append(idx)
            if r['domain'] == 'Archaea':
                arch.append(idx)
            # Reduced-genome pool. At phylum level the WS1.6 ADAPTATION applies
            # (surviving reduced phyla U any genome < 1.5 Mbp) because the panel
            # collapses the pool; at family level V5's own definition survives
            # intact and is used verbatim (see module docstring).
            if r['phylum'] in surviving_reduced or (
                    C.USE_ADAPTED_REDUCED_POOL and
                    r['genome_size'] < C.ADAPTED_REDUCED_SIZE_CUTOFF):
                red.append(idx)
            gen.append(idx)
        self.by_phylum = {k: np.array(v, dtype=np.int64) for k, v in byp.items()}
        self.archaea_indices = np.array(arch, dtype=np.int64)
        self.reduced_indices = np.array(red, dtype=np.int64)
        self.general_indices = np.array(gen, dtype=np.int64)
        self.phylum_list = list(self.by_phylum.keys())
        if logger:
            logger.info(f'  {Path(tsv_path).name}: {len(self.all_genomes):,} genomes '
                        f'({n_missing} FASTA missing), {len(self.phylum_list)} phyla, '
                        f'{len(arch)} archaeal, {len(red)} reduced/small '
                        f'[{time.perf_counter()-t0:.0f}s]')


_kmer_index = None
_nk = None
_gi = None


def init_kmer_globals():
    global _kmer_index, _nk
    codes = load_selected_kmers(str(C.SELECTED_KMERS))
    _kmer_index = build_kmer_index(codes)
    _nk = len(codes)
    _count_kmers_single(np.frombuffer(b'ACGTACGTACGTACGTACGT', dtype=np.uint8),
                        _kmer_index, _nk, K)
    _warm_numba_fragmentation()
    return _nk


def warm_page_cache(paths, n_threads, logger=None):
    t0 = time.perf_counter()

    def rd(p):
        try:
            with open(p, 'rb') as f:
                f.read()
            return True
        except Exception:
            return False
    ok = sum(1 for r in ThreadPoolExecutor(max_workers=n_threads).map(rd, paths) if r)
    el = time.perf_counter() - t0
    if logger:
        logger.info(f'    page cache: {ok}/{len(paths)} files in {el:.1f}s')
    return el


def _count(contigs):
    tot = np.zeros(_nk, dtype=np.int64)
    for c in contigs:
        if len(c) >= K:
            tot += _count_kmers_single(np.frombuffer(c.encode('ascii'), dtype=np.uint8),
                                       _kmer_index, _nk, K)
    return tot


def _trim_contaminants(cc, max_bp, rng):
    """Cap total contaminant bp at max_bp (verbatim V5 logic)."""
    tot = sum(len(c) for c in cc)
    if tot <= max_bp or not cc:
        return cc
    order = np.arange(len(cc))
    rng.shuffle(order)
    kept, kbp = [], 0
    for i in order:
        c = cc[i]
        if kbp + len(c) <= max_bp:
            kept.append(c)
            kbp += len(c)
        else:
            rem = max_bp - kbp
            if rem >= 500:
                kept.append(c[:rem])
            break
    return kept if kept else cc


def _add_contamination(dom_contigs, dom_len, cont_seqs, target_contam, rng):
    """V5 'complete'/'part B' contamination path."""
    if not cont_seqs or target_contam <= 0:
        return dom_contigs, 0.0
    tot_bp = select_contaminant_target_bp(target_contam, dom_len)
    if tot_bp <= 0:
        return dom_contigs, 0.0
    n = len(cont_seqs)
    if n == 1:
        per = [tot_bp]
    else:
        pr = rng.dirichlet(np.ones(n))
        per = (pr * tot_bp).astype(int)
        per[-1] = tot_bp - per[:-1].sum()
    cc = []
    for seq, tbp in zip(cont_seqs, per):
        if tbp <= 0 or not seq:
            continue
        cc.extend(fragment_contaminant(seq, tbp, rng)['contigs'])
    cc = _trim_contaminants(cc, int(target_contam / 100.0 * dom_len), rng)
    actual = compute_contamination_rate(sum(len(c) for c in cc), dom_len)
    out = list(dom_contigs) + cc
    order = np.arange(len(out))
    rng.shuffle(order)
    return [out[i] for i in order], actual


# ===========================================================================
# Workers
# ===========================================================================
def work_v4(args):
    """V4-recipe sample (pure/complete/within/cross/reduced/archaeal)."""
    (sid, stype, di, cis, tcomp, tcontam, qt, seed) = args
    rng = np.random.default_rng(seed)
    try:
        dom = _gi.all_genomes[di]
        cont_seqs = []
        for ci in (cis or []):
            try:
                s = read_fasta(_gi.all_genomes[ci]['fasta_path'])
                if s and len(s) >= 500:
                    cont_seqs.append(s)
            except Exception:
                pass

        if stype == 'complete':
            contigs = load_original_contigs(dom['fasta_path'])
            if not contigs:
                return None
            dlen = sum(len(c) for c in contigs)
            if dlen < 500:
                return None
            contigs, actual_contam = _add_contamination(contigs, dlen, cont_seqs,
                                                        tcontam, rng)
            actual_comp, used_qt = 100.0, 'high'
        else:
            dseq = read_fasta(dom['fasta_path'])
            if dseq is None or len(dseq) < 500:
                return None
            if stype == 'pure':
                smp = generate_pure_sample(dominant_sequence=dseq,
                                           target_completeness=tcomp, rng=rng,
                                           quality_tier=qt)
            else:
                smp = generate_contaminated_sample(
                    dominant_sequence=dseq, contaminant_sequences=cont_seqs,
                    target_completeness=tcomp, target_contamination=tcontam,
                    rng=rng, dominant_quality_tier=qt)
            contigs = smp['contigs']
            if not contigs:
                return None
            actual_comp = smp['completeness'] * 100.0
            actual_contam = smp['contamination']
            dlen = smp['dominant_full_length']
            used_qt = smp.get('dominant_quality_tier', qt)

        cnt = _count(contigs)
        tot = cnt.sum()
        asm = compute_assembly_stats(np.log10(float(tot)) if tot > 0 else 0.0, cnt)
        return (sid, cnt, asm, actual_comp, actual_contam, dom['phylum'], stype,
                used_qt, dom['accession'], dlen, len(cont_seqs))
    except Exception:
        return None


def work_A(args):
    """part A: 100% completeness, 0% contamination, original contigs."""
    sid, di, seed = args
    try:
        info = _gi.all_genomes[di]
        contigs = load_original_contigs(info['fasta_path'])
        if not contigs:
            return None
        dlen = sum(len(c) for c in contigs)
        if dlen < 500:
            return None
        cnt = _count(contigs)
        tot = cnt.sum()
        asm = compute_assembly_stats(np.log10(float(tot)) if tot > 0 else 0.0, cnt)
        return (sid, cnt, asm, 100.0, 0.0, info['phylum'], 'v5_pure_complete', 'high',
                info['accession'], dlen, 0)
    except Exception:
        return None


def work_B(args):
    """part B: 100% completeness, 0-10% contamination."""
    sid, di, cis, tcontam, ctype, seed = args
    rng = np.random.default_rng(seed)
    try:
        info = _gi.all_genomes[di]
        dom_contigs = load_original_contigs(info['fasta_path'])
        if not dom_contigs:
            return None
        dlen = sum(len(c) for c in dom_contigs)
        if dlen < 500:
            return None
        cont_seqs = []
        for ci in (cis or []):
            try:
                s = read_fasta(_gi.all_genomes[ci]['fasta_path'])
                if s and len(s) >= 500:
                    cont_seqs.append(s)
            except Exception:
                pass
        contigs, actual = _add_contamination(dom_contigs, dlen, cont_seqs, tcontam, rng)
        cnt = _count(contigs)
        tot = cnt.sum()
        asm = compute_assembly_stats(np.log10(float(tot)) if tot > 0 else 0.0, cnt)
        return (sid, cnt, asm, 100.0, actual, info['phylum'], f'v5_{ctype}', 'high',
                info['accession'], dlen, len(cis or []))
    except Exception:
        return None


# ===========================================================================
# Planners (seeding formulas identical to V5 scripts 19_v2 / 50)
# ===========================================================================
class V4Planner:
    def __init__(self, gi, batch_id, n_per_type=None):
        self.gi = gi
        self.bid = batch_id
        self.rng = np.random.default_rng(batch_id * 10000)
        self.types = n_per_type or C.SAMPLE_TYPES

    def plan(self):
        samples, needed, sid = [], set(), 0
        for stype, n in self.types.items():
            for _ in range(n):
                p = self._one(sid, stype)
                if p is not None:
                    samples.append(p)
                    needed.add(p[2])
                    if p[3]:
                        needed.update(p[3])
                sid += 1
        self.rng.shuffle(samples)
        return [(i,) + s[1:] for i, s in enumerate(samples)], needed

    def _pick(self, pool):
        if pool is None or len(pool) == 0:
            return None
        return int(pool[self.rng.integers(0, len(pool))])

    def _one(self, sid, stype):
        seed = self.bid * 10000 + sid
        qt = self.rng.choice(QUALITY_TIERS, p=QUALITY_WEIGHTS)
        if stype == 'pure':
            di = self._pick(self.gi.general_indices)
            return None if di is None else (sid, 'pure', di, None,
                                            float(self.rng.uniform(0.5, 1.0)), 0.0, qt, seed)
        if stype == 'complete':
            di = self._pick(self.gi.general_indices)
            if di is None:
                return None
            contam = float(self.rng.uniform(0.0, 100.0))
            ci = self._cross(self.gi.all_genomes[di]['phylum'], int(self.rng.integers(1, 6)))
            return (sid, 'complete', di, ci, 1.0, contam, qt, seed)
        if stype in ('within_phylum', 'cross_phylum', 'reduced_genome', 'archaeal'):
            pool = {'within_phylum': self.gi.general_indices,
                    'cross_phylum': self.gi.general_indices,
                    'reduced_genome': (self.gi.reduced_indices
                                       if len(self.gi.reduced_indices) else
                                       self.gi.general_indices),
                    'archaeal': (self.gi.archaea_indices
                                 if len(self.gi.archaea_indices) else
                                 self.gi.general_indices)}[stype]
            di = self._pick(pool)
            if di is None:
                return None
            comp = float(self.rng.uniform(0.5, 1.0))
            # V4/V5 constraint: contaminant_bp <= dominant_actual_bp
            contam = float(self.rng.uniform(0.0, comp * 100.0))
            meta = self.gi.all_genomes[di]
            if stype == 'within_phylum':
                ci = self._within(meta['phylum'], int(self.rng.integers(1, 4)),
                                  meta['accession'])
                if not ci:
                    ci = self._cross(meta['phylum'], 1)
            elif stype == 'cross_phylum':
                ci = self._cross(meta['phylum'], int(self.rng.integers(1, 6)))
            else:
                ci = self._cross(meta['phylum'], int(self.rng.integers(1, 4)))
            return (sid, stype, di, ci, comp, contam, qt, seed)
        return None

    def _within(self, phylum, n, exclude_acc=''):
        pool = self.gi.by_phylum.get(phylum)
        if pool is None or len(pool) == 0:
            return []
        if exclude_acc:
            m = np.array([self.gi.all_genomes[i]['accession'] != exclude_acc for i in pool])
            pool = pool[m]
        if len(pool) == 0:
            return []
        return [int(i) for i in self.rng.choice(pool, size=min(n, len(pool)), replace=False)]

    def _cross(self, dom_phylum, n):
        other = [p for p in self.gi.phylum_list if p != dom_phylum] or self.gi.phylum_list
        chosen = self.rng.choice(len(other), size=min(n, len(other)), replace=False)
        out = []
        for k in chosen:
            pool = self.gi.by_phylum.get(other[k])
            if pool is not None and len(pool):
                out.append(int(pool[self.rng.integers(0, len(pool))]))
        return out


class ABPlanner:
    def __init__(self, gi, batch_id, part, n=BS):
        self.gi = gi
        self.bid = batch_id
        self.part = part
        self.n = n
        self.rng = np.random.default_rng(batch_id * 100000 + 42)

    def plan(self):
        samples, needed = [], set()
        ng = len(self.gi.general_indices)
        for i in range(self.n):
            di = int(self.gi.general_indices[self.rng.integers(0, ng)])
            needed.add(di)
            if self.part == 'A':
                samples.append((i, di, self.bid * 100000 + i + 1_000_000))
            else:
                tc = float(self.rng.uniform(0.0, 10.0))
                ph = self.gi.all_genomes[di]['phylum']
                if self.rng.random() < 0.5:
                    ctype, cis = 'within_phylum_low_contam', self._within(ph, di)
                else:
                    ctype, cis = 'cross_phylum_low_contam', self._cross(ph)
                needed.update(cis)
                samples.append((i, di, cis, tc, ctype,
                                self.bid * 100000 + i + 2_000_000))
        return samples, needed

    def _within(self, phylum, excl):
        pool = self.gi.by_phylum.get(phylum)
        if pool is None or len(pool) == 0:
            return self._cross(phylum)
        pool = pool[pool != excl]
        if len(pool) == 0:
            return self._cross(phylum)
        n = min(int(self.rng.integers(1, 4)), len(pool))
        return [int(i) for i in self.rng.choice(pool, size=n, replace=False)]

    def _cross(self, dom_phylum):
        other = [p for p in self.gi.phylum_list if p != dom_phylum] or self.gi.phylum_list
        n = min(int(self.rng.integers(1, 4)), len(other))
        out = []
        for k in self.rng.choice(len(other), size=n, replace=False):
            pool = self.gi.by_phylum.get(other[k])
            if pool is not None and len(pool):
                out.append(int(pool[self.rng.integers(0, len(pool))]))
        return out


# ===========================================================================
# Checkpoint
# ===========================================================================
class Ckpt:
    def __init__(self, path):
        self.path = Path(path)
        self.done = set()
        self.times = {}
        self.errors = {}
        self.phases = {}
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                self.done = set(d.get('done', []))
                self.times = d.get('times', {})
                self.errors = d.get('errors', {})
                self.phases = d.get('phases', {})
            except Exception:
                pass

    def _save(self):
        tmp = str(self.path) + '.tmp'
        Path(tmp).write_text(json.dumps({'done': sorted(self.done), 'times': self.times,
                                         'errors': self.errors, 'phases': self.phases,
                                         'updated': time.strftime('%F %T')}, indent=1))
        os.replace(tmp, self.path)

    def is_done(self, k):
        return k in self.done

    def mark(self, k, el):
        self.done.add(k)
        self.times[k] = round(el, 1)
        self._save()

    def err(self, k, e):
        self.errors[k] = str(e)[:500]
        self._save()

    def phase(self, p, v='complete'):
        self.phases[p] = v
        self._save()


# ===========================================================================
# HDF5
# ===========================================================================
def create_h5(path, sizes, logger):
    if Path(path).exists():
        logger.info(f'  {path} exists; appending per checkpoint')
        return
    logger.info(f'  creating {path}')
    with h5py.File(path, 'w') as f:
        for name, n in sizes.items():
            g = f.create_group(name)
            g.create_dataset('kmer_features', shape=(n, NK), dtype=np.float32,
                             chunks=(min(BS, n), NK), compression='gzip',
                             compression_opts=1, fillvalue=0.0)
            g.create_dataset('assembly_features', shape=(n, NA), dtype=np.float32,
                             chunks=(min(BS, n), NA), compression='gzip',
                             compression_opts=1, fillvalue=0.0)
            g.create_dataset('labels', shape=(n, 2), dtype=np.float32,
                             chunks=(min(BS, n), 2), compression='gzip', compression_opts=1)
            g.create_dataset('metadata', shape=(n,), dtype=METADATA_DTYPE,
                             chunks=(min(BS, n),), compression='gzip', compression_opts=1)
            g.attrs['n_written'] = 0
            g.attrs['n_total'] = n
            logger.info(f'    {name}: {n:,}')
        f.attrs['n_kmer_features'] = NK
        f.attrs['n_assembly_features'] = NA
        f.attrs['version'] = f'holdout_{C.PANEL_LEVEL}_v1'
        f.attrs['panel_level'] = C.PANEL_LEVEL
        f.attrs['panel_taxa'] = json.dumps(C.PANEL_TAXA)
        f.attrs['panel_phyla'] = json.dumps(C.PANEL_PHYLA)
        f.attrs['panel_groups'] = json.dumps({k: v['taxa'] for k, v in C.PANEL_GROUPS.items()})
        f.attrs['recipe'] = (f'V5-matched: 800K V4-recipe + 100K partA + 100K partB train, '
                             f'100K val, 100K test; panel {C.PANEL_LEVEL} excluded from '
                             f'dominant AND contaminant roles')
        f.attrs['created'] = time.strftime('%F %T')
        f.attrs['normalized'] = False


def assemble(results, batch_tag, logger):
    n = len(results)
    kc = np.zeros((n, NK), dtype=np.int64)
    asm = np.zeros((n, NA), dtype=np.float64)
    lab = np.zeros((n, 2), dtype=np.float32)
    md = np.zeros(n, dtype=METADATA_DTYPE)
    tc = {}
    for i, r in enumerate(results):
        (sid, k, a, comp, contam, phy, st, qt, acc, fl, nc) = r
        kc[i] = k
        asm[i] = a
        comp = float(np.clip(comp, 0.0, 100.0))
        contam = float(np.clip(contam, 0.0, 100.0))
        lab[i] = (comp, contam)
        md[i]['completeness'] = comp
        md[i]['contamination'] = contam
        md[i]['dominant_phylum'] = phy.encode()[:64]
        md[i]['sample_type'] = st.encode()[:32]
        md[i]['quality_tier'] = (qt if isinstance(qt, str) else str(qt)).encode()[:20]
        md[i]['dominant_accession'] = acc.encode()[:30]
        md[i]['genome_full_length'] = fl
        md[i]['n_contaminants'] = nc
        md[i]['batch_id'] = batch_tag
        tc[st] = tc.get(st, 0) + 1
    return kc, asm, lab, md, tc


def write_h5(path, split, offset, kc, asm, lab, md, logger):
    with h5py.File(path, 'a') as f:
        g = f[split]
        e = offset + len(lab)
        assert e <= int(g.attrs['n_total']), f'{e} > {g.attrs["n_total"]}'
        g['kmer_features'][offset:e] = kc.astype(np.float32)
        g['assembly_features'][offset:e] = asm.astype(np.float32)
        g['labels'][offset:e] = lab
        g['metadata'][offset:e] = md
        g.attrs['n_written'] = max(int(g.attrs['n_written']), e)
        f.flush()
    logger.info(f'    wrote {len(lab)} -> {split}[{offset}:{offset+len(lab)}]')


# ===========================================================================
# Batch runners
# ===========================================================================
def run_batch(kind, bid, split, offset, gi, pool, nw, logger, n_target=BS,
              types=None, async_warm=True):
    global _gi
    t0 = time.perf_counter()
    if kind == 'v4':
        plans, needed = V4Planner(gi, bid, types).plan()
        fn = work_v4
    else:
        plans, needed = ABPlanner(gi, bid, kind, n_target).plan()
        fn = work_A if kind == 'A' else work_B
    paths = list({gi.all_genomes[i]['fasta_path'] for i in needed})
    logger.info(f'  [{kind}:{bid}] {len(plans)} samples, {len(paths)} genome files')
    # Page-cache warming: with ~275 GB of genome pool and 881 GB of RAM, the pool
    # stays resident after the first pass, so a BLOCKING warm costs ~115 s per
    # batch (~3.8 h over 120 batches) for almost no benefit. Warm asynchronously
    # instead, so I/O overlaps with sample generation; workers that reach an
    # uncached file simply read it themselves, as they would anyway.
    warm_pool = None
    if async_warm:
        warm_pool = ThreadPoolExecutor(max_workers=8)
        warm_pool.submit(warm_page_cache, paths, 8, logger)
    else:
        warm_page_cache(paths, nw, logger)
    _gi = gi
    setup = time.perf_counter() - t0

    res, nfail = [], 0
    cs = max(4, len(plans) // (nw * 8))
    for r in pool.imap_unordered(fn, plans, chunksize=cs):
        if r is not None:
            res.append(r)
        else:
            nfail += 1
        if (len(res) + nfail) % 2500 == 0:
            pe = time.perf_counter() - t0 - setup
            logger.info(f'    {len(res)+nfail}/{len(plans)} ({nfail} fail) '
                        f'{(len(res)+nfail)/max(pe,1e-9):.0f}/s')

    # V5 post-synthesis verification
    if kind == 'v4':
        keep, nx, nr = [], 0, 0
        for r in res:
            comp, contam = r[3], r[4]
            if contam > comp:
                nx += 1
                continue
            if not (50.0 <= comp <= 100.0 and 0.0 <= contam <= 100.0):
                nr += 1
                continue
            keep.append(r)
        if len(keep) != len(res):
            logger.info(f'    verification removed {len(res)-len(keep)} '
                        f'({nx} contam>comp, {nr} out-of-range)')
        res = keep
    elif kind == 'A':
        res = [r for r in res if r[3] == 100.0 and r[4] == 0.0]
    else:
        res = [r for r in res if r[3] == 100.0 and 0.0 <= r[4] <= 10.0]

    # Top-up retries (V5 behaviour)
    tries = 0
    while len(res) < n_target and tries < 5:
        tries += 1
        need = n_target - len(res)
        logger.info(f'    retry {tries}: need {need}')
        rr = np.random.default_rng(bid * 10000 + 90000 + tries * 1000 +
                                   {'v4': 0, 'A': 300000, 'B': 600000}[kind])
        rp = []
        for i in range(need * 2):
            sid = n_target + tries * need * 2 + i
            di = int(rr.choice(gi.general_indices))
            if kind == 'v4':
                qt = rr.choice(QUALITY_TIERS, p=QUALITY_WEIGHTS)
                rp.append((sid, 'pure', di, None, float(rr.uniform(0.5, 1.0)), 0.0, qt,
                           bid * 10000 + sid))
            elif kind == 'A':
                rp.append((sid, di, bid * 100000 + sid + 5_000_000))
            else:
                ph = gi.all_genomes[di]['phylum']
                other = [p for p in gi.phylum_list if p != ph]
                cis = []
                if other:
                    cp = gi.by_phylum[other[int(rr.integers(0, len(other)))]]
                    cis = [int(cp[rr.integers(0, len(cp))])]
                rp.append((sid, di, cis, float(rr.uniform(0.0, 10.0)),
                           'cross_phylum_low_contam', bid * 100000 + sid + 6_000_000))
        for r in pool.imap_unordered(fn, rp, chunksize=50):
            if r is None:
                continue
            if kind == 'v4' and not (r[4] <= r[3] and 50.0 <= r[3] <= 100.0):
                continue
            if kind == 'A' and not (r[3] == 100.0 and r[4] == 0.0):
                continue
            if kind == 'B' and not (r[3] == 100.0 and 0.0 <= r[4] <= 10.0):
                continue
            res.append(r)
            if len(res) >= n_target:
                break

    if warm_pool is not None:
        warm_pool.shutdown(wait=False)
    res = res[:n_target]
    if len(res) < n_target:
        logger.warning(f'    [{kind}:{bid}] only {len(res)}/{n_target}')
    tag = {'v4': 0, 'A': 1_000_000, 'B': 2_000_000}[kind] + bid
    kc, asm, lab, md, tc = assemble(res, tag, logger)
    el = time.perf_counter() - t0
    logger.info(f'  [{kind}:{bid}] done {len(res)} in {el:.0f}s '
                f'({setup:.0f}s setup, {len(res)/max(el-setup,1e-9):.0f}/s) types={tc}')
    logger.info(f'      comp mean {lab[:,0].mean():.1f}% [{lab[:,0].min():.1f},'
                f'{lab[:,0].max():.1f}]  contam mean {lab[:,1].mean():.1f}% '
                f'[{lab[:,1].min():.1f},{lab[:,1].max():.1f}]')
    return kc, asm, lab, md, el


# ===========================================================================
# Normalizer persistence
# ===========================================================================
def save_norm(nrm):
    def dump(s):
        return {'count': int(s.count), 'mean': s.mean.tolist(), 'm2': s.m2.tolist(),
                'min_vals': s.min_vals.tolist(), 'max_vals': s.max_vals.tolist(),
                'reservoir_count': int(s.reservoir_count),
                'reservoir': s.reservoir[:s.reservoir_count].tolist()}
    tmp = str(NORMSTATS) + '.tmp'
    Path(tmp).write_text(json.dumps({'kmer': dump(nrm.kmer_stats),
                                     'assembly': dump(nrm.assembly_stats)}))
    os.replace(tmp, NORMSTATS)


def load_norm(nrm):
    if not Path(NORMSTATS).exists():
        return False
    try:
        d = json.loads(Path(NORMSTATS).read_text())
        for key, s in [('kmer', nrm.kmer_stats), ('assembly', nrm.assembly_stats)]:
            x = d[key]
            s.count = x['count']
            s.mean = np.array(x['mean'], dtype=np.float64)
            s.m2 = np.array(x['m2'], dtype=np.float64)
            s.min_vals = np.array(x['min_vals'], dtype=np.float64)
            s.max_vals = np.array(x['max_vals'], dtype=np.float64)
            r = np.array(x['reservoir'], dtype=np.float64)
            n = min(len(r), s.reservoir_size)
            if n:
                s.reservoir[:n] = r[:n]
            s.reservoir_count = n
        return True
    except Exception:
        return False


def finalize(path, nrm, sizes, logger):
    logger.info('Finalizing normalization (log1p + z-score; V5 procedure)')
    nrm.finalize()
    nrm.save(str(C.HOLDOUT_NORM_PARAMS))
    logger.info(f'  saved {C.HOLDOUT_NORM_PARAMS}')
    for split, n in sizes.items():
        logger.info(f'  normalizing {split} ({n:,})')
        for s in range(0, n, BS):
            e = min(s + BS, n)
            with h5py.File(path, 'a') as f:
                g = f[split]
                k = g['kmer_features'][s:e]
                a = g['assembly_features'][s:e]
                g['kmer_features'][s:e] = nrm.normalize_kmer(k).astype(np.float32)
                g['assembly_features'][s:e] = nrm.normalize_assembly(a).astype(np.float32)
                f.flush()
    with h5py.File(path, 'a') as f:
        f.attrs['normalized'] = True
        f.attrs['norm_params'] = str(C.HOLDOUT_NORM_PARAMS)
    logger.info('  normalization applied to all splits')


# ===========================================================================
def main():
    global _gi
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--smoke', action='store_true',
                    help='5,000-sample end-to-end test into a separate HDF5')
    ap.add_argument('--smoke-div', type=int, default=20,
                    help='divisor applied to each per-batch sample-type count in '
                         '--smoke mode. The default 20 reproduces the WS1.6/WS1.9 '
                         'smoke test exactly; a larger value gives a smaller pilot '
                         '(used by WS11.G Phase 1, which had to stay off the CPU).')
    ap.add_argument('--finalize-only', action='store_true')
    ap.add_argument('--prewarm', action='store_true',
                    help='read the whole genome pool once into the page cache first')
    ap.add_argument('--blocking-warm', action='store_true',
                    help='wait for the per-batch page-cache warm (V5 behaviour); '
                         'by default the warm runs asynchronously so I/O overlaps '
                         'with sample generation')
    a = ap.parse_args()

    logger = setup_logging(a.smoke)
    logger.info('=' * 78)
    logger.info(f'{C.WS.upper()} - build leave-{C.PANEL_LEVEL}-out synthetic data'
                + (' [SMOKE TEST]' if a.smoke else ''))
    logger.info(f'panel {C.PANEL_LEVEL} ({len(C.PANEL_TAXA)}): {C.PANEL_TAXA}')
    logger.info(f'panel groups: {list(C.PANEL_GROUPS)}')
    _rm = C.PANEL_PHYLA or (
        f'NONE ({C.PANEL_LEVEL} level: every parent phylum stays in training'
        + (', and so does every parent family' if C.PANEL_LEVEL == 'genus' else '')
        + ')')
    logger.info(f'phyla removed ENTIRELY: {_rm}')
    logger.info(f'adapted reduced-genome pool: {C.USE_ADAPTED_REDUCED_POOL}')
    logger.info(f'output HDF5: {C.HOLDOUT_H5}')
    logger.info(f'workers={a.workers}')
    logger.info('=' * 78)

    if a.smoke:
        _sfx = '' if a.smoke_div == 20 else f'_div{a.smoke_div}'
        h5 = C.HOLDOUT_DIR / f'smoke_features{_sfx}.h5'
        ckpt_path = C.HOLDOUT_DIR / f'smoke_checkpoint{_sfx}.json'
        per = {k: max(1, v // a.smoke_div) for k, v in C.SAMPLE_TYPES.items()}
        n_small = sum(per.values())
        sizes = {'train': n_small * 6, 'val': n_small, 'test': n_small}
        v4a = ([(i, 'train', i * n_small) for i in range(4)] +
               [(4, 'val', 0)] + [(5, 'test', 0)])
        aa = [(0, 'train', n_small * 4)]
        bb = [(0, 'train', n_small * 5)]
        nab = n_small
    else:
        h5 = C.HOLDOUT_H5
        ckpt_path = CKPT
        per = None
        sizes = {'train': C.N_TRAIN_TOTAL, 'val': C.N_VAL, 'test': C.N_TEST}
        v4a, aa, bb, nab = V4_ASSIGN, A_ASSIGN, B_ASSIGN, BS

    ck = Ckpt(ckpt_path)
    logger.info(f'checkpoint: {len(ck.done)} batches already done')

    logger.info('loading k-mer index / warming Numba')
    t0 = time.perf_counter()
    nk = init_kmer_globals()
    assert nk == NK, f'expected {NK} k-mers, got {nk}'
    logger.info(f'  {nk} k-mers, {time.perf_counter()-t0:.1f}s')

    nrm = FeatureNormalizer(n_kmer_features=NK, reservoir_size=50000)
    if load_norm(nrm):
        logger.info(f'  resumed normalizer: {nrm.kmer_stats.count:,} samples')

    create_h5(h5, sizes, logger)

    if not a.finalize_only:
        logger.info('loading panel-free genome pools')
        gis = {}
        for split, tsv in [('train', C.HOLDOUT_DIR / 'train_genomes_holdout.tsv'),
                           ('val', C.HOLDOUT_DIR / 'val_genomes_holdout.tsv'),
                           ('test', C.HOLDOUT_DIR / 'test_genomes_holdout.tsv')]:
            gis[split] = GenomeIndex(tsv, logger)
        # Reduced-genome pool sanity gate (see module docstring). V5's own pool
        # is 1,453 genomes; if the panel collapses the surviving pool below the
        # threshold the run must ADAPT rather than silently train a model that
        # never saw a small genome.
        n_red = len(gis['train'].reduced_indices)
        logger.info(f'reduced-genome training pool: {n_red} genomes '
                    f'(V5: 1453; adaptation applied: {C.USE_ADAPTED_REDUCED_POOL})')
        if not C.USE_ADAPTED_REDUCED_POOL:
            assert n_red >= C.REDUCED_POOL_ADAPT_THRESHOLD * 1453, (
                f'reduced-genome pool collapsed to {n_red}; set '
                f'USE_ADAPTED_REDUCED_POOL and document the adaptation')

        # One-off full-pool page-cache prewarm. The panel-free pools total ~275 GB
        # and RAM is 881 GB, so after this every per-batch warm is a RAM hit and the
        # job becomes CPU-bound instead of HDD-bound.
        if a.prewarm and ck.phases.get('prewarm') != 'complete':
            for split in ['train', 'val', 'test']:
                paths = [g['fasta_path'] for g in gis[split].all_genomes]
                logger.info(f'prewarming page cache for {split}: {len(paths):,} files')
                warm_page_cache(paths, 32, logger)
            ck.phase('prewarm')

        queue = ([('v4', b, s, o) for b, s, o in v4a] +
                 [('A', b, s, o) for b, s, o in aa] +
                 [('B', b, s, o) for b, s, o in bb])
        pool, cur_split = None, None
        t_all = time.perf_counter()
        try:
            for kind, bid, split, off in queue:
                key = f'{kind}:{bid}'
                if ck.is_done(key):
                    logger.info(f'{key} already done, skip')
                    continue
                if split != cur_split:
                    if pool is not None:
                        pool.terminate()
                        pool.join()
                    _gi = gis[split]
                    pool = mp.Pool(processes=a.workers)
                    cur_split = split
                    logger.info(f'worker pool ({a.workers}) for split={split}')
                nrem = sum(1 for k, b, _, _ in queue if not ck.is_done(f'{k}:{b}'))
                avg = np.mean(list(ck.times.values())) if ck.times else 0
                logger.info(f'\n=== {key} -> {split}[{off}] '
                            f'({nrem} left, ETA ~{avg*nrem/3600:.1f} h) ===')
                try:
                    kc, asm, lab, md, el = run_batch(kind, bid, split, off, gis[split],
                                                     pool, a.workers, logger,
                                                     n_target=(sum(per.values()) if per
                                                               else BS) if kind == 'v4'
                                                     else nab,
                                                     types=per,
                                                     async_warm=not a.blocking_warm)
                    write_h5(h5, split, off, kc, asm, lab, md, logger)
                    # V5 procedure: normalizer estimated from the V4-recipe portion only
                    if kind == 'v4':
                        nrm.update_kmer_batch(kc.astype(np.float64))
                        nrm.update_assembly_batch(asm)
                    ck.mark(key, el)
                    if len(ck.done) % 5 == 0:
                        save_norm(nrm)
                    del kc, asm, lab, md
                    gc.collect()
                except Exception as e:
                    logger.error(f'{key} FAILED: {e}\n{traceback.format_exc()}')
                    ck.err(key, e)
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
        save_norm(nrm)
        logger.info(f'synthesis wall time {(time.perf_counter()-t_all)/3600:.2f} h')

    all_keys = ([f'v4:{b}' for b, _, _ in v4a] + [f'A:{b}' for b, _, _ in aa] +
                [f'B:{b}' for b, _, _ in bb])
    if all(ck.is_done(k) for k in all_keys) or a.finalize_only:
        if ck.phases.get('normalized') != 'complete':
            finalize(h5, nrm, sizes, logger)
            ck.phase('normalized')
        else:
            logger.info('already normalized')
        # summary + panel-leak verification.
        # METADATA_DTYPE records dominant_phylum and dominant_accession but NOT
        # dominant_family, so at family level the leak test must run on
        # ACCESSIONS (a stronger check than the phylum-name test anyway).
        import pandas as _pd
        panel_acc = set()
        if C.PANEL_LEVEL != 'phylum':
            for tsv in [C.TRAIN_TSV, C.VAL_TSV, C.TEST_TSV]:
                _d = _pd.read_csv(tsv, sep='\t', usecols=['ncbi_accession',
                                                          'gtdb_taxonomy'])
                _d['_t'] = _d.gtdb_taxonomy.map(
                    lambda t: C.rank_from_taxonomy(t, C.TAXON_COL))
                panel_acc |= set(_d.loc[_d._t.isin(C.PANEL_TAXA), 'ncbi_accession'])
            logger.info(f'panel-family accessions across all splits: {len(panel_acc)}')
        with h5py.File(h5, 'r') as f:
            for s in sizes:
                g = f[s]
                lab = g['labels'][:]
                md = g['metadata'][:]
                st = {k.decode(): int(v) for k, v in
                      zip(*np.unique(md['sample_type'], return_counts=True))}
                phy = np.char.decode(md['dominant_phylum'])
                acc = np.char.decode(md['dominant_accession'])
                bad = set(np.unique(phy)) & set(C.PANEL_PHYLA)
                bad_acc = (set(np.unique(acc)) & panel_acc) if panel_acc else set()
                logger.info(f'{s}: n={len(lab):,} comp={lab[:,0].mean():.2f}% '
                            f'contam={lab[:,1].mean():.2f}% '
                            f'n_phyla={len(np.unique(phy))} '
                            f'n_dominant_genomes={len(np.unique(acc)):,} '
                            f'panel_phylum_leak={bad or "NONE"} '
                            f'panel_taxon_accession_leak='
                            f'{sorted(bad_acc)[:5] if bad_acc else "NONE"}')
                logger.info(f'   types: {st}')
                assert not bad, f'PANEL LEAK in {s}: {bad}'
                assert not bad_acc, f'PANEL-TAXON ACCESSION LEAK in {s}: {bad_acc}'
        logger.info(f'BUILD COMPLETE - no panel {C.PANEL_LEVEL} appears as a dominant, '
                    f'and the genome pools contained no panel {C.PANEL_LEVEL} at all, '
                    f'so none appears as a contaminant either.')
    else:
        missing = [k for k in all_keys if not ck.is_done(k)]
        logger.warning(f'incomplete: {len(missing)} batches missing: {missing[:12]}')


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
