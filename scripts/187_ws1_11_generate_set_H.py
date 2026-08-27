#!/usr/bin/env python3
"""
WS1.11 (R1-m13) -- generate benchmark Set H (``set_H_ncbi``).

WHAT IS HELD CONSTANT
---------------------
The per-genome generation logic is ``generate_clean_cd_genome`` from
scripts/73_generate_clean_cd_benchmarks.py, reused verbatim (which is itself
``generate_set_cd_genome`` from scripts/25_benchmark_generate.py verbatim): same
fragmentation call, same cross-phylum contaminant selection from
``data/splits/test_genomes.tsv``, same inherited cap, same label arithmetic, same
constraint guard, same FASTA writer, same 10 simulations per reference, same target
distributions (completeness ~ U[0.50, 1.00), contamination ~ U[0, 100)).

WHAT CHANGES -- AND ONLY THIS
-----------------------------
How the dominant reference genomes were selected:

    set_{C,D}_clean : held-out test-split genomes, i.e. genomes that PASSED a
                      CheckM2-based curation filter (completeness >= 98 %,
                      contamination <= 2 %, < 100 contigs, N50 > 20 kbp, longest
                      contig > 100 kbp)
    set_H_ncbi      : genomes NCBI/the submitter annotated ``assembly_level ==
                      "Complete Genome"``, with NO CheckM2 filter -- half of them
                      (arm H_fail) would have been REJECTED by that filter, the other
                      half (arm H_pass) are taxonomically matched genomes that would
                      have passed.  See scripts/185_ws1_11_select_ncbi_refs.py.

ONE DELIBERATE DESIGN REFINEMENT
--------------------------------
The two arms are matched in pairs, so the *same* (target completeness, target
contamination) draw is given to both members of a pair at the same replicate index.
This makes the arm comparison paired at the sample level as well as at the reference
level; it does not change the marginal target distributions (each drawn value is used
exactly twice, so the marginals stay U[0.50,1.00) / U[0,100) and are KS-tested below).
Per-sample RNG streams still differ between arms, because the seed contains
``ref_index``.

SEEDS (fully deterministic)
---------------------------
  design RNG        : np.random.default_rng(SET_BASE)  -> n_pairs x 10 target
                      completeness values and n_pairs x 10 target contamination values
  per-sample RNG    : np.random.default_rng(SET_BASE + 1000 * ref_index + replicate)
  SET_BASE          : 7_500_000   (C_clean 7_300_000, D_clean 7_400_000 -- disjoint)
  genome index      : idx = 10 * ref_index + replicate  -> genome_<idx>

OUTPUTS (data/benchmarks/set_H_ncbi/)
  fasta/genome_<idx>.fasta
  metadata.tsv                 same columns as the clean sets, plus arm/pair_id
  labels.npy                   float32 [n, 2]
  generation_metadata.tsv      full per-sample provenance (WS7.7 / R1-m15), 50 columns
  generation_checkpoint.jsonl  append-only, makes the run resumable

Usage:
    python scripts/187_ws1_11_generate_set_H.py --test          # 4 refs x 2 sims
    python scripts/187_ws1_11_generate_set_H.py
    python scripts/187_ws1_11_generate_set_H.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path('/path/to/magicc')
sys.path.insert(0, str(PROJECT_DIR))

from magicc.contamination import generate_contaminated_sample            # noqa: E402
from magicc.fragmentation import (QUALITY_TIERS, load_original_contigs,  # noqa: E402
                                  read_fasta, simulate_fragmentation)

DATA_DIR = PROJECT_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'
SET_DIR = BENCHMARK_DIR / 'set_H_ncbi'
RESULTS_DIR = PROJECT_DIR / 'results' / 'revision' / 'circularity'

LEGACY_ROOTS = ['/path/to/home/projects/magicc2',
                '/mnt/5c77b453-f7e1-48c8-afa3-5641857a41c7/tianrm/projects/magicc2']

N_WORKERS = 24
N_SIMS = 10
SET_BASE = 7_500_000

DROPOUT_PARAMS = {
    'coverage_mean': 30.0,
    'coverage_sigma': 1.0,
    'coverage_threshold': 5.0,
    'gc_loss_strength': 0.3,
    'repeat_loss_prob': 0.15,
    'low_complexity_threshold': 0.6,
}
FRAG_SIGMA_RANGE = '0.8-1.2'


def resolve_path(p: str) -> str:
    for root in LEGACY_ROOTS:
        if isinstance(p, str) and p.startswith(root):
            return p.replace(root, str(PROJECT_DIR), 1)
    return p


def write_fasta(contigs, fasta_path, genome_id='genome'):
    """Verbatim from scripts/25_benchmark_generate.py."""
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as f:
        for i, contig in enumerate(contigs):
            f.write(f'>{genome_id}_contig_{i} len={len(contig)}\n')
            for j in range(0, len(contig), 80):
                f.write(contig[j:j + 80])
                f.write('\n')


_TEST_DF = None


def _init_worker(test_df_path):
    global _TEST_DF
    df = pd.read_csv(test_df_path, sep='\t')
    df['fasta_path'] = [resolve_path(p) for p in df['fasta_path']]
    _TEST_DF = df
    from magicc.fragmentation import _warm_numba_fragmentation
    _warm_numba_fragmentation()


def get_cross_phylum_contaminants(dominant_phylum, test_df, rng, n_contaminants=1):
    """Verbatim from scripts/73 (itself verbatim from scripts/25)."""
    candidates = test_df[test_df['phylum'] != dominant_phylum]
    if len(candidates) == 0:
        candidates = test_df
    selected = candidates.sample(n=min(n_contaminants, len(candidates)),
                                 random_state=int(rng.integers(0, 2 ** 31)))
    sequences, accs, phyla, lens = [], [], [], []
    for _, row in selected.iterrows():
        fasta_path = row['fasta_path']
        if os.path.exists(fasta_path):
            seq = read_fasta(fasta_path)
            if len(seq) > 0:
                sequences.append(seq)
                accs.append(row['gtdb_accession'])
                phyla.append(row['phylum'])
                lens.append(len(seq))
    return sequences, accs, phyla, lens, len(selected)


def generate_set_h_genome(args):
    """Body inherited verbatim from scripts/73_generate_clean_cd_benchmarks.py::
    generate_clean_cd_genome; only the recorded provenance fields differ."""
    (idx, ref_index, replicate, row_dict, target_completeness, target_contamination,
     fasta_dir, seed) = args

    genome_id = f'genome_{idx}'
    fasta_path = os.path.join(fasta_dir, f'{genome_id}.fasta')

    rng = np.random.default_rng(seed)
    accession = row_dict['accession']
    phylum = row_dict['gtdb_phylum']
    src_fasta = row_dict['fasta_path_resolved']

    try:
        test_df = _TEST_DF

        dominant_sequence = read_fasta(src_fasta)
        if len(dominant_sequence) == 0:
            return {'genome_id': genome_id, 'error': 'empty_dominant_sequence'}
        dominant_full_length = len(dominant_sequence)

        cont_accs, cont_phyla, cont_lens = [], [], []
        n_selected = 0
        n_cont_requested = 0
        quality_tier = None
        dom_contigs_final, cont_contigs_final = None, None

        if target_completeness >= 1.0:
            dominant_contigs = load_original_contigs(src_fasta)
            actual_completeness = 1.0
            quality_tier = 'original_contigs'
        else:
            frag_result = simulate_fragmentation(
                dominant_sequence, target_completeness=target_completeness, rng=rng)
            dominant_contigs = frag_result['contigs']
            actual_completeness = frag_result['completeness']
            quality_tier = frag_result['quality_tier']
            if len(dominant_contigs) == 0:
                return {'genome_id': genome_id, 'error': 'no_dominant_contigs'}

        if target_contamination <= 0:
            contigs = dominant_contigs
            actual_contamination = 0.0
            dom_contigs_final, cont_contigs_final = dominant_contigs, []
        else:
            n_cont_requested = int(rng.integers(1, 6))
            (contaminant_sequences, cont_accs, cont_phyla, cont_lens,
             n_selected) = get_cross_phylum_contaminants(
                phylum, test_df, rng, n_cont_requested)

            if len(contaminant_sequences) == 0:
                contigs = dominant_contigs
                actual_contamination = 0.0
                dom_contigs_final, cont_contigs_final = dominant_contigs, []
            else:
                result = generate_contaminated_sample(
                    dominant_sequence=dominant_sequence,
                    contaminant_sequences=contaminant_sequences,
                    target_completeness=target_completeness,
                    target_contamination=target_contamination,
                    rng=rng)
                contigs = result['contigs']
                actual_completeness = result['completeness']
                actual_contamination = result['contamination']
                quality_tier = result['dominant_quality_tier']
                dom_contigs_final = result['dominant_contigs']
                cont_contigs_final = result['contaminant_contigs']

                if actual_contamination > target_contamination * 1.2 + 5:
                    max_cont_bp = int(target_contamination / 100.0
                                      * dominant_full_length)
                    dom_contigs = result['dominant_contigs']
                    cont_contigs = result['contaminant_contigs']
                    kept_cont, kept_bp = [], 0
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
                    dom_contigs_final, cont_contigs_final = dom_contigs, kept_cont

        if len(contigs) == 0:
            return {'genome_id': genome_id, 'error': 'no_contigs'}

        dominant_bp = sum(len(c) for c in dom_contigs_final)
        contaminant_bp = sum(len(c) for c in cont_contigs_final)
        guard_fired = 0
        if contaminant_bp > dominant_bp:
            guard_fired = 1
            kept, kept_bp = [], 0
            for c in cont_contigs_final:
                if kept_bp + len(c) <= dominant_bp:
                    kept.append(c)
                    kept_bp += len(c)
                else:
                    room = dominant_bp - kept_bp
                    if room >= 500:
                        kept.append(c[:room])
                        kept_bp += room
                    break
            cont_contigs_final = kept
            contaminant_bp = kept_bp
            contigs = dom_contigs_final + cont_contigs_final
            indices = list(range(len(contigs)))
            rng.shuffle(indices)
            contigs = [contigs[i] for i in indices]
            actual_contamination = 100.0 * contaminant_bp / dominant_full_length

        total_length = sum(len(c) for c in contigs)
        write_fasta(contigs, fasta_path, genome_id)

        true_completeness = (actual_completeness * 100.0
                             if actual_completeness <= 1.0 else actual_completeness)
        true_contamination = min(actual_contamination, target_contamination + 5)
        true_contamination = min(true_contamination, 100.0)

        tier = QUALITY_TIERS.get(quality_tier)
        rec = {
            'genome_id': genome_id,
            'idx': idx,
            'ref_index': ref_index,
            'replicate': replicate,
            'seed': seed,
            'set': 'set_H_ncbi',
            'dominant_accession': accession,
            'dominant_phylum': phylum,
            'dominant_domain': row_dict.get('domain_src', ''),
            'dominant_taxonomy': row_dict.get('gtdb_taxonomy', ''),
            'dominant_source_split': row_dict.get('source_split', 'none'),
            'dominant_reference_bp': dominant_full_length,
            'dominant_retained_bp': dominant_bp,
            'contaminant_bp': contaminant_bp,
            'contaminant_accessions': ';'.join(map(str, cont_accs)),
            'contaminant_phyla': ';'.join(map(str, cont_phyla)),
            'contaminant_reference_bp': ';'.join(str(int(x)) for x in cont_lens),
            'contaminant_source_split': 'test',
            'n_contaminants_requested': n_cont_requested,
            'n_contaminants_selected': int(n_selected),
            'n_contaminants': len(cont_accs),
            'target_completeness': round(float(target_completeness) * 100.0, 6),
            'observed_completeness': round(float(true_completeness), 6),
            'target_contamination': round(float(target_contamination), 6),
            'observed_contamination': round(float(true_contamination), 6),
            'quality_tier': quality_tier,
            'frag_contig_count_range': (f'{tier[0]}-{tier[1]}' if tier else 'na'),
            'frag_n50_range_bp': (f'{tier[2]}-{tier[3]}' if tier else 'na'),
            'frag_min_contig_bp': (tier[4] if tier else 'na'),
            'frag_lognormal_sigma_range': FRAG_SIGMA_RANGE,
            'contaminant_quality_tier': 'random_per_contaminant_copy',
            'constraint_guard_fired': guard_fired,
            'n_contigs_dominant': len(dom_contigs_final),
            'n_contigs_contaminant': len(cont_contigs_final),
            'n_contigs': len(contigs),
            'total_length': total_length,
            # ---- WS1.11-specific reference provenance (recorded, never used to select)
            'arm': row_dict['arm'],
            'pair_id': row_dict['pair_id'],
            'match_level': row_dict.get('match_level', ''),
            'would_pass_original_filter': bool(row_dict['would_pass_original_filter']),
            'checkm2_severity': row_dict.get('severity', ''),
            'ref_checkm2_completeness': float(row_dict['checkm2_completeness']),
            'ref_checkm2_contamination': float(row_dict['checkm2_contamination']),
            'ref_contig_count': int(row_dict['contig_count']),
            'ref_n50_contigs': int(row_dict['n50_contigs']),
            'ref_ncbi_assembly_level': row_dict.get('ncbi_level_from_summary', ''),
            'error': '',
        }
        rec.update(DROPOUT_PARAMS)
        return rec
    except Exception as e:                                            # noqa: BLE001
        return {'genome_id': genome_id, 'idx': idx, 'ref_index': ref_index,
                'replicate': replicate, 'seed': seed, 'error': repr(e)}


META_COLS = ['genome_id', 'true_completeness', 'true_contamination',
             'dominant_accession', 'dominant_phylum', 'sample_type',
             'n_contigs', 'total_length', 'ref_index', 'replicate',
             'arm', 'pair_id', 'would_pass_original_filter', 'checkm2_severity']

GEN_COLS = ['genome_id', 'idx', 'ref_index', 'replicate', 'seed', 'set',
            'dominant_accession', 'dominant_phylum', 'dominant_domain',
            'dominant_taxonomy', 'dominant_source_split', 'dominant_reference_bp',
            'dominant_retained_bp', 'contaminant_bp', 'contaminant_accessions',
            'contaminant_phyla', 'contaminant_reference_bp', 'contaminant_source_split',
            'n_contaminants_requested', 'n_contaminants_selected', 'n_contaminants',
            'target_completeness', 'observed_completeness', 'target_contamination',
            'observed_contamination', 'quality_tier', 'frag_contig_count_range',
            'frag_n50_range_bp', 'frag_min_contig_bp', 'frag_lognormal_sigma_range',
            'contaminant_quality_tier', 'coverage_mean', 'coverage_sigma',
            'coverage_threshold', 'gc_loss_strength', 'repeat_loss_prob',
            'low_complexity_threshold', 'constraint_guard_fired',
            'n_contigs_dominant', 'n_contigs_contaminant', 'n_contigs',
            'total_length', 'arm', 'pair_id', 'match_level',
            'would_pass_original_filter', 'checkm2_severity',
            'ref_checkm2_completeness', 'ref_checkm2_contamination',
            'ref_contig_count', 'ref_n50_contigs', 'ref_ncbi_assembly_level', 'error']


def load_refs(n_refs=None):
    p = SET_DIR / 'reference_selection_final.tsv'
    if not p.exists():
        raise SystemExit(f'FATAL: {p} missing — run scripts/186_ws1_11_fetch_refs.py')
    refs = pd.read_csv(p, sep='\t')
    refs['source_split'] = np.where(refs['in_test'], 'test', 'none')
    refs = refs.sort_values(['pair_id', 'arm']).reset_index(drop=True)
    if n_refs:
        keep_pairs = sorted(refs['pair_id'].unique())[:max(1, n_refs // 2)]
        refs = refs[refs['pair_id'].isin(keep_pairs)].reset_index(drop=True)
    refs['ref_index'] = range(len(refs))
    return refs


def build_tasks(refs, n_sims, fasta_dir):
    """Targets are drawn per (pair, replicate) and shared by both arms of the pair."""
    pairs = sorted(refs['pair_id'].unique())
    pair_pos = {p: i for i, p in enumerate(pairs)}
    design_rng = np.random.default_rng(SET_BASE)
    n_draw = len(pairs) * n_sims
    comp_targets = design_rng.uniform(0.50, 1.0, size=n_draw)
    cont_targets = design_rng.uniform(0.0, 100.0, size=n_draw)

    tasks = []
    for r in range(len(refs)):
        row = refs.iloc[r].to_dict()
        t0 = pair_pos[row['pair_id']] * n_sims
        for s in range(n_sims):
            idx = r * n_sims + s
            tasks.append((idx, r, s, row, float(comp_targets[t0 + s]),
                          float(cont_targets[t0 + s]), str(fasta_dir),
                          int(SET_BASE + 1000 * r + s)))
    return tasks


def load_checkpoint(path):
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get('error'):
                    done[rec['genome_id']] = rec
    return done


def generate(refs, n_sims, out_dir):
    print('\n' + '=' * 84)
    print(f'Generating set_H_ncbi: {len(refs)} refs x {n_sims} sims '
          f'= {len(refs) * n_sims} genomes '
          f'({int((refs.arm == "H_fail").sum())} H_fail + '
          f'{int((refs.arm == "H_pass").sum())} H_pass references)')
    print('=' * 84)
    fasta_dir = out_dir / 'fasta'
    fasta_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'generation_checkpoint.jsonl'

    tasks = build_tasks(refs, n_sims, fasta_dir)
    done = load_checkpoint(ckpt_path)
    todo = [t for t in tasks
            if not (f'genome_{t[0]}' in done
                    and (fasta_dir / f'genome_{t[0]}.fasta').exists()
                    and (fasta_dir / f'genome_{t[0]}.fasta').stat().st_size > 0)]
    print(f'  checkpoint: {len(done)} recorded, {len(todo)} to generate '
          f'({N_WORKERS} workers)')

    t0 = time.time()
    n_err = 0
    if todo:
        with open(ckpt_path, 'a') as ck, \
             Pool(processes=N_WORKERS, initializer=_init_worker,
                  initargs=(str(SPLITS_DIR / 'test_genomes.tsv'),)) as pool:
            for i, rec in enumerate(pool.imap_unordered(
                    generate_set_h_genome, todo, chunksize=1), 1):
                if rec.get('error'):
                    n_err += 1
                    print(f'    ERROR {rec["genome_id"]}: {rec["error"]}', flush=True)
                else:
                    done[rec['genome_id']] = rec
                ck.write(json.dumps(rec) + '\n')
                ck.flush()
                if i % 200 == 0:
                    el = time.time() - t0
                    print(f'    {i}/{len(todo)}  {el:.0f}s  '
                          f'({el / i:.2f} s/genome)  errors={n_err}', flush=True)
    print(f'  generation: {time.time() - t0:.0f}s, {len(done)} complete, '
          f'{n_err} errors')

    gen = pd.DataFrame([done[f'genome_{t[0]}'] for t in tasks
                        if f'genome_{t[0]}' in done])
    gen = gen.reindex(columns=GEN_COLS).sort_values('idx').reset_index(drop=True)
    gen.to_csv(out_dir / 'generation_metadata.tsv', sep='\t', index=False)

    meta = pd.DataFrame({
        'genome_id': gen['genome_id'],
        'true_completeness': gen['observed_completeness'],
        'true_contamination': gen['observed_contamination'],
        'dominant_accession': gen['dominant_accession'],
        'dominant_phylum': gen['dominant_phylum'],
        'sample_type': 'set_h_ncbi',
        'n_contigs': gen['n_contigs'],
        'total_length': gen['total_length'],
        'ref_index': gen['ref_index'],
        'replicate': gen['replicate'],
        'arm': gen['arm'],
        'pair_id': gen['pair_id'],
        'would_pass_original_filter': gen['would_pass_original_filter'],
        'checkm2_severity': gen['checkm2_severity'],
    })[META_COLS]
    meta.to_csv(out_dir / 'metadata.tsv', sep='\t', index=False)
    np.save(out_dir / 'labels.npy',
            meta[['true_completeness', 'true_contamination']].values.astype(np.float32))
    print(f'  wrote metadata.tsv ({len(meta)}), labels.npy, generation_metadata.tsv '
          f'({len(GEN_COLS)} columns)')
    return gen


def validate(out_dir):
    print('\n' + '-' * 84)
    print('VALIDATION — set_H_ncbi')
    print('-' * 84)
    gen = pd.read_csv(out_dir / 'generation_metadata.tsv', sep='\t')
    meta = pd.read_csv(out_dir / 'metadata.tsv', sep='\t')
    labels = np.load(out_dir / 'labels.npy')
    fasta_dir = out_dir / 'fasta'
    rep = {'set': 'set_H_ncbi', 'n_samples': int(len(meta)),
           'n_generation_metadata_columns': int(gen.shape[1])}

    rep['metadata_matches_generation_metadata'] = bool(
        len(meta) == len(gen)
        and (meta['genome_id'].tolist() == gen['genome_id'].tolist())
        and np.allclose(meta['true_completeness'], gen['observed_completeness'])
        and np.allclose(meta['true_contamination'], gen['observed_contamination']))
    rep['labels_match_metadata'] = bool(
        labels.shape == (len(meta), 2)
        and np.allclose(labels[:, 0], meta['true_completeness'], atol=1e-4)
        and np.allclose(labels[:, 1], meta['true_contamination'], atol=1e-4))

    n_missing = n_empty = n_bad = n_len_mismatch = 0
    for gid, tot, ncont in zip(meta['genome_id'], meta['total_length'],
                               meta['n_contigs']):
        p = fasta_dir / f'{gid}.fasta'
        if not p.exists():
            n_missing += 1
            continue
        if p.stat().st_size == 0:
            n_empty += 1
            continue
        seen_header, bp, nc = False, 0, 0
        with open(p) as f:
            for line in f:
                if line.startswith('>'):
                    seen_header = True
                    nc += 1
                else:
                    bp += len(line.strip())
        if not seen_header or nc == 0:
            n_bad += 1
        elif bp != tot or nc != ncont:
            n_len_mismatch += 1
    rep.update({'fasta_missing': n_missing, 'fasta_empty': n_empty,
                'fasta_unparseable': n_bad, 'fasta_length_mismatch': n_len_mismatch})
    rep['fasta_integrity_pass'] = bool(n_missing == n_empty == n_bad
                                       == n_len_mismatch == 0)

    v_bp = int((gen['contaminant_bp'] > gen['dominant_retained_bp']).sum())
    v_pct = int((gen['observed_contamination']
                 > gen['observed_completeness'] + 1e-6).sum())
    rep['violations_contaminant_bp_gt_dominant_bp'] = v_bp
    rep['violations_contamination_gt_completeness'] = v_pct
    rep['violations_contamination_gt_100'] = int(
        (gen['observed_contamination'] > 100).sum())
    rep['violations_completeness_out_of_50_100'] = int(
        ((gen['observed_completeness'] < 50 - 1e-6)
         | (gen['observed_completeness'] > 100 + 1e-6)).sum())
    rep['constraint_guard_fired_n'] = int(gen['constraint_guard_fired'].sum())
    rep['out_of_domain_samples'] = v_pct          # protocol section 4.4a
    rep['constraints_pass'] = bool(v_bp == 0 and v_pct == 0
                                   and rep['violations_contamination_gt_100'] == 0
                                   and rep['violations_completeness_out_of_50_100'] == 0)

    comp, cont = labels[:, 0], labels[:, 1]
    comp_hist = np.histogram(comp, bins=np.linspace(50, 100, 6))[0]
    cont_hist = np.histogram(cont, bins=np.linspace(0, 100, 6))[0]
    rep['completeness'] = {'mean': round(float(comp.mean()), 3),
                           'min': round(float(comp.min()), 3),
                           'max': round(float(comp.max()), 3),
                           'hist_50_100_5bins': comp_hist.tolist()}
    rep['contamination'] = {'mean': round(float(cont.mean()), 3),
                            'min': round(float(cont.min()), 3),
                            'max': round(float(cont.max()), 3),
                            'hist_0_100_5bins': cont_hist.tolist()}
    from scipy import stats
    ks_comp = stats.kstest((gen['target_completeness'] - 50) / 50, 'uniform')
    ks_cont = stats.kstest(gen['target_contamination'] / 100, 'uniform')
    rep['ks_target_completeness_uniform'] = {'stat': round(float(ks_comp.statistic), 4),
                                             'p': round(float(ks_comp.pvalue), 4)}
    rep['ks_target_contamination_uniform'] = {'stat': round(float(ks_cont.statistic), 4),
                                              'p': round(float(ks_cont.pvalue), 4)}

    rep['n_unique_dominants'] = int(gen['dominant_accession'].nunique())
    rep['sims_per_reference'] = sorted(gen.groupby('ref_index').size().unique().tolist())
    rep['arm_counts'] = gen['arm'].value_counts().to_dict()
    rep['severity_counts'] = gen['checkm2_severity'].value_counts().to_dict()
    rep['n_pairs'] = int(gen['pair_id'].nunique())
    # the paired design: both arms of a pair must share the same target draws
    piv = gen.pivot_table(index=['pair_id', 'replicate'], columns='arm',
                          values=['target_completeness', 'target_contamination'])
    rep['paired_targets_identical'] = bool(
        np.allclose(piv[('target_completeness', 'H_fail')],
                    piv[('target_completeness', 'H_pass')])
        and np.allclose(piv[('target_contamination', 'H_fail')],
                        piv[('target_contamination', 'H_pass')]))

    same_phylum, all_cont = 0, []
    for dp, ca, cp in zip(gen['dominant_phylum'], gen['contaminant_accessions'],
                          gen['contaminant_phyla']):
        if isinstance(cp, str) and cp:
            same_phylum += sum(1 for x in cp.split(';') if x == dp)
        if isinstance(ca, str) and ca:
            all_cont.extend(ca.split(';'))
    rep['contaminant_events_total'] = len(all_cont)
    rep['unique_contaminant_genomes'] = len(set(all_cont))
    rep['contaminant_same_phylum_as_dominant'] = same_phylum
    rep['n_contaminants_distribution'] = gen['n_contaminants'].value_counts(
        ).sort_index().to_dict()
    rep['quality_tier_counts'] = gen['quality_tier'].value_counts().to_dict()

    for k, v in rep.items():
        print(f'  {k}: {v}')
    return rep


def main():
    global N_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--workers', type=int, default=N_WORKERS)
    args = ap.parse_args()
    N_WORKERS = args.workers

    n_refs, n_sims = (4, 2) if args.test else (None, N_SIMS)
    out_dir = (BENCHMARK_DIR / 'set_H_ncbi_TEST') if args.test else SET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = load_refs(n_refs)

    if not args.validate_only:
        generate(refs, n_sims, out_dir)
    rep = validate(out_dir)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = ('ws1_11_generation_validation_TEST.json' if args.test
            else 'ws1_11_generation_validation.json')
    with open(RESULTS_DIR / name, 'w') as f:
        json.dump({'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'generated_by': 'scripts/187_ws1_11_generate_set_H.py',
                   'n_refs': int(len(refs)), 'n_sims_per_ref': n_sims,
                   'n_workers': N_WORKERS, 'set_base_seed': SET_BASE,
                   'seed_formula': 'SET_BASE + 1000 * ref_index + replicate',
                   'design_rng': 'np.random.default_rng(SET_BASE), targets drawn per '
                                 '(pair, replicate) and shared by both arms',
                   'generator': 'scripts/73_generate_clean_cd_benchmarks.py logic, '
                                'verbatim',
                   'report': rep}, f, indent=2)
    print(f'\nwrote {RESULTS_DIR / name}')
    ok = (rep['fasta_integrity_pass'] and rep['constraints_pass']
          and rep['metadata_matches_generation_metadata']
          and rep['labels_match_metadata'] and rep['paired_targets_identical'])
    print(f'\nOVERALL: {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
