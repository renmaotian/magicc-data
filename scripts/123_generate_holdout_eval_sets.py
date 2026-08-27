#!/usr/bin/env python3
"""
WS1.6 step 4a - Generate leave-phylum-out evaluation sets.

One evaluation set per panel group plus an in-distribution control. Generation
logic follows scripts/025_benchmark_generate.py::generate_set_cd_genome:
  uniform completeness U[50,100]%, uniform contamination U[0,100]%,
  1-5 contaminant genomes, contaminants fragmented and trimmed to the target bp,
  explicit re-trim when realized contamination overshoots target*1.2+5.

DIFFERENCES FROM SCRIPT 25 (both are improvements, both documented):
  1. Contaminants are drawn ONLY from the panel-free TEST pool, so no held-out
     phylum ever enters as a contaminant. Script 25 drew from the whole test split.
  2. true_contamination is the exact protocol quantity
     (contaminant_bp / dominant_full_reference_length x 100). Script 25 reported
     min(actual, target+5), which clamps overshoot. The clamped value is ALSO
     stored as `true_contamination_script25` for direct comparability.

Dominants come from the held-out TEST split, so neither the holdout model nor
production V5 trained on the exact genome; the only difference between them is
whether the LINEAGE was in training. DPANN is the exception (only 9 test-split
references exist), so it uses all 83 DPANN references and every sample records
`dominant_v5_split` for a clean test-only subset analysis.

Outputs per group G:
  data/holdout/eval_sets/G/fasta/genome_*.fasta
  data/holdout/eval_sets/G/metadata.tsv        full per-sample metadata + seed
  data/holdout/eval_sets/G/features.h5         RAW k-mer counts + 7 summary features
  data/holdout/eval_sets/manifest.json

Usage
  python scripts/123_generate_holdout_eval_sets.py --workers 24
  python scripts/123_generate_holdout_eval_sets.py --smoke      # 2 refs x 2 sims/group
"""

import os
import sys

os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
import logging
import multiprocessing as mp
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_lib import config as C
from holdout_lib.fragmentation import (read_fasta, load_original_contigs,
                                       simulate_fragmentation, _warm_numba_fragmentation)
from holdout_lib.contamination import generate_contaminated_sample
from holdout_lib.kmer_counter import (load_selected_kmers, build_kmer_index,
                                      _count_kmers_single, K)
from holdout_lib.assembly_stats import compute_assembly_stats

NK = C.N_KMER_FEATURES
NA = C.N_SUMMARY_FEATURES
TAXON_COL = C.TAXON_COL

_kmer_index = None
_nk = None
_cont_pool = None       # list of dicts: panel-free TEST genomes usable as contaminants


def stable_hash(name: str) -> int:
    """CRC-32 of a string - process-stable, unlike Python's salted hash().

    REPRODUCIBILITY DEFECT FIX. This function previously used abs(hash(group)),
    which Python salts per process (PYTHONHASHSEED), so reference selection was
    not reproducible across runs. The same defect was found and fixed in the WS5
    statistics framework (scripts/101 fw.stable_hash). The completed WS1.6 run
    used the old expression; the references it actually drew are recorded
    verbatim in each group's metadata.tsv, so its results remain auditable.
    """
    return zlib.crc32(name.encode('utf-8')) & 0xFFFFFFFF


def setup_logging(smoke=False):
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    p = C.LOGS_DIR / ('%s_eval_sets%s.log' % (C.WS, '_smoke' if smoke else ''))
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(p, mode='a'),
                                  logging.StreamHandler(sys.stdout)], force=True)
    return logging.getLogger('eval_sets')


def init_globals(cont_pool):
    global _kmer_index, _nk, _cont_pool
    codes = load_selected_kmers(str(C.SELECTED_KMERS))
    _kmer_index = build_kmer_index(codes)
    _nk = len(codes)
    _count_kmers_single(np.frombuffer(b'ACGTACGTACGTACGTACGT', dtype=np.uint8),
                        _kmer_index, _nk, K)
    _warm_numba_fragmentation()
    _cont_pool = cont_pool


def write_fasta(contigs, path, gid):
    with open(path, 'w') as f:
        for i, c in enumerate(contigs):
            f.write(f'>{gid}_contig_{i} len={len(c)}\n')
            for j in range(0, len(c), 80):
                f.write(c[j:j + 80] + '\n')


def gen_one(args):
    """Generate one evaluation genome. Mirrors generate_set_cd_genome."""
    (gid, group, dom, dom_split, tcomp, tcontam, cont_idx, seed, fasta_dir) = args
    fpath = os.path.join(fasta_dir, f'{gid}.fasta')
    rng = np.random.default_rng(seed)
    try:
        dseq = read_fasta(dom['fasta_path'])
        if not dseq:
            return None
        dlen = len(dseq)

        if tcomp >= 1.0:
            dom_contigs = load_original_contigs(dom['fasta_path'])
            actual_comp = 1.0
        else:
            fr = simulate_fragmentation(dseq, target_completeness=tcomp, rng=rng)
            dom_contigs = fr['contigs']
            actual_comp = fr['completeness']
        if not dom_contigs:
            return None

        cont_accs, cont_phyla, cont_taxa, cont_seqs = [], [], [], []
        if tcontam > 0:
            for ci in cont_idx:
                g = _cont_pool[ci]
                s = read_fasta(g['fasta_path'])
                if s:
                    cont_seqs.append(s)
                    cont_accs.append(g['accession'])
                    cont_phyla.append(g['phylum'])
                    cont_taxa.append(str(g.get(TAXON_COL, g['phylum'])))

        if not cont_seqs or tcontam <= 0:
            contigs = dom_contigs
            cont_bp = 0
        else:
            r = generate_contaminated_sample(
                dominant_sequence=dseq, contaminant_sequences=cont_seqs,
                target_completeness=tcomp, target_contamination=tcontam, rng=rng)
            contigs = r['contigs']
            actual_comp = r['completeness']
            cont_bp = int(round(r['contamination'] / 100.0 * dlen))
            # script-25 explicit re-trim on large overshoot
            if r['contamination'] > tcontam * 1.2 + 5:
                max_bp = int(tcontam / 100.0 * dlen)
                kept, kbp = [], 0
                for c in r['contaminant_contigs']:
                    if kbp + len(c) <= max_bp:
                        kept.append(c)
                        kbp += len(c)
                    elif max_bp - kbp >= 500:
                        kept.append(c[:max_bp - kbp])
                        kbp = max_bp
                        break
                    else:
                        break
                contigs = list(r['dominant_contigs']) + kept
                order = np.arange(len(contigs))
                rng.shuffle(order)
                contigs = [contigs[i] for i in order]
                cont_bp = kbp
        if not contigs:
            return None

        true_comp = actual_comp * 100.0 if actual_comp <= 1.0 else actual_comp
        true_cont = 100.0 * cont_bp / dlen

        write_fasta(contigs, fpath, gid)

        cnt = np.zeros(_nk, dtype=np.int64)
        for c in contigs:
            if len(c) >= K:
                cnt += _count_kmers_single(np.frombuffer(c.encode('ascii'), dtype=np.uint8),
                                           _kmer_index, _nk, K)
        tot = cnt.sum()
        asm = compute_assembly_stats(np.log10(float(tot)) if tot > 0 else 0.0, cnt)

        return dict(
            genome_id=gid, group=group, dominant_accession=dom['accession'],
            dominant_phylum=dom['phylum'], dominant_domain=dom['domain'],
            dominant_taxon=dom.get('taxon', dom['phylum']),
            dominant_v5_split=dom_split, dominant_genome_size=dlen,
            target_completeness=round(tcomp * 100.0, 4),
            target_contamination=round(tcontam, 4),
            true_completeness=round(true_comp, 6), true_contamination=round(true_cont, 6),
            true_contamination_script25=round(min(true_cont, tcontam + 5), 6),
            contaminant_bp=cont_bp, n_contaminants=len(cont_seqs),
            contaminant_accessions=';'.join(cont_accs),
            contaminant_phyla=';'.join(cont_phyla),
            contaminant_taxa=';'.join(cont_taxa),
            n_contigs=len(contigs), total_length=sum(len(c) for c in contigs),
            seed=seed, fasta=fpath, _kmer=cnt, _asm=asm)
    except Exception as e:
        return {'genome_id': gid, 'error': str(e)[:200]}


def select_refs(group, spec, panel_all, logger):
    """Choose reference genomes for one evaluation group."""
    taxa = (C.PANEL_GROUPS[group]['taxa'] if group in C.PANEL_GROUPS else None)
    rng = np.random.default_rng(C.EVAL_SEED_BASE + stable_hash(group) % 100000)
    if group == 'in_distribution':
        # Stratified by phylum, proportional to sqrt(count) - the same
        # square-root-proportional scheme used for the original 100k curation
        # (scripts/03, 08) - via the largest-remainder method, capped at
        # availability, with the deficit redistributed to the largest phyla.
        pool = panel_all['nonpanel_test']
        target = spec['n_refs']
        vc = pool.phylum.value_counts()
        avail = vc.values.astype(int)
        w = np.sqrt(avail.astype(float))
        w = w / w.sum()
        exact = w * target
        alloc = np.floor(exact).astype(int)
        rem = target - alloc.sum()
        if rem > 0:
            for i in np.argsort(-(exact - alloc))[:rem]:
                alloc[i] += 1
        alloc = np.minimum(alloc, avail)
        # redistribute any deficit created by the availability cap
        while alloc.sum() < target:
            room = avail - alloc
            if room.max() <= 0:
                break
            order = np.argsort(-w * (room > 0))
            for i in order:
                if room[i] > 0:
                    alloc[i] += 1
                    break
        picks = []
        for ph, n in zip(vc.index, alloc):
            if n <= 0:
                continue
            sub = pool[pool.phylum == ph]
            picks.append(sub.sample(n=int(n),
                                    random_state=int(rng.integers(0, 2**31))))
        refs = pd.concat(picks)
        assert len(refs) == target, f'in_distribution: got {len(refs)} refs, want {target}'
        logger.info(f'  in_distribution strata: '
                    f'{dict(refs.phylum.value_counts())}')
    else:
        src = panel_all['panel_test'] if spec['ref_source'] == 'test' else panel_all['panel_all']
        refs = src[src[C.TAXON_COL].isin(taxa)]
        if len(refs) > spec['n_refs']:
            refs = refs.sample(n=spec['n_refs'],
                               random_state=int(rng.integers(0, 2**31)))
    refs = refs.reset_index(drop=True)
    logger.info(f'  {group}: {len(refs)} refs x {spec["sims"]} sims = '
                f'{len(refs)*spec["sims"]} samples '
                f'(source={spec["ref_source"]}, splits='
                f'{dict(refs.split.value_counts()) if "split" in refs else {}})')
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--groups', type=str, default=None, help='comma-separated subset')
    a = ap.parse_args()
    logger = setup_logging(a.smoke)
    logger.info('=' * 78)
    logger.info(f'{C.WS.upper()} - generate leave-{C.PANEL_LEVEL}-out evaluation sets'
                + (' [SMOKE]' if a.smoke else ''))
    logger.info('=' * 78)

    base = C.EVAL_DIR if not a.smoke else C.HOLDOUT_DIR / 'eval_sets_smoke'
    base.mkdir(parents=True, exist_ok=True)

    # ---- reference pools -------------------------------------------------
    dfs = {}
    for s, p in [('train', C.TRAIN_TSV), ('val', C.VAL_TSV), ('test', C.TEST_TSV)]:
        d = pd.read_csv(p, sep='\t')
        d['fasta_path'] = d.fasta_path.map(C.remap_fasta_path)
        C.add_taxon_column(d)          # phylum | family | genus per MAGICC_HOLDOUT_LEVEL
        d['split'] = s
        dfs[s] = d
    allg = pd.concat(dfs.values(), ignore_index=True)
    in_panel_te = dfs['test'][C.TAXON_COL].isin(C.PANEL_TAXA)
    pools = {
        'panel_test': dfs['test'][in_panel_te].copy(),
        'panel_all': allg[allg[C.TAXON_COL].isin(C.PANEL_TAXA)].copy(),
        'nonpanel_test': dfs['test'][~in_panel_te].copy(),
    }
    logger.info(f'pools: panel_test={len(pools["panel_test"])} '
                f'panel_all={len(pools["panel_all"])} '
                f'nonpanel_test={len(pools["nonpanel_test"])}')

    # Contaminant pool: panel-free TEST genomes only (never a held-out phylum,
    # and never a genome either model trained on)
    cp = pools['nonpanel_test']
    cp = cp[cp.fasta_path.map(os.path.exists)]
    _cols = ['ncbi_accession', 'fasta_path', 'phylum', 'domain']
    if C.TAXON_COL not in _cols:
        _cols.append(C.TAXON_COL)
    cont_pool = cp[_cols].rename(columns={'ncbi_accession': 'accession'}).to_dict('records')
    logger.info(f'contaminant pool: {len(cont_pool)} panel-free test genomes, '
                f'{cp.phylum.nunique()} phyla, {cp[C.TAXON_COL].nunique()} '
                f'{C.TAXON_COL}(s)')

    groups = list(C.EVAL_DESIGN) if not a.groups else a.groups.split(',')
    init_globals(cont_pool)

    manifest = {'created': time.strftime('%F %T'), 'seed_base': C.EVAL_SEED_BASE,
                'panel_level': C.PANEL_LEVEL, 'panel_taxa': C.PANEL_TAXA,
                'panel_phyla': C.PANEL_PHYLA,
                'contaminant_pool': {'source': 'panel-free TEST split',
                                     'n': len(cont_pool),
                                     'n_phyla': int(cp.phylum.nunique())},
                'groups': {}}

    pool_mp = mp.Pool(processes=a.workers)
    try:
        for group in groups:
            spec = dict(C.EVAL_DESIGN[group])
            if a.smoke:
                spec['n_refs'], spec['sims'] = 2, 2
            gdir = base / group
            (gdir / 'fasta').mkdir(parents=True, exist_ok=True)
            meta_path = gdir / 'metadata.tsv'
            h5_path = gdir / 'features.h5'
            if meta_path.exists() and h5_path.exists():
                logger.info(f'{group}: already generated, skipping')
                m = pd.read_csv(meta_path, sep='\t')
                manifest['groups'][group] = {**spec, 'n_samples': len(m),
                                             'n_refs_used': m.dominant_accession.nunique()}
                continue

            logger.info(f'\n--- {group} ---')
            refs = select_refs(group, spec, pools, logger)
            rng = np.random.default_rng(C.EVAL_SEED_BASE + sum(map(ord, group)))

            tasks = []
            for ri, row in refs.iterrows():
                dom = {'accession': row['ncbi_accession'], 'fasta_path': row['fasta_path'],
                       'phylum': row['phylum'], 'domain': row['domain'],
                       'taxon': row[C.TAXON_COL]}
                for si in range(spec['sims']):
                    idx = ri * spec['sims'] + si
                    seed = int(C.EVAL_SEED_BASE + sum(map(ord, group)) * 1000003 + idx)
                    r2 = np.random.default_rng(seed)
                    tcomp = float(r2.uniform(0.5, 1.0))
                    tcont = float(r2.uniform(0.0, 100.0))
                    ncont = int(r2.integers(1, 6))
                    cidx = [int(x) for x in r2.choice(len(cont_pool), size=ncont,
                                                      replace=False)]
                    tasks.append((f'genome_{idx}', group, dom, row['split'], tcomp,
                                  tcont, cidx, seed, str(gdir / 'fasta')))

            # page-cache warm
            paths = list({t[2]['fasta_path'] for t in tasks})
            with ThreadPoolExecutor(max_workers=32) as ex:
                list(ex.map(lambda p: open(p, 'rb').read() and None, paths))

            t0 = time.perf_counter()
            rows, errs = [], []
            for r in pool_mp.imap_unordered(gen_one, tasks, chunksize=4):
                if r is None:
                    errs.append('None')
                elif 'error' in r:
                    errs.append(r['error'])
                else:
                    rows.append(r)
                if (len(rows) + len(errs)) % 200 == 0:
                    logger.info(f'    {len(rows)+len(errs)}/{len(tasks)} '
                                f'({len(errs)} err) '
                                f'{(len(rows)+len(errs))/(time.perf_counter()-t0):.1f}/s')
            logger.info(f'  {group}: {len(rows)} ok, {len(errs)} err '
                        f'in {time.perf_counter()-t0:.0f}s')
            if errs:
                logger.warning(f'  first errors: {errs[:3]}')
            rows.sort(key=lambda r: int(r['genome_id'].split('_')[1]))

            kmer = np.stack([r.pop('_kmer') for r in rows]).astype(np.int64)
            asm = np.stack([r.pop('_asm') for r in rows]).astype(np.float64)
            md = pd.DataFrame(rows)
            md.to_csv(meta_path, sep='\t', index=False)
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('kmer_counts_raw', data=kmer, compression='gzip',
                                 compression_opts=1)
                f.create_dataset('summary_features_raw', data=asm, compression='gzip',
                                 compression_opts=1)
                f.create_dataset('labels', data=md[['true_completeness',
                                                    'true_contamination']].values
                                 .astype(np.float32))
                f.attrs['group'] = group
                f.attrs['note'] = ('RAW (un-normalized) counts. Normalize with the '
                                   'target model own normalization_params.json before '
                                   'inference.')
            logger.info(f'  wrote {meta_path} and {h5_path}')
            logger.info(f'  comp {md.true_completeness.mean():.2f}% '
                        f'[{md.true_completeness.min():.1f},{md.true_completeness.max():.1f}]'
                        f'  cont {md.true_contamination.mean():.2f}% '
                        f'[{md.true_contamination.min():.1f},'
                        f'{md.true_contamination.max():.1f}]')
            cph = set()
            for s in md.contaminant_phyla.fillna(''):
                cph |= set(x for x in s.split(';') if x)
            ctx = set()
            for s in md.contaminant_taxa.fillna(''):
                ctx |= set(x for x in s.split(';') if x)
            leak = ctx & set(C.PANEL_TAXA)
            logger.info(f'  contaminant phyla used: {len(cph)}; contaminant '
                        f'{C.TAXON_COL}s used: {len(ctx)}; panel leak: {leak or "NONE"}')
            assert not leak, (f'panel {C.PANEL_LEVEL} used as contaminant in '
                              f'{group}: {leak}')
            # V4/V5 training-domain constraint: contaminant_bp <= dominant_actual_bp,
            # i.e. contamination% <= completeness%. Enforced inside
            # generate_contaminated_sample (magicc/contamination.py, commit c5a9b95);
            # verified here so these sets are directly comparable to set_C/D_clean.
            nviol = int((md.true_contamination > md.true_completeness + 1e-6).sum())
            logger.info(f'  constraint contamination<=completeness: {nviol} violations')
            assert nviol == 0, f'{group}: {nviol} constraint violations'
            manifest['groups'][group] = {
                **spec, 'n_samples': len(md),
                'n_refs_used': int(md.dominant_accession.nunique()),
                'dominant_splits': {k: int(v) for k, v in
                                    md.dominant_v5_split.value_counts().items()},
                'dominant_phyla': {k: int(v) for k, v in
                                   md.dominant_phylum.value_counts().items()},
                'mean_true_completeness': round(float(md.true_completeness.mean()), 3),
                'mean_true_contamination': round(float(md.true_contamination.mean()), 3),
                'contaminant_phyla_used': sorted(cph),
                f'contaminant_{C.TAXON_COL}s_used_n': len(ctx),
                'constraint_violations': nviol,
                'n_errors': len(errs)}
            (base / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    finally:
        pool_mp.terminate()
        pool_mp.join()

    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    logger.info(f'\nmanifest: {base / "manifest.json"}')
    tot = sum(g['n_samples'] for g in manifest['groups'].values())
    logger.info(f'TOTAL {tot} evaluation genomes across {len(manifest["groups"])} groups')


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
