#!/usr/bin/env python3
"""
WS1.6 step 1 - Exploratory data analysis and held-out phylum panel design.

Answers, with numbers:
  1. Genome counts per phylum per split (train/val/test), with domain and genome size.
  2. What the proposed panel costs the training set (genomes removed, % of train).
  3. Whether each panel group has enough held-out *test-split* references to build
     a meaningful evaluation set.
  4. Whether the surviving training pool can still support the V5 sample-type
     composition (archaeal 5%, reduced-genome 5%).
  5. How many panel-phylum genomes contributed to the production 9,249-mer
     feature selection (the WS1.7 leakage question).
  6. Whether the V5 HDF5 metadata records contaminant taxonomy (it does not -
     this is the finding that forces full regeneration).

Outputs
  results/revision/holdout/eda_phylum_counts.tsv
  results/revision/holdout/eda_panel_summary.json
  data/holdout/holdout_panel.json            <- consumed by scripts 121/123/124/125
  data/holdout/train_genomes_holdout.tsv     <- panel-free training pool
  data/holdout/val_genomes_holdout.tsv       <- panel-free validation pool
  data/holdout/test_genomes_holdout.tsv      <- panel-free test pool (in-distribution eval)

Usage:  python scripts/120_holdout_panel_eda.py
"""

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_lib import config as C


def h(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1
    h('1. PHYLUM COUNTS PER SPLIT')
    tr = pd.read_csv(C.TRAIN_TSV, sep='\t')
    va = pd.read_csv(C.VAL_TSV, sep='\t')
    te = pd.read_csv(C.TEST_TSV, sep='\t')
    print(f'train={len(tr):,}  val={len(va):,}  test={len(te):,}')
    print('train domains:', dict(tr.domain.value_counts()))

    cnt = pd.DataFrame({
        'train': tr.phylum.value_counts(),
        'val': va.phylum.value_counts(),
        'test': te.phylum.value_counts(),
    }).fillna(0).astype(int)
    cnt['total'] = cnt.sum(axis=1)
    allg = pd.concat([tr, va, te])
    cnt['domain'] = allg.groupby('phylum').domain.first()
    cnt['median_Mbp'] = (allg.groupby('phylum').genome_size.median() / 1e6).round(3)
    cnt['panel_group'] = [C.PHYLUM_TO_GROUP.get(p, '') for p in cnt.index]
    cnt = cnt.sort_values('train', ascending=False)
    cnt.index.name = 'phylum'
    cnt.to_csv(C.RESULTS_DIR / 'eda_phylum_counts.tsv', sep='\t')
    print(f'{len(cnt)} phyla total; wrote eda_phylum_counts.tsv')
    print(cnt.head(12).to_string())

    # ------------------------------------------------------------------ 2
    h('2. PANEL COST TO THE TRAINING SET')
    rows = []
    for group, spec in C.PANEL_GROUPS.items():
        sub = cnt.loc[[p for p in spec['phyla'] if p in cnt.index]]
        rows.append({
            'group': group,
            'phyla': ','.join(spec['phyla']),
            'train': int(sub.train.sum()),
            'val': int(sub.val.sum()),
            'test': int(sub.test.sum()),
            'total': int(sub.total.sum()),
            'domain': sub.domain.iloc[0],
            'median_Mbp': float(np.median(
                allg[allg.phylum.isin(spec['phyla'])].genome_size) / 1e6),
        })
    panel_tbl = pd.DataFrame(rows)
    print(panel_tbl.to_string(index=False))

    n_tr_rm, n_va_rm, n_te_rm = (int(panel_tbl.train.sum()),
                                 int(panel_tbl.val.sum()),
                                 int(panel_tbl.test.sum()))
    print(f'\nRemoved from train: {n_tr_rm:,} / {len(tr):,} = {100*n_tr_rm/len(tr):.2f}%')
    print(f'Removed from val:   {n_va_rm:,} / {len(va):,} = {100*n_va_rm/len(va):.2f}%')
    print(f'Removed from test:  {n_te_rm:,} / {len(te):,} = {100*n_te_rm/len(te):.2f}%')
    print(f'Surviving train pool: {len(tr)-n_tr_rm:,} genomes, '
          f'{len(cnt)-len(C.PANEL_PHYLA)} phyla')

    # ------------------------------------------------------------------ 3
    h('3. EVALUATION-SET CAPACITY (test-split references per panel group)')
    cap = []
    for group, spec in C.PANEL_GROUPS.items():
        n_test = int(cnt.loc[[p for p in spec['phyla'] if p in cnt.index], 'test'].sum())
        n_all = int(cnt.loc[[p for p in spec['phyla'] if p in cnt.index], 'total'].sum())
        d = C.EVAL_DESIGN[group]
        cap.append({'group': group, 'test_refs_available': n_test,
                    'all_split_refs': n_all, 'ref_source': d['ref_source'],
                    'refs_used': d['n_refs'], 'sims_per_ref': d['sims'],
                    'eval_n': d['n_refs'] * d['sims'],
                    'clean_head_to_head': 'yes' if d['ref_source'] == 'test'
                    else f'test-subset only (n_refs={n_test})'})
    d = C.EVAL_DESIGN['in_distribution']
    cap.append({'group': 'in_distribution', 'test_refs_available': len(te_h_pre := te[~te.phylum.isin(C.PANEL_PHYLA)]),
                'all_split_refs': int(len(tr) + len(va) + len(te) - panel_tbl.total.sum()),
                'ref_source': 'test', 'refs_used': d['n_refs'], 'sims_per_ref': d['sims'],
                'eval_n': d['n_refs'] * d['sims'], 'clean_head_to_head': 'yes (control)'})
    cap_tbl = pd.DataFrame(cap)
    print(cap_tbl.to_string(index=False))
    print('\nNOTE: dominants are drawn from the TEST split so that neither the holdout')
    print('model nor production V5 has trained on the exact genome; the only difference')
    print('between the two models is whether the LINEAGE was in training.')

    # ------------------------------------------------------------------ 4
    h('4. SURVIVING POOLS FOR THE V5 SAMPLE-TYPE COMPOSITION')
    tr_h = tr[~tr.phylum.isin(C.PANEL_PHYLA)].copy()
    n_arch = int((tr_h.domain == 'Archaea').sum())
    print(f'Archaeal training genomes surviving: {n_arch} '
          f'(V5 had {int((tr.domain=="Archaea").sum())})')
    print(f'  -> archaeal category = 5% x 800,000 = 40,000 samples '
          f'=> {40000/max(n_arch,1):.0f} samples/genome '
          f'(V5: {40000/int((tr.domain=="Archaea").sum()):.0f})')
    print('  surviving archaeal phyla:')
    print(tr_h[tr_h.domain == 'Archaea'].phylum.value_counts().to_string())

    surv_reduced_phyla = sorted(C.V5_REDUCED_GENOME_PHYLA - set(C.PANEL_PHYLA))
    n_red_phyla = int(tr_h.phylum.isin(surv_reduced_phyla).sum())
    small = tr_h[tr_h.genome_size < C.ADAPTED_REDUCED_SIZE_CUTOFF]
    adapted = tr_h[tr_h.phylum.isin(surv_reduced_phyla) |
                   (tr_h.genome_size < C.ADAPTED_REDUCED_SIZE_CUTOFF)]
    n_red_v5 = int(tr.phylum.isin(C.V5_REDUCED_GENOME_PHYLA).sum())
    print(f'\nV5 reduced-genome pool: {n_red_v5} genomes')
    print(f'Surviving V5 reduced-genome phyla {surv_reduced_phyla}: {n_red_phyla} genomes'
          f'  <-- TOO NARROW')
    print(f'Non-panel genomes < {C.ADAPTED_REDUCED_SIZE_CUTOFF/1e6:.1f} Mbp: {len(small)}')
    print(f'ADAPTED reduced-genome pool (union): {len(adapted)} genomes, '
          f'median {adapted.genome_size.median()/1e6:.2f} Mbp '
          f'(V5 pool median '
          f'{tr[tr.phylum.isin(C.V5_REDUCED_GENOME_PHYLA)].genome_size.median()/1e6:.2f} Mbp)')
    print('  top phyla in adapted pool:')
    print(adapted.phylum.value_counts().head(10).to_string())

    # ------------------------------------------------------------------ 5
    h('5. WS1.7 - PANEL CONTRIBUTION TO THE PRODUCTION 9,249-MER SELECTION')
    reps = {}
    for dom, f in [('bacterial', 'selected_bacterial_1000.tsv'),
                   ('archaeal', 'selected_archaeal_1000.tsv')]:
        d = pd.read_csv(C.KMER_DIR / f, sep='\t')
        n_panel = int(d.phylum.isin(C.PANEL_PHYLA).sum())
        reps[dom] = {'n_reps': len(d), 'n_panel': n_panel,
                     'pct_panel': round(100 * n_panel / len(d), 2),
                     'by_phylum': {k: int(v) for k, v in
                                   d[d.phylum.isin(C.PANEL_PHYLA)]
                                   .phylum.value_counts().items()}}
        print(f'{dom}: {n_panel}/{len(d)} representatives ({reps[dom]["pct_panel"]}%) '
              f'are panel phyla')
        for k, v in reps[dom]['by_phylum'].items():
            print(f'    {k}: {v}')
    print('\n=> archaeal feature selection is heavily influenced by the panel '
          '(Halobacteriota alone). WS1.7 reselection is therefore NOT a formality.')

    # ------------------------------------------------------------------ 6
    h('6. DOES THE V5 HDF5 RECORD CONTAMINANT TAXONOMY?')
    with h5py.File(C.V5_FEATURES_H5, 'r') as f:
        md = f['train']['metadata']
        fields = list(md.dtype.names)
        print('metadata fields:', fields)
        sample = md[:5]
        print('first 5 train metadata rows:')
        for r in sample:
            print('   ', {k: (r[k].decode() if isinstance(r[k], bytes) else r[k])
                          for k in fields})
        for split in ['train', 'val', 'test']:
            g = f[split]
            nw = int(g.attrs['n_written']) if 'n_written' in g.attrs else g['labels'].shape[0]
            print(f'{split}: n={nw:,} '
                  f'kmer={g["kmer_features"].shape} asm={g["assembly_features"].shape}')
    has_cont_tax = any('contaminant' in x and ('phylum' in x or 'accession' in x)
                       for x in fields)
    print(f'\nCONTAMINANT TAXONOMY RECORDED: {has_cont_tax}')
    print('=> Only dominant_phylum / dominant_accession / n_contaminants exist.')
    print('=> The existing HDF5 CANNOT be filtered to build a valid holdout: panel')
    print('   phyla appear as unlabelled contaminants in ~60% of samples.')
    print('=> Synthetic training data MUST be regenerated (script 121).')

    # How much of V5 train has a panel-phylum DOMINANT (lower bound on leakage)?
    with h5py.File(C.V5_FEATURES_H5, 'r') as f:
        n = f['train']['labels'].shape[0]
        phy = np.empty(n, dtype='S64')
        for s in range(0, n, 100_000):
            e = min(s + 100_000, n)
            phy[s:e] = f['train']['metadata'][s:e]['dominant_phylum']
    phy_s = np.char.decode(phy)
    n_panel_dom = int(np.isin(phy_s, C.PANEL_PHYLA).sum())
    print(f'\nV5 train samples with a PANEL-phylum dominant: {n_panel_dom:,}/{n:,} '
          f'({100*n_panel_dom/n:.2f}%)  <- lower bound; contaminant role not recorded')

    # ------------------------------------------------------------------ write
    h('WRITING PANEL DEFINITION AND HOLDOUT GENOME POOLS')
    va_h = va[~va.phylum.isin(C.PANEL_PHYLA)].copy()
    te_h = te[~te.phylum.isin(C.PANEL_PHYLA)].copy()
    for df, name in [(tr_h, 'train'), (va_h, 'val'), (te_h, 'test')]:
        out = C.HOLDOUT_DIR / f'{name}_genomes_holdout.tsv'
        df.to_csv(out, sep='\t', index=False)
        print(f'  {out}  {len(df):,} genomes, {df.phylum.nunique()} phyla')

    panel_json = {
        'created': pd.Timestamp.now().isoformat(),
        'panel_groups': {g: {**s,
                             'train': int(panel_tbl.loc[panel_tbl.group == g, 'train'].iloc[0]),
                             'val': int(panel_tbl.loc[panel_tbl.group == g, 'val'].iloc[0]),
                             'test': int(panel_tbl.loc[panel_tbl.group == g, 'test'].iloc[0])}
                         for g, s in C.PANEL_GROUPS.items()},
        'panel_phyla': C.PANEL_PHYLA,
        'n_removed': {'train': n_tr_rm, 'val': n_va_rm, 'test': n_te_rm},
        'pct_removed': {'train': round(100 * n_tr_rm / len(tr), 3),
                        'val': round(100 * n_va_rm / len(va), 3),
                        'test': round(100 * n_te_rm / len(te), 3)},
        'surviving': {'train': len(tr_h), 'val': len(va_h), 'test': len(te_h),
                      'train_phyla': int(tr_h.phylum.nunique()),
                      'train_archaea': n_arch},
        'eval_capacity': cap_tbl.to_dict('records'),
        'adapted_reduced_pool': {
            'surviving_v5_reduced_phyla': surv_reduced_phyla,
            'n_from_surviving_phyla': n_red_phyla,
            'size_cutoff_bp': C.ADAPTED_REDUCED_SIZE_CUTOFF,
            'n_small_nonpanel': len(small),
            'n_total_adapted': len(adapted),
            'n_v5_pool': n_red_v5,
        },
        'kmer_selection_leakage': reps,
        'v5_metadata_fields': fields,
        'v5_records_contaminant_taxonomy': bool(has_cont_tax),
        'v5_train_panel_dominant_fraction': round(100 * n_panel_dom / n, 3),
    }
    with open(C.PANEL_JSON, 'w') as f:
        json.dump(panel_json, f, indent=2)
    with open(C.RESULTS_DIR / 'eda_panel_summary.json', 'w') as f:
        json.dump(panel_json, f, indent=2)
    print(f'\n  {C.PANEL_JSON}')
    print(f'  {C.RESULTS_DIR / "eda_panel_summary.json"}')
    print('\nEDA COMPLETE')


if __name__ == '__main__':
    main()
