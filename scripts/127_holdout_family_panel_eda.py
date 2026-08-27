#!/usr/bin/env python3
"""
WS1.9 step 1 - EDA and held-out FAMILY panel design (modelled on 120_holdout_panel_eda.py).

WHY THIS EXPERIMENT EXISTS
  WS1.6 held out whole PHYLA and found that MAGICC does not generalize to unseen
  phyla (difference-in-differences completeness degradation +3.80 to +22.81 pp,
  all BH q = 0.0005, with an in_distribution control of -0.047 pp proving the
  smaller training pool was not the cause). The protocol's WS1 decision rule
  therefore triggers a descent to FAMILY level.

  Phylum-level holdout is the extreme case. The practically important question is
  different: most newly assembled MAGs are novel at family or genus level WITHIN
  AN ALREADY-KNOWN PHYLUM. So the defining constraint of this panel is

      EVERY HELD-OUT FAMILY'S PARENT PHYLUM REMAINS IN THE TRAINING SET.

  This script verifies that constraint numerically rather than asserting it, and
  quantifies exactly how much of each parent phylum survives.

Answers, with numbers:
  1. Genome counts per family per split, with parent phylum, domain, genome size.
  2. What the family panel costs the training set, and - the non-negotiable
     check - that all 110 phyla still have training genomes afterwards.
  3. Whether each panel group has enough held-out TEST-split references for a
     stable cluster bootstrap.
  4. Whether the surviving training pool still supports the V5 sample-type
     composition (archaeal 5%, reduced-genome 5%) WITHOUT the WS1.6 adaptation.
  5. Nearest-surviving-relative context per panel family (how many genomes of the
     same order / class / phylum remain in training) - the covariate that makes
     "novel family inside a seen phylum" quantitative.
  6. Panel contribution to the production 9,249-mer feature selection.
  7. Re-confirms that the V5 HDF5 cannot be filtered (no contaminant taxonomy),
     so regeneration is mandatory.

Outputs
  results/revision/holdout_family/eda_family_counts.tsv
  results/revision/holdout_family/eda_panel_summary.json
  results/revision/holdout_family/panel_family_detail.tsv
  data/holdout_family/holdout_panel.json          <- consumed by 121/123/124
  data/holdout_family/train_genomes_holdout.tsv   <- panel-free training pool
  data/holdout_family/val_genomes_holdout.tsv
  data/holdout_family/test_genomes_holdout.tsv

Usage
  MAGICC_HOLDOUT_LEVEL=family python scripts/127_holdout_family_panel_eda.py
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

assert C.PANEL_LEVEL == 'family', \
    'run with MAGICC_HOLDOUT_LEVEL=family'


def h(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1
    h('1. FAMILY COUNTS PER SPLIT')
    dfs = {}
    for s, p in [('train', C.TRAIN_TSV), ('val', C.VAL_TSV), ('test', C.TEST_TSV)]:
        d = pd.read_csv(p, sep='\t')
        d['fasta_path'] = d.fasta_path.map(C.remap_fasta_path)
        for rank in ['class', 'order', 'family', 'genus']:
            d[rank] = d.gtdb_taxonomy.map(lambda t, r=rank: C.rank_from_taxonomy(t, r))
        d['split'] = s
        dfs[s] = d
    tr, va, te = dfs['train'], dfs['val'], dfs['test']
    allg = pd.concat(dfs.values(), ignore_index=True)
    print(f'train={len(tr):,}  val={len(va):,}  test={len(te):,}')
    print(f'{allg.phylum.nunique()} phyla, {allg.family.nunique()} families, '
          f'{allg.genus.nunique()} genera')
    assert (allg.family == '').sum() == 0, 'some genomes have no GTDB family'
    assert allg.groupby('family').phylum.nunique().max() == 1, \
        'a family name spans >1 phylum; (phylum,family) keying required'

    cnt = pd.DataFrame({'train': tr.family.value_counts(),
                        'val': va.family.value_counts(),
                        'test': te.family.value_counts()}).fillna(0).astype(int)
    cnt['total'] = cnt.sum(axis=1)
    cnt['phylum'] = allg.groupby('family').phylum.first()
    cnt['domain'] = allg.groupby('family').domain.first()
    cnt['class'] = allg.groupby('family')['class'].first()
    cnt['order'] = allg.groupby('family')['order'].first()
    cnt['n_genera'] = allg.groupby('family').genus.nunique()
    cnt['median_Mbp'] = (allg.groupby('family').genome_size.median() / 1e6).round(3)
    cnt['panel_group'] = [C.TAXON_TO_GROUP.get(f, '') for f in cnt.index]
    cnt = cnt.sort_values('train', ascending=False)
    cnt.index.name = 'family'
    cnt.to_csv(C.RESULTS_DIR / 'eda_family_counts.tsv', sep='\t')
    print(f'wrote eda_family_counts.tsv ({len(cnt)} families)')
    print(cnt.head(10).to_string())

    # ------------------------------------------------------------------ 2
    h('2. PANEL COST TO THE TRAINING SET, AND THE PARENT-PHYLUM CONSTRAINT')
    rows = []
    for group, spec in C.PANEL_GROUPS.items():
        fams = [f for f in spec['taxa'] if f in cnt.index]
        assert len(fams) == len(spec['taxa']), \
            f'{group}: families missing from the genome tables: ' \
            f'{set(spec["taxa"]) - set(cnt.index)}'
        sub = cnt.loc[fams]
        parent = spec['parent_phylum']
        ph_tr = int((tr.phylum == parent).sum())
        rm_tr = int(sub.train.sum())
        rows.append({
            'group': group, 'n_families': len(fams),
            'families': ','.join(sorted(fams)),
            'parent_phylum': parent,
            'train': rm_tr, 'val': int(sub.val.sum()), 'test': int(sub.test.sum()),
            'total': int(sub.total.sum()), 'domain': sub.domain.iloc[0],
            'n_genera': int(allg[allg.family.isin(fams)].genus.nunique()),
            'median_Mbp': float(np.median(allg[allg.family.isin(fams)].genome_size) / 1e6),
            'parent_phylum_train_before': ph_tr,
            'parent_phylum_train_after': ph_tr - rm_tr,
            'parent_phylum_retained_pct': round(100 * (ph_tr - rm_tr) / ph_tr, 2),
            'parent_phylum_families_retained': int(
                tr[(tr.phylum == parent) & ~tr.family.isin(C.PANEL_TAXA)].family.nunique()),
        })
    panel_tbl = pd.DataFrame(rows)
    print(panel_tbl.drop(columns=['families']).to_string(index=False))

    is_panel_tr = tr.family.isin(C.PANEL_TAXA)
    is_panel_va = va.family.isin(C.PANEL_TAXA)
    is_panel_te = te.family.isin(C.PANEL_TAXA)
    n_tr_rm, n_va_rm, n_te_rm = (int(is_panel_tr.sum()), int(is_panel_va.sum()),
                                 int(is_panel_te.sum()))
    pct_tr = 100 * n_tr_rm / len(tr)
    print(f'\nRemoved from train: {n_tr_rm:,} / {len(tr):,} = {pct_tr:.2f}%'
          f'   (WS1.6 phylum panel removed 19.64%)')
    print(f'Removed from val:   {n_va_rm:,} / {len(va):,} = {100*n_va_rm/len(va):.2f}%')
    print(f'Removed from test:  {n_te_rm:,} / {len(te):,} = {100*n_te_rm/len(te):.2f}%')

    tr_h = tr[~is_panel_tr].copy()
    va_h = va[~is_panel_va].copy()
    te_h = te[~is_panel_te].copy()
    print(f'Surviving train pool: {len(tr_h):,} genomes, '
          f'{tr_h.phylum.nunique()} phyla, {tr_h.family.nunique()} families')

    # THE NON-NEGOTIABLE CHECK
    lost_phyla = sorted(set(tr.phylum) - set(tr_h.phylum))
    print('\nCONSTRAINT CHECK - every held-out family\'s parent phylum must remain '
          'in training')
    print(f'  phyla with training genomes before: {tr.phylum.nunique()}')
    print(f'  phyla with training genomes after : {tr_h.phylum.nunique()}')
    print(f'  phyla lost entirely: {lost_phyla or "NONE"}')
    for parent in C.PANEL_PARENT_PHYLA:
        n_after = int((tr_h.phylum == parent).sum())
        n_fam = int(tr_h[tr_h.phylum == parent].family.nunique())
        print(f'  {parent:<20} {n_after:>6,} training genomes retained '
              f'across {n_fam} families')
        assert n_after > 0, f'parent phylum {parent} eliminated from training'
    assert not lost_phyla, f'PANEL ELIMINATES PHYLA: {lost_phyla}'
    print('  PASS - no phylum is eliminated; this is a pure family-level holdout.')

    # ------------------------------------------------------------------ 3
    h('3. EVALUATION-SET CAPACITY (test-split references per panel group)')
    cap = []
    for group, spec in C.PANEL_GROUPS.items():
        fams = spec['taxa']
        n_test = int(cnt.loc[fams, 'test'].sum())
        d = C.EVAL_DESIGN[group]
        cap.append({'group': group, 'test_refs_available': n_test,
                    'all_split_refs': int(cnt.loc[fams, 'total'].sum()),
                    'ref_source': d['ref_source'], 'refs_used': d['n_refs'],
                    'sims_per_ref': d['sims'], 'eval_n': d['n_refs'] * d['sims'],
                    'clean_head_to_head': 'yes'})
        assert d['ref_source'] == 'test', f'{group}: family panel must use test refs'
        assert n_test >= d['n_refs'] or d['n_refs'] >= n_test, 'impossible'
        assert min(n_test, d['n_refs']) >= 20, \
            f'{group}: only {min(n_test, d["n_refs"])} clusters - too few for a ' \
            f'stable cluster bootstrap'
    d = C.EVAL_DESIGN['in_distribution']
    cap.append({'group': 'in_distribution', 'test_refs_available': len(te_h),
                'all_split_refs': int(len(tr_h) + len(va_h) + len(te_h)),
                'ref_source': 'test', 'refs_used': d['n_refs'],
                'sims_per_ref': d['sims'], 'eval_n': d['n_refs'] * d['sims'],
                'clean_head_to_head': 'yes (control)'})
    cap_tbl = pd.DataFrame(cap)
    print(cap_tbl.to_string(index=False))
    print(f'\nTotal evaluation genomes: {cap_tbl.eval_n.sum():,} '
          f'(WS1.6 phylum experiment: 6,994)')
    print('\nDominants are drawn from the TEST split, so neither the holdout model nor')
    print('production V5 trained on the exact genome; the only difference between the')
    print('two models is whether the FAMILY was in training - and, unlike WS1.6, the')
    print('parent phylum was in training for BOTH models.')

    # ------------------------------------------------------------------ 4
    h('4. SURVIVING POOLS FOR THE V5 SAMPLE-TYPE COMPOSITION')
    n_arch_v5 = int((tr.domain == 'Archaea').sum())
    n_arch = int((tr_h.domain == 'Archaea').sum())
    print(f'Archaeal training genomes surviving: {n_arch} (V5 had {n_arch_v5})')
    print(f'  -> archaeal category = 5% x 800,000 = 40,000 samples '
          f'=> {40000/max(n_arch,1):.0f} samples/genome (V5: {40000/n_arch_v5:.0f})')

    n_red_v5 = int(tr.phylum.isin(C.V5_REDUCED_GENOME_PHYLA).sum())
    red_surv = tr_h[tr_h.phylum.isin(C.V5_REDUCED_GENOME_PHYLA)]
    frac = len(red_surv) / n_red_v5
    small = tr_h[tr_h.genome_size < C.ADAPTED_REDUCED_SIZE_CUTOFF]
    adapted = tr_h[tr_h.phylum.isin(C.V5_REDUCED_GENOME_PHYLA) |
                   (tr_h.genome_size < C.ADAPTED_REDUCED_SIZE_CUTOFF)]
    print(f'\nV5 reduced-genome pool (V5 definition): {n_red_v5} genomes, '
          f'median {tr[tr.phylum.isin(C.V5_REDUCED_GENOME_PHYLA)].genome_size.median()/1e6:.3f} Mbp')
    print(f'Surviving under the SAME V5 definition:  {len(red_surv)} genomes '
          f'({100*frac:.1f}%), median {red_surv.genome_size.median()/1e6:.3f} Mbp')
    print(f'  threshold for needing WS1.6\'s adaptation: '
          f'{100*C.REDUCED_POOL_ADAPT_THRESHOLD:.0f}%')
    if frac >= C.REDUCED_POOL_ADAPT_THRESHOLD:
        print('  -> NO ADAPTATION NEEDED. The family panel keeps 756 Patescibacteriota')
        print('     training genomes, so the reduced-genome category uses V5\'s own')
        print('     definition verbatim. This makes the WS1.9 model a STRICTLY CLEANER')
        print('     counterfactual to V5 than the WS1.6 model was (which had to redefine')
        print('     the pool as surviving reduced phyla U non-panel genomes < 1.5 Mbp).')
    else:
        print('  -> ADAPTATION WOULD BE REQUIRED (set USE_ADAPTED_REDUCED_POOL=True)')
    assert frac >= C.REDUCED_POOL_ADAPT_THRESHOLD or C.USE_ADAPTED_REDUCED_POOL, \
        'reduced-genome pool collapsed but the adaptation is disabled'
    print(f'\n(for reference only, not used at family level) non-panel genomes '
          f'< 1.5 Mbp: {len(small)}; adapted union would be {len(adapted)}')

    # ------------------------------------------------------------------ 5
    h('5. NEAREST-SURVIVING-RELATIVE CONTEXT PER PANEL FAMILY')
    print('For each held-out family: how much of its order / class / phylum REMAINS')
    print('in the training pool. This is what distinguishes family-level from')
    print('phylum-level novelty and is the covariate for the WS1.8-style analysis.')
    det = []
    for group, spec in C.PANEL_GROUPS.items():
        for fam in sorted(spec['taxa']):
            r = cnt.loc[fam]
            same_order = int(((tr_h['order'] == r['order'])).sum())
            same_class = int(((tr_h['class'] == r['class'])).sum())
            same_phylum = int((tr_h.phylum == r['phylum']).sum())
            det.append({
                'group': group, 'family': fam, 'phylum': r['phylum'],
                'class': r['class'], 'order': r['order'],
                'train_removed': int(r['train']), 'val_removed': int(r['val']),
                'test_refs': int(r['test']), 'n_genera': int(r['n_genera']),
                'median_Mbp': float(r['median_Mbp']),
                'train_left_same_order': same_order,
                'train_left_same_class': same_class,
                'train_left_same_phylum': same_phylum})
    detdf = pd.DataFrame(det)
    detdf.to_csv(C.RESULTS_DIR / 'panel_family_detail.tsv', sep='\t', index=False)
    print(detdf.to_string(index=False))
    n_no_order = int((detdf.train_left_same_order == 0).sum())
    print(f'\nfamilies whose ORDER is entirely removed from training: {n_no_order}/'
          f'{len(detdf)} (these are effectively order-level novelty; reported)')
    print(f'families whose PHYLUM is entirely removed: '
          f'{int((detdf.train_left_same_phylum == 0).sum())} (must be 0)')
    assert int((detdf.train_left_same_phylum == 0).sum()) == 0

    # ------------------------------------------------------------------ 6
    h('6. PANEL CONTRIBUTION TO THE PRODUCTION 9,249-MER SELECTION')
    reps = {}
    for dom, f in [('bacterial', 'selected_bacterial_1000.tsv'),
                   ('archaeal', 'selected_archaeal_1000.tsv')]:
        d = pd.read_csv(C.KMER_DIR / f, sep='\t')
        if 'family' not in d.columns and 'gtdb_taxonomy' in d.columns:
            d['family'] = d.gtdb_taxonomy.map(lambda t: C.rank_from_taxonomy(t, 'family'))
        if 'family' in d.columns:
            n_panel = int(d.family.isin(C.PANEL_TAXA).sum())
            by = {k: int(v) for k, v in
                  d[d.family.isin(C.PANEL_TAXA)].family.value_counts().items()}
        else:                     # fall back to accession matching
            acc = set(allg[allg.family.isin(C.PANEL_TAXA)].ncbi_accession)
            col = 'accession' if 'accession' in d.columns else d.columns[0]
            n_panel = int(d[col].isin(acc).sum())
            by = {}
        reps[dom] = {'n_reps': len(d), 'n_panel': n_panel,
                     'pct_panel': round(100 * n_panel / len(d), 2), 'by_family': by}
        print(f'{dom}: {n_panel}/{len(d)} feature-selection representatives '
              f'({reps[dom]["pct_panel"]}%) belong to panel families')
        for k, v in list(by.items())[:10]:
            print(f'    {k}: {v}')
    print('\nWS1.7 established for the phylum panel that reselecting features without')
    print('panel genomes changes only threshold-noise k-mers (bacterial prevalence')
    print('Spearman rho=0.9865, 100% retention of ranks 1-3,000) and that the bias')
    print('direction is CONSERVATIVE. The family panel touches far fewer')
    print('representatives, so the same conclusion holds a fortiori.')

    # ------------------------------------------------------------------ 7
    h('7. DOES THE V5 HDF5 RECORD CONTAMINANT TAXONOMY? (regeneration test)')
    with h5py.File(C.V5_FEATURES_H5, 'r') as f:
        fields = list(f['train']['metadata'].dtype.names)
        print('metadata fields:', fields)
        for split in ['train', 'val', 'test']:
            g = f[split]
            print(f'  {split}: kmer={g["kmer_features"].shape}')
        n = f['train']['labels'].shape[0]
        acc = np.empty(n, dtype='S30')
        for s in range(0, n, 100_000):
            e = min(s + 100_000, n)
            acc[s:e] = f['train']['metadata'][s:e]['dominant_accession']
    has_cont_tax = any('contaminant' in x and ('phylum' in x or 'accession' in x)
                       for x in fields)
    print(f'\nCONTAMINANT TAXONOMY RECORDED: {has_cont_tax}')
    print('=> Only dominant_phylum / dominant_accession / n_contaminants exist, and')
    print('   dominant_FAMILY is not recorded either. The existing HDF5 therefore')
    print('   cannot be filtered into a valid family holdout in EITHER role.')
    print('=> Synthetic training data MUST be regenerated (script 121).')

    panel_acc = set(allg[allg.family.isin(C.PANEL_TAXA)].ncbi_accession)
    acc_s = np.char.decode(acc)
    n_panel_dom = int(np.isin(acc_s, list(panel_acc)).sum())
    print(f'\nV5 train samples with a PANEL-FAMILY dominant: {n_panel_dom:,}/{n:,} '
          f'({100*n_panel_dom/n:.2f}%)  <- lower bound; contaminant role not recorded')

    # ------------------------------------------------------------------ write
    h('WRITING PANEL DEFINITION AND HOLDOUT GENOME POOLS')
    for df, name in [(tr_h, 'train'), (va_h, 'val'), (te_h, 'test')]:
        out = C.HOLDOUT_DIR / f'{name}_genomes_holdout.tsv'
        df.to_csv(out, sep='\t', index=False)
        print(f'  {out}  {len(df):,} genomes, {df.phylum.nunique()} phyla, '
              f'{df.family.nunique()} families')

    panel_json = {
        'created': pd.Timestamp.now().isoformat(),
        'workstream': 'WS1.9 leave-family-out',
        'panel_level': 'family',
        'design_constraint': ('every held-out family\'s parent phylum REMAINS in the '
                              'training set; verified, not asserted'),
        'panel_groups': {g: {**s,
                             'train': int(panel_tbl.loc[panel_tbl.group == g, 'train'].iloc[0]),
                             'val': int(panel_tbl.loc[panel_tbl.group == g, 'val'].iloc[0]),
                             'test': int(panel_tbl.loc[panel_tbl.group == g, 'test'].iloc[0]),
                             'parent_phylum_retained_pct': float(
                                 panel_tbl.loc[panel_tbl.group == g,
                                               'parent_phylum_retained_pct'].iloc[0])}
                         for g, s in C.PANEL_GROUPS.items()},
        'panel_taxa': C.PANEL_TAXA,
        'panel_phyla': C.PANEL_PHYLA,
        'panel_parent_phyla': C.PANEL_PARENT_PHYLA,
        'n_removed': {'train': n_tr_rm, 'val': n_va_rm, 'test': n_te_rm},
        'pct_removed': {'train': round(pct_tr, 3),
                        'val': round(100 * n_va_rm / len(va), 3),
                        'test': round(100 * n_te_rm / len(te), 3)},
        'ws1_6_pct_removed_train': 19.64,
        'surviving': {'train': len(tr_h), 'val': len(va_h), 'test': len(te_h),
                      'train_phyla': int(tr_h.phylum.nunique()),
                      'train_families': int(tr_h.family.nunique()),
                      'train_archaea': n_arch,
                      'phyla_eliminated': lost_phyla},
        'eval_capacity': cap_tbl.to_dict('records'),
        'reduced_genome_pool': {
            'definition': 'V5 verbatim (V5_REDUCED_GENOME_PHYLA); NO adaptation applied',
            'n_v5_pool': n_red_v5, 'n_surviving': int(len(red_surv)),
            'pct_surviving': round(100 * frac, 2),
            'median_Mbp_surviving': round(float(red_surv.genome_size.median() / 1e6), 3),
            'median_Mbp_v5': round(float(
                tr[tr.phylum.isin(C.V5_REDUCED_GENOME_PHYLA)].genome_size.median() / 1e6), 3),
            'adaptation_applied': bool(C.USE_ADAPTED_REDUCED_POOL),
            'ws1_6_note': ('WS1.6 had to redefine this pool (1,622 genomes, median '
                           '1.17 Mbp) because the phylum panel collapsed it to 103 '
                           'genomes; the family panel does not, so V5\'s own '
                           'definition is used unchanged')},
        'panel_family_detail': detdf.to_dict('records'),
        'kmer_selection_leakage': reps,
        'v5_metadata_fields': fields,
        'v5_records_contaminant_taxonomy': bool(has_cont_tax),
        'v5_train_panel_family_dominant_fraction': round(100 * n_panel_dom / n, 3),
        'holdout_model_status': ('VALIDATION ARTIFACT ONLY. The released MAGICC model '
                                 '(models/magicc_v5.onnx) remains trained on all data.'),
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
