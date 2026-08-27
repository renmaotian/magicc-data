#!/usr/bin/env python3
"""
WS11.G step 1 - EDA and held-out GENUS panel design.

Modelled line-for-line on 127_holdout_family_panel_eda.py (WS1.9), one rank deeper.

WHY THIS EXPERIMENT EXISTS
  WS1.6 held out whole PHYLA (difference-in-differences completeness degradation
  +3.80 to +22.81 pp) and WS1.9 held out FAMILIES inside retained phyla (+2.95 to
  +5.21 pp). The genus level was declined, justified by the argument that "a
  genus-level holdout inside a represented family would be expected to fall below
  the family-level cost". That argument assumes degradation is MONOTONE across
  ranks, and WS1.9's own Helicobacteraceae cell contradicts it: family-level
  novelty there cost as much as phylum-level novelty (attenuation -0.39 pp,
  p = 0.228). WS11.G measures the genus level instead of arguing about it.
  Monotonicity is not assumed anywhere in this experiment.

THE DEFINING DESIGN CONSTRAINT
      EVERY HELD-OUT GENUS'S PARENT FAMILY REMAINS IN THE TRAINING SET,
      and so does its parent phylum.
  This script verifies both numerically rather than asserting them, and
  quantifies exactly how much of each parent family survives.

Answers, with numbers:
  1. Genome counts per genus per split, with parent family, phylum, domain, size.
  2. Re-derivation of the panel from the split tables under the stated selection
     rule, checked against the literal panel recorded in holdout_lib/config.py.
  3. What the genus panel costs the training set, and - the non-negotiable checks
     - that every phylum AND every family still has training genomes afterwards.
  4. Whether each panel group has enough held-out TEST-split references for a
     stable cluster bootstrap.
  5. Whether the surviving training pool still supports the V5 sample-type
     composition (archaeal 5 %, reduced-genome 5 %) WITHOUT the WS1.6 adaptation.
  6. Nearest-surviving-relative context per panel genus (how many genomes of the
     same family / order / class / phylum remain in training).
  7. LADDER COMPLETENESS: every panel genus lies inside a family WS1.9 held out
     and a phylum WS1.6 held out, so the genus/family/phylum ladder can be
     computed on identical evaluation genomes (script 223).
  8. Panel contribution to the production 9,249-mer feature selection.
  9. Re-confirms that the V5 HDF5 cannot be filtered (no contaminant taxonomy),
     so regeneration is mandatory.

Outputs
  results/revision/holdout_genus/eda_genus_counts.tsv
  results/revision/holdout_genus/eda_panel_summary.json
  results/revision/holdout_genus/panel_genus_detail.tsv
  data/holdout_genus/holdout_panel.json          <- consumed by 121/123/124
  data/holdout_genus/train_genomes_holdout.tsv   <- panel-free training pool
  data/holdout_genus/val_genomes_holdout.tsv
  data/holdout_genus/test_genomes_holdout.tsv

Usage
  MAGICC_HOLDOUT_LEVEL=genus python scripts/220_holdout_genus_panel_eda.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_lib import config as C

assert C.PANEL_LEVEL == 'genus', 'run with MAGICC_HOLDOUT_LEVEL=genus'

RANKS = ['class', 'order', 'family', 'genus']


def h(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def derive_panel(gc):
    """Re-derive the held-out genus panel from the genome-count table.

    THE RULE (stated once, here and in holdout_lib/config.py):
      candidates are the genera of the six families WS1.9 held out; a genus is
      eligible if it has >= GENUS_PANEL_MIN_TRAIN training genomes and
      >= GENUS_PANEL_MIN_TEST test-split genomes; eligible genera are held out
      LARGEST-FIRST while the parent family still retains at least
      GENUS_PANEL_RETAIN_BIG (families with >= GENUS_PANEL_BIG_FAMILY training
      genomes) or GENUS_PANEL_RETAIN_SMALL (smaller families) of its training
      genomes, and at least GENUS_PANEL_MIN_RETAIN_GENERA genera.

    Returns {family: [genus, ...]} in the deterministic order the rule visits.
    """
    out = {}
    for fam in C.WS19_FAMILY_PANEL_TAXA:
        sub = gc[gc.family == fam].sort_values(['train', 'test', 'n_species'],
                                               ascending=False)
        total = int(sub.train.sum())
        if total == 0:
            out[fam] = []
            continue
        floor = (C.GENUS_PANEL_RETAIN_BIG if total >= C.GENUS_PANEL_BIG_FAMILY
                 else C.GENUS_PANEL_RETAIN_SMALL)
        elig = sub[(sub.train >= C.GENUS_PANEL_MIN_TRAIN) &
                   (sub.test >= C.GENUS_PANEL_MIN_TEST)]
        held, removed = [], 0
        for g, r in elig.iterrows():
            if total - (removed + int(r.train)) < floor * total:
                continue
            rest = [x for x in sub.index
                    if x not in held + [g] and int(gc.loc[x, 'train']) >= 1]
            if len(rest) < C.GENUS_PANEL_MIN_RETAIN_GENERA:
                continue
            held.append(g)
            removed += int(r.train)
        out[fam] = held
    return out


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1
    h('1. GENUS COUNTS PER SPLIT')
    dfs = {}
    for s, p in [('train', C.TRAIN_TSV), ('val', C.VAL_TSV), ('test', C.TEST_TSV)]:
        d = pd.read_csv(p, sep='\t')
        d['fasta_path'] = d.fasta_path.map(C.remap_fasta_path)
        for rank in RANKS + ['species']:
            d[rank] = d.gtdb_taxonomy.map(lambda t, r=rank: C.rank_from_taxonomy(t, r))
        d['split'] = s
        dfs[s] = d
    tr, va, te = dfs['train'], dfs['val'], dfs['test']
    allg = pd.concat(dfs.values(), ignore_index=True)
    print(f'train={len(tr):,}  val={len(va):,}  test={len(te):,}')
    print(f'{allg.phylum.nunique()} phyla, {allg.family.nunique()} families, '
          f'{allg.genus.nunique()} genera, {allg.species.nunique()} species')
    assert (allg.genus == '').sum() == 0, 'some genomes have no GTDB genus'
    assert allg.groupby('genus').family.nunique().max() == 1, \
        'a genus name spans >1 family; (family,genus) keying required'

    cnt = pd.DataFrame({'train': tr.genus.value_counts(),
                        'val': va.genus.value_counts(),
                        'test': te.genus.value_counts()}).fillna(0).astype(int)
    cnt['total'] = cnt.sum(axis=1)
    for col in ['family', 'order', 'class', 'phylum', 'domain']:
        cnt[col] = allg.groupby('genus')[col].first()
    cnt['n_species'] = allg.groupby('genus').species.nunique()
    cnt['median_Mbp'] = (allg.groupby('genus').genome_size.median() / 1e6).round(3)
    cnt['panel_group'] = [C.TAXON_TO_GROUP.get(g, '') for g in cnt.index]
    cnt = cnt.sort_values('train', ascending=False)
    cnt.index.name = 'genus'
    cnt.to_csv(C.RESULTS_DIR / 'eda_genus_counts.tsv', sep='\t')
    print(f'wrote eda_genus_counts.tsv ({len(cnt)} genera)')

    # ------------------------------------------------------------------ 2
    h('2. RE-DERIVING THE PANEL FROM THE SPLIT TABLES')
    print('The panel literal in holdout_lib/config.py must equal the panel the')
    print('stated selection rule produces from the genome tables. If a future GTDB')
    print('release changes the counts, this check fails loudly rather than')
    print('silently evaluating a different panel.')
    print(f'  rule constants: min_train={C.GENUS_PANEL_MIN_TRAIN} '
          f'min_test={C.GENUS_PANEL_MIN_TEST} '
          f'big_family={C.GENUS_PANEL_BIG_FAMILY} '
          f'retain_big={C.GENUS_PANEL_RETAIN_BIG} '
          f'retain_small={C.GENUS_PANEL_RETAIN_SMALL} '
          f'min_retain_genera={C.GENUS_PANEL_MIN_RETAIN_GENERA}')
    derived_by_family = derive_panel(cnt)
    derived = sorted({g for v in derived_by_family.values() for g in v})
    print(f'  derived: {len(derived)} genera in '
          f'{sum(1 for v in derived_by_family.values() if v)} families')
    print(f'  config : {len(C.PANEL_TAXA)} genera')
    only_derived = sorted(set(derived) - set(C.PANEL_TAXA))
    only_config = sorted(set(C.PANEL_TAXA) - set(derived))
    print(f'  in derived but not config: {only_derived or "NONE"}')
    print(f'  in config but not derived: {only_config or "NONE"}')
    assert set(derived) == set(C.PANEL_TAXA), \
        'the literal panel in config.py no longer matches the selection rule'
    print('  PASS - the recorded panel is exactly what the rule produces.')

    # ------------------------------------------------------------------ 3
    h('3. PANEL COST, AND THE PARENT-FAMILY / PARENT-PHYLUM CONSTRAINTS')
    rows = []
    for group, spec in C.PANEL_GROUPS.items():
        gen = [g for g in spec['taxa'] if g in cnt.index]
        assert len(gen) == len(spec['taxa']), \
            f'{group}: genera missing from the genome tables: ' \
            f'{set(spec["taxa"]) - set(cnt.index)}'
        sub = cnt.loc[gen]
        fams = sorted(sub.family.unique())
        parent = spec['parent_phylum']
        ph_tr = int((tr.phylum == parent).sum())
        fa_tr = int(tr.family.isin(fams).sum())
        rm_tr = int(sub.train.sum())
        rows.append({
            'group': group, 'n_genera': len(gen),
            'genera': ','.join(sorted(gen)),
            'parent_families': ','.join(fams), 'n_parent_families': len(fams),
            'parent_phylum': parent,
            'train': rm_tr, 'val': int(sub.val.sum()), 'test': int(sub.test.sum()),
            'total': int(sub.total.sum()), 'domain': sub.domain.iloc[0],
            'n_species': int(allg[allg.genus.isin(gen)].species.nunique()),
            'median_Mbp': float(np.median(allg[allg.genus.isin(gen)].genome_size) / 1e6),
            'parent_family_train_before': fa_tr,
            'parent_family_train_after': fa_tr - rm_tr,
            'parent_family_retained_pct': round(100 * (fa_tr - rm_tr) / fa_tr, 2),
            'parent_family_genera_retained': int(
                tr[tr.family.isin(fams) & ~tr.genus.isin(C.PANEL_TAXA)].genus.nunique()),
            'parent_phylum_train_before': ph_tr,
            'parent_phylum_train_after': ph_tr - rm_tr,
            'parent_phylum_retained_pct': round(100 * (ph_tr - rm_tr) / ph_tr, 2),
        })
    panel_tbl = pd.DataFrame(rows)
    print(panel_tbl.drop(columns=['genera', 'parent_families']).to_string(index=False))

    is_panel_tr = tr.genus.isin(C.PANEL_TAXA)
    is_panel_va = va.genus.isin(C.PANEL_TAXA)
    is_panel_te = te.genus.isin(C.PANEL_TAXA)
    n_tr_rm, n_va_rm, n_te_rm = (int(is_panel_tr.sum()), int(is_panel_va.sum()),
                                 int(is_panel_te.sum()))
    pct_tr = 100 * n_tr_rm / len(tr)
    print(f'\nRemoved from train: {n_tr_rm:,} / {len(tr):,} = {pct_tr:.2f}%'
          f'   (WS1.9 family panel 6.95%; WS1.6 phylum panel 19.64%)')
    print(f'Removed from val:   {n_va_rm:,} / {len(va):,} = {100*n_va_rm/len(va):.2f}%')
    print(f'Removed from test:  {n_te_rm:,} / {len(te):,} = {100*n_te_rm/len(te):.2f}%')

    tr_h = tr[~is_panel_tr].copy()
    va_h = va[~is_panel_va].copy()
    te_h = te[~is_panel_te].copy()
    print(f'Surviving train pool: {len(tr_h):,} genomes, '
          f'{tr_h.phylum.nunique()} phyla, {tr_h.family.nunique()} families, '
          f'{tr_h.genus.nunique()} genera')

    # THE TWO NON-NEGOTIABLE CHECKS
    lost_phyla = sorted(set(tr.phylum) - set(tr_h.phylum))
    lost_fams = sorted(set(tr.family) - set(tr_h.family))
    print('\nCONSTRAINT CHECK - every held-out genus\'s PARENT FAMILY and PARENT '
          'PHYLUM must remain in training')
    print(f'  phyla   with training genomes before/after: '
          f'{tr.phylum.nunique()} / {tr_h.phylum.nunique()}   '
          f'lost: {lost_phyla or "NONE"}')
    print(f'  families with training genomes before/after: '
          f'{tr.family.nunique()} / {tr_h.family.nunique()}   '
          f'lost: {lost_fams or "NONE"}')
    for fam in sorted({f for grp in C.PANEL_GROUPS.values()
                       for f in cnt.loc[[g for g in grp['taxa']], 'family'].unique()}):
        n_after = int((tr_h.family == fam).sum())
        n_gen = int(tr_h[tr_h.family == fam].genus.nunique())
        n_before = int((tr.family == fam).sum())
        print(f'  {fam:<24} {n_after:>5,} / {n_before:<5,} training genomes retained '
              f'({100*n_after/n_before:5.1f}%) across {n_gen} genera')
        assert n_after > 0, f'parent family {fam} eliminated from training'
        assert n_gen >= 1, f'parent family {fam} has no surviving genus'
    for parent in C.PANEL_PARENT_PHYLA:
        n_after = int((tr_h.phylum == parent).sum())
        print(f'  {parent:<24} {n_after:>5,} training genomes retained '
              f'({100*n_after/int((tr.phylum == parent).sum()):5.1f}%)')
        assert n_after > 0, f'parent phylum {parent} eliminated from training'
    assert not lost_phyla, f'PANEL ELIMINATES PHYLA: {lost_phyla}'
    assert not lost_fams, f'PANEL ELIMINATES FAMILIES: {lost_fams}'
    print('  PASS - no phylum and no family is eliminated; this is a pure '
          'genus-level holdout.')

    # ------------------------------------------------------------------ 4
    h('4. EVALUATION-SET CAPACITY (test-split references per panel group)')
    cap = []
    for group, spec in C.PANEL_GROUPS.items():
        gen = spec['taxa']
        n_test = int(cnt.loc[gen, 'test'].sum())
        d = C.EVAL_DESIGN[group]
        cap.append({'group': group, 'test_refs_available': n_test,
                    'all_split_refs': int(cnt.loc[gen, 'total'].sum()),
                    'ref_source': d['ref_source'], 'refs_used': d['n_refs'],
                    'sims_per_ref': d['sims'], 'eval_n': d['n_refs'] * d['sims'],
                    'clean_head_to_head': 'yes'})
        assert d['ref_source'] == 'test', f'{group}: genus panel must use test refs'
        assert d['n_refs'] <= n_test, \
            f'{group}: design asks for {d["n_refs"]} refs, only {n_test} available'
        assert d['n_refs'] >= 20, \
            f'{group}: only {d["n_refs"]} clusters - too few for a stable ' \
            f'cluster bootstrap'
    d = C.EVAL_DESIGN['in_distribution']
    cap.append({'group': 'in_distribution', 'test_refs_available': len(te_h),
                'all_split_refs': int(len(tr_h) + len(va_h) + len(te_h)),
                'ref_source': 'test', 'refs_used': d['n_refs'],
                'sims_per_ref': d['sims'], 'eval_n': d['n_refs'] * d['sims'],
                'clean_head_to_head': 'yes (control)'})
    cap_tbl = pd.DataFrame(cap)
    print(cap_tbl.to_string(index=False))
    print(f'\nTotal evaluation genomes: {cap_tbl.eval_n.sum():,} '
          f'(WS1.6 6,994; WS1.9 6,998)')
    print('\nDominants are drawn from the TEST split, so neither the holdout model nor')
    print('production V5 trained on the exact genome; the only difference between the')
    print('two models is whether the GENUS was in training - and, unlike WS1.6 and')
    print('WS1.9, the parent FAMILY was in training for BOTH models.')

    # ------------------------------------------------------------------ 5
    h('5. SURVIVING POOLS FOR THE V5 SAMPLE-TYPE COMPOSITION')
    n_arch_v5 = int((tr.domain == 'Archaea').sum())
    n_arch = int((tr_h.domain == 'Archaea').sum())
    print(f'Archaeal training genomes surviving: {n_arch} (V5 had {n_arch_v5})')
    print(f'  -> archaeal category = 5% x 800,000 = 40,000 samples '
          f'=> {40000/max(n_arch,1):.0f} samples/genome (V5: {40000/n_arch_v5:.0f})')

    n_red_v5 = int(tr.phylum.isin(C.V5_REDUCED_GENOME_PHYLA).sum())
    red_surv = tr_h[tr_h.phylum.isin(C.V5_REDUCED_GENOME_PHYLA)]
    frac = len(red_surv) / n_red_v5
    print(f'\nV5 reduced-genome pool (V5 definition): {n_red_v5} genomes, median '
          f'{tr[tr.phylum.isin(C.V5_REDUCED_GENOME_PHYLA)].genome_size.median()/1e6:.3f} Mbp')
    print(f'Surviving under the SAME V5 definition:  {len(red_surv)} genomes '
          f'({100*frac:.1f}%), median {red_surv.genome_size.median()/1e6:.3f} Mbp')
    print(f'  threshold for needing WS1.6\'s adaptation: '
          f'{100*C.REDUCED_POOL_ADAPT_THRESHOLD:.0f}%')
    if frac >= C.REDUCED_POOL_ADAPT_THRESHOLD:
        print('  -> NO ADAPTATION NEEDED. The genus panel removes only '
              f'{int(cnt.loc[[g for g in C.PANEL_TAXA if cnt.loc[g, "phylum"] == "Patescibacteriota"], "train"].sum())} '
              'Patescibacteriota training genomes, so the reduced-genome category')
        print('     uses V5\'s own definition verbatim. This makes WS11.G the '
              'CLEANEST of the')
        print('     three counterfactuals to V5 (WS1.6 needed a redefined pool; '
              'WS1.9 kept 63.5%).')
    else:
        print('  -> ADAPTATION WOULD BE REQUIRED (set USE_ADAPTED_REDUCED_POOL=True)')
    assert frac >= C.REDUCED_POOL_ADAPT_THRESHOLD or C.USE_ADAPTED_REDUCED_POOL, \
        'reduced-genome pool collapsed but the adaptation is disabled'

    # ------------------------------------------------------------------ 6
    h('6. NEAREST-SURVIVING-RELATIVE CONTEXT PER PANEL GENUS')
    print('For each held-out genus: how much of its family / order / class / phylum')
    print('REMAINS in the training pool. This is what distinguishes genus-level from')
    print('family- and phylum-level novelty and is the covariate for the WS1.8-style')
    print('analysis.')
    det = []
    for group, spec in C.PANEL_GROUPS.items():
        for gen in sorted(spec['taxa']):
            r = cnt.loc[gen]
            det.append({
                'group': group, 'genus': gen, 'family': r['family'],
                'phylum': r['phylum'], 'class': r['class'], 'order': r['order'],
                'train_removed': int(r['train']), 'val_removed': int(r['val']),
                'test_refs': int(r['test']), 'n_species': int(r['n_species']),
                'median_Mbp': float(r['median_Mbp']),
                'train_left_same_family': int((tr_h.family == r['family']).sum()),
                'genera_left_same_family': int(
                    tr_h[tr_h.family == r['family']].genus.nunique()),
                'train_left_same_order': int((tr_h['order'] == r['order']).sum()),
                'train_left_same_class': int((tr_h['class'] == r['class']).sum()),
                'train_left_same_phylum': int((tr_h.phylum == r['phylum']).sum()),
                'family_in_ws19_panel': r['family'] in C.WS19_FAMILY_PANEL_TAXA,
                'phylum_in_ws16_panel': r['phylum'] in C.WS16_PHYLUM_PANEL_TAXA})
    detdf = pd.DataFrame(det).sort_values(['group', 'train_removed'],
                                          ascending=[True, False])
    detdf.to_csv(C.RESULTS_DIR / 'panel_genus_detail.tsv', sep='\t', index=False)
    print(detdf.drop(columns=['class', 'order']).to_string(index=False))
    n_no_family = int((detdf.train_left_same_family == 0).sum())
    print(f'\ngenera whose FAMILY is entirely removed from training: {n_no_family} '
          f'(must be 0)')
    print(f'genera whose PHYLUM is entirely removed: '
          f'{int((detdf.train_left_same_phylum == 0).sum())} (must be 0)')
    assert n_no_family == 0
    assert int((detdf.train_left_same_phylum == 0).sum()) == 0

    # ------------------------------------------------------------------ 7
    h('7. LADDER COMPLETENESS (genus -> family -> phylum on identical genomes)')
    print('Every panel genus must lie inside a family the WS1.9 model never saw AND')
    print('a phylum the WS1.6 model never saw. Then the SAME evaluation genome is')
    print('  genus-novel   to models/magicc_holdout_genus.onnx   (WS11.G)')
    print('  family-novel  to models/magicc_holdout_family.onnx  (WS1.9)')
    print('  phylum-novel  to models/magicc_holdout_phylum.onnx  (WS1.6)')
    print('  represented   to models/magicc_v5.onnx              (production)')
    print('and the three novelty effects are paired, not glued together.')
    bad_fam = sorted(detdf.loc[~detdf.family_in_ws19_panel, 'genus'])
    bad_phy = sorted(detdf.loc[~detdf.phylum_in_ws16_panel, 'genus'])
    print(f'  genera whose family is NOT in the WS1.9 panel: {bad_fam or "NONE"}')
    print(f'  genera whose phylum is NOT in the WS1.6 panel: {bad_phy or "NONE"}')
    assert not bad_fam and not bad_phy, 'ladder is not complete'
    print(f'  PASS - all {len(detdf)} panel genera are ladder-complete '
          f'({detdf.family.nunique()} families, {detdf.phylum.nunique()} phyla).')

    # ------------------------------------------------------------------ 8
    h('8. PANEL CONTRIBUTION TO THE PRODUCTION 9,249-MER SELECTION')
    reps = {}
    for dom, fname in [('bacterial', 'selected_bacterial_1000.tsv'),
                       ('archaeal', 'selected_archaeal_1000.tsv')]:
        d = pd.read_csv(C.KMER_DIR / fname, sep='\t')
        C.add_taxon_column(d)
        n_panel = int(d[C.TAXON_COL].isin(C.PANEL_TAXA).sum())
        by = {k: int(v) for k, v in
              d.loc[d[C.TAXON_COL].isin(C.PANEL_TAXA), C.TAXON_COL]
              .value_counts().items()}
        reps[dom] = {'n_reps': len(d), 'n_panel': n_panel,
                     'pct_panel': round(100 * n_panel / len(d), 2), 'by_genus': by}
        print(f'{dom}: {n_panel}/{len(d)} feature-selection representatives '
              f'({reps[dom]["pct_panel"]}%) belong to panel genera')
        for k, v in list(by.items())[:12]:
            print(f'    {k}: {v}')
    print('\nWS1.7 established for the phylum panel (18.7% bacterial / 42.6% archaeal)')
    print('that reselecting features without panel genomes changes only threshold-noise')
    print('k-mers (bacterial prevalence Spearman rho=0.9865, 100% retention of ranks')
    print('1-3,000) and that the bias direction is CONSERVATIVE. The genus panel touches')
    print('far fewer representatives, so the same conclusion holds a fortiori; the')
    print('control is run anyway (script 125 / 222).')

    # ------------------------------------------------------------------ 9
    h('9. DOES THE V5 HDF5 RECORD CONTAMINANT TAXONOMY? (regeneration test)')
    with h5py.File(C.V5_FEATURES_H5, 'r') as f:
        fields = list(f['train']['metadata'].dtype.names)
        print('metadata fields:', fields)
        n = f['train']['labels'].shape[0]
        acc = np.empty(n, dtype='S30')
        for s in range(0, n, 100_000):
            e = min(s + 100_000, n)
            acc[s:e] = f['train']['metadata'][s:e]['dominant_accession']
    has_cont_tax = any('contaminant' in x and ('phylum' in x or 'accession' in x)
                       for x in fields)
    print(f'\nCONTAMINANT TAXONOMY RECORDED: {has_cont_tax}')
    print('=> Only dominant_phylum / dominant_accession / n_contaminants exist, and')
    print('   dominant_GENUS is not recorded either. The existing HDF5 therefore')
    print('   cannot be filtered into a valid genus holdout in EITHER role.')
    print('=> Synthetic training data MUST be regenerated (script 121).')

    panel_acc = set(allg.loc[allg.genus.isin(C.PANEL_TAXA), 'ncbi_accession'])
    acc_s = np.char.decode(acc)
    n_panel_dom = int(np.isin(acc_s, list(panel_acc)).sum())
    print(f'\nV5 train samples with a PANEL-GENUS dominant: {n_panel_dom:,}/{n:,} '
          f'({100*n_panel_dom/n:.2f}%)  <- lower bound; contaminant role not recorded')

    # ------------------------------------------------------------------ write
    h('WRITING PANEL DEFINITION AND HOLDOUT GENOME POOLS')
    for df, name in [(tr_h, 'train'), (va_h, 'val'), (te_h, 'test')]:
        out = C.HOLDOUT_DIR / f'{name}_genomes_holdout.tsv'
        df.to_csv(out, sep='\t', index=False)
        print(f'  {out}  {len(df):,} genomes, {df.phylum.nunique()} phyla, '
              f'{df.family.nunique()} families, {df.genus.nunique()} genera')

    panel_json = {
        'created': pd.Timestamp.now().isoformat(),
        'workstream': 'WS11.G leave-genus-out',
        'panel_level': 'genus',
        'design_constraint': ('every held-out genus\'s PARENT FAMILY and PARENT '
                              'PHYLUM REMAIN in the training set; verified, not '
                              'asserted'),
        'selection_rule': {
            'candidates': 'genera of the six families WS1.9 held out',
            'min_train_genomes': C.GENUS_PANEL_MIN_TRAIN,
            'min_test_genomes': C.GENUS_PANEL_MIN_TEST,
            'big_family_threshold_train_genomes': C.GENUS_PANEL_BIG_FAMILY,
            'family_retention_floor_big': C.GENUS_PANEL_RETAIN_BIG,
            'family_retention_floor_small': C.GENUS_PANEL_RETAIN_SMALL,
            'min_retained_genera_per_family': C.GENUS_PANEL_MIN_RETAIN_GENERA,
            'order': 'largest training count first',
            'rederivation_check': 'PASS (this run)'},
        'panel_groups': {g: {**s,
                             'train': int(panel_tbl.loc[panel_tbl.group == g, 'train'].iloc[0]),
                             'val': int(panel_tbl.loc[panel_tbl.group == g, 'val'].iloc[0]),
                             'test': int(panel_tbl.loc[panel_tbl.group == g, 'test'].iloc[0]),
                             'parent_family_retained_pct': float(
                                 panel_tbl.loc[panel_tbl.group == g,
                                               'parent_family_retained_pct'].iloc[0]),
                             'parent_phylum_retained_pct': float(
                                 panel_tbl.loc[panel_tbl.group == g,
                                               'parent_phylum_retained_pct'].iloc[0])}
                         for g, s in C.PANEL_GROUPS.items()},
        'panel_taxa': C.PANEL_TAXA,
        'panel_phyla': C.PANEL_PHYLA,
        'panel_parent_phyla': C.PANEL_PARENT_PHYLA,
        'panel_parent_families': sorted(detdf.family.unique().tolist()),
        'ws19_family_panel': C.WS19_FAMILY_PANEL_TAXA,
        'ws16_phylum_panel': C.WS16_PHYLUM_PANEL_TAXA,
        'ladder_complete': True,
        'n_removed': {'train': n_tr_rm, 'val': n_va_rm, 'test': n_te_rm},
        'pct_removed': {'train': round(pct_tr, 3),
                        'val': round(100 * n_va_rm / len(va), 3),
                        'test': round(100 * n_te_rm / len(te), 3)},
        'ws1_6_pct_removed_train': 19.64,
        'ws1_9_pct_removed_train': 6.95,
        'surviving': {'train': len(tr_h), 'val': len(va_h), 'test': len(te_h),
                      'train_phyla': int(tr_h.phylum.nunique()),
                      'train_families': int(tr_h.family.nunique()),
                      'train_genera': int(tr_h.genus.nunique()),
                      'train_archaea': n_arch,
                      'phyla_eliminated': lost_phyla,
                      'families_eliminated': lost_fams},
        'eval_capacity': cap_tbl.to_dict('records'),
        'reduced_genome_pool': {
            'definition': 'V5 verbatim (V5_REDUCED_GENOME_PHYLA); NO adaptation applied',
            'n_v5_pool': n_red_v5, 'n_surviving': int(len(red_surv)),
            'pct_surviving': round(100 * frac, 2),
            'median_Mbp_surviving': round(float(red_surv.genome_size.median() / 1e6), 3),
            'median_Mbp_v5': round(float(
                tr[tr.phylum.isin(C.V5_REDUCED_GENOME_PHYLA)].genome_size.median() / 1e6), 3),
            'adaptation_applied': bool(C.USE_ADAPTED_REDUCED_POOL)},
        'panel_genus_detail': detdf.to_dict('records'),
        'kmer_selection_leakage': reps,
        'v5_metadata_fields': fields,
        'v5_records_contaminant_taxonomy': bool(has_cont_tax),
        'v5_train_panel_genus_dominant_fraction': round(100 * n_panel_dom / n, 3),
        'monotonicity_note': ('This experiment exists BECAUSE monotonicity across '
                              'ranks cannot be assumed: WS1.9 Helicobacteraceae showed '
                              'family-level novelty costing as much as phylum-level '
                              '(attenuation -0.39 pp, p = 0.228). The genus cost is '
                              'measured here, never inferred from the family cost.'),
        'holdout_model_status': ('VALIDATION ARTIFACT ONLY. The released MAGICC model '
                                 '(models/magicc_v5.onnx) remains trained on all data.'),
    }
    with open(C.PANEL_JSON, 'w') as f:
        json.dump(panel_json, f, indent=2, default=str)
    with open(C.RESULTS_DIR / 'eda_panel_summary.json', 'w') as f:
        json.dump(panel_json, f, indent=2, default=str)
    print(f'\n  {C.PANEL_JSON}')
    print(f'  {C.RESULTS_DIR / "eda_panel_summary.json"}')
    print('\nEDA COMPLETE')


if __name__ == '__main__':
    main()
