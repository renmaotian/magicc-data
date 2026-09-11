#!/usr/bin/env python3
"""Select ten established-name families using counts and taxonomy only.

The first stage intentionally reads no predictions or model metrics. Selection is
deterministic: >=50 training, >=5 validation and >=20 test genomes; established
family-name syntax; exclude Patescibacteriota; retain each phylum's largest family;
retain >=50% of each parent phylum. Select one eligible family per parent phylum
first, then fill to ten by training abundance, at most two families per phylum.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision/holdout_resubmission5'
SEED = 20260908


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {s: ROOT / f'data/splits/{s}_genomes.tsv' for s in ('train','val','test')}
    splits = {s: pd.read_csv(p, sep='\t') for s,p in inputs.items()}
    for d in splits.values():
        d['family'] = d.gtdb_taxonomy.str.extract(r'f__([^;]+)')
        d['fasta_path'] = d.fasta_path.str.replace('/home/tianrm/projects/magicc2', str(ROOT), regex=False)
        d['fasta_path'] = d.fasta_path.str.replace('/media/Data_1/tianrm/projects/magicc2', str(ROOT), regex=False)
    counts = pd.DataFrame({s:d.family.value_counts() for s,d in splits.items()}).fillna(0).astype(int)
    counts['phylum'] = splits['train'].groupby('family').phylum.first()
    counts['domain'] = splits['train'].groupby('family').domain.first()
    counts['parent_train'] = counts.phylum.map(splits['train'].phylum.value_counts())
    counts = counts.reset_index(names='family').sort_values(['train','family'], ascending=[False,True])
    largest = set(counts.groupby('phylum',sort=False).head(1).family)
    counts['largest_family_in_parent'] = counts.family.isin(largest)
    counts['eligible'] = ((counts.train>=50)&(counts.val>=5)&(counts.test>=20)
                          &counts.family.str.fullmatch(r'[A-Z][a-z]+aceae')
                          &counts.phylum.ne('Patescibacteriota')
                          &~counts.largest_family_in_parent
                          &(counts.train<=.5*counts.parent_train))
    cand = counts[counts.eligible]
    selected=[]
    removed={}
    number={}
    def admit(row):
        p=row.phylum
        if row.family in selected or number.get(p,0)>=2: return False
        if removed.get(p,0)+row.train > .5*row.parent_train: return False
        selected.append(row.family)
        removed[p]=removed.get(p,0)+int(row.train)
        number[p]=number.get(p,0)+1
        return True
    for row in cand.groupby('phylum',sort=False).head(1).itertuples():
        if len(selected)<10: admit(row)
    for row in cand.itertuples():
        if len(selected)<10: admit(row)
    assert len(selected)==10, selected
    counts['selected']=counts.family.isin(selected)
    counts.to_csv(OUT/'family_candidate_census.tsv',sep='\t',index=False)
    panel=counts[counts.selected].copy()
    panel['selection_order']=panel.family.map({f:i+1 for i,f in enumerate(selected)})
    panel['parent_remaining_train']=panel.parent_train-panel.phylum.map(removed)
    panel['parent_remaining_fraction']=panel.parent_remaining_train/panel.parent_train
    panel['eval_references']=panel.test.clip(upper=100)
    panel['simulations_per_reference']=10
    panel['eval_samples']=panel.eval_references*10
    panel.sort_values('selection_order').to_csv(OUT/'lineage_selection_manifest.tsv',sep='\t',index=False)
    for variant in ('holdout','matched_full'):
        dest=OUT/variant/'data'
        dest.mkdir(parents=True,exist_ok=True)
        for split,d in splits.items():
            keep=d[~d.family.isin(selected)] if variant=='holdout' else d
            keep.to_csv(dest/f'{split}_genomes_holdout.tsv',sep='\t',index=False)
        if variant=='holdout':
            assert splits['train'][~splits['train'].family.isin(selected)].phylum.nunique()==splits['train'].phylum.nunique()
    spec={
        'created_utc':pd.Timestamp.now(tz='UTC').isoformat(), 'seed':SEED,
        'selection_inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs.values()},
        'selection_script_sha256':sha(__file__),
        'rules':__doc__, 'rank':'family','families':selected,
        'parent_phyla':sorted(panel.phylum.unique()),
        'n_training_removed':int(panel.train.sum()),
        'n_training_total':len(splits['train']),
        'pct_training_removed':100*float(panel.train.sum())/len(splits['train']),
        'all_phyla_retained':True,
        'n_panel_eval_references':int(panel.eval_references.sum()),
        'n_panel_eval_samples':int(panel.eval_samples.sum()),
        'control':{'n_test_references':100,'simulations_per_reference':10},
        'primary_scope':'Sample-available established-name bacterial families; no archaeal family satisfies all rules.',
        'CPR':'Excluded from primary panel by user-requested scope; prior broad CPR analyses retained as sensitivity evidence.',
        'status':'SELECTION_LOCKED_BEFORE_NEW_PREDICTIONS',
    }
    (OUT/'panel_design.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(panel[['family','phylum','train','test','parent_remaining_fraction']].to_string(index=False))
    print(json.dumps({k:spec[k] for k in ('pct_training_removed','n_panel_eval_references','n_panel_eval_samples')},indent=2))


if __name__=='__main__': main()
