#!/usr/bin/env python3
"""Lock ten named genera using counts/taxonomy before any new model outcomes.

Eligibility: >=50 training, >=5 validation and >=20 test genomes; an unsuffixed
established-name genus ([A-Z][a-z]+); Patescibacteriota excluded from this primary
panel. Keep each parent family's largest genus and >=50% of family training
genomes. Choose one eligible genus per parent family first (training abundance,
then lexical name); fill by abundance, <=2 genera/family and <=4 genera/phylum.
Historical family and broad CPR experiments remain sensitivity evidence.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision/holdout_resubmission7'
SEED=20260928

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs={s:ROOT/f'data/splits/{s}_genomes.tsv' for s in ('train','val','test')}
    splits={s:pd.read_csv(p,sep='\t') for s,p in inputs.items()}
    for d in splits.values():
        for rank,prefix in [('family','f'),('genus','g'),('species','s')]:
            d[rank]=d.gtdb_taxonomy.str.extract(prefix+r'__([^;]+)')
        for old in ['/home/tianrm/projects/magicc2','/media/Data_1/tianrm/projects/magicc2']:
            d['fasta_path']=d.fasta_path.str.replace(old,str(ROOT),regex=False)
    counts=pd.DataFrame({s:d.genus.value_counts() for s,d in splits.items()}).fillna(0).astype(int)
    for col in ['family','phylum','domain']:counts[col]=splits['train'].groupby('genus')[col].first()
    counts['parent_train']=counts.family.map(splits['train'].family.value_counts())
    counts=counts.reset_index(names='genus').sort_values(['train','genus'],ascending=[False,True])
    largest=set(counts.groupby('family',sort=False).head(1).genus)
    counts['largest_genus_in_parent']=counts.genus.isin(largest)
    counts['eligible']=((counts.train>=50)&(counts.val>=5)&(counts.test>=20)
        &counts.genus.str.fullmatch(r'[A-Z][a-z]+')&counts.phylum.ne('Patescibacteriota')
        &~counts.largest_genus_in_parent&(counts.train<=.5*counts.parent_train))
    cand=counts[counts.eligible];selected=[];removed={};number={};phyla={}
    def admit(row):
        f=row.family;p=row.phylum
        if row.genus in selected or number.get(f,0)>=2 or phyla.get(p,0)>=4:return False
        if removed.get(f,0)+row.train>.5*row.parent_train:return False
        selected.append(row.genus);removed[f]=removed.get(f,0)+int(row.train)
        number[f]=number.get(f,0)+1;phyla[p]=phyla.get(p,0)+1
        return True
    for row in cand.groupby('family',sort=False).head(1).itertuples():
        if len(selected)<10:admit(row)
    for row in cand.itertuples():
        if len(selected)<10:admit(row)
    assert len(selected)==10,selected
    counts['selected']=counts.genus.isin(selected)
    panel=counts[counts.selected].copy()
    panel['selection_order']=panel.genus.map({g:i+1 for i,g in enumerate(selected)})
    panel['parent_remaining_train']=panel.parent_train-panel.family.map(removed)
    panel['parent_remaining_fraction']=panel.parent_remaining_train/panel.parent_train
    panel['eval_references']=panel.test.clip(upper=100)
    panel['simulations_per_reference']=10;panel['eval_samples']=panel.eval_references*10
    spec={'seed':SEED,'rank':'genus','genera':selected,
        'selection_inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs.values()},
        'selection_script_sha256':sha(__file__),'rules':__doc__,
        'parent_families':sorted(panel.family.unique()),'parent_phyla':sorted(panel.phylum.unique()),
        'n_training_removed':int(panel.train.sum()),'n_training_total':len(splits['train']),
        'pct_training_removed':100*float(panel.train.sum())/len(splits['train']),
        'all_families_retained':True,'all_phyla_retained':True,
        'n_panel_eval_references':int(panel.eval_references.sum()),'n_panel_eval_samples':int(panel.eval_samples.sum()),
        'control':{'n_test_references':100,'simulations_per_reference':10},
        'primary_scope':'Availability-defined named bacterial genera; no archaeal genus satisfies all eligibility rules.',
        'historical_scope':'Previous family and broad CPR/DPANN findings retained; this new panel is not a paired rank comparison.',
        'status':'SELECTION_LOCKED_BEFORE_NEW_PREDICTIONS'}
    path=OUT/'panel_design.json'
    if path.exists():
        old=json.loads(path.read_text());assert old==spec,'Locked panel/configuration changed; refuse overwrite.'
    else:path.write_text(json.dumps(spec,indent=2)+'\n')
    counts.to_csv(OUT/'genus_candidate_census.tsv',sep='\t',index=False)
    panel.sort_values('selection_order').to_csv(OUT/'lineage_selection_manifest.tsv',sep='\t',index=False)
    for variant in ('holdout','matched_full'):
        dest=OUT/variant/'data';dest.mkdir(parents=True,exist_ok=True)
        for split,d in splits.items():
            keep=d[~d.genus.isin(selected)] if variant=='holdout' else d
            keep.to_csv(dest/f'{split}_genomes_holdout.tsv',sep='\t',index=False)
        if variant=='holdout':
            tr=splits['train'];keep=tr[~tr.genus.isin(selected)]
            assert set(keep.family)==set(tr.family) and set(keep.phylum)==set(tr.phylum)
            assert not set(tr.loc[tr.genus.isin(selected),'species'])&set(keep.species)
    print(panel[['genus','family','phylum','train','val','test','parent_remaining_fraction']].to_string(index=False))
    print(json.dumps({k:spec[k] for k in ('pct_training_removed','n_panel_eval_references','n_panel_eval_samples')},indent=2))

if __name__=='__main__':main()
