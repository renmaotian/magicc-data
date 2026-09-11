#!/usr/bin/env python3
"""Restrict CAMI scientific summaries to provenance-verified microbial sources.

Preserves every prefilter result under construction audit. Neither circular
elements nor mixtures containing them support genome-quality validation.
"""
import hashlib,json,shutil
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'cami_microbial_scope';AN=WORK/'results/revision/cami2/analysis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(exist_ok=True);snapshot=DEST/'before_scope';snapshot.mkdir(exist_ok=True)
    for p in AN.iterdir():
        if p.is_file() and p.suffix in ['.tsv','.json'] and not (snapshot/p.name).exists():shutil.copy2(p,snapshot/p.name)
    status=OUT/'analysis_status/201.json'
    if not (DEST/'201_before_scope.json').exists():shutil.copy2(status,DEST/'201_before_scope.json')
    path=WORK/'scripts/201_ws38_cami2_analysis.py';s=path.read_text()
    if 'def microbial_source_ids(' not in s and not (snapshot/'201_script_before_scope.py').exists():
        shutil.copy2(path,snapshot/'201_script_before_scope.py')
    if 'def microbial_source_ids(' not in s:
        helper='''def microbial_source_ids(dataset):
    setup = 'simulation_short_read' if dataset == 'marine' else 'short_read'
    path = PROJECT_DIR / 'data/real_data/cami2' / dataset / 'setup' / setup / 'metadata.tsv'
    meta = pd.read_csv(path, sep='\\t')
    assert meta.genome_ID.is_unique
    microbial = set(meta.loc[meta.genome_ID.str.startswith('Otu'), 'genome_ID']) if dataset == 'marine' else set(meta.genome_ID)
    assert len(microbial) == (777 if dataset == 'marine' else 408)
    if dataset == 'marine': assert meta.genome_ID.str.startswith(('Otu', 'RNODE_')).all()
    return microbial


def apply_microbial_scope(long):
    valid = {dataset: microbial_source_ids(dataset) for dataset in ('marine', 'strain_madness')}
    mask = [dom in valid[dataset] and (donor in valid[dataset] if isinstance(donor, str) and donor else binset == 'gold')
            for dataset, dom, donor, binset in zip(long.dataset, long.dominant, long.contaminant, long.binset)]
    return long.loc[mask].copy()


'''
        marker='# ------------------------------------------------------------------ main\n';assert marker in s;s=s.replace(marker,helper+marker)
        needle='    long = attach_leakage(load_long())';assert s.count(needle)==1;s=s.replace(needle,'    long = apply_microbial_scope(attach_leakage(load_long()))')
        needle="            g = pd.read_csv(p, sep='\\t')";assert s.count(needle)==1;s=s.replace(needle,needle+"\n            g = g[g.genome.isin(microbial_source_ids(ds))].copy()")
    s=s.replace("('all_attempted', np.ones(len(sub), dtype=bool))","('all_prediction_rows_microbial_scope', np.ones(len(sub), dtype=bool))")
    s=s.replace("'all_prediction_rows_microbi al_scope'.replace(' ', '')", "'all_prediction_rows_microbial_scope'")
    s=s.replace("microbial = set(meta.loc[meta.genome_ID.str.startswith('Otu'), 'genome_ID'])\n", "microbial = set(meta.loc[meta.genome_ID.str.startswith('Otu'), 'genome_ID']) if dataset == 'marine' else set(meta.genome_ID)\n")
    s=s.replace("    assert meta.genome_ID.str.startswith(('Otu', 'RNODE_')).all()", "    if dataset == 'marine': assert meta.genome_ID.str.startswith(('Otu', 'RNODE_')).all()")
    s=s.replace('n_attempted=len(zz)','n_prediction_rows=len(zz)')
    old_guard="    mask = [dom in valid[dataset] and (not isinstance(donor, str) or not donor or donor in valid[dataset])\n            for dataset, dom, donor in zip(long.dataset, long.dominant, long.contaminant)]"
    new_guard="    mask = [dom in valid[dataset] and (donor in valid[dataset] if isinstance(donor, str) and donor else binset == 'gold')\n            for dataset, dom, donor, binset in zip(long.dataset, long.dominant, long.contaminant, long.binset)]"
    if old_guard in s:
        prior=DEST/'201_script_before_donor_guard.py'
        if not prior.exists():prior.write_text(s)
        s=s.replace(old_guard,new_guard)
    path.write_text(s)
    # Record original/cohort construction separately from valid analysis units.
    accounting=[];direct_inputs=[]
    for ds,setup in [('marine','simulation_short_read'),('strain_madness','short_read')]:
        meta=pd.read_csv(ROOT/f'data/real_data/cami2/{ds}/setup/{setup}/metadata.tsv',sep='\t');valid=set(meta.loc[meta.genome_ID.str.startswith('Otu'),'genome_ID']) if ds=='marine' else set(meta.genome_ID)
        audit=pd.read_csv(ROOT/f'results/revision/cami2/provenance/{ds}_source_genome_audit.tsv',sep='\t')
        direct_inputs += [ROOT/f'data/real_data/cami2/{ds}/setup/{setup}/metadata.tsv', ROOT/f'results/revision/cami2/provenance/{ds}_source_genome_audit.tsv']
        for label,z in [('historical_source_audit',audit),('microbial_source_audit',audit[audit.genome.isin(valid)])]:
            accounting.append(dict(dataset=ds,table=label,n_sources=len(z),n_overlap_sources=int(z.leaked.sum()),n_no_detected_overlap_sources=int((~z.leaked).sum())))
        for bs in ['gold','mixed']:
            truth=pd.read_csv(ROOT/f'results/revision/cami2/truth/{ds}_{bs}_truth.tsv',sep='\t');mask=truth.genome.isin(valid) if bs=='gold' else truth.dominant.isin(valid)&truth.contaminant.isin(valid)
            direct_inputs.append(ROOT/f'results/revision/cami2/truth/{ds}_{bs}_truth.tsv')
            if bs=='mixed':assert truth.dominant.notna().all() and truth.contaminant.notna().all()
            for label,z in [('historical_all_truth',truth),('microbial_truth',truth[mask])]:accounting.append(dict(dataset=ds,table=bs+'_'+label,n_sources=None,n_truth_rows=len(z),n_below50=int(z.completeness_pct.lt(50).sum()),fraction_below50=float(z.completeness_pct.lt(50).mean())))
    pd.DataFrame(accounting).to_csv(DEST/'source_and_truth_accounting.tsv',sep='\t',index=False)
    inputs=[Path(__file__).resolve(),ROOT/'scripts/267_audit_cami_marine_source_scope.py',ROOT/'results/revision/holdout_resubmission5/cami_source_audit/audit_summary.json',*direct_inputs]
    result=dict(status='CAMI_MICROBIAL_SCOPE_PREPARED',scientific_scope='Otu microbial source IDs from native CAMI setup metadata; both dominant and donor eligible. RNODE circular elements remain construction-audit records only.',primary_expectation='Corrected four-tool primary membership unchanged, verified after replay.',input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs],modified_script=dict(path=str(path.relative_to(ROOT)),sha256=sha(path)),historical_outputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(snapshot.iterdir())])
    (DEST/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(pd.DataFrame(accounting).to_string(index=False))
if __name__=='__main__':main()
