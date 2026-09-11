#!/usr/bin/env python3
"""Full five-set pooled CI: shared accessions, equal set weights in every draw.

All 5,000 samples and point estimates are retained. A dominant accession is
resampled once across all sets, inducing the same multiplicity wherever it
recurs. Within each resample each set contributes weight 1/5. Sufficient
statistics make this exactly equivalent to resampling complete reference
blocks then normalizing the five set weights, without materializing rows.
"""
import hashlib,importlib.util,json,sys,shutil
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'pooled_cluster_correction'
SETS=['set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E'];TOOLS=['magicc_v5','checkm2','cocopye','deepcheck'];LABEL='POOLED_leakage_free_5_sets'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    spec=importlib.util.spec_from_file_location('pooled101',WORK/'scripts/101_metrics_framework.py');fw=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fw;spec.loader.exec_module(fw)
    cfg=fw.load_config(WORK/'scripts/config_revision_metrics.yaml');objs={b.name:b for b in fw.discover_sets(cfg,include_missing=False)};frames=[]
    for name in SETS:
        d,tools,_=fw.load_set(cfg,objs[name]);assert len(d)==1000 and set(TOOLS)<=set(tools)
        assert np.isfinite(d[[f'pred_{m}__{t}' for t in TOOLS for m in ['completeness','contamination']]]).all().all()
        d['set']=name;frames.append(d)
    refs=sorted(set().union(*(set(d.cluster_id) for d in frames)));nrefs=len(refs);assert nrefs==1648
    refindex={a:i for i,a in enumerate(refs)};rng=np.random.default_rng(cfg.seed+777)
    counts=rng.multinomial(nrefs,np.full(nrefs,1/nrefs),size=cfg.n_boot).astype(float)
    counts=np.concatenate([np.ones((1,nrefs)),counts]);results=[];independent_checks=[]
    for tool in TOOLS:
        for metric in ['completeness','contamination']:
            drawstats=[]
            for name,d in zip(SETS,frames):
                t=d[f'true_{metric}'].to_numpy(float);p=d[f'pred_{metric}__{tool}'].to_numpy(float);e=p-t
                values=np.column_stack([np.ones(len(d)),t,t*t,p,p*p,t*p,np.abs(e),e*e,e])
                idx=np.array([refindex[a] for a in d.cluster_id]);sums=np.zeros((nrefs,9));np.add.at(sums,idx,values)
                weighted=counts@sums;assert (weighted[:,0]>0).all();drawstats.append(weighted[:,1:]/weighted[:,0,None])
                # Direct row weights independently verify the aggregate method
                # on one nontrivial draw for every set/output/tool combination.
                rw=counts[1,idx];direct=np.average(values[:,1:],axis=0,weights=rw)
                err=float(np.max(np.abs(direct-drawstats[-1][1])));assert err<1e-9
                independent_checks.append(dict(set=name,tool=tool,metric=metric,max_sufficient_stat_difference=err))
            avg=np.mean(drawstats,axis=0);t,t2,p,p2,tp,ae,e2,e=avg.T
            quantities={'mae':ae,'rmse':np.sqrt(e2),'bias':e,'r2':1-e2/(t2-t*t),'r2_pearson_sq':((tp-t*p)**2)/((t2-t*t)*(p2-p*p))}
            row=dict(set=LABEL,tool=tool,metric=metric,n=5000,n_clusters=nrefs)
            for name,vals in quantities.items():
                row[name]=float(vals[0]);row[name+'_ci_lo'],row[name+'_ci_hi']=map(float,np.quantile(vals[1:],[.025,.975]))
            results.append(row)
    DEST.mkdir(exist_ok=True);table=WORK/'results/revision/metrics/ws5.5_table_S2_rebuilt.tsv';a=pd.read_csv(table,sep='\t');mask=a.set.eq(LABEL);old=a.loc[mask].copy();assert len(old)==8
    snapshot=DEST/'ws5.5_before_pooled_ci_correction.tsv'
    if not snapshot.exists():shutil.copy2(table,snapshot)
    stagepath=OUT/'analysis_status/104.json'
    if stagepath.exists():
        stage=json.loads(stagepath.read_text())
        for item in stage.get('outputs',[]):
            if item['path']==str(table.relative_to(ROOT)):
                assert item['sha256']==sha(snapshot)
                item['path']=str(snapshot.relative_to(ROOT));item['preservation_note']='Exact stage-104 output retained before script257 replaces pooled CI rows; final source is the stage257 output.'
        stagepath.write_text(json.dumps(stage,indent=2)+'\n')
    historical=DEST/'historical_set_by_reference.tsv'
    if not historical.exists():old.to_csv(historical,sep='\t',index=False)
    corrected=pd.DataFrame(results)
    for row in corrected.to_dict('records'):
        select=mask & a.tool.eq(row['tool']) & a.metric.eq(row['metric']);assert select.sum()==1
        i=a.index[select][0]
        for name in ['mae','rmse','bias','r2','r2_pearson_sq']:assert abs(float(a.loc[i,name])-row[name])<1e-10
        for name,value in row.items():
            if name not in ['set','tool','metric']:a.loc[i,name]=value
    a.loc[mask,'set_label']='Pooled five-set benchmark (equal set weights; shared dominant-accession clusters)'
    a.to_csv(table,sep='\t',index=False);corrected.to_csv(DEST/'corrected_pooled_metrics.tsv',sep='\t',index=False);pd.DataFrame(independent_checks).to_csv(DEST/'direct_weight_equivalence.tsv',sep='\t',index=False)
    pd.DataFrame([dict(accession=acc,n_sets=sum(acc in set(d.cluster_id) for d in frames),n_samples=sum(int(d.cluster_id.eq(acc).sum()) for d in frames)) for acc in refs]).to_csv(DEST/'reference_membership.tsv',sep='\t',index=False)
    summary=dict(status='POOLED_CI_CORRECTION_COMPLETE',n_samples=5000,n_distinct_dominant_accessions=nrefs,n_set_by_reference_blocks=sum(d.cluster_id.nunique() for d in frames),n_bootstrap=cfg.n_boot,seed=cfg.seed+777,method='Multinomial resampling of shared dominant-accession blocks; normalize each set to weight 1/5 within every draw; 95% percentile intervals',point_estimates_unchanged=True,per_set_tests_unchanged=True,input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [Path(__file__).resolve(),WORK/'scripts/config_revision_metrics.yaml',OUT/'corrected_prediction_manifest.tsv']],output_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(DEST.iterdir()) if p.suffix=='.tsv'])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
