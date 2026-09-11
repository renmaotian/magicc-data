#!/usr/bin/env python3
"""Recover and rebuild the historical CAMI realized-dose sensitivity.

First independently verifies every historical point/count from original rows.
Then applies the identical truth filter to the corrected common scoring cohort.
The historical ad-hoc generator was not retained; new intervals explicitly use
2,000 reference-cluster resamples and recorded CRC32-derived seeds.
"""
import hashlib,importlib.util,json,sys,zlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'cami_controlled_dose_recovery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cohort(d):return d[d.binset.eq('mixed') & d.in_domain & d.scoreable & d.leakage_free & d.all_tools_scored & d.true_contamination.le(1.5*d.target_contamination)].copy()
def slope(t,p):return float(np.polyfit(t,p,1)[0]) if np.std(t)>1e-8 else np.nan
def main():
    DEST.mkdir(exist_ok=True);original=ROOT/'results/revision/cami2/analysis';replay=WORK/'results/revision/cami2/analysis'
    oldlong=pd.read_csv(original/'cami2_long_predictions.tsv',sep='\t');old=cohort(oldlong);recorded=pd.read_csv(original/'cami2_wellcontrolled_by_distance.tsv',sep='\t');checks=[]
    for r in recorded.itertuples():
        z=old[old.dataset.eq(r.dataset)&old.tool.eq(r.tool)&old.distance_rank.eq(r.distance_rank)];assert len(z)==r.n and z.cluster.nunique()==r.n_clusters
        t=z.true_contamination.to_numpy(float);p=z.pred_contamination.to_numpy(float)
        for name,value in [('cont_bias',float(np.mean(p-t))),('cont_mae',float(np.mean(np.abs(p-t)))),('slope',slope(t,p))]:
            err=abs(value-getattr(r,name));assert err<1e-10,(r.dataset,r.tool,r.distance_rank,name,err);checks.append(dict(dataset=r.dataset,tool=r.tool,distance_rank=r.distance_rank,quantity=name,absolute_difference=err,n=len(z),n_clusters=z.cluster.nunique()))
    d=pd.read_csv(replay/'cami2_long_predictions.tsv',sep='\t');c=cohort(d);assert c.tool_scored.all()
    spec=importlib.util.spec_from_file_location('controlled101',WORK/'scripts/101_metrics_framework.py');fw=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fw;spec.loader.exec_module(fw)
    rows=[]
    for (ds,tool,rank),z in c.groupby(['dataset','tool','distance_rank'],sort=True):
        t=z.true_contamination.to_numpy(float);p=z.pred_contamination.to_numpy(float);seed=20260726+(zlib.crc32(f'controlled_dose|{ds}|{tool}|{rank}'.encode())&0xffffffff)
        bs=fw.Bootstrapper(clusters=z.cluster.to_numpy(),n_iter=2000,seed=seed)
        ci=bs.ci_multi(lambda i:{'cont_bias':float(np.mean(p[i]-t[i])),'cont_mae':float(np.mean(np.abs(p[i]-t[i]))),'slope':slope(t[i],p[i])})
        r=dict(dataset=ds,tool=tool,distance_rank=rank,n=len(z),n_clusters=int(z.cluster.nunique()),filter='realised contamination <= 1.5 x target; training-overlap-screened, in-domain, completeness >=50%, scored by all four tools',note='Truth-based dose sensitivity. Exact realized truth is retained; stage-1 unscored CoCoPyE rows are excluded from the shared quantitative cohort and counted in coverage.',denominator='Both truth percentages use full dominant-reference bp as denominator.',bootstrap_iterations=2000,seed=seed,cluster_definition='Dominant source reference within dataset')
        for name in ci:r[name]=ci[name]['estimate'];r[name+'_lo']=ci[name]['ci_lo'];r[name+'_hi']=ci[name]['ci_hi']
        rows.append(r)
    pd.DataFrame(rows).to_csv(replay/'cami2_wellcontrolled_by_distance.tsv',sep='\t',index=False);pd.DataFrame(checks).to_csv(DEST/'historical_point_reconstruction.tsv',sep='\t',index=False)
    coverage=[]
    for ds in ['marine','strain_madness']:
        for label,frame in [('historical',oldlong),('corrected',d)]:
            common=frame[frame.dataset.eq(ds)&frame.binset.eq('mixed')&frame.in_domain&frame.scoreable&frame.leakage_free&frame.all_tools_scored]
            wc=cohort(frame);wc=wc[wc.dataset.eq(ds)]
            coverage.append(dict(dataset=ds,version=label,common_n=int(common.bin_id.nunique()),retained_wellcontrolled_n=int(wc.bin_id.nunique()),retained_fraction=float(wc.bin_id.nunique()/common.bin_id.nunique()),n_refs=int(wc.cluster.nunique())))
    pd.DataFrame(coverage).to_csv(DEST/'cohort_coverage.tsv',sep='\t',index=False)
    inputs=[original/'cami2_long_predictions.tsv',original/'cami2_wellcontrolled_by_distance.tsv',replay/'cami2_long_predictions.tsv',Path(__file__).resolve()]
    (DEST/'summary.json').write_text(json.dumps(dict(status='CONTROLLED_DOSE_SENSITIVITY_COMPLETE',historical_point_and_count_checks=len(checks),max_historical_point_difference=max(r['absolute_difference'] for r in checks),interval_provenance='Original point estimates recovered exactly; new CIs use an explicitly recorded reproducible seed because the historical ad-hoc generator was not retained.',input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs],output=dict(path=str((replay/'cami2_wellcontrolled_by_distance.tsv').relative_to(ROOT)),sha256=sha(replay/'cami2_wellcontrolled_by_distance.tsv'))),indent=2)+'\n')
    print(pd.DataFrame(coverage).to_string(index=False))
if __name__=='__main__':main()
