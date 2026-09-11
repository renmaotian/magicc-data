#!/usr/bin/env python3
"""Post-audit CAMI scoring-coverage sensitivity; no inference or cohort replacement.

All reference-screened, in-domain, >=50%-complete bins scored by MAGICC and
CheckM2 are compared with the four-tool common subset and its complement.
Intervals use the same reference-cluster draws for both tools and outputs.
"""
import hashlib,json,zlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision/cocopye_stage_resubmission5'
SRC=OUT/'replay/results/revision/cami2/analysis'
DEST=OUT/'cami_pairwise_coverage'
TOOLS=['MAGICC_V5','CheckM2','CoCoPyE','DeepCheck']
NBOOT=2000
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(exist_ok=True)
    p=SRC/'cami2_long_predictions.tsv';d=pd.read_csv(p,sep='\t')
    assert not d.duplicated(['dataset','binset','bin_id','tool']).any()
    oldp=ROOT/'results/revision/cami2/analysis/cami2_long_predictions.tsv'
    old=pd.read_csv(oldp,sep='\t')
    joined=d[d.tool.isin(TOOLS[:2])].merge(old,on=['dataset','binset','bin_id','tool'],validate='one_to_one',suffixes=('_new','_old'))
    assert len(joined)==len(d[d.tool.isin(TOOLS[:2])])
    for col in ['pred_completeness','pred_contamination','true_completeness','true_contamination']:
        assert np.array_equal(joined[col+'_new'],joined[col+'_old'],equal_nan=True),col
    metrics=[];paired=[];coverage=[];members=[]
    for (ds,bs),raw in d.groupby(['dataset','binset'],sort=True):
        truthcols=['cluster','true_completeness','true_contamination','in_domain','scoreable','leakage_free','all_tools_scored']
        assert (raw.groupby('bin_id')[truthcols].nunique(dropna=False)<=1).all().all()
        truth=raw.drop_duplicates('bin_id').set_index('bin_id')[truthcols]
        base=truth[truth.in_domain & truth.scoreable & truth.leakage_free]
        frames={t:raw[raw.tool.eq(t)].set_index('bin_id').reindex(base.index) for t in TOOLS}
        finite={t:np.isfinite(frames[t][['pred_completeness','pred_contamination']]).all(axis=1) for t in TOOLS}
        both=finite[TOOLS[0]] & finite[TOOLS[1]]
        common=np.logical_and.reduce([finite[t].to_numpy() for t in TOOLS])
        assert np.array_equal(common,base.all_tools_scored.to_numpy())
        for t in TOOLS:
            z=frames[t];has_row=z.tool.notna();stage1=z.selected_stage.eq(1)
            assert not (stage1 & finite[t]).any()
            coverage.append(dict(dataset=ds,binset=bs,tool=t,n_reference_screened_in_domain_above_floor=len(base),n_prediction_rows=int(has_row.sum()),n_no_prediction_row=int((~has_row).sum()),n_stage1_unscored=int(stage1.sum()),n_other_nonfinite=int((has_row & ~finite[t] & ~stage1).sum()),n_finite=int(finite[t].sum()),n_pairwise=int(both.sum()),n_four_tool_common=int(common.sum()),n_pairwise_only=int((both & ~common).sum()),row_availability_scope='No prediction row combines bins not selected for that comparator with any missing outputs; execution attempts are not inferred.'))
        for binid in base.index[both]:
            members.append(dict(dataset=ds,binset=bs,bin_id=binid,cluster=base.loc[binid,'cluster'],four_tool_common=bool(base.loc[binid,'all_tools_scored']),cocopye_stage=frames['CoCoPyE'].loc[binid,'selected_stage'],cocopye_has_row=bool(pd.notna(frames['CoCoPyE'].loc[binid,'tool'])),cocopye_scored=bool(finite['CoCoPyE'].loc[binid])))
        for label,mask in [('pairwise_all_scored',both),('four_tool_common',both & common),('pairwise_only_excluded_from_four_tool',both & ~common)]:
            z=base.loc[mask];n=len(z)
            if not n:continue
            clusters,inv=np.unique(z.cluster.to_numpy(),return_inverse=True);k=len(clusters)
            seed=20260908+(zlib.crc32(f'cami_coverage|{ds}|{bs}|{label}'.encode())&0xffffffff)
            draws=np.random.default_rng(seed).multinomial(k,np.full(k,1/k),size=NBOOT)
            counts=np.bincount(inv,minlength=k);den=draws@counts
            for metric in ['completeness','contamination']:
                estimates=[];boots=[]
                truthv=z['true_'+metric].to_numpy(float)
                for t in TOOLS[:2]:
                    err=frames[t].loc[z.index,'pred_'+metric].to_numpy(float)-truthv
                    sums=np.stack([np.bincount(inv,weights=np.abs(err),minlength=k),np.bincount(inv,weights=err,minlength=k)],axis=1)
                    est=np.array([np.mean(np.abs(err)),np.mean(err)])
                    boot=(draws@sums)/den[:,None];ci=np.quantile(boot,[.025,.975],axis=0)
                    metrics.append(dict(dataset=ds,binset=bs,cohort=label,tool=t,metric=metric,n=n,n_references=k,mae=est[0],mae_ci_lo=ci[0,0],mae_ci_hi=ci[1,0],signed_bias=est[1],bias_ci_lo=ci[0,1],bias_ci_hi=ci[1,1],seed=seed,n_bootstrap=NBOOT))
                    estimates.append(est[0]);boots.append(boot[:,0])
                ci=np.quantile(boots[0]-boots[1],[.025,.975])
                paired.append(dict(dataset=ds,binset=bs,cohort=label,metric=metric,n=n,n_references=k,mean_mae_difference_magicc_minus_checkm2=estimates[0]-estimates[1],ci_lo=ci[0],ci_hi=ci[1],seed=seed,n_bootstrap=NBOOT))
    for name,rows in [('metrics.tsv',metrics),('paired_mae_differences.tsv',paired),('coverage.tsv',coverage),('membership.tsv',members)]:pd.DataFrame(rows).to_csv(DEST/name,sep='\t',index=False)
    inputs=[p,oldp,Path(__file__).resolve()]
    summary=dict(status='CAMI_PAIRWISE_COVERAGE_COMPLETE',purpose='Post-audit scoring-coverage sensitivity; does not replace the fixed four-tool primary cohort.',cohort='Reference overlap screened, true completeness >=50%, true contamination <=true completeness; pairwise MAGICC/CheckM2 finite in both outputs. No CoCoPyE availability requirement in the pairwise cohort.',uncertainty='2000 paired dominant-source-reference cluster resamples; same draws for both tools and outputs. Percentile95% intervals; sample-weighted MAE/bias within each dataset/binset. No new significance designation or multiplicity-selected claim.',prediction_verification='All MAGICC/CheckM2 predictions and truth values exactly match historical rows; changes between common and pairwise cohorts arise only from membership.',normalization='Reference overlap screening does not remove the disclosed historical unsupervised preprocessing overlap.',input_manifest=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x)) for x in inputs],output_manifest=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x)) for x in sorted(DEST.glob('*.tsv'))])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(pd.DataFrame(metrics).query('cohort=="pairwise_all_scored"').to_string(index=False))
if __name__=='__main__':main()
