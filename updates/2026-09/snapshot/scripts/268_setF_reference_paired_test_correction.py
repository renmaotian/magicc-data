#!/usr/bin/env python3
"""Correct SetF paired-test replication units without repeating inference/bootstrap.

The87 comparisons share100 dominant references. The original row-level p/q
values are retained explicitly; primary p/q now use one mean paired difference
per reference. Existing reference-cluster HL intervals are unchanged.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'setF_reference_tests';BASE=WORK/'results/revision/set_F'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bh(p):
    p=np.asarray(p,float);order=np.argsort(p);v=p[order]*len(p)/np.arange(1,len(p)+1);q=np.minimum.accumulate(v[::-1])[::-1];out=np.empty(len(p));out[order]=np.minimum(q,1);return out
def main():
    DEST.mkdir(exist_ok=True);path=BASE/'set_F_paired_comparisons.tsv';snapshot=DEST/'historical_row_level_paired_comparisons.tsv'
    if not snapshot.exists():shutil.copy2(path,snapshot)
    old=pd.read_csv(snapshot,sep='\t');assert len(old)==87
    long=pd.read_csv(BASE/'set_F_long_predictions.tsv',sep='\t');long=long[long.contamination_type.ne('none')];rows=[]
    for r in old.to_dict('records'):
        z=long
        if r['stratum']=='type':z=z[z.contamination_type.eq(r['level'])]
        elif r['stratum']=='distance':z=z[z.distance.eq(r['level'])]
        elif r['stratum']=='cell':
            ty,di=r['level'].split('|');z=z[z.contamination_type.eq(ty)&z.distance.eq(di)]
        elif r['stratum']=='pooled' and r['level']=='NON_REDUNDANT':z=z[z.contamination_type.isin(['replaced','single'])]
        else:assert r['stratum']=='pooled' and r['level']=='ALL'
        a=z[z.tool.eq(r['tool_a'])];b=z[z.tool.eq(r['tool_b'])]
        m=a.merge(b,on='genome_id',validate='one_to_one',suffixes=('_a','_b'))
        assert len(m)==r['n_pairs'] and (m.ref_index_a==m.ref_index_b).all()
        d=m.abs_err_contamination_a.to_numpy()-m.abs_err_contamination_b.to_numpy()
        oldp=float(wilcoxon(d,alternative='two-sided',zero_method='wilcox').pvalue) if len(d)>=3 and not np.allclose(d,0) else 1.
        assert np.isclose(oldp,r['p_wilcoxon'],rtol=1e-10,atol=1e-300),(r['level'],r['tool_b'])
        rd=pd.DataFrame({'reference':m.ref_index_a,'difference':d}).groupby('reference').difference.mean()
        assert len(rd)==100
        p=float(wilcoxon(rd,alternative='two-sided',zero_method='wilcox').pvalue) if not np.allclose(rd,0) else 1.
        r.update(p_wilcoxon_sample_level_historical=r['p_wilcoxon'],q_bh_sample_level_historical=r['q_bh'],winner_sample_level_historical=r['winner'],p_wilcoxon=p,p_wilcoxon_reference_mean=p,n_paired_references=len(rd),reference_mean_difference=float(rd.mean()))
        rows.append(r)
    result=pd.DataFrame(rows);result['q_bh']=bh(result.p_wilcoxon_reference_mean);result['q_bh_reference_mean']=result.q_bh
    result['supported_reference_test_and_hl_interval']=result.q_bh.lt(.05)&((result.hl_ci_lo.gt(0)&result.reference_mean_difference.gt(0))|(result.hl_ci_hi.lt(0)&result.reference_mean_difference.lt(0)))
    result['winner']=np.where(result.supported_reference_test_and_hl_interval,np.where(result.hl.lt(0),'magicc_v5',result.tool_b),'unresolved')
    unchanged=[c for c in old.columns if c not in ['p_wilcoxon','q_bh','winner']]
    pd.testing.assert_frame_equal(result[unchanged],old[unchanged])
    result['inference_definition']='Two-sided Wilcoxon on100 per-reference mean paired absolute-error differences; BH across87 comparisons. Support also requires concordant reference-cluster HL95% interval excluding0; unresolved is not equivalence.'
    result.to_csv(path,sep='\t',index=False)
    result['support_decision_changed']=result.winner.ne(result.winner_sample_level_historical.replace('tie','unresolved'))
    result.to_csv(path,sep='\t',index=False)
    changed=result[result.support_decision_changed].copy();changed.to_csv(DEST/'changed_support.tsv',sep='\t',index=False)
    result[result.stratum.eq('distance')].to_csv(DEST/'distance_comparisons.tsv',sep='\t',index=False)
    stagep=OUT/'analysis_status/147.json';stage=json.loads(stagep.read_text())
    for x in stage['outputs']:
        if x['path']==str(path.relative_to(ROOT)):
            assert x['sha256']==sha(snapshot);x['path']=str(snapshot.relative_to(ROOT));x['preservation_note']='Exact147 output before268 reference-unit inferential correction; final table is separately verified.'
    stagep.write_text(json.dumps(stage,indent=2)+'\n')
    inputs=[snapshot,BASE/'set_F_long_predictions.tsv',Path(__file__).resolve()]
    summary=dict(status='SETF_REFERENCE_TEST_CORRECTION_COMPLETE',n_comparisons=87,n_references_per_comparison=100,uncertainty='Existing reference-cluster HL and Cliff intervals and all point estimates unchanged. Only Wilcoxon replication unit, corresponding87-family BH q-values and support interpretation corrected.',decision_rule='Support requires q<0.05 from reference-mean Wilcoxon and concordant reference-cluster HL95% CI excluding0; no retrospectively prespecified claim.',sample_level_values='Retained in explicitly historical columns and exact input snapshot.',n_support_decisions_changed=len(changed),n_historical_tie_labels_renamed_unresolved=int(result.winner_sample_level_historical.eq('tie').sum()),new_partition=result.winner.value_counts().to_dict(),input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs],output_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [path,DEST/'changed_support.tsv',DEST/'distance_comparisons.tsv']])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(result[result.stratum.eq('distance')][['level','tool_b','hl','hl_ci_lo','hl_ci_hi','q_bh','winner']].to_string(index=False))
if __name__=='__main__':main()
