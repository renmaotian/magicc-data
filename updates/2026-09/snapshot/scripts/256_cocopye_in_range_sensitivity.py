#!/usr/bin/env python3
"""Prespecified secondary four-tool comparison: true contamination <=35%.

The fixed five-set primary panel is unchanged. Eligibility is based exclusively
on known truth (50 <= completeness <=100; contamination <=35); all four tools
must score both quantities. MAEs pool samples, not equally weighted sets. The
pooled resampling unit is the dominant accession across sets, preserving a
shared reference when it recurs. This is a training-dose-range sensitivity, not
proof that the tools share all other training-distribution characteristics.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'in_range_sensitivity'
SETS=['set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E'];TOOLS=['magicc_v5','checkm2','cocopye','deepcheck']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finalize_existing():
    spec=importlib.util.spec_from_file_location('inrange_finalize101',WORK/'scripts/101_metrics_framework.py');fw=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fw;spec.loader.exec_module(fw)
    p=DEST/'paired_comparisons.tsv';t=pd.read_csv(p,sep='\t');assert len(t)==36
    t['q_bh_wilcoxon_naive']=fw.bh_correct(t.p_wilcoxon_two_sided_naive.to_numpy())
    t['significant_all_three_bh']=t.q_bh_wilcoxon_cluster_mean.lt(.05)&t.q_bh_cluster_bootstrap_mean.lt(.05)&t.q_bh_wilcoxon_naive.lt(.05)
    t.to_csv(p,sep='\t',index=False);j=DEST/'summary.json';r=json.loads(j.read_text());r['significance_support_rule']='Agreement of all three BH-adjusted tests; rank-test flags also retained as explicitly named components.'
    r['finalization']='Adds the same conservative agreement flag used for the primary panel from already computed p-values; no bootstrap values or point estimates change.'
    for item in r['input_manifest']+r['output_manifest']:item['sha256']=sha(ROOT/item['path'])
    j.write_text(json.dumps(r,indent=2)+'\n')
def main():
    spec=importlib.util.spec_from_file_location('inrange104',WORK/'scripts/104_clustered_statistics.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    fw=m.fw;cfg=fw.load_config(WORK/'scripts/config_revision_metrics.yaml');DEST.mkdir(exist_ok=True)
    frames=[];coverage=[];objects={b.name:b for b in fw.discover_sets(cfg,include_missing=False)}
    for name in SETS:
        d,tools,prov=fw.load_set(cfg,objects[name]);assert set(TOOLS)<=set(tools)
        assert len(d)==1000
        eligible=d.true_completeness.between(50,100) & d.true_contamination.le(35)
        allscored=np.logical_and.reduce([np.isfinite(d[f'pred_{metric}__{tool}']) for tool in TOOLS for metric in ['completeness','contamination']])
        z=d.loc[eligible & allscored].copy();z['source_set']=name;frames.append(z)
        raw=pd.read_csv(WORK/'data/benchmarks'/name/'cocopye_predictions.tsv',sep='\t')
        for tool in TOOLS:
            scored=np.isfinite(d[f'pred_completeness__{tool}']) & np.isfinite(d[f'pred_contamination__{tool}'])
            coverage.append(dict(set=name,tool=tool,original_n=len(d),eligible_n=int(eligible.sum()),eligible_n_refs=int(d.loc[eligible,'cluster_id'].nunique()),n_scored_eligible=int((eligible & scored).sum()),n_unscored_eligible=int((eligible & ~scored).sum()),common_n=len(z),common_n_refs=int(z.cluster_id.nunique()),cocopye_stage1_entire_set=int(raw.selected_stage.eq(1).sum())))
    pooled=pd.concat(frames,ignore_index=True);groups=list(zip(SETS,frames))+[('POOLED_sample_weighted',pooled)]
    metrics=[];tests=[]
    for name,d in groups:
        seed=cfg.seed+fw.stable_hash('in_range_35|'+name)%100000
        bs=fw.Bootstrapper(clusters=d.cluster_id.to_numpy(),n_iter=cfg.n_boot,ci_level=cfg.ci_level,seed=seed);bs.resample_indices()
        slow=fw.Bootstrapper(clusters=d.cluster_id.to_numpy(),n_iter=cfg.n_boot_slow,ci_level=cfg.ci_level,seed=seed+1);slow.resample_indices()
        obj=type('Secondary',(),dict(name=name,label=name,status='secondary_truth_restriction',tier='secondary_35pct'))()
        metrics.extend(m.accuracy_rows(d,TOOLS,obj,cfg,bs))
        for comp in TOOLS[1:]:
            for metric in ['completeness','contamination']:
                r=m.paired_comparison(d,'magicc_v5',comp,f'abs_err_{metric}',cfg,bs,slow)
                assert r is not None;r['set']=name;r['metric']=metric;tests.append(r)
        print(name,len(d),d.cluster_id.nunique(),'completed',flush=True)
    t=pd.DataFrame(tests)
    assert len(t)==36
    for p,q in [('p_wilcoxon_two_sided_naive','q_bh_wilcoxon_naive'),('p_wilcoxon_two_sided_cluster_mean','q_bh_wilcoxon_cluster_mean'),('p_cluster_bootstrap_mean','q_bh_cluster_bootstrap_mean')]:t[q]=fw.bh_correct(t[p].to_numpy())
    t['bh_family']='all 36 secondary set/output/comparator contrasts';t['significant_bh_primary']=t.q_bh_wilcoxon_cluster_mean.lt(.05)
    t['significant_all_three_bh']=t.q_bh_wilcoxon_cluster_mean.lt(.05) & t.q_bh_cluster_bootstrap_mean.lt(.05) & t.q_bh_wilcoxon_naive.lt(.05)
    pd.DataFrame(metrics).to_csv(DEST/'accuracy.tsv',sep='\t',index=False);t.to_csv(DEST/'paired_comparisons.tsv',sep='\t',index=False);pd.DataFrame(coverage).to_csv(DEST/'scoring_coverage.tsv',sep='\t',index=False)
    keep=['source_set','genome_id','cluster_id','true_completeness','true_contamination']+[f'pred_{metric}__{tool}' for tool in TOOLS for metric in ['completeness','contamination']]
    pooled[keep].to_csv(DEST/'eligible_predictions.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
    summary=dict(status='SECONDARY_COMPLETE',prespecified_before_comparative_outcomes=True,truth_filter='50 <= true completeness <=100 and true contamination <=35',cohort_rule='Both finite estimates from every one of the four tools',pooled_weighting='Each eligible sample has equal weight; sets have their observed sample proportions',pooled_cluster_unit='Dominant accession shared across sets',pooled_n=len(pooled),pooled_n_refs=int(pooled.cluster_id.nunique()),bootstrap_replicates=cfg.n_boot,hl_bootstrap_replicates=cfg.n_boot_slow,bh_family_n=36,interpretation='Secondary CheckM2 published training-dose-range comparison; does not change primary benchmark or correct historical normalization overlap',input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [Path(__file__).resolve(),WORK/'scripts/config_revision_metrics.yaml',WORK/'scripts/104_clustered_statistics.py',OUT/'corrected_prediction_manifest.tsv']],output_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(DEST.iterdir()) if p.suffix in {'.tsv','.gz'}])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':
    if '--finalize-existing' in sys.argv:finalize_existing()
    else:main()
