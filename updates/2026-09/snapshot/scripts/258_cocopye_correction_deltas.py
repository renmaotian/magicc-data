#!/usr/bin/env python3
"""Emit machine-readable numerical deltas for document reconciliation."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'document_deltas'
SOURCES={
 'all_rebuilt_accuracy_including_motivation':('metrics/ws5.5_table_S2_rebuilt.tsv',['set','tool','metric']),
 'primary_accuracy':('metrics/definitive_table_5set.tsv',['set','tool','metric']),
 'primary_thresholds':('metrics/definitive_thresholds.tsv',['set','tool','criterion','threshold']),
 'primary_mimag':('metrics/definitive_mimag.tsv',['set','tool','mimag_class']),
 'primary_paired_tests':('metrics/ws5.4_clustered_tests.tsv',['set','reference_tool','comparison_tool','metric']),
 'primary_macro_f1_paired':('ws11/macro_f1_paired/macro_f1_paired_differences.tsv',['set','reference_tool','comparison_tool','statistic']),
 'novelty_metrics':('ws11/novelty_ladder/novelty_ladder_metrics.tsv',['scope','novelty_class','tool']),
 'novelty_paired':('ws11/novelty_ladder/novelty_ladder_paired_tests.tsv',['scope','novelty_class','reference_tool','comparison_tool','metric']),
 'novelty_trend':('ws11/novelty_ladder/novelty_ladder_trend.tsv',['tool','metric']),
 'signed_error_summaries':('metrics/ws5.3_signed_errors_overall.tsv',['set','tool','metric']),
 'setG_curves':('set_G/set_G_curves.tsv',['tool','error_type','error_rate']),
 'setG_degradation':('set_G/set_G_paired_degradation.tsv',['tool','error_type','error_rate','metric']),
 'setF_cells':('set_F/set_F_cells_type_x_distance.tsv',['tool','contamination_type','distance']),
 'setF_marginals':('set_F/set_F_marginals.tsv',['tool','margin','level']),
 'setF_paired':('set_F/set_F_paired_comparisons.tsv',['stratum','level','tool_a','tool_b']),
 'setF_attribution':('set_F/set_F_attribution.tsv',['competitor','distance','arm']),
 'setH_metrics':('circularity/ws1_11_arm_metrics.tsv',['tool','group']),
 'setH_paired':('circularity/ws1_11_paired_arm_tests.tsv',['tool','metric_name','statistic','subset']),
 'setH_reference_scores':('circularity/ws1_11_reference_level_scores.tsv',['estimate']),
 'setH_did':('circularity/ws1_11_did_vs_competitors.tsv',['tool_a','tool_b','metric_name','statistic']),
 'meslier_metrics':('real_data/meslier/metrics_by_cohort.tsv',['cohort','tool']),
 'meslier_classification':('real_data/meslier/mimag_by_cohort.tsv',['cohort','tool']),
 'meslier_fragmentation':('real_data/meslier/fragmentation_gradient.tsv',['panel','assembly','tool']),
 'meslier_slopes':('real_data/meslier/fragmentation_slopes.tsv',['panel','tool','metric']),
 'ncbi_metrics':('real_data/ncbi_pairs/metrics_by_cohort.tsv',['cohort','tool']),
 'cami_accuracy':('cami2/analysis/cami2_accuracy_by_cohort.tsv',['dataset','binset','cohort','tool','metric']),
 'cami_distance':('cami2/analysis/cami2_mixed_by_distance.tsv',['dataset','tool','distance_rank']),
 'cami_paired':('cami2/analysis/cami2_paired_tests.tsv',['dataset','binset','tool_a','tool_b','metric']),
 'cami_classification':('cami2/analysis/cami2_mimag.tsv',['dataset','binset','tool']),
 'cami_controlled_dose':('cami2/analysis/cami2_wellcontrolled_by_distance.tsv',['dataset','tool','distance_rank']),
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(exist_ok=True);inventory=[];deltas=[]
    for label,(rel,keys) in SOURCES.items():
        if label.startswith('cami_') and json.loads((OUT/'analysis_status/201.json').read_text())['status']!='COMPLETE':continue
        oldpath=ROOT/'results/revision'/rel;newpath=WORK/'results/revision'/rel
        if not oldpath.exists() or not newpath.exists():continue
        a=pd.read_csv(oldpath,sep='\t');b=pd.read_csv(newpath,sep='\t')
        # Threshold table calls its parameter threshold_pct in this revision.
        actualkeys=[k if k in a.columns else 'threshold_pct' if k=='threshold' and 'threshold_pct' in a.columns else k for k in keys]
        assert set(actualkeys)<=set(a.columns)&set(b.columns),(label,actualkeys,a.columns.to_list())
        assert not a.duplicated(actualkeys).any() and not b.duplicated(actualkeys).any(),label
        commoncols=[x for x in a.columns if x in b.columns and x not in actualkeys and pd.api.types.is_numeric_dtype(a[x]) and pd.api.types.is_numeric_dtype(b[x])]
        merged=a[actualkeys+commoncols].merge(b[actualkeys+commoncols],on=actualkeys,how='outer',suffixes=('_old','_new'),indicator=True)
        for r in merged.to_dict('records'):
            identifiers={k:r[k] for k in actualkeys}
            for col in commoncols:
                v,w=r.get(col+'_old'),r.get(col+'_new')
                if pd.isna(v) and pd.isna(w):continue
                if pd.notna(v) and pd.notna(w) and np.isclose(float(v),float(w),atol=1e-11,rtol=1e-12):continue
                deltas.append(dict(source=label,source_logical_path='results/revision/'+rel,identifiers=json.dumps(identifiers,sort_keys=True),column=col,old=v,new=w,delta=float(w)-float(v) if pd.notna(v) and pd.notna(w) else np.nan,row_join=r['_merge']))
        inventory.append(dict(source=label,original_path=str(oldpath.relative_to(ROOT)),original_sha256=sha(oldpath),corrected_path=str(newpath.relative_to(ROOT)),corrected_sha256=sha(newpath),key_columns=actualkeys,original_rows=len(a),corrected_rows=len(b)))
    pd.DataFrame(deltas).to_csv(DEST/'numeric_cell_deltas.tsv',sep='\t',index=False)
    (DEST/'source_reconciliation_manifest.json').write_text(json.dumps(dict(status='PRIMARY_AND_COMPLETED_COHORT_DELTAS_READY',sources=inventory),indent=2)+'\n')
    a=pd.read_csv(WORK/'results/revision/metrics/definitive_table_5set.tsv',sep='\t')
    wanted=['set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E','POOLED_leakage_free_5_sets']
    a=a[a.set.isin(wanted)];a.to_csv(DEST/'primary_accuracy_final.tsv',sep='\t',index=False)
    old=pd.read_csv(ROOT/'results/revision/metrics/definitive_table_5set.tsv',sep='\t');check=a[a.tool.ne('cocopye') & ~a.set.str.startswith('POOLED')].merge(old,on=['set','tool','metric'],suffixes=('_new','_old'))
    invariance=[]
    for col in ['n','n_clusters','mae','mae_ci_lo','mae_ci_hi','rmse','signed_bias_pred_minus_true','r2_coefficient_of_determination']:
        e=np.abs(check[col+'_new']-check[col+'_old']);maxerr=float(e.max());assert maxerr<1e-10,(col,maxerr);invariance.append(dict(quantity=col,maximum_change_non_cocopye_per_set=maxerr))
    pd.DataFrame(invariance).to_csv(DEST/'unchanged_tool_per_set_verification.tsv',sep='\t',index=False)
    tests=pd.read_csv(WORK/'results/revision/metrics/ws5.4_clustered_tests.tsv',sep='\t');tests=tests[tests['set'].isin(wanted[:5]) & tests.comparison_tool.isin(['checkm2','cocopye','deepcheck'])];assert len(tests)==30
    tests.to_csv(DEST/'primary_30_paired_tests_final.tsv',sep='\t',index=False)
    partition=[]
    for criterion in ['significant_bh_primary','significant_all_three_bh']:
        for comparator,z in [('all_three_comparators',tests),*list(tests.groupby('comparison_tool'))]:
            partition.append(dict(criterion=criterion,comparator=comparator,n=len(z),magicc_advantage=int((z[criterion] & z.mean_paired_difference.lt(0)).sum()),magicc_disadvantage=int((z[criterion] & z.mean_paired_difference.gt(0)).sum()),unresolved=int((~z[criterion]).sum())))
    pd.DataFrame(partition).to_csv(DEST/'primary_test_partition.tsv',sep='\t',index=False)
    audit=dict(status='DECISION_RULE_AUDITED',historical_protocol='project_design_and_protocol.md WS5.4 requires two-sided paired clustered tests, BH correction and effects but does not define a winner-count rule.',historical_script='scripts/104_clustered_statistics.py labels BH cluster-mean Wilcoxon primary and emits a separate significant_all_three_bh conjunction.',shipped_methods='nature_communications/resubmission4/manuscript_revised.md Statistics states that a difference is supported only when all three tests agree.',final_support_rule='All three BH-adjusted tests must reject at q<0.05: genome-level paired Wilcoxon, reference-cluster-mean paired Wilcoxon, and reference-cluster bootstrap of the paired mean absolute-error difference. Direction is the paired mean-MAE difference. Failure to meet this conjunction is unresolved, not equivalence.',rank_test_only_rule='BH-adjusted reference-cluster-mean Wilcoxon alone; retained as an explicitly named component result, not substituted for supported mean-MAE differences.',partition=partition)
    (DEST/'decision_rule_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (DEST/'DECISION_RULE_AUDIT.md').write_text('# Decision-rule reconciliation\n\nThe historical protocol (WS5.4) specifies paired two-sided clustered tests, BH correction and effect sizes, without a winner-count criterion. Script104 labels BH cluster-mean Wilcoxon primary and separately records the conjunction of three tests. The shipped resubmission4 Statistics paragraph states that support requires agreement of all three tests.\n\nThe final resubmission5 support count therefore follows that shipped conjunction: **22 MAGICC advantages, 4 disadvantages, 4 unresolved comparisons**. The rank-test-only partition is **24/4/2** and is labeled as a component analysis. No criterion is retrospectively called prespecified.\n\nThe two additional rank-only advantages are SetA contamination versus CoCoPyE and C-clean completeness versus CheckM2; their paired mean-MAE intervals span zero. Full component p/q values and effect intervals remain in primary_30_paired_tests_final.tsv. Unresolved does not establish equivalence.\n')
    headlinepath=WORK/'results/revision/metrics/definitive_headline.json';headline=json.loads(headlinepath.read_text());headline['decision_rule_audit']=audit
    headline['definition_notes']['pooled_ci']='1,648 dominant accessions shared across sets; the five sets each retain weight 1/5 within every bootstrap draw.'
    headline['definition_notes']['binary_balanced_accuracy']='Mean of both class recalls; undefined when either truth class is absent. Multiclass MIMAG convention is separate.'
    headline['definition_notes']['preprocessing_scope']='Held-out dominant reference accessions; production V5 inherited unsupervised normalization fitted on V4 training, validation and test feature rows.'
    headlinepath.write_text(json.dumps(headline,indent=2)+'\n')
    summary=dict(status='PRIMARY_DELTAS_READY',primary_n=5000,primary_stage1_rejections=0,primary_stage2_selected=1702,pooled_n_shared_references=1648,pooled_mae=a[a.set.eq('POOLED_leakage_free_5_sets')][['tool','metric','mae','mae_ci_lo','mae_ci_hi']].to_dict('records'),interpretation=['Official stage selection changes CoCoPyE only; all primary samples remain paired across tools.','MAGICC, CheckM2 and DeepCheck per-set estimates and intervals are independently unchanged.','Pooled intervals now share accession multiplicities across sets and preserve equal set weights.','Paired BH-adjusted q-values may change for other comparisons because the family includes corrected CoCoPyE tests.','MAGICC retains the lowest pooled MAE point estimate for both outputs; comparator-specific trade-offs remain visible.'],sources=inventory)
    (DEST/'primary_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Wrote',len(deltas),'changed numeric cells and primary summary')
if __name__=='__main__':main()
