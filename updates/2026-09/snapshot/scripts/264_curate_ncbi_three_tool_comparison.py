#!/usr/bin/env python3
"""Prespecified three-tool NCBI comparison with explicit contamination proxy.

Consumes existing predictions and species-cluster intervals from corrected
script143. No new inference. Completeness is reference-alignment coverage;
unmatched draft bp is only an upper proxy for contamination, not verified
foreign DNA. Primary913 pairs/124 species; all1096 and species-absent106 are
separate sensitivity strata. DeepCheck was not run and is not fabricated.
"""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'ncbi_three_tool'
COHORTS={'primary_leakage_free':('primary_accession_overlap_screened',913,124),'all_tier1_pairs':('all_valid_pairs_sensitivity',1096,164),'species_absent_from_training':('species_absent_sensitivity',106,None)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    src=WORK/'results/revision/real_data/ncbi_pairs';a=pd.read_csv(src/'metrics_by_cohort.tsv',sep='\t');p=pd.read_csv(src/'paired_tests.tsv',sep='\t');raw=pd.read_csv(src/'per_pair_results.tsv',sep='\t');DEST.mkdir(exist_ok=True);rows=[];comparisons=[];coverage=[]
    assert len(raw)==1096 and raw.genome_id.is_unique
    for original,(label,n,k) in COHORTS.items():
        sub=a[a.cohort.eq(original)];assert len(sub)==3 and sub.n.eq(n).all()
        if k is not None:assert sub.n_clusters.eq(k).all()
        mask=raw.leakage_free if original=='primary_leakage_free' else ~raw.species_in_train if original=='species_absent_from_training' else np.ones(len(raw),dtype=bool)
        z=raw.loc[mask];assert len(z)==n
        for tool in ['magicc','checkm2','cocopye']:
            assert np.isfinite(z[[f'{tool}_completeness',f'{tool}_contamination']]).all().all()
            coverage.append(dict(cohort=label,tool=tool,n_attempted=n,n_scored=n,n_unscored=0,n_species=int(z.species_taxid.nunique()),n_cocopye_stage2=int(z.cocopye_selected_stage.eq(2).sum()) if tool=='cocopye' else None))
        for r in sub.to_dict('records'):
            for metric,prefix in [('completeness','comp'),('contamination_upper_proxy','cont')]:
                rows.append(dict(cohort=label,tool=r['tool'],metric=metric,n=r['n'],n_species=r['n_clusters'],mae=r[prefix+'_mae'],mae_ci_lo=r[prefix+'_mae_ci_lo'],mae_ci_hi=r[prefix+'_mae_ci_hi'],bias_pred_minus_target=r[prefix+'_bias'],bias_ci_lo=r[prefix+'_bias_ci_lo'],bias_ci_hi=r[prefix+'_bias_ci_hi'],target_definition='Aligned full-reference bp fraction' if prefix=='comp' else 'Unmatched draft bp / full matched reference bp; upper proxy, not verified foreign DNA',interval='2000 species-cluster bootstrap resamples;95% percentile interval'))
        for r in p[p.cohort.eq(original)].to_dict('records'):
            comparisons.append(dict(cohort=label,tool_a=r['tool_a'],tool_b=r['tool_b'],metric='contamination_upper_proxy' if r['metric']=='contamination' else 'completeness',n=r['n'],n_species=int(z.species_taxid.nunique()),mean_mae_difference_a_minus_b=r['mean_diff_a_minus_b'],ci_lo=r['diff_ci_lo'],ci_hi=r['diff_ci_hi'],interval_excludes_zero=bool(r['diff_ci_lo']>0 or r['diff_ci_hi']<0),interpretation='Paired difference, resampling the same species clusters for both tools; negative favors tool_a. No winner designation based on a sample-level p-value.'))
    pd.DataFrame(rows).to_csv(DEST/'metrics.tsv',sep='\t',index=False);pd.DataFrame(comparisons).to_csv(DEST/'paired_mae_differences.tsv',sep='\t',index=False);pd.DataFrame(coverage).to_csv(DEST/'scoring_coverage.tsv',sep='\t',index=False)
    inputs=[src/'metrics_by_cohort.tsv',src/'paired_tests.tsv',src/'per_pair_results.tsv',ROOT/'results/revision/ncbi_pairs_resubmission5/existing_prediction_inventory.json',Path(__file__).resolve()]
    summary=dict(status='NCBI_THREE_TOOL_COMPLETE',primary_n=913,primary_n_species=124,tools=['MAGICC V5','CheckM2 1.0.1','CoCoPyE 0.5.0'],deepcheck='No existing predictions; omitted',predeclared_scope='Primary913/124; all1096 and species-absent106 sensitivity; completeness primary and contamination-upper-proxy secondary, regardless comparative outcome',normalization_scope='Accession overlap screened; this does not establish independent historical unsupervised preprocessing for MAGICC.',input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs],output_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(DEST.glob('*.tsv'))])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(pd.DataFrame(rows).query('cohort=="primary_accession_overlap_screened"').to_string(index=False))
if __name__=='__main__':main()
