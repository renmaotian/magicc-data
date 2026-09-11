#!/usr/bin/env python3
"""Verify microbial scope preserves primary results and finalize paired coverage."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';DEST=OUT/'cami_microbial_scope';AN=OUT/'replay/results/revision/cami2/analysis';BEFORE=DEST/'before_scope'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((OUT/'analysis_status/201.json').read_text())['status']=='COMPLETE'
    old=pd.read_csv(BEFORE/'cami2_accuracy_by_cohort.tsv',sep='\t');new=pd.read_csv(AN/'cami2_accuracy_by_cohort.tsv',sep='\t');checks=[]
    keys=['dataset','binset','cohort','tool','metric'];cohort='primary_in_domain_scoreable_LEAKAGE_FREE'
    a=old[old.cohort.eq(cohort)].sort_values(keys).reset_index(drop=True);b=new[new.cohort.eq(cohort)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b);assert len(a)==32
    for name in ['cami2_paired_tests.tsv','cami2_mixed_by_distance.tsv','cami2_mimag.tsv','cami2_gold_by_completeness_decile.tsv']:
        x=pd.read_csv(BEFORE/name,sep='\t');y=pd.read_csv(AN/name,sep='\t');pd.testing.assert_frame_equal(x,y);checks.append(dict(source=name,n_rows=len(x),all_values_unchanged=True))
    oldlong=pd.read_csv(BEFORE/'cami2_long_predictions.tsv',sep='\t');newlong=pd.read_csv(AN/'cami2_long_predictions.tsv',sep='\t')
    def primary(d):return d[d.in_domain & d.scoreable & d.leakage_free & d.all_tools_scored].sort_values(['dataset','binset','bin_id','tool']).reset_index(drop=True)
    pd.testing.assert_frame_equal(primary(oldlong),primary(newlong))
    assert not newlong.dominant.fillna('').str.startswith('RNODE').any() and not newlong.contaminant.fillna('').str.startswith('RNODE').any()
    pd.DataFrame(checks).to_csv(DEST/'primary_result_invariance.tsv',sep='\t',index=False)
    # Preserve diagnostic provenance after its former logical source is superseded.
    diag=OUT/'cami_pairwise_coverage/summary.json';record=json.loads(diag.read_text())
    for item in record['input_manifest']:
        if item['path']==str((AN/'cami2_long_predictions.tsv').relative_to(ROOT)):
            assert item['sha256']==sha(BEFORE/'cami2_long_predictions.tsv');item['path']=str((BEFORE/'cami2_long_predictions.tsv').relative_to(ROOT))
    record['status']='CONSTRUCTION_ONLY_DIAGNOSTIC_SUPERSEDED_BY_MICROBIAL_SCOPE';record['scope_warning']='The pairwise-exclusive marine reference units are RNODE circular elements. These results cannot support prokaryotic genome-quality validation.'
    diag.write_text(json.dumps(record,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('scope266',ROOT/'scripts/266_cami_pairwise_coverage_sensitivity.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    mod.DEST=OUT/'cami_pairwise_coverage_microbial';mod.main()
    coverage=pd.read_csv(mod.DEST/'coverage.tsv',sep='\t');assert coverage.n_pairwise.eq(coverage.n_four_tool_common).all();assert coverage.n_pairwise_only.eq(0).all()
    headp=AN/'cami2_headline.json';head=json.loads(headp.read_text());head['source_scope']='Marine: native Otu microbial references only; all RNODE circular elements excluded from both dominant and donor positions. Strain-madness: all408 native microbial sources. Source audit267 and scope replay270/272 document correction and primary invariance.';headp.write_text(json.dumps(head,indent=2)+'\n')
    inputs=[Path(__file__).resolve(),OUT/'analysis_status/201.json',DEST/'preparation.json',BEFORE/'cami2_long_predictions.tsv',AN/'cami2_long_predictions.tsv',ROOT/'results/revision/holdout_resubmission5/cami_source_audit/audit_summary.json',OUT/'cami_sequence_input_validation/summary.json']
    summary=dict(status='CAMI_MICROBIAL_SCOPE_COMPLETE',n_primary_metric_rows_identical=32,primary_prediction_membership_and_values_identical=True,n_reference_units_excluded='All200 native marine RNODE circular-element IDs;88 unknown-category IDs had entered legacy genome truth/cohorts.',pairwise_result='On eligible microbial sources, four-tool common and MAGICC–CheckM2 pairwise cohorts coincide: marine339gold/864mixed; strain-madness700gold/2250mixed. The apparent additional paired benefit came from out-of-scope circular elements and is not genomic validation.',diagnostic_record='Unscoped354-bin pairwise-exclusive results retained as construction audit; every bin has realACGT sequence and was comparator-input selected, but its source unit is not a prokaryotic genome.',input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs],output_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [DEST/'source_and_truth_accounting.tsv',DEST/'primary_result_invariance.tsv',mod.DEST/'summary.json',diag,headp]])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Primary rows/intervals unchanged; microbial pairwise/common coverage identical')
if __name__=='__main__':main()
