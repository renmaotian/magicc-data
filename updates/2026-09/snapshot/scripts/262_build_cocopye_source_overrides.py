#!/usr/bin/env python3
"""Build a finite verified mapping from historical logical to corrected sources.

This is an analysis-readiness gate, not the submission-completion gate. Final
completion additionally requires figure/workbook and document reconciliation.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay'
REQUIRED=['102','103','104','106','257','108','211','147','153','201','143m','143z','143n','144','204','205','191','192','210']
DIRS=['metrics','set_F','set_G','real_data/meslier','real_data/zymo','real_data/ncbi_pairs','cami2/analysis','circularity','ws11/novelty_ladder','ws11/macro_f1_paired']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--main-123-only',action='store_true');ap.add_argument('--figures-only',action='store_true');args=ap.parse_args();assert not (args.main_123_only and args.figures_only)
    parser=json.loads((OUT/'parser_verification.json').read_text());assert parser['status']=='PARSER_VERIFIED';stages=[]
    required=['102','103','104','106','257','108','211'] if args.main_123_only else REQUIRED
    if args.figures_only:required=[s for s in required if s!='210']
    for stage in required:
        p=OUT/'analysis_status'/f'{stage}.json';r=json.loads(p.read_text());assert r['status']=='COMPLETE',(stage,r['status'])
        for x in r['outputs']:assert sha(ROOT/x['path'])==x['sha256'],(stage,x['path'])
        stages.append(dict(stage=stage,status='COMPLETE',record_path=str(p.relative_to(ROOT)),record_sha256=sha(p)))
    extra=[('in_range_sensitivity','SECONDARY_COMPLETE'),('independent_figures','INDEPENDENT_CORRECTED_FIGURE_AUDIT_PASS'),('pooled_cluster_correction','POOLED_CI_CORRECTION_COMPLETE'),('binary_balanced_accuracy_correction','BINARY_BALANCED_ACCURACY_CORRECTION_COMPLETE'),('cami_controlled_dose_recovery','CONTROLLED_DOSE_SENSITIVITY_COMPLETE'),('ncbi_three_tool','NCBI_THREE_TOOL_COMPLETE'),('setF_reference_tests','SETF_REFERENCE_TEST_CORRECTION_COMPLETE'),('cami_microbial_scope','CAMI_MICROBIAL_SCOPE_COMPLETE'),('cami_sequence_input_validation','CAMI_SEQUENCE_INPUT_AUDIT_COMPLETE'),('diagnostic_context','DIAGNOSTIC_COMPARATOR_CONTEXT_RECONCILED')]
    if args.main_123_only:extra=[x for x in extra if x[0] in ['independent_figures','pooled_cluster_correction','binary_balanced_accuracy_correction']]
    for folder,status in extra:
        p=OUT/folder/'summary.json';r=json.loads(p.read_text());assert r['status']==status;stages.append(dict(stage=folder,status='COMPLETE',record_path=str(p.relative_to(ROOT)),record_sha256=sha(p)))
    overrides={}
    def add(logical,p,reason):
        assert p.resolve().is_relative_to(OUT.resolve()),p
        overrides[logical]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),reason=reason)
    folders=['metrics'] if args.main_123_only else DIRS
    if args.figures_only:folders=[p for p in folders if p!='ws11/novelty_ladder']
    for folder in folders:
        orig=ROOT/'results/revision'/folder;replay=WORK/'results/revision'/folder
        for p in replay.iterdir():
            if not p.is_file() or p.is_symlink() or not (p.suffix in {'.tsv','.json'} or p.name.endswith('.tsv.gz')):continue
            old=orig/p.name
            if not old.exists() or sha(old)!=sha(p):add('results/revision/'+folder+'/'+p.name,p,'Recomputed corrected predictions/statistics, or explicit cohort/definition correction')
    for name in ([] if args.main_123_only else ['synthesis_table.tsv','synthesis_table_with_cami2.tsv']):
        p=WORK/'results/revision/real_data'/name;add('results/revision/real_data/'+name,p,'Rebuilt external-cohort synthesis')
    for row in pd.read_csv(OUT/'prediction_consumer_map.tsv',sep='\t').itertuples():
        p=ROOT/row.consumer;logical=str(p.relative_to(WORK));add(logical,p,'Official selected-stage prediction table; stage1 quantitatively unscored')
    for name in ['independent_duplication_statistics.tsv','duplication_raw_predictions_and_truth.tsv.gz','independent_figure3f_thresholds.tsv']:
        add('results/revision/fig5c_verification_resubmission5/'+name,OUT/'independent_figures'/name,'New independent recalculation from stage-corrected predictions; original verification retained')
    # Every historically touched source is retained: verify the original raw
    # stage inputs and historical tables relevant to the replay, not other
    # agents\' actively generated resubmission5 results.
    historical=pd.read_csv(OUT/'historical_immutability_before.tsv',sep='\t');checks=[]
    prefixes=['data/benchmarks/']+['results/revision/'+p+'/' for p in DIRS]
    for r in historical.itertuples():
        if 'resubmission5' in r.path or not any(r.path.startswith(prefix) for prefix in prefixes):continue
        p=ROOT/r.path;assert p.exists() and sha(p)==r.sha256,('Historical input changed',r.path);checks.append(dict(path=r.path,sha256=r.sha256,unchanged=True))
    pd.DataFrame(checks).to_csv(OUT/'historical_immutability_verified.tsv',sep='\t',index=False)
    additional=[]
    if not args.main_123_only:
        p=ROOT/'results/revision/motivating_threshold_resubmission5/summary.json';r=json.loads(p.read_text());assert r['status']=='MOTIVATING_THRESHOLD_COMPLETE'
        additional=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)),dict(path='results/revision/motivating_threshold_resubmission5/high_contamination_mae.tsv',sha256=sha(p.parent/'high_contamination_mae.tsv'))]
    result=dict(status='ANALYSIS_ARTIFACTS_READY',scope='MAIN_1_2_3_ONLY' if args.main_123_only else 'FIGURES_ONLY_NOVELTY_TABLE_PENDING' if args.figures_only else 'ALL_AFFECTED_ANALYSES',parser_verified=True,required_analysis_stages=stages,overrides=overrides,additional_sources=additional,notes=['Only reads use these overrides; writes are never redirected.','Historical raw inputs and outputs remain available unchanged.','This gate does not imply final figure/workbook/document reconciliation is complete.'])
    (OUT/'source_overrides.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    pd.DataFrame([dict(logical_path=k,**v) for k,v in sorted(overrides.items())]).to_csv(OUT/'corrected_artifact_allowlist.tsv',sep='\t',index=False)
    print('Verified override sources:',len(overrides),'historical files unchanged:',len(checks))
if __name__=='__main__':main()
