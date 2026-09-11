#!/usr/bin/env python3
"""Recompute established metric pipelines inside the stage-correction replay.

Stage completion records are internal execution sentinels, not the final
scientific correction completion.json, which also requires consumer audits.
"""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay'
STAGES={
 '102':('102_mimag_and_thresholds.py',[],['metrics/ws5.1_mimag_overall_metrics.tsv','metrics/ws5.2_threshold_analysis.tsv']),
 '103':('103_signed_errors.py',[],['metrics/ws5.3_signed_errors_overall.tsv']),
 '104':('104_clustered_statistics.py',[],['metrics/ws5.5_table_S2_rebuilt.tsv','metrics/ws5.4_clustered_tests.tsv']),
 '106':('106_domain_size_and_gunc.py',[],['metrics/ws5.8_domain_restriction_accuracy.tsv','metrics/ws5.9_size_bins.tsv']),
 '108':('108_definitive_results_table.py',[],['metrics/definitive_table_5set.tsv','metrics/definitive_thresholds.tsv','metrics/definitive_mimag.tsv']),
 '211':('211_ws11p_macro_f1_paired.py',[],['ws11/macro_f1_paired/macro_f1_paired_differences.tsv']),
 '147':('147_analyze_contamination_types.py',['--workers','1','--skip-provenance','--skip-duplication-scan'],['set_F/set_F_cells_type_x_distance.tsv','set_F/set_F_paired_comparisons.tsv']),
 '153':('153_error_robustness_analysis.py',['--workers','1','--skip-provenance'],['set_G/set_G_curves.tsv','set_G/set_G_paired_degradation.tsv','set_G/set_G_head_to_head.tsv']),
 '201':('201_ws38_cami2_analysis.py',[],['cami2/analysis/cami2_accuracy_by_cohort.tsv','cami2/analysis/cami2_tool_scoring_coverage.tsv','cami2/analysis/cami2_paired_tests.tsv']),
 '143m':('143_realdata_analysis.py',['--track','meslier'],['real_data/meslier/metrics_by_cohort.tsv','real_data/meslier/metrics_full_tool_scoring_coverage.tsv','real_data/meslier/tool_scoring_coverage.tsv','real_data/meslier/fragmentation_gradient.tsv']),
 '143z':('143_realdata_analysis.py',['--track','zymo'],['real_data/zymo/metrics_by_cohort.tsv']),
 '143n':('143_realdata_analysis.py',['--track','ncbi'],['real_data/ncbi_pairs/metrics_by_cohort.tsv','real_data/ncbi_pairs/paired_tests.tsv']),
 '144':('144_realdata_synthesis.py',[],['real_data/synthesis_table.tsv']),
 '191':('191_ws1_11_analysis.py',[],['circularity/ws1_11_arm_metrics.tsv','circularity/ws1_11_did_vs_competitors.tsv']),
 '192':('192_ws1_11_reference_level_scores.py',['--threads','1'],['circularity/ws1_11_reference_level_scores.tsv']),
 '257':('257_correct_pooled_reference_bootstrap.py',[],['metrics/ws5.5_table_S2_rebuilt.tsv']),
 '204':('204_ws38_feed_ws39_synthesis.py',[],['real_data/synthesis_table_with_cami2.tsv']),
 '205':('205_pooled_cluster_definition_sensitivity.py',[],['metrics/ws5.10_pooled_cluster_definition_sensitivity.tsv']),
 '210':('210_ws11n_novelty_ladder.py',[],['ws11/novelty_ladder/novelty_ladder_metrics.tsv','ws11/novelty_ladder/novelty_ladder_paired_tests.tsv'])}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(stage,force=False):
    script,args,expected=STAGES[stage];source=(ROOT if stage=='257' else WORK)/'scripts'/script;config=WORK/'scripts/config_revision_metrics.yaml'
    assert (WORK/'results/revision/metrics').resolve().is_relative_to(WORK.resolve())
    assert (WORK/'results/revision/ws11/macro_f1_paired').resolve().is_relative_to(WORK.resolve())
    for rel in expected:assert (WORK/'results/revision'/rel).resolve().is_relative_to(WORK.resolve()), f'Output symlink escapes replay: {rel}'
    fingerprint={'script_sha256':sha(source),'config_sha256':sha(config),'corrected_predictions_manifest_sha256':sha(OUT/'corrected_prediction_manifest.tsv')}
    statuspath=OUT/'analysis_status'/f'{stage}.json';statuspath.parent.mkdir(exist_ok=True)
    if not force and statuspath.exists():
        old=json.loads(statuspath.read_text())
        # Earlier driver revisions misspelled expected filenames, although the
        # unchanged analysis process succeeded. Repair only this metadata case.
        if old.get('status')=='OUTPUT_SCHEMA_CHECK_NEEDED' and old.get('return_code')==0 and old.get('inputs')==fingerprint and all((WORK/'results/revision'/p).is_file() for p in expected):
            old['status']='COMPLETE';old['outputs']=[{'path':str((WORK/'results/revision'/p).relative_to(ROOT)),'sha256':sha(WORK/'results/revision'/p)} for p in expected]
            old['schema_reconciliation']='Expected filename spelling corrected after successful unchanged analysis; no statistical rerun.';old.pop('missing_expected',None);statuspath.write_text(json.dumps(old,indent=2)+'\n')
        if old.get('status')=='COMPLETE' and old.get('inputs')==fingerprint:
            if all((ROOT/x['path']).is_file() and sha(ROOT/x['path'])==x['sha256'] for x in old['outputs']):return old
    log=OUT/'logs'/f'{stage}.log';log.parent.mkdir(exist_ok=True)
    env=dict(os.environ);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/magicc2-mpl')
    started=time.time();status={'stage':stage,'status':'RUNNING','inputs':fingerprint,'command':[sys.executable,str(source),*args],'log':str(log.relative_to(ROOT))};status['started_epoch']=started;statuspath.write_text(json.dumps(status,indent=2)+'\n')
    with log.open('w') as f:result=subprocess.run(status['command'],cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
    status['return_code']=result.returncode;status['elapsed_seconds']=time.time()-started
    status['status']='COMPLETE' if result.returncode==0 else 'FAILED';status['outputs']=[]
    if result.returncode==0:
        for rel in expected:
            p=WORK/'results/revision'/rel
            if not p.is_file():status['status']='OUTPUT_SCHEMA_CHECK_NEEDED';status.setdefault('missing_expected',[]).append(rel)
            else:status['outputs'].append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    statuspath.write_text(json.dumps(status,indent=2)+'\n');print(stage,status['status'],f"{status['elapsed_seconds']:.1f}s",flush=True)
    if status['status']!='COMPLETE':raise RuntimeError(f'Stage {stage}: {status["status"]}; see {log}')
    return status

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=STAGES);parser.add_argument('--primary',action='store_true');parser.add_argument('--force',action='store_true');args=parser.parse_args()
    assert (OUT/'replay_setup.json').is_file()
    if args.primary:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda stage:run(stage,args.force),['102','103','104','106']))
        run('257',args.force);run('108',args.force);run('211',args.force)
    elif args.stage:run(args.stage,args.force)
    else:parser.error('Choose --primary or --stage')
if __name__=='__main__':main()
