#!/usr/bin/env python3
"""Recompute numerical results from deposited per-sample predictions and truth."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

STAGES={
 'classification': [('102_mimag_and_thresholds.py',[])],
 'signed_errors': [('103_signed_errors.py',[])],
 'accuracy': [('104_clustered_statistics.py',[]),('215_reference_statistics.py',['pooled'])],
 'domain': [('106_domain_size_and_gunc.py',[])],
 'macro_f1': [('211_ws11p_macro_f1_paired.py',[])],
 'set_f': [('147_analyze_contamination_types.py',['--workers','1','--skip-provenance','--skip-duplication-scan']),('215_reference_statistics.py',['set_f'])],
 'set_g': [('153_error_robustness_analysis.py',['--workers','1','--skip-provenance'])],
 'set_h': [('191_ws1_11_analysis.py',[]),('192_ws1_11_reference_level_scores.py',['--threads','1'])],
 'cami': [('201_ws38_cami2_analysis.py',[])],
 'real_data': [('143_realdata_analysis.py',['--track','zymo']),('143_realdata_analysis.py',['--track','meslier']),('143_realdata_analysis.py',['--track','ncbi'])],
 'novelty': [('210_ws11n_novelty_ladder.py',[])],
 'genus': [('287_evaluate_resubmission7_holdout.py',['--from-predictions','results/revision/holdout_resubmission7/per_sample_predictions.tsv.gz','--output-dir','results/revision/holdout_resubmission7/reproduced_statistics'])],
 'primary_tables': [('108_definitive_results_table.py',[])],
}
EXPECTED={
 'classification':['metrics/ws5.1_mimag_overall_metrics.tsv','metrics/ws5.2_threshold_analysis.tsv'],
 'signed_errors':['metrics/ws5.3_signed_errors_overall.tsv'],
 'accuracy':['metrics/ws5.5_table_S2_rebuilt.tsv','metrics/ws5.4_clustered_tests.tsv'],
 'domain':['metrics/ws5.8_domain_restriction_accuracy.tsv','metrics/ws5.9_size_bins.tsv'],
 'macro_f1':['ws11/macro_f1_paired/macro_f1_paired_differences.tsv'],
 'set_f':['set_F/set_F_cells_type_x_distance.tsv','set_F/set_F_paired_comparisons.tsv'],
 'set_g':['set_G/set_G_curves.tsv','set_G/set_G_paired_degradation.tsv','set_G/set_G_head_to_head.tsv'],
 'set_h':['circularity/ws1_11_arm_metrics.tsv','circularity/ws1_11_did_vs_competitors.tsv','circularity/ws1_11_reference_level_scores.tsv'],
 'cami':['cami2/analysis/cami2_accuracy_by_cohort.tsv','cami2/analysis/cami2_tool_scoring_coverage.tsv','cami2/analysis/cami2_paired_tests.tsv'],
 'real_data':['real_data/zymo/metrics_by_cohort.tsv','real_data/meslier/metrics_by_cohort.tsv','real_data/meslier/tool_scoring_coverage.tsv','real_data/ncbi_pairs/metrics_by_cohort.tsv','real_data/ncbi_pairs/paired_tests.tsv'],
 'novelty':['ws11/novelty_ladder/novelty_ladder_metrics.tsv','ws11/novelty_ladder/novelty_ladder_paired_tests.tsv'],
 'genus':['holdout_resubmission7/'+name+'.tsv' for name in ['head_to_head_by_group','per_reference_errors','mimag_confusion_by_group','did','control_model_difference','sensitivity_did']],
 'primary_tables':['metrics/definitive_table_5set.tsv','metrics/definitive_thresholds.tsv','metrics/definitive_mimag.tsv','metrics/definitive_domain_restriction.tsv'],
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compare(a,b):
 import numpy as np
 import pandas as pd
 x=pd.read_csv(a,sep='\t');y=pd.read_csv(b,sep='\t')
 # Historic tables also contain withdrawn datasets and earlier models. The
 # declared replay target is the current five-set/four-tool scientific panel.
 # Statistical context predictions needed for the original BH family are kept
 # as inputs, but superseded result rows are not presented as current evidence.
 current_sets={'set_A_v2','set_B_v2','set_C_clean','set_D_clean','set_E','POOLED_leakage_free_5_sets'}
 current_tools={'magicc_v5','checkm2','cocopye','deepcheck'}
 def current(d):
  scope='set' if 'set' in d else ('scope' if 'scope' in d else None)
  if scope and d[scope].isin(current_sets).any():
   d=d[d[scope].isin(current_sets)]
   for c in ['tool','reference_tool','comparison_tool']:
    if c in d:d=d[d[c].isin(current_tools)]
  return d.reset_index(drop=True)
 x=current(x);y=current(y)
 if list(x.columns)!=list(y.columns):raise AssertionError(f'Columns differ: {a.name}')
 if x.shape!=y.shape:raise AssertionError(f'Shape differs: {a.name}: {x.shape} vs {y.shape}')
 cells=0;max_error=0.
 for c in x.columns:
  if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
   u=x[c].to_numpy(float);v=y[c].to_numpy(float)
   np.testing.assert_allclose(u,v,rtol=1e-9,atol=1e-10,equal_nan=True,err_msg=f'{a.name}:{c}')
   valid=np.isfinite(u)&np.isfinite(v)
   if valid.any():max_error=max(max_error,float(np.max(np.abs(u[valid]-v[valid]))))
  else:
   pd.testing.assert_series_equal(x[c],y[c],check_dtype=False,check_names=False)
  cells+=len(x)
 return dict(rows=len(x),columns=len(x.columns),compared_cells=cells,max_absolute_numeric_difference=max_error)

def output_path(root,stage,rel):
 # The inference evaluator protects its original result directory. Cached
 # genus statistics are independently regenerated in a separate output folder.
 if stage=='genus':return root/'holdout_resubmission7/reproduced_statistics'/Path(rel).name
 return root/rel

def run(stage,workspace):
 root=workspace/'results/revision';audit=workspace/'reproduction_audit';audit.mkdir(exist_ok=True)
 expected=audit/'expected'/stage;expected.mkdir(parents=True,exist_ok=True)
 for rel in EXPECTED[stage]:
  source=root/rel;target=expected/rel
  if not target.exists():
   target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  # Removing declared summaries ensures cached output cannot pass as a recomputation.
  actual=output_path(root,stage,rel)
  if actual.exists():actual.unlink()
 log=audit/(stage+'.log');record=dict(stage=stage,status='RUNNING',started=time.time(),commands=[])
 env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(audit/'matplotlib'),PYTHONPATH=str(workspace))
 with log.open('w') as f:
  for name,args in STAGES[stage]:
   command=[sys.executable,str(workspace/'scripts'/name),*args]
   completed=subprocess.run(command,cwd=workspace,env=env,stdout=f,stderr=subprocess.STDOUT)
   record['commands'].append(dict(command=command,script_sha256=sha(workspace/'scripts'/name),return_code=completed.returncode))
   if completed.returncode:
    record.update(status='FAILED',elapsed_seconds=time.time()-record['started']);(audit/(stage+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    raise RuntimeError(f'{stage} failed; see {log}')
 record['comparisons']=[]
 try:
  for rel in EXPECTED[stage]:
   actual=output_path(root,stage,rel);reference=expected/rel
   comparison=compare(reference,actual)
   record['comparisons'].append(dict(path=rel,expected_sha256=sha(reference),regenerated_sha256=sha(actual),**comparison))
 except Exception as error:
  record.update(status='COMPARISON_FAILED',error=str(error),elapsed_seconds=time.time()-record['started'])
  (audit/(stage+'.json')).write_text(json.dumps(record,indent=2)+'\n')
  raise
 record.update(status='PASS',elapsed_seconds=time.time()-record['started'])
 (audit/(stage+'.json')).write_text(json.dumps(record,indent=2)+'\n')
 print(f'{stage}: PASS ({record["elapsed_seconds"]:.1f}s)',flush=True)
 return record

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workspace',type=Path,required=True)
 ap.add_argument('--stages',nargs='+',choices=STAGES,default=list(STAGES))
 ap.add_argument('--jobs',type=int,default=1,choices=range(1,5));args=ap.parse_args()
 workspace=args.workspace.resolve()
 if not (workspace/'RESTORE_MANIFEST.json').is_file():ap.error('Run 01_restore.py first')
 independent=[s for s in args.stages if s!='primary_tables']
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
  records=list(pool.map(lambda stage:run(stage,workspace),independent))
 # The final presentation tables consume classification, accuracy and domain
 # summaries, and must run after those independent numerical stages finish.
 if 'primary_tables' in args.stages:records.append(run('primary_tables',workspace))
 summary=dict(status='PASS',stages=records,compared_cells=sum(c['compared_cells'] for r in records for c in r['comparisons']))
 (workspace/'reproduction_audit/completion.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(f'Reproduced {len(records)} stages; verified {summary["compared_cells"]} table cells')

if __name__=='__main__':main()
