#!/usr/bin/env python3
"""Make binary balanced accuracy undefined when a truth class is absent.

This targets binary threshold tables only. Multiclass MIMAG summaries retain
their explicitly separate present-class convention. Counts, conditional error
rates, sensitivity and specificity are unchanged.
"""
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';WORK=OUT/'replay';DEST=OUT/'binary_balanced_accuracy_correction'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(exist_ok=True);framework=WORK/'scripts/101_metrics_framework.py';s=framework.read_text()
    old='"balanced_accuracy": float(np.nanmean([d(tp, n_tp_fail), d(tn, n_tp_pass)])),'
    new='"balanced_accuracy": (float((d(tp, n_tp_fail) + d(tn, n_tp_pass)) / 2) if n_tp_fail > 0 and n_tp_pass > 0 else float("nan")),'
    assert old in s or new in s
    framework.write_text(s.replace(old,new));changes=[]
    for name,stage_id in [('ws5.2_threshold_analysis.tsv','102'),('definitive_thresholds.tsv','108')]:
        path=WORK/'results/revision/metrics'/name;a=pd.read_csv(path,sep='\t');snapshot=DEST/(name+'.before.tsv')
        if not snapshot.exists():shutil.copy2(path,snapshot)
        cols=[c for c in a if c=='balanced_accuracy' or c.startswith('balanced_accuracy_ci_')];bad=a.n_true_pass.eq(0) | a.n_true_fail.eq(0)
        for row in a.loc[bad,['set','tool','criterion','threshold','n_true_pass','n_true_fail','balanced_accuracy']].to_dict('records'):row['source']=name;changes.append(row)
        saved=a.drop(columns=cols).copy();a.loc[bad,cols]=np.nan
        assert a.drop(columns=cols).equals(saved)
        a.to_csv(path,sep='\t',index=False)
        stagepath=OUT/'analysis_status'/f'{stage_id}.json';stage=json.loads(stagepath.read_text())
        for item in stage.get('outputs',[]):
            if item['path']==str(path.relative_to(ROOT)):
                assert item['sha256']==sha(snapshot)
                item['path']=str(snapshot.relative_to(ROOT));item['preservation_note']='Exact stage output before binary balanced-accuracy undefined-class correction; use the script261 final table.'
        stagepath.write_text(json.dumps(stage,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('ba101',framework);fw=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fw;spec.loader.exec_module(fw)
    assert np.isnan(fw.threshold_metrics(np.array([0.,1.]),np.array([0.,6.]),5.,'contamination')['balanced_accuracy'])
    assert fw.threshold_metrics(np.array([0.,10.]),np.array([0.,10.]),5.,'contamination')['balanced_accuracy']==1.
    pd.DataFrame(changes).to_csv(DEST/'undefined_rows.tsv',sep='\t',index=False)
    summary=dict(status='BINARY_BALANCED_ACCURACY_CORRECTION_COMPLETE',definition='Mean of both truth-class recalls; undefined if either truth class has zero support',scope='Binary contamination/completeness threshold criteria only; not multiclass MIMAG balanced accuracy',counts_and_other_rates_unchanged=True,n_rows_marked_undefined=len(changes),framework_path=str(framework.relative_to(ROOT)),framework_sha256=sha(framework),outputs=[dict(path=str((WORK/'results/revision/metrics'/name).relative_to(ROOT)),sha256=sha(WORK/'results/revision/metrics'/name)) for name in ['ws5.2_threshold_analysis.tsv','definitive_thresholds.tsv']])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Undefined binary BA rows:',len(changes))
if __name__=='__main__':main()
