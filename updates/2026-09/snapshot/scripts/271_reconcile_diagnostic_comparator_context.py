#!/usr/bin/env python3
"""Link historical reproduction and MAGICC-only sensitivity to final comparators."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5';DEST=OUT/'diagnostic_context'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(exist_ok=True)
    tp=OUT/'independent_figures/independent_figure3f_thresholds.tsv';gp=OUT/'independent_figures/independent_duplication_statistics.tsv'
    t=pd.read_csv(tp,sep='\t');g=pd.read_csv(gp,sep='\t');norm=ROOT/'results/revision/normalization_sensitivity_resubmission5/classification_sensitivity.tsv'
    winners={s:z.loc[z.balanced_accuracy.idxmax(),'tool'] for s,z in t.groupby('dataset') if z.balanced_accuracy.notna().any()}
    assert all(winners[s]=='magicc_v5' for s in ['set_C_clean','set_D_clean','set_E'])
    c=t[t.dataset.eq('set_C_clean')].set_index('tool');assert (c.loc['magicc_v5',['false_fail_n','false_pass_n']].to_numpy(int)==[23,19]).all()
    rows=g[g.added_percent_original.eq(40)]
    block='''
## Final selected-stage comparator reconciliation

The historical numerical tables above are retained as a reproduction record,
not the final CoCoPyE comparison. Script259 independently recalculates the
selected-stage values in `results/revision/cocopye_stage_resubmission5/independent_figures/`.
At40% added/original bp, completeness MAEs are15.985403 (MAGICC),5.519696
(CheckM2),4.890873 (CoCoPyE),6.106747 (DeepCheck); contamination MAEs are
19.329862,22.826661,32.568112,27.160980 respectively. MAGICC's verified
completeness degradation remains a limitation, while it has the lowest
contamination MAE point estimate at that dose; all four contamination errors
increase relative to control. No empirical artifact-incidence claim follows.

At the5% contamination threshold on C-clean, final CoCoPyE false-fail and
false-pass counts are35/52 and122/948; MAGICC remains23/52 and19/948.
MAGICC retains the highest balanced-accuracy point estimate on C-clean,
D-clean and E, while CoCoPyE leads SetB; SetA balanced accuracy is undefined
because one truth class is absent. These point rankings are descriptive.
The final all-three-test agreement count for mean-MAE comparisons is22/4/4;
the distinct BH rank-test component partition is24/4/2.
'''
    block=block.replace('At40','At 40').replace('the5','the 5').replace('are15','are 15').replace('are35','are 35').replace('remains23','remains 23').replace('is22','is 22').replace('is24','is 24')
    report=ROOT/'results/revision/fig5c_verification_resubmission5/VERIFICATION_REPORT.md';s=report.read_text();marker='\n## Final selected-stage comparator reconciliation\n';s=s.split(marker)[0];report.write_text(s.rstrip()+'\n'+block)
    text='''The normalization sensitivity is a fixed-weight MAGICC-only inference perturbation. Its baseline and perturbed predictions, confusion matrices and C-clean23/52 and19/948 threshold counts do not depend on CoCoPyE parsing. Scripts249/250 read historical comparison tables only to verify the unchanged MAGICC baseline rows; they do not compute comparator rankings. Any comparative context in the submission uses selected-stage CoCoPyE results in independent_figures/independent_figure3f_thresholds.tsv and the corrected macro-F1 tables. Small inference-perturbation effects do not quantify or eliminate training-time unsupervised preprocessing overlap.\n'''
    (DEST/'normalization_context.md').write_text(text)
    summary=dict(status='DIAGNOSTIC_COMPARATOR_CONTEXT_RECONCILED',balanced_accuracy_point_leaders=winners,absence_of_true_contaminated_class='SetA: binary balanced accuracy undefined',normalization_scope=text.strip(),historical_verification='Preserved historical reproduction text and explicitly appended final selected-stage comparison linkage.',input_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [tp,gp,norm,ROOT/'scripts/249_normalization_sensitivity.py',ROOT/'scripts/250_audit_normalization_sensitivity.py',Path(__file__).resolve()]],output_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [report,DEST/'normalization_context.md']])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Historical and MAGICC-only diagnostic context reconciled')
if __name__=='__main__':main()
