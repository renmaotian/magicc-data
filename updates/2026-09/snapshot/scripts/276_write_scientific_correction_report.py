#!/usr/bin/env python3
"""Readable scientific audit report from finalized corrected numerical sources."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision/cocopye_stage_resubmission5'
def main():
    scope=json.loads((OUT/'cami_microbial_scope/summary.json').read_text());assert scope['status']=='CAMI_MICROBIAL_SCOPE_COMPLETE'
    p=json.loads((OUT/'document_deltas/primary_summary.json').read_text());g=pd.read_csv(OUT/'independent_figures/independent_duplication_statistics.tsv',sep='\t');f=json.loads((OUT/'setF_reference_tests/summary.json').read_text())
    rows=['| Tool | Completeness MAE [95% CI], pp | Contamination MAE [95% CI], pp |','|---|---:|---:|']
    for tool,label in [('magicc_v5','MAGICC'),('checkm2','CheckM2'),('cocopye','CoCoPyE'),('deepcheck','DeepCheck')]:
        parts=[]
        for metric in ['completeness','contamination']:
            r=next(r for r in p['pooled_mae'] if r['tool']==tool and r['metric']==metric);parts.append(f"{r['mae']:.4f} [{r['mae_ci_lo']:.4f}, {r['mae_ci_hi']:.4f}]")
        rows.append('| '+label+' | '+' | '.join(parts)+' |')
    text='''# Scientific correction and figure-source audit

This report concerns the completed comparator/statistical corrections. The new matched ten-family training and the final submission documents have separate completion gates. Original results remain unchanged in their historical directories; final consumers use the finite, hashed `source_overrides.json` map.

## Official comparator output selection

CoCoPyE0.5.0 emits intermediate marker and neural estimates alongside a selected API/CSV stage. The historical extraction used finite stage3 estimates even when stage2 was selected. The corrected parser uses stage2 marker or stage3 neural values as selected, converts fractions to percentages once, and treats stage1 as quantitatively unscored. It checks every full CSV row against the installed Result API. The paper's stageI/II numbering differs from the CSV/API1/2/3 numbering. No estimate is arbitrarily clipped. The installed source hash and raw-input checksums are in `parser_verification.json` and `affected_cohort_inventory.tsv`; the official methods are at https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giae079/7841111.

All5000 primary benchmark assemblies remain scored by all four tools. Selected-stage reconstruction changes1702 CoCoPyE rows. MAGICC, CheckM2 and DeepCheck predictions and per-set metrics are independently unchanged. The pooled intervals use1648 physical dominant accessions shared across the five sets, with each set retaining weight1/5 in every bootstrap draw.

'''+ '\n'.join(rows)+'''

MAGICC has the lowest pooled MAE point estimate for both outputs. Under the shipped Methods' agreement requirement, the30 paired mean-MAE comparisons give22 MAGICC advantages,4 disadvantages and4 unresolved comparisons. The separate BH reference-mean rank-test component gives24/4/2; it is not substituted for the agreement-supported mean-MAE count. `document_deltas/decision_rule_audit.json` records the distinction without retroactively calling the criterion prespecified. Seven of nine C-clean/D-clean/E macro-F1 comparison intervals exclude zero; the three-class point estimates favor MAGICC on each set.

At5% contamination, MAGICC has the highest balanced-accuracy point estimate on C-clean, D-clean and E; CoCoPyE leads SetB. SetA lacks a contaminated truth class, so binary balanced accuracy is undefined. On C-clean, MAGICC false-fails23/52 truly clean genomes and false-passes19/948 truly contaminated genomes; corrected CoCoPyE counts are35/52 and122/948. Point rankings do not imply universal or statistically supported superiority. All component rates and denominators remain visible.

## Stress-test verification and display

The400 control/duplicated MAGICC outputs were verified by fresh inference, and all320 duplicated FASTAs retain their original contigs plus exact copied subsequences. Script259 independently reconstructs the corrected40 tool/output/dose statistics and20 threshold rows. At40% added/original bp, MAGICC completeness MAE is15.9854pp versus5.5197,4.8909 and6.1067 for CheckM2, CoCoPyE and DeepCheck. Its paired completeness increase is11.5877pp [8.7882,14.1000]. Contamination MAEs are19.3299,22.8267,32.5681 and27.1610 respectively: all tools worsen, while MAGICC has the lowest point estimate at that endpoint.

The intervention is localized exact duplication of assembled sequence, not a measurement of natural coverage variation or artifact incidence.40% added/original bp is28.5714% of final bp. It preserves the selected unique-k-mer set, but not relative k-mer composition; it does not identify a unique internal model mechanism. Full stress-test results remain in FigureS11 and source tables; the main text gives concise quantified findings. Interpolated crossings are diagnostic summaries, not operating cutoffs. Valid unbounded regression estimates are displayed in full. Script265 independently reproduced the DeepCheck outliers with the native network, and script273 audits final plot ranges.

## Reference units and source eligibility

SetF's87 paired comparisons now use one mean paired difference per dominant reference (100 references), with BH correction across the same87 comparisons. Existing reference-cluster Hodges–Lehmann intervals and point estimates are unchanged. Support requires a reference-aware q<0.05 and a concordant interval excluding zero. Two support decisions change;16 historical 'tie' labels are renamed 'unresolved'. The genus CoCoPyE advantage and family MAGICC advantage versus CoCoPyE become unresolved. The original row-level p/q values are retained explicitly.

Native CAMI marine metadata and the official data-generation record identify777 microbial references and200 RNODE circular elements (108 plasmid,4 virus,88 unknown). The old exclusion missed the88 unknown elements. Their taxid32644 is the catchall 'unidentified', so it also cannot establish within-species relatedness. All148 gold and206 mixed bins exclusive to the unscoped MAGICC–CheckM2 pairwise cohort came from RNODE elements. Every examined input was selected for comparator processing; all354 FASTAs contain genuine A/C/G/T sequence and match raw CheckM2 size/prediction records. That sequence validity does not make their reference units suitable for prokaryotic genome-quality validation.

The final scientific analysis excludes circular elements from both dominant and donor positions. The corrected four-tool primary cohorts contain no such elements and remain exactly unchanged: marine339 gold/864 mixed, strain-madness700 gold/2250 mixed. All32 primary metric rows, intervals, paired tests and primary membership/predictions are identical after this source restriction. Correctly scoped MAGICC–CheckM2 pairwise and four-tool common cohorts coincide. Out-of-scope outputs remain construction-audit records, not extra validation. The marine observed-source audit now contains776 microbial sources,340 with detected training overlap; full microbial gold truth has2062/4694 bins below50% completeness. Native-source eligibility and counts are independent of which estimator performs better.

## Remaining interpretation boundaries and reproduction

The fixed-weight normalization sensitivity does not estimate or correct training-time unsupervised preprocessing overlap: production V5 inherited a scaler fitted on V4 training, validation and test feature rows. Its small inference-perturbation effects and unchanged C-clean threshold calls remain valid. Final comparative statements use the selected-stage tables. NCBI's913-pair/124-species primary analysis includes three existing tools; unmatched draft bp is an upper contamination proxy, and no missing DeepCheck result is invented. The contamination≤35% secondary benchmark is separately labeled and does not replace the fixed primary panel.

`completion.json` links finite input/output manifests and completed analysis stages. `figure_input_manifest.tsv` supplies the exact renderer/data dependency closure. Independent corrected duplication/threshold reproduction is `python scripts/259_verify_corrected_figure_comparisons.py`; script241 supplies the separately retained full sequence/fresh-MAGICC audit. Scripts252–276 retain the stage selection, replay, definition corrections, source audits and consumer checks. Generated document and workbook integrity is checked after final rendering; their volatile binary hashes are intentionally outside the statistical completion record.
'''
    # Keep prose readable without altering exact filenames/identifiers.
    import re
    text=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ',text);text=re.sub(r'(?<=\d)(?=[A-Za-z])',' ',text)
    for old,new in [('CheckM 2','CheckM2'),('V 5','V5'),('V 4','V4'),('SetA','Set A'),('SetB','Set B'),('SetF','Set F'),('FigureS 11','Figure S11'),('macro-F 1','macro-F1'),('stageI/II','stage I/II')]:text=text.replace(old,new)
    text=text.replace('scripts/ 259','scripts/259').replace('scripts/ 252','scripts/252').replace('article/doi/10.1093/gigascience/giae 079/7841111','article/doi/10.1093/gigascience/giae079/7841111')
    (OUT/'CORRECTION_REPORT.md').write_text(text);print('Scientific correction report written')
if __name__=='__main__':main()
