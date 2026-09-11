# Scientific correction and figure-source audit

This report concerns the completed comparator/statistical corrections. The new matched ten-family training and the final submission documents have separate completion gates. Original results remain unchanged in their historical directories; final consumers use the finite, hashed `source_overrides.json` map.

## Official comparator output selection

CoCoPyE 0.5.0 emits intermediate marker and neural estimates alongside a selected API/CSV stage. The historical extraction used finite stage 3 estimates even when stage 2 was selected. The corrected parser uses stage 2 marker or stage 3 neural values as selected, converts fractions to percentages once, and treats stage 1 as quantitatively unscored. It checks every full CSV row against the installed Result API. The paper's stage I/II numbering differs from the CSV/API 1/2/3 numbering. No estimate is arbitrarily clipped. The installed source hash and raw-input checksums are in `parser_verification.json` and `affected_cohort_inventory.tsv`; the official methods are at https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giae079/7841111.

All 5000 primary benchmark assemblies remain scored by all four tools. Selected-stage reconstruction changes 1702 CoCoPyE rows. MAGICC, CheckM2 and DeepCheck predictions and per-set metrics are independently unchanged. The pooled intervals use 1648 physical dominant accessions shared across the five sets, with each set retaining weight 1/5 in every bootstrap draw.

| Tool | Completeness MAE [95% CI], pp | Contamination MAE [95% CI], pp |
|---|---:|---:|
| MAGICC | 4.6179 [4.3381, 4.8920] | 5.3086 [5.0623, 5.5755] |
| CheckM2 | 5.8031 [5.6016, 6.0144] | 22.1447 [21.6174, 22.6820] |
| CoCoPyE | 6.8088 [6.5650, 7.0657] | 15.9869 [15.5110, 16.4678] |
| DeepCheck | 9.8310 [9.5223, 10.1713] | 26.4331 [25.8228, 27.0525] |

MAGICC has the lowest pooled MAE point estimate for both outputs. Under the shipped Methods' agreement requirement, the 30 paired mean-MAE comparisons give 22 MAGICC advantages,4 disadvantages and 4 unresolved comparisons. The separate BH reference-mean rank-test component gives 24/4/2; it is not substituted for the agreement-supported mean-MAE count. `document_deltas/decision_rule_audit.json` records the distinction without retroactively calling the criterion prespecified. Seven of nine C-clean/D-clean/E macro-F1 comparison intervals exclude zero; the three-class point estimates favor MAGICC on each set.

At 5% contamination, MAGICC has the highest balanced-accuracy point estimate on C-clean, D-clean and E; CoCoPyE leads Set B. Set A lacks a contaminated truth class, so binary balanced accuracy is undefined. On C-clean, MAGICC false-fails 23/52 truly clean genomes and false-passes 19/948 truly contaminated genomes; corrected CoCoPyE counts are 35/52 and 122/948. Point rankings do not imply universal or statistically supported superiority. All component rates and denominators remain visible.

## Stress-test verification and display

The 400 control/duplicated MAGICC outputs were verified by fresh inference, and all 320 duplicated FASTAs retain their original contigs plus exact copied subsequences. Script 259 independently reconstructs the corrected 40 tool/output/dose statistics and 20 threshold rows. At 40% added/original bp, MAGICC completeness MAE is 15.9854 pp versus 5.5197,4.8909 and 6.1067 for CheckM2, CoCoPyE and DeepCheck. Its paired completeness increase is 11.5877 pp [8.7882,14.1000]. Contamination MAEs are 19.3299,22.8267,32.5681 and 27.1610 respectively: all tools worsen, while MAGICC has the lowest point estimate at that endpoint.

The intervention is localized exact duplication of assembled sequence, not a measurement of natural coverage variation or artifact incidence.40% added/original bp is 28.5714% of final bp. It preserves the selected unique-k-mer set, but not relative k-mer composition; it does not identify a unique internal model mechanism. Full stress-test results remain in Figure S11 and source tables; the main text gives concise quantified findings. Interpolated crossings are diagnostic summaries, not operating cutoffs. Valid unbounded regression estimates are displayed in full. Script 265 independently reproduced the DeepCheck outliers with the native network, and script 273 audits final plot ranges.

## Reference units and source eligibility

Set F's 87 paired comparisons now use one mean paired difference per dominant reference (100 references), with BH correction across the same 87 comparisons. Existing reference-cluster Hodges–Lehmann intervals and point estimates are unchanged. Support requires a reference-aware q<0.05 and a concordant interval excluding zero. Two support decisions change;16 historical 'tie' labels are renamed 'unresolved'. The genus CoCoPyE advantage and family MAGICC advantage versus CoCoPyE become unresolved. The original row-level p/q values are retained explicitly.

Native CAMI marine metadata and the official data-generation record identify 777 microbial references and 200 RNODE circular elements (108 plasmid,4 virus,88 unknown). The old exclusion missed the 88 unknown elements. Their taxid 32644 is the catchall 'unidentified', so it also cannot establish within-species relatedness. All 148 gold and 206 mixed bins exclusive to the unscoped MAGICC–CheckM2 pairwise cohort came from RNODE elements. Every examined input was selected for comparator processing; all 354 FASTAs contain genuine A/C/G/T sequence and match raw CheckM2 size/prediction records. That sequence validity does not make their reference units suitable for prokaryotic genome-quality validation.

The final scientific analysis excludes circular elements from both dominant and donor positions. The corrected four-tool primary cohorts contain no such elements and remain exactly unchanged: marine 339 gold/864 mixed, strain-madness 700 gold/2250 mixed. All 32 primary metric rows, intervals, paired tests and primary membership/predictions are identical after this source restriction. Correctly scoped MAGICC–CheckM2 pairwise and four-tool common cohorts coincide. Out-of-scope outputs remain construction-audit records, not extra validation. The marine observed-source audit now contains 776 microbial sources,340 with detected training overlap; full microbial gold truth has 2062/4694 bins below 50% completeness. Native-source eligibility and counts are independent of which estimator performs better.

## Remaining interpretation boundaries and reproduction

The fixed-weight normalization sensitivity does not estimate or correct training-time unsupervised preprocessing overlap: production V5 inherited a scaler fitted on V4 training, validation and test feature rows. Its small inference-perturbation effects and unchanged C-clean threshold calls remain valid. Final comparative statements use the selected-stage tables. NCBI's 913-pair/124-species primary analysis includes three existing tools; unmatched draft bp is an upper contamination proxy, and no missing DeepCheck result is invented. The contamination≤35% secondary benchmark is separately labeled and does not replace the fixed primary panel.

`completion.json` links finite input/output manifests and completed analysis stages. `figure_input_manifest.tsv` supplies the exact renderer/data dependency closure. Independent corrected duplication/threshold reproduction is `python scripts/259_verify_corrected_figure_comparisons.py`; script 241 supplies the separately retained full sequence/fresh-MAGICC audit. Scripts 252–276 retain the stage selection, replay, definition corrections, source audits and consumer checks. Generated document and workbook integrity is checked after final rendering; their volatile binary hashes are intentionally outside the statistical completion record.
