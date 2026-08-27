# MAGICC headline-claim reproduction (WS7.4 / WS7.11)

Generated 2026-08-25T16:48:35.473431+00:00 by `workflow/Snakefile` via `workflow/scripts/compare_and_report.py`.

**Result: PASS**

## Frozen artefacts

* Model **V5**, SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`
* 9,249 selected 9-mers, SHA256 `02a2857d5dffa6bfe6b26f624c78aa7a72a619e0e2a08919f3dc7fbe5d5a0eae`
* Artefact checks: **PASS**

## Pooled leakage-free five-set panel

Sets: set_A_v2, set_B_v2, set_C_clean, set_D_clean, set_E — 5,000 genomes, 2,586 clusters ((set, reference genome) pair).

| Metric | Reproduced MAE (pp) | 95 % CI | Recorded MAE | Δ (pp) | Status |
|---|---|---|---|---|---|
| completeness | 4.6179 | [4.332, 4.927] | 4.6179 | +0.00000 | MATCH |
| contamination | 5.3086 | [5.013, 5.601] | 5.3086 | +0.00000 | MATCH |

R² (coefficient of determination, 1 − SS_res/SS_tot — protocol §4.4d):
* completeness: reproduced **0.7853**, recorded 0.7853
* contamination: reproduced **0.9149**, recorded 0.9149

## Per set

| Set | n | clusters | completeness MAE | contamination MAE | comp bias | cont bias |
|---|---|---|---|---|---|---|
| set_A_v2 | 1,000 | 798 | 2.1766 | 0.8285 | +0.0922 | +0.8285 |
| set_B_v2 | 1,000 | 803 | 2.9724 | 4.4535 | -2.9724 | -0.1356 |
| set_C_clean | 1,000 | 100 | 6.8216 | 9.0801 | -2.3197 | -3.5536 |
| set_D_clean | 1,000 | 100 | 5.6330 | 6.5229 | -0.2624 | -3.8341 |
| set_E | 1,000 | 785 | 5.4859 | 5.6579 | +1.1277 | -3.1182 |

## What this does and does not establish

* It **does** establish that the released model artefact, the released 9-mer list and normalisation parameters, and the released benchmark genomes together regenerate the pooled leakage-free headline numbers from raw FASTA, end to end, in one command.
* It **does not** establish that the model weights can be retrained bit-exactly: V5 training was never seeded (protocol §4.4b). The weights are pinned by SHA256 and inference is deterministic — see `results/revision/reproducibility/DETERMINISM.md`.
* Sets C and D from the submitted manuscript are **excluded**: their dominant genomes came from the training split (100 % of Set C, 90.3 % of Set D) and those results are withdrawn, not reproduced.
