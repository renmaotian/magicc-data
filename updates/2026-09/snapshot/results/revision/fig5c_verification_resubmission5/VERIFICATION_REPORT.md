> Subsequent comparator parsing audit: this report verifies the historical extracted prediction values. CoCoPyE full-verbosity CSVs also carry a selected `stage`, and the original parser used finite stage-3 predictions even for selected stage-2 rows. The released CoCoPyE API uses stage-2 predictions for those rows. A stage-aware correction is therefore required for the fair CoCoPyE comparison; the independently verified MAGICC duplication effect and fresh-inference agreement remain valid. See `SCIENTIFIC_CONSISTENCY_AUDIT.md`.

# Independent verification of former Figure 5c and Figure 3f

Status: **PASS**. Reproduce with `conda run -n magicc2 python scripts/241_verify_resubmission5_figures.py`. The audit imports no original summary-analysis functions and writes all checks to this directory.

## Design and raw-sequence verification

Set G uses 80 finished, held-out reference genomes from 35 phyla, selected by a seeded round-robin scheme over phyla. Each reference has one base assembly and 24 matched error arms (1,920 assemblies). The base assemblies cross target completeness 60–100%, contamination 0/2/5/10/20%, and four fragmentation tiers. Realized completeness spans 58.0034–100%; contamination spans 0–20%. The same dominant reference, base fragmentation and cross-phylum contaminants are used within a reference across all error arms. This deliberately isolates the injected process; it does not measure artifact incidence in environmental assemblies.

The generator (`scripts/150_error_injection_module.py`, `scripts/151_generate_set_G.py`) samples contigs with weights equal to contig length multiplied by a lognormal draw (sigma = 1). It appends exact subsequences, usually 25–100% of a selected contig, to attain the requested amount of extra sequence, and shuffles the resulting contigs. It does not simulate reads, coverage-dependent assembly or the probability that an assembler produces duplicate contigs.

All **320 duplication FASTAs** were read independently. Every original contig is retained, every added segment is an exact substring of an original contig, and the added length agrees exactly with the saved `dup_bp` and rounded target. The dose is **added duplicate bp / original assembly bp**, not duplicate bp / final assembly bp. The four nominal doses 5%, 10%, 20%, 40% correspond to 4.762%, 9.091%, 16.667%, 28.571% of final assembly bp. The final assembly at the highest dose is 1.4 times its control length.

Primary truth follows represented source content: dominant-derived bp in the control divided by full dominant-reference length for completeness, and foreign-derived control bp divided by that same length for contamination. Appending copies adds no new represented source positions, so both primary labels stay fixed. The alternative sensitivity convention counts all duplicated bp as contamination; it remains available in `results/revision/set_G/set_G_duplication_accounting_sensitivity.tsv`. It changes contamination comparisons but cannot remove the completeness effect.

## Numerical verification

Fresh inference through the released MAGICC CLI on all 400 control-plus-duplication assemblies reproduced saved predictions to at most **0.00005 pp**, entirely within four-decimal TSV rounding. The default model resolution verifies frozen V5 SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096` before inference. The audit also validated all four raw prediction tables (7,680 rows total), recalculated 40 tool × metric × dose statistics, and reproduced published means, biases and 95% confidence endpoints to their four-decimal rounding. Bootstrap intervals use 2,000 reference resamples, CRC32 seeds and the original declared reference ordering; no summary-table values enter the calculations.

At 40% added sequence, completeness MAE changes as follows (pp):

| Tool | Control | 40% added | Paired increase |
|---|---:|---:|---:|
| MAGICC | 4.397746 | 15.985403 | 11.587657 |
| CheckM2 | 5.107523 | 5.519696 | 0.412173 |
| CoCoPyE | 4.232653 | 4.889440 | 0.656787 |
| DeepCheck | 5.907160 | 6.106747 | 0.199587 |

MAGICC's paired increase is 11.587657 pp [8.788160, 14.099986]; its signed completeness bias is +11.287167 pp. This is a real completeness limitation under the defined intervention. On **contamination**, all tools degrade under the primary truth convention: endpoint MAEs are 19.329862 (MAGICC), 22.826661 (CheckM2), 19.135347 (CoCoPyE), and 27.160980 (DeepCheck). Statements that only MAGICC degrades must name completeness. Retention of all original contigs proves that the selected unique-k-mer set is unchanged, but **relative k-mer composition is not fixed** by localized duplication. Do not equate this experiment with the separate uniform count-scaling intervention or claim that it isolates a unique internal model mechanism.

The reported 2.87% boundary is interpolated between the 0% control and first 5% dose, using an upper-CI crossing rule. It is a descriptive benchmark summary, not a directly measured operating cutoff or an empirically validated user recommendation. No evidence in this experiment supports a claim that the tested scenario is rare in nature.

## Figure 3f and balanced comparison

From raw five-set prediction/truth joins, MAGICC has the highest point balanced accuracy at the 5% contamination threshold on C-clean (0.768825), D-clean (0.921556) and E (0.972785). CoCoPyE is higher on B (0.991875 versus MAGICC 0.985000); A has no contaminated genomes and hence undefined balanced accuracy. These are descriptive point rankings, not a claim of significant superiority in every pairwise comparison.

C-clean exposes a useful tradeoff. MAGICC false-fails 23/52 truly clean genomes and false-passes 19/948 truly contaminated genomes. CheckM2 gives 0/52 and 493/948, CoCoPyE 37/52 and 42/948, and DeepCheck 0/52 and 663/948. Leading with MAGICC's strong joint discrimination while immediately reporting its clean-genome false-fail cost is fairer than describing this panel only as a weakness. It does not support calling MAGICC universally best at all thresholds or objectives.

## Presentation decision

The main narrative briefly reports the verified completeness effect and directs readers to Supplementary Figure S11, which retains both axes for all tools and all doses. Reorganized main Figure 5 presents empirical assemblies, sequence-error dose-responses, matched runtime and both CAMI II mixed-bin datasets. Supplementary Figure S17 retains catalogue disagreements, model interventions and the independently ground-truthed reduced-genome error anchor. This change follows the distinction between primary application comparisons and diagnostic stress tests. All material findings, raw predictions, alternative accounting, denominators and uncertainty remain accessible.

## Final selected-stage comparator reconciliation

The historical numerical tables above are retained as a reproduction record,
not the final CoCoPyE comparison. Script259 independently recalculates the
selected-stage values in `results/revision/cocopye_stage_resubmission5/independent_figures/`.
At 40% added/original bp, completeness MAEs are 15.985403 (MAGICC),5.519696
(CheckM2),4.890873 (CoCoPyE),6.106747 (DeepCheck); contamination MAEs are
19.329862,22.826661,32.568112,27.160980 respectively. MAGICC's verified
completeness degradation remains a limitation, while it has the lowest
contamination MAE point estimate at that dose; all four contamination errors
increase relative to control. No empirical artifact-incidence claim follows.

At the 5% contamination threshold on C-clean, final CoCoPyE false-fail and
false-pass counts are 35/52 and122/948; MAGICC remains 23/52 and19/948.
MAGICC retains the highest balanced-accuracy point estimate on C-clean,
D-clean and E, while CoCoPyE leads SetB; SetA balanced accuracy is undefined
because one truth class is absent. These point rankings are descriptive.
The final all-three-test agreement count for mean-MAE comparisons is 22/4/4;
the distinct BH rank-test component partition is 24/4/2.
