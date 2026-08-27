# WS1.11 — Circularity safeguard (Reviewer 1, minor comment 13)

_Generated 2026-08-02T15:34:03.490532+00:00 by `scripts/193_ws1_11_report.py` from the result files in `results/revision/circularity/`. Evaluation only: no model was retrained and `models/magicc_v5.onnx` (SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`) is unchanged._

## Verdict

**NO MATERIAL BIAS.** **Circularity does not materially bias the published conclusions.** The upper 95 % confidence bound on MAGICC's excess error over the genomes the CheckM2 filter removed is 1.39 pp, below the 2.0 pp materiality threshold declared before the analysis.

Per metric, paired over the matched reference pairs (D > 0 = worse on the genomes the CheckM2-based curation removed):

- completeness: D = 0.77 pp [0.30, 1.28], q = 0.012
- contamination: D = 0.82 pp [0.32, 1.39], q = 0.00933

## 1. The objection, stated precisely

Every reference genome in this project passed a CheckM2-based filter: GTDB metadata restricted to CheckM2 completeness ≥ 98 %, contamination ≤ 2 %, < 100 contigs, N50 > 20 kbp and longest contig > 100 kbp → 277,183 genomes → 100,000 sampled → train/val/test. Ground truth for **both** training and benchmarking is therefore drawn from genomes CheckM2 judged near-complete and clean. If CheckM2 systematically mis-scores a class of genome, those genomes were removed from the pool, so MAGICC learned a CheckM2-curated view of genome space **and the benchmark cannot reveal the error, because the affected genomes are absent**. A sensitivity analysis that re-used the same filtered pool would not answer this.

## 2. How the reference genomes were selected — CheckM2 removed from the loop

References were selected by NCBI's **`assembly_level == "Complete Genome"`** annotation taken from `assembly_summary_genbank.txt` / `assembly_summary_refseq.txt` (`version_status == latest`) — a submitter/NCBI assembly annotation that does not involve CheckM2 — with **no CheckM2 filter applied**. CheckM2 scores were read afterwards only to *label* each reference as would-have-passed or would-have-failed.

| stage | n remaining | note |
|---|---|---|
| GTDB genomes (bac120 + ar53) | 732,475 |  |
| NCBI assembly_level == "Complete Genome" (latest) | 47,820 | selector = NCBI/submitter annotation; NO CheckM2 filter applied |
| has an NCBI ftp_path (downloadable) | 47,820 |  |
| taxonomic-consistency filter (GTDB<->NCBI genus + domain agree) | 39,383 | dropped 0 no/placeholder GTDB assignment, 6623 no NCBI assignment, 1814 genus mismatch (of 47820) |
| excluded train / val / 9-mer-selection genomes | 28,054 | references are absent from training entirely |
| FINAL selected references (both arms) | 400 | 200 would-FAIL + 200 matched would-PASS |

**Taxonomic-consistency filter** (protocol WS1.11): a candidate must carry both a GTDB and an NCBI taxonomy and they must agree at genus level after normalising GTDB's alphabetic suffixes. Of 47,820 NCBI-Complete GTDB genomes, 6,623 were dropped for a missing/placeholder NCBI assignment and 1,814 for a genus mismatch; 0 lacked a GTDB assignment.

**The headline design number.** Of the 28,054 eligible NCBI-Complete references remaining after taxonomic consistency and leakage exclusion, **2,449 (8.7 %) would have FAILED the original CheckM2-based curation filter** and 25,605 would have passed. Essentially all failures are CheckM2 failures, not assembly-quality failures: 1,234 fall below 98 % CheckM2 completeness and 1,709 exceed 2 % CheckM2 contamination (criteria are not mutually exclusive), while only 6 have ≥ 100 contigs, 0 have N50 ≤ 20 kbp and 0 have a longest contig ≤ 100 kbp.

**Set H composition.** 200 would-have-failed references were drawn with √-proportional allocation across three CheckM2-deficit severity strata (the same scheme Phase 1 used across phyla, so the severe stratum is represented): mild 87, moderate 67, severe 46 from strata of size mild 1,298, moderate 775, severe 376. Each was matched 1:1 to a would-have-passed reference at the deepest available taxonomic rank and then by closest genome size: species 127, genus 50, family 13, class 4, order 4, phylum 2. The arms are balanced — 13 vs 13 phyla, median genome size 3.8226 vs 3.8174 Mbp, median contig count 2 vs 2 — and differ in CheckM2 score by construction (median completeness 96.65 vs 100.00 %, median contamination 2.69 vs 0.39 %).

**Stated boundary.** Candidates must be present in GTDB, because the taxonomic-consistency filter needs a GTDB taxonomy; 95,272 NCBI prokaryotic Complete Genomes are outside GTDB and are therefore unreachable here. GTDB's own floor (≈ ≥ 50 % completeness, ≤ 10 % contamination) is far weaker than the project's ≥ 98 % / ≤ 2 % filter, so the genomes that matter for this objection — those excluded by the project's filter but not by GTDB's — are exactly the ones this set contains.

## 3. Benchmark generation — only the reference selection changed

`set_H_ncbi` was produced by the generation logic of `scripts/73_generate_clean_cd_benchmarks.py` **verbatim** (itself `scripts/25_benchmark_generate.py` verbatim): same fragmentation call, same cross-phylum contaminant draw from `data/splits/test_genomes.tsv`, same caps, same label arithmetic, same constraint guard, same FASTA writer, 10 independent simulations per reference, targets completeness ~ U[50, 100) % and contamination ~ U[0, 100) %. Design RNG `default_rng(7_500_000)`; per-sample RNG `default_rng(7_500_000 + 1000·ref_index + replicate)`. One deliberate refinement: both members of a matched pair receive the *same* target draw at each replicate, which makes every comparison paired at the sample level without changing the marginal target distributions.

- 4,000 simulations from 400 references (200 matched pairs), exactly 10 simulations each; 0 generation errors.
- FASTA integrity: 0 missing, 0 empty, 0 unparseable, 0 with a contig count or bp total disagreeing with the metadata.
- Constraint violations: 0 (contaminant bp > dominant bp), 0 (contamination % > completeness %), 0 (contamination > 100 %), 0 (completeness outside 50–100 %). **Out-of-domain samples (protocol §4.4a): 0** — the design is entirely in-domain.
- Target uniformity: KS p = 0.4999 (completeness), 0.0511 (contamination). Paired targets identical across arms: True.
- `generation_metadata.tsv` has 53 columns including per-sample seed, both targets, both observed values, every contaminant accession, the fragmentation tier and all six dropout parameters (WS7.7 / R1-m15).
- Determinism: 5 FASTAs deleted with their checkpoint lines removed and regenerated — byte-identical: **True**; `generation_metadata.tsv` unchanged: True.
- All 400/400 reference assemblies downloaded from NCBI and accepted; total bp deviates from the GTDB-recorded genome size by at most 0.0000 %, i.e. the assemblies scored here are exactly the assemblies GTDB scored.

## 4. Provenance and leakage — proven, not asserted

`scripts/188_ws1_11_provenance_audit.py` imports the normalisation and GCA↔GCF cross-map code of `scripts/74_provenance_audit.py` directly, so the two audits cannot drift apart. Cross-map: 277,183 rows → 503,511 accession strings → 277,183 canonical assemblies, 0 inconsistencies. Counts below are identical under all three normalisations.

**DISJOINTNESS VERDICT: PASS.**

| universe | samples (n=4000) | unique references (n=400) |
|---|---|---|
| training split | 0 | 0 |
| validation split | 0 | 0 |
| test split | 350 | 35 |
| 2,000-genome 9-mer feature-selection set | 0 | 0 |
| 277,183-genome CheckM2-filtered curation pool | 2000 | 200 |

The last row is the point of the experiment: the H_fail arm has **0 of 2000** samples inside the CheckM2-filtered pool, i.e. none of those 200 references was ever available to any MAGICC model or any previous MAGICC benchmark. 12,075 contamination events over 6,853 unique genomes all come from the held-out test split (0 from train, 0 from val), and 0 share the dominant's phylum — identical to `set_C_clean`/`set_D_clean`, because the contaminant pool is deliberately unchanged. A SHA256 manifest covers 4,413 files.

## 5. The comparison that answers R1-m13

### 5.1 Error on CheckM2-filtered vs NCBI-selected references

| tool | reference group | n | clusters | completeness MAE (95 % CI) | comp. bias | comp. R² | contamination MAE (95 % CI) | cont. bias | cont. R² |
|---|---|---|---|---|---|---|---|---|---|
| magicc_v5 | would have PASSED the CheckM2 filter | 2000 | 200 | 5.54 (5.11–5.95) | 3.26 | 0.685 | 5.63 (5.23–6.03) | -1.80 | 0.889 |
| magicc_v5 | **would have FAILED it** | 2000 | 200 | 6.31 (5.76–6.87) | 3.24 | 0.591 | 6.45 (5.89–7.07) | -1.41 | 0.849 |
| checkm2 | would have PASSED the CheckM2 filter | 2000 | 200 | 13.89 (13.11–14.65) | 12.81 | -0.516 | 23.14 (21.82–24.51) | -20.62 | -0.464 |
| checkm2 | **would have FAILED it** | 2000 | 200 | 13.94 (13.22–14.70) | 10.63 | -0.507 | 23.35 (21.98–24.71) | -20.47 | -0.445 |
| cocopye | would have PASSED the CheckM2 filter | 2000 | 200 | 5.16 (4.84–5.49) | 2.10 | 0.748 | 22.21 (21.40–23.02) | -21.25 | -0.345 |
| cocopye | **would have FAILED it** | 2000 | 200 | 5.81 (5.47–6.17) | 1.79 | 0.685 | 22.52 (21.75–23.35) | -19.40 | -0.335 |
| deepcheck | would have PASSED the CheckM2 filter | 2000 | 200 | 16.25 (15.21–17.33) | 15.16 | -1.119 | 31.66 (30.54–32.77) | -25.52 | -1.554 |
| deepcheck | **would have FAILED it** | 2000 | 200 | 15.41 (14.37–16.50) | 11.87 | -0.960 | 32.14 (30.98–33.26) | -24.02 | -1.774 |

R² is the coefficient of determination throughout (never squared Pearson). CIs are 95 % cluster bootstrap over reference genomes, 2,000 replicates, seeds from `fw.stable_hash` (CRC-32) with `PYTHONHASHSEED=0`. Denominators: completeness % = retained dominant bp / full reference length × 100; contamination % = total contaminant bp / full reference length × 100.

### 5.2 Primary estimator — paired difference over matched reference pairs

Each would-have-failed reference is compared with its taxonomically matched, size-matched control on simulations that share identical target draws, so composition cannot confound the contrast. `D = MAE(H_fail) − MAE(H_pass)`; positive = worse on the removed genomes. CIs and two-sided p-values are cluster bootstraps over the 200 pairs; q is Benjamini–Hochberg over the whole test family (56 tests).

| tool | metric | mean abs. err. H_pass | H_fail | D (95 % CI) | Hodges–Lehmann | rank-biserial | p | q (BH) |
|---|---|---|---|---|---|---|---|---|
| magicc_v5 | completeness | 5.54 | 6.31 | **0.77** (0.30, 1.28) | 0.56 | 0.110 | 0.003 | 0.012 |
| magicc_v5 | contamination | 5.63 | 6.45 | **0.82** (0.32, 1.39) | 0.46 | 0.106 | 0.002 | 0.00933 |
| checkm2 | completeness | 13.89 | 13.94 | **0.06** (-0.47, 0.60) | 0.01 | 0.002 | 0.822 | 0.902 |
| checkm2 | contamination | 23.14 | 23.35 | **0.22** (-0.45, 0.90) | 0.07 | 0.008 | 0.514 | 0.669 |
| cocopye | completeness | 5.16 | 5.81 | **0.65** (0.35, 0.98) | 0.49 | 0.117 | 0.001 | 0.00509 |
| cocopye | contamination | 22.21 | 22.52 | **0.32** (-0.12, 0.71) | 0.21 | 0.044 | 0.144 | 0.26 |
| deepcheck | completeness | 16.25 | 15.41 | **-0.84** (-1.44, -0.23) | -0.82 | -0.088 | 0.007 | 0.0261 |
| deepcheck | contamination | 31.66 | 32.14 | **0.48** (-0.51, 1.45) | -0.40 | -0.045 | 0.354 | 0.494 |

Restricting to the pairs matched at genus level or better (residual-confounding control):

| tool | metric | n pairs | D (95 % CI) | q (BH) |
|---|---|---|---|---|
| magicc_v5 | completeness | 177 | 0.61 (0.12, 1.12) | 0.0324 |
| magicc_v5 | contamination | 177 | 0.61 (0.14, 1.14) | 0.0324 |
| checkm2 | completeness | 177 | -0.05 (-0.61, 0.52) | 0.949 |
| checkm2 | contamination | 177 | 0.23 (-0.50, 0.97) | 0.669 |
| cocopye | completeness | 177 | 0.70 (0.43, 1.02) | 0.00509 |
| cocopye | contamination | 177 | 0.30 (-0.14, 0.73) | 0.307 |
| deepcheck | completeness | 177 | -0.96 (-1.58, -0.31) | 0.012 |
| deepcheck | contamination | 177 | 0.32 (-0.68, 1.36) | 0.687 |

### 5.3 Difference-in-differences against the tool that defined the filter

Both tools score the identical samples, so `DiD = D(MAGICC) − D(other)` is immune to any property of the removed genomes that makes them intrinsically harder for every method. DiD ≈ 0 means MAGICC is no more affected by the curation boundary than the comparator.

| comparison | metric | D(MAGICC) | D(other) | DiD (95 % CI) | q (BH) |
|---|---|---|---|---|---|
| MAGICC V5 vs checkm2 | completeness | 0.77 | 0.06 | **0.71** (0.08, 1.37) | 0.0576 |
| MAGICC V5 vs checkm2 | contamination | 0.82 | 0.22 | **0.60** (-0.21, 1.47) | 0.247 |
| MAGICC V5 vs cocopye | completeness | 0.77 | 0.65 | **0.12** (-0.39, 0.66) | 0.646 |
| MAGICC V5 vs cocopye | contamination | 0.82 | 0.32 | **0.51** (-0.12, 1.17) | 0.216 |
| MAGICC V5 vs deepcheck | completeness | 0.77 | -0.84 | **1.61** (0.95, 2.29) | 0.004 |
| MAGICC V5 vs deepcheck | contamination | 0.82 | 0.48 | **0.34** (-0.80, 1.43) | 0.635 |

### 5.4 CheckM2's own error on the subgroup its scores excluded

On the 2000 simulations built from references the CheckM2-based curation would have rejected, CheckM2 itself scores completeness MAE 13.94 pp (13.22–14.70) with bias 10.63 pp, and contamination MAE 23.35 pp (21.98–24.71) with bias -20.47 pp, against 13.89 / 23.14 pp on the matched controls.

## 6. Severity gradient inside the removed subgroup

| tool | stratum | n | completeness MAE (95 % CI) | contamination MAE (95 % CI) |
|---|---|---|---|---|
| magicc_v5 | matched controls (would pass) | 2000 | 5.54 (5.11–5.95) | 5.63 (5.23–6.03) |
| magicc_v5 | mild deficit | 870 | 6.54 (5.77–7.35) | 6.29 (5.58–7.19) |
| magicc_v5 | moderate deficit | 670 | 6.58 (5.60–7.71) | 6.72 (5.74–7.91) |
| magicc_v5 | severe deficit | 460 | 5.46 (4.40–6.66) | 6.34 (5.15–7.82) |
| checkm2 | matched controls (would pass) | 2000 | 13.89 (13.11–14.65) | 23.14 (21.82–24.51) |
| checkm2 | mild deficit | 870 | 15.16 (13.95–16.34) | 22.90 (20.92–24.95) |
| checkm2 | moderate deficit | 670 | 13.53 (12.22–14.81) | 22.77 (20.44–25.11) |
| checkm2 | severe deficit | 460 | 12.25 (11.04–13.50) | 25.05 (22.62–27.54) |
| cocopye | matched controls (would pass) | 2000 | 5.16 (4.84–5.49) | 22.21 (21.40–23.02) |
| cocopye | mild deficit | 870 | 6.04 (5.50–6.64) | 23.06 (21.77–24.38) |
| cocopye | moderate deficit | 670 | 5.74 (5.08–6.44) | 21.68 (20.50–22.79) |
| cocopye | severe deficit | 460 | 5.48 (5.01–5.97) | 22.73 (20.99–24.63) |
| deepcheck | matched controls (would pass) | 2000 | 16.25 (15.21–17.33) | 31.66 (30.54–32.77) |
| deepcheck | mild deficit | 870 | 17.15 (15.49–18.86) | 32.21 (30.47–34.02) |
| deepcheck | moderate deficit | 670 | 15.17 (13.32–17.10) | 32.84 (30.99–34.63) |
| deepcheck | severe deficit | 460 | 12.47 (11.23–13.82) | 30.99 (28.68–33.27) |

Strata: *mild* = CheckM2 completeness ≥ 96 % and contamination ≤ 4 % but outside the ≥ 98 % / ≤ 2 % filter; *moderate* = completeness ≥ 90 % or contamination ≤ 10 %; *severe* = completeness < 90 % or contamination > 10 %.

## 7. MIMAG-inspired threshold behaviour

MIMAG-inspired (completeness/contamination only; rRNA and tRNA criteria are not evaluable from these estimates): high ≥ 90 % completeness AND < 5 % contamination; medium ≥ 50 % AND < 10 %.

| tool | arm | n | macro F1 (95 % CI) | Cohen κ | true high / medium / low |
|---|---|---|---|---|---|
| magicc_v5 | H_pass | 2000 | 0.788 (0.716–0.849) | 0.821 | 17 / 176 / 1807 |
| magicc_v5 | H_fail | 2000 | 0.803 (0.726–0.869) | 0.788 | 18 / 183 / 1799 |
| checkm2 | H_pass | 2000 | 0.557 (0.504–0.612) | 0.353 | 17 / 176 / 1807 |
| checkm2 | H_fail | 2000 | 0.568 (0.514–0.623) | 0.383 | 18 / 183 / 1799 |
| cocopye | H_pass | 2000 | 0.812 (0.743–0.870) | 0.769 | 17 / 176 / 1807 |
| cocopye | H_fail | 2000 | 0.672 (0.571–0.756) | 0.538 | 18 / 183 / 1799 |
| deepcheck | H_pass | 2000 | 0.410 (0.372–0.452) | 0.183 | 17 / 176 / 1807 |
| deepcheck | H_fail | 2000 | 0.425 (0.386–0.468) | 0.196 | 18 / 183 / 1799 |

At the 5 % contamination boundary (false-fail denominator = truly clean simulations only; false-pass denominator = truly contaminated simulations only):

| tool | arm | truly clean n | truly contaminated n | false-fail rate (95 % CI) | false-pass rate (95 % CI) | balanced accuracy |
|---|---|---|---|---|---|---|
| magicc_v5 | H_pass | 96 | 1904 | 0.125 (0.058–0.202) | 0.012 (0.007–0.017) | 0.932 |
| magicc_v5 | H_fail | 98 | 1902 | 0.133 (0.061–0.211) | 0.013 (0.007–0.018) | 0.927 |
| checkm2 | H_pass | 96 | 1904 | 0.094 (0.042–0.153) | 0.122 (0.102–0.145) | 0.892 |
| checkm2 | H_fail | 98 | 1902 | 0.265 (0.174–0.366) | 0.098 (0.077–0.121) | 0.819 |
| cocopye | H_pass | 96 | 1904 | 0.219 (0.139–0.306) | 0.014 (0.008–0.021) | 0.883 |
| cocopye | H_fail | 98 | 1902 | 0.582 (0.477–0.684) | 0.007 (0.004–0.012) | 0.706 |
| deepcheck | H_pass | 96 | 1904 | 0.000 (0.000–0.000) | 0.235 (0.206–0.264) | 0.883 |
| deepcheck | H_fail | 98 | 1902 | 0.153 (0.078–0.238) | 0.199 (0.170–0.227) | 0.824 |

## 8. Why the removed genomes are slightly harder — orthogonal evidence

The benchmark defines completeness relative to the **deposited** reference assembly and treats that assembly as complete and clean. For the removed arm that is exactly the assumption in question: if those references really are imperfect, a well calibrated estimator reads low (or high) and is scored as biased by a truth that says otherwise. Every tool was therefore also run on the 400 **unmodified** deposited assemblies, with no simulation at all.

| estimate | mean, would-pass refs | mean, would-FAIL refs | difference (95 % CI) | uses CheckM2? |
|---|---|---|---|---|
| gtdb_checkm2_completeness | 99.87 | 93.96 | -5.91 (-7.08, -4.80) | yes — circular |
| gtdb_checkm2_contamination | 0.59 | 3.04 | 2.45 (2.16, 2.78) | yes — circular |
| magicc_v5_completeness | 99.09 | 97.97 | -1.12 (-1.77, -0.53) | no — independent |
| magicc_v5_contamination | 1.50 | 3.10 | 1.59 (0.47, 3.04) | no — independent |
| checkm2_local_completeness | 99.87 | 94.02 | -5.85 (-7.09, -4.73) | yes — circular |
| checkm2_local_contamination | 0.61 | 3.04 | 2.44 (2.14, 2.76) | yes — circular |
| cocopye_completeness | 99.30 | 98.54 | -0.76 (-1.12, -0.45) | no — independent |
| cocopye_contamination | 1.41 | 7.75 | 6.34 (5.17, 7.61) | no — independent |

MAGICC — which never sees a CheckM2 score — independently reproduces the *direction* of the deficit on the removed references (-1.12 pp completeness, 95 % CI -1.77 to -0.53) but at a much smaller magnitude than CheckM2 claims (-5.91 pp). So these assemblies are genuinely, mildly imperfect — the exclusion was not pure artefact — while the size of the CheckM2 deficit that triggered exclusion is not corroborated. Per integrity rule 3, disagreement is reported as disagreement: nothing here establishes which estimate is correct.

Reproducibility check: local CheckM2 1.0.1 re-run on the same 400 assemblies reproduces the GTDB-recorded CheckM2 completeness to a mean absolute difference of 0.23 pp (median 0.00 pp, r = 0.995); 9 of 200 would-have-failed references would now pass the filter on the local re-run.

Per-reference regression of the tool's mean signed error on the reference's CheckM2 deficit (`ws1_11_reference_incompleteness_mechanism.tsv`):

| tool | signed error | vs | n refs | Spearman ρ | OLS slope ± SE | note |
|---|---|---|---|---|---|---|
| magicc_v5 | comp_bias | checkm2_completeness_deficit | 400 | -0.216 | -0.133 ± 0.040 | independent: this tool never sees the CheckM2 score |
| magicc_v5 | cont_bias | checkm2_contamination_excess | 400 | 0.105 | 1.134 ± 0.184 | independent: this tool never sees the CheckM2 score |
| checkm2 | comp_bias | checkm2_completeness_deficit | 400 | -0.358 | -0.481 ± 0.063 | CIRCULAR: the x variable IS this tool's own estimate |
| checkm2 | cont_bias | checkm2_contamination_excess | 400 | 0.111 | 0.987 ± 0.440 | CIRCULAR: the x variable IS this tool's own estimate |
| cocopye | comp_bias | checkm2_completeness_deficit | 400 | -0.184 | -0.052 ± 0.028 | independent: this tool never sees the CheckM2 score |
| cocopye | cont_bias | checkm2_contamination_excess | 400 | 0.150 | 0.848 ± 0.215 | independent: this tool never sees the CheckM2 score |
| deepcheck | comp_bias | checkm2_completeness_deficit | 400 | -0.412 | -0.656 ± 0.074 | independent: this tool never sees the CheckM2 score |
| deepcheck | cont_bias | checkm2_contamination_excess | 400 | 0.073 | 0.958 ± 0.570 | independent: this tool never sees the CheckM2 score |

Stratified within matched true-contamination bands, which removes any residual difference in the realised label distributions, MAGICC's signed contamination error separates the arms exactly where a genuinely contaminated reference would show up — the low-contamination bands:

| true contamination band | n (pass / FAIL) | MAGICC bias, would-pass refs | MAGICC bias, would-FAIL refs |
|---|---|---|---|
| [0,5) | 96 / 98 | 0.10 | 1.12 |
| [5,10) | 97 / 103 | 0.18 | 0.94 |
| [10,20) | 221 / 210 | 0.11 | 1.30 |
| [20,40) | 399 / 401 | 0.47 | 0.82 |
| [40,70) | 821 / 834 | -1.72 | -1.46 |
| [70,100) | 366 / 354 | -6.61 | -6.79 |

Sensitivity analysis with the truth corrected for the reference genome's own imperfection (first-order model in `scripts/191_ws1_11_analysis.py::add_adjusted_truth`; it uses CheckM2's own reference scores, so it structurally favours CheckM2 and is a sensitivity analysis only):

| tool | arm | completeness MAE raw → adjusted | contamination MAE raw → adjusted |
|---|---|---|---|
| magicc_v5 | H_pass | 5.54 → 5.56 | 5.63 → 5.77 |
| magicc_v5 | H_fail | 6.31 → 9.41 | 6.45 → 6.80 |
| checkm2 | H_pass | 13.89 → 13.88 | 23.14 → 23.56 |
| checkm2 | H_fail | 13.94 → 16.28 | 23.35 → 23.54 |
| cocopye | H_pass | 5.16 → 5.16 | 22.21 → 22.73 |
| cocopye | H_fail | 5.81 → 8.66 | 22.52 → 22.75 |
| deepcheck | H_pass | 16.25 → 16.27 | 31.66 → 32.16 |
| deepcheck | H_fail | 15.41 → 17.52 | 32.14 → 32.73 |

## 9. Context — MAGICC V5 across reference-selection regimes

Reported for scale only; the sets differ in taxonomic composition, so the matched within-`set_H_ncbi` contrast above is the primary comparison.

| benchmark set / group | n | clusters | completeness MAE (95 % CI) | contamination MAE (95 % CI) |
|---|---|---|---|---|
| set_A_v2 | 1000 | 798 | 2.18 (1.94–2.45) | 0.83 (0.64–1.04) |
| set_B_v2 | 1000 | 803 | 2.97 (2.64–3.32) | 4.45 (4.14–4.80) |
| set_C_clean | 1000 | 100 | 6.82 (5.80–7.93) | 9.08 (8.16–10.02) |
| set_D_clean | 1000 | 100 | 5.63 (4.99–6.32) | 6.52 (5.97–7.09) |
| set_E | 1000 | 785 | 5.49 (5.06–5.94) | 5.66 (5.16–6.16) |
| set_H_ncbi, would-have-passed references | 2000 | 200 | 5.54 (5.11–5.95) | 5.63 (5.23–6.03) |
| **set_H_ncbi, would-have-FAILED references** | 2000 | 200 | 6.31 (5.76–6.87) | 6.45 (5.89–7.07) |

## 10. What this experiment does and does not establish

**Establishes.** Reference genomes chosen by an annotation that has nothing to do with CheckM2, deliberately including genomes the project's CheckM2 filter would have rejected, are strictly outside the training data and outside the curated pool (audited, not asserted). On those genomes MAGICC's error changes by the amount quantified in §5.2, measured against taxonomically and size-matched controls on identically parameterised simulations.

**Does not establish.** (i) Nothing here adjudicates whether CheckM2's scores for the excluded references are right — the tools disagree, and disagreement is not correctness. (ii) NCBI-Complete assemblies absent from GTDB cannot be reached, because the taxonomic-consistency filter needs a GTDB taxonomy. (iii) This is an evaluation, not a retrain: it measures whether the *benchmark* was blind to a class of genome, not what a model trained on an uncurated pool would do. A retrain was excluded by author decision because it would also invalidate the V5-anchored WS1.6 / WS1.9 holdout experiments. (iv) GUNC was not run: it yields CSS and a pass/fail flag rather than completeness/contamination percentages, so it cannot contribute to an error comparison.

## 11. Files

| path | content |
|---|---|
| `data/benchmarks/set_H_ncbi/` | benchmark set: 4,000 simulated FASTAs, 400 reference assemblies, metadata, labels, generation metadata (53 columns), checkpoint, per-tool predictions |
| `data/benchmarks/set_H_ncbi/reference_selection_final.tsv` | the 400 references with arm, pair, match level, CheckM2 scores, taxonomy |
| `data/benchmarks/set_H_ncbi/candidate_pool.tsv.gz` | every eligible NCBI-Complete candidate with its would-pass/would-fail label |
| `results/revision/circularity/provenance/` | overlap summary, dominant list, contaminant list, SHA256 manifest, README |
| `results/revision/circularity/ws1_11_arm_metrics.tsv` | MAE / bias / RMSE / R² per tool per arm and per severity stratum, with CIs |
| `results/revision/circularity/ws1_11_paired_arm_tests.tsv` | primary estimator: paired D with CI, effect sizes, p and BH q |
| `results/revision/circularity/ws1_11_did_vs_competitors.tsv` | difference-in-differences against each competitor |
| `results/revision/circularity/ws1_11_mimag.tsv, ws1_11_thresholds.tsv` | MIMAG-inspired classification and 5 %/10 %/50 %/90 % decision thresholds |
| `results/revision/circularity/ws1_11_reference_level_scores.tsv` | tool scores on the 400 unmodified deposited assemblies (orthogonal evidence) |
| `results/revision/circularity/ws1_11_reference_incompleteness_mechanism.tsv` | per-reference signed error regressed on the CheckM2 deficit |
| `results/revision/circularity/ws1_11_adjusted_truth_sensitivity.tsv` | MAE against a truth corrected for the reference genome itself |
| `results/revision/circularity/ws1_11_context_other_sets.tsv` | MAGICC V5 on the CheckM2-filtered benchmark sets, for scale |
| `results/revision/circularity/figures/` | 5 CVD-safe figures with captions in ws1_11_figure_captions.md |
| `scripts/185–193` | selection, fetch, generation, audit, inference, competitor runs, analysis, reference-level scores, this report |
