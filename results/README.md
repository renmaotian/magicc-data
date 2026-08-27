# Result files

Every quantitative result reported in the manuscript, as the file that produced
it. 1,169 files, ~113 MB.

## Path convention

The reports and JSON records in this tree were written inside the analysis
workspace and refer to paths of the form `results/revision/<x>`. In this
repository that material is at **`results/<x>`**, with three deliberate moves:

| Workspace path | Here |
|---|---|
| `results/revision/holdout/` | `results/holdout_phylum/` (renamed for clarity — it is the leave-**phylum**-out evaluation) |
| `results/revision/reproducibility/release/` | `released_artefacts/` (top level; this is the "released artefact bundle" of the Code availability statement) |
| `results/revision/cami2/{truth,provenance}/` | `cami2/{truth,provenance}/` (top level, with the other CAMI II identifiers) |

Scripts are at `scripts/`; the working-tree-to-deposited name mapping is
`scripts/SCRIPT_MAPPING.tsv`.

## Inventory

| Directory | Contents | Manuscript items |
|---|---|---|
| `metrics/` | WS5 definitive metrics on the five leakage-free sets: MAE/bias/R² with cluster-bootstrap CIs, MIMAG-inspired classification and confusion matrices, signed errors per genome and by stratum, threshold analysis, clustered paired tests (Wilcoxon + BH + Hodges–Lehmann + Cliff's δ), the rebuilt Table S2, domain-restriction analysis, and the distribution plots | Table 1, Table S2, Fig. 3, Figs. S1–S6 |
| `benchmark/` | the revised benchmark table, the clean C/D metrics and the leakage-specificity control | Table 1, Table S1 |
| `holdout_phylum/` | leave-phylum-out retraining and evaluation: per-sample predictions, per-reference errors, difference-in-differences, sub-phylum breakdown, MIMAG agreement, the k-mer re-selection control (both prevalence tables under the holdout selection) | Fig. 4a–c, Table S9 |
| `holdout_family/` | leave-family-out retraining and evaluation, plus `dryrun_v5_vs_v5/` (the V5-versus-V5 null control) and the family-versus-phylum comparison | Fig. 4, Fig. S16, Table S10 |
| `holdout_genus/` | leave-genus-out panel definition and EDA — **the retraining and evaluation were not complete at deposition; see `holdout_genus/STATUS.md`** | — |
| `set_F/` | contamination type × donor taxonomic distance: per-genome long table, cell and marginal tables, paired comparisons, the RBH distance characterisation, marker-duplication validation | Fig. 5, Table S11 |
| `set_G/` | sequencing and assembly error robustness: degradation curves, paired degradation, MIMAG agreement, k-mer perturbation mechanism, crossover and boundary tables | Fig. S12, Table S13 |
| `circularity/` | circularity safeguard on 4,000 assemblies from 400 NCBI references in 200 matched pairs: per-reference errors, paired arm tests, threshold false-fail, reference-incompleteness mechanism | Fig. S15, Table S14 |
| `gunc/` | the **8,000-row** per-genome GUNC output (`gunc_per_genome.tsv`, 8 set × database combinations × 1,000 genomes: sets A and B against proGenomes 2.1, sets C-clean, D-clean and E against both proGenomes 2.1 and GTDB r95), power audit, stratified pass/fail, threshold agreement, CSS correlations, proGenomes-2.1-versus-GTDB-r95 sensitivity, and the positive/negative control verification | Table S8, Fig. S11 |
| `cami2/` | CAMI II analysis: per-bin predictions from all four tools, accuracy by cohort, by distance, by completeness decile, paired tests, detection slopes, censoring, the Set F comparison and the figures. Truth tables and identifiers are at `cami2/` (top level) | Fig. 5d–f, Table S12 |
| `real_data/` | Track A real-data validation (Meslier mock communities, ZymoBIOMICS, NCBI strain-matched pairs, NCBI contaminated-flag cohort) and the reduced-genome workstream: cohorts and fetch manifests for SPIRE v1 / GTDB r220 / UHGG v2.0.2, the mechanism models, the ground-truthed `set_C_clean` cross-check, and `reduced_genome/mitigation/` (the size-channel intervention, size-conditioned recalibration, the generalisation ceiling and `BOUNDARY_STATEMENT.md`) | Fig. 1, Fig. S13, Fig. S14, Tables S6, S7, S16 |
| `contamination_evidence/` | the evidence behind the withdrawn real-MAG contamination claims: Kraken2 strict metric under three databases, informative-k-mer density controls, single-copy-gene duplication, matched-novelty cohorts, and the correlation table | Discussion (withdrawal), Table S17 |
| `speed/` | the matched-hardware timing campaign: the 92 per-run records of the campaign (wall clock, peak RSS, stdout and parsed timings, 379 files), thread-scaling summary, cold-versus-warm forensics, setup cost, environment footprint, total-cost table and the rewritten Table S4 | Table 1, Table S4 |
| `speed_v3/` | the follow-up campaign separating code-version from model-version effects (V3 code versus V5 code), 30 per-run records, pooled cell summary, updated ratios and the attribution note | Table S4d |
| `ws11/` | the taxonomic-novelty ladder (genome/species/genus/family/phylum novelty of the dominant, evaluation only) and the paired macro-F1 differences with the power statement | Fig. S17, Table S18 |
| `reproducibility/` | determinism record, and the executed one-command reproduction of the headline benchmark (per-set predictions, metrics and logs) | Methods, Code availability |
| `release_verification/` | evidence that the PyPI and GitHub artefacts are the version the manuscript names, and the recorded `magicc-data` state before this deposition | Code availability |
| top-level files | the frozen model card, the legacy-prediction audit that found the model-version defect, and the WS1.5 clean-C/D metrics | Methods, Table S1 |

## What is deliberately not deposited

Excluded material is intermediate: it is regenerated by the deposited scripts
from the deposited inputs, and none of it is a reported number.

| Excluded | Size | Why |
|---|---|---|
| `results/revision/gunc/runs/` | 11 GB | raw per-genome GUNC output directories (DIAMOND hits, gene calls) for 5 sets × 2 databases. The consolidated 8,000-row table with every power field is deposited as `gunc/gunc_per_genome.tsv` |
| `results/revision/cami2/competitors/` | 9.4 GB | raw CheckM2 / CoCoPyE / DeepCheck output trees over the CAMI II bins. The per-bin predictions from all four tools are deposited as `cami2/predictions/*_all_tools.tsv` |
| `results/revision/real_data/*/checkm2_output/` | 2.9 GB | CheckM2 working directories (protein calls, DIAMOND output). The parsed predictions are deposited |
| `results/revision/benchmark/gunc/` | 1.9 GB | superseded GUNC staging tree |
| `data/benchmarks/*/checkm2_output/` | 22 GB | as above, for the benchmark sets |
| `data/benchmarks/set_F/annotations/`, `set_F/orthology/` | 2.6 GB | per-genome Prodigal annotations and reciprocal-best-hit tables behind the distance characterisation. The derived table is deposited as `set_F/set_F_distance_characterization.tsv` |
| `results/revision/metrics_pre_competitor_backup/` | 17 MB | a superseded snapshot of `metrics/` taken before the competitor re-run |
| `results/revision/set_F_pilot/`, `results/revision/gunc/_pilot/` | 4 MB | superseded pilots. The 20-genome GUNC pilot in particular is unrepresentative and only the n = 1,000 result may be cited |
| `results/revision/speed/deepcheck_features/` | 17 MB | DeepCheck feature cache |
| `models/magicc_holdout_{phylum,family}.onnx` | 162 MB each | the two holdout models are validation artefacts, not released models. They are regenerated by `scripts/122_train_holdout_phylum.py` and the family pipeline from the deposited split files and training data |
| `results/revision/ws11/spire_catalogue/` | 11 MB | a workstream still running at deposition time; see `../DEPOSITION_STATUS.md` |

The two holdout ONNX models and the `spire_catalogue` outputs are the only
excluded items that cannot be regenerated in minutes; both are noted in
`DEPOSITION_STATUS.md` at the repository root.
