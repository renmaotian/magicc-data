# Reconstructing the ten-family experiment

Preserve the supplied submission archive before regenerating files. Extract its
paths into a fresh MAGICC checkout. The scripts resolve their project root from
their own locations; historical absolute paths in the source split tables are
remapped automatically. Completed numerical outputs are directly under
`results/revision/holdout_resubmission5/`, with raw evaluation inputs under
`evaluation/`. `smoke_evaluation/` contains development checks only.

## Software and exact dependency closure

Use Python 3.11 and the package versions in `software_versions.json`. The run
used the existing `magicc2` conda environment. A CPU build of the same PyTorch
version can reproduce the specified FP32 recipe without CUDA; hardware and
library differences can prevent bitwise equality of neural-network weights.

The analysis code closure is:

- Scripts `240_select_resubmission5_holdout.py`, `241_holdout_cpu_benchmark.py`,
  `242_reselect_resubmission5_features.py`,
  `243_build_resubmission5_holdout_data.py`,
  `244_train_resubmission5_holdout.py`,
  `245_generate_resubmission5_holdout_eval.py`,
  `246_evaluate_resubmission5_holdout.py`,
  `247_run_resubmission5_holdout.sh`, `248_report_resubmission5_holdout.py`,
  `249_audit_gorg_sag_source.py`, `250_reconstruct_resubmission5_inputs.py`.
- `scripts/resubmission5_holdout_config.py` and `scripts/holdout_lib/*.py`.
- Reused implementations `scripts/121_build_holdout_training_data.py`,
  `scripts/123_generate_holdout_eval_sets.py`, and
  `scripts/125_kmer_reselection_control.py`. Run the new wrapper scripts,
  which apply the new panel and actual feature reselection; do not run these
  legacy entry points independently for the new experiment.

Script 250 uses only the Python standard library. Script 249 is an independent
SAG metadata audit and is not required to train or evaluate holdout models.

## Restore metadata and versioned reference FASTAs

`metadata_input_manifest.tsv` gives SHA256 values and expected relative paths
for the three original train/validation/test split tables, both representative
selection tables, both original prevalence tables, the production k-mer list,
and the core-gene result tables. Keep these source tables unchanged.

`genomic_source_manifest.tsv` maps all 99,957 reference FASTAs to exact download
accessions, archived filenames, split identities and NCBI FTP URLs. The FASTAs
total 369,151,655,339 bytes in the original checkout. Because GenBank and RefSeq
can differ, the `download_accession` column, derived from the actual archived
FASTA filename, is the retrieval key; do not substitute `ncbi_accession`.

From the new checkout root, restore missing files with:

```bash
python scripts/250_reconstruct_resubmission5_inputs.py fetch
```

This sequential, resumable command downloads only missing files, verifies the
compressed file against NCBI's `md5checksums.txt`, verifies FASTA content,
the original uncompressed byte count and original SHA256, and writes receipts. Failed exact
versions remain failures; the helper does not substitute newer versions or
other assemblies. Review any failed retrieval before proceeding. Existing
files are checked by both size and SHA256. All 99,957 original genomic FASTAs
were fully read to calculate the supplied `original_genomic_sha256` values;
the complete manifest SHA256 is recorded in `reconstruction_summary.json`.

NCBI documents these assembly-specific genomic FASTAs and checksums in its
[Genomes FTP reference](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/data-processing/policies-annotation/genomeftp/).
A bounded verification download of `GCA_041661045.1` reproduced the original
1,339,431-byte FASTA exactly: SHA256
`971c4ee1b8712293954b5645ea6946756ee577e6395ab207a03dfcc2457d14cc`.
This single download validates the retrieval mechanism, not availability of
every historical accession indefinitely. The full original-source checksum
audit is independent of that bounded network verification.

## Restore actual feature-selection inputs

Extract the separate representative core-gene input archive, preserving
`data/kmer_selection/bacterial_core_genes/` and
`data/kmer_selection/archaeal_core_genes/`. The 2,000 FASTAs total 185,471,093
bytes. Every input FASTA, accession and SHA256 is listed in
`core_gene_input_manifest.tsv`. The source genome selection and original
extracted nucleotide core genes are fixed inputs. The new feature-reselection
script does not rerun gene calling or replace these inputs with a newer marker
database. It first reproduces the original full-representative prevalence
counts exactly, then excludes the primary-panel families and selects the
actual 9,243 model features.

Also restore the frozen production comparator to `models/magicc_v5.onnx`,
SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`,
and its original `data/features/normalization_params.json`, SHA256
`b1e3f211a43560d8bed1dd8264921c7ec8ce0529c9bdbdbdc000894c9ce1d7d9`.
These are used only for the separately identified production comparator.

## Full recomputation

```bash
python scripts/240_select_resubmission5_holdout.py
python scripts/242_reselect_resubmission5_features.py --workers 12
bash scripts/247_run_resubmission5_holdout.sh
```

The shell driver uses the active `python`; set `R5_PYTHON` to an explicit
interpreter if needed. Both full arms run concurrently. Each regenerates
1,000,000 training, 100,000 validation and 100,000 auxiliary test samples,
fits training-only normalization, and trains the full architecture with the
locked recipe until validation patience or 150 epochs. Do not substitute the
two-epoch smoke runs for this work. CPU training takes many hours; substantial
reference-reading and feature-storage capacity is also required.

`runtime_affinity.json` records host-specific CPU tuning after a paired
benchmark. Remove this optional file or replace its CPU IDs for another host;
it does not alter the scientific recipe. The original host has 24 physical
cores and 48 logical CPUs, and assigns 12 physical cores to each training arm.

The driver then generates all 9,720 paired evaluation assemblies, counts both
the new and production feature vocabularies, verifies all 8,575 shared features
row for row, evaluates the three models and computes reference-cluster
bootstrap intervals. The final report preserves former phylum/family/genus
results from `legacy_sensitivity/` without needing their original output
directories. On another checkout, numerical equality of synthetic sequences
is expected with the fixed source sequences and software; compressed-file
byte hashes and metadata paths can differ because of timestamps/root paths.

## Completion gates and submission artifacts

For each of `holdout` and `matched_full`, require:

- `models/full/completion.json` with `status: TRAINING_COMPLETE`.
- `models/full/model.onnx`, `training_config.json`, `training_history.json`.
- `data/normalization_params.json` and `data/build_manifest.json` with
  `status: DATA_COMPLETE` and the full sample counts.

At the top analysis directory, require `evaluation_summary.json` with
`status: EVALUATION_COMPLETE`, plus `head_to_head_by_group.tsv`, `did.tsv`,
`control_model_difference.tsv`, `per_sample_predictions.tsv.gz`,
`per_reference_errors.tsv`, `mimag_confusion_by_group.tsv`,
`model_provenance.tsv`, `convergence_check.tsv`, and
`evaluation/verification_manifest.tsv`. Final model ONNX weights are supplied
separately; large raw HDF5/batch artifacts and optimizer checkpoints are
regenerated by the scripts and are not submission-archive requirements.

The independent SAG audit uses the original ENA assembly manifest at
`data/real_data/gorg_tropics/ena_PRJEB33281_assemblies.tsv` and local GTDB R226
`data/gtdb/{bac120_metadata,ar53_metadata}.tsv.gz`; their SHA256 values are
recorded in `sag_source_audit/audit_summary.json`. The audit's joined records
and correction are supplied even if those larger metadata inputs are obtained
separately. No SAG accuracy benchmark is implied by this metadata audit.
