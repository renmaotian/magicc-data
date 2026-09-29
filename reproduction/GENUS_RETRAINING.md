# Genus-exclusion retraining

This is a new paired research experiment. Ten named bacterial genera are
jointly excluded from the shared feature-selection representatives. Both arms
use that same panel-free vocabulary. The genus-excluded arm removes these
genera from its dominant/donor training and validation pools; the matched-full
arm retains them in its synthesis pools. Each arm fits its own training-only
normalizer. Parent
families and phyla remain represented. This availability-defined bacterial
panel is not evidence of universal generalization to archaea or reduced genomes.

## Inputs and environment

Run `01_restore.py` first. It verifies and extracts one representative
core-gene archive supplied as two chunks. The combined archive is 55,236,128 bytes,
SHA256 `a778685d7e25497eac460cf614527229dc696442bbfb8e39a7ef27c61e53a9db`.
All 2,003 member identities, including 2,000 FASTA files, are bound by
`ARCHIVES.json` and `inputs/core_gene_archive/MEMBERS.json`.

The immutable original genomic source manifest remains at
`results/revision/holdout_resubmission5/reconstruction/genomic_source_manifest.tsv`
in the restored workspace. Its historical directory name identifies the
existing exact input collection reused here. It lists 99,957 files and
369,151,655,339 uncompressed bytes. The retrieval helper verifies the provider's
compressed MD5 and the original uncompressed SHA256, refuses substitutions and
resumes by checking existing files. Availability of every historical NCBI
accession indefinitely cannot be guaranteed.

Use Python 3.11 and the numerical requirements plus PyTorch 2.5.1,
h5py 3.15.1, ONNX 1.20.1, ONNX Runtime 1.23.2 and Numba 0.63.1.
`requirements-tested.txt` records the actual environment, including its CUDA
wheel identifier; training itself runs on CPU. A CPU PyTorch wheel can be used.
The full recipe needs many CPU-hours and several hundred gigabytes for source
genomes, plus generated features/intermediates. The supplied launcher runs two
arms concurrently with 18 synthesis workers and 12 training threads per arm;
do not launch other large workflows against the same CPU allowance.

Obtain the unchanged production comparator:

```bash
curl -L --fail -o /path/to/magicc_v5.onnx \
  https://github.com/renmaotian/magicc/releases/download/v0.3.3/magicc_v5.onnx
```

The fourth driver checks its SHA256 against
`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`.
The production vocabulary and normalizer are already restored. The new
research vocabulary has 9,239 features and must not be paired with the
production 9,249-feature normalizer.

## Full recipe

From this repository, with an otherwise idle prepared analysis workspace:

```bash
# A bounded real download can first be tested with --fetch-limit 1.
python reproduction/04_retrain_genus.py --workspace /path/to/magicc-analysis --step fetch
python reproduction/04_retrain_genus.py --workspace /path/to/magicc-analysis --step prepare
python reproduction/04_retrain_genus.py --workspace /path/to/magicc-analysis --step train \
  --production-model /path/to/magicc_v5.onnx
```

`prepare` reruns selection and feature reselection. The full prevalence counts
must match the original recorded counts before exclusions are applied. Each
arm synthesizes 1,000,000 training, 100,000 validation and 100,000 auxiliary test
assemblies, then trains with the locked 150-epoch maximum and 20-epoch validation
patience. The epoch 20 fixed-budget analysis is a sensitivity, not a replacement
for the validation-selected primary comparison. Training checkpoints and raw batches
are reusable only when their recorded input/code identities match.

Before fresh raw training/evaluation, driver 04 moves unchanged deposited run
certificates to `recorded_original_provenance/` under the experiment directory.
It verifies them against the restoration manifest and preserves generated raw
data, model checkpoints and modified records. This prevents a published
completion certificate from being mistaken for locally generated raw files.
The inference-only route preserves the supplied model/normalizer records.

The launcher then generates 4,770 genus-panel and 1,000 control assemblies, evaluates
the research models and unchanged production comparator, computes reference-
cluster intervals, and executes the provenance/numerical audits. The final
study report and completion records are under
`results/revision/holdout_resubmission7/`.

The completed CPU fits stopped after 48 epochs (selected epoch 28) for the
holdout arm and 49 (selected epoch 29) for the matched-full arm. Both reached
20 consecutive validation epochs without improvement. The observed run used
12 threads per fit on separate CPU sockets; this is an execution record, not a
minimum-memory or runtime guarantee.

Research weights are not uploaded here. If the separately supplied research
models are available, restore them to their model-manifest paths and use
`--step evaluate`; pair every model with its recorded vocabulary/normalizer.
This inference-only route does not recreate the training traces required by
the full training audit. The full `train` launcher runs that audit after training.
Reference bootstrap intervals do not quantify training-seed variability.
Seeds and source identities are fixed, but exact weights are not promised
across hardware and library builds.

The code necessity manifest retains the library modules actually imported by
synthesis, feature selection, training and evaluation. The obsolete trainer and
seeding wrappers are omitted; the current trainer owns its seeded recipe.
Optional host-specific CPU-affinity profiling,
smoke runs, optimizer checkpoints and large raw training HDF5 files are not
public inputs; the full recipe creates its own intermediate data.
