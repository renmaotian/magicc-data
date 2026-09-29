# Reproduce the scientific results

The ordered drivers below are the supported entry points. `DEPENDENCIES.tsv`
states why each retained engine is needed; `INPUT_MANIFEST.tsv` records every
restored input and SHA256. Earlier exploratory, superseded-training, editorial
and one-time correction scripts remain in Git history at
`37a1a13622310957762e4fa573c3a6303135b7b9`.

## Numerical replay

Use Python 3.11 and install `requirements-numerical.txt`. From this repository:

```bash
python -m pip install -r reproduction/requirements-numerical.txt
python reproduction/01_restore.py --destination /path/to/magicc-analysis
python reproduction/02_reproduce_statistics.py --workspace /path/to/magicc-analysis
```

The first command verifies all committed inputs before copying them to a new
workspace. The second recomputes accuracy, quality-classification thresholds,
signed errors, contamination-type/distance and error-robustness analyses,
reference-curation sensitivity, CAMI and real-assembly comparisons, and the
production model's observational novelty strata. The genus stage recomputes
head-to-head results, the primary difference-in-differences and all three
prespecified sensitivities from the deposited per-sample predictions and truth,
with the recorded 2,000 bootstrap draws.
It does not rerun neural inference or third-party tools. Use `--jobs 1` (default)
or up to `--jobs 4`; `--stages` selects a subset.

Each declared output is removed before its stage runs. The replay checks the
newly computed tables against preserved reference results and records executed
script hashes, commands, return codes and numerical cell comparisons under
`reproduction_audit/`. Floating-point tolerance is `rtol=1e-9, atol=1e-10`.
The current five-set, four-tool panel is checked; old model/dataset rows in
historical source tables are not current evidence. Three earlier-model
prediction tables are retained only to reproduce the original multiple-test
family for the current primary comparisons. Genus tables are regenerated under
`results/revision/holdout_resubmission7/reproduced_statistics/` and checked
against the original full-evaluation tables. This reuses the evaluator's
statistical implementation without loading models or raw features.

Input restoration resolves corrected CoCoPyE stage selection and microbial
CAMI source eligibility to their final inputs. Users do not need to execute
the old sequence of revision-specific patch scripts. Stage-1 CoCoPyE failures
remain unscored and counted in coverage; no valid estimates are clipped.

## Figures and training

Render the five main and eight supplementary scientific figures with:

```bash
python reproduction/03_reproduce_figures.py --workspace /path/to/magicc-analysis
```

This also recomputes the CheckM2 contamination-dose decomposition. Figure
geometry and source-value ledgers accompany the output. Canonical journal
captions, document rendering and tracked changes are separate author materials.

The [full genus training recipe](GENUS_RETRAINING.md) uses the fourth driver.
It is computationally separate from numerical replay and figure generation.

Heavy reproduction starts from separately obtained benchmark assemblies,
exact-version reference genomes and the documented feature-extraction inputs.
The unchanged benchmark FASTAs are on release `v1.0.0`; see the root download
script and release checksums. The unchanged production model and installation
workflow belong to [MAGICC](https://github.com/renmaotian/magicc).
Research model weights are not uploaded to this data repository. Training
uses a fixed recipe and seeds; numerical weights are not promised to be
bit-identical across hardware and library builds.

## Data identities

Display Set C is `set_C_clean`; display Set D is `set_D_clean`. Their metadata,
assembly identifiers and release checksums are unchanged. Original
training-overlapping C/D datasets remain explicitly withdrawn.

The earlier `updates/2026-09/` directory retains scientific data records. Its
historical path components identify provenance, not additional instructions
to execute. The current dependency manifest is the execution authority.

Both percentages use the full dominant-reference genome length. This sequence
definition is not an assertion of exact equivalence to marker/protein-based
definitions. Source-specific domain, truth and scoring restrictions accompany
the results.
