# September 2026 scientific update

`snapshot/` preserves project-relative `data/`, `results/revision/`, `scripts/`
and supporting package paths. `SCIENTIFIC_FILE_MANIFEST.tsv` records bytes and
SHA256s. Historical `resubmission5` path components identify the unchanged
analysis record, not different numerical results.

## Current results

All paths below are relative to `snapshot/`:

| Analysis | Location |
|---|---|
| New joint ten-family experiment | `results/revision/holdout_resubmission5/analysis_report.md` |
| Per-sample model predictions | `results/revision/holdout_resubmission5/per_sample_predictions.tsv.gz` |
| Raw family/control accuracy | `results/revision/holdout_resubmission5/head_to_head_by_group.tsv` |
| Control-adjusted contrasts | `results/revision/holdout_resubmission5/did.tsv` |
| Official CoCoPyE selector audit | `results/revision/cocopye_stage_resubmission5/parser_verification.json` |
| Corrected source mapping | `results/revision/cocopye_stage_resubmission5/source_overrides.json` |
| Comparator/statistical corrections | `results/revision/cocopye_stage_resubmission5/CORRECTION_REPORT.md` |
| CAMI source-unit audit | `results/revision/holdout_resubmission5/cami_source_audit/` |
| Duplication verification | `results/revision/cocopye_stage_resubmission5/independent_figures/` |
| Preprocessing sensitivity | `results/revision/normalization_sensitivity_resubmission5/` |

`UPDATED_PUBLIC_PATHS.tsv` records old/current hashes at existing public paths.
Their earlier contents remain at commit
`c051b3d8383b560f34942e24ea4cbfe154f079c4`. Historical narrative reports may
describe earlier statistics; consult the corrected source map first.

Display Set C means `set_C_clean`; Set D means `set_D_clean`. Their v1.0.0
assemblies and checksums are unchanged. `DISPLAY_DATASET_MAP.tsv` records the
distinct withdrawn historical IDs. Do not rename data directories or genome IDs.

## Restore a workspace

Run from the data-repository root, choosing a new empty destination:

```bash
python updates/2026-09/restore_workspace.py --destination /path/to/magicc-analysis
```

The helper restores public results, benchmark metadata, splits and original
script names from `SCRIPT_MAPPING.tsv`, then overlays the snapshot. It records
copied hashes. Only literal historical root paths are adapted in the restored
code/configuration, with before/after hashes recorded; no statistical code is
rewritten and neither this repository nor production releases are changed.
Install packages using the supplied analysis-specific software/version records.

Download the unchanged production model from its existing public release:

```bash
curl -L --fail -o /path/to/magicc_v5.onnx \
  https://github.com/renmaotian/magicc/releases/download/v0.3.3/magicc_v5.onnx
```

Pass `--production-model /path/to/magicc_v5.onnx` when restoring, or later place
it at `models/magicc_v5.onnx` after checking SHA256
`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`.

### New holdout

Read the restored `results/revision/holdout_resubmission5/reconstruction/README.md`
and source/metadata manifests. Script 250 fetches exact source accessions and
checks their bytes. The core-gene archive is required for byte-matched feature
reselection; its pending publication is explicit in `RELEASE_ASSETS.json`.
Once those inputs are restored, the full recipe is:

```bash
python scripts/240_select_resubmission5_holdout.py
python scripts/242_reselect_resubmission5_features.py
bash scripts/247_run_resubmission5_holdout.sh
```

This trains two full models with one million training samples each. It is not a
smoke demonstration. Changing hardware/dependencies may change seeded training
trajectories. The research ONNX assets reproduce the recorded weights once
uploaded; pair each with its own normalizer and shared 9,243-feature vocabulary,
not the production 9,249-feature input definition. Reference bootstrap intervals
do not capture seed-to-seed training variability.

### Comparator/statistical corrections

Script 252 reconstructs selected CoCoPyE output from full CSVs and checks the
official 0.5.0 API. Scripts 253–272 and the recorded numerical drivers document
the isolated replay, source eligibility and corrections. Completed tables and
their source map can be inspected directly. Full replay also needs each
analysis's listed third-party/raw inputs; this reduced deposit does not promise
one-command regeneration of every large intermediate or editorial display.
Submission-only gates and private-comment/rendering sources are excluded.

## Pending assets

`RELEASE_ASSETS.json` records exact prepared names/sizes/hashes. Research models,
the core-gene archive and convenience scientific ZIP have not yet been uploaded
as v1.1.0. Tables/scripts in this directory are committed independently of those
assets. The root `DEPOSITION_STATUS.md` is the publication-status authority.
