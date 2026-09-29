# September 2026 scientific records

This directory preserves the corrected per-sample predictions, numerical tables
and provenance of the September analysis. Historical `resubmission5` components
identify the original analysis paths. The executable workflow is now curated
under [reproduction/](../../reproduction/README.md); the old correction and
exploratory scripts are preserved in Git history at
`37a1a13622310957762e4fa573c3a6303135b7b9`.

`SCIENTIFIC_FILE_MANIFEST.tsv` identifies the scientific data files remaining in
`snapshot/`. `UPDATED_PUBLIC_PATHS.tsv` records corrected existing public paths;
`DISPLAY_DATASET_MAP.tsv` maps display C/D to `set_C_clean`/`set_D_clean`.

The records include selected-stage CoCoPyE predictions and source mappings,
shared-reference pooled intervals, reference-level Set F tests, microbial CAMI
source eligibility, duplication and fixed-weight normalization analyses, and
the completed earlier ten-family experiment. The new genus experiment is the
primary retraining study; the family results remain secondary evidence.

Use the current restore driver to obtain a coherent analysis workspace. It
resolves these source mappings and avoids rerunning historical patch chains.
Research weights remain outside this repository. The representative core-gene
inputs are now committed as a verified multipart archive under
`reproduction/inputs/core_gene_archive/`; the old `RELEASE_ASSETS.json` records
historical prepared artifact identities, not new download promises.
