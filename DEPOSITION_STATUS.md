# Deposition status

Updated September 2026. This inventory distinguishes committed results,
downloadable assets and remaining asset publication.

## Committed scientific material

`updates/2026-09/snapshot/` contains corrected CoCoPyE selected-stage predictions,
recomputed statistics, CAMI microbial-source restriction, the new joint
ten-family results, training histories/completion records, feature vocabulary,
arm-specific normalizers, reconstruction manifests and recorded scientific code.
Its explicit SHA256 manifest identifies every supplied file.
`UPDATED_PUBLIC_PATHS.tsv` identifies corrected existing public files.

Production V5 was not retrained or replaced. Research holdout models are a
separate experiment and do not change the deployed predictor.

## Already downloadable assets

Release `v1.0.0` contains the eight benchmark assembly sets and checksums.
Current C/D retain `set_C_clean`/`set_D_clean` IDs and asset names. Release
`v0.1.0` preserves withdrawn original C/D and historical motivating assemblies.
Production V5 remains on the software repository's `v0.3.3` release.

## Prepared assets not yet uploaded

Planned release `v1.1.0` has not been created. The exact sizes and SHA256s are in
`updates/2026-09/RELEASE_ASSETS.json`:

* `magicc_research_matched_full.onnx` — 90-epoch fit, selected epoch 70;
* `magicc_research_family_holdout.onnx` — 49-epoch fit, selected epoch 29;
* `magicc_feature_selection_inputs.zip` — core-gene FASTAs and manifest;
* `magicc_2026_09_scientific_snapshot.zip` — convenience archive of committed data/code.

Research weights and the core-gene archive are not ordinary Git files and are
not publicly downloadable until this inventory is updated. The scientific
snapshot is available from Git independently of its convenience ZIP. Older
phylum/family/genus weights remain outside the previous deposit; their historical
evaluation tables remain available, with the limitations described in the new
holdout documentation.

## External inputs and limits

NCBI/GTDB assemblies are identified by accession and exact input hashes and are
downloaded from their providers. CAMI, Meslier/Zymo and catalogue sequences and
tool databases are not republished here. ATCC Genome Portal data were not used.
Saved parsed results permit inspection without those downloads; sequence
regeneration requires the listed inputs.

This deposit does not claim every historical intermediate or editorial rendering
dependency is public. Private review material, forms, correspondence and
submission-only integrity checks are excluded.
