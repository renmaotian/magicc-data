# Leave-genus-out holdout — complete

This evaluation was still running when the repository was first deposited on
2026-08-27 and was recorded then as an outstanding gap. **It is now complete and
deposited in full**, in the same layout as `../holdout_family/`.

**Design constraint, verified rather than asserted:** every held-out genus's
parent family *and* parent phylum remain in the training set, so what is measured
is genus-level novelty inside a known family — not the family- or phylum-level
novelty already measured in `../holdout_family/` and `../holdout_phylum/`.

| File | Contents |
|---|---|
| `genus_vs_family_vs_phylum_did.tsv` | **the headline table** — the same six lineage groups scored at all three holdout levels, so the genus, family and phylum difference-in-differences sit side by side per group |
| `lineage_novelty_effect_did.tsv` | genus-level DiD in completeness and contamination MAE against the in-distribution control, with 95 % cluster-bootstrap intervals and BH-corrected q-values |
| `head_to_head_by_group.tsv`, `four_model_head_to_head.tsv` | raw signed bias, production V5 against the holdout models, per group |
| `per_sample_predictions.tsv.gz`, `per_reference_errors.tsv` | per-genome and per-reference results |
| `stratified_error_by_band.tsv`, `sub_phylum_breakdown.tsv`, `mimag_confusion_by_group.tsv`, `mimag_threshold_by_group.tsv` | stratified error, sub-lineage breakdown and MIMAG-inspired agreement |
| `clean_sets_evaluation.tsv`, `set_{C,D}_clean_predictions_both_models.tsv` | both models scored on the clean C/D sets |
| `panel_genus_detail.tsv`, `eda_genus_counts.tsv`, `eda_panel_summary.json` | the panel definition and the frame it was drawn from |
| `level_switch_reproduction.json` | verification that the family and genus panels are re-derived by one selection rule at two taxonomic levels |
| `kmer_reselection_*` , `selected_kmers_holdout.txt` | the k-mer re-selection control under the holdout selection |
| `ladder_per_sample.tsv.gz`, `ladder_report.md` | the novelty ladder computed on this panel |
| `dryrun_v5_vs_v5/` | the null control: the whole harness run with production V5 in *both* arms, so any effect it reports is measurement noise |
| `metrics_full.json`, `ws11g_consolidated.json`, `WS11_G_REPORT.md` | consolidated metrics and the workstream report |
| `model_sha256.txt`, `onnx_export_verification.json` | the SHA256 of the retrained model and its ONNX-versus-PyTorch equivalence check (max abs diff 2.29e-05 on real inputs) |

**The retrained model itself is not published** — see the note on holdout models
in the repository README. Its checksum and export verification are here, so a
model regenerated from the deposited scripts and splits can be checked against
what was actually used.
