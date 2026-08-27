# Leave-genus-out holdout — status at deposition (2026-08-27)

**This workstream was still running when this repository was deposited.**
What is here is the panel definition and the exploratory analysis that fixes
it; the retrained model, its evaluation and the difference-in-differences
table are **not** here.

| File | Contents |
|---|---|
| `eda_panel_summary.json` | the selected genus panel: which genera are held out, from which families and phyla, with the training/validation/test genome counts and the retention fractions of every parent family and phylum |
| `panel_genus_detail.tsv` | one row per held-out genus |
| `eda_genus_counts.tsv` | genus-level genome counts across the splits, the frame the panel was drawn from |
| `level_switch_reproduction.json` | verification that the family and genus panels are re-derived by the same selection rule at a different taxonomic level |

**Design constraint, verified rather than asserted:** every held-out genus's
parent family *and* parent phylum remain in the training set, so the
evaluation measures genus-level novelty inside a known family — not the
family- or phylum-level novelty already measured in `../holdout_family/` and
`../holdout_phylum/`.

These four files are a snapshot taken on 2026-08-27 while the workstream was
running; `level_switch_reproduction.json` in particular is rewritten by the
pipeline as it progresses.

The two completed taxonomic-holdout evaluations (phylum and family) are
deposited in full. When the genus evaluation completes it belongs here, in the
same file layout as `../holdout_family/`: `per_sample_predictions.tsv.gz`,
`per_reference_errors.tsv`, `metrics_full.json`, `lineage_novelty_effect_did.tsv`,
`stratified_error_by_band.tsv`, `mimag_confusion_by_group.tsv`,
`clean_sets_evaluation.tsv` and the k-mer re-selection control.
