# Resubmission7 genus holdout

**Status: Complete.**

## Design and scope

Ten named bacterial genera were selected from taxonomy and genome counts before any new model outcomes. Eligibility required at least 50 training, 5 validation and 20 test genomes, an unsuffixed established-name genus and exclusion of Patescibacteriota from this primary panel. Each family retained its largest genus and at least 50% of training genomes. Selection first spread across families by training abundance, then filled remaining slots, at most 2 genera/family and 4 genera/phylum. This is an availability-defined panel, with no archaeal genus meeting all rules.

| genus | family | phylum | train | val | test | eval_references | parent_remaining_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Salmonella | Enterobacteriaceae | Pseudomonadota | 2852 | 352 | 339 | 100 | 0.7625 |
| Corynebacterium | Mycobacteriaceae | Actinomycetota | 733 | 95 | 100 | 100 | 0.8088 |
| Prevotella | Bacteroidaceae | Bacteroidota | 447 | 63 | 64 | 64 | 0.5788 |
| Muribaculum | Muribaculaceae | Bacteroidota | 321 | 49 | 37 | 37 | 0.8227 |
| Lactiplantibacillus | Lactobacillaceae | Bacillota | 227 | 33 | 31 | 31 | 0.8537 |
| Rhizobium | Rhizobiaceae | Pseudomonadota | 227 | 28 | 39 | 39 | 0.8108 |
| Bordetella | Burkholderiaceae | Pseudomonadota | 213 | 15 | 22 | 22 | 0.8671 |
| Stenotrophomonas | Xanthomonadaceae | Pseudomonadota | 194 | 26 | 27 | 27 | 0.7184 |
| Sarcina | Clostridiaceae | Bacillota | 137 | 18 | 21 | 21 | 0.7054 |
| Phocaeicola | Bacteroidaceae | Bacteroidota | 251 | 42 | 36 | 36 | 0.5788 |

Exclusion removes 5,602/79,948 training genomes (7.007%). All original training families, orders, classes and 110 phyla remain. The panel spans 9 parent families/4 phyla and evaluates 477 panel test references plus 100 controls, 10 assemblies/reference (5,770 total). Controls are square-root stratified across nonpanel test phyla. Evaluation donors come only from the nonpanel test pool.

## Actual features and matched retraining

Both new arms use 9,239 reselected canonical 9-mers, selected from 964 bacterial and 1000 archaeal training representatives after excluding target genera. The independent recount exactly reproduces the original production prevalence/tie-breaking rule. The new vocabulary retains 8,897/9,249 production features. This vocabulary is an actual model input, not a post hoc diagnostic.

The holdout arm excludes every target genus from dominant and donor pools in training, validation and its auxiliary test set; the matched-full arm retains them. Both synthesize 1,000,000 training, 100,000 validation and 100,000 auxiliary test examples using the full V5 recipe: 800,000 mixed-quality plus 100,000 complete/clean plus 100,000 complete/low-contamination training examples. Exact training-only moments and summary quantiles fit each arm normalizer. Every accepted example retains its seed, dominant accession and donor identities from its simulation plan. Donor lists bound possible contributors; they do not attribute emitted base pairs to individual donors.

One retraining per arm uses seed 42, CPU FP32, the V5 hidden architecture and output bounds, AdamW 0.001/weight decay 0.0005, batch 512, weighted MSE 2:1, cosine restarts 10/2, 2% masking, noise SD 0.01, gradient clip 1, maximum 150 epochs and validation patience 20. The current session exposes no CUDA device. CPU benchmarks select two 12-thread models pinned to separate physical sockets. No epoch/data shortening is permitted. Validation populations differ by arm; their validation errors are not a same-population comparison.

## Estimand, uncertainty and secondary checks

Primary DiD is (MAE_holdout−MAE_matched_full)_genus −(MAE_holdout−MAE_matched_full)_control on identical test assemblies. Reference-cluster percentile intervals use 2,000 draws, paired predictions within each reference, and independently resampled target/control references. Two-sided centered-bootstrap p-values use a plus-one correction, with BH over 20 genus×outcome contrasts. These intervals measure evaluation-reference uncertainty and omit model-seed uncertainty. Joint pool exclusion changes training composition and normalization; the contrast is not an isolated causal taxonomic-novelty effect.

The supported simulation domain is completeness 50–100% and contamination 0–100% of the full dominant reference, with contamination≤completeness. Prespecified secondary results restrict true contamination to ≤35% or ≤10%, and separately compare both models at fixed epoch 20. The latter uses checkpoints saved independently of validation/test performance; neither primary training run stops at 20. Secondary tests receive separate BH adjustment within each 20-contrast analysis.

Canonical GCA/GCF/linked accession checks include accession versions and an additional version-insensitive source check. Target genera and their species are absent from holdout pools and feature selection; ordinary original splits are accession-independent, not globally species-independent. Per-row donor traces and independent numerical recomputation are required for final completion.

Rank definitions follow GTDB. Three targets retain same-base-name GTDB genera in training: Rhizobium_E (1 genome), Bordetella_A/B/C (10 total), and Phocaeicola_A (4). These are distinct GTDB genera. A shared Latin name alone does not establish phylogenetic proximity; the experiment excludes exact GTDB genera rather than every historical or NCBI genus synonym.

## Historical evidence and reproduction

The complete earlier ten-family experiment remains in legacy_sensitivity/ten_family_* with its original estimates. Broad CPR and DPANN experiments also remain, with their historical preprocessing and DPANN source-overlap caveats. The new genus panel differs in references and composition: smaller or larger effects do not establish a paired rank trend.

Restore original inputs using the hashed genomic/core-gene/metadata manifests, then run scripts 282, 283, 288. Launcher 288 runs 284/285 for both arms and 286/287/290/289 after actual completion. Module dependencies are 121/123/125, resubmission7_holdout_config.py and active holdout_lib Python modules. Optional hardware profiling remains local; runtime affinity may be adapted to available CPUs. The original model remains a descriptive deployment comparator using original features and mixed-split normalization, separate from the primary matched-arm contrast.

Selection and analysis contracts, immutable raw-batch checksums, normalization hashes, model hashes and final numerical audits make the chain reviewable. Atomic epoch checkpoints include model/optimizer/scheduler/RNG states. Run the same launcher to resume; identity/configuration differences are rejected. If synthesis already completed, script 285 can instead resume each arm directly after verifying the completed build manifest and full HDF5/normalizer/feature hashes, avoiding reconstruction. Reconstruction was byte-identical in the isolated smoke resume test.

For statistical reproduction from deposited predictions, run script 287 with --from-predictions per_sample_predictions.tsv.gz --output-dir a_separate_directory. This calls the same aggregation used after inference, restores native float32 predictions and round-trip float64 truth, and recomputes all six primary/secondary tables. Its hashed replay manifest certifies statistical recomputation only; it does not claim new training, ONNX inference or source-FASTA validation.

holdout: completed 48 epochs, selected epoch 28; ONNX SHA256 `aafdf4b8047d5a363997ef224ac23c2128dbcc810243a46055d09c63494c8782`; PyTorch–ONNX maximum difference 0.00002289 pp.

matched_full: completed 49 epochs, selected epoch 29; ONNX SHA256 `66bacf6ca1f442aba9c2f98e326959fc5d21d8266b7fe92244676dd4cc54666f`; PyTorch–ONNX maximum difference 0.00002289 pp.

## Completed primary results

| group | metric | did | ci_low | ci_high | p | delta_mae | control_delta_mae | n_samples | n_refs | control_n_samples | control_n_refs | q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Salmonella | comp | 2.999462 | 2.544091 | 3.475255 | 0.0005 | 2.779787 | -0.219676 | 1000 | 100 | 1000 | 100 | 0.000588 |
| Corynebacterium | comp | 3.00331 | 2.49694 | 3.530187 | 0.0005 | 2.783634 | -0.219676 | 1000 | 100 | 1000 | 100 | 0.000588 |
| Prevotella | comp | 3.622648 | 2.833205 | 4.467984 | 0.0005 | 3.402972 | -0.219676 | 640 | 64 | 1000 | 100 | 0.000588 |
| Muribaculum | comp | 1.529432 | 1.012794 | 2.024177 | 0.0005 | 1.309756 | -0.219676 | 370 | 37 | 1000 | 100 | 0.000588 |
| Lactiplantibacillus | comp | 9.226922 | 8.451101 | 9.992379 | 0.0005 | 9.007246 | -0.219676 | 310 | 31 | 1000 | 100 | 0.000588 |
| Rhizobium | comp | 6.045661 | 5.106164 | 6.975335 | 0.0005 | 5.825985 | -0.219676 | 390 | 39 | 1000 | 100 | 0.000588 |
| Bordetella | comp | 4.548641 | 3.380923 | 5.646171 | 0.0005 | 4.328966 | -0.219676 | 220 | 22 | 1000 | 100 | 0.000588 |
| Stenotrophomonas | comp | 2.955048 | 2.358907 | 3.577359 | 0.0005 | 2.735372 | -0.219676 | 270 | 27 | 1000 | 100 | 0.000588 |
| Sarcina | comp | 2.11057 | 1.485625 | 2.756261 | 0.0005 | 1.890894 | -0.219676 | 210 | 21 | 1000 | 100 | 0.000588 |
| Phocaeicola | comp | 2.360189 | 1.318921 | 3.455811 | 0.0005 | 2.140513 | -0.219676 | 360 | 36 | 1000 | 100 | 0.000588 |
| Salmonella | cont | 4.20374 | 3.625944 | 4.788919 | 0.0005 | 4.341055 | 0.137315 | 1000 | 100 | 1000 | 100 | 0.000588 |
| Corynebacterium | cont | 5.649312 | 4.546736 | 6.863991 | 0.0005 | 5.786627 | 0.137315 | 1000 | 100 | 1000 | 100 | 0.000588 |
| Prevotella | cont | 5.435317 | 4.1062 | 6.83499 | 0.0005 | 5.572632 | 0.137315 | 640 | 64 | 1000 | 100 | 0.000588 |
| Muribaculum | cont | 1.236863 | 0.544044 | 1.997592 | 0.002499 | 1.374177 | 0.137315 | 370 | 37 | 1000 | 100 | 0.002776 |
| Lactiplantibacillus | cont | 4.660275 | 3.357854 | 5.938496 | 0.0005 | 4.79759 | 0.137315 | 310 | 31 | 1000 | 100 | 0.000588 |
| Rhizobium | cont | 1.517801 | 0.554635 | 2.443681 | 0.004498 | 1.655116 | 0.137315 | 390 | 39 | 1000 | 100 | 0.004498 |
| Bordetella | cont | 8.11526 | 6.136085 | 9.807029 | 0.0005 | 8.252575 | 0.137315 | 220 | 22 | 1000 | 100 | 0.000588 |
| Stenotrophomonas | cont | 1.479745 | 0.991852 | 2.000928 | 0.0005 | 1.617059 | 0.137315 | 270 | 27 | 1000 | 100 | 0.000588 |
| Sarcina | cont | 0.724341 | 0.231151 | 1.217432 | 0.003498 | 0.861656 | 0.137315 | 210 | 21 | 1000 | 100 | 0.003682 |
| Phocaeicola | cont | 4.416299 | 3.32692 | 5.495146 | 0.0005 | 4.553614 | 0.137315 | 360 | 36 | 1000 | 100 | 0.000588 |

| group | tool | n_refs | comp_mae | cont_mae | comp_bias | cont_bias |
| --- | --- | --- | --- | --- | --- | --- |
| Salmonella | MAGICC_holdout | 100 | 7.074 | 8.747 | -1.9337 | 4.2644 |
| Salmonella | MAGICC_matched_full | 100 | 4.2942 | 4.4059 | 3.172 | -2.9594 |
| Salmonella | MAGICC_V5 | 100 | 3.2654 | 4.0672 | 2.0025 | -3.1366 |
| Corynebacterium | MAGICC_holdout | 100 | 8.4649 | 12.1283 | 1.3991 | 5.6325 |
| Corynebacterium | MAGICC_matched_full | 100 | 5.6813 | 6.3417 | 2.9006 | -4.3273 |
| Corynebacterium | MAGICC_V5 | 100 | 4.7453 | 5.6921 | 2.3611 | -4.239 |
| Prevotella | MAGICC_holdout | 64 | 10.5235 | 12.2725 | 5.9541 | 4.4552 |
| Prevotella | MAGICC_matched_full | 64 | 7.1205 | 6.6998 | 4.4347 | -2.1444 |
| Prevotella | MAGICC_V5 | 64 | 6.2314 | 5.9039 | 3.7632 | -2.3674 |
| Muribaculum | MAGICC_holdout | 37 | 6.3722 | 6.8171 | 1.3178 | 0.0973 |
| Muribaculum | MAGICC_matched_full | 37 | 5.0624 | 5.443 | 3.1384 | -2.7399 |
| Muribaculum | MAGICC_V5 | 37 | 4.1038 | 4.7772 | 2.4164 | -2.6348 |
| Lactiplantibacillus | MAGICC_holdout | 31 | 14.4256 | 9.6945 | 12.9177 | 6.9382 |
| Lactiplantibacillus | MAGICC_matched_full | 31 | 5.4184 | 4.8969 | 3.8311 | -3.2581 |
| Lactiplantibacillus | MAGICC_V5 | 31 | 4.3564 | 5.1618 | 3.1018 | -4.1268 |
| Rhizobium | MAGICC_holdout | 39 | 12.2413 | 6.6104 | 11.2166 | 3.1696 |
| Rhizobium | MAGICC_matched_full | 39 | 6.4153 | 4.9553 | 3.8415 | -1.1946 |
| Rhizobium | MAGICC_V5 | 39 | 5.2368 | 4.6006 | 2.2648 | -1.21 |
| Bordetella | MAGICC_holdout | 22 | 10.7241 | 13.6605 | -5.6176 | -12.6563 |
| Bordetella | MAGICC_matched_full | 22 | 6.3951 | 5.4079 | 4.2443 | -3.6247 |
| Bordetella | MAGICC_V5 | 22 | 4.9959 | 5.3306 | 2.7086 | -3.8902 |
| Stenotrophomonas | MAGICC_holdout | 27 | 8.6683 | 7.1705 | 5.2571 | -1.3896 |
| Stenotrophomonas | MAGICC_matched_full | 27 | 5.933 | 5.5535 | 3.6801 | -2.8722 |
| Stenotrophomonas | MAGICC_V5 | 27 | 4.8007 | 5.5104 | 2.1789 | -3.895 |
| Sarcina | MAGICC_holdout | 21 | 6.918 | 6.5453 | 4.4862 | -4.4607 |
| Sarcina | MAGICC_matched_full | 21 | 5.0271 | 5.6837 | 2.7884 | -4.0867 |
| Sarcina | MAGICC_V5 | 21 | 4.097 | 5.6792 | 1.3726 | -4.6376 |
| Phocaeicola | MAGICC_holdout | 36 | 8.9594 | 10.2116 | -0.3496 | 4.1999 |
| Phocaeicola | MAGICC_matched_full | 36 | 6.8189 | 5.658 | 5.1902 | -3.3814 |
| Phocaeicola | MAGICC_V5 | 36 | 5.8533 | 5.2887 | 4.4026 | -3.6486 |
| in_distribution | MAGICC_holdout | 100 | 6.915 | 7.3858 | 1.7733 | -1.5597 |
| in_distribution | MAGICC_matched_full | 100 | 7.1347 | 7.2485 | 2.0712 | -1.7115 |
| in_distribution | MAGICC_V5 | 100 | 6.3116 | 6.6389 | 1.5128 | -1.7246 |

## Prespecified secondary sensitivities

| sensitivity | group | metric | holdout_mae | matched_full_mae | delta_mae | control_delta_mae | did | ci_low | ci_high | p | n_samples | n_refs | control_n_samples | control_n_refs | scope | q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contamination_le35 | Salmonella | comp | 6.815814 | 2.13927 | 4.676543 | -0.223532 | 4.900076 | 4.196962 | 5.591106 | 0.0005 | 378 | 100 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Corynebacterium | comp | 7.24191 | 3.357508 | 3.884403 | -0.223532 | 4.107935 | 3.359399 | 4.853045 | 0.0005 | 366 | 100 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Prevotella | comp | 9.03193 | 4.885051 | 4.146879 | -0.223532 | 4.370411 | 3.256599 | 5.405346 | 0.0005 | 260 | 64 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Muribaculum | comp | 4.465096 | 2.190855 | 2.274241 | -0.223532 | 2.497773 | 1.712337 | 3.243016 | 0.0005 | 125 | 37 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Lactiplantibacillus | comp | 14.592989 | 3.074218 | 11.518771 | -0.223532 | 11.742303 | 10.432271 | 13.031367 | 0.0005 | 118 | 31 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Rhizobium | comp | 11.552607 | 4.784803 | 6.767804 | -0.223532 | 6.991336 | 5.498199 | 8.390076 | 0.0005 | 159 | 39 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Bordetella | comp | 12.029554 | 4.156014 | 7.87354 | -0.223532 | 8.097072 | 5.66075 | 10.554955 | 0.0005 | 82 | 22 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Stenotrophomonas | comp | 7.392924 | 3.209102 | 4.183822 | -0.223532 | 4.407354 | 3.435122 | 5.399357 | 0.0005 | 87 | 26 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Sarcina | comp | 5.466549 | 2.788478 | 2.678071 | -0.223532 | 2.901604 | 1.808492 | 4.111165 | 0.0005 | 77 | 21 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Phocaeicola | comp | 7.471354 | 3.479671 | 3.991683 | -0.223532 | 4.215215 | 2.707789 | 5.728356 | 0.0005 | 135 | 36 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Salmonella | cont | 9.811828 | 1.954324 | 7.857504 | 0.326773 | 7.530731 | 6.70997 | 8.442479 | 0.0005 | 378 | 100 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Corynebacterium | cont | 13.996768 | 2.94595 | 11.050818 | 0.326773 | 10.724045 | 8.799589 | 12.723746 | 0.0005 | 366 | 100 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Prevotella | cont | 11.767002 | 4.098006 | 7.668995 | 0.326773 | 7.342223 | 5.205558 | 9.595653 | 0.0005 | 260 | 64 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Muribaculum | cont | 5.540537 | 2.319626 | 3.220911 | 0.326773 | 2.894138 | 1.5925 | 4.423767 | 0.001 | 125 | 37 | 329 | 97 | Prespecified secondary sensitivity | 0.001176 |
| contamination_le35 | Lactiplantibacillus | cont | 7.28273 | 2.358025 | 4.924706 | 0.326773 | 4.597933 | 2.964384 | 6.438748 | 0.0005 | 118 | 31 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Rhizobium | cont | 5.272978 | 3.009941 | 2.263037 | 0.326773 | 1.936264 | 0.98285 | 3.044123 | 0.001499 | 159 | 39 | 329 | 97 | Prespecified secondary sensitivity | 0.001578 |
| contamination_le35 | Bordetella | cont | 7.524274 | 2.76808 | 4.756194 | 0.326773 | 4.429421 | 3.246163 | 5.618853 | 0.0005 | 82 | 22 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le35 | Stenotrophomonas | cont | 4.216916 | 2.475366 | 1.74155 | 0.326773 | 1.414777 | 0.547865 | 2.339876 | 0.001499 | 87 | 26 | 329 | 97 | Prespecified secondary sensitivity | 0.001578 |
| contamination_le35 | Sarcina | cont | 3.066476 | 2.341386 | 0.72509 | 0.326773 | 0.398318 | -0.037144 | 0.848379 | 0.08096 | 77 | 21 | 329 | 97 | Prespecified secondary sensitivity | 0.08096 |
| contamination_le35 | Phocaeicola | cont | 11.673876 | 2.714211 | 8.959666 | 0.326773 | 8.632893 | 6.34416 | 11.052399 | 0.0005 | 135 | 36 | 329 | 97 | Prespecified secondary sensitivity | 0.000625 |
| contamination_le10 | Salmonella | comp | 4.907125 | 1.598396 | 3.308729 | -0.342303 | 3.651032 | 2.778484 | 4.560359 | 0.0005 | 124 | 74 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Corynebacterium | comp | 7.064108 | 2.148754 | 4.915354 | -0.342303 | 5.257656 | 4.049655 | 6.571995 | 0.0005 | 94 | 65 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Prevotella | comp | 9.069835 | 4.551986 | 4.517849 | -0.342303 | 4.860152 | 3.53334 | 6.293185 | 0.0005 | 78 | 48 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Muribaculum | comp | 4.996721 | 1.620601 | 3.37612 | -0.342303 | 3.718423 | 2.44915 | 4.911551 | 0.0005 | 38 | 24 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Lactiplantibacillus | comp | 15.835349 | 1.723174 | 14.112176 | -0.342303 | 14.454478 | 13.069771 | 15.883772 | 0.0005 | 41 | 26 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Rhizobium | comp | 10.99836 | 4.036752 | 6.961608 | -0.342303 | 7.30391 | 5.595687 | 9.199061 | 0.0005 | 43 | 30 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Bordetella | comp | 12.589793 | 2.642852 | 9.946941 | -0.342303 | 10.289243 | 6.105053 | 14.279481 | 0.0005 | 24 | 17 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Stenotrophomonas | comp | 5.212256 | 2.140353 | 3.071903 | -0.342303 | 3.414206 | 2.476447 | 4.44869 | 0.0005 | 25 | 17 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Sarcina | comp | 3.809849 | 1.960255 | 1.849594 | -0.342303 | 2.191896 | 1.023884 | 3.359229 | 0.0005 | 19 | 14 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Phocaeicola | comp | 7.230388 | 2.649643 | 4.580744 | -0.342303 | 4.923047 | 2.762669 | 7.378991 | 0.0005 | 40 | 26 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Salmonella | cont | 8.42859 | 1.153559 | 7.275031 | 0.226734 | 7.048296 | 5.968924 | 8.234843 | 0.0005 | 124 | 74 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Corynebacterium | cont | 10.718604 | 1.809034 | 8.909569 | 0.226734 | 8.682835 | 6.263197 | 11.170342 | 0.0005 | 94 | 65 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Prevotella | cont | 7.080567 | 2.266044 | 4.814523 | 0.226734 | 4.587789 | 2.565976 | 6.70768 | 0.0005 | 78 | 48 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Muribaculum | cont | 6.384724 | 1.689606 | 4.695118 | 0.226734 | 4.468383 | 2.343084 | 6.657629 | 0.001 | 38 | 24 | 84 | 62 | Prespecified secondary sensitivity | 0.001249 |
| contamination_le10 | Lactiplantibacillus | cont | 3.147396 | 1.438318 | 1.709078 | 0.226734 | 1.482343 | 0.162211 | 3.871853 | 0.095452 | 41 | 26 | 84 | 62 | Prespecified secondary sensitivity | 0.106058 |
| contamination_le10 | Rhizobium | cont | 2.001831 | 1.506456 | 0.495375 | 0.226734 | 0.268641 | -0.130662 | 0.632067 | 0.195902 | 43 | 30 | 84 | 62 | Prespecified secondary sensitivity | 0.195902 |
| contamination_le10 | Bordetella | cont | 3.843833 | 1.218575 | 2.625258 | 0.226734 | 2.398523 | 1.797329 | 3.055328 | 0.0005 | 24 | 17 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| contamination_le10 | Stenotrophomonas | cont | 2.077353 | 1.247051 | 0.830302 | 0.226734 | 0.603567 | -0.073771 | 1.474516 | 0.112444 | 25 | 17 | 84 | 62 | Prespecified secondary sensitivity | 0.118362 |
| contamination_le10 | Sarcina | cont | 2.028541 | 1.108101 | 0.920441 | 0.226734 | 0.693706 | 0.342016 | 1.043177 | 0.001499 | 19 | 14 | 84 | 62 | Prespecified secondary sensitivity | 0.001764 |
| contamination_le10 | Phocaeicola | cont | 11.123041 | 1.494505 | 9.628537 | 0.226734 | 9.401802 | 5.973053 | 12.86851 | 0.0005 | 40 | 26 | 84 | 62 | Prespecified secondary sensitivity | 0.000666 |
| fixed_epoch20 | Salmonella | comp | 7.055777 | 5.965924 | 1.089853 | -0.247281 | 1.337134 | 0.83188 | 1.842039 | 0.0005 | 1000 | 100 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Corynebacterium | comp | 8.982495 | 6.028182 | 2.954312 | -0.247281 | 3.201593 | 2.657387 | 3.732132 | 0.0005 | 1000 | 100 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Prevotella | comp | 10.834463 | 8.410341 | 2.424122 | -0.247281 | 2.671403 | 1.930169 | 3.425673 | 0.0005 | 640 | 64 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Muribaculum | comp | 6.610633 | 5.56425 | 1.046384 | -0.247281 | 1.293665 | 0.759943 | 1.791309 | 0.0005 | 370 | 37 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Lactiplantibacillus | comp | 13.877174 | 5.877092 | 8.000081 | -0.247281 | 8.247362 | 7.454598 | 9.012902 | 0.0005 | 310 | 31 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Rhizobium | comp | 13.361832 | 9.63833 | 3.723502 | -0.247281 | 3.970782 | 3.336273 | 4.611963 | 0.0005 | 390 | 39 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Bordetella | comp | 10.707392 | 7.892578 | 2.814814 | -0.247281 | 3.062095 | 1.915383 | 4.279842 | 0.0005 | 220 | 22 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Stenotrophomonas | comp | 9.054963 | 8.327214 | 0.727749 | -0.247281 | 0.97503 | 0.339218 | 1.64552 | 0.004498 | 270 | 27 | 1000 | 100 | Prespecified secondary sensitivity | 0.004998 |
| fixed_epoch20 | Sarcina | comp | 6.655758 | 5.539324 | 1.116434 | -0.247281 | 1.363715 | 0.804162 | 2.084192 | 0.001499 | 210 | 21 | 1000 | 100 | Prespecified secondary sensitivity | 0.001764 |
| fixed_epoch20 | Phocaeicola | comp | 9.371618 | 6.864843 | 2.506774 | -0.247281 | 2.754055 | 1.740548 | 3.792513 | 0.0005 | 360 | 36 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Salmonella | cont | 8.165824 | 5.00918 | 3.156644 | 0.249331 | 2.907313 | 2.355862 | 3.50344 | 0.0005 | 1000 | 100 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Corynebacterium | cont | 12.442632 | 6.562047 | 5.880585 | 0.249331 | 5.631254 | 4.475795 | 6.793561 | 0.0005 | 1000 | 100 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Prevotella | cont | 13.010058 | 7.115874 | 5.894184 | 0.249331 | 5.644853 | 4.20953 | 7.192535 | 0.0005 | 640 | 64 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Muribaculum | cont | 7.095443 | 6.193805 | 0.901638 | 0.249331 | 0.652307 | -0.109536 | 1.539227 | 0.123438 | 370 | 37 | 1000 | 100 | Prespecified secondary sensitivity | 0.123438 |
| fixed_epoch20 | Lactiplantibacillus | cont | 9.598794 | 4.982605 | 4.61619 | 0.249331 | 4.366859 | 3.102795 | 5.616672 | 0.0005 | 310 | 31 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Rhizobium | cont | 8.507198 | 5.262805 | 3.244393 | 0.249331 | 2.995062 | 1.867145 | 4.15198 | 0.0005 | 390 | 39 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Bordetella | cont | 13.881247 | 5.980986 | 7.900261 | 0.249331 | 7.65093 | 5.940445 | 9.307676 | 0.0005 | 220 | 22 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Stenotrophomonas | cont | 8.040094 | 6.023893 | 2.016201 | 0.249331 | 1.76687 | 1.210639 | 2.306624 | 0.0005 | 270 | 27 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
| fixed_epoch20 | Sarcina | cont | 6.778732 | 5.872903 | 0.905829 | 0.249331 | 0.656498 | 0.134483 | 1.205774 | 0.017491 | 210 | 21 | 1000 | 100 | Prespecified secondary sensitivity | 0.018412 |
| fixed_epoch20 | Phocaeicola | cont | 10.672973 | 6.292726 | 4.380247 | 0.249331 | 4.130916 | 3.036059 | 5.295486 | 0.0005 | 360 | 36 | 1000 | 100 | Prespecified secondary sensitivity | 0.000625 |
