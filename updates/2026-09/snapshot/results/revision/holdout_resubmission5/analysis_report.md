# Ten-family holdout analysis for resubmission 5

**Status:** Complete; final matched-model predictions available.

## Selection and scope

Ten established-name bacterial families were selected from taxonomy and split counts before computing new predictions. Eligibility required ≥50 training, ≥5 validation and ≥20 test genomes, exclusion of Patescibacteriota, retention of each parent phylum's largest family and ≥50% of parent training genomes. One family per eligible parent phylum was selected first; remaining slots were filled by training abundance with a maximum of two per parent. The rules yield seven parent phyla; no archaeal family meets all requirements. This is an availability-defined bacterial panel, not a claim of universal taxonomic representativeness.

| family | phylum | train | val | test | eval_references | parent_remaining_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| Streptococcaceae | Bacillota | 4296 | 528 | 478 | 100 | 0.7119 |
| Pseudomonadaceae | Pseudomonadota | 2348 | 305 | 332 | 100 | 0.8407 |
| Bacteroidaceae | Bacteroidota | 1657 | 208 | 229 | 100 | 0.667 |
| Helicobacteraceae | Campylobacterota | 1627 | 210 | 205 | 100 | 0.6728 |
| Bifidobacteriaceae | Actinomycetota | 1209 | 137 | 133 | 100 | 0.8738 |
| Leptospiraceae | Spirochaetota | 305 | 38 | 37 | 37 | 0.6573 |
| Cyanobiaceae | Cyanobacteriota | 304 | 43 | 35 | 35 | 0.7692 |
| Moraxellaceae | Pseudomonadota | 1787 | 205 | 202 | 100 | 0.8407 |
| Lactobacillaceae | Bacillota | 1552 | 199 | 212 | 100 | 0.7119 |
| Flavobacteriaceae | Bacteroidota | 1100 | 153 | 155 | 100 | 0.667 |

The panel removes 16,185/79,948 training genomes (20.24%). All 110 original training phyla remain, with 65.7–87.4% of each selected family's parent phylum retained. The full design targets 872 panel test references ×10 simulations =8,720 assemblies plus 100 nonpanel test references ×10=1,000 control assemblies. Every dominant and contaminant is drawn from the test split during evaluation.

Leptospiraceae and Cyanobiaceae are the only represented families in their respective orders (Leptospirales and PCC-6307), so their exclusion also removes those orders. The other eight families retain their orders; all classes and phyla remain represented. These two rows must be labeled as family exclusions that also create order-level absence. Canonical/linked accession audits find zero overlaps between any two original genome splits.

## Features, controls and retraining

The actual model vocabulary was reselected from 878 panel-free bacterial and all 1,000 archaeal training representatives, using the original top-9,000 bacterial plus top-1,000 archaeal prevalence rule and merging canonical 9-mers. The independent recount reproduced all original prevalence counts exactly. The resulting 9,243 features retain 8,575/9,249 (92.71%) production features; Jaccard=0.864677. Both new arms use this exact vocabulary; feature reselection is not merely a diagnostic.

The holdout arm excludes the ten families from dominant and contaminant roles in training and validation. The matched full-family arm includes them. Both arms regenerate 1,000,000 training samples (800,000 V4 recipe +100,000 complete/clean +100,000 complete/low contamination), 100,000 validation and 100,000 auxiliary test samples. The original six-category synthesis mixture, fragmentation, contamination definition and training-domain constraint are retained. Planners and seed formulas come from the frozen V5 implementation. Every accepted sample has a saved seed, dominant accession, contaminant accessions, target/observed labels and quality tier.

Each arm fits its own normalization on all and only its training samples, using exact k-mer moments and exact summary-feature quantiles. No validation/test values participate. The frozen production model uses the original normalizer, which was fitted to 800,000 training +100,000 validation +100,000 test V4-recipe samples, then reused for the additional 200,000 V5 training samples. Therefore the frozen deployment comparator must not be described as fully independent at preprocessing level.

There is one seeded retraining per arm (seed 42). Hidden layers/output bounds, AdamW learning rate 0.001, weight decay 0.0005, weighted MSE 2:1, batch size 512, cosine warm restarts 10/2, masking 2%, noise SD 0.01, gradient clipping at 1, maximum 150 epochs and patience 20 match the V5 recipe. GPU device access was unavailable to the execution environment, so training used CPU FP32; gradient checkpointing is disabled because memory is ample. The first layer has 9,243 inputs rather than 9,249, identically in both matched arms. Vectorized random augmentation preserves the original independent masking/noise distributions. Atomic checkpoints contain all RNG, optimizer and scheduler states. Training/validation sample counts are matched, but validation taxonomy differs by arm; validation MAEs are not same-population comparisons.

## Estimand and uncertainty

Every model predicts the same evaluation assemblies. Report raw MAE, signed error, R², both-arm MAE differences and the common-control model difference. The primary adjusted contrast is (MAE_holdout−MAE_matched_full)_family −(MAE_holdout−MAE_matched_full)_control. This is the excess error change following joint panel exclusion, after subtracting a global control change. Exclusion changes training-pool composition and normalizers; the adjustment does not isolate taxonomy alone or remove lineage-specific training interactions.

95% percentile intervals resample dominant reference genomes (2,000 replicates), with model predictions paired within references and the family/control reference sets sampled independently. Two-sided centered-bootstrap p-values use a plus-one correction; BH covers all 20 family×outcome contrasts. These intervals cover evaluation-reference variation, not model-seed uncertainty. Primary analyses enforce the stated completeness 50–100% and contamination ≤ completeness domain; all predictions remain available, including any samples outside that domain.

Feature-row identity is checked against independent FASTA recounts under the production vocabulary: every shared k-mer must agree for every row. Simulator genome IDs are local to each group and are stored alongside features; they are checked within group during model evaluation. The pooled evaluation_id combines analysis_group and genome_id and must be globally unique. PyTorch–ONNX equivalence is required within 1e−4. Convergence records include best epoch, stopping reason and proximity of the best epoch to the final epoch; capped or still-improving runs require cautious interpretation.

## Historical CPR and DPANN evidence

Parks et al. (2018) explicitly consolidated the historical Candidate Phyla Radiation into a single GTDB phylum ([primary paper](https://pubmed.ncbi.nlm.nih.gov/30148503/)); [GTDB R226](https://gtdb.ecogenomic.org/stats/r226) lists Patescibacteriota at phylum rank. Its broad evolutionary scope motivates reporting it separately from a common-family panel. It does not make the earlier unfavorable observation invalid, and diversity alone has not been shown to explain that loss.

Exact legacy outputs and provenance are preserved in legacy_sensitivity/. Their historical CPR adjusted completeness errors were 22.8114 pp (phylum), 5.0338 pp (family) and 4.6431 pp (genus), on different evaluation panels; they must not be interpreted as a strictly paired rank ladder. Their frozen production features and mixed-split normalization limit claims of complete training-process exclusion.

The former DPANN phylum panel contains these production-model split categories: [{"dominant_v5_split": "test", "samples": 108, "references": 9}, {"dominant_v5_split": "train", "samples": 768, "references": 64}, {"dominant_v5_split": "val", "samples": 120, "references": 10}]. The full DPANN contrast includes seen dominant references and must remain labeled accordingly.

## Reproduction

Use the software versions and input restoration instructions in reconstruction/. Run scripts 240, 242, then 247. Script 247 runs 243/244 for both arms and then 245/246/248. Script 241 records runtime only. Two simultaneous full-width CPU benchmarks averaged 0.418/0.328 seconds per training batch with each 12-thread model restricted to a separate physical socket, versus 0.834/0.975 seconds unbound. Runtime affinity is recorded separately and changes no scientific setting. Full training remains governed by the specified maximum epochs and validation patience. The persistent exec session is necessary in the current sandbox; detached shell jobs are terminated when their execution namespace closes.

Family selection is deterministic without random sampling. Evaluation base seed is 20260908: reference sampling uses base + CRC32(group) % 100000, while each simulated assembly uses base + sum(character codes of group) × 1000003 + sample index. Bootstrap substreams use (base + CRC32(contrast label)) modulo 2^32. Synthetic training retains the original V5 batch planner formulas; both neural-network arms use seed 42. Every evaluation sample stores its exact seed. All paths, scripts, input hashes and exact legacy copies are recorded in provenance_manifest.tsv. Raw batch artifacts and per-sample provenance make regeneration resumable without treating frozen-model predictions as holdout retraining.

## Final results

Both full training arms completed under the prespecified patience-20 stopping rule: holdout 49 epochs with best epoch 29; matched full-family 90 epochs with best epoch 70. Neither reached the 150-epoch cap. All 9,720 evaluation assemblies from 972 references met the primary domain. The adjusted completeness MAE increases range from 2.95 to 11.16 pp and contamination increases from 3.66 to 16.97 pp. Every 95% reference-bootstrap interval excludes zero; all BH q-values equal the minimum attainable bootstrap p-value, 1/2,001 (approximately 0.000500). The common-control MAE changes are 0.318 pp for completeness and 0.862 pp for contamination. These are results of the joint exclusion experiment with one training seed per arm; they must not be presented as an isolated causal effect of novelty or as accuracy of the unchanged deployed model.

The unchanged production V5 comparator has lower completeness MAE than the new matched full-family model in every panel family. Its contamination MAE is slightly higher in nine families and slightly lower in Leptospiraceae. The exclusion results therefore need to be distinguished from the deployment benchmark, whose predictions and model were unchanged. Full raw errors and adjusted contrasts are retained below. The independent numerical audit reconstructs MAEs by reference means and contrasts by a separate group/model pivot; all agree within 7.11e−15 pp after restoring the native float32 prediction dtype from CSV.

The original long-running shell completed synthesis, training, export and input verification, then encountered a truncated command after an earlier edit to its source file. Scripts 246 and 248 were subsequently invoked directly on the completed artifacts. No data, model weights, training settings or statistical settings were changed. Exact continuation commands and hashes are recorded in reconstruction/execution_completion.json; the final launcher passes bash syntax validation.

{
  "status": "EVALUATION_COMPLETE",
  "total_samples": 9720,
  "primary_samples": 9720,
  "outside_domain_samples": 0,
  "primary_references": 972,
  "n_panel_groups": 10,
  "bootstrap_replicates": 2000,
  "bootstrap_unit": "dominant reference genome",
  "multiplicity": "BH over all 20 family\u00d7metric DiD contrasts",
  "seed": 20260908,
  "interpretation": "Joint family-panel exclusion effect relative to matched full-family training, adjusted for common-control model difference; training-pool interactions and one-seed model variability remain.",
  "feature_selection": "Both newly retrained arms use the same panel-free reselected vocabulary of 9,243 canonical 9-mers. No target family enters feature selection.",
  "frozen_production_role": "Descriptive deployed-model comparator; excluded from primary matched-arm DiD.",
  "convergence": [
    {
      "arm": "holdout",
      "epochs_completed": 49,
      "best_epoch": 29,
      "best_validation_loss": 151.39031105331588,
      "early_stopped": true,
      "hit_maximum_epochs": false,
      "best_in_last_20_epochs": false
    },
    {
      "arm": "matched_full",
      "epochs_completed": 90,
      "best_epoch": 70,
      "best_validation_loss": 128.16071232193278,
      "early_stopped": true,
      "hit_maximum_epochs": false,
      "best_in_last_20_epochs": false
    }
  ]
}

| group | metric | did | ci_low | ci_high | p | delta_mae | control_delta_mae | n_samples | n_refs | control_n_samples | control_n_refs | q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Streptococcaceae | comp | 5.11482 | 4.42963 | 5.83403 | 0.0005 | 5.43241 | 0.31759 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Pseudomonadaceae | comp | 11.15692 | 10.33969 | 11.97844 | 0.0005 | 11.47451 | 0.31759 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Bacteroidaceae | comp | 5.85799 | 5.06838 | 6.72162 | 0.0005 | 6.17558 | 0.31759 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Helicobacteraceae | comp | 5.64341 | 4.97261 | 6.35259 | 0.0005 | 5.961 | 0.31759 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Bifidobacteriaceae | comp | 6.6618 | 5.82872 | 7.4878 | 0.0005 | 6.97939 | 0.31759 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Leptospiraceae | comp | 9.84409 | 8.71968 | 10.94222 | 0.0005 | 10.16168 | 0.31759 | 370 | 37 | 1000 | 100 | 0.0005 |
| Cyanobiaceae | comp | 7.80736 | 5.81702 | 9.93532 | 0.0005 | 8.12495 | 0.31759 | 350 | 35 | 1000 | 100 | 0.0005 |
| Moraxellaceae | comp | 10.36593 | 9.53861 | 11.18762 | 0.0005 | 10.68352 | 0.31759 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Lactobacillaceae | comp | 6.01203 | 5.04594 | 6.95726 | 0.0005 | 6.32961 | 0.31759 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Flavobacteriaceae | comp | 2.94637 | 2.24731 | 3.64783 | 0.0005 | 3.26395 | 0.31759 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Streptococcaceae | cont | 7.07789 | 6.32005 | 7.8604 | 0.0005 | 7.93953 | 0.86164 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Pseudomonadaceae | cont | 8.37873 | 7.49416 | 9.32414 | 0.0005 | 9.24037 | 0.86164 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Bacteroidaceae | cont | 7.77015 | 6.61982 | 8.91203 | 0.0005 | 8.63179 | 0.86164 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Helicobacteraceae | cont | 8.1279 | 7.32426 | 8.93943 | 0.0005 | 8.98954 | 0.86164 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Bifidobacteriaceae | cont | 4.55011 | 3.9048 | 5.20612 | 0.0005 | 5.41175 | 0.86164 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Leptospiraceae | cont | 16.9708 | 14.90404 | 18.97402 | 0.0005 | 17.83244 | 0.86164 | 370 | 37 | 1000 | 100 | 0.0005 |
| Cyanobiaceae | cont | 6.07207 | 4.59912 | 7.67655 | 0.0005 | 6.93371 | 0.86164 | 350 | 35 | 1000 | 100 | 0.0005 |
| Moraxellaceae | cont | 16.27112 | 14.78985 | 17.76668 | 0.0005 | 17.13276 | 0.86164 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Lactobacillaceae | cont | 6.50936 | 5.48303 | 7.6724 | 0.0005 | 7.371 | 0.86164 | 1000 | 100 | 1000 | 100 | 0.0005 |
| Flavobacteriaceae | cont | 3.65868 | 2.80528 | 4.52307 | 0.0005 | 4.52032 | 0.86164 | 1000 | 100 | 1000 | 100 | 0.0005 |

| group | tool | n_refs | comp_mae | cont_mae | comp_bias | cont_bias |
| --- | --- | --- | --- | --- | --- | --- |
| Streptococcaceae | MAGICC_holdout | 100 | 8.8376 | 11.6898 | -5.7363 | 5.7156 |
| Streptococcaceae | MAGICC_matched_full | 100 | 3.4052 | 3.7503 | 2.7244 | -2.5974 |
| Streptococcaceae | MAGICC_V5 | 100 | 3.0741 | 4.0477 | 2.0015 | -3.1718 |
| Pseudomonadaceae | MAGICC_holdout | 100 | 16.7549 | 13.4321 | 16.112 | 12.3641 |
| Pseudomonadaceae | MAGICC_matched_full | 100 | 5.2803 | 4.1918 | 4.4264 | -1.514 |
| Pseudomonadaceae | MAGICC_V5 | 100 | 4.0886 | 4.5002 | 2.1159 | -2.3039 |
| Bacteroidaceae | MAGICC_holdout | 100 | 12.4378 | 13.8993 | 8.513 | 9.5245 |
| Bacteroidaceae | MAGICC_matched_full | 100 | 6.2622 | 5.2675 | 4.3771 | -2.8244 |
| Bacteroidaceae | MAGICC_V5 | 100 | 5.8061 | 5.6768 | 3.5562 | -3.9664 |
| Helicobacteraceae | MAGICC_holdout | 100 | 8.719 | 13.0643 | -2.9872 | -10.4394 |
| Helicobacteraceae | MAGICC_matched_full | 100 | 2.758 | 4.0748 | 1.2753 | -2.9282 |
| Helicobacteraceae | MAGICC_V5 | 100 | 2.5886 | 4.4698 | 0.5895 | -3.753 |
| Bifidobacteriaceae | MAGICC_holdout | 100 | 11.1973 | 10.8266 | -9.222 | -4.7369 |
| Bifidobacteriaceae | MAGICC_matched_full | 100 | 4.2179 | 5.4148 | 2.9382 | -3.4591 |
| Bifidobacteriaceae | MAGICC_V5 | 100 | 3.9334 | 5.5685 | 2.2944 | -3.7225 |
| Leptospiraceae | MAGICC_holdout | 37 | 16.3113 | 22.8479 | 12.0636 | 20.0625 |
| Leptospiraceae | MAGICC_matched_full | 37 | 6.1496 | 5.0155 | 4.0462 | -0.2916 |
| Leptospiraceae | MAGICC_V5 | 37 | 5.1408 | 4.9258 | 1.88 | -0.5786 |
| Cyanobiaceae | MAGICC_holdout | 35 | 14.0002 | 14.2775 | -10.9118 | -9.6815 |
| Cyanobiaceae | MAGICC_matched_full | 35 | 5.8752 | 7.3438 | 0.6062 | -4.4628 |
| Cyanobiaceae | MAGICC_V5 | 35 | 5.758 | 7.598 | 0.0135 | -5.1312 |
| Moraxellaceae | MAGICC_holdout | 100 | 15.9941 | 21.194 | 9.6353 | 16.4788 |
| Moraxellaceae | MAGICC_matched_full | 100 | 5.3106 | 4.0612 | 4.1813 | -1.5694 |
| Moraxellaceae | MAGICC_V5 | 100 | 4.6887 | 4.6867 | 3.3729 | -3.0781 |
| Lactobacillaceae | MAGICC_holdout | 100 | 11.0657 | 12.9049 | 0.3384 | 1.362 |
| Lactobacillaceae | MAGICC_matched_full | 100 | 4.7361 | 5.5339 | 1.9606 | -3.2396 |
| Lactobacillaceae | MAGICC_V5 | 100 | 4.4907 | 5.8974 | 1.3813 | -3.9221 |
| Flavobacteriaceae | MAGICC_holdout | 100 | 10.5772 | 10.9881 | 4.9738 | 2.7566 |
| Flavobacteriaceae | MAGICC_matched_full | 100 | 7.3132 | 6.4678 | 4.8438 | -1.9346 |
| Flavobacteriaceae | MAGICC_V5 | 100 | 6.6648 | 6.592 | 3.3558 | -2.8434 |
| in_distribution | MAGICC_holdout | 100 | 6.9346 | 7.4717 | 1.1638 | -3.1328 |
| in_distribution | MAGICC_matched_full | 100 | 6.617 | 6.6101 | 1.8368 | -2.1345 |
| in_distribution | MAGICC_V5 | 100 | 6.3374 | 6.7191 | 0.6816 | -2.8653 |
