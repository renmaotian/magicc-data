# WS3.10 mitigation diagnostic — mechanism, tractability and the boundary

**Model status: `models/magicc_v5.onnx` is UNCHANGED.** Every result below is either a read-only probe of the frozen model or a post-hoc post-processing function evaluated on disjoint data. No V6 exists.

Scripts `140`–`144`; all outputs under `results/revision/real_data/reduced_genome/mitigation/`. R² is the coefficient of determination throughout; all CIs are 5,000-iteration bootstraps resampling **reference genomes** as clusters; seeds via `fw.stable_hash` under `PYTHONHASHSEED=0`.

## 1. The mechanism, demonstrated causally on the frozen model

Holding the k-mer **composition** of a real assembly exactly fixed and rescaling its absolute size by a factor *s* (binomial thinning for *s*<1, exact multiplication for *s*>1; n=3996 samples, 2604 reference genomes from V5's own held-out test split), define

    phi = d log(predicted completeness) / d log(s)

An estimator whose belief about the reference length depended on composition **alone** would give phi = 1. MAGICC V5 gives **phi = 0.699 [0.687, 0.711]** over ±0.07 in log10 size (0.805 on the subset that never touches the 50 %/100 % output bounds). The complement is the size channel: the elasticity of the model's *implied reference length* to the *observed* assembly size is **0.30** — a genome observed to be half the size of its composition-matched neighbours has its inferred reference length pulled down by only 19 %, and the rest of the shortfall is reported as missing sequence.

**Where the size enters is not where it was assumed to.** Decomposing the intervention by input branch: k-mer branch only phi = 0.644; the 7 k-mer summary features only phi = 0.057. MAGICC's k-mer features are `log1p(absolute count)` z-scored, **not** relative frequencies, so absolute size is carried by all 9,249 of them. The summary features — although `log10_total_kmer_count` correlates r = 0.979 with log10 assembly bp — contribute only ~8 % of the effect. phi is essentially flat across reference-size strata (0.64–0.72 from <1 Mbp to >5 Mbp): the size channel is a **uniform property of the estimator**, not something that switches on for small genomes.

## 2. What that channel does on ground truth, and what it does not explain

On the five leakage-free ground-truth sets (5,000 samples, 1,648 reference genomes), holding **true** completeness *and* true contamination fixed, MAGICC's predicted completeness still rises with reference length in the clean and near-complete regime (true contamination <5% AND true completeness >=90%): **+1.14 pp per SD of log10 reference size [0.508, 1.767], partial R² 0.0364**, versus CheckM2 +0.24 [0.131, 0.342], partial R² 0.0107. The unbiased value is exactly 0, so both tools leak, but MAGICC leaks 3.4× more of it.

The estimator's **implied reference length** (ground-truth retained bp ÷ predicted completeness) makes the same point in units of genome length (clean subset, mean ratio to the true reference length):

| reference size | n | refs | MAGICC implied/true | CheckM2 implied/true |
|---|---|---|---|---|
| <1 Mbp | 62 | 46 | 1.093 [1.047, 1.144] | 0.997 [0.978, 1.017] |
| 1-1.5 Mbp | 86 | 62 | 1.017 [1.002, 1.032] | 1.007 [0.994, 1.019] |
| 1.5-2 Mbp | 142 | 105 | 1.022 [1.005, 1.045] | 1.000 [0.992, 1.008] |
| 2-3 Mbp | 364 | 250 | 1.000 [0.995, 1.005] | 0.985 [0.979, 0.991] |
| 3-5 Mbp | 498 | 353 | 1.000 [0.993, 1.008] | 0.993 [0.988, 0.997] |
| >5 Mbp | 414 | 296 | 0.991 [0.986, 0.996] | 0.969 [0.964, 0.973] |

**A correction to the WS3.10 mechanism claim that must be carried into the manuscript.** The previously reported partial R² of 0.313 for genome size was estimated on real MAGs, where true completeness is unknown and therefore uncontrolled — and a larger assembly of a given lineage genuinely *is* more complete. Once true completeness is controlled on ground-truthed data the partial R² of size falls to 0.036. The **direction** and the **contrast with CheckM2** survive; the **magnitude** does not. Report the controlled figure and the causal phi, not 0.313.

## 3. Two components, not one — and only one of them is about size

In the clean and near-complete regime, regressing MAGICC's signed completeness error jointly on z(log10 reference size) and a Patescibacteriota indicator (n=543, 482 reference clusters, R² = 0.111) gives size **+0.68 pp/SD [0.145, 1.214]** *and* Patescibacteriota **-5.77 pp [-11.006, -0.532]**. Neither term absorbs the other. Stratified, sub-megabase Patescibacteriota carry a completeness bias of −11.0 pp [−17.4, −4.8] and a contamination bias of +5.6 pp [1.5, 9.9], while sub-megabase genomes from *other* phyla (Bacillota, Pseudomonadota) carry −3.2 pp [−9.3, −0.1] and +1.0 pp [0.0, 1.9].

So the reduced-genome failure decomposes into (i) a **uniform size channel**, causally demonstrated and worth a few pp, and (ii) a **lineage-specific composition→reference-length prior error**, roughly three times larger, which is not a function of size at all. This is the quantitative reason the leave-phylum-out experiment (WS1.6) and WS3.10 converge without being the same effect: novelty attacks component (ii), and component (ii) dominates.

## 3b. The completeness under-call and the contamination over-call are largely one event

On clean, near-complete genomes with reference <2 Mbp (n=153, 124 reference clusters) MAGICC's signed completeness error and signed contamination error are strongly **negatively** correlated (Spearman ρ = -0.492 [-0.647, -0.302]); regressing the contamination error on the completeness error leaves an intercept of only +0.60 pp [0.228, 0.981] out of a mean over-call of +2.42 pp. Mechanically this is what one expects: faced with an assembly shorter than the reference length it infers, the model must split the observed sequence into 'dominant' and 'foreign', and one decision simultaneously lowers completeness and raises contamination. It is also why a recalibrator that corrects the two outputs as separate functions of size fixes part of the completeness error and none of the contamination error.

## 4. Is it correctable post hoc? Partly, and only where the lineage is known

A size-conditioned recalibrator — inputs restricted to what is observable at inference (predicted completeness, predicted contamination, log10 assembly Mbp, log10 contig count) — was fitted on the residual and evaluated with fit and test always disjoint and clustered by reference genome.

| protocol | stratum | n | comp MAE raw → recal | ΔMAE [95 % CI] | comp bias raw → recal |
|---|---|---|---|---|---|
| CV-A reference-disjoint | ALL | 5000 | 4.618 → 4.468 | -0.150 [-0.262, -0.041] | -0.867 → -0.087 |
| CV-A reference-disjoint | reduced <2 Mbp | 1911 | 5.777 → 5.649 | -0.128 [-0.386, 0.120] | -1.955 → -0.524 |
| CV-A reference-disjoint | normal ≥2 Mbp | 3089 | 3.901 → 3.737 | -0.164 [-0.257, -0.074] | -0.194 → +0.183 |
| CV-A reference-disjoint | ANCHOR set_C_clean clean+HQ | 30 | 9.382 → 6.307 | -3.074 [-5.190, -1.077] | -8.677 → -5.659 |
| CV-B phylum-disjoint | ALL | 4888 | 4.544 → 4.565 | +0.021 [-0.065, 0.110] | |
| CV-C set-disjoint | normal ≥2 Mbp | 3089 | | +0.708 [0.582, 0.839] | |

**Where it helps.** With the lineage represented in the fitting data (CV-A), the correction is real but small: pooled completeness MAE 4.62 → 4.47 pp (-0.150 [-0.262, -0.041]), contamination MAE 5.31 → 4.95 pp (-0.361 [-0.483, -0.232]). On the ground-truthed reduced-genome anchor — the 30 genuinely clean, high-quality Patescibacteriota of `set_C_clean` that produced the WS3.10 headline −8.68 / +5.09 pp — the completeness bias improves -8.68 → -5.66 pp (MAE -3.074 [-5.190, -1.077]), i.e. about 35 % of the under-call is removed.

**Where it does not.** The contamination over-call on the same anchor is **not corrected at all** (+5.09 → +5.54 pp; ΔMAE +0.281 [-0.111, 0.673]), and the false-fail rate at the 5 % MIMAG contamination threshold on `set_C_clean` is **unchanged at 0.4423 → 0.4423**. The headline harm is untouched.

**Does it degrade normal-size genomes?** Not in aggregate — normal (≥2 Mbp) completeness MAE 3.901 → 3.737 (-0.164 [-0.257, -0.074]). But the error is **relocated, not only removed**: several normal-size strata degrade significantly under CV-A (e.g. reference 2–3 Mbp, `set_D_clean`, Bacillota, Thermoplasmatota; see `recalibration_cv_results.tsv`), the completeness bias flips from negative to positive above 3 Mbp, and under the stricter set-disjoint protocol normal-size genomes degrade outright (+0.708 [0.582, 0.839] pp MAE). Under phylum-disjoint validation the benefit disappears entirely (+0.021 [-0.065, 0.110]).

## 5. The ceiling: it does not generalise to genuinely novel lineages

**First, a bigger fitting corpus does not help — it hurts.** Refitting the same recalibrator on 48,784 ground-truthed samples from V5's own held-out test split (9,706 reference genomes, verified zero reference overlap with the benchmark sets) and applying it to the five benchmark sets makes pooled completeness MAE **worse**: 4.618 → 4.991 pp (+0.373 [0.279, 0.465]), even though the reduced-genome *bias* still improves (-1.96 → -0.84 pp). The residual surface is not stable across data distributions, so the correction cannot simply be trained once on a large corpus and shipped. The modest CV-A gain in §4 is the ceiling for a size-conditioned correction, and it requires fitting on data drawn from the same distribution as the target.

The decisive test uses the WS1.6 leave-phylum-out model, which genuinely never saw six phyla. The recalibrator was fitted on **47,015 samples (7,901 reference genomes)** of that same model's predictions over its own in-distribution test split — verified to contain 0 samples from any panel phylum — and then applied to that model's predictions on the six novel phyla. For comparison, an **oracle** recalibrator was fitted *on each novel lineage itself* with reference-disjoint 5-fold CV, i.e. what a correction could achieve if labelled data from that lineage existed.

| held-out lineage | n | raw comp bias | comp MAE raw → transfer → oracle | bias removed by transfer | bias removed by oracle |
|---|---|---|---|---|---|
| Patescibacteriota | 1000 | -30.00 pp | 30.02 → 25.94 → 11.67 | +4.50 pp (15 %) | +29.73 pp (99 %) |
| DPANN | 996 | -23.65 pp | 23.95 → 21.20 → 11.69 | +3.20 pp (14 %) | +23.10 pp (98 %) |
| Bacteroidota_A | 990 | -1.13 pp | 12.45 → 12.45 → 9.66 | -0.69 pp (-61 %) | +0.97 pp (85 %) |
| in_distribution | 1000 | +0.79 pp | 6.20 → 6.14 → 6.54 | +0.41 pp (52 %) | +0.70 pp (89 %) |
| Campylobacterota | 1000 | +0.74 pp | 6.58 → 6.99 → 5.93 | -0.58 pp (-78 %) | +0.73 pp (99 %) |
| Halobacteriota | 1008 | -0.73 pp | 10.84 → 10.64 → 9.94 | -0.20 pp (-28 %) | +0.61 pp (83 %) |
| Bacteroidota | 1000 | -0.56 pp | 13.44 → 12.97 → 10.70 | -0.39 pp (-69 %) | +0.38 pp (68 %) |

Percentages are meaningless where the raw bias is already near zero; the two reduced-genome groups (Patescibacteriota, DPANN) are the ones the correction is supposed to rescue.

This is the most important result in this workstream and it is negative. A size-conditioned correction learned on lineages the model knows transfers almost none of the way to lineages it does not. The oracle shows the information is *present in the observables* for a lineage you have labels for — so the failure is not that assembly size is uninformative — but the mapping from those observables to the correction is **lineage-specific**. A post-hoc size correction is therefore a **learned prior over lineages, not a mechanism**, and it cannot be relied on exactly where it would be most valuable: on the novel, uncultured, reduced-genome lineages that motivate MAG quality control in the first place.

Two further details sharpen the point. The oracle is not merely better, it is **almost complete**: it drives the Patescibacteriota completeness bias from −30.00 pp to −0.27 pp and DPANN from −23.65 pp to −0.54 pp, so essentially all of the novel-lineage error is recoverable from the observables *once you know which lineage you are looking at*. And blind transfer actively **harms lineages that did not need correcting** — Campylobacterota +0.74 → +1.32 pp bias, Bacteroidota_A −1.13 → −1.82 pp — so it cannot be applied by default either.

## 6. Manuscript-ready boundary statement

> MAGICC V5's completeness target is *retained bases ÷ reference genome length*, and the reference length is not observable at inference: it must be inferred from k-mer composition. Two things follow, and we demonstrate both. First, the inferred reference length is not a function of composition alone — an intervention on the frozen model that holds composition fixed and rescales absolute assembly size shows that 30% of an observed size change is absorbed into the inferred reference length (phi = 0.70 [0.687, 0.711]), a leak carried by the un-normalised k-mer count features rather than by the k-mer summary statistics. Second, and larger, the composition→length map is itself biased for lineages whose genomes are atypically small for their composition. On ground-truthed, genuinely clean, near-complete genomes the two components are separately identifiable (size +0.68 pp/SD [0.145, 1.214]; Patescibacteriota -5.77 pp [-11.006, -0.532]), and the lineage term dominates. Accordingly, **MAGICC V5 is calibrated for genomes ≳2 Mbp whose lineage is represented in GTDB r220. On ground truth the severe regime is sub-megabase references, where completeness is under-called by 9–11 pp and contamination over-called by ~5–6 pp; between 1 and 2 Mbp the under-call is smaller but still non-zero (0.8–2.3 pp). If the lineage is additionally absent from training the completeness under-call reaches 24–30 pp.** (Relative to CheckM2 on real catalogue MAGs the corresponding figures are −12.6 pp completeness in the <1 Mb bin and +13 to +19 pp contamination on the reviewer's genera; WS3.10.) We show that a post-hoc, size-conditioned recalibration removes about 35 % of the ground-truthed completeness under-call when the lineage is represented, leaves the contamination over-call and the 44% false-fail rate at the 5 % MIMAG threshold unchanged, and removes 14–15 % of the bias on genuinely novel reduced-genome lineages while an oracle fitted on the novel lineage itself removes 98–99 %. The failure is therefore mechanistically understood and correctable **in principle**, but not by any post-processing that lacks lineage information — which is why we report it as a boundary rather than shipping a patched model.

## 7. What a future model would have to do differently

1. **Close the size leak in the k-mer branch.** The intervention localises it precisely: k-mer inputs are `log1p(absolute count)` z-scored. Passing relative frequencies (composition) to the k-mer branch, and absolute size only through an explicit, separately modelled channel, would make phi = 1 by construction and leave the reference-length belief a pure function of composition.
2. **Predict reference length as an auxiliary target.** Completeness is a ratio whose denominator the model currently estimates implicitly and never reports. Multi-task training with genome length as a second head makes the composition→length map trainable, diagnosable and reportable, and lets completeness be emitted as retained bp ÷ L̂ with an uncertainty on L̂.
3. **Condition explicitly on lineage-relative size.** The oracle result shows the correction exists but is lineage-indexed; a taxonomic assignment (or a learned lineage embedding) feeding an expected-genome-size prior is the natural carrier. This is effectively what makes CheckM2's marker route size-insensitive here.
4. **Do not expect more training data alone to fix it.** V5 already contains 1,609 Patescibacteriota and 5 % reduced-genome samples. The failure is in the *form* of the estimator, not the sampling of the training set.

## 8. Files

```
results/revision/real_data/reduced_genome/mitigation/BOUNDARY_STATEMENT.md
results/revision/real_data/reduced_genome/mitigation/fit_corpus_HOLDOUT.tsv.gz
results/revision/real_data/reduced_genome/mitigation/fit_corpus_V5.tsv.gz
results/revision/real_data/reduced_genome/mitigation/intervention_phi_by_reference_size.tsv
results/revision/real_data/reduced_genome/mitigation/intervention_phi_pooled.tsv
results/revision/real_data/reduced_genome/mitigation/intervention_size_channel.tsv
results/revision/real_data/reduced_genome/mitigation/intervention_summary.json
results/revision/real_data/reduced_genome/mitigation/mechanism_m1_binned_signed_error.tsv
results/revision/real_data/reduced_genome/mitigation/mechanism_m1_matched_completeness.tsv
results/revision/real_data/reduced_genome/mitigation/mechanism_m2_implied_length_by_stratum.tsv
results/revision/real_data/reduced_genome/mitigation/mechanism_m2_implied_reference_length.tsv
results/revision/real_data/reduced_genome/mitigation/mechanism_m3_feature_size_channel.tsv
results/revision/real_data/reduced_genome/mitigation/mechanism_m4_regression.tsv
results/revision/real_data/reduced_genome/mitigation/mechanism_m4_size_vs_lineage.tsv
results/revision/real_data/reduced_genome/mitigation/mechanism_m5_comp_cont_coupling.tsv
results/revision/real_data/reduced_genome/mitigation/mechanism_summary.json
results/revision/real_data/reduced_genome/mitigation/mitigation_headline.json
results/revision/real_data/reduced_genome/mitigation/pooled_groundtruth_predictions.tsv
results/revision/real_data/reduced_genome/mitigation/recalibration_ceiling.tsv
results/revision/real_data/reduced_genome/mitigation/recalibration_ceiling_summary.json
results/revision/real_data/reduced_genome/mitigation/recalibration_cv_results.tsv
results/revision/real_data/reduced_genome/mitigation/recalibration_false_fail.tsv
results/revision/real_data/reduced_genome/mitigation/recalibration_novel_lineage.tsv
results/revision/real_data/reduced_genome/mitigation/recalibration_summary.json
```

Scripts: `scripts/140_mitigation_mechanism.py`, `scripts/141_size_channel_intervention.py`, `scripts/142_size_conditioned_recalibration.py`, `scripts/143_recalibration_generalization_ceiling.py`, `scripts/144_mitigation_boundary_statement.py`.
