# WS11.N — the taxonomic-novelty ladder on the headline five-set panel

Generated 2026-08-26T17:33:48.019101+00:00 · `scripts/210_ws11n_novelty_ladder.py` · 2000 cluster-bootstrap resamples · `PYTHONHASHSEED=0`

> **This is an observational stratification, not a holdout experiment.**
> Every number below comes from the *same frozen models* scoring the *same*
> five leakage-free benchmark sets that the manuscript already reports. No
> genome was removed from training for this analysis; genomes in the deeper
> novelty classes simply happen to have no training relative at that rank.
> Nothing here may be described as "held out at species/genus level". The
> retrained validation artefacts (WS1.6 phylum, WS1.9 family, WS11.G genus)
> remain the only holdout evidence. What this analysis *does* deliver is the
> genome → species → genus ladder measured on exactly the panel Reviewer 1
> was reading, on identical genomes for all four tools.

## 0. Headline findings

* **Ladder is real.** MAGICC v5 completeness MAE climbs 3.17 pp (lineage_represented) → 6.33 pp (species_novel) → 10.27 pp (genus_novel); contamination MAE climbs 4.02 pp → 7.00 pp → 9.86 pp. Adequately powered classes only (≥ 20 clusters).
* **Trend, MAGICC v5 completeness.** Spearman ρ between novelty depth and cluster-mean absolute error = 0.297 [0.260, 0.331], q = 9.3e-53; monotone over the powered classes: True.
* **Trend, MAGICC v5 contamination.** Spearman ρ between novelty depth and cluster-mean absolute error = 0.208 [0.168, 0.248], q = 3.6e-26; monotone over the powered classes: True.
* **Trend, CheckM2 completeness.** Spearman ρ between novelty depth and cluster-mean absolute error = 0.082 [0.044, 0.121], q = 3.1e-05; monotone over the powered classes: True.
* **Trend, CheckM2 contamination.** Spearman ρ between novelty depth and cluster-mean absolute error = 0.103 [0.067, 0.142], q = 1.9e-07; monotone over the powered classes: True.
* **MAGICC vs CheckM2, completeness.** MAGICC significantly better in `lineage_represented`; significantly worse in `species_novel`, `genus_novel`, `family_novel_or_deeper (pooled)` (BH-corrected, cluster-mean paired Wilcoxon).
* **MAGICC vs CheckM2, contamination.** MAGICC significantly better in `lineage_represented`, `species_novel`, `genus_novel`, `family_novel`, `order_novel`, `family_novel_or_deeper (pooled)`; significantly worse in no class (BH-corrected, cluster-mean paired Wilcoxon).
* **Limits of this panel.** `phylum_novel` does not occur at all on this panel, so it cannot be measured here. `family_novel`, `order_novel`, `class_novel` carry fewer than 20 reference clusters and are reported but must not carry a claim on their own; they are pooled as `family_novel_or_deeper (pooled)`.

## 1. Design

For every dominant reference genome of `set_A_v2`, `set_B_v2`, `set_C_clean`,
`set_D_clean` and `set_E` the GTDB lineage is walked from species upwards and
compared against the taxa present in `data/splits/train_genomes.tsv`
(79,948 genomes). The deepest represented rank fixes the
novelty class:

| class | definition |
|---|---|
| `lineage_represented` | the genome's **species** is present in training (the genome itself is still held out — this is the genome-level-only case) |
| `species_novel` | species absent, **genus** present |
| `genus_novel` | genus absent, **family** present |
| `family_novel` | family absent, **order** present |
| `order_novel` | order absent, **class** present |
| `class_novel` | class absent, **phylum** present |
| `phylum_novel` | phylum absent from training |

Accession matching uses the GCA↔GCF cross-map of the WS1.4 provenance audit
(`scripts/74_provenance_audit.py`), rebuilt here and verified byte-for-byte
against `results/revision/provenance/accession_crossmap_stats.json` (match: True).
GTDB polyphyly suffixes (`Bacteroidota_A`) are treated as distinct taxa, consistent with WS1.6.

**Denominators.** Denominators: completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

## 2. Class balance — reported honestly

Reference clusters (set × dominant reference genome):

| set         | lineage_represented | species_novel | genus_novel | family_novel | order_novel | class_novel | phylum_novel |
|-------------|---------------------|---------------|-------------|--------------|-------------|-------------|--------------|
| set_A_v2    | 643                 | 129           | 21          | 3            | 1           | 1           | 0            |
| set_B_v2    | 625                 | 146           | 28          | 2            | 2           | 0           | 0            |
| set_C_clean | 32                  | 33            | 21          | 10           | 4           | 0           | 0            |
| set_D_clean | 64                  | 29            | 5           | 1            | 1           | 0           | 0            |
| set_E       | 631                 | 132           | 18          | 2            | 2           | 0           | 0            |

Genomes:

| set         | lineage_represented | species_novel | genus_novel | family_novel | order_novel | class_novel | phylum_novel |
|-------------|---------------------|---------------|-------------|--------------|-------------|-------------|--------------|
| set_A_v2    | 799                 | 170           | 25          | 3            | 2           | 1           | 0            |
| set_B_v2    | 775                 | 186           | 32          | 5            | 2           | 0           | 0            |
| set_C_clean | 320                 | 330           | 210         | 100          | 40          | 0           | 0            |
| set_D_clean | 640                 | 290           | 50          | 10           | 10          | 0           | 0            |
| set_E       | 807                 | 165           | 22          | 4            | 2           | 0           | 0            |

**Too thin to support a claim** (< 20 reference clusters pooled across all five sets): `family_novel` (18 clusters, 122 genomes), `order_novel` (10 clusters, 56 genomes), `class_novel` (1 clusters, 1 genomes)

**Absent entirely**: `phylum_novel`. The five-set panel therefore *cannot* speak to phylum-level novelty at all;
that level is only measurable by retraining (WS1.6).

Sensitivity: stripping GTDB polyphyly suffixes moves 47/2586 cluster labels; scoring novelty against
train ∪ val instead of train alone moves 47. Both are reported as
columns of `dominant_novelty_classification.tsv`; neither is the primary definition.

## 3. Pooled five-set metrics by novelty class

All intervals are 95% percentile cluster bootstraps over reference genomes (2000 resamples, clusters = set × dominant accession).

### Completeness (MAE and signed bias, pp)

| novelty class                   | n    | clusters | MAGICC v5 MAE       | MAGICC v5 bias        | CheckM2 MAE       | CheckM2 bias         | CoCoPyE MAE          | CoCoPyE bias            | DeepCheck MAE        | DeepCheck bias          |
|---------------------------------|------|----------|---------------------|-----------------------|-------------------|----------------------|----------------------|-------------------------|----------------------|-------------------------|
| lineage_represented             | 3341 | 1995     | 3.17 [2.98, 3.35]   | +0.30 [0.07, 0.55]    | 5.41 [5.05, 5.78] | +3.38 [2.92, 3.85]   | 5.09 [4.67, 5.53]    | -1.02 [-1.51, -0.52]    | 8.53 [8.08, 8.99]    | +3.23 [2.47, 4.01]      |
| species_novel                   | 1141 | 469      | 6.33 [5.71, 6.93]   | -1.88 [-2.94, -0.89]  | 6.24 [5.64, 6.79] | +1.90 [0.95, 2.88]   | 8.18 [7.22, 9.12]    | -4.63 [-5.78, -3.37]    | 10.92 [9.88, 11.90]  | -1.05 [-2.80, 0.83]     |
| genus_novel                     | 339  | 93       | 10.27 [8.33, 12.30] | -6.14 [-8.92, -3.19]  | 7.47 [6.48, 8.40] | -0.90 [-2.84, 1.45]  | 12.55 [10.84, 14.17] | -9.40 [-11.53, -6.91]   | 15.85 [13.76, 17.81] | -9.70 [-13.12, -5.50]   |
| family_novel                    | 122  | 18       | 10.50 [6.93, 13.60] | -7.38 [-12.26, -1.95] | 7.22 [6.29, 8.03] | -4.30 [-5.81, -2.77] | 14.71 [12.50, 16.83] | -12.91 [-15.43, -10.02] | 17.25 [13.87, 19.82] | -15.64 [-18.91, -11.09] |
| order_novel                     | 56   | 10       | 9.09 [7.13, 11.31]  | -3.91 [-9.30, 3.16]   | 7.17 [4.79, 9.27] | +0.30 [-2.89, 3.50]  | 9.73 [6.78, 11.82]   | -7.84 [-10.28, -4.82]   | 12.84 [7.61, 17.72]  | -11.72 [-16.77, -5.30]  |
| class_novel                     | 1    | 1        | 2.37 [2.37, 2.37]   | +2.37 [2.37, 2.37]    | 3.07 [3.07, 3.07] | +3.07 [3.07, 3.07]   | 10.87 [10.87, 10.87] | +10.87 [10.87, 10.87]   | 4.24 [4.24, 4.24]    | -4.24 [-4.24, -4.24]    |
| family_novel_or_deeper (pooled) | 179  | 29       | 10.01 [7.45, 12.42] | -6.24 [-10.38, -1.94] | 7.18 [6.17, 8.01] | -2.82 [-4.51, -0.92] | 13.13 [11.11, 15.05] | -11.19 [-13.48, -8.80]  | 15.80 [12.79, 18.23] | -14.35 [-17.30, -10.65] |
| ALL (any novelty)               | 5000 | 2586     | 4.62 [4.33, 4.92]   | -0.87 [-1.25, -0.48]  | 5.80 [5.53, 6.08] | +2.53 [2.07, 2.98]   | 6.59 [6.16, 7.01]    | -2.78 [-3.30, -2.24]    | 9.83 [9.37, 10.28]   | +0.75 [-0.08, 1.63]     |

### Contamination (MAE and signed bias, pp)

| novelty class                   | n    | clusters | MAGICC v5 MAE        | MAGICC v5 bias       | CheckM2 MAE          | CheckM2 bias            | CoCoPyE MAE          | CoCoPyE bias            | DeepCheck MAE        | DeepCheck bias          |
|---------------------------------|------|----------|----------------------|----------------------|----------------------|-------------------------|----------------------|-------------------------|----------------------|-------------------------|
| lineage_represented             | 3341 | 1995     | 4.02 [3.78, 4.25]    | -2.05 [-2.35, -1.74] | 19.00 [17.74, 20.21] | -17.77 [-19.05, -16.42] | 16.86 [15.99, 17.75] | -15.98 [-16.90, -15.09] | 23.46 [22.17, 24.71] | -20.67 [-22.08, -19.19] |
| species_novel                   | 1141 | 469      | 7.00 [6.31, 7.69]    | -1.80 [-2.91, -0.67] | 25.95 [23.63, 27.96] | -24.82 [-26.95, -22.35] | 21.66 [19.95, 23.20] | -20.67 [-22.22, -18.90] | 30.48 [28.21, 32.45] | -27.00 [-29.44, -24.26] |
| genus_novel                     | 339  | 93       | 9.86 [7.98, 11.78]   | -2.64 [-5.18, 0.04]  | 34.26 [30.43, 37.51] | -33.89 [-37.22, -29.93] | 27.69 [24.66, 30.52] | -26.20 [-29.34, -22.77] | 37.10 [33.22, 40.40] | -36.64 [-40.09, -32.64] |
| family_novel                    | 122  | 18       | 11.81 [10.27, 13.76] | +0.64 [-4.76, 6.30]  | 34.12 [29.12, 38.14] | -33.84 [-37.97, -28.61] | 26.13 [22.05, 29.61] | -25.02 [-28.83, -20.67] | 36.58 [31.90, 40.43] | -36.25 [-40.18, -31.21] |
| order_novel                     | 56   | 10       | 6.37 [5.12, 8.14]    | -2.00 [-3.28, 0.87]  | 33.09 [23.64, 39.57] | -32.94 [-39.50, -23.20] | 25.65 [18.78, 31.17] | -24.47 [-30.63, -16.73] | 35.39 [25.83, 41.97] | -35.39 [-41.97, -25.83] |
| class_novel                     | 1    | 1        | 0.16 [0.16, 0.16]    | +0.16 [0.16, 0.16]   | 0.07 [0.07, 0.07]    | +0.07 [0.07, 0.07]      | 4.28 [4.28, 4.28]    | +4.28 [4.28, 4.28]      | 0.07 [0.07, 0.07]    | +0.07 [0.07, 0.07]      |
| family_novel_or_deeper (pooled) | 179  | 29       | 10.05 [8.47, 11.77]  | -0.19 [-3.79, 3.80]  | 33.61 [29.43, 37.12] | -33.37 [-36.96, -29.03] | 25.85 [22.50, 28.66] | -24.68 [-27.81, -20.99] | 36.00 [31.93, 39.41] | -35.78 [-39.23, -31.52] |
| ALL (any novelty)               | 5000 | 2586     | 5.31 [5.01, 5.62]    | -1.96 [-2.36, -1.56] | 22.14 [21.00, 23.14] | -21.03 [-22.10, -19.84] | 19.02 [18.21, 19.76] | -18.06 [-18.82, -17.24] | 26.43 [25.30, 27.44] | -23.74 [-24.86, -22.44] |

### MIMAG-inspired 5% contamination boundary

`false_fail` = a truly clean genome (true contamination < 5%) rejected; denominator = truly clean genomes.  
`false_pass` = a truly contaminated genome (true contamination ≥ 5%) let through; denominator = truly contaminated genomes.

| novelty class                   | tool      | false-fail           | false-fail denominator                            | false-pass           | false-pass denominator                              | balanced acc.        |
|---------------------------------|-----------|----------------------|---------------------------------------------------|----------------------|-----------------------------------------------------|----------------------|
| lineage_represented             | MAGICC v5 | 0.006 [0.002, 0.010] | 7/1212 genomes whose TRUE contamination is < 5%   | 0.013 [0.008, 0.017] | 27/2129 genomes whose TRUE contamination is >= 5%   | 0.991 [0.988, 0.994] |
| lineage_represented             | CheckM2   | 0.002 [0.000, 0.004] | 2/1212 genomes whose TRUE contamination is < 5%   | 0.206 [0.179, 0.230] | 439/2129 genomes whose TRUE contamination is >= 5%  | 0.896 [0.884, 0.910] |
| lineage_represented             | CoCoPyE   | 0.040 [0.029, 0.052] | 49/1212 genomes whose TRUE contamination is < 5%  | 0.020 [0.013, 0.028] | 42/2129 genomes whose TRUE contamination is >= 5%   | 0.970 [0.963, 0.977] |
| lineage_represented             | DeepCheck | 0.001 [0.000, 0.003] | 1/1212 genomes whose TRUE contamination is < 5%   | 0.340 [0.308, 0.370] | 723/2129 genomes whose TRUE contamination is >= 5%  | 0.830 [0.815, 0.846] |
| species_novel                   | MAGICC v5 | 0.130 [0.089, 0.175] | 37/285 genomes whose TRUE contamination is < 5%   | 0.009 [0.004, 0.016] | 8/856 genomes whose TRUE contamination is >= 5%     | 0.930 [0.908, 0.950] |
| species_novel                   | CheckM2   | 0.011 [0.000, 0.024] | 3/285 genomes whose TRUE contamination is < 5%    | 0.293 [0.254, 0.332] | 251/856 genomes whose TRUE contamination is >= 5%   | 0.848 [0.829, 0.869] |
| species_novel                   | CoCoPyE   | 0.098 [0.061, 0.140] | 28/285 genomes whose TRUE contamination is < 5%   | 0.019 [0.009, 0.030] | 16/856 genomes whose TRUE contamination is >= 5%    | 0.942 [0.920, 0.961] |
| species_novel                   | DeepCheck | 0.004 [0.000, 0.011] | 1/285 genomes whose TRUE contamination is < 5%    | 0.460 [0.412, 0.505] | 394/856 genomes whose TRUE contamination is >= 5%   | 0.768 [0.745, 0.792] |
| genus_novel                     | MAGICC v5 | 0.275 [0.133, 0.436] | 14/51 genomes whose TRUE contamination is < 5%    | 0.017 [0.004, 0.032] | 5/288 genomes whose TRUE contamination is >= 5%     | 0.854 [0.772, 0.925] |
| genus_novel                     | CheckM2   | 0.000 [0.000, 0.000] | 0/51 genomes whose TRUE contamination is < 5%     | 0.434 [0.347, 0.503] | 125/288 genomes whose TRUE contamination is >= 5%   | 0.783 [0.748, 0.826] |
| genus_novel                     | CoCoPyE   | 0.314 [0.167, 0.471] | 16/51 genomes whose TRUE contamination is < 5%    | 0.045 [0.021, 0.072] | 13/288 genomes whose TRUE contamination is >= 5%    | 0.821 [0.742, 0.896] |
| genus_novel                     | DeepCheck | 0.000 [0.000, 0.000] | 0/51 genomes whose TRUE contamination is < 5%     | 0.601 [0.513, 0.676] | 173/288 genomes whose TRUE contamination is >= 5%   | 0.700 [0.662, 0.744] |
| family_novel                    | MAGICC v5 | 0.750 [0.417, 1.000] | 9/12 genomes whose TRUE contamination is < 5%     | 0.027 [0.000, 0.055] | 3/110 genomes whose TRUE contamination is >= 5%     | 0.611 [0.495, 0.770] |
| family_novel                    | CheckM2   | 0.000 [0.000, 0.000] | 0/12 genomes whose TRUE contamination is < 5%     | 0.564 [0.408, 0.699] | 62/110 genomes whose TRUE contamination is >= 5%    | 0.718 [0.650, 0.796] |
| family_novel                    | CoCoPyE   | 0.583 [0.200, 0.882] | 7/12 genomes whose TRUE contamination is < 5%     | 0.055 [0.019, 0.094] | 6/110 genomes whose TRUE contamination is >= 5%     | 0.681 [0.533, 0.879] |
| family_novel                    | DeepCheck | 0.000 [0.000, 0.000] | 0/12 genomes whose TRUE contamination is < 5%     | 0.764 [0.648, 0.860] | 84/110 genomes whose TRUE contamination is >= 5%    | 0.618 [0.570, 0.676] |
| order_novel                     | MAGICC v5 | 0.200 [0.000, 1.000] | 1/5 genomes whose TRUE contamination is < 5%      | 0.020 [0.000, 0.057] | 1/51 genomes whose TRUE contamination is >= 5%      | 0.890 [0.486, 1.000] |
| order_novel                     | CheckM2   | 0.000 [0.000, 0.000] | 0/5 genomes whose TRUE contamination is < 5%      | 0.549 [0.419, 0.688] | 28/51 genomes whose TRUE contamination is >= 5%     | 0.725 [0.587, 0.788] |
| order_novel                     | CoCoPyE   | 0.600 [0.000, 1.000] | 3/5 genomes whose TRUE contamination is < 5%      | 0.098 [0.000, 0.267] | 5/51 genomes whose TRUE contamination is >= 5%      | 0.651 [0.480, 1.000] |
| order_novel                     | DeepCheck | 0.000 [0.000, 0.000] | 0/5 genomes whose TRUE contamination is < 5%      | 0.647 [0.514, 0.788] | 33/51 genomes whose TRUE contamination is >= 5%     | 0.676 [0.464, 0.741] |
| class_novel                     | MAGICC v5 | 0.000 [0.000, 0.000] | 0/1 genomes whose TRUE contamination is < 5%      | -                    | 0/0 genomes whose TRUE contamination is >= 5%       | 1.000 [1.000, 1.000] |
| class_novel                     | CheckM2   | 0.000 [0.000, 0.000] | 0/1 genomes whose TRUE contamination is < 5%      | -                    | 0/0 genomes whose TRUE contamination is >= 5%       | 1.000 [1.000, 1.000] |
| class_novel                     | CoCoPyE   | 0.000 [0.000, 0.000] | 0/1 genomes whose TRUE contamination is < 5%      | -                    | 0/0 genomes whose TRUE contamination is >= 5%       | 1.000 [1.000, 1.000] |
| class_novel                     | DeepCheck | 0.000 [0.000, 0.000] | 0/1 genomes whose TRUE contamination is < 5%      | -                    | 0/0 genomes whose TRUE contamination is >= 5%       | 1.000 [1.000, 1.000] |
| family_novel_or_deeper (pooled) | MAGICC v5 | 0.556 [0.263, 0.833] | 10/18 genomes whose TRUE contamination is < 5%    | 0.025 [0.006, 0.046] | 4/161 genomes whose TRUE contamination is >= 5%     | 0.710 [0.573, 0.856] |
| family_novel_or_deeper (pooled) | CheckM2   | 0.000 [0.000, 0.000] | 0/18 genomes whose TRUE contamination is < 5%     | 0.559 [0.438, 0.665] | 90/161 genomes whose TRUE contamination is >= 5%    | 0.720 [0.668, 0.781] |
| family_novel_or_deeper (pooled) | CoCoPyE   | 0.556 [0.263, 0.786] | 10/18 genomes whose TRUE contamination is < 5%    | 0.068 [0.023, 0.124] | 11/161 genomes whose TRUE contamination is >= 5%    | 0.688 [0.573, 0.839] |
| family_novel_or_deeper (pooled) | DeepCheck | 0.000 [0.000, 0.000] | 0/18 genomes whose TRUE contamination is < 5%     | 0.727 [0.629, 0.812] | 117/161 genomes whose TRUE contamination is >= 5%   | 0.637 [0.594, 0.685] |
| ALL (any novelty)               | MAGICC v5 | 0.043 [0.032, 0.056] | 68/1566 genomes whose TRUE contamination is < 5%  | 0.013 [0.009, 0.017] | 44/3434 genomes whose TRUE contamination is >= 5%   | 0.972 [0.966, 0.978] |
| ALL (any novelty)               | CheckM2   | 0.003 [0.001, 0.006] | 5/1566 genomes whose TRUE contamination is < 5%   | 0.264 [0.241, 0.286] | 905/3434 genomes whose TRUE contamination is >= 5%  | 0.867 [0.856, 0.878] |
| ALL (any novelty)               | CoCoPyE   | 0.066 [0.052, 0.081] | 103/1566 genomes whose TRUE contamination is < 5% | 0.024 [0.018, 0.030] | 82/3434 genomes whose TRUE contamination is >= 5%   | 0.955 [0.947, 0.963] |
| ALL (any novelty)               | DeepCheck | 0.001 [0.000, 0.003] | 2/1566 genomes whose TRUE contamination is < 5%   | 0.410 [0.383, 0.435] | 1407/3434 genomes whose TRUE contamination is >= 5% | 0.794 [0.782, 0.808] |

## 4. Is MAGICC's error monotone in novelty depth?

Trend test: Spearman ρ between novelty depth (0 = `lineage_represented` … 6 = `phylum_novel`) and the **cluster-mean** absolute error, with a cluster bootstrap CI, BH-corrected over the 8 tool × metric tests. `monotone_nondecreasing_all_classes` includes cells with as few as one cluster and is therefore decided by noise; `monotone_nondecreasing_powered_classes` restricts the check to classes with at least 20 reference clusters and is the one to read.

| tool_label                   | metric        | n_clusters | spearman_rho_depth_vs_error | spearman_rho_ci_lo | spearman_rho_ci_hi | p_spearman_two_sided | q_bh_spearman | monotone_nondecreasing_all_classes | monotone_nondecreasing_powered_classes |
|------------------------------|---------------|------------|-----------------------------|--------------------|--------------------|----------------------|---------------|------------------------------------|----------------------------------------|
| MAGICC v5 (released, v0.3.0) | completeness  | 2586       | 0.2966                      | 0.2601             | 0.331              | 1.163e-53            | 9.302e-53     | False                              | True                                   |
| MAGICC v5 (released, v0.3.0) | contamination | 2586       | 0.2077                      | 0.168              | 0.2477             | 1.368e-26            | 3.649e-26     | False                              | True                                   |
| CheckM2 1.0.1                | completeness  | 2586       | 0.08179                     | 0.04422            | 0.1213             | 3.121e-05            | 3.121e-05     | False                              | True                                   |
| CheckM2 1.0.1                | contamination | 2586       | 0.1033                      | 0.06727            | 0.142              | 1.414e-07            | 1.885e-07     | False                              | True                                   |
| CoCoPyE 0.5.0                | completeness  | 2586       | 0.2315                      | 0.1937             | 0.2669             | 8.345e-33            | 3.338e-32     | False                              | True                                   |
| CoCoPyE 0.5.0                | contamination | 2586       | 0.1084                      | 0.06903            | 0.1457             | 3.287e-08            | 5.26e-08      | False                              | True                                   |
| DeepCheck                    | completeness  | 2586       | 0.1214                      | 0.08247            | 0.1581             | 5.956e-10            | 1.191e-09     | False                              | True                                   |
| DeepCheck                    | contamination | 2586       | 0.09155                     | 0.05329            | 0.1285             | 3.118e-06            | 3.563e-06     | False                              | True                                   |

Ordered class means (cluster-mean absolute error, pp):

* **MAGICC v5 (released, v0.3.0) completeness** — lineage_represented=2.850(k=1995);species_novel=5.786(k=469);genus_novel=10.359(k=93);family_novel=9.667(k=18);order_novel=10.494(k=10);class_novel=2.366(k=1)
* **MAGICC v5 (released, v0.3.0) contamination** — lineage_represented=3.231(k=1995);species_novel=5.640(k=469);genus_novel=8.345(k=93);family_novel=13.044(k=18);order_novel=8.061(k=10);class_novel=0.162(k=1)
* **CheckM2 1.0.1 completeness** — lineage_represented=4.204(k=1995);species_novel=4.390(k=469);genus_novel=5.823(k=93);family_novel=5.824(k=18);order_novel=6.026(k=10);class_novel=3.071(k=1)
* **CheckM2 1.0.1 contamination** — lineage_represented=13.173(k=1995);species_novel=15.426(k=469);genus_novel=22.009(k=93);family_novel=25.236(k=18);order_novel=24.118(k=10);class_novel=0.070(k=1)
* **CoCoPyE 0.5.0 completeness** — lineage_represented=3.667(k=1995);species_novel=5.811(k=469);genus_novel=9.448(k=93);family_novel=11.869(k=18);order_novel=6.418(k=10);class_novel=10.871(k=1)
* **CoCoPyE 0.5.0 contamination** — lineage_represented=13.914(k=1995);species_novel=16.866(k=469);genus_novel=20.969(k=93);family_novel=20.923(k=18);order_novel=19.075(k=10);class_novel=4.281(k=1)
* **DeepCheck completeness** — lineage_represented=7.498(k=1995);species_novel=8.860(k=469);genus_novel=11.716(k=93);family_novel=13.690(k=18);order_novel=10.441(k=10);class_novel=4.239(k=1)
* **DeepCheck contamination** — lineage_represented=17.796(k=1995);species_novel=21.275(k=469);genus_novel=25.050(k=93);family_novel=28.104(k=18);order_novel=27.699(k=10);class_novel=0.068(k=1)

## 5. Paired MAGICC-vs-comparator tests within each novelty class

Paired difference `d = |error|_MAGICC − |error|_comparator` on **identical genomes**; negative favours MAGICC. Primary test is the two-sided paired Wilcoxon on cluster means (unit of analysis = reference genome); Hodges–Lehmann median paired difference with a cluster-bootstrap CI is the effect size; BH correction runs across novelty classes within each (comparator × metric) family.

### MAGICC v5 vs CheckM2

| novelty class                   | metric        | n    | clusters | MAE MAGICC | MAE CheckM2 | HL diff [95% CI]        | mean diff [95% CI]      | Cliff's δ | p (cluster-mean) | q (BH)    | favours    |
|---------------------------------|---------------|------|----------|------------|-------------|-------------------------|-------------------------|-----------|------------------|-----------|------------|
| lineage_represented             | completeness  | 3341 | 1995     | 3.172      | 5.412       | -1.17 [-1.44, -0.91]    | -2.24 [-2.53, -1.92]    | 0.031     | 8.27e-12         | 4.96e-11  | reference  |
| lineage_represented             | contamination | 3341 | 1995     | 4.017      | 19          | -12.95 [-14.40, -11.42] | -14.99 [-16.03, -13.92] | -0.293    | 4.14e-120        | 2.49e-119 | reference  |
| species_novel                   | completeness  | 1141 | 469      | 6.327      | 6.237       | +0.22 [-0.43, 0.93]     | +0.09 [-0.75, 1.01]     | 0.103     | 2.2e-06          | 6.6e-06   | comparison |
| species_novel                   | contamination | 1141 | 469      | 6.996      | 25.95       | -18.27 [-20.20, -15.92] | -18.95 [-20.84, -16.95] | -0.398    | 2.39e-24         | 7.16e-24  | reference  |
| genus_novel                     | completeness  | 339  | 93       | 10.27      | 7.471       | +2.07 [0.13, 4.51]      | +2.80 [0.47, 5.21]      | 0.163     | 0.000938         | 0.00188   | comparison |
| genus_novel                     | contamination | 339  | 93       | 9.858      | 34.26       | -24.30 [-27.70, -20.22] | -24.40 [-27.79, -20.57] | -0.541    | 8.04e-08         | 1.61e-07  | reference  |
| family_novel                    | completeness  | 122  | 18       | 10.5       | 7.219       | +3.06 [-0.59, 6.99]     | +3.28 [-0.09, 6.90]     | 0.164     | 0.0539           | 0.0646    | comparison |
| family_novel                    | contamination | 122  | 18       | 11.81      | 34.12       | -22.09 [-26.47, -16.10] | -22.31 [-26.82, -16.11] | -0.585    | 0.0268           | 0.0322    | reference  |
| order_novel                     | completeness  | 56   | 10       | 9.087      | 7.172       | +1.90 [-0.03, 4.32]     | +1.92 [-0.14, 4.78]     | 0.204     | 0.193            | 0.193     | comparison |
| order_novel                     | contamination | 56   | 10       | 6.367      | 33.09       | -26.76 [-33.91, -16.07] | -26.72 [-32.86, -17.66] | -0.715    | 0.0371           | 0.0371    | reference  |
| family_novel_or_deeper (pooled) | completeness  | 179  | 29       | 10.01      | 7.181       | +2.61 [-0.29, 5.47]     | +2.83 [0.49, 5.57]      | 0.179     | 0.0179           | 0.0268    | comparison |
| family_novel_or_deeper (pooled) | contamination | 179  | 29       | 10.04      | 33.61       | -23.32 [-27.46, -18.28] | -23.56 [-27.49, -18.65] | -0.615    | 0.00215          | 0.00323   | reference  |

### MAGICC v5 vs CoCoPyE

| novelty class                   | metric        | n    | clusters | MAE MAGICC | MAE CoCoPyE | HL diff [95% CI]        | mean diff [95% CI]      | Cliff's δ | p (cluster-mean) | q (BH)    | favours   |
|---------------------------------|---------------|------|----------|------------|-------------|-------------------------|-------------------------|-----------|------------------|-----------|-----------|
| lineage_represented             | completeness  | 3341 | 1995     | 3.172      | 5.091       | -1.05 [-1.35, -0.77]    | -1.92 [-2.37, -1.50]    | -0.115    | 5.78e-13         | 3.47e-12  | reference |
| lineage_represented             | contamination | 3341 | 1995     | 4.017      | 16.86       | -11.25 [-12.29, -10.23] | -12.85 [-13.56, -12.12] | -0.223    | 1.14e-121        | 6.83e-121 | reference |
| species_novel                   | completeness  | 1141 | 469      | 6.327      | 8.185       | -1.10 [-2.00, -0.29]    | -1.86 [-2.94, -0.69]    | -0.09     | 0.313            | 0.591     | reference |
| species_novel                   | contamination | 1141 | 469      | 6.996      | 21.66       | -13.72 [-15.28, -11.97] | -14.67 [-16.12, -13.20] | -0.323    | 1.88e-34         | 5.65e-34  | reference |
| genus_novel                     | completeness  | 339  | 93       | 10.27      | 12.55       | -1.75 [-3.93, 0.36]     | -2.27 [-4.63, -0.01]    | -0.15     | 0.628            | 0.628     | reference |
| genus_novel                     | contamination | 339  | 93       | 9.858      | 27.69       | -17.74 [-20.96, -14.05] | -17.83 [-20.88, -14.73] | -0.453    | 1.26e-08         | 2.51e-08  | reference |
| family_novel                    | completeness  | 122  | 18       | 10.5       | 14.71       | -3.59 [-7.30, -0.41]    | -4.21 [-7.70, -0.90]    | -0.277    | 0.0987           | 0.296     | reference |
| family_novel                    | contamination | 122  | 18       | 11.81      | 26.13       | -13.41 [-17.98, -8.76]  | -14.31 [-18.32, -9.12]  | -0.407    | 0.0483           | 0.0488    | reference |
| order_novel                     | completeness  | 56   | 10       | 9.087      | 9.733       | -0.61 [-2.54, 2.63]     | -0.65 [-2.83, 2.90]     | 0.031     | 0.492            | 0.591     | reference |
| order_novel                     | contamination | 56   | 10       | 6.367      | 25.65       | -18.70 [-25.13, -10.21] | -19.28 [-24.33, -12.56] | -0.598    | 0.0488           | 0.0488    | reference |
| family_novel_or_deeper (pooled) | completeness  | 179  | 29       | 10.01      | 13.13       | -2.58 [-5.36, -0.39]    | -3.12 [-5.46, -0.58]    | -0.195    | 0.405            | 0.591     | reference |
| family_novel_or_deeper (pooled) | contamination | 179  | 29       | 10.04      | 25.86       | -15.26 [-18.88, -10.83] | -15.81 [-19.32, -11.84] | -0.468    | 0.0038           | 0.0057    | reference |

### MAGICC v5 vs DeepCheck

| novelty class                   | metric        | n    | clusters | MAE MAGICC | MAE DeepCheck | HL diff [95% CI]        | mean diff [95% CI]      | Cliff's δ | p (cluster-mean) | q (BH)    | favours   |
|---------------------------------|---------------|------|----------|------------|---------------|-------------------------|-------------------------|-----------|------------------|-----------|-----------|
| lineage_represented             | completeness  | 3341 | 1995     | 3.172      | 8.528         | -3.98 [-4.42, -3.56]    | -5.36 [-5.83, -4.91]    | -0.429    | 6.22e-179        | 3.73e-178 | reference |
| lineage_represented             | contamination | 3341 | 1995     | 4.017      | 23.46         | -17.91 [-19.25, -16.44] | -19.44 [-20.54, -18.35] | -0.373    | 2.31e-181        | 1.39e-180 | reference |
| species_novel                   | completeness  | 1141 | 469      | 6.327      | 10.92         | -4.04 [-5.25, -2.85]    | -4.59 [-5.75, -3.37]    | -0.282    | 7.55e-12         | 2.26e-11  | reference |
| species_novel                   | contamination | 1141 | 469      | 6.996      | 30.48         | -22.40 [-24.21, -20.22] | -23.48 [-25.35, -21.41] | -0.474    | 1.32e-43         | 3.95e-43  | reference |
| genus_novel                     | completeness  | 339  | 93       | 10.27      | 15.85         | -5.77 [-8.64, -2.81]    | -5.58 [-8.34, -2.60]    | -0.313    | 0.146            | 0.219     | reference |
| genus_novel                     | contamination | 339  | 93       | 9.858      | 37.1          | -27.17 [-30.61, -22.90] | -27.24 [-30.77, -23.47] | -0.586    | 2.97e-09         | 5.94e-09  | reference |
| family_novel                    | completeness  | 122  | 18       | 10.5       | 17.25         | -6.99 [-11.66, -1.41]   | -6.75 [-11.31, -1.59]   | -0.391    | 0.108            | 0.217     | reference |
| family_novel                    | contamination | 122  | 18       | 11.81      | 36.58         | -24.46 [-28.74, -18.58] | -24.76 [-29.13, -18.93] | -0.651    | 0.00658          | 0.00789   | reference |
| order_novel                     | completeness  | 56   | 10       | 9.087      | 12.84         | -3.73 [-8.15, 0.66]     | -3.75 [-8.07, 0.90]     | -0.225    | 1                | 1         | reference |
| order_novel                     | contamination | 56   | 10       | 6.367      | 35.39         | -28.79 [-36.08, -18.72] | -29.02 [-35.22, -20.18] | -0.751    | 0.0137           | 0.0137    | reference |
| family_novel_or_deeper (pooled) | completeness  | 179  | 29       | 10.01      | 15.8          | -5.80 [-9.81, -1.98]    | -5.79 [-9.14, -2.00]    | -0.347    | 0.19             | 0.228     | reference |
| family_novel_or_deeper (pooled) | contamination | 179  | 29       | 10.04      | 36            | -25.69 [-29.71, -20.62] | -25.96 [-29.85, -21.38] | -0.672    | 0.000242         | 0.000363  | reference |

## 6. Files

* `results/revision/ws11/novelty_ladder/dominant_novelty_classification.tsv`
* `results/revision/ws11/novelty_ladder/novelty_ladder_metrics.tsv`
* `results/revision/ws11/novelty_ladder/novelty_ladder_paired_tests.tsv`
* `results/revision/ws11/novelty_ladder/novelty_ladder_trend.tsv`
* `results/revision/ws11/novelty_ladder/novelty_ladder_eda.json`
* `results/revision/ws11/novelty_ladder/novelty_ladder_summary.json`
* `results/revision/ws11/novelty_ladder/WS11_N_REPORT.md`

## 7. Input verification

Every input file exists, every set carries exactly 1,000 rows, all four tool prediction tables join 1:1 to `metadata.tsv` with zero NaN predictions and truth columns identical to the metadata, every dominant resolves to a GTDB lineage, and no dominant of the five sets is in the train or val split. No row was dropped.

