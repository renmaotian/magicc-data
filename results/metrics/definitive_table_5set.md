# Definitive leakage-free results table (MAGICC revision, WS5)

Generated from `results/revision/metrics/` after CheckM2 / CoCoPyE / DeepCheck predictions became available for `set_C_clean` and `set_D_clean`.

**Conventions.** R^2 is the coefficient of determination (1 - SS_res/SS_tot), never squared Pearson; it is omitted where the true value has zero variance. All 95% CIs are cluster bootstraps over the dominant reference genome. Signed bias = mean(predicted - true), so negative = under-estimate.

**Denominators.** completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages share the same denominator and are independent measures.


## Set A (completeness gradient, 0% contamination)

n = 1000 genomes; 798 reference-genome clusters (CIs are cluster bootstraps over these).

| Tool | Completeness MAE (95% CI) | Completeness R^2 | Completeness bias | Contamination MAE (95% CI) | Contamination R^2 | Contamination bias |
|---|---|---|---|---|---|---|
| MAGICC v5 | 2.18 [1.93, 2.44] | 0.937 [0.915, 0.955] | 0.09 [-0.22, 0.36] | 0.83 [0.64, 1.06] | n/a | 0.83 [0.64, 1.06] |
| CheckM2 | 2.54 [2.36, 2.71] | 0.949 [0.940, 0.956] | 1.23 [1.00, 1.46] | 0.27 [0.25, 0.30] | n/a | 0.27 [0.25, 0.30] |
| CoCoPyE | 3.63 [3.35, 3.93] | 0.896 [0.878, 0.912] | 1.60 [1.26, 1.93] | 0.77 [0.61, 0.98] | n/a | 0.77 [0.61, 0.98] |
| DeepCheck | 4.26 [3.98, 4.54] | 0.865 [0.843, 0.885] | 1.46 [1.09, 1.85] | 0.40 [0.37, 0.44] | n/a | 0.29 [0.25, 0.32] |

> R^2 omitted for contamination: R^2 is undefined: the true contamination is constant at 0% for all 1000 genomes in this set, so the total sum of squares is zero and no fraction of variance can be explained. MAE, RMSE and mean signed error remain well defined and are reported instead.

## Set B (contamination gradient, 100% completeness)

n = 1000 genomes; 803 reference-genome clusters (CIs are cluster bootstraps over these).

| Tool | Completeness MAE (95% CI) | Completeness R^2 | Completeness bias | Contamination MAE (95% CI) | Contamination R^2 | Contamination bias |
|---|---|---|---|---|---|---|
| MAGICC v5 | 2.97 [2.65, 3.32] | n/a | -2.97 [-3.32, -2.65] | 4.45 [4.12, 4.79] | 0.941 [0.928, 0.951] | -0.14 [-0.59, 0.30] |
| CheckM2 | 0.45 [0.32, 0.60] | n/a | -0.45 [-0.60, -0.32] | 17.66 [16.51, 18.78] | 0.175 [0.086, 0.259] | -14.26 [-15.59, -12.96] |
| CoCoPyE | 2.61 [2.30, 2.95] | n/a | -2.61 [-2.95, -2.30] | 19.14 [18.04, 20.29] | 0.075 [0.009, 0.137] | -18.42 [-19.61, -17.28] |
| DeepCheck | 6.61 [6.06, 7.17] | n/a | 4.40 [3.76, 5.05] | 25.70 [24.35, 27.10] | -0.495 [-0.653, -0.358] | -20.59 [-22.27, -18.87] |

> R^2 omitted for completeness: R^2 is undefined: the true completeness is constant at 100% for all 1000 genomes in this set, so the total sum of squares is zero and no fraction of variance can be explained. MAE, RMSE and mean signed error remain well defined and are reported instead.

## Set C-clean (Patescibacteriota, held-out test split)

n = 1000 genomes; 100 reference-genome clusters (CIs are cluster bootstraps over these).

| Tool | Completeness MAE (95% CI) | Completeness R^2 | Completeness bias | Contamination MAE (95% CI) | Contamination R^2 | Contamination bias |
|---|---|---|---|---|---|---|
| MAGICC v5 | 6.82 [5.78, 7.86] | 0.611 [0.483, 0.714] | -2.32 [-3.83, -0.81] | 9.08 [8.19, 10.10] | 0.770 [0.711, 0.816] | -3.55 [-5.13, -2.00] |
| CheckM2 | 7.71 [7.34, 8.11] | 0.653 [0.617, 0.683] | -4.21 [-4.83, -3.59] | 39.62 [38.00, 41.23] | -2.195 [-2.429, -1.981] | -39.60 [-41.22, -37.98] |
| CoCoPyE | 15.80 [15.12, 16.49] | -0.313 [-0.433, -0.204] | -13.88 [-14.69, -13.08] | 31.07 [29.57, 32.59] | -1.170 [-1.340, -1.021] | -30.23 [-31.86, -28.67] |
| DeepCheck | 17.81 [16.86, 18.79] | -0.569 [-0.751, -0.404] | -17.06 [-18.10, -16.02] | 41.67 [40.00, 43.35] | -2.496 [-2.754, -2.266] | -41.67 [-43.34, -39.99] |

## Set D-clean (Archaea, held-out test split)

n = 1000 genomes; 100 reference-genome clusters (CIs are cluster bootstraps over these).

| Tool | Completeness MAE (95% CI) | Completeness R^2 | Completeness bias | Contamination MAE (95% CI) | Contamination R^2 | Contamination bias |
|---|---|---|---|---|---|---|
| MAGICC v5 | 5.63 [5.00, 6.34] | 0.705 [0.617, 0.774] | -0.26 [-1.32, 0.78] | 6.52 [5.96, 7.11] | 0.854 [0.823, 0.881] | -3.83 [-4.65, -2.97] |
| CheckM2 | 9.17 [8.49, 9.91] | 0.316 [0.206, 0.414] | 7.74 [6.85, 8.70] | 34.70 [33.20, 36.14] | -1.659 [-1.899, -1.446] | -34.66 [-36.10, -33.15] |
| CoCoPyE | 5.89 [5.32, 6.56] | 0.709 [0.633, 0.770] | -0.57 [-1.46, 0.23] | 22.49 [21.36, 23.63] | -0.318 [-0.432, -0.217] | -21.82 [-23.06, -20.57] |
| DeepCheck | 8.78 [8.02, 9.63] | 0.381 [0.251, 0.498] | 4.73 [3.43, 6.00] | 38.96 [37.49, 40.49] | -2.311 [-2.590, -2.070] | -38.84 [-40.37, -37.37] |

## Set E (realistic mixture)

n = 1000 genomes; 785 reference-genome clusters (CIs are cluster bootstraps over these).

| Tool | Completeness MAE (95% CI) | Completeness R^2 | Completeness bias | Contamination MAE (95% CI) | Contamination R^2 | Contamination bias |
|---|---|---|---|---|---|---|
| MAGICC v5 | 5.49 [5.04, 5.95] | 0.707 [0.662, 0.752] | 1.13 [0.55, 1.68] | 5.66 [5.19, 6.15] | 0.913 [0.898, 0.927] | -3.12 [-3.68, -2.54] |
| CheckM2 | 9.14 [8.48, 9.80] | 0.249 [0.165, 0.327] | 8.32 [7.63, 9.04] | 18.47 [17.25, 19.75] | 0.303 [0.230, 0.374] | -16.90 [-18.27, -15.52] |
| CoCoPyE | 5.02 [4.67, 5.37] | 0.801 [0.774, 0.826] | 1.56 [1.10, 2.01] | 21.60 [20.31, 22.97] | 0.102 [0.040, 0.157] | -20.60 [-22.04, -19.25] |
| DeepCheck | 11.69 [10.99, 12.41] | 0.009 [-0.098, 0.108] | 10.21 [9.41, 11.00] | 25.43 [23.88, 27.10] | -0.296 [-0.495, -0.123] | -17.88 [-19.91, -15.73] |

## POOLED, leakage-free 5 sets (n=5,000)

n = 5000 genomes; 2586 reference-genome clusters (CIs are cluster bootstraps over these).

| Tool | Completeness MAE (95% CI) | Completeness R^2 | Completeness bias | Contamination MAE (95% CI) | Contamination R^2 | Contamination bias |
|---|---|---|---|---|---|---|
| MAGICC v5 | 4.62 [4.32, 4.92] | 0.785 [0.756, 0.811] | -0.87 [-1.26, -0.46] | 5.31 [5.00, 5.63] | 0.915 [0.904, 0.925] | -1.96 [-2.38, -1.56] |
| CheckM2 | 5.80 [5.52, 6.10] | 0.666 [0.639, 0.691] | 2.53 [2.08, 2.94] | 22.14 [21.08, 23.20] | -0.118 [-0.195, -0.047] | -21.03 [-22.16, -19.90] |
| CoCoPyE | 6.59 [6.16, 7.01] | 0.628 [0.586, 0.669] | -2.78 [-3.29, -2.24] | 19.02 [18.20, 19.78] | 0.141 [0.099, 0.185] | -18.06 [-18.83, -17.23] |
| DeepCheck | 9.83 [9.35, 10.30] | 0.305 [0.250, 0.358] | 0.75 [-0.07, 1.57] | 26.43 [25.37, 27.48] | -0.503 [-0.592, -0.416] | -23.74 [-24.93, -22.48] |

## POOLED, submitted-manuscript 5 sets (leaky C/D)

n = 5000 genomes; 4386 reference-genome clusters (CIs are cluster bootstraps over these).

| Tool | Completeness MAE (95% CI) | Completeness R^2 | Completeness bias | Contamination MAE (95% CI) | Contamination R^2 | Contamination bias |
|---|---|---|---|---|---|---|
| MAGICC v5 | 3.79 [3.63, 3.96] | 0.837 [0.821, 0.852] | 0.13 [-0.05, 0.33] | 5.39 [5.18, 5.62] | 0.915 [0.909, 0.922] | -3.11 [-3.35, -2.86] |
| CheckM2 | 6.00 [5.78, 6.24] | 0.637 [0.612, 0.660] | 2.52 [2.25, 2.79] | 23.14 [22.41, 23.84] | -0.136 [-0.178, -0.093] | -22.03 [-22.76, -21.29] |
| CoCoPyE | 6.62 [6.38, 6.85] | 0.619 [0.594, 0.645] | -2.86 [-3.13, -2.58] | 20.26 [19.63, 20.89] | 0.102 [0.076, 0.130] | -19.24 [-19.89, -18.59] |
| DeepCheck | 10.10 [9.81, 10.39] | 0.269 [0.230, 0.307] | 0.72 [0.33, 1.11] | 27.57 [26.74, 28.35] | -0.508 [-0.568, -0.445] | -24.74 [-25.61, -23.86] |

## Honest counter-finding: `set_C_clean` false-fail rate at the 5% contamination threshold (MIMAG-inspired)

Denominator = genomes whose TRUE contamination is < 5% (n = 52); a false fail is such a genome predicted at >= 5% contamination.

| Tool | False-fail rate (95% CI) | n false fails / n truly clean |
|---|---|---|
| MAGICC v5 | 0.442 [0.273, 0.617] | 23 / 52 |
| CheckM2 | 0.000 [0.000, 0.000] | 0 / 52 |
| CoCoPyE | 0.712 [0.580, 0.830] | 37 / 52 |
| DeepCheck | 0.000 [0.000, 0.000] | 0 / 52 |
