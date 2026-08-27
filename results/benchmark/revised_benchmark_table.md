# Revised five-set benchmark table (WS1.5)

Generated 2026-07-26T16:58:44.482793+00:00 by `scripts/94_revised_benchmark_table.py`.

All numbers come from one frozen model version, **MAGICC V5** (`models/magicc_v5.onnx`, SHA256 b843466…). Sets C and D of the submitted manuscript are **withdrawn** and replaced by `C_clean` / `D_clean`, built from 100 strictly held-out test-split references × 10 simulations each (0 train/val overlap, 0 overlap with the 2,000 k-mer feature-selection genomes; `results/revision/provenance/`).

Denominator for both metrics: the dominant genome's **full reference length**. Completeness = retained dominant bp / reference bp × 100; contamination = total contaminant bp / reference bp × 100.

MAE, percentage points, with 95 % CI from a cluster bootstrap over `dominant_accession` (2000 resamples, seed 7600); the OVERALL row resamples clusters within each set (stratified), keeping the 1,000-per-set composition fixed.


## Completeness MAE (95 % CI)

| Set | subset | n | clusters | MAGICC V5 | CheckM2 1.0.1 | CoCoPyE 0.5.0 | DeepCheck |
|---|---|---|---|---|---|---|---|
| A_v2 | all | 1000 | 798 | 2.18 (1.94–2.42) | 2.54 (2.36–2.72) | 3.63 (3.36–3.90) | 4.26 (3.98–4.55) |
| B_v2 | all | 1000 | 803 | 2.97 (2.63–3.32) | 0.45 (0.32–0.60) | 2.61 (2.31–2.93) | 6.61 (6.04–7.21) |
| C_clean | all | 1000 | 100 | 6.82 (5.81–7.83) | 7.71 (7.33–8.09) | 15.80 (15.14–16.48) | 17.81 (16.88–18.75) |
| D_clean | all | 1000 | 100 | 5.63 (5.02–6.34) | 9.17 (8.44–9.89) | 5.89 (5.31–6.56) | 8.78 (8.00–9.61) |
| E | all | 1000 | 785 | 5.49 (5.04–5.96) | 9.14 (8.47–9.81) | 5.02 (4.69–5.37) | 11.69 (10.99–12.43) |
| E | in_domain | 868 | 714 | 4.14 (3.77–4.54) | 6.51 (5.95–7.07) | 4.17 (3.86–4.47) | 9.26 (8.63–9.91) |
| OVERALL | all | 5000 | 2586 | 4.62 (4.34–4.91) | 5.80 (5.58–6.03) | 6.59 (6.38–6.79) | 9.83 (9.51–10.15) |
| OVERALL | E_in_domain | 4868 | 2515 | 4.35 (4.08–4.64) | 5.24 (5.04–5.46) | 6.48 (6.28–6.70) | 9.35 (9.06–9.67) |

## Contamination MAE (95 % CI)

| Set | subset | n | clusters | MAGICC V5 | CheckM2 1.0.1 | CoCoPyE 0.5.0 | DeepCheck |
|---|---|---|---|---|---|---|---|
| A_v2 | all | 1000 | 798 | 0.83 (0.65–1.05) | 0.27 (0.25–0.30) | 0.77 (0.59–0.97) | 0.40 (0.37–0.44) |
| B_v2 | all | 1000 | 803 | 4.45 (4.12–4.79) | 17.66 (16.54–18.77) | 19.14 (18.07–20.23) | 25.70 (24.39–27.09) |
| C_clean | all | 1000 | 100 | 9.08 (8.15–9.98) | 39.62 (38.12–41.12) | 31.07 (29.63–32.49) | 41.67 (40.15–43.17) |
| D_clean | all | 1000 | 100 | 6.52 (5.96–7.12) | 34.70 (33.16–36.13) | 22.49 (21.29–23.63) | 38.96 (37.41–40.44) |
| E | all | 1000 | 785 | 5.66 (5.17–6.17) | 18.47 (17.21–19.82) | 21.60 (20.27–23.03) | 25.43 (23.77–27.04) |
| E | in_domain | 868 | 714 | 3.94 (3.58–4.32) | 14.95 (13.81–16.11) | 16.60 (15.32–17.87) | 20.54 (19.11–22.04) |
| OVERALL | all | 5000 | 2586 | 5.31 (5.07–5.57) | 22.14 (21.58–22.71) | 19.02 (18.50–19.54) | 26.43 (25.81–27.07) |
| OVERALL | E_in_domain | 4868 | 2515 | 4.99 (4.75–5.23) | 21.62 (21.06–22.19) | 18.05 (17.55–18.55) | 25.59 (25.01–26.21) |

## Mean signed error (bias = predicted − true, 95 % CI)

| Set | subset | Target | MAGICC V5 | CheckM2 1.0.1 | CoCoPyE 0.5.0 | DeepCheck |
|---|---|---|---|---|---|---|
| A_v2 | all | completeness | 0.09 (-0.21–0.38) | 1.23 (1.00–1.46) | 1.60 (1.26–1.92) | 1.46 (1.07–1.84) |
| A_v2 | all | contamination | 0.83 (0.65–1.05) | 0.27 (0.25–0.30) | 0.77 (0.59–0.97) | 0.29 (0.25–0.33) |
| B_v2 | all | completeness | -2.97 (-3.32–-2.63) | -0.45 (-0.60–-0.32) | -2.61 (-2.93–-2.31) | 4.40 (3.73–5.11) |
| B_v2 | all | contamination | -0.14 (-0.57–0.32) | -14.26 (-15.58–-12.90) | -18.42 (-19.55–-17.30) | -20.59 (-22.28–-18.92) |
| C_clean | all | completeness | -2.32 (-3.81–-0.83) | -4.21 (-4.82–-3.58) | -13.88 (-14.69–-13.11) | -17.06 (-18.11–-16.01) |
| C_clean | all | contamination | -3.55 (-5.07–-2.01) | -39.60 (-41.11–-38.11) | -30.23 (-31.68–-28.73) | -41.67 (-43.17–-40.15) |
| D_clean | all | completeness | -0.26 (-1.35–0.78) | 7.74 (6.78–8.68) | -0.57 (-1.43–0.21) | 4.73 (3.45–5.98) |
| D_clean | all | contamination | -3.83 (-4.65–-2.94) | -34.66 (-36.10–-33.12) | -21.82 (-23.05–-20.50) | -38.84 (-40.31–-37.27) |
| E | all | completeness | 1.13 (0.56–1.67) | 8.32 (7.63–9.02) | 1.56 (1.08–1.99) | 10.21 (9.40–11.00) |
| E | all | contamination | -3.12 (-3.71–-2.52) | -16.90 (-18.29–-15.53) | -20.60 (-22.10–-19.19) | -17.88 (-19.89–-15.74) |
| E | in_domain | completeness | -0.68 (-1.14–-0.24) | 5.59 (5.00–6.20) | 0.25 (-0.18–0.67) | 7.61 (6.87–8.33) |
| E | in_domain | contamination | -1.09 (-1.54–-0.63) | -13.19 (-14.50–-11.93) | -15.46 (-16.76–-14.14) | -15.99 (-17.78–-14.18) |
| OVERALL | all | completeness | -0.87 (-1.28–-0.47) | 2.53 (2.25–2.80) | -2.78 (-3.04–-2.51) | 0.75 (0.34–1.14) |
| OVERALL | all | contamination | -1.96 (-2.35–-1.60) | -21.03 (-21.63–-20.45) | -18.06 (-18.62–-17.52) | -23.74 (-24.43–-23.03) |
| OVERALL | E_in_domain | completeness | -1.24 (-1.64–-0.85) | 1.89 (1.63–2.16) | -3.13 (-3.40–-2.87) | 0.03 (-0.38–0.43) |
| OVERALL | E_in_domain | contamination | -1.57 (-1.94–-1.21) | -20.48 (-21.07–-19.92) | -17.07 (-17.60–-16.54) | -23.56 (-24.22–-22.91) |

## R² = coefficient of determination (1 − SS_res/SS_tot, `sklearn.metrics.r2_score`)

Negative values mean the predictor is worse than always predicting the mean of the true values; they are correct and reported plainly. The squared Pearson correlation - which ignores bias and scale error, is always ≥ this quantity, and is what the submitted Table S2c mixed in for some tools - is kept in the TSV as `*_r2_pearson_sq` and is never called R². '-' marks cells where the true value has zero variance (Set A_v2 contamination is 0 % throughout, Set B_v2 completeness is 100 % throughout), so R² is undefined (WS5.5).

| Set | subset | Target | MAGICC V5 | CheckM2 1.0.1 | CoCoPyE 0.5.0 | DeepCheck |
|---|---|---|---|---|---|---|
| A_v2 | all | completeness | 0.937 | 0.949 | 0.896 | 0.865 |
| A_v2 | all | contamination | - | - | - | - |
| B_v2 | all | completeness | - | - | - | - |
| B_v2 | all | contamination | 0.941 | 0.175 | 0.075 | -0.495 |
| C_clean | all | completeness | 0.611 | 0.653 | -0.313 | -0.569 |
| C_clean | all | contamination | 0.770 | -2.195 | -1.170 | -2.496 |
| D_clean | all | completeness | 0.705 | 0.316 | 0.709 | 0.381 |
| D_clean | all | contamination | 0.854 | -1.659 | -0.318 | -2.311 |
| E | all | completeness | 0.707 | 0.249 | 0.801 | 0.009 |
| E | all | contamination | 0.913 | 0.303 | 0.102 | -0.296 |
| E | in_domain | completeness | 0.821 | 0.556 | 0.853 | 0.328 |
| E | in_domain | contamination | 0.950 | 0.402 | 0.266 | -0.053 |
| OVERALL | all | completeness | 0.785 | 0.666 | 0.628 | 0.305 |
| OVERALL | all | contamination | 0.915 | -0.118 | 0.141 | -0.503 |
| OVERALL | E_in_domain | completeness | 0.805 | 0.728 | 0.627 | 0.362 |
| OVERALL | E_in_domain | contamination | 0.921 | -0.145 | 0.164 | -0.492 |

## Caveats that must travel with this table

| Set | n | unique dominants | samples with contamination % > completeness % (outside the V5 training domain) | R² undefined |
|---|---|---|---|---|
| A_v2 | 1000 | 798 | 0 | contamination (true value constant) |
| B_v2 | 1000 | 803 | 0 | completeness (true value constant) |
| C_clean | 1000 | 100 | 0 | - |
| D_clean | 1000 | 100 | 0 | - |
| E | 1000 | 785 | 132 | - |

* **Set E carries 132/1,000 out-of-domain samples.** The `in_domain` row restricts Set E to its 868 constraint-satisfying samples, and `OVERALL / E_in_domain` restricts only Set E inside the pooled estimate (4,868 genomes). Use one variant consistently throughout the manuscript.

* Sets A_v2, B_v2, C_clean and D_clean contain **no** out-of-domain samples, so their rows are unaffected.

* Descriptive statistics only. Paired two-sided clustered tests with Benjamini–Hochberg correction and effect sizes come from the statistics framework (`scripts/101`–`105`, `results/revision/metrics/`).
