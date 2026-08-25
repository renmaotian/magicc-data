# Leakage-specificity control (WS1.5)

Generated 2026-07-26T16:55:59.712713+00:00 by `scripts/93_leakage_specificity_control.py`.

MAGICC V5 was fitted on the superseded Sets C/D dominants; CheckM2, CoCoPyE and DeepCheck were not. A drop confined to MAGICC isolates the effect to training-set memorisation.

The superseded sets contain samples that violate the training constraint `contamination % <= completeness %` (set_C 141/1,000, set_D 213/1,000) while the clean sets contain none, so each comparison is given twice. **The `constraint_ok` rows are the confounder-free comparison and the ones to quote.**


## Confounder-free: superseded set restricted to contamination % <= completeness %

| Lineage | Tool | Trained on these genomes | n leaked | n clean | comp MAE leaked | comp MAE clean | Δ comp MAE (95% CI) | cont MAE leaked | cont MAE clean | Δ cont MAE (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| Patescibacteriota (CPR) | MAGICC V5 | **yes** | 859 | 1000 | 2.38 | 6.82 | +4.44 (+3.45, +5.50) | 6.05 | 9.08 | +3.03 (+2.10, +4.08) |
| Patescibacteriota (CPR) | CheckM2 1.0.1 | no | 859 | 1000 | 7.87 | 7.71 | -0.16 (-0.69, +0.37) | 36.86 | 39.62 | +2.76 (+0.52, +5.01) |
| Patescibacteriota (CPR) | CoCoPyE 0.5.0 | no | 859 | 1000 | 17.01 | 15.80 | -1.20 (-2.19, -0.27) | 28.72 | 31.07 | +2.35 (+0.34, +4.37) |
| Patescibacteriota (CPR) | DeepCheck | no | 859 | 1000 | 18.67 | 17.81 | -0.86 (-1.99, +0.28) | 39.01 | 41.67 | +2.66 (+0.34, +4.96) |
| Archaea | MAGICC V5 | **yes** | 787 | 1000 | 3.69 | 5.63 | +1.95 (+1.24, +2.68) | 4.85 | 6.52 | +1.67 (+1.00, +2.36) |
| Archaea | CheckM2 1.0.1 | no | 787 | 1000 | 6.72 | 9.17 | +2.45 (+1.62, +3.32) | 29.25 | 34.70 | +5.45 (+3.39, +7.33) |
| Archaea | CoCoPyE 0.5.0 | no | 787 | 1000 | 5.69 | 5.89 | +0.21 (-0.49, +1.00) | 17.76 | 22.49 | +4.73 (+3.06, +6.25) |
| Archaea | DeepCheck | no | 787 | 1000 | 7.42 | 8.78 | +1.37 (+0.45, +2.34) | 33.41 | 38.96 | +5.55 (+3.35, +7.57) |

## Confounded: superseded set as-is

| Lineage | Tool | Trained on these genomes | n leaked | n clean | comp MAE leaked | comp MAE clean | Δ comp MAE (95% CI) | cont MAE leaked | cont MAE clean | Δ cont MAE (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| Patescibacteriota (CPR) | MAGICC V5 | **yes** | 1000 | 1000 | 3.15 | 6.82 | +3.67 (+2.65, +4.76) | 7.97 | 9.08 | +1.11 (+0.10, +2.24) |
| Patescibacteriota (CPR) | CheckM2 1.0.1 | no | 1000 | 1000 | 7.99 | 7.71 | -0.28 (-0.82, +0.24) | 42.37 | 39.62 | -2.75 (-5.04, -0.40) |
| Patescibacteriota (CPR) | CoCoPyE 0.5.0 | no | 1000 | 1000 | 15.73 | 15.80 | +0.08 (-0.79, +0.99) | 34.07 | 31.07 | -3.00 (-5.14, -0.81) |
| Patescibacteriota (CPR) | DeepCheck | no | 1000 | 1000 | 17.97 | 17.81 | -0.16 (-1.30, +1.02) | 44.56 | 41.67 | -2.89 (-5.20, -0.48) |
| Archaea | MAGICC V5 | **yes** | 1000 | 1000 | 5.18 | 5.63 | +0.46 (-0.29, +1.27) | 8.06 | 6.52 | -1.54 (-2.37, -0.70) |
| Archaea | CheckM2 1.0.1 | no | 1000 | 1000 | 9.89 | 9.17 | -0.72 (-1.63, +0.23) | 36.92 | 34.70 | -2.23 (-4.30, -0.11) |
| Archaea | CoCoPyE 0.5.0 | no | 1000 | 1000 | 6.14 | 5.89 | -0.25 (-0.97, +0.53) | 25.73 | 22.49 | -3.23 (-5.04, -1.46) |
| Archaea | DeepCheck | no | 1000 | 1000 | 9.97 | 8.78 | -1.19 (-2.17, -0.21) | 41.77 | 38.96 | -2.82 (-5.08, -0.58) |

## Specificity summary (Δ = clean − superseded, percentage points)

| Lineage | Comparison | Target | MAGICC Δ MAE | competitor Δ MAE (mean / min / max) | MAGICC − mean competitor | competitors whose Δ CI excludes 0 |
|---|---|---|---|---|---|---|
| Patescibacteriota (CPR) | full | completeness | +3.67 | -0.12 / -0.28 / +0.08 | +3.79 | 0/3 |
| Patescibacteriota (CPR) | full | contamination | +1.11 | -2.88 / -3.00 / -2.75 | +3.99 | 3/3 |
| Patescibacteriota (CPR) | constraint_ok | completeness | +4.44 | -0.74 / -1.20 / -0.16 | +5.19 | 1/3 |
| Patescibacteriota (CPR) | constraint_ok | contamination | +3.03 | +2.59 / +2.35 / +2.76 | +0.44 | 3/3 |
| Archaea | full | completeness | +0.46 | -0.72 / -1.19 / -0.25 | +1.18 | 1/3 |
| Archaea | full | contamination | -1.54 | -2.76 / -3.23 / -2.23 | +1.22 | 3/3 |
| Archaea | constraint_ok | completeness | +1.95 | +1.34 / +0.21 / +2.45 | +0.61 | 2/3 |
| Archaea | constraint_ok | contamination | +1.67 | +5.24 / +4.73 / +5.55 | -3.57 | 3/3 |

R² columns are the coefficient of determination (1 − SS_res/SS_tot, = `sklearn.metrics.r2_score`); the squared Pearson correlation is kept separately as `*_pearson_r2_legacy` and is never called R².


CIs: cluster bootstrap over `dominant_accession`, 2000 resamples, seed 7600; the clean−leaked delta resamples the two (disjoint) sets independently. Descriptive only — the paired two-sided clustered tests with BH correction and effect sizes are produced by the statistics framework (`scripts/101`–`105`, `results/revision/metrics/`).
