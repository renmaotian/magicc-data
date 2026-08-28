# WS11.G - genus vs family vs phylum novelty ladder

Generated 2026-08-28T10:25:40.752711

Four models score the SAME evaluation genomes: production V5, the WS11.G
genus-holdout model, the WS1.9 family-holdout model and the WS1.6
phylum-holdout model. Every panel genus lies inside a family WS1.9 held
out and a phylum WS1.6 held out, so the ladder is paired on identical
samples rather than assembled from separate experiments.

**Monotonicity across ranks is not assumed.** It is measured. WS1.9 already
produced a counterexample (Helicobacteraceae: family-level novelty as costly
as phylum-level, attenuation -0.39 pp, p = 0.228).

Common control: 790/1000 control samples (79 references) are in-distribution for all
three holdout models (dominant phylum outside the WS1.6 panel).

## Control deltas vs V5 (cost of the smaller training pool alone)

| model | completeness dMAE | contamination dMAE |
|---|---|---|
| genusHO | -0.012 | -0.283 |
| familyHO | +0.085 | -0.403 |
| phylumHO | +0.105 | -0.281 |

## Completeness DiD ladder (pp of MAE)

| group | n_refs | DiD genus [95% CI] | q(BH) | DiD family | DiD phylum | attenuation family-vs-genus | attenuation phylum-vs-genus |
|---|---|---|---|---|---|---|---|
| Bacteroidota_Muribaculaceae_genera | 100 | +2.95 [2.31, 3.62] | 0.0006 | +4.46 | +7.76 | +1.51 | +4.81 |
| Bacteroidota_Flavobacteriaceae_genera | 96 | +2.67 [2.11, 3.24] | 0.0006 | +4.40 | +7.67 | +1.73 | +5.00 |
| Campylobacterota_Helicobacteraceae_genera | 51 | +2.86 [1.92, 3.94] | 0.0006 | +2.83 | +2.99 | -0.03 | +0.12 |
| Campylobacterota_Arcobacteraceae_genera | 33 | +4.28 [3.43, 5.12] | 0.0006 | +4.77 | +8.24 | +0.49 | +3.96 |
| Halobacteriota_halophilic_genera | 24 | +2.11 [1.41, 2.86] | 0.0006 | +2.45 | +3.82 | +0.33 | +1.71 |
| Patescibacteriota_genera | 30 | +4.73 [3.02, 6.48] | 0.0006 | +6.16 | +25.68 | +1.43 | +20.95 |

## Contamination DiD ladder (pp of MAE)

| group | n_refs | DiD genus [95% CI] | q(BH) | DiD family | DiD phylum | attenuation family-vs-genus | attenuation phylum-vs-genus |
|---|---|---|---|---|---|---|---|
| Bacteroidota_Muribaculaceae_genera | 100 | +3.18 [2.65, 3.69] | 0.0006 | +4.64 | +6.53 | +1.46 | +3.35 |
| Bacteroidota_Flavobacteriaceae_genera | 96 | +4.35 [3.34, 5.35] | 0.0006 | +5.57 | +15.38 | +1.21 | +11.03 |
| Campylobacterota_Helicobacteraceae_genera | 51 | +3.04 [2.42, 3.60] | 0.0006 | +5.68 | +10.84 | +2.64 | +7.79 |
| Campylobacterota_Arcobacteraceae_genera | 33 | +3.43 [2.91, 3.91] | 0.0006 | +2.80 | +2.44 | -0.63 | -0.99 |
| Halobacteriota_halophilic_genera | 24 | +0.91 [0.32, 1.54] | 0.0090 | +1.87 | +3.41 | +0.96 | +2.50 |
| Patescibacteriota_genera | 30 | +2.95 [2.07, 3.88] | 0.0006 | +3.68 | +13.18 | +0.73 | +10.23 |
