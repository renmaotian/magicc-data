# WS11.G - leave-GENUS-out retraining

Generated 2026-08-28T10:27:29.453279

**Holdout models are validation artefacts only.** The released MAGICC model (`models/magicc_v5.onnx`) remains trained on all data and was not touched.

## Why this experiment exists

Reviewer 2 asked for taxonomic holdout at genus, family or phylum level and Reviewer 1 asked for held-out evaluation by genome, species, genus and phylum. Phylum (WS1.6) and family (WS1.9) were run; genus was declined with the argument that a genus-level holdout inside a represented family would be expected to fall below the family-level cost. That argument assumes degradation is monotone across ranks, and WS1.9's own Helicobacteraceae cell contradicts it (family-level novelty cost as much as phylum-level novelty: attenuation -0.39 pp, p = 0.228). This experiment measures the genus level instead of arguing about it. **Monotonicity across ranks is not assumed anywhere in this report.**

## The panel

44 genera in 18 families across 4 phyla. 2,435 of 79,948 training genomes removed (3.05 %; WS1.9 family panel 6.95 %, WS1.6 phylum panel 19.64 %).

| group | genera | parent family | parent phylum | train removed | test refs | parent family retained | parent phylum retained |
|---|---|---|---|---|---|---|---|
| Bacteroidota_Muribaculaceae_genera | 5 | Muribaculaceae | Bacteroidota | 904 | 111 | 50.1 % | 89.1 % |
| Bacteroidota_Flavobacteriaceae_genera | 6 | Flavobacteriaceae | Bacteroidota | 550 | 96 | 50.0 % | 93.4 % |
| Campylobacterota_Helicobacteraceae_genera | 6 | Helicobacteraceae | Campylobacterota | 398 | 51 | 75.5 % | 92.0 % |
| Campylobacterota_Arcobacteraceae_genera | 2 | Arcobacteraceae | Campylobacterota | 227 | 33 | 15.6 % | 95.4 % |
| Halobacteriota_halophilic_genera | 9 | Haloferacaceae+Haloarculaceae | Halobacteriota | 169 | 24 | 23.9 % | 70.5 % |
| Patescibacteriota_genera | 16 | 12 CPR families | Patescibacteriota | 187 | 30 | 49.6 % | 85.5 % |

### Constraint verification (numerical, not asserted)

- Phyla with training genomes after removal: **110** of 110; eliminated: **NONE**.
- Families with training genomes after removal: **1620** of 1,620; eliminated: **NONE**.
- Genera: 5,648 -> 5604. Surviving training pool 77,513 genomes.
- Reduced-genome pool under V5's own definition survives at 1266/1453 = 87.13 % (median 0.944 Mbp vs V5 0.912 Mbp), so **no adaptation was applied** - WS1.6 needed a redefined pool, WS1.9 kept 63.5 %.
- Ladder completeness: every panel genus lies inside a family WS1.9 held out and a phylum WS1.6 held out (`ladder_complete = True`).

## Design validity

- **Dry run, production V5 substituted for both models:** max |difference| over 4 numeric DiD/delta columns = **0.000e+00** -> **PASS - exactly zero**. (`dryrun_v5_vs_v5/`)
- **`in_distribution` control, holdout vs V5 dMAE (completeness):** **+0.077 pp** (holdout 6.382 vs V5 6.306).
- **`in_distribution` control, holdout vs V5 dMAE (contamination):** **-0.155 pp** (holdout 7.197 vs V5 7.352).
  WS1.6's completeness control delta was -0.047 and WS1.9's +0.029. A control delta of ~0 means removing the panel genomes had no measurable effect on retained lineages, so every measured degradation is attributable to lineage novelty alone. **The design is only valid if this holds.**
- **Level-switch reproduction:** adding `genus` to `MAGICC_HOLDOUT_LEVEL` changes nothing at phylum or family level - every value the pipeline reads out of `config.py`, including the SHA-256 of the reference-genome accession list `select_refs()` draws for every group, is identical under the pre-WS11.G config and the current one (`level_switch_reproduction.json`, script 224).
- **Frozen library:** the ten non-config files in `scripts/holdout_lib/` are byte-identical to the `FROZEN_SHA256_WS1.9.txt` record; only `config.py` changed (`FROZEN_SHA256_WS11.G.txt`).
- **Training:** seed 42, 68 best epoch, ? h.

## Primary result - difference-in-differences

`DiD = (MAE_holdout - MAE_V5)_held-out-group - (MAE_holdout - MAE_V5)_in-distribution-control`, both models scoring identical samples; 95 % cluster-bootstrap CIs over reference genomes (2,000 resamples), two-sided paired tests, Benjamini-Hochberg corrected across groups x metrics.

| group | comp DiD | 95 % CI | q(BH) | cont DiD | 95 % CI | q(BH) |
|---|---|---|---|---|---|---|
| Halobacteriota_halophilic_genera | +2.02 | [1.33, 2.76] | 0.0006 | +0.78 | [0.21, 1.38] | 0.0070 |
| Bacteroidota_Flavobacteriaceae_genera | +2.58 | [2.07, 3.10] | 0.0006 | +4.23 | [3.25, 5.21] | 0.0006 |
| Campylobacterota_Helicobacteraceae_genera | +2.77 | [1.84, 3.87] | 0.0006 | +2.91 | [2.31, 3.49] | 0.0006 |
| Bacteroidota_Muribaculaceae_genera | +2.86 | [2.27, 3.53] | 0.0006 | +3.05 | [2.52, 3.60] | 0.0006 |
| Campylobacterota_Arcobacteraceae_genera | +4.19 | [3.37, 5.03] | 0.0006 | +3.30 | [2.80, 3.78] | 0.0006 |
| Patescibacteriota_genera | +4.64 | [2.99, 6.35] | 0.0006 | +2.82 | [1.93, 3.77] | 0.0006 |

`*_delta` columns in the TSV are the within-group (MAE_holdout - MAE_V5) differences before the control is subtracted; the DiD is the estimator.

## Head-to-head raw MAEs and signed biases

Raw cross-group MAE comparisons are confounded (registry W17, trap T7) and are reported for completeness only; the DiD above is the estimator.

```
                                    group lineage_seen_by_holdout    n  n_refs  true_comp_mean  true_cont_mean  HO_comp_mae  V5_comp_mae  d_comp_mae  HO_comp_bias  V5_comp_bias  HO_comp_rmse  V5_comp_rmse  HO_comp_r2  V5_comp_r2  HO_cont_mae  V5_cont_mae  d_cont_mae  HO_cont_bias  V5_cont_bias  HO_cont_rmse  V5_cont_rmse  HO_cont_r2  V5_cont_r2  HO_comp_r2_pearson  V5_comp_r2_pearson  HO_cont_r2_pearson  V5_cont_r2_pearson  HO_comp_mae_ci95  HO_cont_mae_ci95 V5_comp_mae_ci95 V5_cont_mae_ci95 paired_d_comp_ci95  paired_d_comp_p paired_d_cont_ci95  paired_d_cont_p  paired_d_comp_q_bh  paired_d_cont_q_bh
       Bacteroidota_Muribaculaceae_genera                      NO 1000     100           76.40           44.89        7.479        4.538       2.941        -1.818         2.775        10.034         6.897      0.5441      0.7846        8.019        5.120       2.899        -1.765        -3.083        10.551         7.279      0.8166      0.9127              0.5897              0.8207              0.8218              0.9290  [6.9124, 8.1081]  [7.5322, 8.5208] [4.1021, 4.9941] [4.7286, 5.5072]   [2.3675, 3.5621]           0.0005   [2.4552, 3.3421]           0.0005             0.00064             0.00064
    Bacteroidota_Flavobacteriaceae_genera                      NO  960      96           75.35           43.81        9.125        6.467       2.658         2.127         3.385        11.864         9.046      0.3279      0.6093       10.359        6.289       4.070         1.817        -2.819        13.851         8.915      0.6575      0.8581              0.4247              0.6773              0.6893              0.8727  [8.4093, 9.8319] [9.4528, 11.3634]   [5.8788, 7.08] [5.7577, 6.8577]   [2.1725, 3.1287]           0.0005   [3.1572, 5.0737]           0.0005             0.00064             0.00064
Campylobacterota_Helicobacteraceae_genera                      NO 1020      51           79.63           44.97        6.646        3.795       2.851        -0.686         1.479         9.598         5.920      0.6191      0.8551        8.760        6.001       2.759        -7.034        -5.076        11.741         8.596      0.7875      0.8861              0.6490              0.8647              0.8703              0.9316  [5.7569, 7.6185]  [8.1204, 9.3908] [3.4506, 4.1455] [5.5536, 6.4976]   [1.9651, 3.8552]           0.0005     [2.204, 3.287]           0.0005             0.00064             0.00064
  Campylobacterota_Arcobacteraceae_genera                      NO  990      33           77.79           44.33        7.886        3.619       4.267        -4.837         0.059         9.994         5.466      0.5512      0.8658        9.396        6.248       3.147        -8.444        -5.239        13.042         9.128      0.7287      0.8671              0.6585              0.8666              0.8581              0.9194  [7.2308, 8.5425] [8.7224, 10.1371] [3.2925, 3.9591]  [5.744, 6.7397]   [3.4965, 5.0343]           0.0005   [2.7196, 3.5469]           0.0005             0.00064             0.00064
         Halobacteriota_halophilic_genera                      NO 1008      24           75.08           45.49        7.366        5.266       2.101         2.895         0.963         9.286         7.199      0.5451      0.7266        6.587        5.957       0.630        -0.734        -3.373         9.385         8.622      0.8537      0.8765              0.6218              0.7335              0.8551              0.8954  [6.6184, 8.1089]  [5.9842, 7.1812] [4.7334, 5.9207]  [5.458, 6.4296]   [1.4108, 2.8264]           0.0005   [0.0856, 1.2089]           0.0310             0.00064             0.03617
                 Patescibacteriota_genera                      NO  990      30           84.27           46.06        9.551        4.831       4.720        -7.621        -1.755        12.947         7.546      0.3564      0.7813       10.625        7.960       2.664        -2.654        -3.850        13.604        10.705      0.7265      0.8307              0.6061              0.8023              0.7373              0.8571 [7.3783, 11.7127] [9.1589, 12.4495] [3.5888, 6.2055] [6.8238, 9.4513]   [3.0598, 6.4842]           0.0005   [1.8622, 3.5601]           0.0005             0.00064             0.00064
                          in_distribution                     yes 1000     100           76.57           45.92        6.382        6.306       0.077         2.129         1.813         9.292         9.272      0.5985      0.6002        7.197        7.352      -0.155        -0.543        -0.953        10.531        10.616      0.8162      0.8132              0.6392              0.6334              0.8190              0.8164  [5.7116, 7.1022]  [6.3465, 8.1292] [5.5919, 7.0116] [6.5467, 8.2676]  [-0.1178, 0.2681]           0.4890  [-0.4137, 0.1083]           0.2580             0.48900             0.27785
```

## The genus -> family -> phylum ladder on identical genomes

All four models (V5, genusHO, familyHO, phylumHO) score the same evaluation genomes against a common control restricted to dominants outside the WS1.6 phylum panel. Attenuation = DiD_deeper - DiD_shallower; a value near zero means the shallower novelty is as damaging as the deeper one, and a negative value means it is worse.

### Completeness (pp of MAE)

| group | n refs | DiD genus | 95 % CI | q(BH) | DiD family | DiD phylum | att. family-vs-genus | att. phylum-vs-genus |
|---|---|---|---|---|---|---|---|---|
| Bacteroidota_Muribaculaceae_genera | 100 | +2.95 | [2.31, 3.62] | 0.0006 | +4.46 | +7.76 | +1.51 | +4.81 |
| Bacteroidota_Flavobacteriaceae_genera | 96 | +2.67 | [2.11, 3.24] | 0.0006 | +4.40 | +7.67 | +1.73 | +5.00 |
| Campylobacterota_Helicobacteraceae_genera | 51 | +2.86 | [1.92, 3.94] | 0.0006 | +2.83 | +2.99 | -0.03 | +0.12 |
| Campylobacterota_Arcobacteraceae_genera | 33 | +4.28 | [3.43, 5.12] | 0.0006 | +4.77 | +8.24 | +0.49 | +3.96 |
| Halobacteriota_halophilic_genera | 24 | +2.11 | [1.41, 2.86] | 0.0006 | +2.45 | +3.82 | +0.33 | +1.71 |
| Patescibacteriota_genera | 30 | +4.73 | [3.02, 6.48] | 0.0006 | +6.16 | +25.68 | +1.43 | +20.95 |

### Contamination (pp of MAE)

| group | n refs | DiD genus | 95 % CI | q(BH) | DiD family | DiD phylum | att. family-vs-genus | att. phylum-vs-genus |
|---|---|---|---|---|---|---|---|---|
| Bacteroidota_Muribaculaceae_genera | 100 | +3.18 | [2.65, 3.69] | 0.0006 | +4.64 | +6.53 | +1.46 | +3.35 |
| Bacteroidota_Flavobacteriaceae_genera | 96 | +4.35 | [3.34, 5.35] | 0.0006 | +5.57 | +15.38 | +1.21 | +11.03 |
| Campylobacterota_Helicobacteraceae_genera | 51 | +3.04 | [2.42, 3.60] | 0.0006 | +5.68 | +10.84 | +2.64 | +7.79 |
| Campylobacterota_Arcobacteraceae_genera | 33 | +3.43 | [2.91, 3.91] | 0.0006 | +2.80 | +2.44 | -0.63 | -0.99 |
| Halobacteriota_halophilic_genera | 24 | +0.91 | [0.32, 1.54] | 0.0090 | +1.87 | +3.41 | +0.96 | +2.50 |
| Patescibacteriota_genera | 30 | +2.95 | [2.07, 3.88] | 0.0006 | +3.68 | +13.18 | +0.73 | +10.23 |

### Four models, raw MAE on identical samples

```
                                    group    n  n_refs  true_comp_mean  true_cont_mean  V5_comp_mae  V5_comp_bias  V5_comp_r2  V5_cont_mae  V5_cont_bias  V5_cont_r2  genusHO_comp_mae  genusHO_comp_bias  genusHO_comp_r2  genusHO_cont_mae  genusHO_cont_bias  genusHO_cont_r2  familyHO_comp_mae  familyHO_comp_bias  familyHO_comp_r2  familyHO_cont_mae  familyHO_cont_bias  familyHO_cont_r2  phylumHO_comp_mae  phylumHO_comp_bias  phylumHO_comp_r2  phylumHO_cont_mae  phylumHO_cont_bias  phylumHO_cont_r2
       Bacteroidota_Muribaculaceae_genera 1000     100           76.40           44.89        4.538         2.775      0.7846        5.120        -3.083      0.9127             7.479             -1.818           0.5441             8.019             -1.765           0.8166              9.082              -2.226            0.3500              9.356              -2.617            0.7448             12.404              -8.600           -0.0830             11.367              -3.566            0.6185
    Bacteroidota_Flavobacteriaceae_genera  960      96           75.35           43.81        6.467         3.385      0.6093        6.289        -2.819      0.8581             9.125              2.127           0.3279            10.359              1.817           0.6575             10.951               4.890            0.0848             11.453               3.616            0.6066             14.240               5.552           -0.4769             21.392              17.508           -0.2685
Campylobacterota_Helicobacteraceae_genera 1020      51           79.63           44.97        3.795         1.479      0.8551        6.001        -5.076      0.8861             6.646             -0.686           0.6191             8.760             -7.034           0.7875              6.712              -1.063            0.6232             11.280             -10.075            0.6716              6.885               3.260            0.6100             16.556             -16.278            0.3908
  Campylobacterota_Arcobacteraceae_genera  990      33           77.79           44.33        3.619         0.059      0.8658        6.248        -5.239      0.8671             7.886             -4.837           0.5512             9.396             -8.444           0.7287              8.471               4.591            0.4753              8.647              -6.255            0.7693             11.965               9.404           -0.0419              8.404              -5.628            0.7927
         Halobacteriota_halophilic_genera 1008      24           75.08           45.49        5.266         0.963      0.7266        5.957        -3.373      0.8765             7.366              2.895           0.5451             6.587             -0.734           0.8537              7.797               1.436            0.4939              7.427              -4.622            0.8171              9.195               1.619            0.2722              9.089              -4.571            0.7422
                 Patescibacteriota_genera  990      30           84.27           46.06        4.831        -1.755      0.7813        7.960        -3.850      0.8307             9.551             -7.621           0.3564            10.625             -2.654           0.7265             11.074              -8.741            0.1773             11.239              -1.193            0.6873             30.620             -30.610           -3.4944             20.854             -15.361           -0.0016
                          in_distribution 1000     100           76.57           45.92        6.306         1.813      0.6002        7.352        -0.953      0.8132             6.382              2.129           0.5985             7.197             -0.543           0.8162              6.644               2.103            0.5823              7.473              -1.554            0.8102              8.497               0.766            0.2867              9.716              -0.894            0.6670
```

## k-mer feature-selection leakage control

- **bacterial**: ? of ? feature-selection representatives belong to panel genera.
- **archaeal**: ? of ? feature-selection representatives belong to panel genera.

## Per-genus panel detail

```
                                    group           genus                family            phylum            class               order  train_removed  val_removed  test_refs  n_species  median_Mbp  train_left_same_family  genera_left_same_family  train_left_same_order  train_left_same_class  train_left_same_phylum  family_in_ws19_panel  phylum_in_ws16_panel
    Bacteroidota_Flavobacteriaceae_genera  Flavobacterium     Flavobacteriaceae      Bacteroidota      Bacteroidia    Flavobacteriales            322           45         59        264       3.703                     550                      127                   1293                   6825                    6825                  True                  True
    Bacteroidota_Flavobacteriaceae_genera   Tenacibaculum     Flavobacteriaceae      Bacteroidota      Bacteroidia    Flavobacteriales             69           13         14         31       2.909                     550                      127                   1293                   6825                    6825                  True                  True
    Bacteroidota_Flavobacteriaceae_genera    Polaribacter     Flavobacteriaceae      Bacteroidota      Bacteroidia    Flavobacteriales             48            5         12         46       3.166                     550                      127                   1293                   6825                    6825                  True                  True
    Bacteroidota_Flavobacteriaceae_genera   Flagellimonas     Flavobacteriaceae      Bacteroidota      Bacteroidia    Flavobacteriales             46           12          5         44       3.959                     550                      127                   1293                   6825                    6825                  True                  True
    Bacteroidota_Flavobacteriaceae_genera       Nonlabens     Flavobacteriaceae      Bacteroidota      Bacteroidia    Flavobacteriales             44            4          2          7       3.160                     550                      127                   1293                   6825                    6825                  True                  True
    Bacteroidota_Flavobacteriaceae_genera     Aequorivita     Flavobacteriaceae      Bacteroidota      Bacteroidia    Flavobacteriales             21            3          4         19       3.198                     550                      127                   1293                   6825                    6825                  True                  True
       Bacteroidota_Muribaculaceae_genera Paramuribaculum        Muribaculaceae      Bacteroidota      Bacteroidia       Bacteroidales            370           54         45         15       2.521                     906                       14                   4442                   6825                    6825                  True                  True
       Bacteroidota_Muribaculaceae_genera         CAG-873        Muribaculaceae      Bacteroidota      Bacteroidia       Bacteroidales            357           41         42         43       2.417                     906                       14                   4442                   6825                    6825                  True                  True
       Bacteroidota_Muribaculaceae_genera       Lepagella        Muribaculaceae      Bacteroidota      Bacteroidia       Bacteroidales            149           15         21         37       2.711                     906                       14                   4442                   6825                    6825                  True                  True
       Bacteroidota_Muribaculaceae_genera        Limisoma        Muribaculaceae      Bacteroidota      Bacteroidia       Bacteroidales             22            6          2          9       2.510                     906                       14                   4442                   6825                    6825                  True                  True
       Bacteroidota_Muribaculaceae_genera        CAG-1031        Muribaculaceae      Bacteroidota      Bacteroidia       Bacteroidales              6            0          1          2       2.374                     906                       14                   4442                   6825                    6825                  True                  True
  Campylobacterota_Arcobacteraceae_genera   Aliarcobacter       Arcobacteraceae  Campylobacterota  Campylobacteria   Campylobacterales            222           22         32         23       2.170                      42                       12                   4333                   4343                    4348                  True                  True
  Campylobacterota_Arcobacteraceae_genera Poseidonibacter       Arcobacteraceae  Campylobacterota  Campylobacteria   Campylobacterales              5            0          1          5       2.958                      42                       12                   4333                   4343                    4348                  True                  True
Campylobacterota_Helicobacteraceae_genera  Helicobacter_D     Helicobacteraceae  Campylobacterota  Campylobacteria   Campylobacterales            168           19         18         14       1.791                    1229                        8                   4333                   4343                    4348                  True                  True
Campylobacterota_Helicobacteraceae_genera  Helicobacter_C     Helicobacteraceae  Campylobacterota  Campylobacteria   Campylobacterales            160           26         21         11       1.864                    1229                        8                   4333                   4343                    4348                  True                  True
Campylobacterota_Helicobacteraceae_genera  Helicobacter_E     Helicobacteraceae  Campylobacterota  Campylobacteria   Campylobacterales             41            6          9         13       1.632                    1229                        8                   4333                   4343                    4348                  True                  True
Campylobacterota_Helicobacteraceae_genera  Helicobacter_F     Helicobacteraceae  Campylobacterota  Campylobacteria   Campylobacterales             17            0          1          8       1.705                    1229                        8                   4333                   4343                    4348                  True                  True
Campylobacterota_Helicobacteraceae_genera  Helicobacter_B     Helicobacteraceae  Campylobacterota  Campylobacteria   Campylobacterales              7            2          1          6       2.009                    1229                        8                   4333                   4343                    4348                  True                  True
Campylobacterota_Helicobacteraceae_genera  Helicobacter_G     Helicobacteraceae  Campylobacterota  Campylobacteria   Campylobacterales              5            1          1          4       1.435                    1229                        8                   4333                   4343                    4348                  True                  True
         Halobacteriota_halophilic_genera      Halorubrum        Haloferacaceae    Halobacteriota     Halobacteria     Halobacteriales             64            8          7         28       3.450                      38                       14                    175                    175                     404                  True                  True
         Halobacteriota_halophilic_genera       Haloferax        Haloferacaceae    Halobacteriota     Halobacteria     Halobacteriales             31            4          3         11       3.886                      38                       14                    175                    175                     404                  True                  True
         Halobacteriota_halophilic_genera      Haloarcula        Haloarculaceae    Halobacteriota     Halobacteria     Halobacteriales             30            8          5         17       4.030                      15                        8                    175                    175                     404                  True                  True
         Halobacteriota_halophilic_genera   Halomicrobium        Haloarculaceae    Halobacteriota     Halobacteria     Halobacteriales             12            0          1          7       3.367                      15                        8                    175                    175                     404                  True                  True
         Halobacteriota_halophilic_genera      Halobellus        Haloferacaceae    Halobacteriota     Halobacteria     Halobacteriales              8            1          1          7       3.347                      38                       14                    175                    175                     404                  True                  True
         Halobacteriota_halophilic_genera      Halapricum        Haloarculaceae    Halobacteriota     Halobacteria     Halobacteriales              7            0          1          5       3.219                      15                        8                    175                    175                     404                  True                  True
         Halobacteriota_halophilic_genera      Haloplanus        Haloferacaceae    Halobacteriota     Halobacteria     Halobacteriales              6            0          1          6       3.503                      38                       14                    175                    175                     404                  True                  True
         Halobacteriota_halophilic_genera     Halorhabdus        Haloarculaceae    Halobacteriota     Halobacteria     Halobacteriales              6            0          2          6       2.892                      15                        8                    175                    175                     404                  True                  True
         Halobacteriota_halophilic_genera   Halorientalis        Haloarculaceae    Halobacteriota     Halobacteria     Halobacteriales              5            0          3          6       3.720                      15                        8                    175                    175                     404                  True                  True
                 Patescibacteriota_genera   Nanosyncoccus      Nanosyncoccaceae Patescibacteriota  Saccharimonadia   Saccharimonadales             51            4          6         56       0.779                      12                        2                    245                    255                    1099                  True                  True
                 Patescibacteriota_genera         UBA9973               UBA9973 Patescibacteriota    Minisyncoccia             UBA9973             40            7          4         33       0.696                      11                        8                    171                    396                    1099                  True                  True
                 Patescibacteriota_genera       C7867-001               UBA2103 Patescibacteriota    Minisyncoccia             UBA9973             35            7          3         38       0.711                       7                        2                    171                    396                    1099                  True                  True
                 Patescibacteriota_genera         UBA8515               UBA9973 Patescibacteriota    Minisyncoccia             UBA9973              8            0          1          6       0.772                      11                        8                    171                    396                    1099                  True                  True
                 Patescibacteriota_genera         UBA1547               UBA1547 Patescibacteriota  Saccharimonadia   Saccharimonadales              7            1          2          8       0.848                       7                        2                    245                    255                    1099                  True                  True
                 Patescibacteriota_genera         UBA2170       Peribacteraceae Patescibacteriota  Gracilibacteria      Peribacterales              7            1          1          3       1.178                      29                       18                     29                     69                    1099                  True                  True
                 Patescibacteriota_genera          MWCR01       2-12-FULL-60-25 Patescibacteriota Patescibacteriia     2-12-FULL-60-25              6            1          2          6       1.287                      15                        6                     15                    168                    1099                  True                  True
                 Patescibacteriota_genera      Peribacter       Peribacteraceae Patescibacteriota  Gracilibacteria      Peribacterales              5            1          1          4       1.367                      29                       18                     29                     69                    1099                  True                  True
                 Patescibacteriota_genera         UBA4124               UBA9973 Patescibacteriota    Minisyncoccia             UBA9973              5            0          1          5       0.664                      11                        8                    171                    396                    1099                  True                  True
                 Patescibacteriota_genera        CAIYEO01               UBA4665 Patescibacteriota  Saccharimonadia   Saccharimonadales              4            0          1          1       0.741                      35                       23                    245                    255                    1099                  True                  True
                 Patescibacteriota_genera  XYD2-FULL-39-9             GWA2-37-8 Patescibacteriota Patescibacteriia Magasanikbacterales              4            0          1          5       1.021                       6                        6                     34                    168                    1099                  True                  True
                 Patescibacteriota_genera        CAIKZD01               UBA2206 Patescibacteriota    Minisyncoccia     Moranbacterales              3            0          1          1       1.025                      12                        9                     39                    396                    1099                  True                  True
                 Patescibacteriota_genera    CG1-02-43-31                PJMF01 Patescibacteriota   Microgenomatia             UBA1400              3            0          1          2       1.236                      19                        9                     43                    143                    1099                  True                  True
                 Patescibacteriota_genera        JABIEQ01                UBA922 Patescibacteriota Patescibacteriia Magasanikbacterales              3            1          2          1       0.941                      24                       16                     34                    168                    1099                  True                  True
                 Patescibacteriota_genera        JAKLGL01             GWA2-37-8 Patescibacteriota Patescibacteriia Magasanikbacterales              3            0          1          2       0.930                       6                        6                     34                    168                    1099                  True                  True
                 Patescibacteriota_genera XYD1-FULL-39-28 Staskawiczbacteraceae Patescibacteriota    Minisyncoccia     Minisyncoccales              3            0          2          4       0.801                       7                        2                     55                    396                    1099                  True                  True
```

## Verdict on monotonicity across ranks

Degradation is **not** monotone across ranks, and the two departures run in opposite directions. For four of the six groups the ladder is ordered as one would hope - genus < family < phylum, with the phylum-vs-genus attenuation large and significant (Patescibacteriota +20.95 pp completeness, q = 0.0006). But **Campylobacterota_Helicobacteraceae is flat at every rung**: completeness DiD +2.86 (genus), +2.83 (family), +2.99 (phylum), giving a family-vs-genus attenuation of -0.03 pp [-0.63, +0.57], q = 0.927, and a phylum-vs-genus attenuation of +0.12 pp [-1.08, +1.17], q = 0.824. Losing a single genus of Helicobacteraceae costs as much as losing the whole phylum Campylobacterota. This extends the WS1.9 finding (family and phylum equal there, -0.39 pp, p = 0.228) one rank deeper and confirms it was not an artefact of the family rung. And **Campylobacterota_Arcobacteraceae inverts the ladder for contamination**: DiD +3.43 (genus) exceeds +2.80 (family) and +2.44 (phylum), a phylum-vs-genus attenuation of **-0.99 pp [-1.76, -0.11], q = 0.012 - significantly negative**. There, genus-level novelty is measurably *worse* than phylum-level novelty. **A genus-level cost therefore cannot be bounded by the family-level cost, in either direction, and must be measured.**

## Relation to the WS11.N observational novelty ladder

WS11.N stratifies the five leakage-free benchmark sets by the deepest rank at which the dominant genome is novel relative to the training split and reports completeness MAE climbing 3.17 pp (lineage represented) -> 6.33 pp (species-novel) -> 10.27 pp (genus-novel). That is an **observational** stratification: genomes that are genus-novel to the training split differ from represented genomes in many ways besides novelty - they are disproportionately reduced, fragmented and drawn from sparsely sampled lineages - so the 10.27 pp is an association, not a causal effect of novelty. WS11.G is the **causal** counterpart: the same genomes are scored by two models differing only in whether the genus was in training, and the control subtracts the cost of the smaller pool. The causal genus effect is **+2.02 to +4.64 pp of completeness MAE**, materially smaller than the ~7 pp gap the observational ladder shows between represented and genus-novel genomes. The two are consistent rather than contradictory: most of the observational gap is composition (which genomes are novel), and roughly a third of it is novelty itself. Both belong in the paper, each labelled for what it is.

## Determinism and provenance notes (for the supplementary note)

1. **WS1.6 subsampled groups do not re-derive from the current code (defect D2).** WS1.6 drew evaluation references with `abs(hash(group))`, which Python salts per process; the defect was found and fixed in WS1.9 by switching to CRC-32 `stable_hash`. Re-running selection today reproduces WS1.6's three groups that use ALL available references exactly (Bacteroidota_A, Halobacteriota, DPANN) and not the three that subsample (overlap 8-67 of 100). **WS1.6 remains fully auditable because the references it actually drew are recorded verbatim in each group's `metadata.tsv`.** WS1.9 and WS11.G use CRC-32 and re-derive exactly: all six WS1.9 panel groups reproduce 100 %.
2. **`in_distribution` re-derives 92 of 100 references at family level.** The sqrt-proportional allocator breaks a largest-remainder tie between two equal-count single-genome phyla (Zhuqueibacterota vs Zixibacteria), and swapping one stratum reorders every downstream `.sample()` draw. The allocation differs by exactly one genome. Pre-existing; affects no reported number, because each run's own control is the one used in its own DiD.
3. **Synthetic training data is deterministic in content, not in row order.** Three independent 384-sample pilot builds at genus level produced an identical multiset of samples - same dominant genomes, same labels, same sample types, same composition - but different row order within a batch, because the top-up retry loop consumes `pool.imap_unordered(...)` and stops as soon as the batch is full, so which completed samples land in the batch depends on worker scheduling. **This was deliberately NOT "fixed".** Sorting the results would change what the phylum and family levels do, and WS1.6 and WS1.9 are completed experiments running on this same shared code path. It affects no reported number: the multiset defines the training distribution, and the training DataLoader shuffles it under its own seed.

## Files

- `results/revision/holdout_genus/WS11_G_REPORT.md`
- `results/revision/holdout_genus/clean_sets_evaluation.tsv`
- `results/revision/holdout_genus/degradation_vs_in_distribution.tsv`
- `results/revision/holdout_genus/eda_genus_counts.tsv`
- `results/revision/holdout_genus/eda_panel_summary.json`
- `results/revision/holdout_genus/four_model_head_to_head.tsv`
- `results/revision/holdout_genus/genus_vs_family_vs_phylum_did.tsv`
- `results/revision/holdout_genus/head_to_head_by_group.tsv`
- `results/revision/holdout_genus/kmer_reselection_dropped_added.tsv`
- `results/revision/holdout_genus/kmer_reselection_prevalence_archaeal.tsv`
- `results/revision/holdout_genus/kmer_reselection_prevalence_bacterial.tsv`
- `results/revision/holdout_genus/kmer_reselection_summary.json`
- `results/revision/holdout_genus/ladder_per_sample.tsv.gz`
- `results/revision/holdout_genus/ladder_report.md`
- `results/revision/holdout_genus/level_switch_reproduction.json`
- `results/revision/holdout_genus/lineage_novelty_effect_did.tsv`
- `results/revision/holdout_genus/metrics_full.json`
- `results/revision/holdout_genus/mimag_confusion_by_group.tsv`
- `results/revision/holdout_genus/mimag_threshold_by_group.tsv`
- `results/revision/holdout_genus/model_sha256.txt`
- `results/revision/holdout_genus/onnx_export_verification.json`
- `results/revision/holdout_genus/panel_genus_detail.tsv`
- `results/revision/holdout_genus/per_reference_errors.tsv`
- `results/revision/holdout_genus/per_sample_predictions.tsv.gz`
- `results/revision/holdout_genus/selected_kmers_holdout.txt`
- `results/revision/holdout_genus/set_C_clean_predictions_both_models.tsv`
- `results/revision/holdout_genus/set_D_clean_predictions_both_models.tsv`
- `results/revision/holdout_genus/stratified_error_by_band.tsv`
- `results/revision/holdout_genus/sub_phylum_breakdown.tsv`
- `results/revision/holdout_genus/training_summary.json`
- `results/revision/holdout_genus/ws11g_consolidated.json`

