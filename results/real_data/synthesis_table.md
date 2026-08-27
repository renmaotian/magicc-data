# WS3 Track A — real-data performance with genuine ground truth

_Generated 2026-07-26T22:34:07.984261+00:00_

Denominators (identical for ground truth and for every tool): completeness = retained dominant-organism bp / dominant reference FULL length x 100; contamination = contaminant bp / dominant reference FULL length x 100. MIMAG-inspired thresholds (completeness/contamination only): high quality >=90 % completeness AND <5 % contamination; medium quality >=50 % AND <10 %. R2 is the coefficient of determination (1 - SS_res/SS_tot), never squared Pearson.

| dataset                                                     | tool          | n   | n_clusters | comp_MAE | comp_MAE_95CI      | comp_bias | comp_R2_CoD | cont_MAE | cont_MAE_95CI      | cont_bias | cont_R2_CoD |
|-------------------------------------------------------------|---------------|-----|------------|----------|--------------------|-----------|-------------|----------|--------------------|-----------|-------------|
| Meslier MOCK1 (leakage-free, >=50 %, ORF-intact assemblies) | MAGICC V5     | 93  | 18         | 3.955    | [2.6956, 5.2865]   | 2.775     | 0.7331      | 0.866    | [0.4418, 1.3409]   | 0.6919    | -11.47      |
| Meslier MOCK1 (leakage-free, >=50 %, ORF-intact assemblies) | CheckM2 1.0.1 | 93  | 18         | 4.534    | [3.1893, 6.138]    | -2.182    | 0.5797      | 2.478    | [1.7305, 3.2614]   | 2.462     | -47.01      |
| Meslier MOCK1 (leakage-free, >=50 %, ORF-intact assemblies) | CoCoPyE 0.5.0 | 93  | 18         | 4.218    | [2.7305, 5.9553]   | 4.11      | 0.6145      | 8.197    | [5.1715, 11.5311]  | 8.167     | -669.7      |
| Meslier MOCK1 (leakage-free, >=50 %, ORF-intact assemblies) | DeepCheck     | 93  | 18         | 5.657    | [3.5243, 8.2106]   | -4.772    | 0.255       | 1.684    | [1.1243, 2.3044]   | 1.559     | -25.26      |
| Meslier MOCK1 (leakage-free, >=50 %, all 7 assemblies)      | MAGICC V5     | 108 | 18         | 4.196    | [3.0508, 5.4291]   | 1.96      | 0.6538      | 0.8793   | [0.465, 1.3233]    | 0.6912    | -10.42      |
| Meslier MOCK1 (leakage-free, >=50 %, all 7 assemblies)      | CheckM2 1.0.1 | 108 | 18         | 11.01    | [9.2807, 12.9716]  | -8.982    | -3.337      | 2.635    | [1.9963, 3.3101]   | 2.62      | -43.94      |
| Meslier MOCK1 (leakage-free, >=50 %, all 7 assemblies)      | CoCoPyE 0.5.0 | 108 | 18         | 3.939    | [2.5859, 5.502]    | 3.352     | 0.6336      | 10.07    | [7.5058, 12.8157]  | 10.05     | -744        |
| Meslier MOCK1 (leakage-free, >=50 %, all 7 assemblies)      | DeepCheck     | 108 | 18         | 13.25    | [10.7154, 16.0897] | -12.49    | -4.956      | 1.863    | [1.4322, 2.3328]   | 1.719     | -25.32      |
| Meslier MOCK1 (indel-dense MinION assembly only)            | MAGICC V5     | 39  | 39         | 4.905    | [2.8494, 7.4549]   | -2.92     | 0.4043      | 1.572    | [0.7579, 2.5755]   | 1.34      | -8.688      |
| Meslier MOCK1 (indel-dense MinION assembly only)            | CheckM2 1.0.1 | 39  | 39         | 50.69    | [45.5215, 55.3439] | -50.69    | -20.69      | 3.45     | [2.8736, 4.0496]   | 3.412     | -13.38      |
| Meslier MOCK1 (indel-dense MinION assembly only)            | CoCoPyE 0.5.0 | 39  | 39         | 3.115    | [2.2177, 4.1123]   | -1.834    | 0.8591      | 22.17    | [20.6778, 23.7017] | 22.17     | -473.8      |
| Meslier MOCK1 (indel-dense MinION assembly only)            | DeepCheck     | 39  | 39         | 56.83    | [51.7475, 61.7072] | -56.83    | -25.76      | 4.195    | [3.24, 5.2562]     | 4.095     | -24.7       |
| Meslier MOCK1 (all organisms, >=50 %, all 7 assemblies)     | MAGICC V5     | 308 | 52         | 4.135    | [2.877, 5.9859]    | 1.272     | 0.5903      | 1.377    | [0.7935, 2.2583]   | 1.221     | -18.8       |
| Meslier MOCK1 (all organisms, >=50 %, all 7 assemblies)     | CheckM2 1.0.1 | 308 | 52         | 11.96    | [10.563, 13.4025]  | -10.25    | -2.36       | 1.891    | [1.5366, 2.2878]   | 1.837     | -15.89      |
| Meslier MOCK1 (all organisms, >=50 %, all 7 assemblies)     | CoCoPyE 0.5.0 | 308 | 52         | 4.651    | [3.5881, 5.7676]   | 2.4       | 0.626       | 10.23    | [8.5202, 12.0357]  | 10.22     | -455        |
| Meslier MOCK1 (all organisms, >=50 %, all 7 assemblies)     | DeepCheck     | 308 | 52         | 13.98    | [12.3291, 15.6535] | -13.17    | -3.294      | 1.85     | [1.5451, 2.1565]   | 1.694     | -16.54      |
| Zymo isolate drafts (n=8)                                   | MAGICC V5     | 8   | 8          | 1.752    | [1.0462, 2.8506]   | 1.752     | -1.706      | 0.4317   | [0.277, 0.5801]    | 0.4317    | -154.1      |
| Zymo isolate drafts (n=8)                                   | CheckM2 1.0.1 | 8   | 8          | 1.851    | [1.1712, 2.9119]   | 1.851     | -1.852      | 0.1669   | [0.0402, 0.3747]   | 0.1625    | -65.43      |
| Zymo isolate drafts (n=8)                                   | CoCoPyE 0.5.0 | 8   | 8          | 1.742    | [1.0765, 2.8103]   | 1.742     | -1.665      | 0.4713   | [0.0544, 1.147]    | 0.4412    | -640.9      |
| Zymo isolate drafts (n=8)                                   | DeepCheck     | 8   | 8          | 0.9794   | [0.3292, 1.9789]   | 0.961     | -0.3892     | 0.2836   | [0.2214, 0.3528]   | -0.1272   | -58.03      |
