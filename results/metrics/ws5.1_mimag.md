# WS5.1 - MIMAG-inspired quality classification, per benchmark set

MIMAG-inspired quality classes, applied to completeness and contamination ONLY: **high** = completeness >= 90% AND contamination < 5%; **medium** = completeness >= 50% AND contamination < 10% (and not high); **low** = all others. NOTE: the full MIMAG standard (Bowers et al. 2017) additionally requires 23S/16S/5S rRNA genes and >= 18 tRNAs for the high-quality tier; those criteria cannot be evaluated from completeness/contamination estimates and are NOT applied here, so the tier names are MIMAG-inspired rather than strict MIMAG. The submitted manuscript is internally inconsistent about the completeness boundary (line 53 uses '>= 90% complete', the Results text uses '> 90% completeness'); the inclusive form '>= 90%' is used throughout, matching Bowers et al.

Denominators: completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

All 95% confidence intervals are percentile bootstrap intervals (2000 replicates) with **clusters resampled by dominant reference genome** (`dominant_accession`), so genomes simulated from the same reference are resampled together.

Sets flagged **SUPERSEDED** were built with dominant genomes drawn from the training split (Set C: 1000/1000 dominants in train; Set D: 796 train / 107 val / 97 test) and are reported only for transparency; the clean replacements are `set_C_clean` and `set_D_clean`.

## set_A_v2 - Set A (completeness gradient, 0% contamination)

*n* = 1000, clusters (distinct dominant reference genomes) = 798. Design: 1,000 NCBI finished test-split dominants; completeness 50-100%; no contaminants

**True class balance:** high = 276 (27.6%), medium = 724 (72.4%), low = 0 (0.0%)

> **Class(es) absent from the ground truth by design: low.** The 3-class macro F1 therefore averages a structurally undefined class scored as 0 and is bounded above by 2/3; the `macro F1 (observed classes)` column is the interpretable summary for this set. The submitted manuscript's Table S3 used the 3-class form.

### Overall classification performance (estimate [95% CI])

| tool                                                 | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | 0.955 [0.941, 0.969] | 0.633 [0.622, 0.643] | 0.949 [0.934, 0.964]        | 0.955 [0.941, 0.969] | 0.960 [0.948, 0.972] | 0.942 [0.923, 0.960] | 0.889 [0.855, 0.920] |
| MAGICC v4 (numbers in submitted manuscript)          | 0.917 [0.898, 0.935] | 0.600 [0.584, 0.614] | 0.899 [0.876, 0.921]        | 0.917 [0.898, 0.935] | 0.924 [0.906, 0.942] | 0.869 [0.841, 0.896] | 0.788 [0.743, 0.831] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | 0.965 [0.953, 0.976] | 0.640 [0.631, 0.649] | 0.960 [0.946, 0.973]        | 0.965 [0.953, 0.976] | 0.968 [0.957, 0.978] | 0.950 [0.934, 0.966] | 0.912 [0.884, 0.939] |
| CheckM2 1.0.1                                        | 0.943 [0.928, 0.956] | 0.633 [0.622, 0.642] | 0.949 [0.933, 0.962]        | 0.943 [0.928, 0.956] | 0.954 [0.941, 0.965] | 0.954 [0.940, 0.966] | 0.866 [0.829, 0.898] |
| CoCoPyE 0.5.0                                        | 0.943 [0.927, 0.957] | 0.625 [0.614, 0.636] | 0.938 [0.920, 0.954]        | 0.943 [0.927, 0.957] | 0.952 [0.938, 0.964] | 0.933 [0.912, 0.952] | 0.862 [0.824, 0.895] |
| DeepCheck                                            | 0.936 [0.920, 0.950] | 0.630 [0.620, 0.639] | 0.945 [0.930, 0.958]        | 0.936 [0.920, 0.950] | 0.949 [0.936, 0.961] | 0.949 [0.936, 0.961] | 0.851 [0.817, 0.885] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                                                 | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | high   | 276              | 269         | 0.937 [0.906, 0.963] | 0.913 [0.876, 0.947] | 0.925 [0.900, 0.947] |
| MAGICC v5 (released, v0.3.0)                         | low    | 0                | 11          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MAGICC v5 (released, v0.3.0)                         | medium | 724              | 720         | 0.976 [0.965, 0.987] | 0.971 [0.957, 0.983] | 0.974 [0.965, 0.982] |
| MAGICC v4 (numbers in submitted manuscript)          | high   | 276              | 222         | 0.946 [0.916, 0.973] | 0.761 [0.706, 0.814] | 0.843 [0.806, 0.878] |
| MAGICC v4 (numbers in submitted manuscript)          | low    | 0                | 22          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MAGICC v4 (numbers in submitted manuscript)          | medium | 724              | 756         | 0.935 [0.916, 0.954] | 0.977 [0.965, 0.987] | 0.955 [0.944, 0.966] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | high   | 276              | 261         | 0.969 [0.946, 0.989] | 0.917 [0.884, 0.948] | 0.942 [0.922, 0.962] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | low    | 0                | 7           | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | medium | 724              | 732         | 0.973 [0.961, 0.984] | 0.983 [0.973, 0.992] | 0.978 [0.970, 0.985] |
| CheckM2 1.0.1                                        | high   | 276              | 299         | 0.903 [0.867, 0.934] | 0.978 [0.959, 0.993] | 0.939 [0.916, 0.958] |
| CheckM2 1.0.1                                        | low    | 0                | 22          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CheckM2 1.0.1                                        | medium | 724              | 679         | 0.991 [0.984, 0.997] | 0.930 [0.910, 0.947] | 0.959 [0.948, 0.969] |
| CoCoPyE 0.5.0                                        | high   | 276              | 277         | 0.906 [0.871, 0.939] | 0.909 [0.871, 0.944] | 0.908 [0.879, 0.933] |
| CoCoPyE 0.5.0                                        | low    | 0                | 18          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CoCoPyE 0.5.0                                        | medium | 724              | 705         | 0.982 [0.970, 0.992] | 0.956 [0.940, 0.970] | 0.969 [0.959, 0.977] |
| DeepCheck                                            | high   | 276              | 301         | 0.897 [0.861, 0.929] | 0.978 [0.960, 0.993] | 0.936 [0.915, 0.954] |
| DeepCheck                                            | low    | 0                | 27          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| DeepCheck                                            | medium | 724              | 672         | 0.991 [0.983, 0.997] | 0.920 [0.899, 0.939] | 0.954 [0.943, 0.965] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 252  | 17     | 7   |
| medium      | 17   | 703    | 4   |
| low         | 0    | 0      | 0   |

**MAGICC v4 (numbers in submitted manuscript)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 210  | 49     | 17  |
| medium      | 12   | 707    | 5   |
| low         | 0    | 0      | 0   |

**MAGICC v3 (SUPERSEDED - uses 20 assembly statistics)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 253  | 20     | 3   |
| medium      | 8    | 712    | 4   |
| low         | 0    | 0      | 0   |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 270  | 6      | 0   |
| medium      | 29   | 673    | 22  |
| low         | 0    | 0      | 0   |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 251  | 13     | 12  |
| medium      | 26   | 692    | 6   |
| low         | 0    | 0      | 0   |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 270  | 6      | 0   |
| medium      | 31   | 666    | 27  |
| low         | 0    | 0      | 0   |

## set_B_v2 - Set B (contamination gradient, 100% completeness)

*n* = 1000, clusters (distinct dominant reference genomes) = 803. Design: 1,000 NCBI finished test-split dominants at 100% completeness; cross-phylum contamination 0-80%

**True class balance:** high = 200 (20.0%), medium = 2 (0.2%), low = 798 (79.8%)

> **Small-support class(es): medium (n=2).** Per-class F1 for these classes is estimated from fewer than 10 genomes; the CI is correspondingly wide and the macro average is dominated by sampling noise in that class. Individual data points are shown for these classes in the figures.

### Overall classification performance (estimate [95% CI])

| tool                                                 | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | 0.989 [0.982, 0.995] | 0.778 [0.654, 0.897] | 0.778 [0.654, 0.897]        | 0.989 [0.982, 0.995] | 0.991 [0.985, 0.996] | 0.983 [0.967, 0.992] | 0.966 [0.945, 0.984] |
| MAGICC v4 (numbers in submitted manuscript)          | 0.950 [0.937, 0.963] | 0.634 [0.607, 0.675] | 0.634 [0.607, 0.675]        | 0.950 [0.937, 0.963] | 0.963 [0.952, 0.974] | 0.752 [0.573, 0.931] | 0.845 [0.808, 0.880] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | 0.990 [0.984, 0.996] | 0.762 [0.657, 0.870] | 0.762 [0.657, 0.870]        | 0.990 [0.984, 0.996] | 0.993 [0.989, 0.997] | 0.986 [0.971, 0.994] | 0.969 [0.950, 0.987] |
| CheckM2 1.0.1                                        | 0.852 [0.830, 0.875] | 0.598 [0.580, 0.619] | 0.598 [0.580, 0.619]        | 0.852 [0.830, 0.875] | 0.892 [0.875, 0.910] | 0.772 [0.599, 0.945] | 0.655 [0.609, 0.701] |
| CoCoPyE 0.5.0                                        | 0.985 [0.977, 0.992] | 0.735 [0.659, 0.821] | 0.735 [0.659, 0.821]        | 0.985 [0.977, 0.992] | 0.990 [0.986, 0.995] | 0.990 [0.980, 0.995] | 0.955 [0.932, 0.976] |
| DeepCheck                                            | 0.671 [0.642, 0.700] | 0.499 [0.480, 0.517] | 0.499 [0.480, 0.517]        | 0.671 [0.642, 0.700] | 0.744 [0.718, 0.770] | 0.529 [0.517, 0.802] | 0.410 [0.370, 0.448] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                                                 | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | high   | 200              | 190         | 1.000 [1.000, 1.000] | 0.950 [0.918, 0.977] | 0.974 [0.957, 0.988] |
| MAGICC v5 (released, v0.3.0)                         | low    | 798              | 801         | 0.995 [0.990, 0.999] | 0.999 [0.996, 1.000] | 0.997 [0.994, 0.999] |
| MAGICC v5 (released, v0.3.0)                         | medium | 2                | 9           | 0.222 [0.000, 0.556] | 1.000 [0.000, 1.000] | 0.364 [0.000, 0.714] |
| MAGICC v4 (numbers in submitted manuscript)          | high   | 200              | 151         | 1.000 [1.000, 1.000] | 0.755 [0.696, 0.812] | 0.860 [0.821, 0.896] |
| MAGICC v4 (numbers in submitted manuscript)          | low    | 798              | 812         | 0.983 [0.973, 0.991] | 1.000 [1.000, 1.000] | 0.991 [0.986, 0.996] |
| MAGICC v4 (numbers in submitted manuscript)          | medium | 2                | 37          | 0.027 [0.000, 0.088] | 0.500 [0.000, 1.000] | 0.051 [0.000, 0.158] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | high   | 200              | 192         | 1.000 [1.000, 1.000] | 0.960 [0.931, 0.985] | 0.980 [0.964, 0.992] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | low    | 798              | 797         | 0.999 [0.996, 1.000] | 0.997 [0.994, 1.000] | 0.998 [0.996, 1.000] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | medium | 2                | 11          | 0.182 [0.000, 0.455] | 1.000 [0.000, 1.000] | 0.308 [0.000, 0.625] |
| CheckM2 1.0.1                                        | high   | 200              | 257         | 0.778 [0.730, 0.827] | 1.000 [1.000, 1.000] | 0.875 [0.844, 0.905] |
| CheckM2 1.0.1                                        | low    | 798              | 651         | 1.000 [1.000, 1.000] | 0.816 [0.789, 0.844] | 0.899 [0.882, 0.915] |
| CheckM2 1.0.1                                        | medium | 2                | 92          | 0.011 [0.000, 0.035] | 0.500 [0.000, 1.000] | 0.021 [0.000, 0.067] |
| CoCoPyE 0.5.0                                        | high   | 200              | 198         | 0.995 [0.983, 1.000] | 0.985 [0.966, 1.000] | 0.990 [0.978, 0.998] |
| CoCoPyE 0.5.0                                        | low    | 798              | 786         | 1.000 [1.000, 1.000] | 0.985 [0.976, 0.993] | 0.992 [0.988, 0.996] |
| CoCoPyE 0.5.0                                        | medium | 2                | 16          | 0.125 [0.000, 0.312] | 1.000 [0.000, 1.000] | 0.222 [0.000, 0.476] |
| DeepCheck                                            | high   | 200              | 328         | 0.607 [0.556, 0.659] | 0.995 [0.983, 1.000] | 0.754 [0.713, 0.793] |
| DeepCheck                                            | low    | 798              | 472         | 1.000 [1.000, 1.000] | 0.591 [0.558, 0.625] | 0.743 [0.716, 0.769] |
| DeepCheck                                            | medium | 2                | 200         | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 190  | 6      | 4   |
| medium      | 0    | 2      | 0   |
| low         | 0    | 1      | 797 |

**MAGICC v4 (numbers in submitted manuscript)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 151  | 36     | 13  |
| medium      | 0    | 1      | 1   |
| low         | 0    | 0      | 798 |

**MAGICC v3 (SUPERSEDED - uses 20 assembly statistics)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 192  | 7      | 1   |
| medium      | 0    | 2      | 0   |
| low         | 0    | 2      | 796 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 200  | 0      | 0   |
| medium      | 1    | 1      | 0   |
| low         | 56   | 91     | 651 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 197  | 3      | 0   |
| medium      | 0    | 2      | 0   |
| low         | 1    | 11     | 786 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 199  | 1      | 0   |
| medium      | 2    | 0      | 0   |
| low         | 127  | 199    | 472 |

## set_C - Set C (Patescibacteriota) - SUPERSEDED (training leakage)  **[SUPERSEDED - training-set leakage]**

*n* = 1000, clusters (distinct dominant reference genomes) = 1000. Design: 1,000 Patescibacteriota dominants drawn from train+val+test; completeness 50-100%, cross-phylum contamination 0-100%

**True class balance:** high = 34 (3.4%), medium = 83 (8.3%), low = 883 (88.3%)

### Overall classification performance (estimate [95% CI])

| tool                                                 | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | 0.958 [0.945, 0.970] | 0.852 [0.803, 0.894] | 0.852 [0.803, 0.894]        | 0.958 [0.945, 0.970] | 0.960 [0.948, 0.971] | 0.913 [0.876, 0.945] | 0.817 [0.765, 0.864] |
| MAGICC v4 (numbers in submitted manuscript)          | 0.964 [0.953, 0.975] | 0.872 [0.827, 0.916] | 0.872 [0.827, 0.916]        | 0.964 [0.953, 0.975] | 0.965 [0.955, 0.976] | 0.911 [0.869, 0.948] | 0.840 [0.793, 0.887] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | 0.971 [0.960, 0.981] | 0.892 [0.845, 0.933] | 0.892 [0.845, 0.933]        | 0.971 [0.960, 0.981] | 0.972 [0.962, 0.982] | 0.913 [0.866, 0.957] | 0.869 [0.822, 0.915] |
| CheckM2 1.0.1                                        | 0.287 [0.260, 0.314] | 0.282 [0.247, 0.315] | 0.282 [0.247, 0.315]        | 0.287 [0.260, 0.314] | 0.353 [0.321, 0.385] | 0.599 [0.552, 0.643] | 0.063 [0.043, 0.084] |
| CoCoPyE 0.5.0                                        | 0.798 [0.772, 0.822] | 0.420 [0.395, 0.448] | 0.420 [0.395, 0.448]        | 0.798 [0.772, 0.822] | 0.817 [0.793, 0.839] | 0.501 [0.466, 0.537] | 0.297 [0.231, 0.363] |
| DeepCheck                                            | 0.262 [0.237, 0.289] | 0.267 [0.216, 0.316] | 0.267 [0.216, 0.316]        | 0.262 [0.237, 0.289] | 0.326 [0.293, 0.359] | 0.443 [0.381, 0.506] | 0.033 [0.014, 0.052] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                                                 | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | high   | 34               | 48          | 0.688 [0.558, 0.816] | 0.971 [0.900, 1.000] | 0.805 [0.704, 0.889] |
| MAGICC v5 (released, v0.3.0)                         | low    | 883              | 863         | 0.995 [0.991, 0.999] | 0.973 [0.962, 0.983] | 0.984 [0.978, 0.989] |
| MAGICC v5 (released, v0.3.0)                         | medium | 83               | 89          | 0.742 [0.653, 0.833] | 0.795 [0.702, 0.875] | 0.767 [0.692, 0.833] |
| MAGICC v4 (numbers in submitted manuscript)          | high   | 34               | 41          | 0.756 [0.622, 0.886] | 0.912 [0.808, 1.000] | 0.827 [0.722, 0.914] |
| MAGICC v4 (numbers in submitted manuscript)          | low    | 883              | 868         | 0.994 [0.989, 0.999] | 0.977 [0.967, 0.987] | 0.986 [0.980, 0.991] |
| MAGICC v4 (numbers in submitted manuscript)          | medium | 83               | 91          | 0.769 [0.685, 0.857] | 0.843 [0.762, 0.918] | 0.805 [0.741, 0.866] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | high   | 34               | 35          | 0.829 [0.692, 0.944] | 0.853 [0.725, 0.967] | 0.841 [0.735, 0.923] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | low    | 883              | 871         | 0.995 [0.991, 0.999] | 0.982 [0.972, 0.991] | 0.989 [0.983, 0.993] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | medium | 83               | 94          | 0.798 [0.716, 0.879] | 0.904 [0.835, 0.965] | 0.847 [0.786, 0.904] |
| CheckM2 1.0.1                                        | high   | 34               | 162         | 0.198 [0.140, 0.263] | 0.941 [0.848, 1.000] | 0.327 [0.242, 0.412] |
| CheckM2 1.0.1                                        | low    | 883              | 205         | 0.990 [0.974, 1.000] | 0.230 [0.204, 0.256] | 0.373 [0.338, 0.408] |
| CheckM2 1.0.1                                        | medium | 83               | 633         | 0.082 [0.062, 0.104] | 0.627 [0.523, 0.729] | 0.145 [0.110, 0.180] |
| CoCoPyE 0.5.0                                        | high   | 34               | 0           | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CoCoPyE 0.5.0                                        | low    | 883              | 787         | 0.944 [0.928, 0.959] | 0.841 [0.817, 0.865] | 0.890 [0.874, 0.905] |
| CoCoPyE 0.5.0                                        | medium | 83               | 213         | 0.258 [0.201, 0.320] | 0.663 [0.562, 0.762] | 0.372 [0.300, 0.444] |
| DeepCheck                                            | high   | 34               | 39          | 0.282 [0.143, 0.429] | 0.324 [0.162, 0.500] | 0.301 [0.159, 0.435] |
| DeepCheck                                            | low    | 883              | 195         | 0.949 [0.915, 0.978] | 0.210 [0.183, 0.236] | 0.343 [0.306, 0.378] |
| DeepCheck                                            | medium | 83               | 766         | 0.086 [0.067, 0.105] | 0.795 [0.707, 0.877] | 0.155 [0.122, 0.187] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 33   | 1      | 0   |
| medium      | 13   | 66     | 4   |
| low         | 2    | 22     | 859 |

**MAGICC v4 (numbers in submitted manuscript)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 31   | 3      | 0   |
| medium      | 8    | 70     | 5   |
| low         | 2    | 18     | 863 |

**MAGICC v3 (SUPERSEDED - uses 20 assembly statistics)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 29   | 4      | 1   |
| medium      | 5    | 75     | 3   |
| low         | 1    | 15     | 867 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 32   | 2      | 0   |
| medium      | 29   | 52     | 2   |
| low         | 101  | 579    | 203 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 0    | 18     | 16  |
| medium      | 0    | 55     | 28  |
| low         | 0    | 140    | 743 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 11   | 23     | 0   |
| medium      | 7    | 66     | 10  |
| low         | 21   | 677    | 185 |

## set_D - Set D (Archaea) - SUPERSEDED (training leakage)  **[SUPERSEDED - training-set leakage]**

*n* = 1000, clusters (distinct dominant reference genomes) = 1000. Design: 1,000 archaeal dominants drawn from train+val+test; completeness 50-100%, cross-phylum contamination 0-100%

**True class balance:** high = 11 (1.1%), medium = 103 (10.3%), low = 886 (88.6%)

### Overall classification performance (estimate [95% CI])

| tool                                                 | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | 0.973 [0.963, 0.983] | 0.877 [0.795, 0.936] | 0.877 [0.795, 0.936]        | 0.973 [0.963, 0.983] | 0.973 [0.964, 0.983] | 0.923 [0.850, 0.968] | 0.870 [0.822, 0.917] |
| MAGICC v4 (numbers in submitted manuscript)          | 0.973 [0.963, 0.983] | 0.824 [0.722, 0.903] | 0.824 [0.722, 0.903]        | 0.973 [0.963, 0.983] | 0.973 [0.963, 0.983] | 0.830 [0.728, 0.936] | 0.867 [0.818, 0.914] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | 0.978 [0.969, 0.987] | 0.889 [0.807, 0.948] | 0.889 [0.807, 0.948]        | 0.978 [0.969, 0.987] | 0.978 [0.969, 0.987] | 0.903 [0.818, 0.972] | 0.894 [0.850, 0.937] |
| CheckM2 1.0.1                                        | 0.499 [0.468, 0.530] | 0.373 [0.327, 0.416] | 0.373 [0.327, 0.416]        | 0.499 [0.468, 0.530] | 0.587 [0.557, 0.615] | 0.716 [0.639, 0.767] | 0.135 [0.105, 0.166] |
| CoCoPyE 0.5.0                                        | 0.935 [0.919, 0.949] | 0.710 [0.617, 0.789] | 0.710 [0.617, 0.789]        | 0.935 [0.919, 0.949] | 0.940 [0.926, 0.953] | 0.774 [0.664, 0.878] | 0.718 [0.652, 0.780] |
| DeepCheck                                            | 0.262 [0.235, 0.291] | 0.205 [0.178, 0.234] | 0.205 [0.178, 0.234]        | 0.262 [0.235, 0.291] | 0.315 [0.282, 0.348] | 0.504 [0.398, 0.611] | 0.036 [0.019, 0.054] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                                                 | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | high   | 11               | 15          | 0.667 [0.400, 0.909] | 0.909 [0.700, 1.000] | 0.769 [0.538, 0.929] |
| MAGICC v5 (released, v0.3.0)                         | low    | 886              | 882         | 0.990 [0.983, 0.996] | 0.985 [0.977, 0.993] | 0.988 [0.982, 0.993] |
| MAGICC v5 (released, v0.3.0)                         | medium | 103              | 103         | 0.874 [0.808, 0.938] | 0.874 [0.807, 0.933] | 0.874 [0.824, 0.920] |
| MAGICC v4 (numbers in submitted manuscript)          | high   | 11               | 12          | 0.583 [0.286, 0.857] | 0.636 [0.333, 1.000] | 0.609 [0.333, 0.818] |
| MAGICC v4 (numbers in submitted manuscript)          | low    | 886              | 887         | 0.989 [0.981, 0.995] | 0.990 [0.983, 0.996] | 0.989 [0.984, 0.994] |
| MAGICC v4 (numbers in submitted manuscript)          | medium | 103              | 101         | 0.881 [0.813, 0.942] | 0.864 [0.792, 0.927] | 0.873 [0.820, 0.919] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | high   | 11               | 12          | 0.750 [0.462, 1.000] | 0.818 [0.571, 1.000] | 0.783 [0.545, 0.933] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | low    | 886              | 883         | 0.992 [0.985, 0.998] | 0.989 [0.982, 0.995] | 0.990 [0.986, 0.994] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | medium | 103              | 105         | 0.886 [0.824, 0.942] | 0.903 [0.844, 0.957] | 0.894 [0.848, 0.938] |
| CheckM2 1.0.1                                        | high   | 11               | 77          | 0.130 [0.060, 0.206] | 0.909 [0.700, 1.000] | 0.227 [0.111, 0.340] |
| CheckM2 1.0.1                                        | low    | 886              | 415         | 0.986 [0.973, 0.995] | 0.462 [0.428, 0.494] | 0.629 [0.596, 0.659] |
| CheckM2 1.0.1                                        | medium | 103              | 508         | 0.157 [0.127, 0.190] | 0.777 [0.691, 0.854] | 0.262 [0.217, 0.307] |
| CoCoPyE 0.5.0                                        | high   | 11               | 17          | 0.353 [0.133, 0.591] | 0.545 [0.222, 0.857] | 0.429 [0.174, 0.645] |
| CoCoPyE 0.5.0                                        | low    | 886              | 853         | 0.989 [0.982, 0.995] | 0.953 [0.938, 0.966] | 0.971 [0.962, 0.978] |
| CoCoPyE 0.5.0                                        | medium | 103              | 130         | 0.654 [0.566, 0.739] | 0.825 [0.750, 0.894] | 0.730 [0.660, 0.790] |
| DeepCheck                                            | high   | 11               | 131         | 0.046 [0.015, 0.085] | 0.545 [0.222, 0.857] | 0.085 [0.028, 0.152] |
| DeepCheck                                            | low    | 886              | 183         | 0.967 [0.939, 0.990] | 0.200 [0.174, 0.227] | 0.331 [0.294, 0.367] |
| DeepCheck                                            | medium | 103              | 686         | 0.115 [0.093, 0.139] | 0.767 [0.677, 0.849] | 0.200 [0.164, 0.239] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 10   | 0      | 1   |
| medium      | 5    | 90     | 8   |
| low         | 0    | 13     | 873 |

**MAGICC v4 (numbers in submitted manuscript)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 7    | 3      | 1   |
| medium      | 5    | 89     | 9   |
| low         | 0    | 9      | 877 |

**MAGICC v3 (SUPERSEDED - uses 20 assembly statistics)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 9    | 2      | 0   |
| medium      | 3    | 93     | 7   |
| low         | 0    | 10     | 876 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 10   | 1      | 0   |
| medium      | 17   | 80     | 6   |
| low         | 50   | 427    | 409 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 6    | 5      | 0   |
| medium      | 9    | 85     | 9   |
| low         | 2    | 40     | 844 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 6    | 5      | 0   |
| medium      | 18   | 79     | 6   |
| low         | 107  | 602    | 177 |

## set_E - Set E (realistic mixture)

*n* = 1000, clusters (distinct dominant reference genomes) = 785. Design: 1,000 NCBI finished test-split dominants; 200 pure + 200 complete + 600 mixed (70% cross-/30% within-phylum)

**True class balance:** high = 66 (6.6%), medium = 215 (21.5%), low = 719 (71.9%)

### Overall classification performance (estimate [95% CI])

| tool                                                 | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | 0.954 [0.941, 0.966] | 0.903 [0.871, 0.933] | 0.903 [0.871, 0.933]        | 0.954 [0.941, 0.966] | 0.954 [0.942, 0.967] | 0.904 [0.870, 0.937] | 0.895 [0.866, 0.923] |
| MAGICC v4 (numbers in submitted manuscript)          | 0.956 [0.943, 0.968] | 0.912 [0.884, 0.939] | 0.912 [0.884, 0.939]        | 0.956 [0.943, 0.968] | 0.956 [0.943, 0.968] | 0.899 [0.863, 0.932] | 0.898 [0.869, 0.926] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | 0.964 [0.952, 0.975] | 0.924 [0.894, 0.950] | 0.924 [0.894, 0.950]        | 0.964 [0.952, 0.975] | 0.964 [0.951, 0.975] | 0.908 [0.873, 0.940] | 0.917 [0.889, 0.942] |
| CheckM2 1.0.1                                        | 0.865 [0.843, 0.887] | 0.814 [0.779, 0.847] | 0.814 [0.779, 0.847]        | 0.865 [0.843, 0.887] | 0.873 [0.853, 0.892] | 0.878 [0.844, 0.909] | 0.723 [0.679, 0.765] |
| CoCoPyE 0.5.0                                        | 0.945 [0.930, 0.958] | 0.888 [0.853, 0.917] | 0.888 [0.853, 0.917]        | 0.945 [0.930, 0.958] | 0.945 [0.930, 0.958] | 0.883 [0.845, 0.917] | 0.873 [0.841, 0.903] |
| DeepCheck                                            | 0.737 [0.710, 0.765] | 0.695 [0.657, 0.732] | 0.695 [0.657, 0.732]        | 0.737 [0.710, 0.765] | 0.758 [0.733, 0.782] | 0.809 [0.774, 0.841] | 0.523 [0.477, 0.570] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                                                 | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | high   | 66               | 63          | 0.841 [0.746, 0.927] | 0.803 [0.705, 0.897] | 0.822 [0.746, 0.889] |
| MAGICC v5 (released, v0.3.0)                         | low    | 719              | 709         | 0.987 [0.978, 0.995] | 0.974 [0.962, 0.984] | 0.980 [0.973, 0.987] |
| MAGICC v5 (released, v0.3.0)                         | medium | 215              | 228         | 0.882 [0.836, 0.922] | 0.935 [0.901, 0.966] | 0.907 [0.876, 0.934] |
| MAGICC v4 (numbers in submitted manuscript)          | high   | 66               | 57          | 0.912 [0.836, 0.981] | 0.788 [0.681, 0.883] | 0.846 [0.774, 0.908] |
| MAGICC v4 (numbers in submitted manuscript)          | low    | 719              | 720         | 0.978 [0.967, 0.989] | 0.979 [0.968, 0.989] | 0.978 [0.971, 0.986] |
| MAGICC v4 (numbers in submitted manuscript)          | medium | 215              | 223         | 0.897 [0.856, 0.934] | 0.930 [0.895, 0.963] | 0.913 [0.886, 0.939] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | high   | 66               | 52          | 0.981 [0.935, 1.000] | 0.773 [0.672, 0.869] | 0.864 [0.793, 0.923] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | low    | 719              | 710         | 0.992 [0.984, 0.997] | 0.979 [0.967, 0.989] | 0.985 [0.979, 0.991] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | medium | 215              | 238         | 0.878 [0.831, 0.920] | 0.972 [0.948, 0.991] | 0.923 [0.895, 0.948] |
| CheckM2 1.0.1                                        | high   | 66               | 84          | 0.679 [0.575, 0.780] | 0.864 [0.770, 0.943] | 0.760 [0.676, 0.833] |
| CheckM2 1.0.1                                        | low    | 719              | 615         | 0.992 [0.984, 0.998] | 0.848 [0.820, 0.875] | 0.915 [0.897, 0.930] |
| CheckM2 1.0.1                                        | medium | 215              | 301         | 0.658 [0.604, 0.713] | 0.921 [0.884, 0.955] | 0.767 [0.725, 0.808] |
| CoCoPyE 0.5.0                                        | high   | 66               | 61          | 0.836 [0.742, 0.923] | 0.773 [0.667, 0.868] | 0.803 [0.720, 0.871] |
| CoCoPyE 0.5.0                                        | low    | 719              | 715         | 0.979 [0.968, 0.989] | 0.974 [0.961, 0.985] | 0.976 [0.968, 0.984] |
| CoCoPyE 0.5.0                                        | medium | 215              | 224         | 0.866 [0.818, 0.907] | 0.902 [0.862, 0.940] | 0.884 [0.850, 0.913] |
| DeepCheck                                            | high   | 66               | 107         | 0.533 [0.437, 0.630] | 0.864 [0.780, 0.940] | 0.659 [0.573, 0.739] |
| DeepCheck                                            | low    | 719              | 495         | 0.992 [0.983, 0.998] | 0.683 [0.649, 0.716] | 0.809 [0.784, 0.832] |
| DeepCheck                                            | medium | 215              | 398         | 0.475 [0.425, 0.525] | 0.879 [0.835, 0.921] | 0.617 [0.570, 0.663] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 53   | 10     | 3   |
| medium      | 8    | 201    | 6   |
| low         | 2    | 17     | 700 |

**MAGICC v4 (numbers in submitted manuscript)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 52   | 9      | 5   |
| medium      | 4    | 200    | 11  |
| low         | 1    | 14     | 704 |

**MAGICC v3 (SUPERSEDED - uses 20 assembly statistics)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 51   | 15     | 0   |
| medium      | 0    | 209    | 6   |
| low         | 1    | 14     | 704 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 57   | 8      | 1   |
| medium      | 13   | 198    | 4   |
| low         | 14   | 95     | 610 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 51   | 11     | 4   |
| medium      | 10   | 194    | 11  |
| low         | 0    | 19     | 700 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 57   | 9      | 0   |
| medium      | 22   | 189    | 4   |
| low         | 28   | 200    | 491 |

## set_C_clean - Set C-clean (Patescibacteriota, held-out test split)

*n* = 1000, clusters (distinct dominant reference genomes) = 100. Design: 100 test-split Patescibacteriota references x 10 simulations; completeness 50-100%, cross-phylum contamination 0-100% from test split only

**True class balance:** high = 30 (3.0%), medium = 70 (7.0%), low = 900 (90.0%)

### Overall classification performance (estimate [95% CI])

| tool                         | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | 0.930 [0.913, 0.946] | 0.692 [0.621, 0.756] | 0.692 [0.621, 0.756]        | 0.930 [0.913, 0.946] | 0.927 [0.909, 0.944] | 0.664 [0.595, 0.738] | 0.594 [0.514, 0.670] |
| CheckM2 1.0.1                | 0.271 [0.240, 0.304] | 0.263 [0.226, 0.303] | 0.263 [0.226, 0.303]        | 0.271 [0.240, 0.304] | 0.346 [0.306, 0.383] | 0.594 [0.553, 0.636] | 0.048 [0.027, 0.072] |
| CoCoPyE 0.5.0                | 0.802 [0.775, 0.828] | 0.415 [0.390, 0.440] | 0.415 [0.390, 0.440]        | 0.802 [0.775, 0.828] | 0.832 [0.809, 0.853] | 0.525 [0.490, 0.560] | 0.315 [0.256, 0.376] |
| DeepCheck                    | 0.250 [0.219, 0.280] | 0.243 [0.195, 0.289] | 0.243 [0.195, 0.289]        | 0.250 [0.219, 0.280] | 0.328 [0.288, 0.366] | 0.413 [0.350, 0.475] | 0.021 [0.003, 0.039] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                         | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | high   | 30               | 23          | 0.652 [0.458, 0.842] | 0.500 [0.310, 0.700] | 0.566 [0.383, 0.717] |
| MAGICC v5 (released, v0.3.0) | low    | 900              | 914         | 0.962 [0.947, 0.975] | 0.977 [0.965, 0.987] | 0.969 [0.960, 0.977] |
| MAGICC v5 (released, v0.3.0) | medium | 70               | 63          | 0.571 [0.443, 0.703] | 0.514 [0.403, 0.632] | 0.541 [0.434, 0.642] |
| CheckM2 1.0.1                | high   | 30               | 161         | 0.186 [0.127, 0.258] | 1.000 [1.000, 1.000] | 0.314 [0.226, 0.410] |
| CheckM2 1.0.1                | low    | 900              | 206         | 0.981 [0.960, 0.995] | 0.224 [0.193, 0.256] | 0.365 [0.322, 0.406] |
| CheckM2 1.0.1                | medium | 70               | 633         | 0.062 [0.043, 0.081] | 0.557 [0.436, 0.676] | 0.111 [0.079, 0.144] |
| CoCoPyE 0.5.0                | high   | 30               | 0           | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CoCoPyE 0.5.0                | low    | 900              | 772         | 0.972 [0.960, 0.982] | 0.833 [0.809, 0.857] | 0.897 [0.881, 0.912] |
| CoCoPyE 0.5.0                | medium | 70               | 228         | 0.228 [0.179, 0.282] | 0.743 [0.640, 0.839] | 0.349 [0.284, 0.414] |
| DeepCheck                    | high   | 30               | 39          | 0.231 [0.105, 0.400] | 0.300 [0.143, 0.472] | 0.261 [0.123, 0.395] |
| DeepCheck                    | low    | 900              | 199         | 0.955 [0.926, 0.979] | 0.211 [0.180, 0.243] | 0.346 [0.303, 0.387] |
| DeepCheck                    | medium | 70               | 762         | 0.067 [0.050, 0.084] | 0.729 [0.603, 0.847] | 0.123 [0.093, 0.153] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 15   | 6      | 9   |
| medium      | 8    | 36     | 26  |
| low         | 0    | 21     | 879 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 30   | 0      | 0   |
| medium      | 27   | 39     | 4   |
| low         | 104  | 594    | 202 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 0    | 26     | 4   |
| medium      | 0    | 52     | 18  |
| low         | 0    | 150    | 750 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 9    | 21     | 0   |
| medium      | 10   | 51     | 9   |
| low         | 20   | 690    | 190 |

## set_D_clean - Set D-clean (Archaea, held-out test split)

*n* = 1000, clusters (distinct dominant reference genomes) = 100. Design: 100 test-split archaeal references x 10 simulations; cross-phylum contamination 0-100% from test split only

**True class balance:** high = 13 (1.3%), medium = 100 (10.0%), low = 887 (88.7%)

### Overall classification performance (estimate [95% CI])

| tool                         | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | 0.958 [0.946, 0.970] | 0.788 [0.692, 0.864] | 0.788 [0.692, 0.864]        | 0.958 [0.946, 0.970] | 0.959 [0.947, 0.970] | 0.811 [0.705, 0.904] | 0.800 [0.742, 0.857] |
| CheckM2 1.0.1                | 0.480 [0.440, 0.523] | 0.382 [0.330, 0.432] | 0.382 [0.330, 0.432]        | 0.480 [0.440, 0.523] | 0.567 [0.528, 0.608] | 0.726 [0.662, 0.774] | 0.137 [0.106, 0.171] |
| CoCoPyE 0.5.0                | 0.930 [0.909, 0.949] | 0.756 [0.674, 0.825] | 0.756 [0.674, 0.825]        | 0.930 [0.909, 0.949] | 0.936 [0.919, 0.953] | 0.867 [0.787, 0.928] | 0.703 [0.629, 0.775] |
| DeepCheck                    | 0.240 [0.207, 0.273] | 0.221 [0.184, 0.255] | 0.221 [0.184, 0.255]        | 0.240 [0.207, 0.273] | 0.273 [0.229, 0.317] | 0.636 [0.572, 0.682] | 0.044 [0.029, 0.059] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                         | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | high   | 13               | 15          | 0.533 [0.267, 0.786] | 0.615 [0.333, 0.875] | 0.571 [0.300, 0.774] |
| MAGICC v5 (released, v0.3.0) | low    | 887              | 878         | 0.986 [0.977, 0.994] | 0.976 [0.967, 0.985] | 0.981 [0.975, 0.987] |
| MAGICC v5 (released, v0.3.0) | medium | 100              | 107         | 0.785 [0.705, 0.859] | 0.840 [0.766, 0.906] | 0.812 [0.750, 0.868] |
| CheckM2 1.0.1                | high   | 13               | 71          | 0.169 [0.091, 0.253] | 0.923 [0.750, 1.000] | 0.286 [0.164, 0.400] |
| CheckM2 1.0.1                | low    | 887              | 386         | 1.000 [1.000, 1.000] | 0.435 [0.392, 0.480] | 0.606 [0.563, 0.649] |
| CheckM2 1.0.1                | medium | 100              | 543         | 0.151 [0.121, 0.182] | 0.820 [0.743, 0.894] | 0.255 [0.210, 0.300] |
| CoCoPyE 0.5.0                | high   | 13               | 24          | 0.458 [0.250, 0.684] | 0.846 [0.611, 1.000] | 0.595 [0.378, 0.766] |
| CoCoPyE 0.5.0                | low    | 887              | 847         | 0.989 [0.982, 0.995] | 0.945 [0.921, 0.964] | 0.967 [0.954, 0.977] |
| CoCoPyE 0.5.0                | medium | 100              | 129         | 0.628 [0.529, 0.734] | 0.810 [0.727, 0.887] | 0.707 [0.631, 0.780] |
| DeepCheck                    | high   | 13               | 118         | 0.102 [0.054, 0.152] | 0.923 [0.750, 1.000] | 0.183 [0.101, 0.262] |
| DeepCheck                    | low    | 887              | 147         | 0.993 [0.978, 1.000] | 0.165 [0.132, 0.199] | 0.282 [0.234, 0.332] |
| DeepCheck                    | medium | 100              | 735         | 0.112 [0.090, 0.133] | 0.820 [0.745, 0.894] | 0.196 [0.162, 0.229] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 8    | 3      | 2   |
| medium      | 6    | 84     | 10  |
| low         | 1    | 20     | 866 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 12   | 1      | 0   |
| medium      | 18   | 82     | 0   |
| low         | 41   | 460    | 386 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 11   | 1      | 1   |
| medium      | 11   | 81     | 8   |
| low         | 2    | 47     | 838 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 12   | 1      | 0   |
| medium      | 17   | 82     | 1   |
| low         | 89   | 652    | 146 |

## set_A - Set A v1 (superseded by Set A v2)

*n* = 600, clusters (distinct dominant reference genomes) = 582. Design: 600 genomes, completeness gradient; replaced by set_A_v2

**True class balance:** high = 182 (30.3%), medium = 418 (69.7%), low = 0 (0.0%)

> **Class(es) absent from the ground truth by design: low.** The 3-class macro F1 therefore averages a structurally undefined class scored as 0 and is bounded above by 2/3; the `macro F1 (observed classes)` column is the interpretable summary for this set. The submitted manuscript's Table S3 used the 3-class form.

### Overall classification performance (estimate [95% CI])

| tool                                                 | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | 0.970 [0.956, 0.983] | 0.646 [0.636, 0.655] | 0.969 [0.953, 0.983]        | 0.970 [0.956, 0.983] | 0.973 [0.960, 0.985] | 0.968 [0.951, 0.983] | 0.930 [0.897, 0.960] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | 0.972 [0.958, 0.985] | 0.645 [0.634, 0.654] | 0.967 [0.950, 0.982]        | 0.972 [0.958, 0.985] | 0.973 [0.959, 0.985] | 0.958 [0.936, 0.976] | 0.932 [0.898, 0.962] |
| CheckM2 1.0.1                                        | 0.930 [0.910, 0.950] | 0.632 [0.621, 0.643] | 0.948 [0.931, 0.965]        | 0.930 [0.910, 0.950] | 0.948 [0.932, 0.964] | 0.942 [0.925, 0.959] | 0.846 [0.802, 0.888] |
| CoCoPyE 0.5.0                                        | 0.950 [0.931, 0.966] | 0.632 [0.619, 0.644] | 0.948 [0.928, 0.965]        | 0.950 [0.931, 0.966] | 0.956 [0.940, 0.971] | 0.939 [0.916, 0.961] | 0.883 [0.841, 0.920] |
| DeepCheck                                            | 0.958 [0.940, 0.974] | 0.642 [0.631, 0.652] | 0.963 [0.947, 0.978]        | 0.958 [0.940, 0.974] | 0.966 [0.951, 0.979] | 0.967 [0.953, 0.980] | 0.906 [0.866, 0.940] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                                                 | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | high   | 182              | 183         | 0.956 [0.923, 0.984] | 0.962 [0.931, 0.988] | 0.959 [0.937, 0.979] |
| MAGICC v5 (released, v0.3.0)                         | low    | 0                | 4           | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MAGICC v5 (released, v0.3.0)                         | medium | 418              | 413         | 0.985 [0.972, 0.995] | 0.974 [0.957, 0.988] | 0.980 [0.969, 0.989] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | high   | 182              | 171         | 0.982 [0.960, 1.000] | 0.923 [0.882, 0.959] | 0.952 [0.927, 0.973] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | low    | 0                | 2           | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | medium | 418              | 427         | 0.972 [0.955, 0.986] | 0.993 [0.983, 1.000] | 0.982 [0.972, 0.991] |
| CheckM2 1.0.1                                        | high   | 182              | 191         | 0.927 [0.888, 0.961] | 0.973 [0.947, 0.995] | 0.949 [0.926, 0.970] |
| CheckM2 1.0.1                                        | low    | 0                | 23          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CheckM2 1.0.1                                        | medium | 418              | 386         | 0.987 [0.975, 0.997] | 0.911 [0.883, 0.938] | 0.948 [0.932, 0.963] |
| CoCoPyE 0.5.0                                        | high   | 182              | 176         | 0.943 [0.904, 0.976] | 0.912 [0.869, 0.952] | 0.927 [0.898, 0.953] |
| CoCoPyE 0.5.0                                        | low    | 0                | 8           | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CoCoPyE 0.5.0                                        | medium | 418              | 416         | 0.971 [0.954, 0.986] | 0.967 [0.948, 0.982] | 0.969 [0.956, 0.980] |
| DeepCheck                                            | high   | 182              | 194         | 0.928 [0.889, 0.963] | 0.989 [0.972, 1.000] | 0.957 [0.934, 0.976] |
| DeepCheck                                            | low    | 0                | 9           | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| DeepCheck                                            | medium | 418              | 397         | 0.995 [0.987, 1.000] | 0.945 [0.921, 0.967] | 0.969 [0.956, 0.981] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 175  | 6      | 1   |
| medium      | 8    | 407    | 3   |
| low         | 0    | 0      | 0   |

**MAGICC v3 (SUPERSEDED - uses 20 assembly statistics)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 168  | 12     | 2   |
| medium      | 3    | 415    | 0   |
| low         | 0    | 0      | 0   |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 177  | 5      | 0   |
| medium      | 14   | 381    | 23  |
| low         | 0    | 0      | 0   |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 166  | 12     | 4   |
| medium      | 10   | 404    | 4   |
| low         | 0    | 0      | 0   |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 180  | 2      | 0   |
| medium      | 14   | 395    | 9   |
| low         | 0    | 0      | 0   |

## set_B - Set B v1 (superseded by Set B v2)

*n* = 600, clusters (distinct dominant reference genomes) = 588. Design: 600 genomes, contamination gradient; replaced by set_B_v2

**True class balance:** high = 103 (17.2%), medium = 95 (15.8%), low = 402 (67.0%)

### Overall classification performance (estimate [95% CI])

| tool                                                 | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | 0.920 [0.898, 0.942] | 0.869 [0.834, 0.901] | 0.869 [0.834, 0.901]        | 0.920 [0.898, 0.942] | 0.915 [0.890, 0.939] | 0.854 [0.819, 0.889] | 0.834 [0.787, 0.876] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | 0.898 [0.874, 0.923] | 0.836 [0.796, 0.872] | 0.836 [0.796, 0.872]        | 0.898 [0.874, 0.923] | 0.891 [0.864, 0.919] | 0.813 [0.774, 0.852] | 0.784 [0.734, 0.835] |
| CheckM2 1.0.1                                        | 0.797 [0.765, 0.829] | 0.693 [0.649, 0.736] | 0.693 [0.649, 0.736]        | 0.797 [0.765, 0.829] | 0.795 [0.764, 0.827] | 0.741 [0.706, 0.777] | 0.618 [0.564, 0.674] |
| CoCoPyE 0.5.0                                        | 0.882 [0.854, 0.906] | 0.792 [0.746, 0.832] | 0.792 [0.746, 0.832]        | 0.882 [0.854, 0.906] | 0.867 [0.833, 0.896] | 0.777 [0.737, 0.813] | 0.746 [0.688, 0.796] |
| DeepCheck                                            | 0.612 [0.574, 0.652] | 0.529 [0.493, 0.567] | 0.529 [0.493, 0.567]        | 0.612 [0.574, 0.652] | 0.642 [0.607, 0.679] | 0.617 [0.584, 0.651] | 0.385 [0.337, 0.438] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                                                 | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0)                         | high   | 103              | 110         | 0.891 [0.830, 0.945] | 0.951 [0.905, 0.990] | 0.920 [0.880, 0.956] |
| MAGICC v5 (released, v0.3.0)                         | low    | 402              | 420         | 0.938 [0.913, 0.960] | 0.980 [0.965, 0.993] | 0.959 [0.944, 0.972] |
| MAGICC v5 (released, v0.3.0)                         | medium | 95               | 70          | 0.857 [0.770, 0.934] | 0.632 [0.535, 0.722] | 0.727 [0.647, 0.798] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | high   | 103              | 100         | 0.940 [0.891, 0.981] | 0.913 [0.859, 0.962] | 0.926 [0.888, 0.962] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | low    | 402              | 432         | 0.910 [0.880, 0.935] | 0.978 [0.962, 0.991] | 0.942 [0.925, 0.958] |
| MAGICC v3 (SUPERSEDED - uses 20 assembly statistics) | medium | 95               | 68          | 0.765 [0.662, 0.864] | 0.547 [0.447, 0.649] | 0.638 [0.549, 0.720] |
| CheckM2 1.0.1                                        | high   | 103              | 166         | 0.620 [0.551, 0.695] | 1.000 [1.000, 1.000] | 0.766 [0.710, 0.820] |
| CheckM2 1.0.1                                        | low    | 402              | 359         | 0.944 [0.920, 0.967] | 0.843 [0.807, 0.878] | 0.891 [0.866, 0.913] |
| CheckM2 1.0.1                                        | medium | 95               | 75          | 0.480 [0.366, 0.594] | 0.379 [0.279, 0.480] | 0.424 [0.323, 0.516] |
| CoCoPyE 0.5.0                                        | high   | 103              | 111         | 0.874 [0.810, 0.933] | 0.942 [0.891, 0.982] | 0.907 [0.861, 0.944] |
| CoCoPyE 0.5.0                                        | low    | 402              | 438         | 0.897 [0.865, 0.925] | 0.978 [0.962, 0.991] | 0.936 [0.917, 0.952] |
| CoCoPyE 0.5.0                                        | medium | 95               | 51          | 0.765 [0.636, 0.878] | 0.411 [0.304, 0.516] | 0.534 [0.420, 0.627] |
| DeepCheck                                            | high   | 103              | 224         | 0.455 [0.390, 0.521] | 0.990 [0.968, 1.000] | 0.624 [0.559, 0.683] |
| DeepCheck                                            | low    | 402              | 241         | 0.996 [0.987, 1.000] | 0.597 [0.550, 0.645] | 0.747 [0.709, 0.783] |
| DeepCheck                                            | medium | 95               | 135         | 0.185 [0.122, 0.252] | 0.263 [0.176, 0.351] | 0.217 [0.147, 0.286] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 98   | 3      | 2   |
| medium      | 11   | 60     | 24  |
| low         | 1    | 7      | 394 |

**MAGICC v3 (SUPERSEDED - uses 20 assembly statistics)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 94   | 8      | 1   |
| medium      | 5    | 52     | 38  |
| low         | 1    | 8      | 393 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 103  | 0      | 0   |
| medium      | 39   | 36     | 20  |
| low         | 24   | 39     | 339 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 97   | 4      | 2   |
| medium      | 13   | 39     | 43  |
| low         | 1    | 8      | 393 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 102  | 1      | 0   |
| medium      | 69   | 25     | 1   |
| low         | 53   | 109    | 240 |

## motivating_v2_set_A - Motivating Set A (Table S1a): completeness gradient

*n* = 1000, clusters (distinct dominant reference genomes) = 806. Design: 1,000 finished test-split dominants, completeness gradient; existing tools only in the manuscript

**True class balance:** high = 288 (28.8%), medium = 712 (71.2%), low = 0 (0.0%)

> **Class(es) absent from the ground truth by design: low.** The 3-class macro F1 therefore averages a structurally undefined class scored as 0 and is bounded above by 2/3; the `macro F1 (observed classes)` column is the interpretable summary for this set. The submitted manuscript's Table S3 used the 3-class form.

### Overall classification performance (estimate [95% CI])

| tool                         | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | 0.960 [0.947, 0.972] | 0.639 [0.629, 0.647] | 0.958 [0.944, 0.970]        | 0.960 [0.947, 0.972] | 0.965 [0.953, 0.975] | 0.954 [0.938, 0.968] | 0.904 [0.872, 0.933] |
| CheckM2 1.0.1                | 0.952 [0.939, 0.965] | 0.640 [0.632, 0.648] | 0.961 [0.948, 0.973]        | 0.952 [0.939, 0.965] | 0.963 [0.952, 0.973] | 0.960 [0.948, 0.971] | 0.889 [0.860, 0.919] |
| CoCoPyE 0.5.0                | 0.951 [0.937, 0.965] | 0.634 [0.624, 0.644] | 0.951 [0.936, 0.965]        | 0.951 [0.937, 0.965] | 0.960 [0.947, 0.971] | 0.947 [0.930, 0.964] | 0.885 [0.852, 0.916] |
| DeepCheck                    | 0.931 [0.915, 0.946] | 0.625 [0.614, 0.635] | 0.937 [0.921, 0.952]        | 0.931 [0.915, 0.946] | 0.942 [0.929, 0.955] | 0.944 [0.930, 0.957] | 0.843 [0.808, 0.877] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                         | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | high   | 288              | 288         | 0.941 [0.912, 0.967] | 0.941 [0.912, 0.966] | 0.941 [0.920, 0.959] |
| MAGICC v5 (released, v0.3.0) | low    | 0                | 10          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MAGICC v5 (released, v0.3.0) | medium | 712              | 702         | 0.981 [0.971, 0.990] | 0.968 [0.953, 0.981] | 0.975 [0.966, 0.983] |
| CheckM2 1.0.1                | high   | 288              | 302         | 0.934 [0.905, 0.962] | 0.979 [0.962, 0.993] | 0.956 [0.938, 0.972] |
| CheckM2 1.0.1                | low    | 0                | 22          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CheckM2 1.0.1                | medium | 712              | 676         | 0.991 [0.984, 0.997] | 0.941 [0.924, 0.957] | 0.965 [0.956, 0.975] |
| CoCoPyE 0.5.0                | high   | 288              | 292         | 0.925 [0.893, 0.954] | 0.938 [0.907, 0.967] | 0.931 [0.908, 0.953] |
| CoCoPyE 0.5.0                | low    | 0                | 18          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CoCoPyE 0.5.0                | medium | 712              | 690         | 0.987 [0.978, 0.994] | 0.956 [0.942, 0.970] | 0.971 [0.962, 0.980] |
| DeepCheck                    | high   | 288              | 320         | 0.878 [0.845, 0.914] | 0.976 [0.956, 0.993] | 0.924 [0.902, 0.945] |
| DeepCheck                    | low    | 0                | 23          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| DeepCheck                    | medium | 712              | 657         | 0.989 [0.981, 0.997] | 0.913 [0.892, 0.932] | 0.950 [0.937, 0.961] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 271  | 13     | 4   |
| medium      | 17   | 689    | 6   |
| low         | 0    | 0      | 0   |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 282  | 6      | 0   |
| medium      | 20   | 670    | 22  |
| low         | 0    | 0      | 0   |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 270  | 9      | 9   |
| medium      | 22   | 681    | 9   |
| low         | 0    | 0      | 0   |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 281  | 7      | 0   |
| medium      | 39   | 650    | 23  |
| low         | 0    | 0      | 0   |

## motivating_v2_set_B - Motivating Set B (Table S1b): contamination gradient

*n* = 1000, clusters (distinct dominant reference genomes) = 798. Design: 1,000 finished test-split dominants at 100% completeness, cross-phylum contamination 0-80%

**True class balance:** high = 200 (20.0%), medium = 2 (0.2%), low = 798 (79.8%)

> **Small-support class(es): medium (n=2).** Per-class F1 for these classes is estimated from fewer than 10 genomes; the CI is correspondingly wide and the macro average is dominated by sampling noise in that class. Individual data points are shown for these classes in the figures.

### Overall classification performance (estimate [95% CI])

| tool                         | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | 0.984 [0.976, 0.991] | 0.743 [0.652, 0.841] | 0.743 [0.652, 0.841]        | 0.984 [0.976, 0.991] | 0.988 [0.982, 0.993] | 0.978 [0.960, 0.988] | 0.951 [0.926, 0.972] |
| CheckM2 1.0.1                | 0.876 [0.855, 0.897] | 0.607 [0.593, 0.618] | 0.607 [0.593, 0.618]        | 0.876 [0.855, 0.897] | 0.912 [0.896, 0.928] | 0.616 [0.608, 0.929] | 0.698 [0.653, 0.743] |
| CoCoPyE 0.5.0                | 0.981 [0.972, 0.989] | 0.695 [0.654, 0.770] | 0.695 [0.654, 0.770]        | 0.981 [0.972, 0.989] | 0.987 [0.981, 0.993] | 0.821 [0.649, 0.992] | 0.943 [0.916, 0.966] |
| DeepCheck                    | 0.689 [0.662, 0.718] | 0.516 [0.499, 0.532] | 0.516 [0.499, 0.532]        | 0.689 [0.662, 0.718] | 0.764 [0.740, 0.788] | 0.538 [0.527, 0.814] | 0.432 [0.393, 0.473] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                         | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | high   | 200              | 188         | 1.000 [1.000, 1.000] | 0.940 [0.905, 0.970] | 0.969 [0.950, 0.985] |
| MAGICC v5 (released, v0.3.0) | low    | 798              | 799         | 0.994 [0.988, 0.999] | 0.995 [0.990, 0.999] | 0.994 [0.991, 0.998] |
| MAGICC v5 (released, v0.3.0) | medium | 2                | 13          | 0.154 [0.000, 0.375] | 1.000 [0.000, 1.000] | 0.267 [0.000, 0.545] |
| CheckM2 1.0.1                | high   | 200              | 243         | 0.823 [0.775, 0.868] | 1.000 [1.000, 1.000] | 0.903 [0.873, 0.929] |
| CheckM2 1.0.1                | low    | 798              | 677         | 0.999 [0.995, 1.000] | 0.847 [0.822, 0.873] | 0.917 [0.902, 0.931] |
| CheckM2 1.0.1                | medium | 2                | 80          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CoCoPyE 0.5.0                | high   | 200              | 198         | 0.990 [0.974, 1.000] | 0.980 [0.960, 0.995] | 0.985 [0.972, 0.995] |
| CoCoPyE 0.5.0                | low    | 798              | 786         | 0.997 [0.994, 1.000] | 0.982 [0.972, 0.991] | 0.990 [0.984, 0.994] |
| CoCoPyE 0.5.0                | medium | 2                | 16          | 0.062 [0.000, 0.214] | 0.500 [0.000, 1.000] | 0.111 [0.000, 0.333] |
| DeepCheck                    | high   | 200              | 308         | 0.649 [0.600, 0.700] | 1.000 [1.000, 1.000] | 0.787 [0.750, 0.824] |
| DeepCheck                    | low    | 798              | 489         | 1.000 [1.000, 1.000] | 0.613 [0.581, 0.648] | 0.760 [0.735, 0.786] |
| DeepCheck                    | medium | 2                | 203         | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 188  | 7      | 5   |
| medium      | 0    | 2      | 0   |
| low         | 0    | 4      | 794 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 200  | 0      | 0   |
| medium      | 1    | 0      | 1   |
| low         | 42   | 80     | 676 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 196  | 2      | 2   |
| medium      | 1    | 1      | 0   |
| low         | 1    | 13     | 784 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 200  | 0      | 0   |
| medium      | 2    | 0      | 0   |
| low         | 106  | 203    | 489 |

## motivating_v2_set_C - Motivating Set C (Table S1c): realistic mixture

*n* = 1000, clusters (distinct dominant reference genomes) = 773. Design: 1,000 finished test-split dominants, realistic mixture

**True class balance:** high = 44 (4.4%), medium = 240 (24.0%), low = 716 (71.6%)

### Overall classification performance (estimate [95% CI])

| tool                         | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | 0.968 [0.957, 0.979] | 0.908 [0.871, 0.941] | 0.908 [0.871, 0.941]        | 0.968 [0.957, 0.979] | 0.968 [0.957, 0.979] | 0.923 [0.885, 0.958] | 0.925 [0.899, 0.950] |
| CheckM2 1.0.1                | 0.854 [0.831, 0.877] | 0.795 [0.753, 0.831] | 0.795 [0.753, 0.831]        | 0.854 [0.831, 0.877] | 0.863 [0.843, 0.884] | 0.895 [0.864, 0.919] | 0.701 [0.657, 0.746] |
| CoCoPyE 0.5.0                | 0.952 [0.937, 0.965] | 0.887 [0.847, 0.922] | 0.887 [0.847, 0.922]        | 0.952 [0.937, 0.965] | 0.952 [0.938, 0.966] | 0.901 [0.855, 0.941] | 0.887 [0.855, 0.917] |
| DeepCheck                    | 0.714 [0.686, 0.743] | 0.669 [0.628, 0.709] | 0.669 [0.628, 0.709]        | 0.714 [0.686, 0.743] | 0.734 [0.709, 0.761] | 0.834 [0.810, 0.856] | 0.489 [0.445, 0.535] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                         | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | high   | 44               | 51          | 0.745 [0.617, 0.860] | 0.864 [0.756, 0.957] | 0.800 [0.696, 0.882] |
| MAGICC v5 (released, v0.3.0) | low    | 716              | 720         | 0.988 [0.979, 0.995] | 0.993 [0.986, 0.999] | 0.990 [0.985, 0.995] |
| MAGICC v5 (released, v0.3.0) | medium | 240              | 229         | 0.956 [0.927, 0.982] | 0.912 [0.875, 0.947] | 0.934 [0.909, 0.956] |
| CheckM2 1.0.1                | high   | 44               | 73          | 0.575 [0.455, 0.683] | 0.955 [0.878, 1.000] | 0.718 [0.613, 0.803] |
| CheckM2 1.0.1                | low    | 716              | 601         | 0.993 [0.986, 0.998] | 0.834 [0.806, 0.863] | 0.907 [0.890, 0.924] |
| CheckM2 1.0.1                | medium | 240              | 326         | 0.660 [0.608, 0.714] | 0.896 [0.852, 0.934] | 0.760 [0.718, 0.799] |
| CoCoPyE 0.5.0                | high   | 44               | 51          | 0.725 [0.600, 0.840] | 0.841 [0.707, 0.955] | 0.779 [0.675, 0.864] |
| CoCoPyE 0.5.0                | low    | 716              | 722         | 0.975 [0.962, 0.986] | 0.983 [0.974, 0.992] | 0.979 [0.971, 0.986] |
| CoCoPyE 0.5.0                | medium | 240              | 227         | 0.930 [0.897, 0.959] | 0.879 [0.833, 0.921] | 0.904 [0.874, 0.931] |
| DeepCheck                    | high   | 44               | 98          | 0.439 [0.337, 0.540] | 0.977 [0.927, 1.000] | 0.606 [0.500, 0.696] |
| DeepCheck                    | low    | 716              | 462         | 0.994 [0.985, 1.000] | 0.641 [0.605, 0.677] | 0.779 [0.753, 0.806] |
| DeepCheck                    | medium | 240              | 440         | 0.482 [0.435, 0.531] | 0.883 [0.842, 0.922] | 0.624 [0.579, 0.668] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 38   | 6      | 0   |
| medium      | 12   | 219    | 9   |
| low         | 1    | 4      | 711 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 42   | 2      | 0   |
| medium      | 21   | 215    | 4   |
| low         | 10   | 109    | 597 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 37   | 4      | 3   |
| medium      | 14   | 211    | 15  |
| low         | 0    | 12     | 704 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 43   | 1      | 0   |
| medium      | 25   | 212    | 3   |
| low         | 30   | 227    | 459 |

## motivating_set_A - Motivating Set A v1 (superseded)

*n* = 600, clusters (distinct dominant reference genomes) = 585. Design: 600 genomes; replaced by motivating_v2/set_A

**True class balance:** high = 184 (30.7%), medium = 416 (69.3%), low = 0 (0.0%)

> **Class(es) absent from the ground truth by design: low.** The 3-class macro F1 therefore averages a structurally undefined class scored as 0 and is bounded above by 2/3; the `macro F1 (observed classes)` column is the interpretable summary for this set. The submitted manuscript's Table S3 used the 3-class form.

### Overall classification performance (estimate [95% CI])

| tool                         | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | 0.947 [0.926, 0.963] | 0.628 [0.612, 0.641] | 0.941 [0.919, 0.961]        | 0.947 [0.926, 0.963] | 0.952 [0.933, 0.968] | 0.924 [0.895, 0.948] | 0.873 [0.827, 0.913] |
| CheckM2 1.0.1                | 0.958 [0.942, 0.974] | 0.646 [0.636, 0.655] | 0.969 [0.954, 0.982]        | 0.958 [0.942, 0.974] | 0.969 [0.955, 0.981] | 0.965 [0.951, 0.979] | 0.906 [0.869, 0.941] |
| CoCoPyE 0.5.0                | 0.950 [0.930, 0.968] | 0.631 [0.616, 0.644] | 0.947 [0.925, 0.965]        | 0.950 [0.930, 0.968] | 0.958 [0.941, 0.974] | 0.928 [0.899, 0.952] | 0.883 [0.837, 0.922] |
| DeepCheck                    | 0.940 [0.919, 0.958] | 0.632 [0.618, 0.643] | 0.948 [0.927, 0.965]        | 0.940 [0.919, 0.958] | 0.951 [0.933, 0.966] | 0.939 [0.914, 0.959] | 0.864 [0.817, 0.904] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                         | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | high   | 184              | 164         | 0.970 [0.940, 0.994] | 0.864 [0.811, 0.913] | 0.914 [0.879, 0.943] |
| MAGICC v5 (released, v0.3.0) | low    | 0                | 8           | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MAGICC v5 (released, v0.3.0) | medium | 416              | 428         | 0.956 [0.933, 0.976] | 0.983 [0.970, 0.993] | 0.969 [0.956, 0.981] |
| CheckM2 1.0.1                | high   | 184              | 190         | 0.953 [0.920, 0.982] | 0.984 [0.963, 1.000] | 0.968 [0.948, 0.986] |
| CheckM2 1.0.1                | low    | 0                | 13          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CheckM2 1.0.1                | medium | 416              | 397         | 0.992 [0.983, 1.000] | 0.947 [0.923, 0.968] | 0.969 [0.956, 0.981] |
| CoCoPyE 0.5.0                | high   | 184              | 165         | 0.970 [0.941, 0.994] | 0.870 [0.812, 0.919] | 0.917 [0.882, 0.947] |
| CoCoPyE 0.5.0                | low    | 0                | 11          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| CoCoPyE 0.5.0                | medium | 416              | 424         | 0.967 [0.947, 0.984] | 0.986 [0.973, 0.995] | 0.976 [0.965, 0.986] |
| DeepCheck                    | high   | 184              | 182         | 0.945 [0.909, 0.975] | 0.935 [0.891, 0.971] | 0.940 [0.912, 0.964] |
| DeepCheck                    | low    | 0                | 14          | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| DeepCheck                    | medium | 416              | 404         | 0.970 [0.949, 0.987] | 0.942 [0.918, 0.963] | 0.956 [0.940, 0.970] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 159  | 19     | 6   |
| medium      | 5    | 409    | 2   |
| low         | 0    | 0      | 0   |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 181  | 3      | 0   |
| medium      | 9    | 394    | 13  |
| low         | 0    | 0      | 0   |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 160  | 14     | 10  |
| medium      | 5    | 410    | 1   |
| low         | 0    | 0      | 0   |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 172  | 12     | 0   |
| medium      | 10   | 392    | 14  |
| low         | 0    | 0      | 0   |

## motivating_set_B - Motivating Set B v1 (superseded)

*n* = 1100, clusters (distinct dominant reference genomes) = 1031. Design: 1,100 genomes; replaced by motivating_v2/set_B

**True class balance:** high = 197 (17.9%), medium = 96 (8.7%), low = 807 (73.4%)

### Overall classification performance (estimate [95% CI])

| tool                         | accuracy             | macro F1 (3 classes) | macro F1 (observed classes) | micro F1 (=accuracy) | weighted F1          | balanced acc.        | Cohen kappa          |
|------------------------------|----------------------|----------------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | 0.936 [0.922, 0.950] | 0.852 [0.821, 0.883] | 0.852 [0.821, 0.883]        | 0.936 [0.922, 0.950] | 0.937 [0.922, 0.951] | 0.853 [0.819, 0.887] | 0.849 [0.816, 0.882] |
| CheckM2 1.0.1                | 0.758 [0.734, 0.784] | 0.614 [0.582, 0.648] | 0.614 [0.582, 0.648]        | 0.758 [0.734, 0.784] | 0.783 [0.761, 0.806] | 0.675 [0.640, 0.714] | 0.527 [0.484, 0.570] |
| CoCoPyE 0.5.0                | 0.898 [0.880, 0.916] | 0.768 [0.732, 0.802] | 0.768 [0.732, 0.802]        | 0.898 [0.880, 0.916] | 0.900 [0.882, 0.919] | 0.766 [0.726, 0.806] | 0.757 [0.715, 0.798] |
| DeepCheck                    | 0.580 [0.552, 0.609] | 0.486 [0.460, 0.512] | 0.486 [0.460, 0.512]        | 0.580 [0.552, 0.609] | 0.629 [0.603, 0.656] | 0.584 [0.554, 0.616] | 0.335 [0.301, 0.370] |

### Per-class precision / recall / F1 (estimate [95% CI])

| tool                         | class  | support (true n) | n predicted | precision            | recall               | F1                   |
|------------------------------|--------|------------------|-------------|----------------------|----------------------|----------------------|
| MAGICC v5 (released, v0.3.0) | high   | 197              | 193         | 0.938 [0.903, 0.968] | 0.919 [0.879, 0.954] | 0.928 [0.901, 0.953] |
| MAGICC v5 (released, v0.3.0) | low    | 807              | 808         | 0.972 [0.959, 0.983] | 0.973 [0.961, 0.983] | 0.972 [0.964, 0.980] |
| MAGICC v5 (released, v0.3.0) | medium | 96               | 99          | 0.646 [0.546, 0.738] | 0.667 [0.571, 0.763] | 0.656 [0.577, 0.732] |
| CheckM2 1.0.1                | high   | 197              | 295         | 0.593 [0.538, 0.648] | 0.888 [0.842, 0.932] | 0.711 [0.667, 0.755] |
| CheckM2 1.0.1                | low    | 807              | 642         | 0.972 [0.959, 0.984] | 0.773 [0.743, 0.801] | 0.861 [0.842, 0.879] |
| CheckM2 1.0.1                | medium | 96               | 163         | 0.215 [0.152, 0.280] | 0.365 [0.266, 0.467] | 0.270 [0.198, 0.340] |
| CoCoPyE 0.5.0                | high   | 197              | 175         | 0.914 [0.870, 0.953] | 0.812 [0.756, 0.868] | 0.860 [0.820, 0.897] |
| CoCoPyE 0.5.0                | low    | 807              | 814         | 0.956 [0.940, 0.969] | 0.964 [0.951, 0.977] | 0.960 [0.950, 0.970] |
| CoCoPyE 0.5.0                | medium | 96               | 111         | 0.450 [0.356, 0.543] | 0.521 [0.419, 0.619] | 0.483 [0.398, 0.563] |
| DeepCheck                    | high   | 197              | 404         | 0.478 [0.430, 0.528] | 0.980 [0.956, 0.995] | 0.642 [0.597, 0.687] |
| DeepCheck                    | low    | 807              | 421         | 1.000 [1.000, 1.000] | 0.522 [0.487, 0.555] | 0.686 [0.655, 0.714] |
| DeepCheck                    | medium | 96               | 275         | 0.087 [0.056, 0.120] | 0.250 [0.163, 0.338] | 0.129 [0.083, 0.175] |

### Confusion matrices (rows = true class, columns = predicted class)

**MAGICC v5 (released, v0.3.0)**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 181  | 14     | 2   |
| medium      | 11   | 64     | 21  |
| low         | 1    | 21     | 785 |

**CheckM2 1.0.1**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 175  | 22     | 0   |
| medium      | 43   | 35     | 18  |
| low         | 77   | 106    | 624 |

**CoCoPyE 0.5.0**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 160  | 33     | 4   |
| medium      | 14   | 50     | 32  |
| low         | 1    | 28     | 778 |

**DeepCheck**

| true \ pred | high | medium | low |
|-------------|------|--------|-----|
| high        | 193  | 4      | 0   |
| medium      | 72   | 24     | 0   |
| low         | 139  | 247    | 421 |

