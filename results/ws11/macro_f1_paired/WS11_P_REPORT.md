# WS11.P — paired tests for the macro-F1 claim, and the power statement for the negative results

Generated 2026-08-26T17:13:32.772734+00:00 · `scripts/211_ws11p_macro_f1_paired.py` · 2000 cluster-bootstrap resamples · `PYTHONHASHSEED=0`

## 1. What was wrong with the old form of the claim

Macro F1 was reported as two point estimates with **separate** confidence intervals. Overlapping marginal intervals do not mean the difference is indistinguishable from zero, and non-overlapping ones are not the test either. The object a reviewer will ask for is the **paired** difference: resample the same reference-genome clusters, recompute both tools' macro F1 on that resample, and take the difference inside the replicate. That is what this analysis does, with the same seed script 102 used, so these differences come from exactly the replicates behind the published marginal intervals.

Point-estimate check: every recomputed macro F1 reproduces `results/revision/metrics/definitive_mimag.tsv` (max |diff| = 8.33e-17).

## 2. Paired macro-F1 differences (MAGICC v5 − comparator)

Primary inference is the paired cluster-bootstrap interval and its two-sided p. The confirmatory test is a one-sample t-test on delete-one-cluster jackknife pseudo-values — a genuinely different estimator, which agrees with the bootstrap in every cell. A Wilcoxon signed-rank test on the same pseudo-values is emitted in the TSV but is **not** a valid confirmatory test here: most reference clusters have zero leverage on macro F1 while a few have large leverage, so the pseudo-values are strongly skewed and the rank test answers whether their *median* is zero rather than whether the difference is.

Positive favours MAGICC. Class balance per set is printed below the table because Set A contains **zero** low-quality genomes and Set B only two medium-quality ones, which caps the three-class macro F1 by design and is a property of the set, not of the tool.

| set         | informative | comparator | n    | clusters | macro F1 MAGICC | macro F1 comparator | paired diff [95% CI]   | CI width | excludes 0 | p (boot) | q (BH)  | p (jackknife t, confirm.) |
|-------------|-------------|------------|------|----------|-----------------|---------------------|------------------------|----------|------------|----------|---------|---------------------------|
| set_A_v2    | False       | CheckM2    | 1000 | 798      | 0.6328          | 0.6328              | -0.000 [-0.012, 0.014] | 0.0262   | False      | 0.955    | 0.955   | 0.998                     |
| set_A_v2    | False       | CoCoPyE    | 1000 | 798      | 0.6328          | 0.6254              | +0.007 [-0.006, 0.022] | 0.0278   | False      | 0.295    | 0.442   | 0.296                     |
| set_A_v2    | False       | DeepCheck  | 1000 | 798      | 0.6328          | 0.63                | +0.003 [-0.010, 0.016] | 0.0254   | False      | 0.711    | 0.761   | 0.671                     |
| set_B_v2    | False       | CheckM2    | 1000 | 803      | 0.7783          | 0.5984              | +0.180 [0.057, 0.291]  | 0.2339   | True       | 0.001    | 0.00167 | 0.00817                   |
| set_B_v2    | False       | CoCoPyE    | 1000 | 803      | 0.7783          | 0.7349              | +0.043 [-0.017, 0.132] | 0.149    | False      | 0.382    | 0.49    | 0.248                     |
| set_B_v2    | False       | DeepCheck  | 1000 | 803      | 0.7783          | 0.499               | +0.279 [0.148, 0.399]  | 0.2514   | True       | 0.001    | 0.00167 | 0.000156                  |
| set_C_clean | True        | CheckM2    | 1000 | 100      | 0.6922          | 0.2635              | +0.429 [0.353, 0.507]  | 0.1541   | True       | 0.001    | 0.00167 | 5.03e-19                  |
| set_C_clean | True        | CoCoPyE    | 1000 | 100      | 0.6922          | 0.4154              | +0.277 [0.208, 0.340]  | 0.1316   | True       | 0.001    | 0.00167 | 1.39e-12                  |
| set_C_clean | True        | DeepCheck  | 1000 | 100      | 0.6922          | 0.2431              | +0.449 [0.370, 0.518]  | 0.148    | True       | 0.001    | 0.00167 | 3.82e-20                  |
| set_D_clean | True        | CheckM2    | 1000 | 100      | 0.7881          | 0.3824              | +0.406 [0.316, 0.485]  | 0.169    | True       | 0.001    | 0.00167 | 1.96e-15                  |
| set_D_clean | True        | CoCoPyE    | 1000 | 100      | 0.7881          | 0.7562              | +0.032 [-0.076, 0.131] | 0.2067   | False      | 0.557    | 0.642   | 0.54                      |
| set_D_clean | True        | DeepCheck  | 1000 | 100      | 0.7881          | 0.2207              | +0.567 [0.474, 0.647]  | 0.173    | True       | 0.001    | 0.00167 | 2.53e-23                  |
| set_E       | True        | CheckM2    | 1000 | 785      | 0.9032          | 0.814               | +0.089 [0.053, 0.125]  | 0.0722   | True       | 0.001    | 0.00167 | 3.25e-06                  |
| set_E       | True        | CoCoPyE    | 1000 | 785      | 0.9032          | 0.8878              | +0.015 [-0.020, 0.055] | 0.0758   | False      | 0.392    | 0.49    | 0.422                     |
| set_E       | True        | DeepCheck  | 1000 | 785      | 0.9032          | 0.6948              | +0.208 [0.170, 0.247]  | 0.0768   | True       | 0.001    | 0.00167 | 1.23e-24                  |

* `set_A_v2` true class balance: {'medium': 724, 'high': 276}
* `set_B_v2` true class balance: {'low': 798, 'high': 200, 'medium': 2}
* `set_C_clean` true class balance: {'low': 900, 'medium': 70, 'high': 30}
* `set_D_clean` true class balance: {'low': 887, 'medium': 100, 'high': 13}
* `set_E` true class balance: {'low': 719, 'medium': 215, 'high': 66}

## 3. Per-class F1 — is a macro-F1 margin driven by one class?

| set         | comparator | class  | F1 MAGICC | F1 comparator | paired diff [95% CI]   | excludes 0 | q (BH)  |
|-------------|------------|--------|-----------|---------------|------------------------|------------|---------|
| set_A_v2    | CheckM2    | high   | 0.9248    | 0.9391        | -0.014 [-0.043, 0.015] | False      | 0.404   |
| set_A_v2    | CheckM2    | medium | 0.9737    | 0.9594        | +0.014 [0.002, 0.028]  | True       | 0.03    |
| set_A_v2    | CheckM2    | low    | 0         | 0             | +0.000 [0.000, 0.000]  | False      | 1       |
| set_A_v2    | CoCoPyE    | high   | 0.9248    | 0.9078        | +0.017 [-0.014, 0.050] | False      | 0.404   |
| set_A_v2    | CoCoPyE    | medium | 0.9737    | 0.9685        | +0.005 [-0.007, 0.017] | False      | 0.389   |
| set_A_v2    | CoCoPyE    | low    | 0         | 0             | +0.000 [0.000, 0.000]  | False      | 1       |
| set_A_v2    | DeepCheck  | high   | 0.9248    | 0.9359        | -0.011 [-0.038, 0.017] | False      | 0.517   |
| set_A_v2    | DeepCheck  | medium | 0.9737    | 0.9542        | +0.020 [0.007, 0.033]  | True       | 0.00562 |
| set_A_v2    | DeepCheck  | low    | 0         | 0             | +0.000 [0.000, 0.000]  | False      | 1       |
| set_B_v2    | CheckM2    | high   | 0.9744    | 0.8753        | +0.099 [0.066, 0.134]  | True       | 0.00375 |
| set_B_v2    | CheckM2    | medium | 0.3636    | 0.0213        | +0.342 [0.000, 0.667]  | False      | 0.292   |
| set_B_v2    | CheckM2    | low    | 0.9969    | 0.8986        | +0.098 [0.081, 0.115]  | True       | 0.00167 |
| set_B_v2    | CoCoPyE    | high   | 0.9744    | 0.9899        | -0.016 [-0.036, 0.003] | False      | 0.145   |
| set_B_v2    | CoCoPyE    | medium | 0.3636    | 0.2222        | +0.141 [-0.026, 0.398] | False      | 0.389   |
| set_B_v2    | CoCoPyE    | low    | 0.9969    | 0.9924        | +0.004 [-0.001, 0.009] | False      | 0.0709  |
| set_B_v2    | DeepCheck  | high   | 0.9744    | 0.7538        | +0.221 [0.178, 0.263]  | True       | 0.00375 |
| set_B_v2    | DeepCheck  | medium | 0.3636    | 0             | +0.364 [0.000, 0.714]  | False      | 0.292   |
| set_B_v2    | DeepCheck  | low    | 0.9969    | 0.7433        | +0.254 [0.227, 0.281]  | True       | 0.00167 |
| set_C_clean | CheckM2    | high   | 0.566     | 0.3141        | +0.252 [0.066, 0.424]  | True       | 0.0129  |
| set_C_clean | CheckM2    | medium | 0.5414    | 0.111         | +0.430 [0.324, 0.529]  | True       | 0.00214 |
| set_C_clean | CheckM2    | low    | 0.9691    | 0.3653        | +0.604 [0.563, 0.648]  | True       | 0.00167 |
| set_C_clean | CoCoPyE    | high   | 0.566     | 0             | +0.566 [0.383, 0.717]  | True       | 0.00375 |
| set_C_clean | CoCoPyE    | medium | 0.5414    | 0.349         | +0.192 [0.095, 0.288]  | True       | 0.00214 |
| set_C_clean | CoCoPyE    | low    | 0.9691    | 0.8971        | +0.072 [0.057, 0.088]  | True       | 0.00167 |
| set_C_clean | DeepCheck  | high   | 0.566     | 0.2609        | +0.305 [0.103, 0.484]  | True       | 0.006   |
| set_C_clean | DeepCheck  | medium | 0.5414    | 0.1226        | +0.419 [0.311, 0.521]  | True       | 0.00214 |
| set_C_clean | DeepCheck  | low    | 0.9691    | 0.3458        | +0.623 [0.582, 0.668]  | True       | 0.00167 |
| set_D_clean | CheckM2    | high   | 0.5714    | 0.2857        | +0.286 [0.043, 0.492]  | True       | 0.045   |
| set_D_clean | CheckM2    | medium | 0.8116    | 0.2551        | +0.557 [0.503, 0.607]  | True       | 0.00214 |
| set_D_clean | CheckM2    | low    | 0.9813    | 0.6064        | +0.375 [0.333, 0.416]  | True       | 0.00167 |
| set_D_clean | CoCoPyE    | high   | 0.5714    | 0.5946        | -0.023 [-0.310, 0.249] | False      | 0.829   |
| set_D_clean | CoCoPyE    | medium | 0.8116    | 0.7074        | +0.104 [0.035, 0.175]  | True       | 0.00666 |
| set_D_clean | CoCoPyE    | low    | 0.9813    | 0.9666        | +0.015 [0.004, 0.027]  | True       | 0.0105  |
| set_D_clean | DeepCheck  | high   | 0.5714    | 0.1832        | +0.388 [0.141, 0.587]  | True       | 0.0075  |
| set_D_clean | DeepCheck  | medium | 0.8116    | 0.1964        | +0.615 [0.561, 0.664]  | True       | 0.00214 |
| set_D_clean | DeepCheck  | low    | 0.9813    | 0.2824        | +0.699 [0.650, 0.746]  | True       | 0.00167 |
| set_E       | CheckM2    | high   | 0.8217    | 0.76          | +0.062 [-0.019, 0.146] | False      | 0.187   |
| set_E       | CheckM2    | medium | 0.9074    | 0.7674        | +0.140 [0.097, 0.185]  | True       | 0.00214 |
| set_E       | CheckM2    | low    | 0.9804    | 0.9145        | +0.066 [0.049, 0.084]  | True       | 0.00167 |
| set_E       | CoCoPyE    | high   | 0.8217    | 0.8031        | +0.019 [-0.070, 0.112] | False      | 0.68    |
| set_E       | CoCoPyE    | medium | 0.9074    | 0.8838        | +0.024 [-0.008, 0.056] | False      | 0.207   |
| set_E       | CoCoPyE    | low    | 0.9804    | 0.9763        | +0.004 [-0.004, 0.013] | False      | 0.455   |
| set_E       | DeepCheck  | high   | 0.8217    | 0.659         | +0.163 [0.081, 0.248]  | True       | 0.00375 |
| set_E       | DeepCheck  | medium | 0.9074    | 0.6166        | +0.291 [0.248, 0.336]  | True       | 0.00214 |
| set_E       | DeepCheck  | low    | 0.9804    | 0.8089        | +0.171 [0.147, 0.197]  | True       | 0.00167 |

## 4. Power statement for the negative results

Exact n, the named denominator, the cluster count, the 95% cluster-bootstrap interval and its **width** — for the 44.2% false-fail rate, its companion false-pass rate and balanced accuracy, and for the ground-truthed anchor.

**set_C_clean, MIMAG-inspired 5% contamination boundary**

| quantity          | tool      | estimate [95% CI]    | CI width | numerator | denominator n | cohort clusters | denominator                                                                                    |
|-------------------|-----------|----------------------|----------|-----------|---------------|-----------------|------------------------------------------------------------------------------------------------|
| false_fail_rate   | MAGICC v5 | 0.442 [0.273, 0.617] | 0.344    | 23        | 52            | 40              | the 52 Set C-clean genomes whose TRUE contamination is < 5% (NOT the 1,000 genomes of the set) |
| false_pass_rate   | MAGICC v5 | 0.020 [0.013, 0.028] | 0.0158   | 19        | 948           | 100             | the 948 Set C-clean genomes whose TRUE contamination is >= 5%                                  |
| balanced_accuracy | MAGICC v5 | 0.769 [0.682, 0.854] | 0.1712   |           | 1000          | 100             | mean of sensitivity (948 truly contaminated) and specificity (52 truly clean)                  |
| false_fail_rate   | CheckM2   | 0.000 [0.000, 0.000] | 0        | 0         | 52            | 40              | the 52 Set C-clean genomes whose TRUE contamination is < 5% (NOT the 1,000 genomes of the set) |
| false_pass_rate   | CheckM2   | 0.520 [0.485, 0.557] | 0.0718   | 493       | 948           | 100             | the 948 Set C-clean genomes whose TRUE contamination is >= 5%                                  |
| balanced_accuracy | CheckM2   | 0.740 [0.722, 0.757] | 0.0359   |           | 1000          | 100             | mean of sensitivity (948 truly contaminated) and specificity (52 truly clean)                  |
| false_fail_rate   | CoCoPyE   | 0.712 [0.580, 0.830] | 0.2498   | 37        | 52            | 40              | the 52 Set C-clean genomes whose TRUE contamination is < 5% (NOT the 1,000 genomes of the set) |
| false_pass_rate   | CoCoPyE   | 0.044 [0.030, 0.061] | 0.0314   | 42        | 948           | 100             | the 948 Set C-clean genomes whose TRUE contamination is >= 5%                                  |
| balanced_accuracy | CoCoPyE   | 0.622 [0.565, 0.686] | 0.1214   |           | 1000          | 100             | mean of sensitivity (948 truly contaminated) and specificity (52 truly clean)                  |
| false_fail_rate   | DeepCheck | 0.000 [0.000, 0.000] | 0        | 0         | 52            | 40              | the 52 Set C-clean genomes whose TRUE contamination is < 5% (NOT the 1,000 genomes of the set) |
| false_pass_rate   | DeepCheck | 0.699 [0.664, 0.735] | 0.0712   | 663       | 948           | 100             | the 948 Set C-clean genomes whose TRUE contamination is >= 5%                                  |
| balanced_accuracy | DeepCheck | 0.650 [0.632, 0.668] | 0.0356   |           | 1000          | 100             | mean of sensitivity (948 truly contaminated) and specificity (52 truly clean)                  |

**set_C_clean ground-truthed reduced-genome anchor: genomes whose TRUE MIMAG-inspired class is high (completeness >= 90% AND contamination < 5%)**

| quantity                  | tool      | estimate [95% CI]          | CI width | numerator | denominator n | cohort clusters | denominator                                                                                                             |
|---------------------------|-----------|----------------------------|----------|-----------|---------------|-----------------|-------------------------------------------------------------------------------------------------------------------------|
| signed_bias_completeness  | MAGICC v5 | -8.677 [-14.169, -3.745]   | 10.42    |           | 30            | 25              | 30 of the 1,000 Set C-clean genomes (25 reference clusters) are truly high-quality; signed error = predicted - true, pp |
| signed_bias_contamination | MAGICC v5 | 5.091 [1.800, 8.687]       | 6.887    |           | 30            | 25              | 30 of the 1,000 Set C-clean genomes (25 reference clusters) are truly high-quality; signed error = predicted - true, pp |
| signed_bias_completeness  | CheckM2   | -0.701 [-1.436, 0.035]     | 1.472    |           | 30            | 25              | 30 of the 1,000 Set C-clean genomes (25 reference clusters) are truly high-quality; signed error = predicted - true, pp |
| signed_bias_contamination | CheckM2   | -1.162 [-1.754, -0.607]    | 1.148    |           | 30            | 25              | 30 of the 1,000 Set C-clean genomes (25 reference clusters) are truly high-quality; signed error = predicted - true, pp |
| signed_bias_completeness  | CoCoPyE   | -21.358 [-23.286, -19.288] | 3.998    |           | 30            | 25              | 30 of the 1,000 Set C-clean genomes (25 reference clusters) are truly high-quality; signed error = predicted - true, pp |
| signed_bias_contamination | CoCoPyE   | 5.896 [4.631, 7.137]       | 2.506    |           | 30            | 25              | 30 of the 1,000 Set C-clean genomes (25 reference clusters) are truly high-quality; signed error = predicted - true, pp |
| signed_bias_completeness  | DeepCheck | -12.168 [-14.309, -10.096] | 4.213    |           | 30            | 25              | 30 of the 1,000 Set C-clean genomes (25 reference clusters) are truly high-quality; signed error = predicted - true, pp |
| signed_bias_contamination | DeepCheck | -1.775 [-2.255, -1.303]    | 0.9518   |           | 30            | 25              | 30 of the 1,000 Set C-clean genomes (25 reference clusters) are truly high-quality; signed error = predicted - true, pp |

## 5. Ready-to-paste sentences

See `ready_to_paste_sentences.md` in this directory.

## 6. Determinism (defect D1)

All TSV outputs byte-identical under `PYTHONHASHSEED=0` and `PYTHONHASHSEED=99999`: **True**

| file                                     | sha256_hashseed0                                                 | sha256_hashseed99999                                             | identical |
|------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|-----------|
| macro_f1_paired_differences.tsv          | 86235d911ac7b6021005a0294ccdd8601ad2d129931238cdc1442e85a5a27d13 | 86235d911ac7b6021005a0294ccdd8601ad2d129931238cdc1442e85a5a27d13 | True      |
| macro_f1_point_estimate_verification.tsv | 15b5afaae767da7b3db96e3731935ab8989b2dbb9b7c61d4b345703501537d18 | 15b5afaae767da7b3db96e3731935ab8989b2dbb9b7c61d4b345703501537d18 | True      |
| per_class_f1_paired.tsv                  | 0d1de6f1cda865121ac6cba9548131ed5b55c3e0d5995ee6e788c41d5fbf53b9 | 0d1de6f1cda865121ac6cba9548131ed5b55c3e0d5995ee6e788c41d5fbf53b9 | True      |
| power_statement.tsv                      | 3a03378dc3acf94d15d032cd5c84a42a038136b109eea7370ce00f28437de997 | 3a03378dc3acf94d15d032cd5c84a42a038136b109eea7370ce00f28437de997 | True      |

## 7. Input verification

Every set carries exactly 1,000 rows, all four tool prediction tables join 1:1 to `metadata.tsv`, and every macro-F1 point estimate reproduces the WS5 definitive table exactly. No row was dropped.

