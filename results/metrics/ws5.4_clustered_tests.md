# WS5.4 - Two-sided, cluster-aware, multiplicity-corrected statistics

Denominators: completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

### Statistical methods (text for the Methods section)

Every tool-vs-tool comparison is a **two-sided** test; the one-sided Wilcoxon
signed-rank tests of the original submission have been withdrawn because they
presuppose the direction of the effect they are used to establish.

For each benchmark set and each accuracy measure (absolute completeness error,
absolute contamination error) we form the paired per-genome difference
d_i = |error_MAGICC,i| - |error_competitor,i|, so that d_i < 0 means MAGICC was
more accurate on genome i. Three complementary tests are reported:

1. **Genome-level two-sided Wilcoxon signed-rank test.** This treats the
   simulated genomes as independent and is included only for comparability with
   the original submission; it is anti-conservative whenever several simulated
   genomes derive from the same reference genome.

2. **Cluster-mean two-sided Wilcoxon signed-rank test.** Absolute errors are
   first averaged within each dominant reference genome; the test is then run on
   the resulting per-reference means, so the unit of analysis is the reference
   genome and the number of independent units equals the number of distinct
   reference genomes (e.g. 100 for Sets C-clean/D-clean, which contain 10
   independent simulations per reference).

3. **Cluster bootstrap.** Reference genomes (clusters), not individual simulated
   genomes, are resampled with replacement 2,000 times, taking all simulated
   genomes belonging to each sampled reference. Percentile 95% confidence
   intervals and two-sided p-values (p = 2 x min[Pr(theta* <= 0), Pr(theta* >= 0)],
   with a +1/(B+1) continuity correction, so the smallest attainable p-value is
   ~1/1000 at 2,000 replicates) are obtained for the mean paired difference -
   which is exactly the difference in MAE between the two tools - and for the
   median paired difference. A coverage simulation with 40 clusters of 10
   observations and an intraclass correlation of 0.8 gives 95.0% coverage for the
   cluster bootstrap versus 54.0% for the naive bootstrap (script 101
   --selftest), confirming that ignoring clustering understates uncertainty
   severely in exactly the design used for Sets C-clean/D-clean.

**Which test is the primary one.** Test 2 (cluster-mean Wilcoxon) is the primary
decision rule: it is rank-based, two-sided, and treats the reference genome as
the sampling unit, which is precisely what Reviewer 1 asks for. Test 3 on the
mean paired difference is the confirmatory test, because the difference in MAE is
the quantity the manuscript's accuracy claims are about. Test 1 is reported for
comparability only. A comparison is called robust only when all three survive BH
correction (column `significant_all_three_bh`). Note that the bootstrap p-value
for the MEDIAN paired difference can be large even when the mean difference is
decisively non-zero: when most paired differences are near zero and the advantage
comes from the tails (as for completeness on Set A), the median of d is close to
zero by construction. That is a property of the statistic, not a failure of the
test, which is why the mean (dMAE) version is the confirmatory one.

A sensitivity analysis repeats (3) with clusters defined by dominant **phylum**
instead of dominant genome, i.e. the most conservative taxonomic grouping the
data support.

**Multiplicity.** Benjamini-Hochberg FDR correction at q < 0.05 is applied
across the entire family of tests reported in one run, separately for the
leakage-free ("primary") sets, for the superseded leaky Sets C/D ("reported"),
and for the secondary/legacy sets, so that the primary family is not diluted by
withdrawn results.

**Effect sizes.** Every p-value is accompanied by (i) the Hodges-Lehmann
estimator of the median paired difference (median of the Walsh averages of d)
in percentage points, with a cluster-bootstrap 95% CI; (ii) Cliff's delta
comparing the two absolute-error distributions, with a cluster-bootstrap 95% CI
and the conventional magnitude labels (|delta| < 0.147 negligible, < 0.33 small,
< 0.474 medium, else large); and (iii) the matched-pairs rank-biserial
correlation, the effect size that corresponds directly to the signed-rank
statistic. Negative Hodges-Lehmann differences and negative Cliff's delta both
indicate that MAGICC is the more accurate tool. To keep run time acceptable the
O(n^2) Walsh-average bootstrap CI for the Hodges-Lehmann estimator is computed
for the primary and reported (superseded) sets only; for the secondary/legacy
sets the Hodges-Lehmann point estimate is given without an interval.


Bootstrap replicates: 2000 (median/mean/Cliff's delta), 1000 (Hodges-Lehmann). BH alpha = 0.05.

**Sign convention: negative values favour MAGICC.**

## Primary family (leakage-free sets)

Family size m = 36 tests; BH applied within this family.

### abs_err_completeness

| set         | MAGICC v5 vs | n / clusters | MAE ref | MAE comp | HL median diff (pp) [95% CI] | Cliff delta [95% CI]    | |delta|    | r_rb   | mean diff = dMAE (pp) [95% CI] | p naive  | p cluster-mean (PRIMARY) | q BH cluster-mean (PRIMARY) | p cluster-boot (dMAE) | q BH cluster-boot (dMAE) | survives BH (all 3) | favours    |
|-------------|--------------|--------------|---------|----------|------------------------------|-------------------------|------------|--------|--------------------------------|----------|--------------------------|-----------------------------|-----------------------|--------------------------|---------------------|------------|
| set_A_v2    | magicc_v4    | 1000/798     | 2.177   | 2.591    | -0.19 [-0.25, -0.13]         | -0.148 [-0.179, -0.116] | small      | -0.213 | -0.41 [-0.54, -0.30]           | 5e-09    | 2.3e-07                  | 2.9e-07                     | 0.001                 | 0.0013                   | True                | reference  |
| set_A_v2    | checkm2      | 1000/798     | 2.177   | 2.542    | -0.50 [-0.67, -0.32]         | -0.075 [-0.115, -0.034] | negligible | -0.229 | -0.37 [-0.64, -0.07]           | 3.5e-10  | 4e-11                    | 6e-11                       | 0.02                  | 0.024                    | True                | reference  |
| set_A_v2    | cocopye      | 1000/798     | 2.177   | 3.627    | -1.25 [-1.48, -1.00]         | -0.206 [-0.248, -0.164] | small      | -0.443 | -1.45 [-1.80, -1.08]           | 6.5e-34  | 6.2e-32                  | 1.9e-31                     | 0.001                 | 0.0013                   | True                | reference  |
| set_A_v2    | deepcheck    | 1000/798     | 2.177   | 4.255    | -1.72 [-1.97, -1.47]         | -0.404 [-0.444, -0.362] | medium     | -0.598 | -2.08 [-2.45, -1.70]           | 3.3e-60  | 3.5e-53                  | 1.2e-52                     | 0.001                 | 0.0013                   | True                | reference  |
| set_B_v2    | magicc_v4    | 1000/803     | 2.972   | 3.124    | -0.04 [-0.11, +0.02]         | -0.105 [-0.133, -0.079] | negligible | -0.049 | -0.15 [-0.33, +0.03]           | 0.18     | 0.2                      | 0.21                        | 0.1                   | 0.11                     | False               | reference  |
| set_B_v2    | checkm2      | 1000/803     | 2.972   | 0.451    | +1.15 [+0.94, +1.40]         | +0.828 [+0.797, +0.858] | large      | 0.812  | +2.52 [+2.14, +2.90]           | 1e-109   | 2.8e-88                  | 2e-87                       | 0.001                 | 0.0013                   | True                | comparison |
| set_B_v2    | cocopye      | 1000/803     | 2.972   | 2.608    | +0.20 [+0.13, +0.30]         | +0.273 [+0.229, +0.317] | small      | 0.18   | +0.36 [-0.05, +0.80]           | 8.6e-07  | 0.00052                  | 0.00062                     | 0.087                 | 0.098                    | False               | comparison |
| set_B_v2    | deepcheck    | 1000/803     | 2.972   | 6.611    | -2.30 [-2.81, -1.87]         | -0.381 [-0.422, -0.338] | medium     | -0.537 | -3.64 [-4.26, -3.02]           | 6.1e-49  | 1.9e-43                  | 6.3e-43                     | 0.001                 | 0.0013                   | True                | reference  |
| set_E       | magicc_v4    | 1000/785     | 5.486   | 5.71     | -0.16 [-0.26, -0.07]         | -0.030 [-0.049, -0.011] | negligible | -0.121 | -0.22 [-0.38, -0.07]           | 0.0009   | 0.00085                  | 0.00099                     | 0.008                 | 0.01                     | True                | reference  |
| set_E       | checkm2      | 1000/785     | 5.486   | 9.14     | -2.81 [-3.50, -2.16]         | -0.050 [-0.096, -0.004] | negligible | -0.372 | -3.65 [-4.29, -3.03]           | 2.2e-24  | 1.6e-23                  | 4e-23                       | 0.001                 | 0.0013                   | True                | reference  |
| set_E       | cocopye      | 1000/785     | 5.486   | 5.022    | +0.13 [-0.16, +0.44]         | -0.008 [-0.051, +0.033] | negligible | 0.034  | +0.46 [+0.04, +0.89]           | 0.34     | 0.35                     | 0.36                        | 0.035                 | 0.041                    | False               | comparison |
| set_E       | deepcheck    | 1000/785     | 5.486   | 11.7     | -5.36 [-6.03, -4.71]         | -0.369 [-0.410, -0.329] | medium     | -0.634 | -6.21 [-6.93, -5.51]           | 1.5e-67  | 9.1e-60                  | 3.6e-59                     | 0.001                 | 0.0013                   | True                | reference  |
| set_C_clean | checkm2      | 1000/100     | 6.822   | 7.711    | -1.44 [-2.40, -0.33]         | -0.183 [-0.284, -0.086] | small      | -0.177 | -0.89 [-2.02, +0.23]           | 1.3e-06  | 0.016                    | 0.018                       | 0.13                  | 0.14                     | False               | reference  |
| set_C_clean | cocopye      | 1000/100     | 6.822   | 15.8     | -8.75 [-9.96, -7.46]         | -0.560 [-0.634, -0.489] | large      | -0.701 | -8.98 [-10.26, -7.75]          | 4.4e-82  | 3.9e-16                  | 6.1e-16                     | 0.001                 | 0.0013                   | True                | reference  |
| set_C_clean | deepcheck    | 1000/100     | 6.822   | 17.81    | -11.16 [-12.40, -9.76]       | -0.642 [-0.706, -0.580] | large      | -0.795 | -10.99 [-12.41, -9.66]         | 3.1e-105 | 3.9e-17                  | 6.3e-17                     | 0.001                 | 0.0013                   | True                | reference  |
| set_D_clean | checkm2      | 1000/100     | 5.633   | 9.171    | -3.28 [-4.16, -2.44]         | -0.206 [-0.266, -0.143] | small      | -0.407 | -3.54 [-4.47, -2.49]           | 6.5e-29  | 5.1e-11                  | 7.4e-11                     | 0.001                 | 0.0013                   | True                | reference  |
| set_D_clean | cocopye      | 1000/100     | 5.633   | 5.891    | -0.35 [-0.92, +0.25]         | -0.064 [-0.137, +0.011] | negligible | -0.068 | -0.26 [-1.15, +0.62]           | 0.062    | 0.4                      | 0.4                         | 0.57                  | 0.59                     | False               | reference  |
| set_D_clean | deepcheck    | 1000/100     | 5.633   | 8.782    | -2.70 [-3.52, -1.88]         | -0.257 [-0.326, -0.186] | small      | -0.385 | -3.15 [-4.21, -2.05]           | 4.9e-26  | 1.5e-09                  | 2.1e-09                     | 0.001                 | 0.0013                   | True                | reference  |

### abs_err_contamination

| set         | MAGICC v5 vs | n / clusters | MAE ref | MAE comp | HL median diff (pp) [95% CI] | Cliff delta [95% CI]    | |delta|    | r_rb   | mean diff = dMAE (pp) [95% CI] | p naive  | p cluster-mean (PRIMARY) | q BH cluster-mean (PRIMARY) | p cluster-boot (dMAE) | q BH cluster-boot (dMAE) | survives BH (all 3) | favours    |
|-------------|--------------|--------------|---------|----------|------------------------------|-------------------------|------------|--------|--------------------------------|----------|--------------------------|-----------------------------|-----------------------|--------------------------|---------------------|------------|
| set_A_v2    | magicc_v4    | 1000/798     | 0.829   | 1.478    | -0.20 [-0.24, -0.16]         | -0.162 [-0.184, -0.143] | small      | -0.649 | -0.65 [-0.78, -0.53]           | 1.2e-70  | 1.5e-63                  | 6.6e-63                     | 0.001                 | 0.0013                   | True                | reference  |
| set_A_v2    | checkm2      | 1000/798     | 0.829   | 0.274    | +0.15 [+0.12, +0.18]         | +0.295 [+0.249, +0.340] | small      | 0.422  | +0.55 [+0.37, +0.79]           | 5.7e-31  | 4.6e-25                  | 1.3e-24                     | 0.001                 | 0.0013                   | True                | comparison |
| set_A_v2    | cocopye      | 1000/798     | 0.829   | 0.774    | +0.19 [+0.16, +0.22]         | +0.565 [+0.512, +0.615] | large      | 0.412  | +0.05 [-0.23, +0.33]           | 1.4e-29  | 1.9e-23                  | 4.7e-23                     | 0.74                  | 0.74                     | False               | comparison |
| set_A_v2    | deepcheck    | 1000/798     | 0.829   | 0.404    | +0.07 [+0.04, +0.10]         | +0.118 [+0.064, +0.171] | negligible | 0.168  | +0.42 [+0.23, +0.66]           | 4.1e-06  | 0.00048                  | 0.00059                     | 0.001                 | 0.0013                   | True                | comparison |
| set_B_v2    | magicc_v4    | 1000/803     | 4.453   | 5.11     | -0.45 [-0.62, -0.29]         | -0.097 [-0.126, -0.066] | negligible | -0.203 | -0.66 [-0.88, -0.44]           | 2.7e-08  | 5.2e-09                  | 6.9e-09                     | 0.001                 | 0.0013                   | True                | reference  |
| set_B_v2    | checkm2      | 1000/803     | 4.453   | 17.66    | -11.21 [-12.55, -9.91]       | -0.420 [-0.459, -0.379] | medium     | -0.743 | -13.21 [-14.28, -12.11]        | 5.7e-92  | 1.2e-83                  | 7.3e-83                     | 0.001                 | 0.0013                   | True                | reference  |
| set_B_v2    | cocopye      | 1000/803     | 4.453   | 19.14    | -13.77 [-14.71, -12.63]      | -0.392 [-0.430, -0.350] | medium     | -0.772 | -14.69 [-15.74, -13.68]        | 2.3e-99  | 1.8e-90                  | 2.2e-89                     | 0.001                 | 0.0013                   | True                | reference  |
| set_B_v2    | deepcheck    | 1000/803     | 4.453   | 25.7     | -19.59 [-21.06, -18.13]      | -0.576 [-0.614, -0.534] | large      | -0.895 | -21.25 [-22.57, -19.95]        | 1e-132   | 1.4e-114                 | 5.2e-113                    | 0.001                 | 0.0013                   | True                | reference  |
| set_E       | magicc_v4    | 1000/785     | 5.658   | 5.876    | -0.06 [-0.16, +0.03]         | -0.041 [-0.060, -0.022] | negligible | -0.049 | -0.22 [-0.39, -0.05]           | 0.18     | 0.092                    | 0.1                         | 0.009                 | 0.011                    | False               | reference  |
| set_E       | checkm2      | 1000/785     | 5.658   | 18.47    | -10.78 [-12.09, -9.35]       | -0.346 [-0.380, -0.308] | medium     | -0.74  | -12.81 [-13.94, -11.70]        | 2.7e-91  | 4e-82                    | 2.1e-81                     | 0.001                 | 0.0013                   | True                | reference  |
| set_E       | cocopye      | 1000/785     | 5.658   | 21.6     | -15.08 [-16.65, -13.50]      | -0.333 [-0.371, -0.294] | medium     | -0.764 | -15.94 [-17.16, -14.76]        | 3.3e-97  | 2.3e-88                  | 2e-87                       | 0.001                 | 0.0013                   | True                | reference  |
| set_E       | deepcheck    | 1000/785     | 5.658   | 25.43    | -17.58 [-19.07, -15.94]      | -0.458 [-0.495, -0.420] | medium     | -0.848 | -19.77 [-21.33, -18.30]        | 2e-119   | 1.5e-103                 | 2.7e-102                    | 0.001                 | 0.0013                   | True                | reference  |
| set_C_clean | checkm2      | 1000/100     | 9.08    | 39.62    | -30.54 [-32.21, -28.84]      | -0.768 [-0.801, -0.732] | large      | -0.959 | -30.54 [-32.18, -28.86]        | 3.9e-152 | 3.9e-18                  | 6.7e-18                     | 0.001                 | 0.0013                   | True                | reference  |
| set_C_clean | cocopye      | 1000/100     | 9.08    | 31.07    | -21.86 [-23.46, -20.17]      | -0.615 [-0.658, -0.568] | large      | -0.882 | -21.99 [-23.57, -20.37]        | 5.4e-129 | 3.9e-18                  | 6.7e-18                     | 0.001                 | 0.0013                   | True                | reference  |
| set_C_clean | deepcheck    | 1000/100     | 9.08    | 41.67    | -32.63 [-34.32, -30.88]      | -0.791 [-0.823, -0.756] | large      | -0.967 | -32.59 [-34.27, -30.85]        | 1.1e-154 | 3.9e-18                  | 6.7e-18                     | 0.001                 | 0.0013                   | True                | reference  |
| set_D_clean | checkm2      | 1000/100     | 6.523   | 34.7     | -28.01 [-29.43, -26.55]      | -0.787 [-0.811, -0.761] | large      | -0.986 | -28.18 [-29.48, -26.82]        | 8.7e-161 | 3.9e-18                  | 6.7e-18                     | 0.001                 | 0.0013                   | True                | reference  |
| set_D_clean | cocopye      | 1000/100     | 6.523   | 22.49    | -15.40 [-16.58, -14.24]      | -0.563 [-0.600, -0.528] | large      | -0.89  | -15.97 [-17.01, -14.93]        | 2.5e-131 | 3.9e-18                  | 6.7e-18                     | 0.001                 | 0.0013                   | True                | reference  |
| set_D_clean | deepcheck    | 1000/100     | 6.523   | 38.96    | -32.37 [-33.86, -30.90]      | -0.815 [-0.839, -0.792] | large      | -0.989 | -32.43 [-33.81, -31.04]        | 9.2e-162 | 3.9e-18                  | 6.7e-18                     | 0.001                 | 0.0013                   | True                | reference  |

## Superseded family (Sets C/D - training-set leakage; reported for transparency only)

Family size m = 16 tests; BH applied within this family.

### abs_err_completeness

| set   | MAGICC v5 vs | n / clusters | MAE ref | MAE comp | HL median diff (pp) [95% CI] | Cliff delta [95% CI]    | |delta|    | r_rb   | mean diff = dMAE (pp) [95% CI] | p naive  | p cluster-mean (PRIMARY) | q BH cluster-mean (PRIMARY) | p cluster-boot (dMAE) | q BH cluster-boot (dMAE) | survives BH (all 3) | favours   |
|-------|--------------|--------------|---------|----------|------------------------------|-------------------------|------------|--------|--------------------------------|----------|--------------------------|-----------------------------|-----------------------|--------------------------|---------------------|-----------|
| set_C | magicc_v4    | 1000/1000    | 3.154   | 3.365    | -0.09 [-0.16, -0.04]         | -0.031 [-0.051, -0.011] | negligible | -0.137 | -0.21 [-0.32, -0.10]           | 0.00017  | 0.00017                  | 0.0002                      | 0.001                 | 0.0011                   | True                | reference |
| set_C | checkm2      | 1000/1000    | 3.154   | 7.994    | -4.82 [-5.25, -4.35]         | -0.558 [-0.597, -0.518] | large      | -0.696 | -4.84 [-5.27, -4.42]           | 4.1e-81  | 4.1e-81                  | 7.2e-81                     | 0.001                 | 0.0011                   | True                | reference |
| set_C | cocopye      | 1000/1000    | 3.154   | 15.72    | -12.59 [-13.42, -11.87]      | -0.766 [-0.799, -0.733] | large      | -0.856 | -12.57 [-13.31, -11.83]        | 1.9e-121 | 1.9e-121                 | 3.8e-121                    | 0.001                 | 0.0011                   | True                | reference |
| set_C | deepcheck    | 1000/1000    | 3.154   | 17.97    | -14.88 [-15.60, -14.19]      | -0.849 [-0.873, -0.824] | large      | -0.931 | -14.81 [-15.54, -14.11]        | 2.5e-143 | 2.5e-143                 | 6.7e-143                    | 0.001                 | 0.0011                   | True                | reference |
| set_D | magicc_v4    | 1000/1000    | 5.176   | 5.204    | -0.05 [-0.14, +0.05]         | -0.012 [-0.030, +0.008] | negligible | -0.038 | -0.03 [-0.15, +0.11]           | 0.3      | 0.3                      | 0.3                         | 0.67                  | 0.67                     | False               | reference |
| set_D | checkm2      | 1000/1000    | 5.176   | 9.887    | -4.00 [-4.64, -3.40]         | -0.253 [-0.294, -0.213] | small      | -0.547 | -4.71 [-5.27, -4.17]           | 9.8e-51  | 9.8e-51                  | 1.4e-50                     | 0.001                 | 0.0011                   | True                | reference |
| set_D | cocopye      | 1000/1000    | 5.176   | 6.14     | -0.88 [-1.21, -0.54]         | -0.150 [-0.192, -0.108] | small      | -0.181 | -0.96 [-1.42, -0.48]           | 7.2e-07  | 7.2e-07                  | 8.9e-07                     | 0.001                 | 0.0011                   | True                | reference |
| set_D | deepcheck    | 1000/1000    | 5.176   | 9.975    | -4.13 [-4.68, -3.64]         | -0.348 [-0.385, -0.307] | medium     | -0.606 | -4.80 [-5.35, -4.23]           | 6.3e-62  | 6.3e-62                  | 1e-61                       | 0.001                 | 0.0011                   | True                | reference |

### abs_err_contamination

| set   | MAGICC v5 vs | n / clusters | MAE ref | MAE comp | HL median diff (pp) [95% CI] | Cliff delta [95% CI]    | |delta|    | r_rb   | mean diff = dMAE (pp) [95% CI] | p naive  | p cluster-mean (PRIMARY) | q BH cluster-mean (PRIMARY) | p cluster-boot (dMAE) | q BH cluster-boot (dMAE) | survives BH (all 3) | favours    |
|-------|--------------|--------------|---------|----------|------------------------------|-------------------------|------------|--------|--------------------------------|----------|--------------------------|-----------------------------|-----------------------|--------------------------|---------------------|------------|
| set_C | magicc_v4    | 1000/1000    | 7.974   | 7.439    | +0.53 [+0.38, +0.67]         | +0.049 [+0.032, +0.067] | negligible | 0.268  | +0.54 [+0.37, +0.70]           | 2.1e-13  | 2.1e-13                  | 2.8e-13                     | 0.001                 | 0.0011                   | True                | comparison |
| set_C | checkm2      | 1000/1000    | 7.974   | 42.37    | -34.24 [-35.58, -32.75]      | -0.777 [-0.800, -0.753] | large      | -0.997 | -34.40 [-35.73, -33.01]        | 4.6e-164 | 4.6e-164                 | 3.7e-163                    | 0.001                 | 0.0011                   | True                | reference  |
| set_C | cocopye      | 1000/1000    | 7.974   | 34.07    | -25.85 [-27.07, -24.53]      | -0.652 [-0.680, -0.625] | large      | -0.978 | -26.09 [-27.35, -24.86]        | 3.1e-158 | 3.1e-158                 | 9.8e-158                    | 0.001                 | 0.0011                   | True                | reference  |
| set_C | deepcheck    | 1000/1000    | 7.974   | 44.56    | -36.49 [-37.88, -34.94]      | -0.799 [-0.821, -0.775] | large      | -0.998 | -36.58 [-37.96, -35.18]        | 1.2e-164 | 1.2e-164                 | 1.9e-163                    | 0.001                 | 0.0011                   | True                | reference  |
| set_D | magicc_v4    | 1000/1000    | 8.058   | 7.767    | +0.20 [+0.09, +0.32]         | +0.019 [+0.001, +0.035] | negligible | 0.119  | +0.29 [+0.14, +0.45]           | 0.0012   | 0.0012                   | 0.0012                      | 0.001                 | 0.0011                   | True                | comparison |
| set_D | checkm2      | 1000/1000    | 8.058   | 36.92    | -28.63 [-29.82, -27.36]      | -0.746 [-0.772, -0.721] | large      | -0.989 | -28.87 [-30.09, -27.60]        | 1.6e-161 | 1.6e-161                 | 6.3e-161                    | 0.001                 | 0.0011                   | True                | reference  |
| set_D | cocopye      | 1000/1000    | 8.058   | 25.73    | -17.08 [-18.21, -15.83]      | -0.499 [-0.532, -0.468] | large      | -0.916 | -17.67 [-18.80, -16.60]        | 5.5e-139 | 5.5e-139                 | 1.2e-138                    | 0.001                 | 0.0011                   | True                | reference  |
| set_D | deepcheck    | 1000/1000    | 8.058   | 41.77    | -33.41 [-34.69, -32.06]      | -0.786 [-0.810, -0.762] | large      | -0.995 | -33.72 [-35.16, -32.34]        | 1.1e-163 | 1.1e-163                 | 5.8e-163                    | 0.001                 | 0.0011                   | True                | reference  |

## Secondary family (v1 sets and motivating sets)

Family size m = 42 tests; BH applied within this family.

### abs_err_completeness

| set                 | MAGICC v5 vs | n / clusters | MAE ref | MAE comp | HL median diff (pp) [95% CI] | Cliff delta [95% CI]    | |delta|    | r_rb   | mean diff = dMAE (pp) [95% CI] | p naive  | p cluster-mean (PRIMARY) | q BH cluster-mean (PRIMARY) | p cluster-boot (dMAE) | q BH cluster-boot (dMAE) | survives BH (all 3) | favours    |
|---------------------|--------------|--------------|---------|----------|------------------------------|-------------------------|------------|--------|--------------------------------|----------|--------------------------|-----------------------------|-----------------------|--------------------------|---------------------|------------|
| set_A               | checkm2      | 600/582      | 1.787   | 2.197    | -0.38                        | -0.067 [-0.116, -0.014] | negligible | -0.198 | -0.41 [-0.66, -0.14]           | 2.6e-05  | 1.9e-05                  | 2.5e-05                     | 0.007                 | 0.0086                   | True                | reference  |
| set_A               | cocopye      | 600/582      | 1.787   | 3.365    | -1.37                        | -0.233 [-0.288, -0.180] | small      | -0.474 | -1.58 [-1.93, -1.23]           | 9.2e-24  | 8.9e-23                  | 1.5e-22                     | 0.001                 | 0.0013                   | True                | reference  |
| set_A               | deepcheck    | 600/582      | 1.787   | 3.291    | -1.35                        | -0.413 [-0.463, -0.367] | medium     | -0.587 | -1.50 [-1.82, -1.18]           | 1.1e-35  | 3.2e-34                  | 6.7e-34                     | 0.001                 | 0.0013                   | True                | reference  |
| set_B               | checkm2      | 600/588      | 2.485   | 0.42     | +0.90                        | +0.753 [+0.708, +0.793] | large      | 0.721  | +2.07 [+1.69, +2.48]           | 7.6e-53  | 1.1e-52                  | 3.1e-52                     | 0.001                 | 0.0013                   | True                | comparison |
| set_B               | cocopye      | 600/588      | 2.485   | 2.77     | +0.11                        | +0.176 [+0.120, +0.230] | small      | 0.118  | -0.28 [-0.81, +0.21]           | 0.012    | 0.015                    | 0.017                       | 0.29                  | 0.31                     | False               | reference  |
| set_B               | deepcheck    | 600/588      | 2.485   | 5.411    | -1.73                        | -0.411 [-0.462, -0.356] | medium     | -0.551 | -2.93 [-3.58, -2.26]           | 1.5e-31  | 7.6e-31                  | 1.5e-30                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_A | checkm2      | 1000/806     | 2.195   | 2.541    | -0.42                        | -0.053 [-0.093, -0.013] | negligible | -0.196 | -0.35 [-0.58, -0.10]           | 7.6e-08  | 2.7e-08                  | 3.7e-08                     | 0.005                 | 0.0064                   | True                | reference  |
| motivating_v2_set_A | cocopye      | 1000/806     | 2.195   | 3.471    | -1.04                        | -0.156 [-0.196, -0.114] | small      | -0.363 | -1.28 [-1.56, -1.00]           | 3e-23    | 5e-23                    | 8.8e-23                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_A | deepcheck    | 1000/806     | 2.195   | 4.147    | -1.60                        | -0.372 [-0.412, -0.334] | medium     | -0.553 | -1.95 [-2.25, -1.65]           | 6.6e-52  | 3.4e-48                  | 7.4e-48                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_B | checkm2      | 1000/798     | 2.564   | 0.416    | +0.99                        | +0.828 [+0.798, +0.858] | large      | 0.799  | +2.15 [+1.81, +2.49]           | 3.5e-106 | 5.5e-84                  | 2.6e-83                     | 0.001                 | 0.0013                   | True                | comparison |
| motivating_v2_set_B | cocopye      | 1000/798     | 2.564   | 2.572    | +0.16                        | +0.270 [+0.222, +0.316] | small      | 0.175  | -0.01 [-0.42, +0.39]           | 1.6e-06  | 0.00059                  | 0.00073                     | 0.93                  | 0.96                     | False               | reference  |
| motivating_v2_set_B | deepcheck    | 1000/798     | 2.564   | 6.743    | -2.54                        | -0.422 [-0.464, -0.381] | medium     | -0.593 | -4.18 [-4.80, -3.55]           | 2.8e-59  | 5.3e-51                  | 1.3e-50                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_C | checkm2      | 1000/773     | 5.403   | 9.13     | -2.90                        | -0.067 [-0.111, -0.021] | negligible | -0.39  | -3.73 [-4.36, -3.09]           | 1.1e-26  | 2.3e-26                  | 4.4e-26                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_C | cocopye      | 1000/773     | 5.403   | 4.985    | -0.04                        | -0.023 [-0.065, +0.020] | negligible | -0.01  | +0.42 [-0.00, +0.85]           | 0.79     | 0.75                     | 0.75                        | 0.055                 | 0.064                    | False               | comparison |
| motivating_v2_set_C | deepcheck    | 1000/773     | 5.403   | 11.96    | -5.47                        | -0.361 [-0.403, -0.320] | medium     | -0.649 | -6.55 [-7.34, -5.77]           | 1e-70    | 2.1e-58                  | 6.2e-58                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_set_A    | checkm2      | 600/585      | 2.364   | 2.372    | -0.31                        | -0.034 [-0.088, +0.022] | negligible | -0.142 | -0.01 [-0.36, +0.39]           | 0.0025   | 0.0024                   | 0.0028                      | 1                     | 1                        | False               | reference  |
| motivating_set_A    | cocopye      | 600/585      | 2.364   | 3.744    | -1.04                        | -0.195 [-0.246, -0.144] | small      | -0.378 | -1.38 [-1.76, -1.01]           | 1.1e-15  | 1.7e-15                  | 2.5e-15                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_set_A    | deepcheck    | 600/585      | 2.364   | 4.034    | -1.53                        | -0.383 [-0.434, -0.330] | medium     | -0.505 | -1.67 [-2.08, -1.25]           | 8.1e-27  | 6.7e-26                  | 1.2e-25                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_set_B    | checkm2      | 1100/1031    | 2.197   | 0.514    | +0.48                        | +0.725 [+0.691, +0.760] | large      | 0.692  | +1.68 [+1.40, +1.99]           | 5.9e-88  | 5.7e-83                  | 2.4e-82                     | 0.001                 | 0.0013                   | True                | comparison |
| motivating_set_B    | cocopye      | 1100/1031    | 2.197   | 3.268    | -0.07                        | +0.136 [+0.093, +0.180] | negligible | -0.044 | -1.07 [-1.47, -0.66]           | 0.2      | 0.17                     | 0.18                        | 0.001                 | 0.0013                   | False               | reference  |
| motivating_set_B    | deepcheck    | 1100/1031    | 2.197   | 5.808    | -1.69                        | -0.461 [-0.499, -0.422] | medium     | -0.589 | -3.61 [-4.18, -3.02]           | 3.1e-64  | 2.7e-60                  | 8.6e-60                     | 0.001                 | 0.0013                   | True                | reference  |

### abs_err_contamination

| set                 | MAGICC v5 vs | n / clusters | MAE ref | MAE comp | HL median diff (pp) [95% CI] | Cliff delta [95% CI]    | |delta|    | r_rb   | mean diff = dMAE (pp) [95% CI] | p naive  | p cluster-mean (PRIMARY) | q BH cluster-mean (PRIMARY) | p cluster-boot (dMAE) | q BH cluster-boot (dMAE) | survives BH (all 3) | favours    |
|---------------------|--------------|--------------|---------|----------|------------------------------|-------------------------|------------|--------|--------------------------------|----------|--------------------------|-----------------------------|-----------------------|--------------------------|---------------------|------------|
| set_A               | checkm2      | 600/582      | 0.605   | 0.456    | +0.04                        | +0.119 [+0.062, +0.179] | negligible | 0.108  | +0.15 [+0.03, +0.28]           | 0.022    | 0.025                    | 0.027                       | 0.014                 | 0.017                    | True                | comparison |
| set_A               | cocopye      | 600/582      | 0.605   | 1.079    | -0.10                        | +0.198 [+0.132, +0.271] | small      | -0.086 | -0.47 [-0.67, -0.29]           | 0.068    | 0.048                    | 0.052                       | 0.001                 | 0.0013                   | False               | reference  |
| set_A               | deepcheck    | 600/582      | 0.605   | 0.519    | -0.02                        | -0.025 [-0.092, +0.039] | negligible | -0.044 | +0.09 [-0.04, +0.22]           | 0.35     | 0.29                     | 0.3                         | 0.2                   | 0.22                     | False               | comparison |
| set_B               | checkm2      | 600/588      | 3.989   | 15.14    | -8.50                        | -0.418 [-0.466, -0.364] | medium     | -0.71  | -11.16 [-12.47, -9.82]         | 2.7e-51  | 6.2e-51                  | 1.5e-50                     | 0.001                 | 0.0013                   | True                | reference  |
| set_B               | cocopye      | 600/588      | 3.989   | 16.73    | -11.17                       | -0.376 [-0.427, -0.328] | medium     | -0.721 | -12.74 [-14.14, -11.37]        | 7e-53    | 4.9e-52                  | 1.3e-51                     | 0.001                 | 0.0013                   | True                | reference  |
| set_B               | deepcheck    | 600/588      | 3.989   | 21.56    | -15.86                       | -0.561 [-0.609, -0.512] | large      | -0.885 | -17.57 [-19.04, -16.05]        | 8.8e-79  | 5.7e-78                  | 2e-77                       | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_A | checkm2      | 1000/806     | 0.786   | 0.276    | +0.14                        | +0.280 [+0.231, +0.325] | small      | 0.403  | +0.51 [+0.33, +0.74]           | 2.8e-28  | 7.5e-22                  | 1.2e-21                     | 0.001                 | 0.0013                   | True                | comparison |
| motivating_v2_set_A | cocopye      | 1000/806     | 0.786   | 0.727    | +0.18                        | +0.572 [+0.522, +0.620] | large      | 0.403  | +0.06 [-0.22, +0.35]           | 2.4e-28  | 2.9e-21                  | 4.4e-21                     | 0.68                  | 0.71                     | False               | comparison |
| motivating_v2_set_A | deepcheck    | 1000/806     | 0.786   | 0.429    | +0.05                        | +0.088 [+0.037, +0.139] | negligible | 0.129  | +0.36 [+0.17, +0.59]           | 0.00039  | 0.012                    | 0.013                       | 0.001                 | 0.0013                   | True                | comparison |
| motivating_v2_set_B | checkm2      | 1000/798     | 4.361   | 17.08    | -10.96                       | -0.422 [-0.462, -0.380] | medium     | -0.75  | -12.71 [-13.78, -11.69]        | 9.2e-94  | 2.2e-86                  | 1.2e-85                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_B | cocopye      | 1000/798     | 4.361   | 19.14    | -13.66                       | -0.400 [-0.443, -0.357] | medium     | -0.768 | -14.78 [-15.83, -13.72]        | 3.2e-98  | 3.3e-91                  | 2.3e-90                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_B | deepcheck    | 1000/798     | 4.361   | 25.7     | -19.86                       | -0.582 [-0.620, -0.544] | large      | -0.904 | -21.34 [-22.65, -20.03]        | 3.1e-135 | 7.4e-115                 | 1.6e-113                    | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_C | checkm2      | 1000/773     | 5.482   | 18.13    | -10.81                       | -0.360 [-0.399, -0.322] | medium     | -0.759 | -12.65 [-13.75, -11.55]        | 4.9e-96  | 3.4e-81                  | 1.3e-80                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_C | cocopye      | 1000/773     | 5.482   | 20.32    | -13.48                       | -0.342 [-0.383, -0.300] | medium     | -0.767 | -14.84 [-15.99, -13.72]        | 4e-98    | 6.5e-88                  | 3.9e-87                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_v2_set_C | deepcheck    | 1000/773     | 5.482   | 24.37    | -16.95                       | -0.454 [-0.492, -0.416] | medium     | -0.848 | -18.89 [-20.37, -17.42]        | 2.4e-119 | 1.5e-101                 | 1.6e-100                    | 0.001                 | 0.0013                   | True                | reference  |
| motivating_set_A    | checkm2      | 600/585      | 0.76    | 0.282    | +0.14                        | +0.302 [+0.247, +0.353] | small      | 0.403  | +0.48 [+0.33, +0.65]           | 1.2e-17  | 3.6e-17                  | 5.4e-17                     | 0.001                 | 0.0013                   | True                | comparison |
| motivating_set_A    | cocopye      | 600/585      | 0.76    | 0.938    | +0.14                        | +0.437 [+0.373, +0.502] | medium     | 0.251  | -0.18 [-0.44, +0.05]           | 1e-07    | 3.2e-08                  | 4.4e-08                     | 0.14                  | 0.16                     | False               | reference  |
| motivating_set_A    | deepcheck    | 600/585      | 0.76    | 0.379    | +0.06                        | +0.149 [+0.085, +0.209] | small      | 0.177  | +0.38 [+0.23, +0.56]           | 0.00017  | 0.00031                  | 0.00039                     | 0.001                 | 0.0013                   | True                | comparison |
| motivating_set_B    | checkm2      | 1100/1031    | 4.519   | 17.27    | -9.70                        | -0.460 [-0.496, -0.425] | medium     | -0.805 | -12.75 [-13.86, -11.56]        | 2e-118   | 5.1e-113                 | 7.1e-112                    | 0.001                 | 0.0013                   | True                | reference  |
| motivating_set_B    | cocopye      | 1100/1031    | 4.519   | 18.43    | -10.23                       | -0.363 [-0.398, -0.330] | medium     | -0.748 | -13.91 [-15.14, -12.75]        | 2.6e-102 | 1.8e-99                  | 1.5e-98                     | 0.001                 | 0.0013                   | True                | reference  |
| motivating_set_B    | deepcheck    | 1100/1031    | 4.519   | 24.06    | -16.83                       | -0.620 [-0.653, -0.587] | large      | -0.919 | -19.54 [-20.85, -18.28]        | 1.4e-153 | 5e-146                   | 2.1e-144                    | 0.001                 | 0.0013                   | True                | reference  |

## Sensitivity analysis: clusters defined by dominant phylum

The most conservative grouping the data support (clusters = phyla, so only a handful of independent units per set). Reported because Reviewer 1 asks for clustering by reference genome **or taxonomic group**.

### abs_err_completeness

| set         | MAGICC v5 vs | n phyla (clusters) | HL median diff (pp) [95% CI] | median diff [95% CI] | p cluster-boot (phylum) | p Wilcoxon phylum-mean |
|-------------|--------------|--------------------|------------------------------|----------------------|-------------------------|------------------------|
| set_A_v2    | magicc_v4    | 31                 | -0.19 [-0.33, -0.11]         | -0.17 [-0.29, -0.13] | 0.001                   | 0.053                  |
| set_A_v2    | checkm2      | 31                 | -0.50 [-0.70, -0.06]         | -0.05 [-0.27, +0.05] | 0.67                    | 0.71                   |
| set_A_v2    | cocopye      | 31                 | -1.25 [-2.01, -0.71]         | -0.52 [-1.17, -0.10] | 0.01                    | 0.069                  |
| set_A_v2    | deepcheck    | 31                 | -1.72 [-2.63, -0.98]         | -1.10 [-1.81, -0.77] | 0.001                   | 0.053                  |
| set_B_v2    | magicc_v4    | 30                 | -0.04 [-0.17, +0.03]         | +0.00 [-0.07, +0.04] | 0.83                    | 0.44                   |
| set_B_v2    | checkm2      | 30                 | +1.15 [+0.68, +2.15]         | +0.50 [+0.34, +0.94] | 0.001                   | 8.3e-07                |
| set_B_v2    | cocopye      | 30                 | +0.20 [-0.00, +0.39]         | +0.16 [+0.11, +0.23] | 0.001                   | 0.5                    |
| set_B_v2    | deepcheck    | 30                 | -2.30 [-4.26, -0.52]         | -0.99 [-2.17, -0.57] | 0.002                   | 0.61                   |
| set_D       | magicc_v4    | 11                 | -0.05 [-0.09, +0.02]         | -0.03 [-0.06, +0.03] | 0.4                     | 0.52                   |
| set_D       | checkm2      | 11                 | -4.00 [-5.82, -2.09]         | -2.54 [-4.47, -1.38] | 0.004                   | 0.24                   |
| set_D       | cocopye      | 11                 | -0.88 [-2.40, -0.22]         | -0.61 [-1.41, -0.30] | 0.002                   | 0.014                  |
| set_D       | deepcheck    | 11                 | -4.13 [-6.03, -2.62]         | -3.59 [-4.96, -2.03] | 0.001                   | 0.019                  |
| set_E       | magicc_v4    | 31                 | -0.16 [-0.25, +0.00]         | -0.14 [-0.19, -0.01] | 0.034                   | 0.66                   |
| set_E       | checkm2      | 31                 | -2.81 [-4.13, -0.50]         | -1.02 [-2.00, +0.10] | 0.16                    | 0.61                   |
| set_E       | cocopye      | 31                 | +0.13 [-0.31, +0.60]         | +0.09 [-0.01, +0.39] | 0.062                   | 0.14                   |
| set_E       | deepcheck    | 31                 | -5.36 [-7.02, -2.89]         | -3.82 [-5.30, -1.39] | 0.001                   | 0.21                   |
| set_D_clean | checkm2      | 6                  | -3.28 [-5.37, -1.75]         | -2.27 [-4.16, -0.94] | 0.001                   | 0.031                  |
| set_D_clean | cocopye      | 6                  | -0.35 [-2.21, +0.14]         | -0.32 [-1.62, +0.00] | 0.081                   | 0.84                   |
| set_D_clean | deepcheck    | 6                  | -2.70 [-4.82, -1.21]         | -2.10 [-3.60, -1.08] | 0.001                   | 0.031                  |

### abs_err_contamination

| set         | MAGICC v5 vs | n phyla (clusters) | HL median diff (pp) [95% CI] | median diff [95% CI]    | p cluster-boot (phylum) | p Wilcoxon phylum-mean |
|-------------|--------------|--------------------|------------------------------|-------------------------|-------------------------|------------------------|
| set_A_v2    | magicc_v4    | 31                 | -0.20 [-0.42, -0.14]         | -0.08 [-0.27, -0.05]    | 0.001                   | 5.3e-06                |
| set_A_v2    | checkm2      | 31                 | +0.15 [+0.09, +0.40]         | +0.11 [+0.08, +0.34]    | 0.001                   | 3.5e-07                |
| set_A_v2    | cocopye      | 31                 | +0.19 [+0.13, +0.37]         | +0.17 [+0.13, +0.32]    | 0.001                   | 0.2                    |
| set_A_v2    | deepcheck    | 31                 | +0.07 [+0.01, +0.34]         | +0.05 [+0.01, +0.26]    | 0.004                   | 1e-05                  |
| set_B_v2    | magicc_v4    | 30                 | -0.45 [-0.71, -0.29]         | -0.40 [-0.56, -0.27]    | 0.001                   | 0.0093                 |
| set_B_v2    | checkm2      | 30                 | -11.21 [-17.85, -7.55]       | -7.24 [-16.34, -4.38]   | 0.001                   | 1.4e-05                |
| set_B_v2    | cocopye      | 30                 | -13.77 [-14.51, -11.73]      | -8.16 [-8.75, -6.22]    | 0.001                   | 2e-07                  |
| set_B_v2    | deepcheck    | 30                 | -19.59 [-24.16, -16.94]      | -16.42 [-23.40, -13.55] | 0.001                   | 1e-07                  |
| set_D       | magicc_v4    | 11                 | +0.20 [+0.14, +0.25]         | +0.17 [+0.10, +0.24]    | 0.005                   | 0.58                   |
| set_D       | checkm2      | 11                 | -28.63 [-32.24, -25.60]      | -27.37 [-33.46, -24.28] | 0.001                   | 0.00098                |
| set_D       | cocopye      | 11                 | -17.08 [-18.91, -16.64]      | -14.38 [-16.35, -13.63] | 0.001                   | 0.0029                 |
| set_D       | deepcheck    | 11                 | -33.41 [-36.21, -31.37]      | -33.20 [-36.96, -29.82] | 0.001                   | 0.00098                |
| set_E       | magicc_v4    | 31                 | -0.06 [-0.33, +0.02]         | -0.02 [-0.24, +0.00]    | 0.077                   | 0.026                  |
| set_E       | checkm2      | 31                 | -10.78 [-15.47, -8.51]       | -6.62 [-9.98, -4.88]    | 0.001                   | 6e-07                  |
| set_E       | cocopye      | 31                 | -15.08 [-15.87, -13.84]      | -6.83 [-7.51, -4.75]    | 0.001                   | 9.3e-09                |
| set_E       | deepcheck    | 31                 | -17.58 [-19.04, -16.11]      | -13.02 [-14.42, -9.66]  | 0.001                   | 2.4e-07                |
| set_D_clean | checkm2      | 6                  | -28.01 [-30.65, -25.09]      | -28.90 [-32.06, -25.52] | 0.001                   | 0.031                  |
| set_D_clean | cocopye      | 6                  | -15.40 [-18.11, -14.45]      | -13.40 [-15.69, -12.52] | 0.001                   | 0.031                  |
| set_D_clean | deepcheck    | 6                  | -32.37 [-34.27, -30.27]      | -33.38 [-36.59, -30.64] | 0.001                   | 0.031                  |

