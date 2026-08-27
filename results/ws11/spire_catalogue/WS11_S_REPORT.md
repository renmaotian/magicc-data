# WS11.S — catalogue-scale SPIRE run: how often does the MIMAG-inspired class change?

**This is a disagreement analysis, not an error measurement.** Catalogue MAGs carry no ground truth. Everything below compares MAGICC V5 against SPIRE v1's *published* CheckM2 values; where the two disagree, nothing here says which is right. The ground-truthed anchor remains `set_C_clean`, where **MAGICC is the tool in error** (−8.68 pp completeness and +5.09 pp contamination against truth, versus CheckM2's −0.70 and −1.16), and that is the interpretation that governs the direction of every disagreement reported here.

Frozen `models/magicc_v5.onnx`, SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`, verified before every inference run (scripts 231 and 232).

## 1. Cohorts and denominators

Frame: the **1,158,468** SPIRE v1 MAGs carrying a published CheckM2 completeness, contamination and genome size (85 of the 1,158,553 metadata rows have none and are out of frame). 183 phyla, 3,449 families, 107,008 95 %-ANI clusters.

| Cohort | What it is | n analysed | Denominator meaning |
|---|---|---|---|
| `catalogue_weighted` | stratified probability sample of the catalogue, design-weighted and post-stratified for non-response | 18,647 | the whole catalogue |
| `catalogue_weighted_no_nr_adjustment` | same, design weight only | 18,647 | sensitivity check |
| `S3_srs_unweighted` | the 200,000-genome simple random sample alone, unweighted | 892 | stratum S3 (93.6 % of the catalogue) |
| `S1_small_census` | every frame MAG < 1 Mbp | 17,201 | the sub-megabase catalogue |
| `S2_rare_phylum_census` | every frame MAG in a phylum with < 1,000 catalogue MAGs | 554 | the rare-phylum catalogue |
| `reps_census` | every SPIRE 95 %-ANI MAG cluster representative | 6,566 | the dereplicated catalogue — a DIFFERENT population |

Design (script 230, seed 2248018003 = CRC-32("WS11.S/spire_catalogue/SRS"), numpy PCG64): **S1_small** N=59,297 n=59,297 f=1.0000 w=1.00000; **S2_rare_phylum** N=14,586 n=14,586 f=1.0000 w=1.00000; **S3_main** N=1,084,585 n=200,000 f=0.1844 w=5.42293.

The sampled cohort is **273,883 genomes = 23.64 % of the catalogue**; 18,647 of them were retrieved and scored (**1.61 % of the catalogue**). Cluster bootstrap: 2,000 iterations, primary unit **family** (1,381 families in the analysed cohort), secondary unit **spire_cluster** (7,301 95 %-ANI clusters).

## 2. Headline — proportion of catalogue MAGs whose MIMAG-inspired class changes

**32.82 % [29.11, 36.33]** of SPIRE v1 catalogue MAGs change MIMAG-inspired class when MAGICC V5 is substituted for the published CheckM2 values.

- denominator: all 1,158,468 catalogue MAGs, estimated from 18,647 analysed genomes;
- cohort: `catalogue_weighted`; clustering unit: **family**; 2,000-iteration cluster bootstrap. This is the conservative, model-based interval and uses the same convention as WS3.10.

**Design-based interval for the same quantity: 32.82 % [29.86, 35.78].** The catalogue proportion is a finite-population descriptive parameter: its only sampling uncertainty comes from which S3 genomes were drawn (the S1 and S2 strata are censuses and contribute exactly zero variance), and because S3 is a simple random sample of *genomes*, family clustering does not enter the design variance. Quote the design interval for statements about **this** catalogue and the family-clustered interval for statements about MAGICC's behaviour in general.

Clustering on 95 %-ANI species clusters instead of families gives 32.82 % [29.54, 36.26] — the point estimate is identical by construction; only the interval changes.

| Direction | Rate [95 % CI] | Denominator |
|---|---|---|
| any class change | 32.82 % [29.11, 36.33] (design 32.82 % [29.86, 35.78]) | all MAGs |
| downgrade (class falls) | 23.59 % [20.35, 26.83] | all MAGs |
| upgrade (class rises) | 9.23 % [7.42, 11.26] | all MAGs |
| HQ → not HQ | 53.96 % [46.97, 60.65] | MAGs SPIRE publishes as HQ (n = 3,582 analysed) |
| HQ → MQ | 43.57 % [36.66, 50.62] | published HQ |
| HQ → LQ | 10.40 % [6.78, 14.33] | published HQ |
| MQ → HQ | 13.48 % [10.95, 16.28] | published MQ |
| MQ → LQ | 9.74 % [7.16, 12.71] | published MQ |

Overall high-quality fraction: SPIRE's published CheckM2 calls 31.35 % [27.76, 34.85] of the catalogue HQ; MAGICC calls 23.66 % [20.62, 26.77].

Design check — the unweighted 200,000-genome SRS alone (stratum S3, 93.6 % of the catalogue) gives 37.00 % [33.84, 40.30] for any class change; the weighted catalogue estimate above is the one to quote.

### 2.1 Confusion matrix (published CheckM2 class × MAGICC class)

Weighted proportion of the catalogue in each cell, `catalogue_weighted`, family-clustered bootstrap. Row-conditional rates are in the table above.

| published \ MAGICC | HQ | MQ | LQ | row n analysed |
|---|---|---|---|---|
| **HQ** | 14.430 % [11.738, 17.343] | 13.657 % [11.013, 16.414] | 3.259 % [2.116, 4.544] | 3,582 |
| **MQ** | 9.232 % [7.415, 11.256] | 52.603 % [48.967, 56.536] | 6.671 % [4.873, 8.746] | 15,063 |
| **LQ** | 0.000 % [0.000, 0.000] | 0.000 % [0.000, 0.001] | 0.149 % [0.000, 0.465] | 2 |

MAGICC cannot express completeness below 50, so it can only reach LQ through contamination ≥ 10 — read the LQ column with that in mind.

## 3. Stratified rates

### 3.1 By genome size (`catalogue_weighted`)

| size bin | n analysed | any class change | HQ → not HQ | published HQ % | MAGICC HQ % |
|---|---|---|---|---|---|
| <1Mb | 17,201 | 18.37 % [15.56, 21.68] | 69.57 % [64.59, 75.10] | 16.04 % [13.18, 19.15] | 6.50 % [4.79, 8.41] |
| 1-2Mb | 428 | 24.10 % [18.26, 30.22] | 41.54 % [27.05, 58.17] | 18.09 % [13.08, 23.60] | 17.62 % [12.58, 22.68] |
| 2-3Mb | 418 | 38.37 % [32.16, 44.67] | 65.04 % [54.47, 75.67] | 40.14 % [32.86, 46.85] | 22.43 % [17.19, 27.51] |
| 3-5Mb | 464 | 44.73 % [38.82, 50.46] | 45.83 % [37.94, 54.00] | 49.26 % [43.43, 54.81] | 44.36 % [38.24, 50.38] |
| >5Mb | 136 | 56.48 % [46.03, 67.53] | 33.73 % [21.24, 47.78] | 48.09 % [37.54, 58.82] | 56.03 % [44.23, 67.09] |

The `<1Mb` row is a **census** of every sub-megabase MAG in the catalogue (17,201 analysed), not a sample.

### 3.2 By reduction relative to the lineage — log2(size / phylum median)

| stratum | n analysed | any class change | HQ → not HQ | mean Δcompleteness |
|---|---|---|---|---|
| <-1 (>=2x reduced) | 6,405 | 2.52 % [1.23, 4.47] | 55.56 % [44.61, 95.51] | -2.24 pp [-5.96, +1.40] |
| -1 to -0.5 | 4,082 | 13.69 % [6.94, 21.03] | 54.20 % [25.19, 81.52] | -4.65 pp [-6.65, -2.50] |
| -0.5 to -0.25 | 1,766 | 22.36 % [13.79, 31.83] | 53.59 % [29.81, 76.71] | -2.92 pp [-5.22, -0.72] |
| -0.25 to +0.25 (typical) | 4,150 | 32.10 % [26.01, 38.25] | 62.33 % [49.78, 74.09] | -3.60 pp [-5.32, -1.95] |
| >+0.25 (larger than lineage) | 2,244 | 50.46 % [43.82, 57.06] | 47.53 % [38.97, 55.71] | +1.04 pp [-0.42, +2.57] |

### 3.3 By phylum

Phyla with ≥ 200 analysed genomes, ordered by class-change rate; `catalogue_weighted`, family-clustered.

| phylum | n analysed | any class change | HQ → not HQ | mean Δcompleteness |
|---|---|---|---|---|
| Nanoarchaeota | 670 | 82.51 % [37.46, 93.36] | 97.00 % [90.77, 100.00] | +1.54 pp [-17.90, +22.92] |
| Thermoplasmatota | 282 | 51.65 % [0.22, 91.04] | 100.00 % [100.00, 100.00] | +1.54 pp [-12.29, +3.83] |
| Patescibacteria | 7,343 | 48.54 % [35.32, 60.05] | 47.48 % [31.17, 65.94] | -2.23 pp [-6.32, +2.07] |
| Thermoproteota | 361 | 46.25 % [24.77, 65.41] | 66.31 % [24.11, 100.00] | -1.20 pp [-7.99, +7.77] |
| other (<200 in cohort) | 2,251 | 36.37 % [30.33, 42.91] | 56.26 % [45.54, 68.13] | -1.26 pp [-2.94, +0.38] |
| Bacteroidota | 363 | 31.30 % [22.81, 39.41] | 57.28 % [40.82, 72.85] | -4.66 pp [-6.94, -2.40] |
| Firmicutes | 1,884 | 28.86 % [11.20, 54.77] | 75.26 % [28.93, 99.68] | -0.88 pp [-7.10, +2.12] |
| Proteobacteria | 1,621 | 27.13 % [20.00, 35.04] | 55.20 % [41.03, 68.90] | -3.56 pp [-5.64, -1.48] |
| Actinobacteriota | 973 | 24.92 % [13.66, 38.16] | 44.09 % [16.52, 73.41] | -1.97 pp [-4.57, +0.82] |
| Firmicutes_A | 2,487 | 24.04 % [14.15, 38.00] | 41.98 % [11.77, 100.00] | -0.39 pp [-4.57, +2.59] |
| Firmicutes_C | 412 | 9.32 % [0.00, 37.03] | 0.00 % [0.00, 0.00] | +1.68 pp [-3.05, +6.82] |

## 4. Delta distributions and the size dose–response

MAGICC − CheckM2, `catalogue_weighted`, weighted medians and means with family-clustered bootstrap CIs.

| stratum | metric | weighted median [95 % CI] | weighted mean [95 % CI] |
|---|---|---|---|
| ALL | completeness | -0.89 [-1.99, -0.21] | -2.12 [-3.05, -1.19] |
| ALL | contamination | -0.66 [-0.82, -0.42] | +0.98 [+0.30, +1.78] |
| <1Mb | completeness | -6.31 [-7.55, -5.11] | -8.21 [-9.21, -7.28] |
| <1Mb | contamination | -0.03 [-0.07, +0.01] | +1.18 [+0.77, +1.67] |
| 1-2Mb | completeness | -1.22 [-2.50, -0.08] | -1.69 [-3.46, -0.12] |
| 1-2Mb | contamination | -0.55 [-0.77, -0.27] | +1.59 [+0.42, +3.00] |
| 2-3Mb | completeness | -1.51 [-5.01, -0.01] | -3.35 [-4.92, -1.70] |
| 2-3Mb | contamination | -0.82 [-1.34, -0.39] | +0.28 [-0.67, +1.29] |
| 3-5Mb | completeness | +0.71 [-0.00, +1.84] | +1.00 [-0.32, +2.38] |
| 3-5Mb | contamination | -1.12 [-1.64, -0.76] | +0.14 [-0.75, +1.08] |
| >5Mb | completeness | +2.08 [+0.15, +4.63] | +2.97 [+0.97, +5.01] |
| >5Mb | contamination | -0.49 [-1.66, +0.33] | +6.19 [+2.17, +10.95] |

**Size dose–response** (weighted OLS of Δ on log10 assembly Mbp, family-clustered):

| cohort | subset | n | metric | slope pp per log10 Mbp [95 % CI] |
|---|---|---|---|---|
| catalogue_weighted | overall:ALL | 18,647 | completeness | +8.38 [+5.06, +11.76] |
| catalogue_weighted | overall:ALL | 18,647 | contamination | -0.88 [-3.75, +1.98] |
| catalogue_weighted_no_nr_adjustment | overall:ALL | 18,647 | completeness | +10.23 [+7.50, +12.85] |
| catalogue_weighted_no_nr_adjustment | overall:ALL | 18,647 | contamination | +2.43 [-1.27, +6.27] |
| S3_srs_unweighted | overall:ALL | 892 | completeness | +8.71 [+5.08, +12.67] |
| S3_srs_unweighted | overall:ALL | 892 | contamination | +3.21 [-1.50, +8.39] |
| S1_small_census | overall:ALL | 17,201 | completeness | +21.29 [+16.12, +26.14] |
| S1_small_census | overall:ALL | 17,201 | contamination | +2.30 [+0.43, +4.45] |
| S2_rare_phylum_census | overall:ALL | 554 | completeness | +22.40 [+15.80, +30.01] |
| S2_rare_phylum_census | overall:ALL | 554 | contamination | +22.57 [+7.61, +33.93] |
| reps_census | overall:ALL | 6,566 | completeness | +14.34 [+12.98, +15.66] |
| reps_census | overall:ALL | 6,566 | contamination | +0.70 [-1.14, +2.72] |

## 5. Floor censoring

MAGICC cannot express completeness below 50, so a prediction sitting at the floor makes its completeness delta a **lower bound** on the disagreement.

| cohort | stratum | n | n ≤ 50.5 | n ≤ 50.0 | rate [95 % CI] | min MAGICC completeness |
|---|---|---|---|---|---|---|
| catalogue_weighted | ALL | 18,647 | 1368 | 0 | 0.586 % [0.288, 1.050] | 50.00 |
| catalogue_weighted | <1Mb | 17,201 | 1366 | 0 | 8.100 % [5.506, 11.712] | 50.00 |
| catalogue_weighted | 1-2Mb | 428 | 2 | 0 | 0.416 % [0.000, 1.477] | 50.31 |
| catalogue_weighted | 2-3Mb | 418 | 0 | 0 | 0.000 % [0.000, 0.000] | 52.30 |
| catalogue_weighted | 3-5Mb | 464 | 0 | 0 | 0.000 % [0.000, 0.000] | 53.18 |
| catalogue_weighted | >5Mb | 136 | 0 | 0 | 0.000 % [0.000, 0.000] | 71.02 |
| S1_small_census | ALL | 17,201 | 1366 | 0 | 7.941 % [5.360, 11.384] | 50.00 |
| S1_small_census | <1Mb | 17,201 | 1366 | 0 | 7.941 % [5.468, 11.349] | 50.00 |
| reps_census | ALL | 6,566 | 60 | 0 | 0.914 % [0.626, 1.247] | 50.00 |
| reps_census | <1Mb | 1,200 | 55 | 0 | 4.583 % [2.975, 6.259] | 50.00 |
| reps_census | 1-2Mb | 1,485 | 5 | 0 | 0.337 % [0.069, 0.666] | 50.31 |
| reps_census | 2-3Mb | 1,509 | 0 | 0 | 0.000 % [0.000, 0.000] | 50.70 |
| reps_census | 3-5Mb | 1,797 | 0 | 0 | 0.000 % [0.000, 0.000] | 52.97 |
| reps_census | >5Mb | 575 | 0 | 0 | 0.000 % [0.000, 0.000] | 63.89 |

## 6. Non-response

17,201 of 19,128 sampled genomes were retrieved (**89.93 %**). Non-response is a persistent server-side HTTP 404 on `spire.embl.de/download_file/<id>`, verified by slow single retries, and it is clustered by originating study. Every frame variable is known for the non-respondents, so the analysis post-stratifies on (stratum × size bin × published class); the `catalogue_weighted_no_nr_adjustment` cohort is the unadjusted sensitivity check.

| grouping | level | n sampled | n retrieved | response rate |
|---|---|---|---|---|
| overall | ALL | 19,128 | 17,201 | 89.93 % |
| stratum | S1_small | 19,128 | 17,201 | 89.93 % |
| size_bin | <1Mb | 19,128 | 17,201 | 89.93 % |
| checkm2_class | HQ | 3,185 | 3,043 | 95.54 % |
| checkm2_class | LQ | 1 | 1 | 100.00 % |
| checkm2_class | MQ | 15,942 | 14,157 | 88.80 % |
| recovery | HTTP404_recovered_from_representatives_tar | 970 | 171 | 17.63 % |

Non-response adjustment moves the headline from 36.26 % [33.04, 39.34] to 32.82 % [29.11, 36.33].

## 7. Catalogue scale versus the 750-genome cohort

The cohort the manuscript currently cites is **750 GTDB r220 MAGs drawn 150 per genome-size bin** (WS3.10, script 131, seed 13100). That design over-samples `<1Mb` by about 6× and `>5Mb` by about 13× relative to the catalogue, so its pooled numbers are not catalogue rates. Two things separate it from the catalogue estimate — the **design** and the **catalogue** (GTDB r220 is not SPIRE v1). The design effect is isolated below by redrawing the 150-per-size-bin design 500 times from the SPIRE data measured here; the catalogue difference cannot be isolated without rescoring GTDB and is stated, not estimated.

| quantity | cohort | n | estimate [95 % CI] |
|---|---|---|---|
| size_slope_completeness_pp_per_log10Mbp | WS3.10 750-MAG GTDB size-stratified | 750 | +15.31 [+12.3, +18.3] |
| size_slope_completeness_pp_per_log10Mbp | SPIRE catalogue, 150-per-size-bin design re-drawn (n=750 x 500 replicates) | 750 | +21.22 [+7.81, +37] |
| size_slope_completeness_pp_per_log10Mbp | SPIRE catalogue_weighted | 17147 | +13.83 [+12.1, +15.5] |
| median_delta_completeness[<1Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | -12.6 |
| median_delta_completeness[<1Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | -6.307 [-8.58, -4.6] |
| mean_delta_completeness[<1Mb] | SPIRE catalogue_weighted | 15701 | -8.047 [-9.06, -7.1] |
| HQ_downgrade_rate[<1Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | +0.2733 |
| HQ_downgrade_rate[<1Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +0.6944 [+0.524, +0.884] |
| HQ_downgrade_rate[<1Mb] | SPIRE catalogue_weighted | 15701 | +0.6789 [+0.623, +0.738] |
| median_delta_completeness[1-2Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | -2.42 |
| median_delta_completeness[1-2Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +nan |
| mean_delta_completeness[1-2Mb] | SPIRE catalogue_weighted | 428 | -2.306 [-3.88, -0.954] |
| HQ_downgrade_rate[1-2Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | +0.1933 |
| HQ_downgrade_rate[1-2Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +nan |
| HQ_downgrade_rate[1-2Mb] | SPIRE catalogue_weighted | 428 | +0.4612 [+0.33, +0.602] |
| median_delta_completeness[2-3Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | +0.28 |
| median_delta_completeness[2-3Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +nan |
| mean_delta_completeness[2-3Mb] | SPIRE catalogue_weighted | 418 | -2.361 [-3.77, -0.984] |
| HQ_downgrade_rate[2-3Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | +0.1067 |
| HQ_downgrade_rate[2-3Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +nan |
| HQ_downgrade_rate[2-3Mb] | SPIRE catalogue_weighted | 418 | +0.6281 [+0.531, +0.728] |
| median_delta_completeness[3-5Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | +0.65 |
| median_delta_completeness[3-5Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +nan |
| mean_delta_completeness[3-5Mb] | SPIRE catalogue_weighted | 464 | +1.349 [+0.154, +2.6] |
| HQ_downgrade_rate[3-5Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | +0.2067 |
| HQ_downgrade_rate[3-5Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +nan |
| HQ_downgrade_rate[3-5Mb] | SPIRE catalogue_weighted | 464 | +0.4606 [+0.381, +0.538] |
| median_delta_completeness[>5Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | +0.64 |
| median_delta_completeness[>5Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +nan |
| mean_delta_completeness[>5Mb] | SPIRE catalogue_weighted | 136 | +3.029 [+0.998, +5.09] |
| HQ_downgrade_rate[>5Mb] | WS3.10 750-MAG GTDB size-stratified | 150 | +0.24 |
| HQ_downgrade_rate[>5Mb] | SPIRE catalogue, 150-per-bin design re-drawn | 150 | +nan |
| HQ_downgrade_rate[>5Mb] | SPIRE catalogue_weighted | 136 | +0.3476 [+0.222, +0.488] |
| p_class_change | SPIRE catalogue_weighted | 17147 | +0.2298 [+0.203, +0.258] |
| p_class_change | SPIRE catalogue_weighted_no_nr_adjustment | 17147 | +0.236 [+0.209, +0.267] |
| p_class_change | SPIRE S3_srs_unweighted | 892 | +0.37 [+0.338, +0.403] |
| p_class_change | SPIRE S1_small_census | 15701 | +0.1855 [+0.157, +0.218] |
| p_class_change | SPIRE reps_census | 6566 | +0.3718 [+0.356, +0.388] |
| HQ_downgrade_rate | SPIRE catalogue_weighted | 17147 | +0.6006 [+0.563, +0.642] |
| HQ_downgrade_rate | SPIRE catalogue_weighted_no_nr_adjustment | 17147 | +0.5975 [+0.559, +0.637] |
| HQ_downgrade_rate | SPIRE S3_srs_unweighted | 892 | +0.4735 [+0.417, +0.532] |
| HQ_downgrade_rate | SPIRE S1_small_census | 15701 | +0.6789 [+0.623, +0.739] |
| HQ_downgrade_rate | SPIRE reps_census | 6566 | +0.5008 [+0.476, +0.528] |
| MQ_to_HQ_rate | SPIRE catalogue_weighted | 17147 | +0.0559 [+0.0453, +0.0675] |
| MQ_to_HQ_rate | SPIRE catalogue_weighted_no_nr_adjustment | 17147 | +0.0591 [+0.0479, +0.0723] |
| MQ_to_HQ_rate | SPIRE S3_srs_unweighted | 892 | +0.2053 [+0.173, +0.241] |
| MQ_to_HQ_rate | SPIRE S1_small_census | 15701 | +0.01986 [+0.0137, +0.0281] |
| MQ_to_HQ_rate | SPIRE reps_census | 6566 | +0.1809 [+0.166, +0.197] |
| MQ_to_LQ_rate | SPIRE catalogue_weighted | 17147 | +0.07334 [+0.0587, +0.0899] |
| MQ_to_LQ_rate | SPIRE catalogue_weighted_no_nr_adjustment | 17147 | +0.07428 [+0.0595, +0.0918] |
| MQ_to_LQ_rate | SPIRE S3_srs_unweighted | 892 | +0.107 [+0.0785, +0.136] |
| MQ_to_LQ_rate | SPIRE S1_small_census | 15701 | +0.0628 [+0.0471, +0.0828] |
| MQ_to_LQ_rate | SPIRE reps_census | 6566 | +0.1148 [+0.0974, +0.132] |
| floor_rate | SPIRE catalogue_weighted | 17147 | +0.06109 [+0.0418, +0.09] |
| floor_rate | SPIRE catalogue_weighted_no_nr_adjustment | 17147 | +0.05924 [+0.0403, +0.0861] |
| floor_rate | SPIRE S3_srs_unweighted | 892 | +0.001121 [+0, +0.00376] |
| floor_rate | SPIRE S1_small_census | 15701 | +0.07917 [+0.054, +0.115] |
| floor_rate | SPIRE reps_census | 6566 | +0.009138 [+0.00626, +0.0125] |
| mean_d_comp | SPIRE catalogue_weighted | 17147 | -6.271 [-7.05, -5.48] |
| mean_d_comp | SPIRE catalogue_weighted_no_nr_adjustment | 17147 | -6.158 [-6.95, -5.45] |
| mean_d_comp | SPIRE S3_srs_unweighted | 892 | -0.5846 [-1.42, +0.249] |
| mean_d_comp | SPIRE S1_small_census | 15701 | -8.082 [-9.12, -7.12] |
| mean_d_comp | SPIRE reps_census | 6566 | -2.245 [-2.8, -1.71] |
| mean_d_cont | SPIRE catalogue_weighted | 17147 | +1.138 [+0.745, +1.6] |
| mean_d_cont | SPIRE catalogue_weighted_no_nr_adjustment | 17147 | +1.154 [+0.759, +1.6] |
| mean_d_cont | SPIRE S3_srs_unweighted | 892 | +1.004 [+0.238, +1.86] |
| mean_d_cont | SPIRE S1_small_census | 15701 | +1.057 [+0.667, +1.54] |
| mean_d_cont | SPIRE reps_census | 6566 | +1.581 [+1.02, +2.19] |
| floor_rate | WS3.10 SPIRE five reviewer genera | 591 | +0.03215 |

## 8. Files

- `WS11_S_REPORT.md` (20,726 B)
- `checksums.tsv` (1,686,005 B)
- `classification_change_by_stratum.tsv` (984,771 B)
- `classification_change_matrix.tsv` (9,518 B)
- `cohort_definition.tsv.gz` (10,608,635 B)
- `delta_distributions.tsv` (28,740 B)
- `design_effect.json` (206 B)
- `feasibility_and_design.md` (6,030 B)
- `fetch_status.tsv` (591,841 B)
- `floor_censoring.tsv` (13,594 B)
- `magicc_predictions.tsv` (631,438 B)
- `nonresponse.tsv` (2,573 B)
- `reps_checkpoint.json` (77 B)
- `reps_checksums.tsv` (599,514 B)
- `reps_predictions.tsv` (224,071 B)
- `sampling_frame.json` (4,320 B)
- `size_dose_response.tsv` (17,949 B)
- `small_vs_large_cohort_comparison.tsv` (6,925 B)
- `ws11s_headline.json` (12,745 B)

## 9. Reproduction

```
python3 scripts/230_ws11s_sampling_frame.py
python3 scripts/231_ws11s_fetch_and_score.py --workers 14 --feat 9
python3 scripts/232_ws11s_representatives.py --feat 6
PYTHONHASHSEED=0 python3 scripts/233_ws11s_analysis.py
PYTHONHASHSEED=0 python3 scripts/234_ws11s_compare_and_report.py
PYTHONHASHSEED=0 python3 scripts/235_ws11s_report.py
```

