# WS3 Track A — real data with genuine ground truth

_Generated 2026-07-26T23:11:42.540534+00:00_

Protocol §4.4e withdrew the 141-MAG real-MAG contamination claims, so these three cohorts are the only real-data validation routes with genuine ground truth. Reviewer 2 named mock communities and isolate draft/complete pairs specifically (R2-M3).

**Conventions.** R² is the coefficient of determination (1 − SS_res/SS_tot) everywhere; squared Pearson is emitted separately as `*_r2_pearson` and is never called R². MIMAG-inspired thresholds use completeness/contamination only (HQ ≥90 % and <5 %; MQ ≥50 % and <10 %); the rRNA/tRNA criteria of the full standard are not evaluable here. All CIs are 95 % percentile cluster bootstraps (2,000 replicates) resampling reference organisms / species, because the same reference recurs across assemblies. Tests are two-sided Wilcoxon signed-rank on paired absolute errors.

**Denominator (identical for the ground truth and for every tool):** completeness = retained dominant-organism bp / dominant reference FULL length × 100; contamination = contaminant bp / dominant reference FULL length × 100. This is MAGICC's own convention, so no denominator re-definition is needed (R1-M4, R1-M5).

## A1 — Meslier et al. 2022 MOCK1 (headline: fragmentation gradient on real data)

Source: Meslier V. *et al.* *Benchmarking second and third-generation sequencing platforms for microbial metagenomics.* Sci Data 9, 694 (2022), doi:10.1038/s41597-022-01762-z. Reference genomes and the seven pre-computed MOCK1 assemblies come from the authors' public GitLab (`https://forge.inrae.fr/metagenopolis/benchmark_mock`); **no assembly compute was performed and no reads were downloaded**. 22 of the 91 reference strains carry ATCC designations with public GenBank accessions (ATCC MSA-1002 is a component of the mocks), which is how this work covers Reviewer 2's request for ATCC standards without the ATCC Genome Portal Data Use Agreement.

> **Erratum to report if the raw runs are cited.** Meslier Table 4 prints the Illumina runs as ERR9765446-ERR9765449. Those accessions do not exist; the correct Illumina HiSeq 3000 runs, verified against the ENA portal API, are **ERR9765746 (MOCK_001), ERR9765747 (MOCK_002), ERR9765748 + ERR9765749 (MOCK_003)**. Table 4's ONT/Ion/PacBio accessions do resolve.

91 reference genomes (68 bacterial / 23 archaeal, 29 GTDB phyla, 86/91 Complete). Split membership, GCA↔GCF cross-mapped: all 91 → {'train': 49, 'none': 27, 'val': 8, 'test': 7}; the 71 MOCK1 organisms → {'train': 39, 'none': 18, 'val': 8, 'test': 6}. Leakage-free (not in TRAIN or VAL): **34 of 91**, of which **24 are MOCK1 organisms**.

Alignment QC per assembly (minimap2 `-x asm10 --secondary=no`, contigs → pooled 91-reference index):

| assembly    | total_contigs | total_bp  | unassigned_bp_pct | n_bins_mock1 | bp_in_nonmock1_bins |
|-------------|---------------|-----------|-------------------|--------------|---------------------|
| pacbio      | 436           | 163865108 | 0.004             | 55           | 128893              |
| minion      | 1276          | 131421630 | 0.004             | 51           | 0                   |
| illumina    | 71367         | 159271694 | 0.08              | 70           | 5299                |
| s5          | 52315         | 135869544 | 0.027             | 66           | 21282               |
| proton      | 50680         | 133276450 | 0.026             | 66           | 14395               |
| mgiseq_2000 | 106741        | 136053186 | 0.04              | 68           | 7897                |
| mgiseq_t7   | 115274        | 134358760 | 0.045             | 69           | 1930                |

**394 reference-anchored bins** over 62 organisms × 7 assemblies. True completeness spans 0.9–100.0 % (median 94.5), driven by the mock's three-orders-of-magnitude abundance spread. True contamination is low by construction of reference-anchored binning (median 0.011 %, max 6.02 %), so the contamination axis here is a **real-data false-positive test**, not a quantification benchmark. Bin bp per covered reference bp: median 1.011, max 1.157 — the bins carry almost no redundant sequence.

### The seven assemblies differ along TWO axes, not one

Contig N50 (fragmentation) and per-base accuracy (platform error mode) are separate. Indel-dense assemblies frameshift ORFs, which a nucleotide k-mer method never sees and a protein-based method cannot survive. Coding density and mean gene length below come from **CheckM2's own output**, and the indel rates from base-level alignment of each assembly's largest contigs to the known references (`scripts/147`), so the diagnosis does not depend on this pipeline. Short mean gene length alone is not sufficient evidence — a highly fragmented short-read assembly also truncates genes at contig ends — hence the joint coding-density criterion.

| assembly    | n_bins | median_coding_density | median_gene_length | median_n_cds | median_bin_n50 | orf_integrity_compromised | indel_bp_per_kb | substitution_per_kb | mean_bp_between_indels |
|-------------|--------|-----------------------|--------------------|--------------|----------------|---------------------------|-----------------|---------------------|------------------------|
| pacbio      | 52     | 0.884                 | 304.5              | 2362         | 1.854e+06      | False                     | 0.4119          | 0.0055              | 2428                   |
| minion      | 51     | 0.754                 | 132.2              | 3728         | 3.137e+05      | True                      | 8.947           | 0.872               | 111.8                  |
| illumina    | 62     | 0.8815                | 265.7              | 2465         | 1.07e+04       | False                     | 0.6545          | 0.0788              | 1528                   |
| s5          | 56     | 0.8715                | 233.1              | 2650         | 6522           | False                     | 8.385           | 0.7805              | 119.3                  |
| proton      | 56     | 0.8625                | 207                | 2754         | 5244           | False                     | 9.428           | 0.629               | 106.1                  |
| mgiseq_2000 | 58     | 0.8835                | 193.9              | 2577         | 2232           | False                     | 0.7249          | 0.1189              | 1380                   |
| mgiseq_t7   | 59     | 0.883                 | 182                | 2549         | 1645           | False                     | 0.7983          | 0.0985              | 1253                   |

### PRIMARY (fragmentation axis) — leakage-free organisms, true completeness ≥50 %, ORF-intact assemblies only

| tool          | n  | clusters | comp MAE [95% CI] | comp bias [95% CI]   | comp R2 (CoD) | cont MAE [95% CI]  | cont bias [95% CI] | cont R2 (CoD)  |
|---------------|----|----------|-------------------|----------------------|---------------|--------------------|--------------------|----------------|
| MAGICC V5     | 93 | 18       | 3.96 [2.70, 5.29] | 2.77 [0.81, 4.63]    | 0.733         | 0.87 [0.44, 1.34]  | 0.69 [0.22, 1.19]  | n/a (SS_tot~0) |
| CheckM2 1.0.1 | 93 | 18       | 4.53 [3.19, 6.14] | -2.18 [-4.60, -0.02] | 0.58          | 2.48 [1.73, 3.26]  | 2.46 [1.71, 3.25]  | n/a (SS_tot~0) |
| CoCoPyE 0.5.0 | 93 | 18       | 4.22 [2.73, 5.96] | 4.11 [2.59, 5.86]    | 0.615         | 8.20 [5.17, 11.53] | 8.17 [5.15, 11.49] | n/a (SS_tot~0) |
| DeepCheck     | 93 | 18       | 5.66 [3.52, 8.21] | -4.77 [-7.69, -2.21] | 0.255         | 1.68 [1.12, 2.30]  | 1.56 [0.95, 2.23]  | n/a (SS_tot~0) |

### PRIMARY (all seven assemblies) — leakage-free organisms, true completeness ≥50 %

| tool          | n   | clusters | comp MAE [95% CI]    | comp bias [95% CI]     | comp R2 (CoD) | cont MAE [95% CI]   | cont bias [95% CI]  | cont R2 (CoD)  |
|---------------|-----|----------|----------------------|------------------------|---------------|---------------------|---------------------|----------------|
| MAGICC V5     | 108 | 18       | 4.20 [3.05, 5.43]    | 1.96 [-0.28, 3.90]     | 0.654         | 0.88 [0.47, 1.32]   | 0.69 [0.23, 1.18]   | n/a (SS_tot~0) |
| CheckM2 1.0.1 | 108 | 18       | 11.01 [9.28, 12.97]  | -8.98 [-11.58, -6.56]  | -3.337        | 2.63 [2.00, 3.31]   | 2.62 [1.98, 3.30]   | n/a (SS_tot~0) |
| CoCoPyE 0.5.0 | 108 | 18       | 3.94 [2.59, 5.50]    | 3.35 [2.07, 4.82]      | 0.634         | 10.07 [7.51, 12.82] | 10.05 [7.48, 12.78] | n/a (SS_tot~0) |
| DeepCheck     | 108 | 18       | 13.25 [10.72, 16.09] | -12.49 [-15.60, -9.66] | -4.956        | 1.86 [1.43, 2.33]   | 1.72 [1.22, 2.26]   | n/a (SS_tot~0) |

### Per-base-accuracy axis — the indel-dense assembly alone

| tool          | n  | clusters | comp MAE [95% CI]    | comp bias [95% CI]      | comp R2 (CoD) | cont MAE [95% CI]    | cont bias [95% CI]   | cont R2 (CoD) |
|---------------|----|----------|----------------------|-------------------------|---------------|----------------------|----------------------|---------------|
| MAGICC V5     | 39 | 39       | 4.90 [2.85, 7.45]    | -2.92 [-5.78, -0.59]    | 0.404         | 1.57 [0.76, 2.58]    | 1.34 [0.49, 2.38]    | -8.688        |
| CheckM2 1.0.1 | 39 | 39       | 50.69 [45.52, 55.34] | -50.69 [-55.34, -45.52] | -20.69        | 3.45 [2.87, 4.05]    | 3.41 [2.81, 4.03]    | -13.38        |
| CoCoPyE 0.5.0 | 39 | 39       | 3.11 [2.22, 4.11]    | -1.83 [-3.09, -0.60]    | 0.859         | 22.17 [20.68, 23.70] | 22.17 [20.68, 23.70] | -473.8        |
| DeepCheck     | 39 | 39       | 56.83 [51.75, 61.71] | -56.83 [-61.71, -51.75] | -25.76        | 4.20 [3.24, 5.26]    | 4.09 [3.05, 5.17]    | -24.7         |

### Secondary — all MOCK1 organisms, ≥50 %, ORF-intact assemblies

| tool          | n   | clusters | comp MAE [95% CI] | comp bias [95% CI]   | comp R2 (CoD) | cont MAE [95% CI]  | cont bias [95% CI] | cont R2 (CoD)  |
|---------------|-----|----------|-------------------|----------------------|---------------|--------------------|--------------------|----------------|
| MAGICC V5     | 269 | 52       | 4.02 [2.77, 5.93] | 1.88 [-0.34, 3.39]   | 0.616         | 1.35 [0.76, 2.24]  | 1.20 [0.58, 2.10]  | n/a (SS_tot~0) |
| CheckM2 1.0.1 | 269 | 52       | 6.35 [4.89, 7.87] | -4.39 [-6.32, -2.48] | 0.29          | 1.66 [1.29, 2.09]  | 1.61 [1.22, 2.06]  | n/a (SS_tot~0) |
| CoCoPyE 0.5.0 | 269 | 52       | 4.87 [3.70, 6.12] | 3.01 [1.37, 4.52]    | 0.591         | 8.49 [6.48, 10.52] | 8.48 [6.47, 10.51] | n/a (SS_tot~0) |
| DeepCheck     | 269 | 52       | 7.76 [5.90, 9.77] | -6.84 [-9.10, -4.74] | -0.048        | 1.51 [1.19, 1.85]  | 1.35 [0.98, 1.73]  | n/a (SS_tot~0) |

### Secondary — all MOCK1 organisms, ≥50 %, all seven assemblies

| tool          | n   | clusters | comp MAE [95% CI]    | comp bias [95% CI]      | comp R2 (CoD) | cont MAE [95% CI]   | cont bias [95% CI]  | cont R2 (CoD)  |
|---------------|-----|----------|----------------------|-------------------------|---------------|---------------------|---------------------|----------------|
| MAGICC V5     | 308 | 52       | 4.14 [2.88, 5.99]    | 1.27 [-1.01, 2.86]      | 0.59          | 1.38 [0.79, 2.26]   | 1.22 [0.61, 2.11]   | n/a (SS_tot~0) |
| CheckM2 1.0.1 | 308 | 52       | 11.96 [10.56, 13.40] | -10.25 [-12.02, -8.49]  | -2.36         | 1.89 [1.54, 2.29]   | 1.84 [1.47, 2.25]   | n/a (SS_tot~0) |
| CoCoPyE 0.5.0 | 308 | 52       | 4.65 [3.59, 5.77]    | 2.40 [0.98, 3.73]       | 0.626         | 10.23 [8.52, 12.04] | 10.22 [8.51, 12.03] | n/a (SS_tot~0) |
| DeepCheck     | 308 | 52       | 13.98 [12.33, 15.65] | -13.17 [-15.04, -11.30] | -3.294        | 1.85 [1.55, 2.16]   | 1.69 [1.35, 2.03]   | n/a (SS_tot~0) |

### Below MAGICC's stated 50 % completeness floor (reported for transparency; MAGICC does not claim this regime)

| tool          | n  | clusters | comp MAE [95% CI]    | comp bias [95% CI]   | comp R2 (CoD) | cont MAE [95% CI]  | cont bias [95% CI] | cont R2 (CoD)  |
|---------------|----|----------|----------------------|----------------------|---------------|--------------------|--------------------|----------------|
| MAGICC V5     | 86 | 24       | 29.97 [25.01, 34.76] | 29.97 [25.01, 34.76] | -3.445        | 0.34 [0.14, 0.58]  | 0.27 [0.07, 0.53]  | n/a (SS_tot~0) |
| CheckM2 1.0.1 | 86 | 24       | 6.79 [4.93, 9.01]    | -4.82 [-7.65, -2.27] | 0.641         | 1.15 [0.78, 1.58]  | 1.10 [0.74, 1.54]  | n/a (SS_tot~0) |
| CoCoPyE 0.5.0 | 86 | 24       | 39.20 [34.58, 43.64] | 39.20 [34.58, 43.64] | -5.9          | 8.38 [5.36, 11.82] | 8.36 [5.34, 11.80] | n/a (SS_tot~0) |
| DeepCheck     | 86 | 24       | 11.94 [10.25, 14.11] | 0.88 [-3.31, 5.03]   | 0.209         | 2.58 [1.95, 3.35]  | 2.57 [1.94, 3.34]  | n/a (SS_tot~0) |

### Fragmentation gradient — completeness MAE (pp) by assembly

Balanced panel: only organisms recovered in **every** assembly at ≥50 % true completeness, so the same genomes are compared at every fragmentation level. The `ORF_broken` row is NOT a point on the fragmentation axis — see the two-axes section above.

| assembly    | assembly_median_bin_n50 | ORF_broken | CheckM2 1.0.1 | CoCoPyE 0.5.0 | DeepCheck | MAGICC V5 |
|-------------|-------------------------|------------|---------------|---------------|-----------|-----------|
| pacbio      | 1853991                 | False      | 0.303         | 1.347         | 1.393     | 1.735     |
| minion      | 313692                  | True       | 51.8          | 3.049         | 57.47     | 4.876     |
| illumina    | 10696                   | False      | 3.189         | 2.887         | 2.992     | 3.636     |
| s5          | 6522                    | False      | 5.601         | 4.307         | 7.78      | 5.115     |
| proton      | 5244                    | False      | 8.706         | 4.469         | 10.5      | 5.326     |
| mgiseq_2000 | 2231                    | False      | 6.75          | 5.111         | 7.721     | 4.528     |
| mgiseq_t7   | 1645                    | False      | 7.455         | 6.011         | 7.854     | 4.826     |

### Fragmentation gradient — contamination MAE (pp) by assembly

Balanced panel: only organisms recovered in **every** assembly at ≥50 % true completeness, so the same genomes are compared at every fragmentation level. The `ORF_broken` row is NOT a point on the fragmentation axis — see the two-axes section above.

| assembly    | assembly_median_bin_n50 | ORF_broken | CheckM2 1.0.1 | CoCoPyE 0.5.0 | DeepCheck | MAGICC V5 |
|-------------|-------------------------|------------|---------------|---------------|-----------|-----------|
| pacbio      | 1853991                 | False      | 0.4933        | 0.9046        | 0.4581    | 2.055     |
| minion      | 313692                  | True       | 3.414         | 21.88         | 4.277     | 1.619     |
| illumina    | 10696                   | False      | 1.199         | 2.857         | 0.8867    | 1.442     |
| s5          | 6522                    | False      | 2.05          | 9.42          | 1.89      | 1.498     |
| proton      | 5244                    | False      | 2.34          | 12.29         | 1.839     | 1.148     |
| mgiseq_2000 | 2231                    | False      | 1.778         | 8.334         | 1.713     | 1.46      |
| mgiseq_t7   | 1645                    | False      | 1.648         | 8.615         | 1.613     | 1.36      |

### Degradation slopes (MAE per log10 decrease in bin N50)

ORF-intact assemblies only.

| panel                   | tool          | metric   | slope_per_log10_N50 | slope_se | p_two_sided | pearson_r | value_at_best_N50 | value_at_worst_N50 | degradation | assemblies_used                                 |
|-------------------------|---------------|----------|---------------------|----------|-------------|-----------|-------------------|--------------------|-------------|-------------------------------------------------|
| balanced_panel_comp>=50 | MAGICC V5     | comp_mae | -1.084              | 0.2554   | 0.01323     | -0.9045   | 1.735             | 4.826              | 3.091       | pacbio,illumina,s5,proton,mgiseq_2000,mgiseq_t7 |
| balanced_panel_comp>=50 | MAGICC V5     | cont_mae | 0.2385              | 0.0631   | 0.01948     | 0.8838    | 2.055             | 1.36               | -0.6958     | pacbio,illumina,s5,proton,mgiseq_2000,mgiseq_t7 |
| balanced_panel_comp>=50 | CheckM2 1.0.1 | comp_mae | -2.401              | 0.6837   | 0.02465     | -0.8689   | 0.303             | 7.455              | 7.152       | pacbio,illumina,s5,proton,mgiseq_2000,mgiseq_t7 |
| balanced_panel_comp>=50 | CheckM2 1.0.1 | cont_mae | -0.4721             | 0.1753   | 0.05447     | -0.8029   | 0.4933            | 1.648              | 1.155       | pacbio,illumina,s5,proton,mgiseq_2000,mgiseq_t7 |
| balanced_panel_comp>=50 | CoCoPyE 0.5.0 | comp_mae | -1.363              | 0.2973   | 0.01015     | -0.9166   | 1.347             | 6.011              | 4.664       | pacbio,illumina,s5,proton,mgiseq_2000,mgiseq_t7 |
| balanced_panel_comp>=50 | CoCoPyE 0.5.0 | cont_mae | -2.877              | 1.272    | 0.08652     | -0.7491   | 0.9046            | 8.615              | 7.71        | pacbio,illumina,s5,proton,mgiseq_2000,mgiseq_t7 |
| balanced_panel_comp>=50 | DeepCheck     | comp_mae | -2.368              | 0.9796   | 0.07295     | -0.7705   | 1.393             | 7.854              | 6.461       | pacbio,illumina,s5,proton,mgiseq_2000,mgiseq_t7 |
| balanced_panel_comp>=50 | DeepCheck     | cont_mae | -0.4363             | 0.1457   | 0.04016     | -0.8316   | 0.4581            | 1.613              | 1.155       | pacbio,illumina,s5,proton,mgiseq_2000,mgiseq_t7 |

### Paired MAGICC-vs-competitor tests (two-sided Wilcoxon on paired absolute errors)

| cohort                                   | metric        | tool_b        | n   | mae_a  | mae_b | mean_diff_a_minus_b | diff_ci_lo | diff_ci_hi | hodges_lehmann_shift | wilcoxon_p_two_sided | winner        |
|------------------------------------------|---------------|---------------|-----|--------|-------|---------------------|------------|------------|----------------------|----------------------|---------------|
| primary_leakage_free_comp>=50            | completeness  | CheckM2 1.0.1 | 108 | 4.196  | 11.01 | -6.811              | -8.919     | -4.726     | -1.358               | 0.008164             | MAGICC V5     |
| primary_leakage_free_comp>=50            | contamination | CheckM2 1.0.1 | 108 | 0.8793 | 2.635 | -1.755              | -2.518     | -0.9677    | -1.731               | 4.998e-10            | MAGICC V5     |
| primary_leakage_free_comp>=50            | completeness  | CoCoPyE 0.5.0 | 108 | 4.196  | 3.939 | 0.2562              | -1.232     | 1.5        | 0.3722               | 0.2779               | CoCoPyE 0.5.0 |
| primary_leakage_free_comp>=50            | contamination | CoCoPyE 0.5.0 | 108 | 0.8793 | 10.07 | -9.195              | -11.99     | -6.617     | -9.987               | 1.68e-09             | MAGICC V5     |
| primary_leakage_free_comp>=50            | completeness  | DeepCheck     | 108 | 4.196  | 13.25 | -9.058              | -11.86     | -6.407     | -3.242               | 0.0008165            | MAGICC V5     |
| primary_leakage_free_comp>=50            | contamination | DeepCheck     | 108 | 0.8793 | 1.863 | -0.9837             | -1.592     | -0.3956    | -0.9172              | 9.788e-05            | MAGICC V5     |
| primary_leakage_free_comp>=50_ORF_intact | completeness  | CheckM2 1.0.1 | 93  | 3.955  | 4.534 | -0.5786             | -2.209     | 0.9117     | -0.0423              | 0.686                | MAGICC V5     |
| primary_leakage_free_comp>=50_ORF_intact | contamination | CheckM2 1.0.1 | 93  | 0.866  | 2.478 | -1.612              | -2.553     | -0.667     | -1.61                | 1.42e-07             | MAGICC V5     |
| primary_leakage_free_comp>=50_ORF_intact | completeness  | CoCoPyE 0.5.0 | 93  | 3.955  | 4.218 | -0.2633             | -1.785     | 1.02       | 0.1332               | 0.5564               | MAGICC V5     |
| primary_leakage_free_comp>=50_ORF_intact | contamination | CoCoPyE 0.5.0 | 93  | 0.866  | 8.197 | -7.331              | -10.78     | -4.1       | -8.314               | 3.648e-06            | MAGICC V5     |
| primary_leakage_free_comp>=50_ORF_intact | completeness  | DeepCheck     | 93  | 3.955  | 5.657 | -1.702              | -4.175     | 0.5918     | -0.5698              | 0.2                  | MAGICC V5     |
| primary_leakage_free_comp>=50_ORF_intact | contamination | DeepCheck     | 93  | 0.866  | 1.684 | -0.8183             | -1.605     | -0.0513    | -0.7252              | 0.003315             | MAGICC V5     |
| ORF_compromised_assembly_only            | completeness  | CheckM2 1.0.1 | 39  | 4.905  | 50.69 | -45.78              | -51.01     | -40.06     | -46.14               | 3.638e-12            | MAGICC V5     |
| ORF_compromised_assembly_only            | contamination | CheckM2 1.0.1 | 39  | 1.572  | 3.45  | -1.879              | -2.877     | -0.7382    | -2.265               | 9.755e-05            | MAGICC V5     |
| ORF_compromised_assembly_only            | completeness  | CoCoPyE 0.5.0 | 39  | 4.905  | 3.115 | 1.79                | -0.5367    | 4.593      | 0.373                | 0.6839               | CoCoPyE 0.5.0 |
| ORF_compromised_assembly_only            | contamination | CoCoPyE 0.5.0 | 39  | 1.572  | 22.17 | -20.6               | -22.29     | -18.93     | -20.72               | 3.638e-12            | MAGICC V5     |
| ORF_compromised_assembly_only            | completeness  | DeepCheck     | 39  | 4.905  | 56.83 | -51.93              | -57.55     | -46.21     | -52.15               | 3.638e-12            | MAGICC V5     |
| ORF_compromised_assembly_only            | contamination | DeepCheck     | 39  | 1.572  | 4.195 | -2.623              | -3.565     | -1.639     | -2.821               | 6.288e-06            | MAGICC V5     |

### MIMAG-inspired classification — PRIMARY (ORF-intact)

| tool          | n  | true_HQ | pred_HQ | true_MQ | pred_MQ | class_agreement_pct | n_truly_below_5pct_cont | false_fail_5pct_n | false_fail_5pct_rate | comp90_recall_pct |
|---------------|----|---------|---------|---------|---------|---------------------|-------------------------|-------------------|----------------------|-------------------|
| MAGICC V5     | 93 | 73      | 79      | 20      | 13      | 91.4                | 93                      | 1                 | 1.08                 | 100               |
| CheckM2 1.0.1 | 93 | 73      | 61      | 20      | 25      | 78.49               | 93                      | 18                | 19.35                | 87.67             |
| CoCoPyE 0.5.0 | 93 | 73      | 55      | 20      | 4       | 58.06               | 93                      | 38                | 40.86                | 100               |
| DeepCheck     | 93 | 73      | 59      | 20      | 28      | 78.49               | 93                      | 11                | 11.83                | 80.82             |

### MIMAG-inspired classification — all seven assemblies

| tool          | n   | true_HQ | pred_HQ | true_MQ | pred_MQ | class_agreement_pct | n_truly_below_5pct_cont | false_fail_5pct_n | false_fail_5pct_rate | comp90_recall_pct |
|---------------|-----|---------|---------|---------|---------|---------------------|-------------------------|-------------------|----------------------|-------------------|
| MAGICC V5     | 108 | 87      | 90      | 21      | 17      | 89.81               | 108                     | 2                 | 1.85                 | 96.55             |
| CheckM2 1.0.1 | 108 | 87      | 61      | 21      | 32      | 68.52               | 108                     | 23                | 21.3                 | 73.56             |
| CoCoPyE 0.5.0 | 108 | 87      | 55      | 21      | 4       | 50                  | 108                     | 53                | 49.07                | 100               |
| DeepCheck     | 108 | 87      | 59      | 21      | 33      | 67.59               | 108                     | 15                | 13.89                | 67.82             |

### Where the contamination false positives live

Computed on truly-clean bins only (true contamination <1 %, true completeness ≥50 %), aggregated per reference organism. Whether a tool's false positives are spread thinly over many organisms or concentrated in a few is the difference between a noise floor and a systematic blind spot, and only the latter yields a usable limitation statement. Organisms (of 52) whose MEDIAN predicted contamination is ≥5 % / ≥2 %: magicc 2 / 5; checkm2 1 / 20; cocopye 27 / 33; deepcheck 1 / 18.

Ten worst organisms for MAGICC:

| accession       | organism                                 | gtdb_phylum        | kingdom  | leakage_free | n_bins | true_cont_max | magicc_cont_median | magicc_n_bins_ge5pct | checkm2_cont_median | checkm2_n_bins_ge5pct | cocopye_cont_median | cocopye_n_bins_ge5pct | deepcheck_cont_median | deepcheck_n_bins_ge5pct |
|-----------------|------------------------------------------|--------------------|----------|--------------|--------|---------------|--------------------|----------------------|---------------------|-----------------------|---------------------|-----------------------|-----------------------|-------------------------|
| GCA_000166095.1 | Methanothermus fervidus                  | Methanobacteriota  | Archaea  | False        | 7      | 0             | 16.06              | 7                    | 0.11                | 0                     | 0                   | 1                     | -0.093                | 1                       |
| GCA_000020465.1 | Chlorobium limicola                      | Bacteroidota       | Bacteria | False        | 5      | 0.0052        | 5.466              | 4                    | 0.18                | 0                     | 0                   | 1                     | -0.355                | 0                       |
| GCA_000020325.1 | Sulfurihydrogenibium sp000020325         | Aquificota         | Bacteria | False        | 7      | 0.6226        | 4.905              | 3                    | 1.42                | 0                     | 25.67               | 6                     | 2.589                 | 2                       |
| GCA_000016545.1 | Caldicellulosiruptor saccharolyticus     | Firmicutes_A       | Bacteria | True         | 7      | 0.4436        | 2.735              | 0                    | 0.12                | 0                     | 0                   | 1                     | 0.694                 | 1                       |
| GCA_000021565.1 | Persephonella marina                     | Aquificota         | Bacteria | False        | 6      | 0             | 2.046              | 0                    | 0.41                | 0                     | 0.812               | 1                     | 0.548                 | 0                       |
| GCA_002952055.1 | Desulfobulbus oralis                     | Desulfobacterota   | Bacteria | False        | 7      | 0.2882        | 1.826              | 0                    | 0.2                 | 0                     | 0                   | 2                     | 0.097                 | 0                       |
| GCA_000018565.1 | Herpetosiphon aurantiacus                | Chloroflexota      | Bacteria | False        | 6      | 0.0387        | 1.725              | 0                    | 0.525               | 0                     | 4.182               | 2                     | -0.382                | 1                       |
| GCA_003019295.1 | Fusobacterium nucleatum subsp. nucleatum | Fusobacteriota     | Bacteria | False        | 7      | 0.7141        | 1.652              | 0                    | 1.22                | 0                     | 3.696               | 3                     | 0.168                 | 0                       |
| GCA_000013645.1 | Paraburkholderia xenovorans              | Proteobacteria (g) | Bacteria | True         | 4      | 0.1829        | 1.617              | 0                    | 4.745               | 2                     | 20.85               | 3                     | 3.099                 | 0                       |
| GCA_000008085.1 | Nanoarchaeum equitans                    | Nanoarchaeota      | Archaea  | False        | 5      | 0.3107        | 1.615              | 0                    | 0.29                | 0                     | 3.125               | 1                     | 1.903                 | 0                       |

### Per-base-accuracy effect on the same balanced panel

| cohort                     | tool          | n   | comp_mae | comp_bias | cont_mae | cont_bias | assemblies                                      |
|----------------------------|---------------|-----|----------|-----------|----------|-----------|-------------------------------------------------|
| ORF_compromised_assemblies | MAGICC V5     | 37  | 4.876    | -3.372    | 1.619    | 1.374     | minion                                          |
| ORF_compromised_assemblies | CheckM2 1.0.1 | 37  | 51.8     | -51.8     | 3.414    | 3.373     | minion                                          |
| ORF_compromised_assemblies | CoCoPyE 0.5.0 | 37  | 3.049    | -2.167    | 21.88    | 21.88     | minion                                          |
| ORF_compromised_assemblies | DeepCheck     | 37  | 57.47    | -57.47    | 4.277    | 4.172     | minion                                          |
| ORF_intact_assemblies      | MAGICC V5     | 222 | 4.194    | 1.633     | 1.494    | 1.336     | illumina,mgiseq_2000,mgiseq_t7,pacbio,proton,s5 |
| ORF_intact_assemblies      | CheckM2 1.0.1 | 222 | 5.334    | -3.027    | 1.585    | 1.52      | illumina,mgiseq_2000,mgiseq_t7,pacbio,proton,s5 |
| ORF_intact_assemblies      | CoCoPyE 0.5.0 | 222 | 4.022    | 2.878     | 7.069    | 7.056     | illumina,mgiseq_2000,mgiseq_t7,pacbio,proton,s5 |
| ORF_intact_assemblies      | DeepCheck     | 222 | 6.373    | -5.255    | 1.4      | 1.201     | illumina,mgiseq_2000,mgiseq_t7,pacbio,proton,s5 |

## A2 — ZymoBIOMICS isolate draft/complete pairs

8 bacterial SPAdes drafts (Nicholls et al. 2019) vs the ZymoBIOMICS complete references; the 2 yeasts are excluded as eukaryotes. Each draft was aligned against the POOLED 10-organism reference so that cross-isolate carry-over is detectable as contamination. True completeness 94.68–99.18 %, true contamination 0.000–0.118 % — a **precision / false-positive test** in the near-complete, near-clean regime.

| organism                      | n_contigs | draft_bp | ref_len | true_completeness | true_contamination | true_contamination_upper | magicc_completeness | checkm2_completeness | cocopye_completeness | deepcheck_completeness | magicc_contamination | checkm2_contamination | cocopye_contamination | deepcheck_contamination |
|-------------------------------|-----------|----------|---------|-------------------|--------------------|--------------------------|---------------------|----------------------|----------------------|------------------------|----------------------|-----------------------|-----------------------|-------------------------|
| Bacillus subtilis             | 29        | 3985829  | 4045677 | 98.41             | 0.118              | 0.1325                   | 99.94               | 100                  | 100                  | 98.96                  | 0.2503               | 0.1                   | 0                     | -0.3085                 |
| Escherichia coli              | 103       | 4787036  | 4875441 | 98.1              | 0                  | 0.0352                   | 99.9                | 100                  | 99.42                | 98.9                   | 0.7164               | 0.15                  | 0                     | -0.1679                 |
| Enterococcus faecalis         | 30        | 2820294  | 2845392 | 99.18             | 0                  | 0.0212                   | 99.89               | 99.99                | 99.83                | 99.34                  | 0.1991               | 0.13                  | 0                     | -0.3791                 |
| Limosilactobacillus fermentum | 84        | 1805517  | 1905333 | 94.68             | 0                  | 0.082                    | 99.93               | 99.96                | 99.93                | 98.87                  | 0.2607               | 0.03                  | 0.001126              | 0.3882                  |
| Listeria monocytogenes        | 16        | 2958570  | 2992342 | 98.9              | 0                  | 0.0105                   | 99.73               | 99.99                | 100                  | 99.07                  | 0.6478               | 0.87                  | 2.705                 | -0.2119                 |
| Pseudomonas aeruginosa        | 83        | 6728151  | 6792330 | 99.01             | 0.0024             | 0.025                    | 99.91               | 100                  | 100                  | 98.94                  | 0.646                | 0.07                  | 0                     | 0.24                    |
| Staphylococcus aureus         | 67        | 2689205  | 2730326 | 98.39             | 0                  | 0.0555                   | 99.97               | 100                  | 99.95                | 99.42                  | 0.2805               | 0.02                  | 0.5136                | -0.2774                 |
| Salmonella enterica           | 66        | 4738357  | 4809318 | 98.46             | 0                  | 0.0316                   | 99.89               | 100                  | 99.95                | 99.32                  | 0.5729               | 0.05                  | 0.4305                | -0.1804                 |

| tool          | n | clusters | comp MAE [95% CI] | comp bias [95% CI] | comp R2 (CoD) | cont MAE [95% CI] | cont bias [95% CI]  | cont R2 (CoD)  |
|---------------|---|----------|-------------------|--------------------|---------------|-------------------|---------------------|----------------|
| MAGICC V5     | 8 | 8        | 1.75 [1.05, 2.85] | 1.75 [1.05, 2.85]  | -1.706        | 0.43 [0.28, 0.58] | 0.43 [0.28, 0.58]   | n/a (SS_tot~0) |
| CheckM2 1.0.1 | 8 | 8        | 1.85 [1.17, 2.91] | 1.85 [1.17, 2.91]  | -1.852        | 0.17 [0.04, 0.37] | 0.16 [0.03, 0.37]   | n/a (SS_tot~0) |
| CoCoPyE 0.5.0 | 8 | 8        | 1.74 [1.08, 2.81] | 1.74 [1.08, 2.81]  | -1.665        | 0.47 [0.05, 1.15] | 0.44 [-0.00, 1.13]  | n/a (SS_tot~0) |
| DeepCheck     | 8 | 8        | 0.98 [0.33, 1.98] | 0.96 [0.29, 1.97]  | -0.389        | 0.28 [0.22, 0.35] | -0.13 [-0.30, 0.08] | n/a (SS_tot~0) |

## A3 — NCBI Tier-1 same-BioSample draft/complete pairs

## WS3.9 — cross-dataset synthesis

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

Figures: `results/revision/real_data/figures/` (`fig_ws3_fragmentation_gradient`, `fig_ws3_meslier_scatter`, `fig_ws3_signed_errors`, `fig_ws3_realdata_synthesis`), captions in `figures/captions.md`.
