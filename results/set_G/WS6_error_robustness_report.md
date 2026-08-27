# WS6 — Sequencing and assembly error robustness (Set G)

Generated 2026-07-28T03:36:49.943959+00:00 · model `models/magicc_v5.onnx` (frozen) · addresses **R2-o3**.

## Set G composition

- **1,920 assemblies** = 80 held-out reference genomes x 24 error arms, spanning 35 phyla (Bacteria 67 references, Archaea 13 references), genome size 0.62-9.79 Mbp (median 2.70), fragmentation tiers balanced over medium 20, low 20, highly_fragmented 20, high 20 (base contig count median 18, max 326).
- Dominants from `data/splits/test_finished_genomes.tsv` (complete genomes, so the full reference length used in the denominator is exact); contaminants cross-phylum from the same held-out test split.
- True completeness 58.0-100.0 % (mean 79.4); true contamination 0.0-20.0 % (mean 6.9). **0** samples outside the training domain (contamination % <= completeness %).
- Every arm of a reference is applied to **one shared base assembly**, so control and error arms have an identical ground truth, fragmentation realisation and contaminants; all degradation estimates are paired within reference and every CI is a cluster bootstrap over reference genomes.
- **Provenance / disjointness verdict: PASS** (GCA<->GCF cross-mapped, 1927 files hashed) — `results/revision/provenance/set_G_provenance_audit.json`.

## Ground truth is unchanged by error injection

Substitutions change which base occupies a position, not which organism the position derives from, and change no length — truth is exactly invariant. Chimeras re-partition exactly the same multiset of bases. Uneven-coverage duplication re-emits sequence already present, so the set of represented reference positions is unchanged. Indels are the only length-changing process; they are balanced 50/50 and the residual effect on truth is bounded at **0.052 pp** of completeness (column `true_completeness_indel_adjusted`). An alternative accounting in which duplicated bp are scored as contamination is emitted as `true_contamination_dup_counted`.

## Degradation curves

### Substitutions (uniform)

```
               comp_mae                             cont_mae                            
tool            checkm2 cocopye deepcheck magicc_v5  checkm2 cocopye deepcheck magicc_v5
error_rate_pct                                                                          
0.0               5.108   4.233     5.907     4.398    4.329   2.721     5.178     2.721
0.1               5.152   4.099     5.906     4.367    4.308   3.152     5.156     2.763
0.5               6.530   4.435     6.810     4.472    4.086   6.195     4.904     2.921
1.0               8.735   4.621     9.332     4.878    4.382   9.906     4.885     3.338
2.0              14.540   4.696    15.415     6.908    4.474  14.697     5.083     4.678
5.0              31.067   6.110    33.285    12.184    4.944  18.242     5.387     7.878
```

### Substitutions (Ti/Tv = 2)

```
               comp_mae                             cont_mae                            
tool            checkm2 cocopye deepcheck magicc_v5  checkm2 cocopye deepcheck magicc_v5
error_rate_pct                                                                          
0.0               5.108   4.233     5.907     4.398    4.329   2.721     5.178     2.721
0.1               5.331   4.304     5.933     4.364    4.213   3.042     5.129     2.765
0.5               5.556   4.586     6.081     4.455    4.189   5.156     5.055     3.129
1.0               7.092   4.371     7.846     5.039    4.247   7.730     4.759     3.281
2.0              11.471   4.562    12.211     6.704    4.649  12.301     5.070     4.481
5.0              23.897   5.824    25.669    11.763    4.742  16.897     4.982     8.260
```

### Indels (50 % ins / 50 % del)

```
               comp_mae                             cont_mae                            
tool            checkm2 cocopye deepcheck magicc_v5  checkm2 cocopye deepcheck magicc_v5
error_rate_pct                                                                          
0.0               5.108   4.233     5.907     4.398    4.329   2.721     5.178     2.721
0.1              15.248   4.442    18.001     4.385    4.534  16.876     5.119     2.768
0.5              44.231   5.801    48.780     4.455    6.126  19.832     6.216     2.809
1.0              46.575   6.453    54.610     4.970    6.286  19.149     6.558     3.027
2.0              49.875   8.308    46.493     6.862    7.158  19.279     6.954     3.601
5.0              54.631  11.304    36.422    11.927    7.226  20.215    11.223     5.812
```

### Chimeric mis-joins

```
               comp_mae                             cont_mae                            
tool            checkm2 cocopye deepcheck magicc_v5  checkm2 cocopye deepcheck magicc_v5
error_rate_pct                                                                          
0.0               5.108   4.233     5.907     4.398    4.329   2.721     5.178     2.721
5.0               5.112   4.233     5.905     4.398    4.324   2.720     5.186     2.721
10.0              5.119   4.234     5.920     4.397    4.327   2.718     5.181     2.723
20.0              5.125   4.230     5.915     4.396    4.330   2.733     5.192     2.722
40.0              5.178   4.233     5.888     4.398    4.321   2.725     5.192     2.720
```

### Uneven coverage (duplication)

```
               comp_mae                             cont_mae                            
tool            checkm2 cocopye deepcheck magicc_v5  checkm2 cocopye deepcheck magicc_v5
error_rate_pct                                                                          
0.0               5.108   4.233     5.907     4.398    4.329   2.721     5.178     2.721
5.0               5.267   4.353     5.836     5.373    4.929   5.001     5.036     2.984
10.0              5.115   4.531     6.074     7.670    6.857   8.346     6.402     4.280
20.0              5.196   4.850     5.778    12.167   12.017  12.288    11.480     7.913
40.0              5.520   4.889     6.107    15.985   22.827  19.135    27.161    19.330
```

## Applicability boundary (WS6.4)

```
     tool        error_type        metric  max_rate_tested_pct rate_pct_where_upperCI_delta_mae_exceeds_1pp rate_pct_where_delta_mae_exceeds_1pp rate_pct_where_upperCI_delta_mae_exceeds_2pp rate_pct_where_delta_mae_exceeds_2pp rate_pct_where_upperCI_delta_mae_exceeds_5pp rate_pct_where_delta_mae_exceeds_5pp first_significant_rate_pct  delta_mae_at_max_rate delta_mae_at_max_rate_ci
magicc_v5      substitution  completeness                  5.0                                       0.9642                               1.1943                                        1.306                               1.6801                                       2.5119                               3.0817                        2.0                 7.7867           [6.293, 9.396]
magicc_v5      substitution contamination                  5.0                                       0.9395                               1.2191                                       1.4964                               2.0248                                       3.4772                               4.7806                        2.0                 5.1567           [3.831, 6.523]
magicc_v5 substitution_titv  completeness                  5.0                                       0.9037                               1.1611                                       1.3352                               1.7608                                       2.6212                               3.2578                        1.0                 7.3654           [5.786, 9.196]
magicc_v5 substitution_titv contamination                  5.0                                       1.0103                               1.2899                                       1.6032                               2.1201                                       3.3567                               4.3878                        0.5                 5.5387           [4.162, 6.940]
magicc_v5             indel  completeness                  5.0                                       0.8645                               1.1698                                       1.2822                               1.6871                                       2.5572                                3.164                        1.0                 7.5293           [5.934, 9.163]
magicc_v5             indel contamination                  5.0                                       1.4163                               2.1026                                       2.4303                               3.1819                                                                                                          1.0                 3.0909           [2.102, 4.136]
magicc_v5           chimera  completeness                 40.0                                                                                                                                                                                                                                                                                                  0.0001          [-0.003, 0.003]
magicc_v5           chimera contamination                 40.0                                                                                                                                                                                                                                                                                                 -0.0009          [-0.004, 0.002]
magicc_v5   uneven_coverage  completeness                 40.0                                       2.8716                               5.0373                                       5.3202                               6.8115                                      10.5258                               13.051                        5.0                11.5877          [8.788, 14.100]
magicc_v5   uneven_coverage contamination                 40.0                                       5.7145                               7.4171                                       8.4838                              10.8782                                      14.1187                              19.2807                       10.0                16.6085         [12.540, 20.660]
```

Placed against real per-base error rates:

```
                                     context  per_base_error_rate  per_base_error_rate_pct                          note completeness_within_1pp_boundary  completeness_boundary_pct_1pp  completeness_margin_x_1pp completeness_within_5pp_boundary  completeness_boundary_pct_5pp  completeness_margin_x_5pp contamination_within_1pp_boundary  contamination_boundary_pct_1pp  contamination_margin_x_1pp contamination_within_5pp_boundary  contamination_boundary_pct_5pp  contamination_margin_x_5pp
   Illumina raw read, per base (Q30 nominal)              0.00100                    0.100       typical modern Illumina                              yes                         0.9642                       9.64                              yes                         2.5119                      25.12                               yes                          0.9395                        9.39                               yes                          3.4772                       34.77
      Illumina raw read, per base (Q20 tail)              0.01000                    1.000      poor-quality read region                               no                         0.9642                       0.96                              yes                         2.5119                       2.51                                no                          0.9395                        0.94                               yes                          3.4772                        3.48
           Nanopore R10.4 raw read, per base              0.02000                    2.000       modern simplex nanopore                               no                         0.9642                       0.48                              yes                         2.5119                       1.26                                no                          0.9395                        0.47                               yes                          3.4772                        1.74
     Short-read assembly consensus, per base              0.00001                    0.001      SPAdes/MEGAHIT consensus                              yes                         0.9642                     964.20                              yes                         2.5119                    2511.90                               yes                          0.9395                      939.50                               yes                          3.4772                     3477.20
      Long-read assembly consensus, polished              0.00010                    0.010 HiFi / polished ONT consensus                              yes                         0.9642                      96.42                              yes                         2.5119                     251.19                               yes                          0.9395                       93.95                               yes                          3.4772                      347.72
Long-read assembly consensus, unpolished ONT              0.00100                    0.100           older ONT consensus                              yes                         0.9642                       9.64                              yes                         2.5119                      25.12                               yes                          0.9395                        9.39                               yes                          3.4772                       34.77
```

## Head-to-head crossover

```
comparator        error_type        metric  delta_mae_at_zero_error winner_at_zero_error crossover_rate_pct_magicc_becomes_worse highest_rate_pct_magicc_still_better  delta_mae_at_max_rate delta_mae_at_max_rate_ci winner_at_max_rate
   checkm2      substitution  completeness                  -0.7098                  tie                                                                          5.0               -18.8823       [-21.657, -16.069]             MAGICC
   checkm2      substitution contamination                  -1.6074               MAGICC                                     5.0                                                      2.9346           [1.112, 4.778]            checkm2
   checkm2 substitution_titv  completeness                  -0.7098                  tie                                                                          5.0               -12.1335        [-14.796, -9.459]             MAGICC
   checkm2 substitution_titv contamination                  -1.6074               MAGICC                                     5.0                                                      3.5184           [1.587, 5.381]            checkm2
   checkm2             indel  completeness                  -0.7098                  tie                                                                          5.0               -42.7040       [-47.716, -37.651]             MAGICC
   checkm2             indel contamination                  -1.6074               MAGICC                                                                          2.0                -1.4137          [-3.284, 0.475]                tie
   checkm2           chimera  completeness                  -0.7098                  tie                                                                                             -0.7801          [-2.537, 1.141]                tie
   checkm2           chimera contamination                  -1.6074               MAGICC                                                                                             -1.6009         [-2.869, -0.374]             MAGICC
   checkm2   uneven_coverage  completeness                  -0.7098                  tie                                    10.0                                                     10.4657          [8.522, 12.509]            checkm2
   checkm2   uneven_coverage contamination                  -1.6074               MAGICC                                                                         20.0                -3.4968          [-6.939, 0.169]                tie
   cocopye      substitution  completeness                   0.1651                  tie                                     2.0                                                      6.0746           [4.018, 8.306]            cocopye
   cocopye      substitution contamination                   0.0006                  tie                                                                          5.0               -10.3640        [-12.287, -8.408]             MAGICC
   cocopye substitution_titv  completeness                   0.1651                  tie                                     2.0                                                      5.9397           [3.912, 8.051]            cocopye
   cocopye substitution_titv contamination                   0.0006                  tie                                                                          5.0                -8.6368        [-10.700, -6.490]             MAGICC
   cocopye             indel  completeness                   0.1651                  tie                                                                                              0.6235          [-1.708, 2.746]                tie
   cocopye             indel contamination                   0.0006                  tie                                                                          5.0               -14.4027       [-16.679, -12.172]             MAGICC
   cocopye           chimera  completeness                   0.1651                  tie                                                                                              0.1646          [-1.291, 1.705]                tie
   cocopye           chimera contamination                   0.0006                  tie                                                                                             -0.0044          [-0.893, 0.944]                tie
   cocopye   uneven_coverage  completeness                   0.1651                  tie                                    10.0                                                     11.0960          [8.735, 13.288]            cocopye
   cocopye   uneven_coverage contamination                   0.0006                  tie                                                                         20.0                 0.1945          [-4.014, 4.650]                tie
 deepcheck      substitution  completeness                  -1.5094                  tie                                                                          5.0               -21.1007       [-24.071, -18.377]             MAGICC
 deepcheck      substitution contamination                  -2.4566               MAGICC                                     5.0                                  1.0                 2.4912           [0.707, 4.265]          deepcheck
 deepcheck substitution_titv  completeness                  -1.5094                  tie                                                                          5.0               -13.9059       [-16.357, -11.464]             MAGICC
 deepcheck substitution_titv contamination                  -2.4566               MAGICC                                     5.0                                  1.0                 3.2779           [1.436, 5.176]          deepcheck
 deepcheck             indel  completeness                  -1.5094                  tie                                                                          5.0               -24.4946       [-28.477, -20.550]             MAGICC
 deepcheck             indel contamination                  -2.4566               MAGICC                                                                          5.0                -5.4107         [-7.530, -3.069]             MAGICC
 deepcheck           chimera  completeness                  -1.5094                  tie                                                                                             -1.4900          [-3.022, 0.164]                tie
 deepcheck           chimera contamination                  -2.4566               MAGICC                                                                         40.0                -2.4720         [-3.920, -1.181]             MAGICC
 deepcheck   uneven_coverage  completeness                  -1.5094                  tie                                    20.0                                                      9.8787          [7.660, 12.120]          deepcheck
 deepcheck   uneven_coverage contamination                  -2.4566               MAGICC                                                                         40.0                -7.8311        [-11.733, -3.522]             MAGICC
```

## Mechanism

```
       error_type  error_rate_pct  observed_kmer_corruption predicted_kmer_position_corruption  total_count_ratio  l1_relative  normalised_l2_per_dim  summary_l2_per_dim
          chimera             5.0                    0.0000                                                1.0000       0.0000                 0.0002              0.0000
          chimera            10.0                    0.0000                                                1.0000       0.0000                 0.0007              0.0003
          chimera            20.0                    0.0000                                                1.0000       0.0000                 0.0006              0.0001
          chimera            40.0                    0.0000                                                1.0000       0.0000                 0.0011              0.0008
            indel             0.1                    0.0060                                                0.9980       0.0100                 0.0330              0.0246
            indel             0.5                    0.0208                                                0.9899       0.0315                 0.0737              0.1083
            indel             1.0                    0.0347                                                0.9802       0.0497                 0.1054              0.1749
            indel             2.0                    0.0589                                                0.9608       0.0787                 0.1513              0.2940
            indel             5.0                    0.1188                                                0.9087       0.1463                 0.2452              0.5433
     substitution             0.1                    0.0066                           0.008964             0.9981       0.0113                 0.0360              0.0415
     substitution             0.5                    0.0221                            0.04411             0.9906       0.0348                 0.0808              0.1458
     substitution             1.0                    0.0362                           0.086483             0.9813       0.0538                 0.1147              0.2560
     substitution             2.0                    0.0606                           0.166252             0.9631       0.0843                 0.1633              0.4433
     substitution             5.0                    0.1226                           0.369751             0.9111       0.1563                 0.2602              0.8693
substitution_titv             0.1                    0.0065                           0.008964             0.9983       0.0113                 0.0363              0.0358
substitution_titv             0.5                    0.0218                            0.04411             0.9912       0.0348                 0.0817              0.1554
substitution_titv             1.0                    0.0360                           0.086483             0.9823       0.0542                 0.1171              0.2880
substitution_titv             2.0                    0.0593                           0.166252             0.9657       0.0843                 0.1657              0.4966
substitution_titv             5.0                    0.1198                           0.369751             0.9165       0.1561                 0.2651              0.9240
  uneven_coverage             5.0                    0.0000                                                1.0507       0.0507                 0.0690              0.0397
  uneven_coverage            10.0                    0.0000                                                1.1010       0.1010                 0.1190              0.0694
  uneven_coverage            20.0                    0.0000                                                1.1997       0.1997                 0.2081              0.1390
  uneven_coverage            40.0                    0.0000                                                1.3995       0.3995                 0.3540              0.2363
```

```
       error_type  error_rate_pct  n  Total_Coding_Sequences_ratio_vs_control  Coding_Density_ratio_vs_control  Average_Gene_Length_ratio_vs_control  GC_Content_ratio_vs_control  Genome_Size_ratio_vs_control  Contig_N50_ratio_vs_control
          chimera             5.0 80                                   0.9997                           0.9999                                1.0002                       1.0000                          1.00                       1.0252
          chimera            10.0 80                                   0.9995                           0.9999                                1.0004                       1.0000                          1.00                       1.0615
          chimera            20.0 80                                   0.9993                           0.9999                                1.0006                       1.0000                          1.00                       1.1146
          chimera            40.0 80                                   0.9984                           0.9997                                1.0014                       1.0000                          1.00                       1.2301
            indel             0.1 80                                   1.4314                           0.9525                                0.6778                       0.9998                          1.00                       1.0000
            indel             0.5 80                                   2.1012                           0.8520                                0.4485                       1.0002                          1.00                       1.0000
            indel             1.0 80                                   2.0836                           0.7633                                0.4207                       1.0010                          1.00                       1.0001
            indel             2.0 80                                   1.5952                           0.6152                                0.4249                       1.0010                          1.00                       0.9996
            indel             5.0 80                                   1.1530                           0.4325                                0.3647                       1.0032                          1.00                       0.9994
     substitution             0.1 80                                   1.0243                           0.9968                                0.9732                       0.9998                          1.00                       1.0000
     substitution             0.5 80                                   1.1160                           0.9841                                0.8825                       1.0013                          1.00                       1.0000
     substitution             1.0 80                                   1.2240                           0.9690                                0.7935                       1.0010                          1.00                       1.0000
     substitution             2.0 80                                   1.4145                           0.9410                                0.6683                       1.0019                          1.00                       1.0000
     substitution             5.0 80                                   1.8087                           0.8726                                0.4880                       1.0069                          1.00                       1.0000
substitution_titv             0.1 80                                   1.0183                           0.9976                                0.9797                       0.9998                          1.00                       1.0000
substitution_titv             0.5 80                                   1.0870                           0.9880                                0.9094                       1.0018                          1.00                       1.0000
substitution_titv             1.0 80                                   1.1727                           0.9766                                0.8338                       1.0014                          1.00                       1.0000
substitution_titv             2.0 80                                   1.3289                           0.9552                                0.7207                       1.0028                          1.00                       1.0000
substitution_titv             5.0 80                                   1.6877                           0.8980                                0.5355                       1.0073                          1.00                       1.0000
  uneven_coverage             5.0 80                                   1.0500                           1.0000                                1.0001                       0.9999                          1.05                       0.9876
  uneven_coverage            10.0 80                                   1.1014                           1.0001                                0.9989                       1.0004                          1.10                       0.9865
  uneven_coverage            20.0 80                                   1.2036                           0.9999                                0.9971                       0.9997                          1.20                       0.9871
  uneven_coverage            40.0 80                                   1.4031                           1.0006                                0.9985                       1.0008                          1.40                       0.9819
```

## Duplication accounting sensitivity

Under MAGICC's denominator a duplicated segment adds no reference coverage, so ground truth is unchanged; a marker-duplication tool may legitimately read the same segment as contamination. Both accountings:

```
     tool  error_rate_pct  n  true_cont_primary  true_cont_dup_counted  cont_mae_primary  cont_bias_primary  cont_mae_dup_counted  cont_bias_dup_counted
  checkm2             5.0 80             6.9103                11.2235            4.9294             0.6270                5.1069                -3.6861
  checkm2            10.0 80             6.9103                15.5366            6.8569             4.6324                5.4073                -3.9938
  checkm2            20.0 80             6.9103                24.1628           12.0168            11.0718                7.3640                -6.1807
  checkm2            40.0 80             6.9103                41.4153           22.8267            22.7802               12.3292               -11.7248
  cocopye             5.0 80             6.9103                11.2235            5.0014             4.1271                3.9369                -0.1861
  cocopye            10.0 80             6.9103                15.5366            8.3456             7.7193                5.6723                -0.9069
  cocopye            20.0 80             6.9103                24.1628           12.2882            12.2151                8.0322                -5.0374
  cocopye            40.0 80             6.9103                41.4153           19.1353            18.9190               15.7764               -15.5860
deepcheck             5.0 80             6.9103                11.2235            5.0358            -1.0099                5.9127                -5.3230
deepcheck            10.0 80             6.9103                15.5366            6.4018             2.9073                6.5752                -5.7190
deepcheck            20.0 80             6.9103                24.1628           11.4797            10.4510                7.9297                -6.8015
deepcheck            40.0 80             6.9103                41.4153           27.1610            27.1610               12.0655                -7.3440
magicc_v5             5.0 80             6.9103                11.2235            2.9839             1.5513                4.3704                -2.7619
magicc_v5            10.0 80             6.9103                15.5366            4.2801             2.7128                7.2880                -5.9134
magicc_v5            20.0 80             6.9103                24.1628            7.9134             6.6668               13.8550               -10.5857
magicc_v5            40.0 80             6.9103                41.4153           19.3299            18.4980               19.9332               -16.0070
```

## Chimera dose by base contig count

The achievable chimera dose is bounded by the number of contigs available to mis-join, so the arm is also reported on the fragmented half of the panel where the dose is large.

```
     tool error_type  error_rate_pct    stratify_by          stratum  n  mean_realised_dose  mean_chimera_events  delta_comp_mae  delta_cont_mae
magicc_v5    chimera             5.0 contig_stratum  base_contigs<20 42             0.08506                 0.21         -0.0001         -0.0002
magicc_v5    chimera             5.0 contig_stratum base_contigs>=20 38             0.04915                 2.42          0.0000         -0.0000
magicc_v5    chimera            10.0 contig_stratum  base_contigs<20 42             0.10475                 0.43         -0.0007          0.0006
magicc_v5    chimera            10.0 contig_stratum base_contigs>=20 38             0.09802                 4.79          0.0000          0.0034
magicc_v5    chimera            20.0 contig_stratum  base_contigs<20 42             0.19549                 0.67         -0.0011         -0.0002
magicc_v5    chimera            20.0 contig_stratum base_contigs>=20 38             0.20284                 9.71         -0.0017          0.0017
magicc_v5    chimera            40.0 contig_stratum  base_contigs<20 42             0.32900                 1.26         -0.0013         -0.0001
magicc_v5    chimera            40.0 contig_stratum base_contigs>=20 38             0.39970                19.32          0.0015         -0.0017
  checkm2    chimera             5.0 contig_stratum  base_contigs<20 42             0.08506                 0.21         -0.0057          0.0000
  checkm2    chimera             5.0 contig_stratum base_contigs>=20 38             0.04915                 2.42          0.0168         -0.0095
  checkm2    chimera            10.0 contig_stratum  base_contigs<20 42             0.10475                 0.43          0.0124          0.0062
  checkm2    chimera            10.0 contig_stratum base_contigs>=20 38             0.09802                 4.79          0.0101         -0.0100
  checkm2    chimera            20.0 contig_stratum  base_contigs<20 42             0.19549                 0.67         -0.0012          0.0048
  checkm2    chimera            20.0 contig_stratum base_contigs>=20 38             0.20284                 9.71          0.0372         -0.0029
  checkm2    chimera            40.0 contig_stratum  base_contigs<20 42             0.32900                 1.26          0.0738         -0.0155
  checkm2    chimera            40.0 contig_stratum base_contigs>=20 38             0.39970                19.32          0.0667          0.0016
  cocopye    chimera             5.0 contig_stratum  base_contigs<20 42             0.08506                 0.21          0.0000          0.0000
  cocopye    chimera             5.0 contig_stratum base_contigs>=20 38             0.04915                 2.42          0.0003         -0.0006
  cocopye    chimera            10.0 contig_stratum  base_contigs<20 42             0.10475                 0.43          0.0000          0.0000
  cocopye    chimera            10.0 contig_stratum base_contigs>=20 38             0.09802                 4.79          0.0019         -0.0052
  cocopye    chimera            20.0 contig_stratum  base_contigs<20 42             0.19549                 0.67          0.0000          0.0000
  cocopye    chimera            20.0 contig_stratum base_contigs>=20 38             0.20284                 9.71         -0.0049          0.0258
  cocopye    chimera            40.0 contig_stratum  base_contigs<20 42             0.32900                 1.26         -0.0016          0.0002
  cocopye    chimera            40.0 contig_stratum base_contigs>=20 38             0.39970                19.32          0.0029          0.0086
deepcheck    chimera             5.0 contig_stratum  base_contigs<20 42             0.08506                 0.21          0.0010          0.0007
deepcheck    chimera             5.0 contig_stratum base_contigs>=20 38             0.04915                 2.42         -0.0058          0.0155
deepcheck    chimera            10.0 contig_stratum  base_contigs<20 42             0.10475                 0.43          0.0133          0.0021
deepcheck    chimera            10.0 contig_stratum base_contigs>=20 38             0.09802                 4.79          0.0127          0.0049
deepcheck    chimera            20.0 contig_stratum  base_contigs<20 42             0.19549                 0.67          0.0057          0.0027
deepcheck    chimera            20.0 contig_stratum base_contigs>=20 38             0.20284                 9.71          0.0108          0.0258
deepcheck    chimera            40.0 contig_stratum  base_contigs<20 42             0.32900                 1.26          0.0016          0.0007
deepcheck    chimera            40.0 contig_stratum base_contigs>=20 38             0.39970                19.32         -0.0426          0.0297
```

## Conventions honoured

R^2 = coefficient of determination (1 - SS_res/SS_tot) throughout, blank where the truth has near-zero variance (R1-m19). `fw.stable_hash()` (CRC-32) for every bootstrap seed with `PYTHONHASHSEED=0`. Two-sided paired Wilcoxon tests, cluster bootstrap over reference genomes (2,000 iterations), BH correction, Hodges-Lehmann and Cliff's delta with CIs. MIMAG-inspired thresholds (completeness/contamination only). Palette CVD-verified in-script; no red/green discrimination. Denominator stated in every caption (`figures/captions.md`).
