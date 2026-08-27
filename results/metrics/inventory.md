# Benchmark data inventory (WS5 Stage A)

Config: `/path/to/magicc/scripts/config_revision_metrics.yaml`  |  benchmark root: `data/benchmarks`

Denominators: completeness (%) = retained dominant-genome bp / full reference length of the dominant genome x 100; contamination (%) = total contaminant bp / full reference length of the dominant genome x 100. Both percentages use the same denominator (the full reference length of the dominant genome) and are therefore independent measures; contamination is not normalised to total assembly length or to retained dominant length.

## Sets

| set                 | status            | tier      | n_genomes | n_clusters_dominant_accession | max_cluster_size | tools_available                                         | tools_missing       | dominant_split_counts                  | note                              |
|---------------------|-------------------|-----------|-----------|-------------------------------|------------------|---------------------------------------------------------|---------------------|----------------------------------------|-----------------------------------|
| set_A_v2            | current           | primary   | 1000      | 798                           | 5                | magicc_v5,magicc_v4,magicc_v3,checkm2,cocopye,deepcheck |                     | {"test": 1000}                         |                                   |
| set_B_v2            | current           | primary   | 1000      | 803                           | 4                | magicc_v5,magicc_v4,magicc_v3,checkm2,cocopye,deepcheck |                     | {"test": 1000}                         |                                   |
| set_C               | superseded_leaky  | reported  | 1000      | 1000                          | 1                | magicc_v5,magicc_v4,magicc_v3,checkm2,cocopye,deepcheck |                     | {"train": 1000}                        | SUPERSEDED - training-set leakage |
| set_D               | superseded_leaky  | reported  | 1000      | 1000                          | 1                | magicc_v5,magicc_v4,magicc_v3,checkm2,cocopye,deepcheck |                     | {"train": 796, "val": 107, "test": 97} | SUPERSEDED - training-set leakage |
| set_E               | current           | primary   | 1000      | 785                           | 4                | magicc_v5,magicc_v4,magicc_v3,checkm2,cocopye,deepcheck |                     | {"test": 1000}                         |                                   |
| set_C_clean         | current           | primary   | 1000      | 100                           | 10               | magicc_v5,checkm2,cocopye,deepcheck                     | magicc_v4,magicc_v3 | {"test": 1000}                         |                                   |
| set_D_clean         | current           | primary   | 1000      | 100                           | 10               | magicc_v5,checkm2,cocopye,deepcheck                     | magicc_v4,magicc_v3 | {"test": 1000}                         |                                   |
| set_A               | legacy_superseded | secondary | 600       | 582                           | 2                | magicc_v5,magicc_v3,checkm2,cocopye,deepcheck           | magicc_v4           | {"test": 600}                          |                                   |
| set_B               | legacy_superseded | secondary | 600       | 588                           | 3                | magicc_v5,magicc_v3,checkm2,cocopye,deepcheck           | magicc_v4           | {"test": 600}                          |                                   |
| motivating_v2_set_A | motivating        | secondary | 1000      | 806                           | 3                | magicc_v5,checkm2,cocopye,deepcheck                     | magicc_v4,magicc_v3 | {"test": 1000}                         |                                   |
| motivating_v2_set_B | motivating        | secondary | 1000      | 798                           | 4                | magicc_v5,checkm2,cocopye,deepcheck                     | magicc_v4,magicc_v3 | {"test": 1000}                         |                                   |
| motivating_v2_set_C | motivating        | secondary | 1000      | 773                           | 5                | magicc_v5,checkm2,cocopye,deepcheck                     | magicc_v4,magicc_v3 | {"test": 1000}                         |                                   |
| motivating_set_A    | legacy_superseded | secondary | 600       | 585                           | 2                | magicc_v5,checkm2,cocopye,deepcheck                     | magicc_v4,magicc_v3 | {"test": 600}                          |                                   |
| motivating_set_B    | legacy_superseded | secondary | 1100      | 1031                          | 3                | magicc_v5,checkm2,cocopye,deepcheck                     | magicc_v4,magicc_v3 | {"test": 1100}                         |                                   |

## Files

| set                 | tool      | file                      | present | n_rows | n_cols | n_matched_to_metadata | n_nan_predictions | note                                                          |
|---------------------|-----------|---------------------------|---------|--------|--------|-----------------------|-------------------|---------------------------------------------------------------|
| set_A_v2            |           | metadata.tsv              | True    | 1000   | 9      |                       |                   |                                                               |
| set_A_v2            |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| set_A_v2            | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A_v2            | magicc_v4 | magicc_v4_predictions.tsv | True    | 1000   | 3      | 1000                  | 0                 | no embedded truth columns - joined on genome id               |
| set_A_v2            | magicc_v3 | magicc_predictions.tsv    | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A_v2            | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A_v2            | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A_v2            | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_B_v2            |           | metadata.tsv              | True    | 1000   | 9      |                       |                   |                                                               |
| set_B_v2            |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| set_B_v2            | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_B_v2            | magicc_v4 | magicc_v4_predictions.tsv | True    | 1000   | 3      | 1000                  | 0                 | no embedded truth columns - joined on genome id               |
| set_B_v2            | magicc_v3 | magicc_predictions.tsv    | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_B_v2            | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_B_v2            | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_B_v2            | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_C               |           | metadata.tsv              | True    | 1000   | 8      |                       |                   |                                                               |
| set_C               |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| set_C               | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_C               | magicc_v4 | magicc_v4_predictions.tsv | True    | 1000   | 3      | 1000                  | 0                 | no embedded truth columns - joined on genome id               |
| set_C               | magicc_v3 | magicc_predictions.tsv    | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_C               | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_C               | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_C               | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_D               |           | metadata.tsv              | True    | 1000   | 8      |                       |                   |                                                               |
| set_D               |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| set_D               | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_D               | magicc_v4 | magicc_v4_predictions.tsv | True    | 1000   | 3      | 1000                  | 0                 | no embedded truth columns - joined on genome id               |
| set_D               | magicc_v3 | magicc_predictions.tsv    | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_D               | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_D               | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_D               | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_E               |           | metadata.tsv              | True    | 1000   | 11     |                       |                   |                                                               |
| set_E               |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| set_E               | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_E               | magicc_v4 | magicc_v4_predictions.tsv | True    | 1000   | 3      | 1000                  | 0                 | no embedded truth columns - joined on genome id               |
| set_E               | magicc_v3 | magicc_predictions.tsv    | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_E               | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_E               | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_E               | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_C_clean         |           | metadata.tsv              | True    | 1000   | 10     |                       |                   |                                                               |
| set_C_clean         |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| set_C_clean         | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_C_clean         | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| set_C_clean         | magicc_v3 | magicc_predictions.tsv    | False   |        |        |                       | 0                 | file absent                                                   |
| set_C_clean         | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 16     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_C_clean         | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 17     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_C_clean         | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_D_clean         |           | metadata.tsv              | True    | 1000   | 10     |                       |                   |                                                               |
| set_D_clean         |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.80e-06                             |
| set_D_clean         | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_D_clean         | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| set_D_clean         | magicc_v3 | magicc_predictions.tsv    | False   |        |        |                       | 0                 | file absent                                                   |
| set_D_clean         | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 16     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_D_clean         | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 17     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_D_clean         | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A               |           | metadata.tsv              | True    | 600    | 8      |                       |                   |                                                               |
| set_A               |           | labels.npy                | True    | 600    | 2      |                       |                   | max\|labels-metadata\| = 3.78e-06                             |
| set_A               | magicc_v5 | magicc_v5_predictions.tsv | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A               | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| set_A               | magicc_v3 | magicc_predictions.tsv    | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A               | checkm2   | checkm2_predictions.tsv   | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A               | cocopye   | cocopye_predictions.tsv   | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_A               | deepcheck | deepcheck_predictions.tsv | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| set_B               |           | metadata.tsv              | True    | 600    | 8      |                       |                   |                                                               |
| set_B               |           | labels.npy                | True    | 600    | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| set_B               | magicc_v5 | magicc_v5_predictions.tsv | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_B               | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| set_B               | magicc_v3 | magicc_predictions.tsv    | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_B               | checkm2   | checkm2_predictions.tsv   | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_B               | cocopye   | cocopye_predictions.tsv   | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| set_B               | deepcheck | deepcheck_predictions.tsv | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_v2_set_A |           | metadata.tsv              | True    | 1000   | 9      |                       |                   |                                                               |
| motivating_v2_set_A |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| motivating_v2_set_A | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| motivating_v2_set_A | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_v2_set_A | magicc_v3 | magicc_predictions.tsv    | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_v2_set_A | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| motivating_v2_set_A | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| motivating_v2_set_A | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| motivating_v2_set_B |           | metadata.tsv              | True    | 1000   | 9      |                       |                   |                                                               |
| motivating_v2_set_B |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| motivating_v2_set_B | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_v2_set_B | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_v2_set_B | magicc_v3 | magicc_predictions.tsv    | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_v2_set_B | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 12     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_v2_set_B | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_v2_set_B | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 13     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_v2_set_C |           | metadata.tsv              | True    | 1000   | 11     |                       |                   |                                                               |
| motivating_v2_set_C |           | labels.npy                | True    | 1000   | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| motivating_v2_set_C | magicc_v5 | magicc_v5_predictions.tsv | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_v2_set_C | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_v2_set_C | magicc_v3 | magicc_predictions.tsv    | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_v2_set_C | checkm2   | checkm2_predictions.tsv   | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_v2_set_C | cocopye   | cocopye_predictions.tsv   | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_v2_set_C | deepcheck | deepcheck_predictions.tsv | True    | 1000   | 15     | 1000                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_set_A    |           | metadata.tsv              | True    | 600    | 8      |                       |                   |                                                               |
| motivating_set_A    |           | labels.npy                | True    | 600    | 2      |                       |                   | max\|labels-metadata\| = 3.81e-06                             |
| motivating_set_A    | magicc_v5 | magicc_v5_predictions.tsv | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| motivating_set_A    | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_set_A    | magicc_v3 | magicc_predictions.tsv    | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_set_A    | checkm2   | checkm2_predictions.tsv   | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| motivating_set_A    | cocopye   | cocopye_predictions.tsv   | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| motivating_set_A    | deepcheck | deepcheck_predictions.tsv | True    | 600    | 12     | 600                   | 0                 | embedded truth agrees with metadata (max \|diff\| = 0.00e+00) |
| motivating_set_B    |           | metadata.tsv              | True    | 1100   | 8      |                       |                   |                                                               |
| motivating_set_B    |           | labels.npy                | True    | 1100   | 2      |                       |                   | max\|labels-metadata\| = 3.80e-06                             |
| motivating_set_B    | magicc_v5 | magicc_v5_predictions.tsv | True    | 1100   | 12     | 1100                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_set_B    | magicc_v4 | magicc_v4_predictions.tsv | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_set_B    | magicc_v3 | magicc_predictions.tsv    | False   |        |        |                       | 0                 | file absent                                                   |
| motivating_set_B    | checkm2   | checkm2_predictions.tsv   | True    | 1100   | 12     | 1100                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_set_B    | cocopye   | cocopye_predictions.tsv   | True    | 1100   | 12     | 1100                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
| motivating_set_B    | deepcheck | deepcheck_predictions.tsv | True    | 1100   | 12     | 1100                  | 0                 | embedded truth agrees with metadata (max \|diff\| = 3.55e-15) |
