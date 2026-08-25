# Train / validation / test splits (WS7.6)

| Split | accessions selected | genomes available | download failures |
|---|---|---|---|
| train | 79,978 | 79,948 | 30 |
| val | 10,017 | 10,010 | 7 |
| test | 10,005 | 9,999 | 6 |

The splits are stratified by phylum and **mutually disjoint** (verified here on
both the selected-accession sets and the available-genome sets).

`*_accessions.txt` lists NCBI GenBank accessions (`GCA_x.y`). The full per-genome tables
(`*_genomes.tsv`, 35 MB, deposited at figshare) are keyed by GTDB accession (`GB_GCA_x.y` /
`RS_GCF_x.y`) and carry the NCBI accession in column `ncbi_accession`; that is
the join key. Naive string matching between the two conventions undercounts
overlap and is the exact mistake that hid the Set C/D leakage originally --
see `../provenance/README.md`.

43 selected accessions (43 across the three splits) failed to
download and are listed in `missing_accessions.txt`. They have no row in
`*_genomes.tsv` and were never used for training, feature selection or
benchmarking. The counts used throughout the manuscript are the
**available-genome** counts: train 79,948, val 10,010, test 9,999.
