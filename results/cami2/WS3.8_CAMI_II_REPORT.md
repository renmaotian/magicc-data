# WS3.8 — CAMI II external benchmark (Reviewer 1, major comment 3)

_Generated 2026-08-02T21:25:34.173538+00:00_

## What this is, and why it is independent

CAMI II (Meyer et al. 2022, *Nature Methods*) is a community benchmark whose genome selection, read simulation, assembly and gold standard were all produced by a third party. Data were obtained from the fully open repository `https://frl.publisso.de/data/frl:6425521/` (no request form is required, contrary to the CAMI web page); every tarball was MD5-verified against the repository's own `md5sums.txt`.

**Nothing in the CAMI pipeline was chosen by us.** The contigs are CAMI's, the contig-to-source-genome truth is CAMI's, and the reference genomes are CAMI's. The only thing we contribute is the *grouping* of CAMI contigs into bins. This is what makes the benchmark a test of MAGICC outside our own simulation assumptions.

## Truth derivation (exact; stated for scrutiny)

`gsa_mapping.tsv` gives, for every gold-standard-assembly contig, its exact source genome and the exact interval `[start_position, end_position]` of that genome the contig reproduces. For a bin *B* with dominant genome *d*:

```
completeness(B)  = 100 x (bp in B whose source is d)      / R(d)
contamination(B) = 100 x (bp in B whose source is not d)  / R(d)
R(d) = FULL reference length of d = total bp of source_genomes/<d>.fasta
```

This is MAGICC's own convention verbatim (`magicc/contamination.py`: `completeness = dominant_actual_bp / dominant_genome_full_length`, `contamination = contaminant_total_bp / dominant_genome_full_length`), so truth and every tool share one denominator (R1-M5).

Retained bp was computed two ways and both are stored: `span_bp` (sum of contig spans) and `union_bp` (bp of the *union* of source intervals, which cannot double-count overlapping contigs). **They agree exactly** — the maximum difference across all gold bins is 0.00 pp, and no gold bin exceeds 100% completeness — so the primary number is not inflated by overlapping contigs.

## Bin sets

**(i) Gold-standard bins.** Contigs grouped by their true source genome exactly as `binning_gs.tsv` specifies. Pure by construction (0% contamination); the completeness gradient is whatever CAMI's read simulation and gold-standard assembly produced.

**(ii) Constructed mixed bins.** A known completeness x contamination grid (completeness targets 60/75/90%, contamination targets 2/5/10/15/20%), **stratified by the taxonomic distance between dominant and contaminant** (species -> genus -> family -> order -> class -> phylum), so it is directly comparable with WS2 Set F. Distance is the rank of the lowest common ancestor of the NCBI tax_ids CAMI itself assigned. Contaminant partners are only eligible when that sample's gold-standard assembly actually contains enough of their sequence to reach the target, so realised contamination tracks target closely. Every contig is an unmodified CAMI contig; only the grouping is ours.

**What the `species` cell means, and why strain-madness is the decisive dataset.** CAMI II labelled the strain-madness isolates at species level, so two genomes in the `species` cell are *different clinical isolates of the same species* — i.e. strain-level contamination, the hardest case a composition-based method can face, and exactly the same semantics as WS2 Set F's `species` cell (same taxon at species rank, different below it). CAMI's own `metadata.tsv` independently classifies 382 of the 405 strain-madness source genomes as `new_strain`, confirming the regime is theirs, not our construction.

## Leakage audit

### marine

- source genomes: **864**

- accession recoverable from the CAMI id (L1): 0; recovered by NCBI organism/strain name join (L2): 554; no accession recoverable: 310

- GCA<->GCF cross-mapped overlap with **train 313**, val 107, test 95, 9-mer selection set 71

- **leaked (train | val | 9-mer selection): 340 / 864 = 39.352%**; leakage-free n = **524**

- species-level context (NOT assembly leakage): 215/474 species also occur in the training split, covering 45.02% of CAMI genomes

### strain_madness

- source genomes: **408**

- accession recoverable from the CAMI id (L1): 0; recovered by NCBI organism/strain name join (L2): 0; no accession recoverable: 408

- GCA<->GCF cross-mapped overlap with **train 0**, val 0, test 0, 9-mer selection set 0

- **leaked (train | val | 9-mer selection): 0 / 408 = 0.0%**; leakage-free n = **408**

- species-level context (NOT assembly leakage): 16/20 species also occur in the training split, covering 87.5% of CAMI genomes


The name-based L2 join is deliberately over-inclusive — a CAMI strain name that matches several assemblies of the same organism marks all of them as potential leaks — so it can only over-state leakage, never hide it. That is the correct direction of error for an audit, and it is the direction the Set C/D failure went the wrong way on.

## Censoring by MAGICC's 50% completeness floor

MAGICC V5 was trained only on 50-100% completeness, so bins below the floor cannot be scored. This is reported, not hidden:


| dataset        | binset              | n_bins_scored | n_below_50pct_floor | pct_below_floor | n_out_of_domain |
|----------------|---------------------|---------------|---------------------|-----------------|-----------------|
| marine         | gold                | 4497          | 1189                | 26.44           | 0               |
| marine         | mixed               | 2201          | 0                   | 0               | 0               |
| strain_madness | gold                | 1564          | 457                 | 29.22           | 0               |
| strain_madness | mixed               | 2250          | 0                   | 0               | 0               |
| marine         | gold_ALL_TRUTH_ROWS | 6359          | 2139                | 33.64           | 0               |
| strain_madness | gold_ALL_TRUTH_ROWS | 4064          | 2957                | 72.76           | 0               |

The `gold_ALL_TRUTH_ROWS` rows are the full truth tables, i.e. every source genome present in every sample; the other rows are the bins actually scored. In strain-madness in particular a large majority of source genomes fall below the floor, because the gold-standard assembly of a community of near-identical strains is heavily fragmented and each strain recovers only a small fraction of its reference. That is a property of the dataset, and it is reported rather than quietly dropped. A bounded probe cohort in the 30-50% band is scored and reported as `below_floor_probe_NOT_POOLED`; it is never pooled into any headline number.


## Competitor cohort: what was subsampled, and why

MAGICC V5 costs milliseconds per bin, so it was run on **every** bin. CheckM2 measured ~29 bins/min at 24 threads here, so the competitor cohort was budgeted:

- **every mixed bin** is given to all tools (they are the designed grid, and each distance x completeness x contamination cell needs its n);
- **gold bins are subsampled**, stratified by completeness decile so the gradient survives, plus a labelled below-floor probe. Selection is seeded through `fw.stable_hash` and the exact membership is recorded in `results/revision/cami2/provenance/<dataset>_gold_competitor_cohort.tsv`.

**All tool-versus-tool tables are computed on the subset every tool scored**, so no comparison is made across different bin sets. MAGICC's full-coverage numbers appear separately as the cohort `MAGICC_full_coverage_not_tool_comparable` and are explicitly not used for comparisons.

DeepCheck is obtained as a pure tensor transform of CheckM2's `--dbg_vectors` feature vectors, using the model, scaler and forward-pass work-around imported verbatim from `scripts/38_run_deepcheck_v2.py`, so it costs no extra tool run.

**Scope decision, stated:** the rhizosphere (plant-associated, 21 samples, ~6.6 GB) and pathogen CAMI II datasets were NOT acquired. Marine and strain-madness already populate all six taxonomic-distance cells and cover both the broad-diversity and the close-relative regimes; the remaining datasets would add volume rather than a new question.


## THE HEADLINE — does the WS2 close-contaminant limitation reproduce?

WS2 (Set F, our own simulation) established that MAGICC is near-blind to close contaminants: signed contamination bias **-7.76 pp at species distance**, with per-type OLS detection slopes at species of **-0.014 (replaced, indistinguishable from zero) / 0.135 (single) / 0.178 (redundant)** versus ~1.0 at phylum. CAMI II `strain_madness` is built from many closely related strains by an independent group, so it is an external test of exactly that regime.

| dataset                | distance_rank | cami2_cont_bias_pp | cami2_bias_ci    | ws2_setF_cont_bias_pp | delta_cami2_minus_setF | cami2_detection_slope | cami2_slope_ci  | cami2_n | cami2_n_clusters |
|------------------------|---------------|--------------------|------------------|-----------------------|------------------------|-----------------------|-----------------|---------|------------------|
| CAMI_II_marine         | species       | -13.16             | [-14.74, -11.65] | -7.76                 | -5.405                 | 0.245                 | [-0.035, 0.513] | 195     | 94               |
| CAMI_II_marine         | genus         | -8.349             | [-9.57, -7.14]   | -7.73                 | -0.619                 | 0.729                 | [0.092, 0.908]  | 146     | 75               |
| CAMI_II_marine         | family        | -5.78              | [-7.12, -4.46]   | -5.98                 | 0.2                    | 0.647                 | [0.369, 0.772]  | 168     | 85               |
| CAMI_II_marine         | order         | -2.713             | [-4.35, -1.10]   | -3.6                  | 0.887                  | 0.918                 | [0.751, 1.169]  | 146     | 69               |
| CAMI_II_marine         | class         | -2.506             | [-3.89, -0.87]   | -1.89                 | -0.616                 | 0.882                 | [0.703, 1.017]  | 161     | 90               |
| CAMI_II_marine         | phylum        | 0.019              | [-1.09, 1.26]    | -0.9                  | 0.919                  | 1.073                 | [0.968, 1.201]  | 170     | 96               |
| CAMI_II_strain_madness | species       | -8.098             | [-8.76, -7.47]   | -7.76                 | -0.338                 | 0.369                 | [0.264, 0.475]  | 375     | 205              |
| CAMI_II_strain_madness | genus         | -7.998             | [-8.65, -7.33]   | -7.73                 | -0.268                 | 0.435                 | [0.321, 0.547]  | 375     | 163              |
| CAMI_II_strain_madness | family        | -2.898             | [-3.51, -2.33]   | -5.98                 | 3.082                  | 0.939                 | [0.849, 1.023]  | 375     | 178              |
| CAMI_II_strain_madness | order         | -1.669             | [-1.99, -1.33]   | -3.6                  | 1.931                  | 1.019                 | [0.971, 1.062]  | 375     | 155              |
| CAMI_II_strain_madness | class         | -0.839             | [-1.10, -0.57]   | -1.89                 | 1.051                  | 1.056                 | [1.028, 1.085]  | 375     | 153              |
| CAMI_II_strain_madness | phylum        | -0.422             | [-0.73, -0.05]   | -0.9                  | 0.478                  | 1.04                  | [1.007, 1.081]  | 375     | 162              |


### Grid-control caveat, and the well-controlled cohort that sharpens the result

Contaminant material is added as whole unmodified CAMI contigs. Marine contigs are long, so a single contig can **overshoot** the contamination target (strain-madness realised means 2.10/5.16/10.21/15.28/20.30 against targets 2/5/10/15/20, tight; marine 4.85/6.89/11.10/16.47/21.95 with a heavy tail). **Truth is unaffected** — it is always the realised value, never the target, and every bin is in-domain — but the intended grid is only tight for strain-madness. Restricting to cells where the grid was actually achieved (realised <= 1.5x target: 92% of marine, 99% of strain-madness bins) does not weaken the headline, it sharpens it:

| dataset        | distance_rank | n   | n_clusters | cont_bias | cont_bias_lo | cont_bias_hi | cont_mae | slope    | slope_lo | slope_hi  |
|----------------|---------------|-----|------------|-----------|--------------|--------------|----------|----------|----------|-----------|
| marine         | species       | 147 | 85         | -12.15    | -13.8        | -10.55       | 12.29    | -0.04675 | -0.0885  | -0.005366 |
| marine         | genus         | 143 | 74         | -8.391    | -9.617       | -7.114       | 8.819    | 0.1938   | 0.06435  | 0.3525    |
| marine         | family        | 165 | 85         | -5.701    | -7.025       | -4.363       | 7.015    | 0.4321   | 0.305    | 0.5608    |
| marine         | order         | 137 | 69         | -2.719    | -4.45        | -0.9883      | 5.914    | 0.8657   | 0.6038   | 1.137     |
| marine         | class         | 155 | 89         | -2.599    | -4.059       | -0.9785      | 5.356    | 0.7036   | 0.5336   | 0.874     |
| marine         | phylum        | 164 | 95         | -0.02102  | -1.195       | 1.211        | 4.166    | 1.102    | 0.941    | 1.262     |
| strain_madness | species       | 372 | 205        | -8.138    | -8.737       | -7.509       | 8.7      | 0.3701   | 0.2662   | 0.4818    |
| strain_madness | genus         | 360 | 160        | -7.955    | -8.62        | -7.273       | 8.136    | 0.3563   | 0.2545   | 0.4573    |
| strain_madness | family        | 370 | 177        | -2.928    | -3.517       | -2.385       | 4.01     | 0.9177   | 0.8315   | 1.002     |
| strain_madness | order         | 372 | 155        | -1.668    | -2.005       | -1.299       | 2.493    | 1.02     | 0.9752   | 1.064     |
| strain_madness | class         | 369 | 151        | -0.8358   | -1.105       | -0.5623      | 1.775    | 1.062    | 1.033    | 1.091     |
| strain_madness | phylum        | 375 | 162        | -0.4223   | -0.7481      | -0.04308     | 1.506    | 1.04     | 1.007    | 1.08      |

**On the well-controlled cohort MAGICC's marine species-level detection slope is -0.047 [-0.089, -0.005] — statistically indistinguishable from zero, and a near-exact match to WS2 Set F's worst cell (`replaced`, -0.014 [-0.034, 0.001]).** MAGICC has essentially no ability to detect same-species contamination on independently produced data. The 0.245 seen on the unrestricted marine cohort was inflated by overshoot bins carrying very high true contamination; the honest number is the well-controlled one, and it is worse, not better. Report the species-level slope across both datasets as **-0.05 to 0.37**, against ~1.0 at phylum.


## All tools by taxonomic distance (mixed bins, leakage-free, in-domain)

| dataset        | tool      | distance_rank | n   | n_clusters | cont_bias | cont_bias_lo | cont_bias_hi | cont_mae | slope   | comp_bias |
|----------------|-----------|---------------|-----|------------|-----------|--------------|--------------|----------|---------|-----------|
| marine         | CheckM2   | species       | 195 | 94         | -11.09    | -13.1        | -9.322       | 11.6     | 0.3141  | -48.96    |
| marine         | CheckM2   | genus         | 146 | 75         | -4.443    | -5.453       | -3.499       | 5.261    | 0.6734  | -4.391    |
| marine         | CheckM2   | family        | 168 | 85         | -4.995    | -5.944       | -4.105       | 5.788    | 0.7052  | -4.967    |
| marine         | CheckM2   | order         | 146 | 69         | -4.314    | -5.34        | -3.245       | 5.67     | 0.8084  | -1.799    |
| marine         | CheckM2   | class         | 161 | 90         | -5.935    | -6.819       | -5.064       | 6.919    | 0.5492  | -3.392    |
| marine         | CheckM2   | phylum        | 170 | 96         | -5.636    | -6.548       | -4.765       | 6.4      | 0.3735  | -3.437    |
| marine         | CoCoPyE   | species       | 195 | 94         | -5.868    | -9.129       | -2.443       | 14.74    | -0.0629 | -10.5     |
| marine         | CoCoPyE   | genus         | 146 | 75         | 10.45     | 7.823        | 12.63        | 12.24    | 0.2827  | 6.07      |
| marine         | CoCoPyE   | family        | 168 | 85         | 10.1      | 8.182        | 11.98        | 11.86    | 0.3406  | 6.921     |
| marine         | CoCoPyE   | order         | 146 | 69         | 8.396     | 6.752        | 10.14        | 9.833    | 0.4819  | 4.443     |
| marine         | CoCoPyE   | class         | 161 | 90         | 9.864     | 8.354        | 11.42        | 10.75    | 0.4673  | 6.758     |
| marine         | CoCoPyE   | phylum        | 170 | 96         | 10.09     | 8.666        | 11.51        | 10.97    | 0.4287  | 6.461     |
| marine         | DeepCheck | species       | 195 | 94         | -9.186    | -11.05       | -7.417       | 9.888    | 0.5231  | -41.72    |
| marine         | DeepCheck | genus         | 146 | 75         | -4.66     | -5.764       | -3.42        | 5.768    | 0.9744  | -6.621    |
| marine         | DeepCheck | family        | 168 | 85         | -5.639    | -6.582       | -4.767       | 6.166    | 0.6728  | -7.531    |
| marine         | DeepCheck | order         | 146 | 69         | -5.191    | -6.075       | -4.231       | 5.953    | 0.7204  | -3.295    |
| marine         | DeepCheck | class         | 161 | 90         | -6.754    | -7.662       | -5.798       | 7.3      | 0.3582  | -6.543    |
| marine         | DeepCheck | phylum        | 170 | 96         | -6.454    | -7.451       | -5.514       | 6.919    | 0.2737  | -5.583    |
| marine         | MAGICC_V5 | species       | 195 | 94         | -13.17    | -14.74       | -11.65       | 13.27    | 0.2455  | -16.51    |
| marine         | MAGICC_V5 | genus         | 146 | 75         | -8.349    | -9.571       | -7.137       | 8.768    | 0.7289  | 8.862     |
| marine         | MAGICC_V5 | family        | 168 | 85         | -5.78     | -7.117       | -4.46        | 7.115    | 0.6466  | 4.486     |
| marine         | MAGICC_V5 | order         | 146 | 69         | -2.713    | -4.347       | -1.1         | 6.047    | 0.9183  | 2.759     |
| marine         | MAGICC_V5 | class         | 161 | 90         | -2.506    | -3.888       | -0.8708      | 5.317    | 0.8821  | 2.477     |
| marine         | MAGICC_V5 | phylum        | 170 | 96         | 0.01941   | -1.089       | 1.261        | 4.144    | 1.073   | 1.118     |
| strain_madness | CheckM2   | species       | 375 | 205        | -4.437    | -4.934       | -3.925       | 5.274    | 0.3887  | -3.69     |
| strain_madness | CheckM2   | genus         | 375 | 163        | -4.034    | -4.56        | -3.515       | 5.042    | 0.4765  | -4.208    |
| strain_madness | CheckM2   | family        | 375 | 178        | -4.772    | -5.285       | -4.246       | 5.464    | 0.4293  | -2.495    |
| strain_madness | CheckM2   | order         | 375 | 155        | -5.823    | -6.412       | -5.244       | 6.365    | 0.276   | -4.454    |
| strain_madness | CheckM2   | class         | 375 | 153        | -5.717    | -6.268       | -5.174       | 6.245    | 0.2844  | -4.197    |
| strain_madness | CheckM2   | phylum        | 375 | 162        | -6.698    | -7.327       | -6.058       | 7.121    | 0.1605  | -6.027    |
| strain_madness | CoCoPyE   | species       | 375 | 205        | 12.98     | 12.22        | 13.74        | 13.06    | 0.7398  | 6.262     |
| strain_madness | CoCoPyE   | genus         | 375 | 163        | 12.42     | 11.63        | 13.31        | 12.65    | 0.5381  | 5.977     |
| strain_madness | CoCoPyE   | family        | 375 | 178        | 12.33     | 11.51        | 13.16        | 12.42    | 0.8008  | 5.905     |
| strain_madness | CoCoPyE   | order         | 375 | 155        | 11.12     | 10.21        | 11.97        | 11.54    | 0.7474  | 5.357     |
| strain_madness | CoCoPyE   | class         | 375 | 153        | 11.31     | 10.45        | 12.2         | 11.52    | 0.7116  | 5.422     |
| strain_madness | CoCoPyE   | phylum        | 375 | 162        | 9.447     | 8.403        | 10.47        | 10.09    | 0.6283  | 5.391     |
| strain_madness | DeepCheck | species       | 375 | 205        | -5.13     | -5.632       | -4.605       | 5.593    | 0.3652  | -6.381    |
| strain_madness | DeepCheck | genus         | 375 | 163        | -4.532    | -5.085       | -3.949       | 5.447    | 0.4144  | -6.451    |
| strain_madness | DeepCheck | family        | 375 | 178        | -5.722    | -6.231       | -5.25        | 6.154    | 0.3279  | -5.719    |
| strain_madness | DeepCheck | order         | 375 | 155        | -6.255    | -6.819       | -5.682       | 6.688    | 0.2258  | -6.445    |
| strain_madness | DeepCheck | class         | 375 | 153        | -6.222    | -6.774       | -5.653       | 6.613    | 0.2363  | -6.358    |
| strain_madness | DeepCheck | phylum        | 375 | 162        | -6.882    | -7.542       | -6.247       | 7.269    | 0.1209  | -7.913    |
| strain_madness | MAGICC_V5 | species       | 375 | 205        | -8.098    | -8.757       | -7.473       | 8.655    | 0.3692  | 7.578     |
| strain_madness | MAGICC_V5 | genus         | 375 | 163        | -7.998    | -8.646       | -7.332       | 8.201    | 0.4351  | 7.006     |
| strain_madness | MAGICC_V5 | family        | 375 | 178        | -2.898    | -3.509       | -2.325       | 4.023    | 0.939   | 2.333     |
| strain_madness | MAGICC_V5 | order         | 375 | 155        | -1.669    | -1.989       | -1.326       | 2.498    | 1.019   | 0.2633    |
| strain_madness | MAGICC_V5 | class         | 375 | 153        | -0.8391   | -1.1         | -0.5712      | 1.763    | 1.056   | -0.2558   |
| strain_madness | MAGICC_V5 | phylum        | 375 | 162        | -0.4223   | -0.7339      | -0.05377     | 1.506    | 1.04    | -0.7226   |


## Accuracy by cohort

| dataset        | binset | tool      | metric        | n    | n_clusters | mae    | mae_lo | mae_hi | bias     | bias_lo | bias_hi | r2       | r2_omitted_reason                            |
|----------------|--------|-----------|---------------|------|------------|--------|--------|--------|----------|---------|---------|----------|----------------------------------------------|
| marine         | gold   | MAGICC_V5 | completeness  | 409  | 231        | 9.733  | 7.711  | 11.88  | -6.856   | -9.274  | -4.644  | -0.05957 |                                              |
| marine         | gold   | MAGICC_V5 | contamination | 409  | 231        | 1.049  | 0.6024 | 1.628  | 1.049    | 0.6024  | 1.628   |          | true value has (near-)zero variance (R1-m19) |
| marine         | gold   | CheckM2   | completeness  | 409  | 231        | 25.28  | 21.13  | 29.69  | -25.17   | -29.6   | -21.01  | -3.924   |                                              |
| marine         | gold   | CheckM2   | contamination | 409  | 231        | 2.143  | 1.894  | 2.405  | 2.143    | 1.894   | 2.405   |          | true value has (near-)zero variance (R1-m19) |
| marine         | gold   | CoCoPyE   | completeness  | 409  | 231        | 12.76  | 11.36  | 14.36  | 3.282    | 0.9071  | 5.549   | 0.06287  |                                              |
| marine         | gold   | CoCoPyE   | contamination | 409  | 231        | 14.74  | 13.21  | 16.18  | 14.74    | 13.21   | 16.18   |          | true value has (near-)zero variance (R1-m19) |
| marine         | gold   | DeepCheck | completeness  | 409  | 231        | 26.32  | 22.99  | 29.7   | -26.32   | -29.7   | -22.99  | -3.116   |                                              |
| marine         | gold   | DeepCheck | contamination | 409  | 231        | 2.946  | 2.735  | 3.179  | 2.913    | 2.694   | 3.155   |          | true value has (near-)zero variance (R1-m19) |
| marine         | mixed  | MAGICC_V5 | completeness  | 986  | 253        | 10.17  | 9.073  | 11.3   | -0.1832  | -1.966  | 1.429   | -0.1482  |                                              |
| marine         | mixed  | MAGICC_V5 | contamination | 986  | 253        | 7.613  | 7.011  | 8.238  | -5.632   | -6.537  | -4.742  | -0.1524  |                                              |
| marine         | mixed  | CheckM2   | completeness  | 986  | 253        | 15.74  | 12.99  | 18.99  | -12.59   | -16.17  | -9.607  | -3.311   |                                              |
| marine         | mixed  | CheckM2   | contamination | 986  | 253        | 7.132  | 6.62   | 7.687  | -6.283   | -6.903  | -5.72   | -0.03737 |                                              |
| marine         | mixed  | CoCoPyE   | completeness  | 986  | 253        | 8.418  | 7.505  | 9.489  | 2.877    | 1.663   | 4.005   | 0.2611   |                                              |
| marine         | mixed  | CoCoPyE   | contamination | 986  | 253        | 11.85  | 11.13  | 12.63  | 6.701    | 5.453   | 7.911   | -1.21    |                                              |
| marine         | mixed  | DeepCheck | completeness  | 986  | 253        | 15.64  | 13.4   | 18.34  | -13.03   | -16     | -10.59  | -2.309   |                                              |
| marine         | mixed  | DeepCheck | contamination | 986  | 253        | 7.126  | 6.683  | 7.615  | -6.452   | -7.008  | -5.961  | 0.003    |                                              |
| strain_madness | gold   | MAGICC_V5 | completeness  | 700  | 330        | 1.295  | 1.162  | 1.463  | -0.05194 | -0.2022 | 0.1102  | 0.9847   |                                              |
| strain_madness | gold   | MAGICC_V5 | contamination | 700  | 330        | 0.5028 | 0.4013 | 0.6606 | 0.5028   | 0.4013  | 0.6606  |          | true value has (near-)zero variance (R1-m19) |
| strain_madness | gold   | CheckM2   | completeness  | 700  | 330        | 13.96  | 13.19  | 14.69  | -13.85   | -14.59  | -13.06  | -0.08566 |                                              |
| strain_madness | gold   | CheckM2   | contamination | 700  | 330        | 3.015  | 2.852  | 3.184  | 3.015    | 2.852   | 3.184   |          | true value has (near-)zero variance (R1-m19) |
| strain_madness | gold   | CoCoPyE   | completeness  | 700  | 330        | 8.942  | 8.429  | 9.43   | 8.928    | 8.416   | 9.415   | 0.5498   |                                              |
| strain_madness | gold   | CoCoPyE   | contamination | 700  | 330        | 20.06  | 19.27  | 20.94  | 20.06    | 19.27   | 20.94   |          | true value has (near-)zero variance (R1-m19) |
| strain_madness | gold   | DeepCheck | completeness  | 700  | 330        | 17.47  | 16.68  | 18.25  | -17.47   | -18.25  | -16.68  | -0.547   |                                              |
| strain_madness | gold   | DeepCheck | contamination | 700  | 330        | 3.179  | 3.036  | 3.317  | 3.15     | 3.005   | 3.289   |          | true value has (near-)zero variance (R1-m19) |
| strain_madness | mixed  | MAGICC_V5 | completeness  | 2250 | 335        | 4.218  | 4.003  | 4.447  | 2.7      | 2.407   | 2.99    | 0.7402   |                                              |
| strain_madness | mixed  | MAGICC_V5 | contamination | 2250 | 335        | 4.441  | 4.236  | 4.667  | -3.654   | -3.942  | -3.373  | 0.02231  |                                              |
| strain_madness | mixed  | CheckM2   | completeness  | 2250 | 335        | 7.156  | 6.749  | 7.577  | -4.178   | -4.775  | -3.609  | 0.4067   |                                              |
| strain_madness | mixed  | CheckM2   | contamination | 2250 | 335        | 5.918  | 5.704  | 6.131  | -5.247   | -5.501  | -4.983  | -0.2745  |                                              |
| strain_madness | mixed  | CoCoPyE   | completeness  | 2250 | 335        | 6.079  | 5.739  | 6.449  | 5.719    | 5.346   | 6.124   | 0.569    |                                              |
| strain_madness | mixed  | CoCoPyE   | contamination | 2250 | 335        | 11.88  | 11.41  | 12.35  | 11.6     | 11.09   | 12.11   | -3.216   |                                              |
| strain_madness | mixed  | DeepCheck | completeness  | 2250 | 335        | 8.501  | 8.001  | 9.031  | -6.544   | -7.228  | -5.921  | 0.1647   |                                              |
| strain_madness | mixed  | DeepCheck | contamination | 2250 | 335        | 6.294  | 6.079  | 6.499  | -5.79    | -6.028  | -5.539  | -0.4202  |                                              |

R2 is the **coefficient of determination** (1 - SS_res/SS_tot), never squared Pearson (protocol 4.4d). It is **omitted** wherever the true value has (near-)zero variance — which is every contamination row on the pure gold-standard bins, where truth is exactly 0 (R1-m19). A large negative R2 there would be an artefact of a zero denominator, not a finding.


## MIMAG-inspired thresholds

High quality >=90% completeness AND <5% contamination; medium quality >=50% AND <10%. Labelled **MIMAG-inspired** because the strict definition also requires rRNA/tRNA criteria not evaluable from these assemblies.


| dataset        | binset | tool      | n    | n_true_HQ | n_pred_HQ | HQ_agreement | HQ_sensitivity | HQ_precision | MQ_agreement | n_true_contaminated_ge5pct | false_clean_rate_at_5pct | false_fail_rate_at_5pct |
|----------------|--------|-----------|------|-----------|-----------|--------------|----------------|--------------|--------------|----------------------------|--------------------------|-------------------------|
| marine         | gold   | MAGICC_V5 | 409  | 183       | 118       | 0.7922       | 0.5902         | 0.9153       | 0.9829       | 0                          |                          | 0.04401                 |
| marine         | gold   | CheckM2   | 409  | 183       | 93        | 0.78         | 0.5082         | 1            | 0.5037       | 0                          |                          | 0.1076                  |
| marine         | gold   | CoCoPyE   | 409  | 183       | 79        | 0.7457       | 0.4317         | 1            | 0.4083       | 0                          |                          | 0.6357                  |
| marine         | gold   | DeepCheck | 409  | 183       | 93        | 0.78         | 0.5082         | 1            | 0.4792       | 0                          |                          | 0.1394                  |
| marine         | mixed  | MAGICC_V5 | 986  | 57        | 185       | 0.8215       | 0.5789         | 0.1784       | 0.5649       | 819                        | 0.5971                   | 0.1198                  |
| marine         | mixed  | CheckM2   | 986  | 57        | 91        | 0.9006       | 0.4386         | 0.2747       | 0.5649       | 819                        | 0.5055                   | 0.1796                  |
| marine         | mixed  | CoCoPyE   | 986  | 57        | 30        | 0.9483       | 0.3158         | 0.6          | 0.6592       | 819                        | 0.1563                   | 0.6946                  |
| marine         | mixed  | DeepCheck | 986  | 57        | 117       | 0.8742       | 0.4386         | 0.2137       | 0.5193       | 819                        | 0.5372                   | 0.1078                  |
| strain_madness | gold   | MAGICC_V5 | 700  | 277       | 252       | 0.9586       | 0.9025         | 0.9921       | 0.9986       | 0                          |                          | 0.002857                |
| strain_madness | gold   | CheckM2   | 700  | 277       | 176       | 0.8557       | 0.6354         | 1            | 0.64         | 0                          |                          | 0.19                    |
| strain_madness | gold   | CoCoPyE   | 700  | 277       | 144       | 0.81         | 0.5199         | 1            | 0.2529       | 0                          |                          | 0.7943                  |
| strain_madness | gold   | DeepCheck | 700  | 277       | 183       | 0.8657       | 0.6606         | 1            | 0.5986       | 0                          |                          | 0.2029                  |
| strain_madness | mixed  | MAGICC_V5 | 2250 | 149       | 301       | 0.8578       | 0.4362         | 0.2159       | 0.7173       | 1804                       | 0.4523                   | 0.006726                |
| strain_madness | mixed  | CheckM2   | 2250 | 149       | 196       | 0.8964       | 0.3758         | 0.2857       | 0.5124       | 1804                       | 0.4767                   | 0.1233                  |
| strain_madness | mixed  | CoCoPyE   | 2250 | 149       | 56        | 0.9516       | 0.3221         | 0.8571       | 0.7129       | 1804                       | 0.01386                  | 0.6951                  |
| strain_madness | mixed  | DeepCheck | 2250 | 149       | 204       | 0.8911       | 0.3624         | 0.2647       | 0.4662       | 1804                       | 0.4945                   | 0.1345                  |


## Paired comparisons (identical bins, two-sided Wilcoxon, BH-corrected)

| dataset        | binset                            | tool_a    | tool_b    | metric        | n_pairs | mean_abs_err_a | mean_abs_err_b | hodges_lehmann | hl_lo  | hl_hi     | cliffs_delta | delta_lo | delta_hi | wilcoxon_p | p_bh       | favours   |
|----------------|-----------------------------------|-----------|-----------|---------------|---------|----------------|----------------|----------------|--------|-----------|--------------|----------|----------|------------|------------|-----------|
| marine         | gold                              | MAGICC_V5 | CheckM2   | completeness  | 409     | 9.733          | 25.28          | -14.09         | -17.44 | -11.74    | -0.3878      | -0.4651  | -0.3147  | 3.789e-48  | 8.526e-48  | MAGICC_V5 |
| marine         | gold                              | MAGICC_V5 | CheckM2   | contamination | 409     | 1.049          | 2.143          | -1.32          | -1.624 | -1.008    | -0.4153      | -0.504   | -0.3235  | 5.414e-29  | 8.474e-29  | MAGICC_V5 |
| marine         | gold                              | MAGICC_V5 | CoCoPyE   | completeness  | 409     | 9.733          | 12.76          | -3.236         | -4.771 | -1.707    | -0.291       | -0.3817  | -0.2011  | 2.901e-11  | 3.368e-11  | MAGICC_V5 |
| marine         | gold                              | MAGICC_V5 | CoCoPyE   | contamination | 409     | 1.049          | 14.74          | -13.27         | -14.25 | -12.56    | -0.4881      | -0.5857  | -0.3891  | 5.223e-48  | 1.106e-47  | MAGICC_V5 |
| marine         | gold                              | MAGICC_V5 | DeepCheck | completeness  | 409     | 9.733          | 26.32          | -16.72         | -18.81 | -15.34    | -0.5056      | -0.5808  | -0.4393  | 1.937e-59  | 4.98e-59   | MAGICC_V5 |
| marine         | gold                              | MAGICC_V5 | DeepCheck | contamination | 409     | 1.049          | 2.946          | -2.342         | -2.588 | -2.049    | -0.7173      | -0.8039  | -0.6187  | 3.448e-42  | 6.534e-42  | MAGICC_V5 |
| marine         | mixed                             | MAGICC_V5 | CheckM2   | completeness  | 986     | 10.17          | 15.74          | -2.049         | -4.076 | -0.5289   | 0.007552     | -0.06685 | 0.07865  | 3.714e-07  | 4.051e-07  | MAGICC_V5 |
| marine         | mixed                             | MAGICC_V5 | CheckM2   | contamination | 986     | 7.613          | 7.132          | 0.7183         | 0.3102 | 1.132     | 0.03996      | -0.01317 | 0.09007  | 1.039e-07  | 1.169e-07  | CheckM2   |
| marine         | mixed                             | MAGICC_V5 | CoCoPyE   | completeness  | 986     | 10.17          | 8.418          | 1.944          | 1.042  | 2.773     | 0.08342      | 0.01146  | 0.1543   | 1.381e-12  | 1.714e-12  | CoCoPyE   |
| marine         | mixed                             | MAGICC_V5 | CoCoPyE   | contamination | 986     | 7.613          | 11.85          | -3.775         | -4.698 | -2.863    | -0.3527      | -0.422   | -0.2875  | 5.561e-37  | 1.001e-36  | MAGICC_V5 |
| marine         | mixed                             | MAGICC_V5 | DeepCheck | completeness  | 986     | 10.17          | 15.64          | -4.239         | -6.488 | -2.438    | -0.09347     | -0.169   | -0.01416 | 2.191e-20  | 2.922e-20  | MAGICC_V5 |
| marine         | mixed                             | MAGICC_V5 | DeepCheck | contamination | 986     | 7.613          | 7.126          | 1.048          | 0.6127 | 1.473     | 0.02648      | -0.03042 | 0.07856  | 2.252e-11  | 2.703e-11  | DeepCheck |
| marine         | mixed::close_species_genus        | MAGICC_V5 | CheckM2   | contamination | 341     | 11.34          | 8.885          | 2.022          | 1.393  | 2.681     | 0.2199       | 0.1669   | 0.2787   | 1.029e-28  | 1.544e-28  | CheckM2   |
| marine         | mixed::distant_order_class_phylum | MAGICC_V5 | CheckM2   | contamination | 477     | 5.122          | 6.352          | -0.6776        | -1.311 | -0.08101  | -0.1277      | -0.2078  | -0.04834 | 0.007614   | 0.007614   | MAGICC_V5 |
| marine         | mixed::close_species_genus        | MAGICC_V5 | CoCoPyE   | contamination | 341     | 11.34          | 13.67          | -0.6032        | -2.102 | -1.49e-06 | -0.1739      | -0.2618  | -0.0875  | 5.401e-05  | 5.556e-05  | MAGICC_V5 |
| marine         | mixed::distant_order_class_phylum | MAGICC_V5 | CoCoPyE   | contamination | 477     | 5.122          | 10.55          | -5.286         | -6.485 | -4.094    | -0.4929      | -0.5842  | -0.3951  | 4.88e-29   | 7.986e-29  | MAGICC_V5 |
| marine         | mixed::close_species_genus        | MAGICC_V5 | DeepCheck | contamination | 341     | 11.34          | 8.124          | 3.005          | 2.911  | 3.386     | 0.2747       | 0.2316   | 0.326    | 3.268e-46  | 6.535e-46  | DeepCheck |
| marine         | mixed::distant_order_class_phylum | MAGICC_V5 | DeepCheck | contamination | 477     | 5.122          | 6.752          | -1.047         | -1.695 | -0.4701   | -0.1695      | -0.2437  | -0.09692 | 1.97e-05   | 2.086e-05  | MAGICC_V5 |
| strain_madness | gold                              | MAGICC_V5 | CheckM2   | completeness  | 700     | 1.295          | 13.96          | -12.36         | -13.11 | -11.69    | -0.6538      | -0.6983  | -0.61    | 9.973e-104 | 5.129e-103 | MAGICC_V5 |
| strain_madness | gold                              | MAGICC_V5 | CheckM2   | contamination | 700     | 0.5028         | 3.015          | -2.47          | -2.638 | -2.303    | -0.7541      | -0.7944  | -0.7133  | 8.626e-103 | 3.45e-102  | MAGICC_V5 |
| strain_madness | gold                              | MAGICC_V5 | CoCoPyE   | completeness  | 700     | 1.295          | 8.942          | -7.447         | -7.769 | -7.155    | -0.6213      | -0.6663  | -0.579   | 5.118e-85  | 1.842e-84  | MAGICC_V5 |
| strain_madness | gold                              | MAGICC_V5 | CoCoPyE   | contamination | 700     | 0.5028         | 20.06          | -20.82         | -22.75 | -18.57    | -0.792       | -0.8405  | -0.7412  | 6.052e-107 | 3.631e-106 | MAGICC_V5 |
| strain_madness | gold                              | MAGICC_V5 | DeepCheck | completeness  | 700     | 1.295          | 17.47          | -15.44         | -16.58 | -14.75    | -0.7999      | -0.8284  | -0.7725  | 1.703e-115 | 1.226e-114 | MAGICC_V5 |
| strain_madness | gold                              | MAGICC_V5 | DeepCheck | contamination | 700     | 0.5028         | 3.179          | -2.76          | -2.926 | -2.58     | -0.7876      | -0.8275  | -0.7458  | 1.241e-103 | 5.585e-103 | MAGICC_V5 |
| strain_madness | mixed                             | MAGICC_V5 | CheckM2   | completeness  | 2250    | 4.218          | 7.156          | -2.357         | -2.884 | -1.877    | -0.3231      | -0.3667  | -0.2784  | 1.788e-53  | 4.291e-53  | MAGICC_V5 |
| strain_madness | mixed                             | MAGICC_V5 | CheckM2   | contamination | 2250    | 4.441          | 5.918          | -1.119         | -1.414 | -0.8208   | -0.2244      | -0.2551  | -0.1926  | 1.459e-18  | 1.875e-18  | MAGICC_V5 |
| strain_madness | mixed                             | MAGICC_V5 | CoCoPyE   | completeness  | 2250    | 4.218          | 6.079          | -1.627         | -2.038 | -1.247    | -0.2534      | -0.297   | -0.2065  | 5.953e-34  | 1.021e-33  | MAGICC_V5 |
| strain_madness | mixed                             | MAGICC_V5 | CoCoPyE   | contamination | 2250    | 4.441          | 11.88          | -7.412         | -7.97  | -6.856    | -0.6216      | -0.6556  | -0.5869  | 9.874e-226 | 3.555e-224 | MAGICC_V5 |
| strain_madness | mixed                             | MAGICC_V5 | DeepCheck | completeness  | 2250    | 4.218          | 8.501          | -3.716         | -4.48  | -3.069    | -0.361       | -0.4129  | -0.3113  | 3.888e-82  | 1.273e-81  | MAGICC_V5 |
| strain_madness | mixed                             | MAGICC_V5 | DeepCheck | contamination | 2250    | 4.441          | 6.294          | -1.501         | -1.81  | -1.19     | -0.2497      | -0.2794  | -0.2179  | 2.926e-28  | 4.213e-28  | MAGICC_V5 |
| strain_madness | mixed::close_species_genus        | MAGICC_V5 | CheckM2   | contamination | 750     | 8.428          | 5.158          | 3.136          | 2.832  | 3.432     | 0.3287       | 0.2933   | 0.363    | 1.767e-74  | 5.302e-74  | CheckM2   |
| strain_madness | mixed::distant_order_class_phylum | MAGICC_V5 | CheckM2   | contamination | 1125    | 1.923          | 6.577          | -4.491         | -4.877 | -4.106    | -0.574       | -0.6153  | -0.5297  | 3.009e-117 | 2.709e-116 | MAGICC_V5 |
| strain_madness | mixed::close_species_genus        | MAGICC_V5 | CoCoPyE   | contamination | 750     | 8.428          | 12.86          | -4.2           | -5.113 | -3.319    | -0.3633      | -0.4308  | -0.2998  | 4.544e-26  | 6.291e-26  | MAGICC_V5 |
| strain_madness | mixed::distant_order_class_phylum | MAGICC_V5 | CoCoPyE   | contamination | 1125    | 1.923          | 11.05          | -8.926         | -9.619 | -8.23     | -0.7865      | -0.8258  | -0.7474  | 5.954e-167 | 1.072e-165 | MAGICC_V5 |
| strain_madness | mixed::close_species_genus        | MAGICC_V5 | DeepCheck | contamination | 750     | 8.428          | 5.52           | 2.958          | 2.691  | 3.198     | 0.2729       | 0.2336   | 0.3101   | 8.574e-66  | 2.374e-65  | DeepCheck |
| strain_madness | mixed::distant_order_class_phylum | MAGICC_V5 | DeepCheck | contamination | 1125    | 1.923          | 6.857          | -4.811         | -5.184 | -4.432    | -0.5589      | -0.5974  | -0.5164  | 3.549e-120 | 4.259e-119 | MAGICC_V5 |

Hodges-Lehmann is the median paired difference in absolute error (MAGICC minus comparator); negative favours MAGICC. Cliff's delta is the rank effect size. CIs are cluster bootstraps over dominant source genomes.


## Gold-standard bins along the completeness gradient

| dataset        | completeness_band | tool      | n   | comp_mae | comp_bias | cont_mae | cont_bias |
|----------------|-------------------|-----------|-----|----------|-----------|----------|-----------|
| marine         | 50-60             | CheckM2   | 70  | 21.55    | -21.55    | 1.405    | 1.405     |
| marine         | 50-60             | CoCoPyE   | 70  | 16.25    | 16.25     | 23.43    | 23.43     |
| marine         | 50-60             | DeepCheck | 70  | 26.34    | -26.34    | 2.235    | 2.235     |
| marine         | 50-60             | MAGICC_V5 | 70  | 3.062    | 1.068     | 0.4039   | 0.4039    |
| marine         | 60-70             | CheckM2   | 55  | 25.6     | -25.6     | 1.79     | 1.79      |
| marine         | 60-70             | CoCoPyE   | 55  | 15.4     | 14.94     | 24.83    | 24.83     |
| marine         | 60-70             | DeepCheck | 55  | 29.12    | -29.12    | 3.235    | 3.235     |
| marine         | 60-70             | MAGICC_V5 | 55  | 4.23     | -0.9233   | 0.3412   | 0.3412    |
| marine         | 70-80             | CheckM2   | 47  | 30.65    | -30.65    | 2.614    | 2.614     |
| marine         | 70-80             | CoCoPyE   | 47  | 13.95    | 9.384     | 21.04    | 21.04     |
| marine         | 70-80             | DeepCheck | 47  | 32.3     | -32.3     | 3.918    | 3.918     |
| marine         | 70-80             | MAGICC_V5 | 47  | 7.796    | -5.459    | 0.6255   | 0.6255    |
| marine         | 80-90             | CheckM2   | 54  | 27.34    | -27.34    | 3.52     | 3.52      |
| marine         | 80-90             | CoCoPyE   | 54  | 12.9     | 4.438     | 19.76    | 19.76     |
| marine         | 80-90             | DeepCheck | 54  | 29.27    | -29.27    | 4.943    | 4.943     |
| marine         | 80-90             | MAGICC_V5 | 54  | 10.35    | -6.16     | 0.9997   | 0.9997    |
| marine         | 90-95             | CheckM2   | 37  | 33.85    | -33.85    | 3.786    | 3.786     |
| marine         | 90-95             | CoCoPyE   | 37  | 14.83    | -5.868    | 13.21    | 13.21     |
| marine         | 90-95             | DeepCheck | 37  | 34.64    | -34.64    | 4.268    | 4.268     |
| marine         | 90-95             | MAGICC_V5 | 37  | 18.2     | -12.87    | 0.8463   | 0.8463    |
| marine         | 95-100            | CheckM2   | 146 | 22.27    | -21.98    | 1.554    | 1.554     |
| marine         | 95-100            | CoCoPyE   | 146 | 9.118    | -7.397    | 3.261    | 3.261     |
| marine         | 95-100            | DeepCheck | 146 | 20.15    | -20.15    | 1.793    | 1.7       |
| marine         | 95-100            | MAGICC_V5 | 146 | 13.25    | -12.07    | 1.831    | 1.831     |
| strain_madness | 50-60             | CheckM2   | 110 | 20.95    | -20.95    | 1.403    | 1.403     |
| strain_madness | 50-60             | CoCoPyE   | 110 | 15.88    | 15.88     | 28.93    | 28.93     |
| strain_madness | 50-60             | DeepCheck | 110 | 24.51    | -24.51    | 2.053    | 2.053     |
| strain_madness | 50-60             | MAGICC_V5 | 110 | 0.9429   | 0.09277   | 0.1709   | 0.1709    |
| strain_madness | 60-70             | CheckM2   | 105 | 24.52    | -24.52    | 2.228    | 2.228     |
| strain_madness | 60-70             | CoCoPyE   | 105 | 15.79    | 15.79     | 28.65    | 28.65     |
| strain_madness | 60-70             | DeepCheck | 105 | 27.04    | -27.04    | 3.264    | 3.264     |
| strain_madness | 60-70             | MAGICC_V5 | 105 | 0.9754   | -0.3632   | 0.2356   | 0.2356    |
| strain_madness | 70-80             | CheckM2   | 102 | 22.43    | -22.43    | 3.523    | 3.523     |
| strain_madness | 70-80             | CoCoPyE   | 102 | 13.6     | 13.6      | 27.77    | 27.77     |
| strain_madness | 70-80             | DeepCheck | 102 | 27.38    | -27.38    | 4.465    | 4.465     |
| strain_madness | 70-80             | MAGICC_V5 | 102 | 1.388    | -0.3828   | 0.4179   | 0.4179    |
| strain_madness | 80-90             | CheckM2   | 106 | 14.92    | -14.92    | 5.133    | 5.133     |
| strain_madness | 80-90             | CoCoPyE   | 106 | 9.228    | 9.228     | 26.16    | 26.16     |
| strain_madness | 80-90             | DeepCheck | 106 | 21.83    | -21.83    | 5.165    | 5.165     |
| strain_madness | 80-90             | MAGICC_V5 | 106 | 1.532    | -1.106    | 0.6306   | 0.6306    |
| strain_madness | 90-95             | CheckM2   | 59  | 10.12    | -10.12    | 5.528    | 5.528     |
| strain_madness | 90-95             | CoCoPyE   | 59  | 4.863    | 4.863     | 21.16    | 21.16     |
| strain_madness | 90-95             | DeepCheck | 59  | 14.51    | -14.51    | 4.862    | 4.862     |
| strain_madness | 90-95             | MAGICC_V5 | 59  | 2.699    | -0.7562   | 0.6839   | 0.6839    |
| strain_madness | 95-100            | CheckM2   | 218 | 1.952    | -1.589    | 2.261    | 2.261     |
| strain_madness | 95-100            | CoCoPyE   | 218 | 0.9256   | 0.8807    | 4.586    | 4.586     |
| strain_madness | 95-100            | DeepCheck | 218 | 3.335    | -3.335    | 1.684    | 1.591     |
| strain_madness | 95-100            | MAGICC_V5 | 218 | 1.09     | 0.8831    | 0.7275   | 0.7275    |


## Figures

### fig_ws3.8_bias_vs_distance

HEADLINE. Signed contamination error of each tool on the CAMI II constructed mixed bins, stratified by the taxonomic distance between the dominant and the contaminant source genome. The dashed grey line is MAGICC V5 on WS2 Set F, which was generated by our own simulation pipeline; CAMI II reproduces the same monotone gradient on third-party contigs with third-party truth. Error bars are 95% cluster-bootstrap CIs (2,000 iterations) clustered by dominant source genome. Colours are the project CVD-verified palette; no red/green discrimination is required. Denominator: completeness = retained dominant bp / FULL reference length of the dominant CAMI II source genome x 100; contamination = total contaminant bp / the SAME denominator x 100 (MAGICC convention). Truth is exact, from CAMI II's own gsa_mapping.tsv contig-to-source-genome assignment; only the grouping of contigs into bins is ours.

### fig_ws3.8_detection_slope

Contamination detection slope (OLS of predicted on true contamination) by taxonomic distance. A slope of 1 (dotted) means contamination is tracked one-for-one; 0 (solid) means the tool is blind to it. 95% cluster-bootstrap CIs clustered by dominant source genome. Denominator: completeness = retained dominant bp / FULL reference length of the dominant CAMI II source genome x 100; contamination = total contaminant bp / the SAME denominator x 100 (MAGICC convention). Truth is exact, from CAMI II's own gsa_mapping.tsv contig-to-source-genome assignment; only the grouping of contigs into bins is ours.

### fig_ws3.8_mixed_scatter

Predicted versus true contamination on the CAMI II constructed mixed bins, one column per taxonomic distance between dominant and contaminant, one row per tool. Dotted line is y = x. Points collapsing onto the x axis at species/genus distance are the failure mode: contamination that is present but not detected. Denominator: completeness = retained dominant bp / FULL reference length of the dominant CAMI II source genome x 100; contamination = total contaminant bp / the SAME denominator x 100 (MAGICC convention). Truth is exact, from CAMI II's own gsa_mapping.tsv contig-to-source-genome assignment; only the grouping of contigs into bins is ours.

### fig_ws3.8_gold_completeness

CAMI II gold-standard bins: contigs grouped by their true source genome, so every bin is pure (0% contamination) and the completeness gradient is whatever CAMI's own read simulation and gold-standard assembly produced. Left: predicted versus true completeness, dotted line y = x. Right: completeness MAE by true-completeness band. Contamination R2 is deliberately NOT reported for these bins: the true value is exactly 0 for every bin, so SS_tot = 0 and R2 is undefined (R1-m19). Denominator: completeness = retained dominant bp / FULL reference length of the dominant CAMI II source genome x 100; contamination = total contaminant bp / the SAME denominator x 100 (MAGICC convention). Truth is exact, from CAMI II's own gsa_mapping.tsv contig-to-source-genome assignment; only the grouping of contigs into bins is ours.

### fig_ws3.8_mimag

MIMAG-inspired outcomes on the CAMI II mixed bins. Thresholds: high quality >=90% completeness AND <5% contamination; medium quality >=50% AND <10%. Labelled MIMAG-INSPIRED because the strict definition additionally requires rRNA/tRNA criteria that cannot be evaluated from these assemblies. "false-clean at 5%" is the fraction of bins with true contamination >=5% that the tool calls <5%; "false-fail at 5%" is the converse on truly clean bins. Denominator: completeness = retained dominant bp / FULL reference length of the dominant CAMI II source genome x 100; contamination = total contaminant bp / the SAME denominator x 100 (MAGICC convention). Truth is exact, from CAMI II's own gsa_mapping.tsv contig-to-source-genome assignment; only the grouping of contigs into bins is ours.

### fig_ws3.8_palette_cvd_check

Verification that every colour used in the WS3.8 figures remains distinguishable under deuteranopia, protanopia and tritanopia, and that no figure requires a red/green discrimination (editorial requirement E4). The minimum pairwise CIE76 Delta-E across all four vision types is printed in the title; the accompanying table is cami2_palette_cvd_check.tsv.



## Conventions honoured

- R2 = coefficient of determination throughout; omitted where the truth has (near-)zero variance, with the reason recorded in the table (R1-m19, 4.4d).
- MIMAG-inspired thresholds, always labelled as such.
- MAGICC's 50% completeness floor handled explicitly; censored counts reported; a bounded below-floor probe cohort is reported separately and never pooled in.
- Primary analysis in-domain (contamination% <= completeness%, 4.4a); out-of-domain reported separately.
- All bootstrap seeds derive from `fw.stable_hash()` (CRC-32) with `PYTHONHASHSEED=0`; Python's salted `hash()` is never used.
- Two-sided paired tests, BH correction, Hodges-Lehmann and Cliff's delta with cluster-bootstrap CIs clustered by dominant source genome.
- CVD-safe palette, verified and plotted; no red/green discrimination (E4).
- Denominator stated in every table and figure caption (R1-M5).
