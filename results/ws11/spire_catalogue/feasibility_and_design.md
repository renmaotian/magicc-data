# WS11.S — feasibility measurements and sampling design (recorded before the transfer)

Measured 2026-08-26/27 on the project host. Every rate below is a measurement, not an
estimate, unless marked *extrapolated*.

## 1. How SPIRE assemblies are obtained

| Route | Endpoint | Object |
|---|---|---|
| Per genome | `https://spire.embl.de/download_file/<genome_id>` | one gzipped FASTA |
| Bulk, dereplicated | `https://black.embl.de/~fullam/spire/representatives/spire_representative_genomes.tar` | **86,849,024,000 B (86.85 GB)**, `spire_representative_genomes/<id>.fa.gz`, 107,078 members (92,063 `spire_mag_*` + 14,943 proGenomes `specI_v4_*`) |
| Bulk, per study | `https://black.embl.de/~fullam/spire/compiled/<study>_spire_v1_MAGs.tar` | 709 study tarballs, **1.418 TB total**, ~2.80 M MAG files (all MAGs SPIRE produced, incl. the ~1.64 M that failed catalogue QC) |

Both bulk hosts honour `Accept-Ranges: bytes`, so a bulk transfer is resumable.

## 2. Measured transfer rates

| Test | Result |
|---|---|
| `black.embl.de`, 1 stream | **11.72 / 12.83 MB/s** (two 200 MB / 100 MB range reads) |
| `black.embl.de`, 2 concurrent | 13.3 MB/s aggregate |
| `black.embl.de`, 3–4 concurrent | one connection immediately returns **HTTP 403** — the host caps concurrency; aggregate does not improve |
| `speed.cloudflare.com`, 4 streams | **27.8 MB/s** — our uplink is *not* the binding constraint at 12 MB/s |
| `spire.embl.de` GET, 4 workers | 2.98 genomes/s, 2.06 MB/s (n = 300) |
| `spire.embl.de` GET, 16 workers | **11.23 genomes/s, 7.57 MB/s** (n = 120) |
| `spire.embl.de` GET, 24 workers | 12.67 genomes/s, 8.80 MB/s, p90 latency 1.34 → 2.61 s (queueing begins) |
| `spire.embl.de` HEAD, 4→32 workers | 6.3 → 41.8 req/s, **linear**, median latency flat at 0.62–0.65 s, no throttling |
| End-to-end fetch+score, 14–16 workers | **13–15 genomes/s** including feature extraction and inference |

Per-genome compressed size (n = 3,877 exact `content-length` values from a seeded random
probe): **median 641,809 B, mean 703,136 B**, 0.30956 B per genome bp. Whole-frame total
**≈ 813 GB**.

**Non-response:** `spire.embl.de` returns **HTTP 404 for 123 / 4,000 = 3.08 %** of a
seeded random sample of catalogue ids (the earlier five-genus cohort saw 21/612 = 3.4 %).
Missingness is close to non-differential: 404 genomes have median 2.102 Mbp vs 2.078 Mbp
for retrieved ones, 33.3 % vs 29.9 % published-HQ, and 404 rates by size bin of
1.0 / 3.3 / 3.5 / 2.3 / 1.1 % (<1, 1–2, 2–3, 3–5, >5 Mbp). It does track the SPIRE
cluster-assignment method (0.95 % for `95_ANI`, 3.6 % `pg_v3_mapped_marker_gene`, 4.5 %
`pg_v3_mapped_ANI_95`). Non-response is recorded per genome and tested on the achieved
sample.

## 3. The three requested cost estimates (per-genome route, 16 workers)

| Cohort | n | Transfer | Wall clock |
|---|---|---|---|
| **Entire catalogue** | 1,158,468 | **813 GB** | **28.7 h** (19.2 h *extrapolated* at 24 workers) |
| **200,000 sample** | 200,000 | 141 GB | **4.9 h** |
| **50,000 sample** | 50,000 | 35 GB | **1.2 h** |

Bulk alternatives: the 709 per-study MAG tarballs would take **32.8 h and 1.418 TB** to
deliver the same 1.16 M catalogue MAGs plus 1.64 M non-catalogue ones — strictly worse,
**rejected**. The single representatives tarball takes **2.0 h and 86.85 GB** and is
**used**, because it is the cheapest genomes-per-byte route on offer, is 404-free, and
delivers a *complete* population rather than a sample.

## 4. Chosen design

Frame: the **1,158,468** SPIRE v1 MAGs that carry a published CheckM2 completeness,
contamination and genome size (85 of the 1,158,553 metadata rows do not and are out of
frame). Stratified random sample, strata defined on frame variables only:

| Stratum | Definition | N | n | f | weight |
|---|---|---|---|---|---|
| `S1_small` | genome_size < 1 Mbp | 59,297 | 59,297 | 1.000 | 1.000 |
| `S2_rare_phylum` | not S1, phylum with < 1,000 frame MAGs | 14,586 | 14,586 | 1.000 | 1.000 |
| `S3_main` | remainder | 1,084,585 | 200,000 | 0.1844 | 5.42293 |

**Total n = 273,883 = 23.64 % of the catalogue.** Seed = CRC-32("WS11.S/spire_catalogue/SRS")
= 2248018003, `numpy.random.default_rng` (PCG64). The S3 sample is a prefix of one seeded
permutation, so any prefix reached is still a valid SRS of S3 — the design survives
truncation. Genomes are processed S1, then S2, then S3 in SRS order for that reason.

An `extreme_reduction` stratum (log2(size / phylum median) < −2) was evaluated and **not**
created: 1,188 of those 1,192 genomes already fall inside S1 ∪ S2.

Headline catalogue rate = design-weighted (Hájek) estimate over S1 ∪ S2 ∪ S3, with the
unweighted S3-only SRS estimate reported beside it as a design check. Size, phylum and
lineage-reduction strata are reported from whichever cohort has power, each labelled.

**Second cohort, `reps`:** all **92,063** SPIRE 95 %-ANI cluster representatives that are
MAGs and in frame — a **complete census of the dereplicated catalogue**, 86.85 GB / 2.0 h,
no sampling error and no 404 loss. Representatives are the best genome of their cluster,
so this is a *different population*; its rate is never quoted as the per-MAG rate.

**Score representatives or all genomes? Both, and label them.** The reviewer's question
("what proportion of catalogue MAGs change class") has the individual MAG as its unit, so
the headline must come from the probability sample of MAGs. The dereplicated census
answers the question a downstream user of the catalogue actually faces, and being a census
it carries no sampling caveat at all. Reporting only one of the two would repeat the
ambiguity the 750-genome cohort left behind.

## 5. Binding interpretation

Catalogue MAGs carry **no ground truth**. Everything here is **disagreement between
MAGICC V5 and SPIRE's published CheckM2**, never error. The ground-truthed anchor is
`set_C_clean`, where MAGICC is the tool in error: −8.68 pp completeness and +5.09 pp
contamination against truth, versus CheckM2's −0.70 and −1.16.
