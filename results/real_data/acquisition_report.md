# WS3 Phase 1 — Real-data acquisition report

**Date:** 2026-07-26
**Workstream:** WS3 (real MAG / SAG / mock-community validation), protocol §4.4e — now the critical path
**Scope of this document:** discovery + acquisition + verification only. No heavy pipeline was started.
**Machine state during this phase:** 48 CPU, load average 19 → 56 while running; ≤6 threads used; total new download **901 MB**; disk 2.4 TB free.

---

## 0. Summary table

| # | Dataset | Form obtained | Ground truth? | Verified accessible | Difficulty | Downloaded now |
|---|---|---|---|---|---|---|
| 1 | **Meslier et al. 2022 mocks** (MOCK1/2/3, 91 strains) | **ready-made assemblies (7 technologies) + 91 reference genomes + exact composition** | **YES — exact, genome-level** | ✅ | 🟡 L2 | ✅ 604 MB |
| 2 | **Zymo D6300/D6310 + 10 isolates** (Nicholls 2019) | **ready-made SPAdes isolate drafts + 8 complete refs**; community reads in ENA | **YES — exact for isolates** | ✅ | 🟢 L1 (isolates) / 🟠 L3 (community MAGs) | ✅ 187 MB |
| 3 | **NCBI strain-matched draft/complete pairs** | metadata only (FTP paths recorded) | **YES — reference-anchored** | ✅ | 🟡 L2 | ✅ metadata (local file) |
| 4 | **UHGG / MGnify human-gut v2.0.2** | metadata (289,232 genomes) + per-genome FASTA on FTP | **NO** (disagreement only) | ✅ | 🟠 L3 | ✅ 110 MB metadata |
| 5 | **GTDB reviewer-genera cohort** | metadata already local; published **CheckM2** values | **NO** (disagreement only) | ✅ | 🟢 L1 | already local |
| 6 | **GORG-Tropics SAGs** (12,715) | ENA assembly accessions | **NO** / partial | ✅ | 🟡 L2 | ✅ manifest |
| 7 | **GTDB SAG cohort** (1,596; 453 with conspecific complete ref) | metadata already local | **partial — conspecific reference** | ✅ | 🟡 L2 | already local |
| 8 | **CAMI II** (marine / strain-madness / rhizosphere / pathogen) | gold-standard assembly + gold-standard binning + source genomes | **YES — exact** | ✅ open, no form needed | 🟠 L3 | ✅ 1 sample probe (377 MB, scratch) |
| 9 | **ATCC mock standards** (MSA-1002/1003) | reads only (PacBio); references behind ATCC agreement | partial | ⚠️ constrained | 🔴 L4 for direct use | ✅ manifest |

Everything marked ✅ was verified by actually retrieving the resource or its file listing/HTTP headers in this session. Nothing is recommended on the strength of a secondary summary.

---

## 1. Meslier et al. 2022 — **the single best resource found** (Reviewer 2 Priority 2)

**Publication.** Meslier V., Quinquis B., Da Silva K., Oñate F.P., Pons N., Roume H., Podar M., Almeida M.
*Benchmarking second and third-generation sequencing platforms for microbial metagenomics.*
**Sci Data 9, 694 (2022).** doi:10.1038/s41597-022-01762-z · PMID 36369227 · PMC9652401
Full text used: `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9652401/fullTextXML`

### 1.1 What it actually is — correction to the protocol
Protocol WS3.7 lists Meslier 2022 under **SAGs**. **That is wrong.** Meslier 2022 is a **mock-community / sequencing-platform benchmark**; there are no single-amplified genomes in it. Reviewer 2 cites it correctly, in the mock-community sentence: *"draft genomes, MAGs, and SAGs derived from isolates and mock communities (e.g. ATCC, Zymo standards or Meslier et al. 2022)"*. WS3.7 must be re-pointed at GORG-Tropics / GTDB SAGs (§6, §7).

### 1.2 Known composition — exact, quantitative, machine-readable
- **91 strains** total; three uneven mocks: **MOCK1 = 71**, **MOCK2 = 87**, **MOCK3 = 64** species.
- **29 GTDB phyla**; **68 Bacteria + 23 Archaea**.
- Genome sizes **0.49 – 9.73 Mbp** (median 2.74 Mbp); GC 27–69%; 21 strains carry plasmids/extra chromosomes.
- Relative abundance spans **three orders of magnitude**: MOCK1 0.0033 – 7.35 %, MOCK2 0.0033 – 7.25 %, MOCK3 0.0326 – 11.12 % (sum = 100 % in each).
- Source material: 58 strains of archived gDNA from Shakya et al.; 9 newly cultured; 4 from the Gilmour lab; plus **ATCC MSA-1002 (20 Strain Even Mix)** — so ATCC strains are embedded here, with NCBI accessions, and **without ATCC's download agreement**. 22 of the 91 strains carry ATCC designations, 18 DSM.
- **Every strain has a GenBank assembly accession** in Supplementary Table S1, and **86/91 are "Complete"** (5 "Draft").

Extracted to machine-readable form:
- `data/real_data/meslier2022/composition_S1.tsv` — 91 rows × 18 cols (organism, strain, genome label, **GenBank accession**, kingdom, NCBI+GTDB phylum, size, #scaffolds, #Ns, status, GC, #rRNA operons, **MOCK1/2/3 % abundance**, phylogenetic position)
- `data/real_data/meslier2022/mock1_abundance_and_binning_S3.tsv` — 71 rows, MOCK1 abundance + which binning run recovered each organism

### 1.3 Reference genomes — obtained
`https://forge.inrae.fr/metagenopolis/benchmark_mock` (public GitLab; note the paper prints the old host `forgemia.inra.fr`, which now 301-redirects to `forge.inrae.fr`; project id **5970**, branch `main`, visibility **public**).

- `reference/all_genomes_listed/*.fna.gz` — **91 individual reference genomes** → `data/real_data/meslier2022/references/` (88 MB)
  Verified locally: **65/91 are single-sequence closed chromosomes**, 16 have 2–3 sequences (chromosome + plasmids), 10 have >3, and only 4 have >10 sequences (max 136, *Methanobrevibacter oralis* DSM 7256). Total 303.7 Mbp. Stats: `references/reference_stats.json`.
- `reference/MOCK_00{1,2,3}.fasta.gz` — concatenated per-mock references (73/89/66 MB), also downloaded.

### 1.4 Assemblies — **already computed and published, no assembly compute needed**
`assembly/assemblies/MOCK_001.<tech>.fasta.gz` → `data/real_data/meslier2022/assemblies/` (313 MB). Verified locally:

| Assembly | contigs | total bp | N50 | longest |
|---|---|---|---|---|
| MOCK_001.illumina (SPAdes) | 71,367 | 159,271,694 | 14,599 | 1,630,540 |
| MOCK_001.mgiseq_2000 | 106,741 | 136,053,186 | 4,502 | 1,255,468 |
| MOCK_001.mgiseq_t7 | 115,274 | 134,358,760 | 4,042 | 1,063,407 |
| MOCK_001.minion (metaFlye) | 1,276 | 131,421,630 | 787,227 | 7,136,649 |
| MOCK_001.pacbio (metaFlye) | 436 | 163,865,108 | 2,013,697 | 7,147,004 |
| MOCK_001.proton | 50,680 | 133,276,450 | 7,462 | 426,841 |
| MOCK_001.s5 | 52,315 | 135,869,544 | 8,444 | 794,907 |

Seven independent assemblies of the *same known community* spanning N50 4 kb → 2 Mb. This is a **fragmentation gradient on real data with known truth** — it directly answers Reviewer 2's "assembly fragmentation, uneven coverage, chimeric contigs" objection without any simulation.

### 1.5 Binning already demonstrated by the authors
Supplementary Table S3 records DAS Tool-optimised bins (CONCOCT + MetaBAT2 + MaxBin2, min contig 1500 nt) for MOCK1:
**47/71** organisms recovered from Illumina×PacBio hybrid, **43/71** from Illumina×MinION, **41/71** from Illumina alone. **0 of the 13 organisms below 0.1 % abundance** were recovered by any run. KBase narrative: `https://narrative.kbase.us/narrative/125743`.
⚠️ The **bin FASTA files themselves are not in the git repo** — only the YES/NO recovery table. To get MAGs we must either re-bin ourselves or pull them from the KBase narrative (KBase export is interactive; not verified as a scripted download).

### 1.6 Reads (only needed if we re-bin)
ENA **PRJEB52977**, 42 runs, **186.90 GB** total. Saved manifest: `data/real_data/meslier2022/ena_PRJEB52977_runs.tsv`.
Per platform: DNBSEQ-T7 121.70 GB (2 runs) · **Illumina HiSeq 3000 20.07 GB (4 runs)** · DNBSEQ-G400 12.43 · Ion S5 10.40 · MinION 10.21 · Ion Proton 8.11 · PacBio Sequel 3.98.
**MOCK1 Illumina HiSeq 3000 = ERR9765746 (3.85 GB, 6.14 Gbp).**

> ⚠️ **Erratum to report to the reviewer if we cite the accessions.** Meslier Table 4 prints the Illumina runs as ERR9765446 / ERR9765447 / ERR9765448-49. **Those accessions do not exist.** Verified against the ENA portal API: the correct Illumina HiSeq 3000 runs are **ERR9765746 (MOCK_001), ERR9765747 (MOCK_002), ERR9765748 + ERR9765749 (MOCK_003)**. Table 4's ONT/Ion/PacBio accessions (ERR9765780/781/782, ERR9765783, ERR9765777-779) do resolve.

### 1.7 How ground truth is derived
`completeness(bin) = (reference bp of the dominant organism covered by the bin's contigs) / (that organism's full reference length) × 100`
`contamination(bin) = (bin bp aligning to any other reference organism) / (dominant organism's full reference length) × 100`
This is **exactly MAGICC's own denominator convention** (progress doc, Phase 3) — no re-definition, no MIMAG/CheckM2 denominator mismatch to argue about, which is directly useful for R1-M4/R1-M5.
Requires: minimap2 (contig → 91-reference alignment). Nothing else.

### 1.8 Licence / access
Repo carries a `LICENSE` file, public visibility. Reads are ENA open, "publicly available without restriction". No agreement, no registration. ✅

### 1.9 Training-set overlap — must be reported
Checked the 91 GenBank accessions against `data/splits/*`: **49 in TRAIN, 8 in VAL, 7 in TEST, 27 in none.**
These are canonical type strains, so overlap was inevitable. Two consequences:
1. The **evaluation inputs** (fragmented bins from real reads) were never seen by the model — the model never saw these assemblies.
2. But composition-based prediction could still benefit from having seen the same organism's sequence. So the analysis must report the full 71-organism result **and** a leakage-free primary subset (**34 organisms not in TRAIN/VAL**).

---

## 2. Zymo mock standards + 10 isolate genomes (Reviewer 2 Priority 1)

### 2.1 Reference genomes — obtained, no restrictions
Verified live and downloaded to `data/real_data/zymo/`:

| File | Bytes | SHA256 (prefix) | Contents |
|---|---|---|---|
| `https://s3.amazonaws.com/zymo-files/BioPool/ZymoBIOMICS.STD.refseq.v2.zip` | 21,464,466 | a66d2085… | D6300/D6305/D6306/D6310: 8 **complete** bacterial genomes + 2 **draft** yeast genomes + ssrRNAs |
| `https://s3.amazonaws.com/zymo-files/BioPool/D6331.refseq.zip` | 27,198,072 | 4795b7df… | D6331 Gut Microbiome Standard: 21 genomes incl. *Methanobrevibacter smithii* (archaeon) and **5 distinct *E. coli* strains** |
| `https://s3.amazonaws.com/zymo-files/BioPool/D6320.refseq.zip` | 1,683,367 | fcb9e2aa… | D6320 spike-in: 2 genomes |

### 2.2 Ready-made isolate draft assemblies — obtained
Nicholls S.M., Quick J.C., Tang S., Loman N.J. *Ultra-deep, long-read nanopore sequencing of mock microbial community standards.* **GigaScience 8(5):giz043 (2019).** doi:10.1093/gigascience/giz043 · PMC6520541
Project page `https://lomanlab.github.io/mockcommunity/` · code `https://github.com/LomanLab/mockcommunity`

Downloaded: `http://nanopore.s3.climb.ac.uk/mockcommunity/v2/Zymo-Isolates-SPAdes-Illumina.fasta` (71,433,493 B, 2,428 contigs). It is a **single multi-FASTA of all 10 isolates' SPAdes drafts**, contigs prefixed by a 2-letter organism code — split trivially. Verified against the Zymo complete references:

| code | organism | draft bp | contigs | reference bp | ref seqs | draft/ref |
|---|---|---|---|---|---|---|
| bs | *Bacillus subtilis* | 3,985,829 | 29 | 4,045,677 | 1 | 0.9852 |
| ec | *Escherichia coli* | 4,787,036 | 103 | 4,875,441 | 2 | 0.9819 |
| ef | *Enterococcus faecalis* | 2,820,294 | 30 | 2,845,392 | 1 | 0.9912 |
| lf | *Lactobacillus fermentum* | 1,805,517 | 84 | 1,905,333 | 1 | 0.9476 |
| lm | *Listeria monocytogenes* | 2,958,570 | 16 | 2,992,342 | 1 | 0.9887 |
| pa | *Pseudomonas aeruginosa* | 6,728,151 | 83 | 6,792,330 | 1 | 0.9906 |
| sa | *Staphylococcus aureus* | 2,689,205 | 67 | 2,730,326 | 4 | 0.9849 |
| se | *Salmonella enterica* | 4,738,357 | 66 | 4,809,318 | 2 | 0.9852 |
| (cn) | *Cryptococcus neoformans* | 28,089,685 | 1562 | 29,176,277 | 7154 | eukaryote — **out of MAGICC scope** |
| (sc) | *Saccharomyces cerevisiae* | 11,546,282 | 388 | 12,843,354 | 51 | eukaryote — **out of MAGICC scope** |

**8 usable isolate draft/complete pairs, zero assembly compute, exact truth.** Expected truth band: completeness ≈ 95–99 %, contamination ≈ 0 %. Small n, but this is precisely the regime where MAGICC's worst honest counter-finding lives (44.2 % false-fail at the 5 % contamination threshold on `set_C_clean`), so it is a *targeted* false-positive test on real isolate assemblies.

### 2.3 Community data (needed only for Zymo MAGs)
ENA **PRJEB29504**; manifest saved to `data/real_data/zymo/ena_PRJEB29504_runs.tsv`. Exact byte counts from the ENA API:

| Run | Content | fastq bytes |
|---|---|---|
| ERR2984773 | **Even (D6300) community, Illumina MiSeq**, 8.79 M pairs | 1.96 GB |
| ERR2935805 | **Log (D6310) community, Illumina HiSeq 1500**, 47.8 M pairs | 6.54 GB |
| ERR3152364 | Even, GridION nanopore, 14.38 Gbp | 14.99 GB |
| ERR3152366 | Log, GridION nanopore, 16.51 Gbp | 17.15 GB |
| ERR3152365 / ERR3152367 | Even/Log **PromethION** | 157.6 GB / 160.2 GB — **do not fetch** |
| ERR2935848–ERR2935857 | the 10 isolate Illumina runs | 16.98 GB total (14.47 GB for the 8 bacteria) |

Composition: Even = 8 bacteria at 12 % each + 2 yeasts at 2 %. Log (D6310) = log-scaled cell counts from **89.1 % (*L. monocytogenes*) down to 0.000089 % (*S. aureus*)** — a 6-order-of-magnitude gradient, which would produce MAGs across the full completeness range.

### 2.4 ATCC — the one reviewer suggestion that is materially constrained
- **ATCC MSA-1003** ("20 Strain Staggered Mix", 0.02–18 % composition): only **BioProject PRJNA546278** public data found — **2 PacBio Sequel II runs only** (SRR9328980 16.14 GB, SRR9202034 14.99 GB; no Illumina). Manifest: `data/real_data/atcc/ena_PRJNA546278_MSA-1003_runs.tsv`.
- **ATCC reference genomes** live in the **ATCC Genome Portal** (`https://genomes.atcc.org`), which is gated by a **Data Use Agreement** and membership/registration, with no public bulk API found. That is incompatible with a Nature Communications Data Availability statement that must let a reader reproduce the benchmark.
- **Resolution:** we do not need the ATCC portal. **ATCC MSA-1002 strains are already inside Meslier MOCK1/2/3 with public GenBank accessions**, and 22 of the 91 Meslier references carry ATCC designations. We therefore satisfy "ATCC standards" through Meslier and say so explicitly in the response. This is a finding, not a failure.

---

## 3. NCBI strain-matched draft/complete pairs (Reviewer 2 Priority 3) — **largest ground-truth N available**

Script: `scripts/130_ncbi_draft_complete_pairs.py` (read-only on `data/ncbi/`, no network, ~1 min, single-threaded).
Input: `data/ncbi/assembly_summary_genbank.txt` (3,379,307 latest assemblies).
Outputs: `results/revision/real_data/ncbi_pairs/{tier1_same_biosample_pairs.tsv, tier2_same_strain_pairs.tsv, pairs_summary.json}` — each row carries both FTP paths, so acquisition is a scripted `wget` list.

**Assembly-level census:** Contig 2,528,743 · Scaffold 498,080 · Complete Genome 307,494 · Chromosome 44,990.

### Tier 1 — same **BioSample**, one Complete/Chromosome + one Contig/Scaffold (strongest possible match)
Same BioSample = **same physical DNA isolate**, so draft sequence absent from the complete assembly is assembly artefact or genuine contamination, *not* strain divergence.
- **4,834 pairs**, 4,654 unique complete references, **3,666 species_taxids**
- Bacteria/archaea only: **1,121 pairs**; with draft/complete size ratio in 0.80–1.20: **1,097 pairs, 165 species, 0 archaea**
- Size-ratio quantiles (all groups): p05 0.365 · p25 0.910 · **p50 0.983** · p75 1.001 · p95 1.193
- 20 drafts already flagged `contaminated` by NCBI; 2 `derived from metagenome`
- Download volume: 1,097 drafts (4.65 Gbp) + 1,006 completes (4.29 Gbp) ≈ **2.6 GB gzipped**
- ⚠️ Domain filter is **mandatory**: unfiltered Tier 1 contains *Homo sapiens* (36), *Ovis aries* (30), "marine metagenome" (38). The `group` column is now emitted for this purpose.
- Training overlap: only **50/4,834** draft accessions and 154/4,834 complete accessions appear in TRAIN/VAL; **only 1,005/4,834 pairs** have a binomial matching a GTDB TRAIN species name → ~79 % of pairs are species not named in training. Good novelty spread.

### Tier 2 — same species_taxid + identical normalised strain designation, different BioSample
- **23,327 pairs**; bacteria/archaea **22,748**; in the 0.80–1.20 ratio band **22,146** across **1,338 species (83 archaeal)**
- Heavily skewed: *E. coli* 5,238 · *K. pneumoniae* 4,796 · *M. tuberculosis* 2,691 · *C. jejuni* 2,486 · *S. aureus* 1,641
- Species-capped subsamples (all 1,338 species retained): cap 2 → **1,687**; cap 5 → **2,109**; cap 10 → 2,482; cap 20 → 2,908 pairs
- 125 drafts flagged `contaminated`, 34 `derived from metagenome`, **11 `derived from single cell`**
- Full-tier download ≈ 37.5 GB gzipped; capped subsample ≈ 4 GB
- Weakness: different BioSamples means real strain/accessory divergence can masquerade as contamination. Tier 2 is a **secondary/sensitivity** cohort, not primary.

### Bonus cohort discovered — NCBI's own contamination flag
**23,048 assemblies** carry `contaminated` in `excluded_from_refseq` (Contig 14,637 · Scaffold 7,907 · Complete Genome 387 · Chromosome 117). This is NCBI's Foreign Contamination Screen verdict: **binary, not quantitative**, but it is an independent positive-control label for *detection* (precision/recall at the 5 %/10 % thresholds) that no reviewer can call circular. Co-occurring tokens: `derived from metagenome` 13,662, `fragmented assembly` 6,099, `genome length too large` 657.

### Ground-truth derivation for pairs
Align draft → complete with minimap2 (asm5/asm10) or nucmer:
`completeness = covered reference bp / reference total bp × 100`; `contamination = draft bp with no reference alignment / reference total bp × 100`.
**Stated limitation:** the "unaligned" fraction conflates true contamination with (i) accessory/plasmid content absent from the chosen complete assembly and (ii) assembly artefacts. Tier 1 (same BioSample) minimises (i). Report the unaligned fraction as an **upper bound** on contamination and cross-check against the NCBI `contaminated` flag and GUNC.

---

## 4. UHGG / MGnify human-gut catalogue (Reviewer 2 Priority 5) — **no ground truth**

**Release used:** MGnify human-gut **v2.0.2** (latest; FTP shows only v1.0, v2.0, v2.0.1, v2.0.2).
Base URL `https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v2.0.2/`
Downloaded: `genomes-all_metadata.tsv` (115,140,145 B, **289,232 genomes**) + `README_v2.0.2.txt` → `data/real_data/uhgg/`.

- 289,232 genomes = **278,263 MAGs + 10,969 isolates**, clustered into **4,744 species reps**
- Taxonomy: **GTDB r202**
- Shipped `Completeness` / `Contamination`: **CheckM v1**, not CheckM2. Range: completeness min 50.03, median 90.59; contamination min 0.00, median 0.84, **max 5.00** (catalogue inclusion criteria are comp ≥ 50, cont ≤ 5)
- Per-genome download works: `FTP_download` column → Prokka `.gff.gz` with an embedded `##FASTA` section (verified: `MGYG000000001.gff.gz`, 1,162,441 B, `##FASTA` at line 3376). Path also resolves under `v2.0.2/`. FASTA must be split out of the GFF.
- Volume: mean assembly 2.47 Mbp → **5,000 genomes ≈ 5.6 GB**, 20,000 ≈ 22 GB

### The four reviewer-named genera in UHGG
| genus | genomes in UHGG | UHGG (CheckM1) completeness median | contamination median | length median | contigs median |
|---|---|---|---|---|---|
| **CAG-557** | **89** | 77.16 (min 50.5) | 0.67 (max 4.13) | 1,067,093 | 98 |
| **UMGS1491** | **30** | 67.52 (min 54.2) | 0.67 (max 4.24) | 960,731 | 135 |
| **HGM10766** | **7** | 91.01 (min 80.1) | 0.28 (max 2.25) | 1,129,673 | 52 |
| **Caccenecus** | **0** | — | — | — | — |
| **Faecimonas** | **0** | — | — | — | — |

`Caccenecus` and `Faecimonas` **do not exist in UHGG v2.0.2** because its taxonomy is GTDB r202 and those names were introduced later. So **UHGG alone cannot reproduce the reviewer's full observation.**

### The cohort that *can*: GTDB (already on disk, published **CheckM2** values)
`data/gtdb/{bac120,ar53}_metadata.tsv.gz` — 732,475 genomes, **113 columns including `checkm2_completeness` and `checkm2_contamination` for every genome**. All five reviewer genera are present, and all are **100 % `derived from metagenome`** (i.e. MAGs):

| genus | genomes | CheckM2 comp median | CheckM2 cont median | genome size median |
|---|---|---|---|---|
| CAG-557 | 60 | 97.36 | 0.81 | 981,188 |
| UMGS1491 | 61 | 94.52 | 0.27 | 1,095,451 |
| Caccenecus | 101 | 92.41 | 0.82 | 1,286,632 |
| HGM10766 | 15 | 89.05 | 1.27 | 1,091,647 |
| Faecimonas | 61 | 92.84 | 2.54 | 1,410,926 |
| **total** | **298** | | | |

**All five are reduced-genome lineages (0.98–1.41 Mbp)** — exactly the regime where `set_C_clean` exposed MAGICC's worst behaviour (completeness bias −8.68 pp, contamination +5.09 pp, 44.2 % false-fail at 5 %). The reviewer's observation is therefore *a priori* plausible and WS3.3 is a real falsification test.

**Sign-convention analysis (important for WS3.3 design).** The reviewer's completeness deltas are stated as MAGICC − CheckM2 (positive = MAGICC higher). Their contamination deltas are negative for genera where MAGICC would be expected to be *higher*; with UHGG's contamination medians (0.67, 0.67, 0.28) the numbers **only reconcile if the contamination column is CheckM2 − MAGICC**, i.e. MAGICC ≈ 26.7 % (CAG-557), 36.6 % (UMGS1491), 22.1 % (HGM10766). That is fully consistent with MAGICC over-calling contamination on reduced genomes, and with `set_C_clean`. It also means the reviewer's comparator baseline for *completeness* (~62.9 % for CAG-557 to make +37.1 reach 100) is lower than both UHGG-CheckM1 (77.16) and GTDB-CheckM2 (97.36) — so **the exact catalogue the reviewer used is not identified**. Unverified candidates worth one hour of checking before WS3.3 runs: **SPIRE** (`http://spire.embl.de`, 1.16 M MAGs, GTDB r207+) and **HRGM2**. We must state in the response which catalogue we used and reproduce the *direction and magnitude* rather than the exact digits.

**Status: NO GROUND TRUTH.** Per protocol §8.3, results from UHGG/GTDB cohorts may only be reported as **disagreement characterisation**.

---

## 5. CAMI II (Reviewer 1 M3 "independent procedure") — open, verified

**No request form is required** for the challenge data, contrary to the CAMI II web page: everything is served openly from the PUBLISSO repository.

Base: `https://frl.publisso.de/data/frl:6425521/` (= `https://repository.publisso.de/resource/frl:6425521/`)
Contents verified by directory listing: `marine/`, `strain/`, `plant_associated/`, `patmgCAMI2.tar.gz`, `md5sums.txt`, `readme.txt`.
Toy datasets: `https://frl.publisso.de/data/frl:6425518/` (human body sites: `airskinurogenital/`, `gastrooral/`, ± `_pbsim` long-read) and `https://frl.publisso.de/data/frl:6421672/` (`dataset/`, `dataset_pbsim/`). `https://data.cami-challenge.org/participate` also still responds (HTTP 200).

Sizes verified by HTTP `Content-Length`:

| File | Bytes |
|---|---|
| `marine/marmgCAMI2_genomes.tar.gz` (source genomes) | 836,461,952 |
| `strain/strmgCAMI2_genomes.tar.gz` | 452,195,617 |
| `plant_associated/rhimgCAMI2_genomes.tar.gz` | 1,639,376,960 |
| `patmgCAMI2.tar.gz` (pathogen detection) | 584,127,020 |
| `marine/short_read/marmgCAMI2_sample_0_contigs.tar.gz` | 377,315,812 |
| `strain/short_read/strmgCAMI2_sample_0_contigs.tar.gz` | 203,698,672 |
| `marine/short_read/marmgCAMI2_sample_0_reads.tar.gz` | 5,550,020,311 |
| `marine/short_read/marmgCAMI2_sample_0_bam.tar.gz` | 4,936,471,735 |

**Structure confirmed by actually downloading and listing `marmgCAMI2_sample_0_contigs.tar.gz`:**
```
simulation_short_read/2018.08.15_09.49.32_sample_0/contigs/anonymous_gsa.fasta.gz   <- gold-standard assembly
simulation_short_read/2018.08.15_09.49.32_sample_0/contigs/binning_gs.tsv           <- gold-standard contig->genome map
simulation_short_read/2018.08.15_09.49.32_sample_0/contigs/gsa_mapping.tsv.gz
```
Ground truth is therefore **exact and free**: for every source genome, completeness = its bp present in the gold-standard assembly / its full length; the gold-standard bins have contamination = 0 by construction. To obtain a *contamination* axis we must either (a) run a real binner using the provided BAMs (+4.9 GB per sample) or (b) recover CAMI II participant binning submissions (Zenodo community `https://zenodo.org/communities/cami/` — **not verified in this session**).
Dataset scale: CAMI II sampled **1,680 microbial genomes and 599 circular elements**; 10 samples per environment.
Licence: open, CC-style repository deposit; no registration used.

---

## 6. GORG-Tropics SAGs (Reviewer 1 M9 / protocol WS3.7) — obtainable, but mostly outside MAGICC's operating range

Pachiadaki M.G. et al. *Charting the complexity of the marine microbiome through single-cell genomics.* **Cell 179:1623–1635 (2019).** 12,715 SAGs (>20 kbp assemblies, "no detectable contamination"), 8.1 Gbp cumulative, 28 tropical/subtropical epipelagic samples.
Access verified: **ENA/NCBI BioProject PRJEB33281** returns **12,715 assembly accessions** (`GCA_9025…`) — manifest saved to `data/real_data/gorg_tropics/ena_PRJEB33281_assemblies.tsv`. Also on OSF: doi:10.17605/OSF.IO/PCWJ9.
Download volume ≈ 8.1 Gbp ⇒ **~2.5 GB gzipped** for all 12,715.

**Decisive quality census (computed from local GTDB metadata, which carries published CheckM2 values):**
Only **3,151 of 12,715 (24.8 %)** GORG-Tropics genomes are present in GTDB — GTDB's floor is completeness ≥ 50 %, so **~9,564 (75 %) of GORG-Tropics SAGs fall below MAGICC's 50 % completeness floor and cannot be scored at all.**
For the 3,151 that qualify: CheckM2 completeness min 50.0, p10 59.4, **median 75.4**, p90 89.3, max 100.0; only 278 ≥ 90 %. CheckM2 contamination median 0.05, p90 0.36, **max 3.11** — i.e. this cohort is a near-pure **false-positive contamination test**. Phyla: Pseudomonadota 2,467, Bacteroidota 248, Cyanobacteriota 185, Actinomycetota 123.

**No ground truth**: GORG SAGs are uncultured marine lineages with no complete references.

## 7. A genuine, if limited, reference-anchored SAG cohort
From local GTDB metadata: **1,596 genomes are `derived from single cell`** (published CheckM2 completeness min 42.3, median 75.1, p90 93.8; contamination median 0.12, max 8.28; 1,595 have completeness ≥ 50). Top BioProjects PRJNA846736 (571), PRJNA445865 (446), PRJNA1084198 (96). Phyla: Pseudomonadota 552, Cyanobacteriota 437, Bacteroidota 149, Actinomycetota 96, Thermoproteota 57.

Cross-referencing `ncbi_species_taxid` against the 16,938 bacterial/archaeal species_taxids that have ≥1 latest **Complete Genome** in GenBank:
**453 SAGs belong to a species with a complete reference genome** (CheckM2 completeness median 72.3, all ≥ 50). Phyla: Pseudomonadota 179, Bacteroidota 129, Actinomycetota 25, Thermoproteota 18, Verrucomicrobiota 17, Thermoplasmatota 17.

**This is the only route to SAG ground truth found.** Truth is *conspecific*, not same-strain, so accessory-genome divergence inflates apparent contamination and deflates completeness; restrict to ANI ≥ 97–98 % and report as bounded truth.

---

## 8. Resources checked but not pursued (with reasons)

- **ATCC Genome Portal bulk download** — Data Use Agreement + membership gate, no public API found; incompatible with Nature Comms data availability. Superseded by Meslier (which embeds ATCC MSA-1002 with public GCA accessions).
- **Zymo PromethION runs** (ERR3152365/ERR3152367, 157.6/160.2 GB) — no analytical benefit over GridION for this purpose.
- **Meslier DNBSEQ-T7 reads** (121.7 GB of the 186.9 GB project) — the T7 assembly is already published in the repo.
- **CAMI II reads and BAMs** — only needed if we run a real binner; 5.6/4.9 GB *per sample*, 10 samples per environment.
- **SPIRE / HRGM2** — flagged as unverified candidates for identifying the reviewer's exact catalogue. Not recommended for use until confirmed.

---

## 9. What is on disk after this phase (901 MB total)

```
data/real_data/
├── meslier2022/            604 MB  91 references + 7 MOCK1 assemblies + composition + profiling + ENA manifest
├── zymo/                   187 MB  3 reference bundles (unzipped STD) + 10-isolate SPAdes drafts + ENA manifest
├── uhgg/                   110 MB  genomes-all_metadata.tsv (289,232 rows) + README
├── gorg_tropics/           988 KB  ENA PRJEB33281 assembly manifest (12,715 rows)
└── atcc/                   < 1 KB  ENA PRJNA546278 run manifest
results/revision/real_data/ncbi_pairs/   tier1 (4,834) + tier2 (23,327) pair tables + pairs_summary.json
```
