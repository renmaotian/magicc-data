# WS3 Phase 1 — Ranked feasibility plan and compute budget

**Date:** 2026-07-26 · **Companion:** `acquisition_report.md` (all accessions, URLs, verification evidence)
**Decision this document requests:** approval to start Track A (and only Track A) heavy work.

---

## 0. The single most important distinction

| Provides **genuine ground truth** (may be called *validation*) | Provides **tool disagreement only** (may NOT be called validation) |
|---|---|
| Meslier 2022 mock communities (91 known references, exact abundances) | UHGG human-gut catalogue |
| Zymo 8 isolate drafts vs 8 complete references | GTDB reviewer-genera cohort (298 genomes) |
| Zymo Even/Log community MAGs (10 known references) | GORG-Tropics SAG cohort |
| NCBI same-BioSample draft/complete pairs (1,097 prokaryote) | GTDB `derived from single cell` cohort (agreement part) |
| CAMI II gold-standard bins (1,680 known genomes) | NCBI `contaminated` flag cohort (*detection* label only, binary) |
| GTDB SAGs with a conspecific complete reference (453) — **bounded** truth | |

Protocol §8.3 is binding: the right-hand column can only ever be written up as "MAGICC and CheckM2 disagree", with the disagreement characterised.

---

## 1. Ranked recommendation

### Rank 1 — **Meslier et al. 2022 MOCK1 (+MOCK2/3 optional)** 🟡 L2 · **do this first**
**Why first:** it is the only resource that simultaneously (a) is a resource Reviewer 2 named by citation, (b) has exact genome-level ground truth in *MAGICC's own denominator convention*, (c) is already 100 % on disk, (d) needs **no assembly**, (e) covers 71 organisms × 29 phyla including 23 archaea, and (f) supplies a **fragmentation gradient on real data** (N50 4 kb → 2 Mb across 7 sequencing technologies) that answers Reviewer 2's "real MAGs have fragmentation, uneven coverage, chimeric contigs" objection without simulation.

**Pipeline**
1. minimap2 `-x asm10` each of the 7 MOCK1 assemblies (436–115,274 contigs) against the 91-reference set (303.7 Mbp). *No reads needed.*
2. **Bin set A — reference-anchored bins.** Assign each contig to its best-hit reference; bins are near-pure. Gives **completeness across a wide real range** (driven by the 3-orders-of-magnitude abundance spread) at **≈0 % contamination** → a real-data **false-positive contamination test**, which is exactly where MAGICC's 44.2 % false-fail rate on `set_C_clean` needs an independent check.
3. **Bin set B — real binner bins.** Download only `ERR9765746` (MOCK1 Illumina HiSeq 3000, **3.85 GB**), map with bowtie2/minimap2, run MetaBAT2 (+ optionally MaxBin2/CONCOCT + DAS Tool to mirror the authors). Gives **realistic contamination and chimerism**; truth from the same minimap2 alignment.
4. Truth per bin: `completeness = covered dominant-reference bp / dominant reference length × 100`; `contamination = bp aligning to other references / dominant reference length × 100`.
5. Run MAGICC V5, CheckM2, CoCoPyE, DeepCheck, GUNC on both bin sets. Report MAE + signed error + MIMAG confusion, stratified by technology, abundance decile, and reduced-genome status.
6. **Report two cohorts:** all 71 organisms, and the **34 organisms whose reference is not in TRAIN/VAL** as the primary leakage-free result.

**Compute.** minimap2 7 assemblies × 91 refs at 6 threads ≈ **1.5–3 CPU-h**. Read mapping ERR9765746 (6.1 Gbp) at 6 threads ≈ **3–5 CPU-h**; MetaBAT2 ≈ 0.5 CPU-h. MAGICC on ≤ 500 bins: **< 2 min**. CheckM2 on ~500 bins at 0.8 genomes/min/thread: **~2 CPU-h wall at 6 threads / 20 min at 32**. GUNC ~500 genomes ≈ 2–4 CPU-h. **Total ≈ 12–18 CPU-h.**
**Disk.** +4 GB reads, +10 GB intermediates. Already-downloaded 604 MB stays.
**What it lets us claim.** "On three real mock communities of 64–87 known strains spanning 29 phyla, assembled by seven independent sequencing technologies, MAGICC's completeness/contamination MAE against exact known composition was X/Y, versus CheckM2's …" — a genuine, independent, real-data validation with ground truth. This is the replacement for the withdrawn §4.4e claims.

---

### Rank 2 — **NCBI same-BioSample draft/complete pairs** 🟡 L2 · **largest ground-truth N, cheapest per genome**
1,097 prokaryotic pairs in the 0.80–1.20 size band across 165 species (4,834 pairs / 3,666 species before the mandatory bacteria/archaea filter). Tables and FTP URLs already emitted by `scripts/130_ncbi_draft_complete_pairs.py`.

**Pipeline.** Download drafts+completes (~2.6 GB) → minimap2 `asm5`/`asm10` draft→complete → completeness = covered ref bp / ref length; contamination = unaligned draft bp / ref length (report as an **upper bound**) → MAGICC + CheckM2 (+ GUNC on the tail).
**Compute.** Download ≈ 20 min. minimap2 1,097 pairs at 4 threads ≈ **2–3 CPU-h**. MAGICC **< 1 min**. CheckM2 1,097 genomes = **~6 CPU-h wall at 6 threads / ~45 min at 32**. **Total ≈ 10 CPU-h.**
**Disk.** ~6 GB.
**What it lets us claim.** "Across 1,097 isolate draft assemblies whose complete reference genome derives from the *same BioSample*, MAGICC's completeness error was X pp and its contamination false-positive rate at the 5 % MIMAG threshold was Y%." This is a **large-N, high-taxonomic-diversity, real-assembly test of exactly the failure mode we found on `set_C_clean`**, and 79 % of pairs are species not named in training.
**Optional extension (cheap, high value):** the **23,048 NCBI `contaminated`-flagged assemblies** as an independent *detection* positive-control set (binary label, precision/recall at 5 %/10 %). Add 1,000 flagged + 1,000 matched unflagged: +2 GB, +6 CPU-h.
**Optional Tier 2** (same strain name, different BioSample), species-capped at 2/species → **1,687 pairs across 1,338 species** including 83 archaeal, ~4 GB, +8 CPU-h. Use only as a sensitivity analysis; strain divergence contaminates the contamination estimate.

---

### Rank 3 — **Zymo isolate drafts (8 pairs)** 🟢 L1 · **do immediately, essentially free**
Everything is on disk. Split `Zymo-Isolates-SPAdes-Illumina.fasta` by the 2-letter prefix, drop the two eukaryotes, minimap2 each of the 8 drafts against its Zymo complete reference.
**Compute:** **< 15 CPU-min** total (plus CheckM2 on 8 genomes, ~2 min at 6 threads).
**Claim:** a small, crisp, fully-independent real-isolate check in the ≈95–99 % completeness / ≈0 % contamination regime. n = 8 is too small to stand alone but it is a clean sanity exhibit and it names *Zymo standards* explicitly, which Reviewer 2 asked for.

---

### Rank 4 — **UHGG + GTDB reviewer-genera reproduction (WS3.3)** 🟡 L2 · **mandatory, but disagreement only**
This is a *falsification test we are obliged to run*, not validation.

**Cohort 4a — GTDB (free comparator).** 298 genomes across all five named genera, with **published CheckM2** values already on disk (`data/gtdb/*_metadata.tsv.gz`). Download 298 FASTAs (~0.4 GB), run MAGICC, compute per-genus median deltas. **No CheckM2 run needed.** ≈ 1 CPU-h.
**Cohort 4b — UHGG (the reviewer's likely source).** 126 genomes for the three genera present (CAG-557 89, UMGS1491 30, HGM10766 7). `Caccenecus` and `Faecimonas` are **absent from UHGG v2.0.2** (GTDB r202 taxonomy) — this is a reportable finding. UHGG ships **CheckM v1**, not CheckM2, so CheckM2 must be run ourselves on this subset. ≈ 1 CPU-h + 30 CPU-min CheckM2.
**Cohort 4c — broad UHGG background.** 5,000–20,000 genomes stratified by lineage, genome size, GC, contig count and reduced-genome status, to place the five genera in context and satisfy WS3.4. **5,000 genomes = 5.6 GB download, MAGICC < 5 min, CheckM2 ≈ 3.3 h at 32 threads (17 h at 6 threads).** 20,000 genomes = 22 GB and ≈ 13 h CheckM2 at 32 threads.
**Recommendation: 5,000 first.** Extend to 20,000 only if the 5,000-genome result is ambiguous.
**Note:** the exact catalogue the reviewer used is **not identified** — their implied comparator completeness baseline (~63 % for CAG-557) matches neither UHGG-CheckM1 (77.2) nor GTDB-CheckM2 (97.4). Budget **1 h** to check SPIRE and HRGM2 before running 4c. In the response we reproduce *direction and magnitude*, name our catalogue, and do not claim to reproduce their digits.
**Claim ceiling:** "MAGICC and CheckM2 disagree systematically on reduced-genome gut lineages; the disagreement is concentrated in [strata]; no ground truth exists for this cohort, so we do not assert which tool is correct." Combined with Rank 1/2/3, we can additionally say *"on genomes of comparable size and fragmentation where ground truth does exist, MAGICC's error is X"* — which is the honest way to convert a disagreement into information.

---

### Rank 5 — **CAMI II** 🟠 L3 · independent-procedure external benchmark (Reviewer 1 M3)
Openly downloadable (no request form). Recommended minimal slice: **marine, samples 0–4**.
`marmgCAMI2_genomes.tar.gz` (836 MB) + 5 × `*_contigs.tar.gz` (~1.9 GB) = **~2.7 GB** → gold-standard bins with exact completeness truth and 0 % contamination.
Add `strain/` (strain-madness, 452 MB genomes + ~1 GB contigs) as the **strain-heterogeneity stress case** — directly relevant to Reviewer 1's "close contaminants" point and to WS2.
**Compute.** Untar/parse + per-genome bin extraction ≈ 2 CPU-h. MAGICC on ~1,700 bins < 5 min. CheckM2 on 1,700 ≈ 9 CPU-h at 6 threads / 1.1 h at 32. **Total ≈ 12 CPU-h.**
**If a contamination axis is wanted**, add `marmgCAMI2_sample_0_bam.tar.gz` (4.9 GB) and run MetaBAT2 on the gold-standard assembly: +6 GB disk, +4 CPU-h.
**Claim.** "On CAMI II gold-standard genome bins — simulated by CAMISIM, a procedure entirely independent of our generator — MAGICC's completeness MAE was X." Strong answer to "synthetic data generated under assumptions similar to the training setup".

---

### Rank 6 — **SAGs** 🟡 L2 · **run it, but expect to scope the claim down**
Two cohorts, in this order:
1. **Reference-anchored (bounded truth), n = 453.** GTDB `derived from single cell` SAGs whose species has a complete GenBank reference. Download SAGs + one complete reference per species (~1.5 GB), minimap2, filter to ANI ≥ 97–98 %, derive bounded completeness/contamination. ≈ 4 CPU-h + 45 min CheckM2 at 32 threads.
2. **Agreement-only, n = 3,151 + 1,596.** GORG-Tropics SAGs present in GTDB (published CheckM2) and the full GTDB SAG set. ~2.5 GB, MAGICC minutes, no CheckM2 needed (values published).
**The number that decides the claim:** only **3,151 of 12,715 (24.8 %)** GORG-Tropics SAGs reach completeness ≥ 50 %; **~75 % lie below MAGICC's 50 % floor and cannot be scored at all.**
**Recommended wording** (satisfies R1-M9 either way): report the 453-SAG bounded-truth result and the agreement statistics, and state plainly that *MAGICC is applicable to SAGs only above its 50 % completeness floor, which excludes roughly three quarters of a representative marine SAG collection.* If the 453-SAG result is good, the claim becomes "validated within the ≥50 % completeness regime on n = 453 SAGs"; if not, it becomes an explicit limitation. **Do not leave the manuscript's current unqualified SAG claim standing.**

---

### Rank 7 — **ATCC, direct** 🔴 not recommended
Public data for MSA-1003 is **PacBio Sequel II only** (2 runs, 31 GB) and the matching reference genomes sit behind the **ATCC Genome Portal Data Use Agreement**, which we cannot re-distribute under a Nature Communications data-availability statement. **Covered instead via Meslier**, which embeds ATCC MSA-1002 and 22 ATCC-designated strains with public GenBank accessions. Report this to the reviewer as the reason.

---

## 2. Consolidated budget

| Track | Datasets | Download | Disk (peak) | CPU-h | Ground truth |
|---|---|---|---|---|---|
| **A (recommended now)** | Meslier MOCK1 (Rank 1) + NCBI Tier-1 pairs (Rank 2) + Zymo isolates (Rank 3) | ~8 GB | ~25 GB | **~30** | **YES, all three** |
| **B** | WS3.3 reproduction: GTDB 298 + UHGG 126 + UHGG 5,000 background (Rank 4) | ~6 GB | ~15 GB | **~20** (CheckM2 at 32 threads) or ~35 (at 6) | no |
| **C** | CAMI II marine + strain-madness (Rank 5) | ~5 GB | ~20 GB | **~15** | YES |
| **D** | SAGs (Rank 6) | ~4 GB | ~10 GB | **~6** | partial (n=453) |
| **E (optional)** | NCBI `contaminated` positive controls + Tier-2 sensitivity + Meslier MOCK2/3 + Zymo community MAGs | ~25 GB | ~60 GB | **~30** | YES (detection label / bounded) |
| **Total A–E** | | **~48 GB** | **~130 GB** | **~100** | |

Disk headroom is 2.4 TB — not a constraint. The binding constraint is **CheckM2 wall-clock at low thread counts** (0.8 genomes/min/thread). Every plan above is written so that CheckM2 is the only expensive comparator and is applied to the smallest defensible cohort.

**Thread policy while the GPU/CheckM2/Kraken2 jobs are running:** ≤ 6 threads, i.e. Track A alone takes ~5 wall-hours of the ~30 CPU-h. Recommend deferring Track B's 5,000-genome CheckM2 run until the machine frees up, or reusing GTDB's published CheckM2 values (Cohort 4a) which need no run at all.

---

## 3. Reviewer 2 Major comment 3 — feasibility verdict, item by item

| Reviewer 2 asked for | Feasible? | How / why not |
|---|---|---|
| "one or more **large public MAG collections**, compare with CheckM2" | **YES** | UHGG v2.0.2, 289,232 genomes, metadata on disk; 5,000-genome stratified subset. UHGG ships **CheckM1**, so CheckM2 must be run by us (3.3 h at 32 threads). Cheaper alternative with published CheckM2: GTDB MAG subset (351,646 MAGs with `checkm2_*` columns). **No ground truth — disagreement characterisation only.** |
| "quantify the proportion of genomes whose quality classification changes" | **YES** | MIMAG reclassification matrix on the same cohort. Already-built machinery: `scripts/101`–`105`. |
| "identify taxonomic groups / genome types where disagreements are concentrated" | **YES** | UHGG metadata has lineage, length, N50, contigs, GC, MAG-vs-isolate, country. Stratification is a table join. |
| "investigate representative cases with large discrepancies" | **YES** | GUNC (installed, `gunc_env`) + Kraken2 + SCG duplication already exist as `scripts/76`–`80`. **Caveat established in §4.4e: none of these adjudicate novel lineages.** Present them as characterisation, not proof. |
| "**CAG-557, UMGS1491, Caccenecus, HGM10766**" (and Faecimonas) | **PARTIALLY, and we can say exactly why** | **UHGG v2.0.2 contains 3 of 5**: CAG-557 (89 genomes), UMGS1491 (30), HGM10766 (7). **Caccenecus and Faecimonas are absent** — UHGG uses GTDB r202 and those names postdate it. **GTDB contains all five (298 genomes) with published CheckM2 values**, so the full observation *is* testable there. All five are reduced-genome MAGs (0.98–1.41 Mbp median), which is consistent with the reviewer's finding being real. The reviewer's exact catalogue is unidentified (their implied comparator baseline matches neither UHGG-CheckM1 nor GTDB-CheckM2) — we will name ours. |
| "**ATCC** … standards" | **INDIRECTLY** | ATCC reference genomes are gated behind the ATCC Genome Portal Data Use Agreement; MSA-1003's only public reads are PacBio. **But ATCC MSA-1002 is a component of Meslier MOCK1/2/3**, and 22 of the 91 Meslier references are ATCC strains with public GenBank accessions. We therefore cover ATCC strains without the licence problem, and say so. |
| "**Zymo** standards" | **YES, fully** | Complete references for D6300/D6310/D6331/D6320 are open S3 downloads (already on disk). **8 isolate SPAdes drafts with matched complete references are already on disk.** Community reads (Even 1.96 GB, Log 6.54 GB) available if MAGs are wanted; the Log standard's 6-order-of-magnitude gradient would give a full completeness range. |
| "**Meslier et al. 2022**" | **YES — best resource found** | 91 references (86 complete), exact abundances for 3 mocks, **7 pre-computed MOCK1 assemblies**, 100 % downloaded, open licence, no assembly compute required. Two corrections to record: it is a **mock-community** paper (not SAGs, as our protocol WS3.7 mis-stated), and its **Table 4 Illumina accessions are misprinted** (ERR9765446-49 do not exist; use ERR9765746-49). |
| "draft genomes, MAGs and SAGs **derived from isolates** for which complete references are available" | **YES, at scale** | 1,097 prokaryotic same-BioSample draft/complete pairs (165 species) + 1,687 species-capped same-strain pairs (1,338 species incl. 83 archaeal) mined from `assembly_summary_genbank.txt`. This is the largest genuine-ground-truth cohort available and needs no assembly. |
| "validation on these genome types should occupy a **much more prominent place**" | **YES** | Ranks 1–3 + 5 are all ground-truth real-data or independent-procedure benchmarks. Recommend they become a top-level Results section, replacing the withdrawn §4.4e material. |
| **SAGs** (also Reviewer 1 M9) | **PARTIALLY — claim must be scoped** | Bounded truth for **453** GTDB SAGs with a conspecific complete reference; agreement-only for 3,151 GORG-Tropics + 1,596 GTDB SAGs. **~75 % of GORG-Tropics SAGs are below MAGICC's 50 % completeness floor and cannot be scored.** The manuscript's SAG claim must be either narrowed to the ≥50 % regime with n = 453 evidence, or marked unvalidated. |

---

## 4. Decisions requested from the coordinator

1. **Approve Track A now** (Meslier MOCK1 + NCBI Tier-1 pairs + Zymo isolates): ~8 GB download, ~25 GB disk, ~30 CPU-h at ≤6 threads. All three give genuine ground truth. This is the direct replacement for the withdrawn real-MAG contamination claims.
2. **Choose the WS3.3 comparator.** (a) GTDB's **published** CheckM2 values — free, covers all five reviewer genera, 298 genomes; or (b) UHGG + our own CheckM2 run — matches the reviewer's likely source but covers only 3 of 5 genera and costs 3.3 h at 32 threads for a 5,000-genome background. **Recommendation: do (a) immediately and (b) when threads free up.**
3. **Confirm 5,000 (not 20,000) for the UHGG background cohort** in the first pass. 20,000 quadruples CheckM2 cost for a characterisation-only result.
4. **Authorise 1 h to check SPIRE and HRGM2** before Track B, to try to identify the reviewer's exact catalogue. If neither matches, we state ours and reproduce direction/magnitude.
5. **Ratify the two protocol corrections:** WS3.7 must stop describing **Meslier 2022 as a SAG dataset** (it is a mock community; move it to WS3.6), and WS3.7's SAG work should point at **GORG-Tropics + the 453-SAG GTDB reference-anchored cohort**.
6. **Ratify the ATCC position:** direct ATCC use is blocked by the Data Use Agreement; ATCC coverage comes via Meslier. Needs to be stated in the point-by-point response.
7. **Ratify the leakage-reporting rule for Meslier:** report all 71 MOCK1 organisms **and** the 34 whose reference is not in TRAIN/VAL, with the 34 as primary.
8. **Decide whether to add the NCBI `contaminated`-flag cohort** (23,048 assemblies available; proposed 1,000 + 1,000 matched controls, +2 GB, +6 CPU-h) as an independent binary *detection* positive control. It is cheap and no reviewer can call it circular.

---

## 5. Scripts to be written for Phase 2 (numbering continues from 130)

| Script | Purpose |
|---|---|
| `130_ncbi_draft_complete_pairs.py` | ✅ **done** — pair discovery from `assembly_summary_genbank.txt` |
| `131_fetch_meslier_mock.py` | ✅ effectively done by hand; formalise the download + checksum manifest |
| `132_mock_reference_alignment.py` | minimap2 contigs → 91 references; per-contig best-hit table |
| `133_mock_derive_bins_and_truth.py` | reference-anchored bins (set A) + truth from alignment |
| `134_mock_binning_metabat.sh` | ERR9765746 mapping + MetaBAT2/DAS Tool bins (set B) |
| `135_fetch_ncbi_pairs.py` | download Tier-1 draft+complete FASTAs from the emitted FTP paths |
| `136_pair_truth_from_alignment.py` | draft→complete alignment; completeness + unaligned-bp upper bound |
| `137_zymo_isolate_split_and_truth.py` | split the 10-isolate multi-FASTA; align 8 bacteria to Zymo references |
| `138_fetch_uhgg_subset.py` | stratified UHGG subset; GFF→FASTA extraction |
| `139_reviewer_genera_cohort.py` | GTDB 298 + UHGG 126 cohorts; per-genus median deltas (WS3.3) |
| `140_fetch_cami2.py` / `141_cami2_bins_and_truth.py` | CAMI II slice + gold-standard bin extraction |
| `142_sag_cohorts.py` / `143_sag_reference_anchored_truth.py` | GORG-Tropics + GTDB SAG cohorts; 453-SAG conspecific-reference truth |
| `144_real_data_synthesis.py` | cross-dataset synthesis figure/table (WS3.9) |
