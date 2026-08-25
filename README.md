# MAGICC benchmark data

Benchmark metadata, split files, provenance audits and generator scripts for
[MAGICC](https://github.com/renmaotian/magicc), the genome
completeness/contamination estimator.

> ## ⚠️ Read this first: benchmark sets C and D have been WITHDRAWN
>
> The sets published here as `benchmark/set_C` and `benchmark/set_D` up to
> 2026-08 were built from reference genomes that **overlapped MAGICC's own
> training split**: 1,000/1,000 (**100 %**) of set C's dominant genomes and
> 903/1,000 (**90.3 %**) of set D's were in train or validation. Every number
> derived from them is invalid and is withdrawn from the manuscript.
>
> They now live under **[`withdrawn/`](withdrawn/)**, each with a
> `WITHDRAWN.md` giving the audited counts, the root cause and the size of the
> effect. **No number from `withdrawn/` may be cited.**
>
> Their replacements are **[`benchmark/set_C_clean`](benchmark/set_C_clean)**
> and **[`benchmark/set_D_clean`](benchmark/set_D_clean)**, rebuilt from
> held-out test-split references only (audited: 0 samples in train, 0 in
> validation, 0 in the k-mer feature-selection set).

## What is here, and what is at figshare

Genome FASTA files are far too large for git (the benchmark sets alone are
123 GB). This repository holds everything *about* the datasets; the assemblies
themselves are deposited at figshare.

| | Where |
|---|---|
| Per-sample ground-truth metadata, generation metadata and seeds, reference-selection tables, per-tool predictions | **here**, under `benchmark/` |
| Split accession lists and split statistics | **here**, under `splits/` |
| Provenance / leakage audits and SHA256 manifests | **here**, under `provenance/` |
| CAMI II sample and source-genome identifiers, derived truth tables, bin definitions, leakage audit, competitor cohorts | **here**, under `cami2/` |
| Generator and audit scripts | **here**, under `data_generating_scripts/` |
| **Benchmark genome assemblies (FASTA)** | figshare, DOI `[DOI]` |
| **Full per-genome result files, statistics, figures** | figshare, DOI `[DOI]` |
| **Software, released model, reproduction workflow, containers** | [github.com/renmaotian/magicc](https://github.com/renmaotian/magicc) |
| **Analysis, training and figure code** | released with the paper |

No third-party sequence is redistributed here. CAMI II data are openly
available from <https://frl.publisso.de/data/frl:6425521/>; only identifiers
and derived tables are deposited in this repository.

## Benchmark sets

All sets are synthetic assemblies built from real reference genomes by
controlled fragmentation and contamination simulation. **Every set below draws
its dominant reference genomes from the held-out test split**; this is the
property whose absence invalidated the withdrawn sets, and it is audited in
[`provenance/overlap_summary.tsv`](provenance/overlap_summary.tsv).

| Set | n | Dominants | Design |
|---|---|---|---|
| [`set_A`](benchmark/set_A) | 1,000 | finished test-split genomes (798 unique) | completeness gradient, 50–100 % in 6 levels, 0 % contamination |
| [`set_B`](benchmark/set_B) | 1,000 | finished test-split genomes (803 unique) | contamination gradient, 0–80 % in 5 levels, 100 % completeness |
| [`set_C_clean`](benchmark/set_C_clean) | 1,000 | **100** test-split Patescibacteriota (CPR) references × 10 simulations | uniform completeness 50–100 %, uniform contamination 0–100 % |
| [`set_D_clean`](benchmark/set_D_clean) | 1,000 | **100** test-split archaeal references × 10 simulations | uniform completeness 50–100 %, uniform contamination 0–100 % |
| [`set_E`](benchmark/set_E) | 1,000 | finished test-split genomes (785 unique) | realistic mixed: 200 pure, 200 complete-but-contaminated, 600 mixed |
| [`set_F`](benchmark/set_F) | 1,900 | test-split references | contamination **type** × donor **taxonomic distance** factorial (`design.tsv`) |
| [`set_G`](benchmark/set_G) | 1,920 | test-split references | sequencing and assembly **error robustness** gradient (`design.tsv`) |
| [`set_H`](benchmark/set_H) | 4,000 | 400 NCBI-selected references in **200 matched pairs** | circularity safeguard on references never filtered by CheckM2 |

Sets A, B, C-clean, D-clean and E together form the **leakage-free five-set
panel** (5,000 assemblies) on which the headline benchmark is computed, and
which the reproduction workflow in the code repository regenerates end to end
from one command.

**Because `set_C_clean`, `set_D_clean`, `set_F`, `set_G` and `set_H` reuse each
reference genome across several simulations, any statistic computed on them
must be clustered by `dominant_accession`.** A naive i.i.d. bootstrap achieves
54 % coverage where a cluster bootstrap achieves 95 %.

### Set H: reference-selection material

`benchmark/set_H` additionally ships `reference_selection_final.tsv` (the 400
selected references and their pairing) and `candidate_pool.tsv.gz` (the full
candidate pool with would-pass / would-fail labels for the original filter), so
the selection can be re-derived rather than taken on trust.

### Motivating sets

[`motivating/`](motivating/) holds the three sets behind the motivating
analysis (Fig. 1): a completeness gradient, a contamination gradient and a
realistic mixed set, 1,000 genomes each, all from finished test-split
references. They are audited in the same table and carry no leakage.

## Per-set file inventory

| File | Contents |
|---|---|
| `metadata.tsv` | one row per simulated genome: `genome_id`, `true_completeness`, `true_contamination`, `dominant_accession`, `dominant_phylum`, `sample_type`, `n_contigs`, `total_length`, plus set-specific design columns |
| `labels.npy` | the same ground truth as an (n, 2) float array `[completeness, contamination]` |
| `generation_metadata.tsv` | full generation record where one exists, including the **literal per-sample RNG seed** (`seed` column) |
| `design.tsv` | the factorial design (sets F and G) |
| `reference_selection.tsv` | the reference genomes chosen and why |
| `checkm2_predictions.tsv`, `cocopye_predictions.tsv`, `deepcheck_predictions.tsv`, `magicc_v5_predictions.tsv` | per-genome predictions from each tool |

Sets whose per-sample seeds were not persisted are reproducible from a recorded
rule instead — `default_rng(BASE + row_index + OFFSET)`, with every constant
released in [`provenance/seed_provenance.tsv`](provenance/seed_provenance.tsv).
That table states, for all sets, which of the two tiers applies.

## Quality metric definitions

Ground truth is **sequence-based**, not gene-based, and both metrics share one
denominator:

- **Completeness (%)** = retained dominant-genome bp ÷ **full reference length
  of the dominant genome** × 100.
- **Contamination (%)** = total contaminant bp ÷ **the same full reference
  length** × 100.

Using the full reference length for both makes the two metrics independent: a
genome at 60 % completeness with 20 % contamination contains contaminant DNA
equal to 20 % of the original reference size, regardless of how much dominant
sequence survived. These definitions correspond to CheckM2's protein-based
ones (completeness = annotated proteins ÷ complete-genome proteins;
contamination = duplicated proteins ÷ complete-genome proteins) under an
approximately uniform protein density.

## Splits

[`splits/`](splits/) holds the training / validation / test accession lists,
the split statistics, and the list of accessions that failed to download.
Splits are stratified by phylum and mutually disjoint.

| Split | accessions selected | genomes available |
|---|---|---|
| train | 79,978 | **79,948** |
| val | 10,017 | **10,010** |
| test | 10,005 | **9,999** |

The 43-genome difference is the download failures listed in
`splits/missing_accessions.txt`; they were never used for training, feature
selection or benchmarking. **The counts quoted in the manuscript are the
available-genome counts.** See [`splits/README.md`](splits/README.md) for the
GTDB-vs-NCBI accession join, whose naive form is precisely the mistake that
originally hid the set C/D leakage.

## Provenance and the leakage audit

[`provenance/`](provenance/) is the audit trail.

| File | Contents |
|---|---|
| [`README.md`](provenance/README.md) | the full provenance-audit report: method, accession cross-mapping, findings |
| [`overlap_summary.tsv`](provenance/overlap_summary.tsv) | **the headline table** — for every set: unique dominants, how many samples fall in train / val / test, how many touched the k-mer feature-selection set, leakage %, and label-constraint violations |
| `audit_summary.json`, `accession_crossmap_stats.json` | machine-readable form |
| `set_*_dominants.txt`, `contaminants.txt` | the exact reference accessions used |
| `sha256_manifest.txt`, `set_{F,G,H}_sha256_manifest.txt` | SHA256 of every generated assembly |
| [`seed_provenance.tsv`](provenance/seed_provenance.tsv) | RNG construction for every set |
| [`benchmark_inventory.tsv`](provenance/benchmark_inventory.tsv) | per-set status, generator, base seed, seed tier, metadata SHA256 |
| [`leakage_specificity_control.tsv`](provenance/leakage_specificity_control.tsv) | the control showing the effect is leakage, not lineage |
| [`withdrawn_vs_clean_cd_metrics.tsv`](provenance/withdrawn_vs_clean_cd_metrics.tsv) | withdrawn vs clean accuracy under the *same frozen model* |

Accession matching is GCA/GCF-aware. Counts are identical under raw
GTDB-string matching, under prefix-and-version-stripped matching and under the
cross-map, so they are not normalisation artefacts.

Two further defects the audit surfaced, both recorded rather than smoothed
over: the withdrawn sets C and D contain 141 and 213 samples whose
contamination exceeds their completeness — outside the region the model was
ever trained on, because the training-domain constraint post-dates their
construction — and `set_E` contains 132 such samples.

## CAMI II

[`cami2/`](cami2/) covers the external benchmark: all 10 marine samples and 10
of the 100 strain-madness samples of the CAMI II short-read challenge.
**No CAMI II sequence is redistributed.** Obtain it from
<https://frl.publisso.de/data/frl:6425521/>.

| Path | Contents |
|---|---|
| `truth/*_truth.tsv` | derived per-bin truth: source genome, taxid, reference length, spanned bp, contig count, CAMI novelty category, completeness and contamination under the definitions above, and the exclusion / scoreability flags |
| `truth/*_derivation.json` | how the truth was derived from CAMI II's own `gsa_mapping.tsv` contig→source-genome assignment |
| `provenance/*_source_genomes.tsv`, `*_accessions_matched.txt` | the source-genome identifiers |
| `provenance/*_leaked.txt`, `*_leakage_summary.json`, `*_species_vs_training.tsv` | the leakage audit against MAGICC's training split |
| `provenance/*_competitor_cohort.tsv` | membership of the subsampled cohorts the comparator tools were run on |
| `provenance/sha256_manifest.txt` | checksums |
| `cami2_censoring.tsv` | bin counts and censoring rates per bin set, **with the denominator spelled out in a column** |

**Denominators differ between tables, deliberately.** The truth tables hold
14,874 rows in total; **10,512 bins were actually written and scored**
(marine gold 4,497, marine mixed 2,201, strain-madness gold 1,564,
strain-madness mixed 2,250), and 5,406 of the 6,359 marine gold truth rows are
genuine-genome bins (953 are flagged `excluded_non_genome`). Any percentage
computed from these files must state which denominator it uses;
`cami2_censoring.tsv` reports both the scored-bin and all-truth-row versions
side by side.

## Generator scripts

[`data_generating_scripts/`](data_generating_scripts/) contains the scripts
that built the sets and audited them, including the original
`25_benchmark_generate.py` — **whose module docstring is where the set C/D
leakage is visible in the source**, drawing dominants from "train+val+test".
It is kept for exactly that reason.

[`all_project_scripts/`](all_project_scripts/) is a snapshot of the earlier
project pipeline (scripts 01–44) and is retained for continuity; it predates
this revision. The complete, current analysis code is released with the paper.

Scripts contain workspace-relative paths of the form `/path/to/magicc/...`,
which must be repointed at a local checkout before they will run.

## Citation

Tian, R., Zhou, J., Imanian, B. MAGICC: genome quality assessment from
single-copy core-gene k-mer profiles. *Manuscript under revision.*

## License

MIT License. See [LICENSE](LICENSE).
