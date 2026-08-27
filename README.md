# MAGICC benchmark data, result files and analysis code

Everything the Nature Communications manuscript's **Data availability** and
**Code availability** statements promise, in one place: the benchmark genome
assemblies (as release assets), the per-sample ground truth, the provenance and
leakage audits, every result file behind every reported number, and the complete
analysis code.

The software itself — the released model, the reproduction workflow and the
containers — is at [github.com/renmaotian/magicc](https://github.com/renmaotian/magicc),
current release tag `v0.3.3`. Every result deposited here was produced with the
code of tag `v0.3.1` and model V5, SHA256
`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`; `v0.3.2`
and `v0.3.3` change only how that model is fetched and checksummed, and their
predictions are byte-identical to `v0.3.1`'s.

---

> # ⚠️ Withdrawn datasets: benchmark sets C and D
>
> The sets published here as `benchmark/set_C` and `benchmark/set_D` up to
> 2026-08, and still served as `WITHDRAWN_benchmark_set_C.tar.gz` and
> `WITHDRAWN_benchmark_set_D.tar.gz` on release
> [**v0.1.0**](https://github.com/renmaotian/magicc-data/releases/tag/v0.1.0),
> were built from reference genomes that **overlapped MAGICC's own training
> split**:
>
> * **set C — 1,000 of 1,000 (100 %)** dominant genomes were training-split
>   genomes; 23 also entered the 9-mer feature-selection set.
> * **set D — 796 of 1,000 in train, 107 in validation, 97 in test (90.3 %
>   leakage)**; 511 entered feature selection.
>
> The root cause is visible in the generator's own docstring
> ([`scripts/025_benchmark_generate.py`](scripts/025_benchmark_generate.py)):
> dominants were drawn from "train+val+test". **Every number derived from these
> two sets is invalid and is withdrawn from the manuscript. No number from them
> may be cited.**
>
> **Their leakage-free replacements are
> [`benchmark/set_C_clean`](benchmark/set_C_clean) and
> [`benchmark/set_D_clean`](benchmark/set_D_clean)**, rebuilt from held-out
> test-split references only and audited at 0 samples in train, 0 in validation
> and 0 in the feature-selection set. Their assemblies are release assets
> `set_C_clean.tar.gz` and `set_D_clean.tar.gz` on
> [**v1.0.0**](https://github.com/renmaotian/magicc-data/releases/tag/v1.0.0).
>
> The withdrawn sets are **retained, not deleted**, under
> [`withdrawn/`](withdrawn/) and on `v0.1.0` under their `WITHDRAWN_` names, so
> that the leakage stays auditable and the size of its effect stays measurable
> ([`provenance/withdrawn_vs_clean_cd_metrics.tsv`](provenance/withdrawn_vs_clean_cd_metrics.tsv)
> compares the withdrawn and clean sets under the *same frozen model*).

---

## Get the benchmark assemblies

The assemblies are far too large for git, so they ship as **GitHub Release
assets** on tag [`v1.0.0`](https://github.com/renmaotian/magicc-data/releases/tag/v1.0.0).
Any archive over 1.8 GB is uploaded as 1.5 GiB parts, safely under GitHub's
2 GiB per-asset limit; `download_benchmarks.sh` concatenates them for you.

```bash
git clone https://github.com/renmaotian/magicc-data
cd magicc-data
bash download_benchmarks.sh                    # all eight sets, ~15 GB
bash download_benchmarks.sh set_C_clean set_E  # or just the ones you need
```

The script downloads, concatenates any parts, verifies every file against the
published SHA256 manifests, extracts, and is safe to re-run — it resumes rather
than restarting. Add `--full-verify` to check every one of the 12,820 assemblies
against `provenance/assemblies/`. To verify by hand:

```bash
cd benchmarks                          # wherever the assets were downloaded
sha256sum -c ../SHA256SUMS             # the assets exactly as uploaded
cat set_H.tar.gz.part?? > set_H.tar.gz
sha256sum -c ../SHA256SUMS.tar         # the reassembled archives
tar -xzf set_H.tar.gz
sha256sum -c ../provenance/assemblies/set_H_sha256.txt   # all 4,000 of its assemblies
```

### Release v1.0.0 — asset inventory

| Set | Assemblies | Release asset(s) | Size | Design |
|---|---|---|---|---|
| **set_A** | 1,000 | `set_A.tar.gz` | 914 MB | completeness gradient, 50-100 % in six levels, 0 % contamination |
| **set_B** | 1,000 | `set_B.tar.gz` | 1.66 GB | contamination gradient, 0-80 % in five levels, 100 % completeness |
| **set_C_clean** | 1,000 | `set_C_clean.tar.gz` | 355 MB | 100 test-split Patescibacteriota (CPR) references x 10 simulations |
| **set_D_clean** | 1,000 | `set_D_clean.tar.gz` | 839 MB | 100 test-split archaeal references x 10 simulations |
| **set_E** | 1,000 | `set_E.tar.gz` | 1.46 GB | realistic mixed: 200 pure, 200 complete-but-contaminated, 600 mixed |
| **set_F** | 1,900 | `set_F.tar.gz.part00`<br>`set_F.tar.gz.part01` | 1.91 GB | contamination type x donor taxonomic distance factorial |
| **set_G** | 1,920 | `set_G.tar.gz` | 1.62 GB | sequencing and assembly error-robustness gradient |
| **set_H** | 4,000 | `set_H.tar.gz.part00`<br>`set_H.tar.gz.part01`<br>`set_H.tar.gz.part02`<br>`set_H.tar.gz.part03` | 5.99 GB | circularity safeguard: 400 NCBI references in 200 matched pairs |
| | **12,820** | **12 assets** + `SHA256SUMS` + `SHA256SUMS.tar` | **14.75 GB** (from 49.13 GB of FASTA) | |

Each archive expands to `<set>/metadata.tsv`, the generation metadata where it
exists, and `<set>/fasta/` — one FASTA per simulated assembly.

### Release v0.1.0 — retained, relabelled

`v0.1.0` (February 2026) predates the leakage audit. It is kept so that nothing
that was ever downloadable disappears, but every asset is now labelled for what
it is. Its two withdrawn assets carry a `WITHDRAWN_` prefix. Of the other six,
three (A, B, E) hold current data superseded only in *packaging* — `v1.0.0`
re-ships the same assemblies with per-file checksums, split parts and a download
script — and the three `motivating_*` assets are not superseded at all.

Asset identity was established by evidence rather than by name: a ranged read of
each archive was decompressed far enough to list its first members, whose exact
byte sizes were matched against the analysis workspace.

| `v0.1.0` asset | Contents | Status |
|---|---|---|
| `benchmark_set_A.tar.gz` | the same 1,000 assemblies as `set_A.tar.gz` on v1.0.0 | current data, superseded packaging |
| `benchmark_set_B.tar.gz` | the same 1,000 assemblies as `set_B.tar.gz` on v1.0.0 | current data, superseded packaging |
| **`WITHDRAWN_benchmark_set_C.tar.gz`** | the leakage-contaminated set C | **WITHDRAWN — do not use** |
| **`WITHDRAWN_benchmark_set_D.tar.gz`** | the leakage-contaminated set D | **WITHDRAWN — do not use** |
| `benchmark_set_E.tar.gz` | the same 1,000 assemblies as `set_E.tar.gz` on v1.0.0 | current data, superseded packaging |
| `motivating_set_A.tar.gz` | motivating completeness gradient, 1,000 assemblies | **current** — the only deposition of these assemblies |
| `motivating_set_B.tar.gz` | motivating contamination gradient, 1,000 assemblies | **current** |
| `motivating_set_C.tar.gz` | motivating realistic mixed set, 1,000 assemblies | **current** |

The three `motivating_*` assets are **not** superseded: they hold the assemblies
behind the motivating analysis of Figure 1 and are not re-uploaded to `v1.0.0`.
Their per-sample metadata and predictions are in [`motivating/`](motivating/).

---

## What is in this repository

| Path | Contents |
|---|---|
| [`benchmark/`](benchmark/) | per-sample ground truth, generation metadata and seeds, reference-selection tables and per-genome predictions from all four tools, for the eight benchmark sets |
| [`withdrawn/`](withdrawn/) | the two withdrawn sets, prominently labelled, with the audited counts and the root cause |
| [`motivating/`](motivating/) | the three motivating sets behind Figure 1 |
| [`splits/`](splits/) | train / validation / test accession lists, the full per-genome tables and the split statistics |
| [`curation/`](curation/) | the 277,183-row GTDB filtered-genome table the curation produced, and the 100,000-genome selection |
| [`provenance/`](provenance/) | the leakage audit, per-set dominant and contaminant accession lists, seed provenance, benchmark inventory and **SHA256 manifests for every assembly of every set** |
| [`cami2/`](cami2/) | CAMI II sample and source-genome identifiers, derived truth tables and bin definitions, the leakage audit and competitor-cohort membership. **No CAMI II sequence is redistributed** |
| [`results/`](results/) | every result file behind every reported number — see [`results/README.md`](results/README.md) for the inventory and for what is deliberately excluded |
| [`scripts/`](scripts/) | the complete analysis code, plus [`scripts/SCRIPT_MAPPING.tsv`](scripts/SCRIPT_MAPPING.tsv) |
| [`released_artefacts/`](released_artefacts/) | the selected 9-mer list, **both complete 131,072-k-mer prevalence tables**, the split files, seed provenance, normalization parameters and a SHA256 manifest |
| [`data_generating_scripts/`](data_generating_scripts/) | a convenience subset of `scripts/`: the generators and audits that built the benchmark sets |
| [`download_benchmarks.sh`](download_benchmarks.sh) | fetch and verify the release assets |
| `SHA256SUMS`, `SHA256SUMS.tar` | checksums of the release assets and of the reassembled archives |
| `benchmark_release_manifest.tsv` | one row per set: workspace directory, assembly count, raw and archive bytes, asset names and the archive checksum |
| [`DEPOSITION_STATUS.md`](DEPOSITION_STATUS.md) | **what is not here, and why** — outstanding items, deliberate non-deposition, and the known limitations of what is here |

---

## Benchmark sets

All eight sets are synthetic assemblies built from real reference genomes by
controlled fragmentation and contamination simulation. **Every set draws its
dominant reference genomes from the held-out test split** — the property whose
absence invalidated the withdrawn sets, audited in
[`provenance/overlap_summary.tsv`](provenance/overlap_summary.tsv).

| Set | n | Dominants | Design |
|---|---|---|---|
| [`set_A`](benchmark/set_A) | 1,000 | finished test-split genomes (798 unique) | completeness gradient, 50–100 % in 6 levels, 0 % contamination |
| [`set_B`](benchmark/set_B) | 1,000 | finished test-split genomes (803 unique) | contamination gradient, 0–80 % in 5 levels, 100 % completeness |
| [`set_C_clean`](benchmark/set_C_clean) | 1,000 | **100** test-split Patescibacteriota (CPR) references × 10 simulations | uniform completeness 50–100 %, uniform contamination 0–100 % |
| [`set_D_clean`](benchmark/set_D_clean) | 1,000 | **100** test-split archaeal references × 10 simulations | uniform completeness 50–100 %, uniform contamination 0–100 % |
| [`set_E`](benchmark/set_E) | 1,000 | finished test-split genomes (785 unique) | realistic mixed: 200 pure, 200 complete-but-contaminated, 600 mixed |
| [`set_F`](benchmark/set_F) | 1,900 | **100** test-split references | contamination **type** × donor **taxonomic distance** factorial ([`design.tsv`](benchmark/set_F/design.tsv)) |
| [`set_G`](benchmark/set_G) | 1,920 | **80** test-split references | sequencing and assembly **error robustness** gradient ([`design.tsv`](benchmark/set_G/design.tsv)) |
| [`set_H`](benchmark/set_H) | 4,000 | **400** NCBI-selected references in **200 matched pairs** | circularity safeguard on references never filtered by CheckM2 |

**12,820 assemblies in total.** Sets A, B, C-clean, D-clean and E form the
**leakage-free five-set panel** (5,000 assemblies) on which the headline
benchmark is computed, and which the reproduction workflow in the code
repository regenerates end to end from one command.

`set_A` here is the set the manuscript calls Set A; in the analysis workspace it
is the directory `set_A_v2`, and likewise `set_B` ↔ `set_B_v2` and `set_H` ↔
`set_H_ncbi`. The mapping is recorded in
[`provenance/benchmark_inventory.tsv`](provenance/benchmark_inventory.tsv) and in
the release manifest.

**Because `set_C_clean`, `set_D_clean`, `set_F`, `set_G` and `set_H` reuse each
reference genome across several simulations, any statistic computed on them must
be clustered by `dominant_accession`.** A naive i.i.d. bootstrap achieves 54 %
coverage where a cluster bootstrap achieves 95 %.

### Set H: reference-selection material

[`benchmark/set_H`](benchmark/set_H) additionally ships
`reference_selection_final.tsv` (the 400 selected references and their pairing)
and `candidate_pool.tsv.gz` (the full candidate pool with would-pass /
would-fail labels for the original filter), so the selection can be re-derived
rather than taken on trust. The 400 NCBI reference assemblies themselves are not
redistributed; their accessions are in
[`provenance/set_H_dominants.txt`](provenance/set_H_dominants.txt).

### Motivating sets

[`motivating/`](motivating/) holds the three sets behind the motivating analysis
(Figure 1): a completeness gradient, a contamination gradient and a realistic
mixed set, 1,000 genomes each, all from finished test-split references. They are
audited in the same table and carry no leakage. Their assemblies are the
`motivating_*` assets on release `v0.1.0`.

## Per-set file inventory

| File | Contents |
|---|---|
| `metadata.tsv` | one row per simulated genome: `genome_id`, `true_completeness`, `true_contamination`, `dominant_accession`, `dominant_phylum`, `sample_type`, `n_contigs`, `total_length`, plus set-specific design columns |
| `labels.npy` | the same ground truth as an (n, 2) float array `[completeness, contamination]` |
| `generation_metadata.tsv` | the full generation record — 43 to 48 columns including the literal per-sample RNG `seed`, the contaminant accessions and taxonomy, target *and* observed completeness and contamination, every fragmentation parameter, and the per-component contig counts. Present for sets C-clean, D-clean, F, G and H |
| `design.tsv` | the factorial design (sets F and G) |
| `reference_selection.tsv` | the reference genomes chosen and why |
| `checkm2_predictions.tsv`, `cocopye_predictions.tsv`, `deepcheck_predictions.tsv`, `magicc_v5_predictions.tsv` | per-genome predictions from each tool |

**Sets A, B and E predate the full generation record.** For them
`metadata.tsv` carries 9–11 columns — ground truth, dominant accession and
phylum, sample type, contig count, total length and the design target — and the
per-sample contaminant accessions and fragmentation parameters were not
persisted. What is released for them instead is the *rule* that reproduces them:
[`provenance/seed_provenance.tsv`](provenance/seed_provenance.tsv) gives the
generator, the base seed and the exact per-sample RNG construction
(`np.random.default_rng(BASE + row_index)`), so every sample is regenerable from
the released generator. That table states, for all sets, which of the two tiers
applies.

## Quality metric definitions

Ground truth is **sequence-based**, not gene-based, and both metrics share one
denominator:

- **Completeness (%)** = retained dominant-genome bp ÷ **full reference length of
  the dominant genome** × 100.
- **Contamination (%)** = total contaminant bp ÷ **the same full reference
  length** × 100.

Using the full reference length for both makes the two metrics independent: a
genome at 60 % completeness with 20 % contamination contains contaminant DNA
equal to 20 % of the original reference size, regardless of how much dominant
sequence survived. These definitions correspond to CheckM2's protein-based ones
(completeness = annotated proteins ÷ complete-genome proteins; contamination =
duplicated proteins ÷ complete-genome proteins) under an approximately uniform
protein density.

## Splits

[`splits/`](splits/) holds the training / validation / test accession lists, the
full per-genome tables, the split statistics, and the list of accessions that
failed to download. Splits are stratified by phylum and mutually disjoint.

| Split | accessions selected | genomes available |
|---|---|---|
| train | 79,978 | **79,948** |
| val | 10,017 | **10,010** |
| test | 10,005 | **9,999** |

The 43-genome difference is the download failures listed in
`splits/missing_accessions.txt`; they were never used for training, feature
selection or benchmarking. **The counts quoted in the manuscript are the
available-genome counts.** See [`splits/README.md`](splits/README.md) for the
GTDB-versus-NCBI accession join, whose naive form is precisely the mistake that
originally hid the set C/D leakage.

## Provenance and the leakage audit

[`provenance/`](provenance/) is the audit trail.

| File | Contents |
|---|---|
| [`README.md`](provenance/README.md) | the full provenance-audit report: method, accession cross-mapping, findings |
| [`overlap_summary.tsv`](provenance/overlap_summary.tsv) | **the headline table** — for every set: unique dominants, how many samples fall in train / val / test, how many touched the k-mer feature-selection set, leakage %, and label-constraint violations |
| `audit_summary.json`, `accession_crossmap_stats.json` | machine-readable form |
| `set_*_dominants.txt`, `contaminants.txt` | the exact reference accessions used |
| [`assemblies/`](provenance/assemblies/) | **12,848 SHA256 entries** — every assembly and every metadata file of all eight sets, with paths relative to the extracted archive, so `sha256sum -c provenance/assemblies/set_A_sha256.txt` works straight after `tar -xzf` |
| `sha256_manifest.txt`, `set_{A,B,E,F,G,H}_sha256_manifest.txt` | the original audit manifests, in workspace-relative paths, kept as written |
| [`seed_provenance.tsv`](provenance/seed_provenance.tsv) | RNG construction for every set |
| [`benchmark_inventory.tsv`](provenance/benchmark_inventory.tsv) | per-set status, generator, base seed, seed tier, metadata SHA256 |
| [`leakage_specificity_control.tsv`](provenance/leakage_specificity_control.tsv) | the control showing the effect is leakage, not lineage |
| [`withdrawn_vs_clean_cd_metrics.tsv`](provenance/withdrawn_vs_clean_cd_metrics.tsv) | withdrawn versus clean accuracy under the *same frozen model* |

Accession matching is GCA/GCF-aware, against the cross-map built from
[`curation/gtdb_filtered_genomes.tsv.gz`](curation/). Counts are identical under
raw GTDB-string matching, under prefix-and-version-stripped matching and under
the cross-map, so they are not normalisation artefacts.

Two further defects the audit surfaced, both recorded rather than smoothed over:
the withdrawn sets C and D contain 141 and 213 samples whose contamination
exceeds their completeness — outside the region the model was ever trained on,
because the training-domain constraint post-dates their construction — and
`set_E` contains 132 such samples.

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

Per-bin predictions from all four tools and the full analysis are in
[`results/cami2/`](results/cami2/).

**Denominators differ between tables, deliberately.** The truth tables hold
14,874 rows in total; **10,512 bins were actually written and scored** (marine
gold 4,497, marine mixed 2,201, strain-madness gold 1,564, strain-madness mixed
2,250), and 5,406 of the 6,359 marine gold truth rows are genuine-genome bins
(953 are flagged `excluded_non_genome`). Any percentage computed from these files
must state which denominator it uses; `cami2_censoring.tsv` reports both the
scored-bin and all-truth-row versions side by side.

## Result files

[`results/`](results/) holds every quantitative result reported in the paper —
1,177 files, 128 MB. [`results/README.md`](results/README.md) is the inventory:
which directory answers which figure and table, the workspace-to-repository path
convention, and **every exclusion with its reason**. Nothing excluded is a
reported number; it is raw tool output that the deposited scripts regenerate.

## Analysis code

[`scripts/`](scripts/) is the complete analysis code: 227 numbered scripts, the
frozen `holdout_lib/` module copy used by the holdout retrainings, and
`manuscript_figures/` — the builders that draw every display item in the paper
and the supplement.

**Script numbers were resolved for this deposition.** In the analysis workspace
thirty scripts carried a number another workstream also used: `136`–`148` were
claimed by four workstreams at once (the real-data track, the reduced-genome
mechanism, its mitigation arm and the contamination-type benchmark) and
`190`–`198` by three. In this copy every number is unique and zero-padded to
three digits, so **a lexical sort of `scripts/` is a valid execution order**.
The working tree was not renumbered, so
[`scripts/SCRIPT_MAPPING.tsv`](scripts/SCRIPT_MAPPING.tsv) maps every
working-tree name to its deposited name and names the workstream. The moved
blocks are:

| Workstream | Workspace numbers | Deposited numbers |
|---|---|---|
| WS3 real-data track A | 136–148 | **136–148** (kept) |
| WS4.2 GUNC detection comparator | 138–139 | **095–096** |
| WS2 contamination type × distance (set F) | 144–148 | **154–158** |
| WS3.10 reduced-genome mechanism | 136–137 | **206–207** |
| WS3.10 mitigation and recalibration | 140–144 | **212–216** |
| WS11.T timing campaign v3 | 190–198 | **250–258** |
| legacy pathogen / Kraken2 / NCBI-comparison scripts | 45, 46, 60, 62 | **048, 049, 063–066** |

Scripts contain workspace-relative paths of the form `/path/to/magicc/...`,
which must be repointed at a local checkout before they will run.

## Released artefacts

[`released_artefacts/`](released_artefacts/) is the bundle the Code availability
statement names:

| Path | Contents |
|---|---|
| `kmers/bacterial_kmer_prevalence.tsv`, `kmers/archaeal_kmer_prevalence.tsv` | **both complete 131,072-k-mer prevalence tables** (all canonical 9-mers, one row each), so the feature selection can be recomputed from scratch |
| `kmers/selected_9mers.tsv` | the 9,249 selected features with domain assignment, per-domain prevalence and the selection reason for each |
| `kmers/selected_{bacterial,archaeal}_1000.tsv`, `kmer_selection_criteria.md`, `kmer_selection_stats.json` | the per-domain rankings and the criteria |
| `normalization/normalization_params.json` | the normalization parameters of the released model |
| `splits/`, `generator/`, `benchmarks/` | split files and statistics, per-sample seed provenance, the per-sample metadata index and the benchmark inventory |
| `MANIFEST.sha256`, `RELEASE_MANIFEST.json` | checksums and the machine-readable manifest |

## What is not deposited here, and why

| Not here | Why |
|---|---|
| CAMI II sequence data | third-party; openly available at <https://frl.publisso.de/data/frl:6425521/>. Only identifiers and derived tables are deposited |
| ATCC Genome Portal genomes | the ATCC Data Use Agreement is incompatible with unrestricted redistribution; these genomes were **not used** |
| Meslier mock-community and ZymoBIOMICS assemblies | third-party; available from `https://forge.inrae.fr/metagenopolis/benchmark_mock` (ENA PRJEB52977) and ENA PRJEB29504. The accessions, derived truth and predictions are deposited |
| SPIRE v1, GTDB r220 and UHGG v2.0.2 genomes | third-party catalogues; the genome identifiers analysed are deposited in [`results/real_data/reduced_genome/`](results/real_data/reduced_genome/) |
| Reference genomes (GTDB / NCBI) | 345 GB; identified by accession in `splits/`, `curation/` and `provenance/`, and downloadable from NCBI |
| Raw tool output trees (GUNC, CheckM2, CoCoPyE, DeepCheck working directories) | about **53 GB** of intermediates; the parsed per-genome predictions and the consolidated tables *are* deposited. Itemised with sizes and reasons in [`results/README.md`](results/README.md) |
| The two taxonomic-holdout ONNX models | 162 MB each; validation artefacts, not released models, and regenerable by the deposited training scripts from the deposited splits |
| The leave-genus-out **evaluation** | still running at deposition. Its panel definition and its complete V5-versus-V5 null control *are* deposited; see [`results/holdout_genus/STATUS.md`](results/holdout_genus/STATUS.md) and [`DEPOSITION_STATUS.md`](DEPOSITION_STATUS.md) |

## Citation

Tian, R., Zhou, J., Imanian, B. MAGICC: genome quality assessment from
single-copy core-gene k-mer profiles. *Manuscript under revision.*

## License

MIT License. See [LICENSE](LICENSE).
