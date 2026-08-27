# MAGICC revision — benchmark provenance audit (WS1.4)

Generated 2026-07-26T14:26:25.065636+00:00 by `data_generating_scripts/74_provenance_audit.py`.

## Why this audit exists

Both reviewers observed that benchmark Sets C (Patescibacteria) and D (Archaea) in the
submitted manuscript were not independent of the training data. They were right. This
directory contains the evidence, the exact leakage numbers, and the proof that the
replacement sets are clean.

## Method

### Accession universes

| Universe | File | Rows |
|---|---|---|
| train split | `splits/train_accessions.txt` | 79948 |
| val split | `splits/val_accessions.txt` | 10010 |
| test split | `splits/test_accessions.txt` | 9999 |
| 9-mer feature-selection genomes | `data/kmer_selection/selected_bacterial_1000.tsv` + `selected_archaeal_1000.tsv` | 2000 |

The three splits are disjoint by construction and stratified by phylum. The 2,000
feature-selection genomes were sampled from the train split only
(2000/2000 verified to be inside the
train split here).

### The GCA/GCF pitfall, and how it is handled

The same assembly is written three different ways across project files — GTDB
(`GB_GCA_001822065.1`, `RS_GCF_035810195.1`), GenBank (`GCA_001822065.1`) and RefSeq
(`GCF_035810195.1`). Naive string matching therefore **undercounts** overlap: a
benchmark genome recorded as `GB_GCA_x.1` can sit in a split file as `RS_GCF_x.1`.

Three normalisations are computed for every accession and all three are reported in
`overlap_summary.tsv`:

* **raw** — the GTDB accession string exactly as written in the file.
* **strict** — drop the `GB_`/`RS_` prefix and the `.version` suffix
  (`GB_GCA_001822065.1 -> GCA_001822065`). Keeps `GCA` and `GCF` distinct.
* **crossmap** (authoritative) — a canonical assembly-pair key. Every row of
  the GTDB filtered-genome table (`curation/gtdb_filtered_genomes.tsv.gz`) supplies a `(gtdb_accession, ncbi_accession,
  gcf_accession)` triple that names one assembly, so all three strings are mapped to one
  key. Accessions absent from that table fall back to the 9-digit assembly number, which
  a GCA/GCF pair shares by NCBI construction.

Cross-map statistics: 277183 rows of
`filtered_genomes.tsv` -> 503511 distinct
accession strings collapsing to 277183 canonical
assemblies (1.817 strings per assembly on average,
0 inconsistent rows).

## Result 1 — the superseded Sets C and D were leaked

Counts are **per sample** (n = 1,000 each), using the authoritative crossmap
normalisation. All three normalisations agree for these two sets (see
`overlap_summary.tsv`), so the numbers below are not normalisation artefacts.

| Set | n | in TRAIN | in VAL | in TEST | in no split | in 9-mer selection | leakage (train+val) |
|---|---|---|---|---|---|---|---|
| set_C | 1000 | **1000** | 0 | 0 | 0 | 23 | 100.0% |
| set_D | 1000 | **796** | 107 | 97 | 0 | 511 | 90.3% |

Per-normalisation TRAIN / VAL / TEST counts:

| Set | raw GTDB string | strict (prefix+version stripped) | crossmap (authoritative) |
|---|---|---|---|
| set_C | 1000 / 0 / 0 | 1000 / 0 / 0 | 1000 / 0 / 0 |
| set_D | 796 / 107 / 97 | 796 / 107 / 97 | 796 / 107 / 97 |

> **Correction to the internal record.** The figures previously logged in
> the internal project log — "Set C: 985/1,000 dominants in TRAIN" and
> "Set D: only 36/1,000 in test" — were themselves incomplete audits. 985 is the number of
> Set C dominants carrying a `GB_` prefix, not the number in TRAIN; the true count is
> **1000/1,000 in TRAIN and 0 in TEST**. For Set D the
> complete accounting is **796 TRAIN /
> 107 VAL / 97 TEST**.
> Set C leakage is therefore total, not 98.5 %.

A further overlap that the original audit missed: **23**
Set C samples and **511** Set D samples use a
dominant reference that was also one of the 2,000 genomes used to *select the 9-mer feature
set*, so those samples leak into feature selection as well as into model fitting.

Root cause, quoted from the generator that produced them
(`data_generating_scripts/25_benchmark_generate.py`): *"Set C: ALL Patescibacteriota from
train+val+test (1608 total)"*, *"Set D: ALL Archaea from train+val+test (1976 total),
sample 1000"*.

The sets could never have been built from the test split alone: the test split holds only
161 Patescibacteriota and 198 archaeal genomes, and reference curation had already taken
**all** 1,609 Patescibacteriota and **all** 1,976 archaeal genomes of the
277,183-genome filtered pool into the 100,000-genome working set, so there are zero
unused genomes of these lineages to fall back on.

Contaminant provenance cannot be audited for the superseded sets: their generator did not
record contaminant accessions. That gap is one of the reasons the new generator writes a
complete `generation_metadata.tsv` (reviewer comment R1-m15).

### A second, independent defect of the old sets: out-of-constraint labels

The V5 training distribution enforces `contaminant_bp <= dominant_retained_bp`, i.e.
contamination % <= completeness % (both share the dominant's full reference length as
denominator). That cap was added to `magicc/contamination.py` in commit `c5a9b95`
(2026-03-14). Sets C, D, E and `motivating_v2/set_C` were generated on 2026-03-10,
**before the cap existed**, so a fraction of their samples lie outside the region the
model was ever trained on:

| Set | n | samples with contamination % > completeness % | max excess |
|---|---|---|---|
| set_C | 1000 | 141 | 43.791 pp |
| set_D | 1000 | 213 | 43.904 pp |
| set_E | 1000 | 132 | 46.328 pp |
| motivating_v2/set_C | 1000 | 122 | 46.385 pp |

The clean sets have zero such samples (verified in
the generation-validation record (`results/ws1_23_generation_validation.json`)). The leaked -> clean difference in
Set C/D accuracy therefore mixes two effects; `provenance/withdrawn_vs_clean_cd_metrics.tsv`
decomposes them by also reporting the superseded sets restricted to their
constraint-satisfying subsets.

## Result 2 — the clean Sets C_clean and D_clean are disjoint from all training input

| Set | n samples | unique dominants | in TRAIN | in VAL | in TEST | in 9-mer selection |
|---|---|---|---|---|---|---|
| set_C_clean | 1000 | 100 | **0** | **0** | 1000 | **0** |
| set_D_clean | 1000 | 100 | **0** | **0** | 1000 | **0** |

Dominant accession lists with per-genome split membership flags:
`set_C_clean_dominants.txt`, `set_D_clean_dominants.txt`.

### Contaminants

Contaminants are drawn only from `splits/test_accessions.txt`, excluding the dominant's
own phylum. Every contamination event is listed in `contaminants.txt`.

| Set | events | unique contaminant genomes | from TEST | from TRAIN | from VAL | same phylum as dominant |
|---|---|---|---|---|---|---|
| set_C_clean | 2977 | 2560 | 2977 | 0 | 0 | 0 |
| set_D_clean | 3006 | 2608 | 3006 | 0 | 0 | 0 |

## Result 3 — the retained sets were already clean

| Set | n | in TRAIN | in VAL | in TEST |
|---|---|---|---|---|
| set_A_v2 | 1000 | 0 | 0 | 1000 |
| set_B_v2 | 1000 | 0 | 0 | 1000 |
| set_E | 1000 | 0 | 0 | 1000 |
| set_A | 600 | 0 | 0 | 600 |
| set_B | 600 | 0 | 0 | 600 |
| motivating_v2/set_A | 1000 | 0 | 0 | 1000 |
| motivating_v2/set_B | 1000 | 0 | 0 | 1000 |
| motivating_v2/set_C | 1000 | 0 | 0 | 1000 |
| motivating/set_A | 600 | 0 | 0 | 600 |
| motivating/set_B | 1100 | 0 | 0 | 1100 |

## Files

| File | Contents |
|---|---|
| `overlap_summary.tsv` | one row per benchmark set; per-sample and per-unique-genome overlap counts against train/val/test/9-mer-selection, under all three normalisations, plus the label-constraint violation counts |
| `set_C_clean_dominants.txt` | the 100 Patescibacteriota dominants with taxonomy, canonical key and split flags |
| `set_D_clean_dominants.txt` | the 100 archaeal dominants, same columns |
| `contaminants.txt` | every contamination event: genome_id, contaminant accession, split, phylum, dominant phylum |
| `sha256_manifest.txt` | SHA256 of every generated FASTA, every metadata/label file, the frozen ONNX model, the normalisation parameters, the 9-mer list and the split files |
| `accession_crossmap_stats.json` | cross-map construction statistics |

## Reproduce

```bash
python data_generating_scripts/72_select_clean_cd_refs.py            # WS1.1  reference selection
python data_generating_scripts/73_generate_clean_cd_benchmarks.py C D  # WS1.2/1.3  generation
python data_generating_scripts/74_provenance_audit.py                # WS1.4  this audit
python scripts/75_run_magicc_clean_cd.py             # WS1.5  MAGICC V5 inference
```

---

## SHA256 manifests in this repository

| Manifest | Covers | Path convention |
|---|---|---|
| `sha256_manifest.txt` | `set_C_clean`, `set_D_clean` (+ the model and shared inputs) | analysis-workspace paths (`data/benchmarks/...`) |
| `set_F_sha256_manifest.txt` | `set_F` | analysis-workspace paths |
| `set_G_sha256_manifest.txt` | `set_G` | analysis-workspace paths |
| `set_H_sha256_manifest.txt` | `set_H` | analysis-workspace paths |
| `set_A_sha256_manifest.txt` | `set_A` | this repository's paths (`benchmark/set_A/...`) |
| `set_B_sha256_manifest.txt` | `set_B` | this repository's paths |
| `set_E_sha256_manifest.txt` | `set_E` | this repository's paths |

Each manifest lists every generated assembly FASTA plus `metadata.tsv` and
`labels.npy`. Verify a release-asset download with, from the directory holding the
`fasta/` tree:

```bash
sha256sum -c set_A_sha256_manifest.txt
```

The manifests for sets A, B and E were computed on 2026-08-25 from the same
files that produced every reported number; the others were written by the
provenance audit at generation time. The path prefixes differ for that
historical reason and are stated per row above.
