# Data-Generating Scripts

These scripts generated the benchmark and motivating analysis datasets used in the MAGICC paper. They depend on the `magicc` Python package ([github.com/renmaotian/magicc](https://github.com/renmaotian/magicc)) for genome fragmentation and contamination simulation modules.

> **Paths.** Every path of the form `/path/to/magicc/...` is a placeholder for a
> local checkout of the analysis workspace and must be repointed before these
> scripts will run. Output paths named below (`data/benchmarks/...`) are the
> workspace layout, not this repository's layout; the corresponding metadata is
> published here under `benchmark/`.

> **Numbering.** This directory keeps the **working-tree** script numbers it has
> been published under since February 2026, so existing links stay valid. The
> complete analysis code is at [`../scripts/`](../scripts/), where the numbers
> were made unique for the deposition — for example `145_generate_set_F.py` here
> is `155_generate_set_F.py` there. [`../scripts/SCRIPT_MAPPING.tsv`](../scripts/SCRIPT_MAPPING.tsv)
> maps every name. The logic is identical; the only difference is that the
> copies in `../scripts/` have their cross-references repointed at the deposited
> numbers, so that set runs as it stands.

> **`25_benchmark_generate.py` produced the WITHDRAWN sets C and D.** Its module
> docstring states that dominants come from "ALL ... from train+val+test" — that
> is the leakage, in the source. It is kept here for exactly that reason. See
> [`../withdrawn/`](../withdrawn/).

## Scripts added in the 2026 revision

| Script | Produces |
|---|---|
| `72_select_clean_cd_refs.py` | selects the 100 test-split Patescibacteriota and 100 test-split archaeal references for the clean sets |
| `73_generate_clean_cd_benchmarks.py` | generates `set_C_clean` and `set_D_clean` (100 references x 10 simulations each), recording a **literal per-sample seed** |
| `74_provenance_audit.py` | the GCA/GCF-aware leakage audit behind `../provenance/overlap_summary.tsv` |
| `144_contamination_type_module.py`, `145_generate_set_F.py` | set F: contamination type x donor taxonomic distance |
| `150_error_injection_module.py`, `151_generate_set_G.py` | set G: sequencing and assembly error robustness |
| `185_ws1_11_select_ncbi_refs.py`, `187_ws1_11_generate_set_H.py` | set H: 400 NCBI references in 200 matched pairs, circularity safeguard |
| `188_ws1_11_provenance_audit.py` | the set H provenance audit (delegates to `74_provenance_audit.py`, so the two cannot drift apart) |

## Scripts

### `33_filter_finished_genomes.py`

**Prerequisite step.** Filters the test genome split to retain only NCBI-finished genomes (assembly level "Complete Genome" or "Chromosome") by cross-referencing GTDB metadata. These finished genomes serve as dominant references for the finished-genome benchmark and motivating datasets, providing clean ground truth without contig-level assembly artifacts.

- **Input**: `data/splits/test_genomes.tsv`, GTDB metadata (`bac120_metadata.tsv.gz`, `ar53_metadata.tsv.gz`)
- **Output**: `data/splits/test_finished_genomes.tsv` (1,810 finished genomes out of 9,999 test genomes)

### `25_benchmark_generate.py`

Generates the original benchmark datasets (Sets A--D) using all test reference genomes as dominants. These sets evaluate tool performance on controlled quality gradients and underrepresented lineages.

| Set | N | Completeness | Contamination | Description |
|-----|---|--------------|---------------|-------------|
| A | 600 | 50--100% (6 levels, 100 each) | 0% | Completeness gradient |
| B | 600 | 100% (original contigs) | 0--80% (6 levels, 100 each) | Contamination gradient |
| C | 1,000 | Uniform 50--100% | Uniform 0--100% | Patescibacteria (all 1,608 refs) |
| D | 1,000 | Uniform 50--100% | Uniform 0--100% | Archaea (1,000 of 1,976 refs) |

- **Output**: `data/benchmarks/set_{A,B,C,D}/` each containing `fasta/`, `metadata.tsv`, `labels.npy`

### `34_generate_finished_benchmarks.py`

Generates five datasets using **only finished genomes** as dominant references (contaminants drawn from all test references). This is the primary generation script for the paper's final benchmark and motivating analyses.

| Dataset | Seed | N | Completeness | Contamination | Description |
|---------|------|---|--------------|---------------|-------------|
| Motivating Set A | 100 | 1,000 | 50--100% (6 levels) | 0% | Completeness gradient |
| Motivating Set B | 200 | 1,000 | 100% (original contigs) | 0--80% (5 levels) | Contamination gradient |
| Benchmark Set A_v2 | 300 | 1,000 | 50--100% (6 levels) | 0% | Completeness gradient |
| Benchmark Set B_v2 | 400 | 1,000 | 100% (original contigs) | 0--80% (5 levels) | Contamination gradient |
| Set E | 500 | 1,000 | Mixed | Mixed | Realistic (200 pure + 200 complete + 600 other) |

- **Output**: `data/benchmarks/motivating_v2/set_{A,B}/`, `data/benchmarks/set_{A_v2,B_v2,E}/`
- **Requires**: `data/splits/test_finished_genomes.tsv` (from `33_filter_finished_genomes.py`)

### `41_generate_motivating_set_c.py`

Generates Motivating Set C: a realistic mixed dataset of 1,000 genomes using finished dominant genomes. This set complements Motivating Sets A and B by combining completeness and contamination variation together.

- **Composition**: 200 pure (0% contamination) + 200 complete (100% completeness) + 600 mixed (70% cross-phylum, 30% within-phylum contamination)
- **Seed**: 300
- **Output**: `data/benchmarks/motivating_v2/set_C/` containing `fasta/`, `metadata.tsv`, `labels.npy`
- **Requires**: `data/splits/test_finished_genomes.tsv` (from `33_filter_finished_genomes.py`)

## Execution Order

```
33_filter_finished_genomes.py    # Step 1: filter to finished genomes
25_benchmark_generate.py         # Step 2a: original benchmark sets A-D
34_generate_finished_benchmarks.py  # Step 2b: finished-genome sets (motivating A/B, benchmark A_v2/B_v2, E)
41_generate_motivating_set_c.py  # Step 2c: motivating set C
```

Steps 2a--2c are independent and can run in any order after Step 1.
