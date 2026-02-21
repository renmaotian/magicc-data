# Data-Generating Scripts

These scripts generated the benchmark and motivating analysis datasets used in the MAGICC paper. They depend on the `magicc` Python package ([github.com/renmaotian/magicc](https://github.com/renmaotian/magicc)) for genome fragmentation and contamination simulation modules.

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
