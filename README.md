# MAGICC Benchmark Data

Benchmark and motivating analysis datasets for [MAGICC](https://github.com/renmaotian/magicc).

## Datasets

### Motivating Analysis (Sets A-C)

Demonstrate limitations of existing tools. All use NCBI finished genomes as dominant references.

| Set | N | Completeness | Contamination | Description |
|-----|---|--------------|---------------|-------------|
| A | 1,000 | 50-100% (6 levels) | 0% | Completeness gradient |
| B | 1,000 | 100% | 0-80% (5 levels) | Contamination gradient |
| C | 1,000 | 50-100% | 0-100% | Realistic mixed |

### Benchmark (Sets A-E)

Full evaluation datasets. All use NCBI finished genomes as dominant references.

| Set | N | Completeness | Contamination | Description |
|-----|---|--------------|---------------|-------------|
| A | 1,000 | 50-100% (6 levels) | 0% | Completeness gradient |
| B | 1,000 | 100% | 0-80% (5 levels) | Contamination gradient |
| C | 1,000 | 50-100% | 0-100% | Patescibacteria (CPR) |
| D | 1,000 | 50-100% | 0-100% | Archaea |
| E | 1,000 | 50-100% | 0-100% | Realistic mixed |

## Data Generation Methods

All synthetic benchmark genomes were generated from real NCBI reference genomes using controlled fragmentation and contamination simulation. The generation scripts are in [`data_generating_scripts/`](data_generating_scripts/).

### Reference genomes

Dominant genomes for the motivating sets (A--C) and finished-genome benchmark sets (A, B, E) were restricted to NCBI-finished references (assembly level "Complete Genome" or "Chromosome") from the MAGICC test split, providing clean ground truth. Benchmark Sets C and D used all available Patescibacteria (1,608) and Archaea (1,976) reference genomes respectively, drawn from train+val+test splits due to limited availability. Contaminant genomes were drawn from all test reference genomes regardless of assembly level.

### Quality metric definitions

Ground truth quality metrics are sequence-based (not gene-based):

- **Completeness** (%) = (total retained bp from the dominant genome / dominant genome full reference length) x 100. A genome retaining 4 Mb of a 5 Mb reference has 80% completeness.
- **Contamination** (%) = (total contaminant bp / dominant genome full reference length) x 100. Contamination is measured relative to the dominant genome's full reference length, not the retained length after incompleteness. This ensures the two metrics are independent: a genome at 60% completeness with 20% contamination contains contaminant DNA equal to 20% of the original reference size, regardless of how much dominant sequence was retained.

These sequence-based definitions are equivalent to CheckM2's protein-based definitions (completeness = annotated proteins / complete genome proteins; contamination = duplicated proteins / complete genome proteins), since protein density is approximately uniform across the genome.

### Fragmentation (incompleteness simulation)

Genome incompleteness was simulated by fragmenting a reference genome into contigs and then dropping contigs to reach a target completeness level:

1. **Contig generation**: The reference sequence is split into contigs with lengths drawn from a log-normal distribution (mu = log(N50), sigma = 0.8--1.2), where N50 is sampled uniformly within a quality tier. Quality tiers control assembly fragmentation: High (10--50 contigs, N50 100--500 kb), Medium (50--200 contigs, N50 20--100 kb), Low (200--500 contigs, N50 5--20 kb), and Highly Fragmented (500--2,000 contigs, N50 1--5 kb).
2. **Assembly bias simulation**: Three biologically-motivated dropout mechanisms are applied sequentially before completeness targeting: (a) coverage dropout, where each contig is assigned a coverage value from a log-normal distribution (mean 30x) and contigs below a 5x floor are removed; (b) GC-biased loss, where the drop probability follows a saturating exponential of the contig GC |z-score| (p = 0.3 x (1 - exp(-|z|))); and (c) repeat exclusion, where contigs with a high composite repeat score (homopolymer fraction + maximum dinucleotide frequency > 0.6) are removed with probability 0.15.
3. **Completeness targeting**: Contigs are sorted smallest-first, shuffled, then greedily accumulated until the total retained length reaches the target completeness (fraction of original genome length). An overshoot/undershoot heuristic decides borderline contigs, and a safety net adds back the smallest dropped contigs if completeness falls below 50%. For genomes requiring 100% completeness, the original reference contigs are used directly without fragmentation.

### Contamination simulation

Contamination was simulated by mixing fragmented contaminant genome(s) into the dominant genome's contigs:

1. **Contaminant selection**: 1--3 within-phylum or 1--5 cross-phylum contaminant genomes are selected randomly from the reference pool.
2. **Target allocation**: The total contaminant base pairs are determined by the target contamination rate (see definitions above). When multiple contaminants are used, the total is distributed among them via Dirichlet allocation.
3. **Contaminant fragmentation**: Each contaminant genome is independently fragmented into contigs and trimmed to its allocated base pair target. If the target exceeds the contaminant genome size, multiple copies are used.
4. **Merging**: Contaminant contigs are concatenated with dominant genome contigs to produce the final synthetic genome.

### Set-specific generation details

- **Motivating Sets A / Benchmark Sets A, A_v2**: Pure genomes (0% contamination) at 6 completeness levels (50%, 60%, 70%, 80%, 90%, 100%), ~167 genomes per level.
- **Motivating Sets B / Benchmark Sets B, B_v2**: Complete genomes (100% completeness, original contigs) with cross-phylum contamination at 5 levels (0%, 20%, 40%, 60%, 80%), 200 genomes per level.
- **Motivating Set C / Benchmark Set E**: Realistic mixed composition -- 200 pure genomes (0% contamination, 50--100% completeness), 200 complete genomes (100% completeness, 0--100% contamination), and 600 mixed genomes (50--100% completeness, 0--100% contamination, 70% cross-phylum / 30% within-phylum). Motivating Set C and Benchmark Set E use identical generation logic with different random seeds (300 vs 500).
- **Benchmark Set C (Patescibacteria)**: 1,000 genomes from all 1,608 Patescibacteriota references with uniform completeness (50--100%) and contamination (0--100%), testing performance on reduced-genome organisms.
- **Benchmark Set D (Archaea)**: 1,000 genomes from 1,976 archaeal references with uniform completeness (50--100%) and contamination (0--100%), testing performance on underrepresented lineages.

## Download FASTA Files

FASTA archives are available as GitHub Release assets (~6.3 GB total):

```bash
# Download all archives
gh release download v0.1.0 --repo renmaotian/magicc-data --dir .

# Or download individual sets
gh release download v0.1.0 --repo renmaotian/magicc-data --pattern "motivating_set_A.tar.gz"
```

Each archive contains:
- `fasta/` directory with 1,000 genome FASTA files
- `metadata.tsv` with ground truth labels
- `labels.npy` with numpy array of [completeness, contamination]

Extract with:
```bash
tar xzf motivating_set_A.tar.gz -C motivating/set_A/
```

## Metadata Format

Each `metadata.tsv` contains:

| Column | Description |
|--------|-------------|
| genome_id | Genome identifier (matches FASTA filename) |
| true_completeness | Ground truth completeness (%) |
| true_contamination | Ground truth contamination (%) |
| dominant_accession | GTDB accession of dominant genome |
| dominant_phylum | Phylum of dominant genome |
| sample_type | Generation method |
| n_contigs | Number of contigs |
| total_length | Total genome length (bp) |

## Predictions

Each set directory includes prediction TSV files from:
- MAGICC (`magicc_predictions.tsv`)
- CheckM2 (`checkm2_predictions.tsv`)
- CoCoPyE (`cocopye_predictions.tsv`)
- DeepCheck (`deepcheck_predictions.tsv`)

## License

MIT License.
