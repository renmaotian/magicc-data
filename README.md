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
