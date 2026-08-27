# Selected 9-mer feature set — selection criteria and provenance  (WS7.5, R1-m14)

MAGICC's k-mer branch consumes a fixed vector of **9,249 canonical 9-mers**.
`selected_9mers.tsv` in this directory is the authoritative release of that list.

## Feature order is load-bearing

Column `feature_index` is the position of the k-mer in the ONNX model input
`kmer_features [batch, 9249]`. It is the line order of
`data/kmer_selection/selected_kmers.txt` (and of the copy shipped inside the
Python package at `magicc/data/selected_kmers.txt`). Re-ordering the list
invalidates the model.

## How the list was built

1. **Reference genomes.** 1,000 bacterial and 1,000 archaeal
   representative genomes were sampled **from the TRAIN split only**, seed 42
   (`data/kmer_selection/selected_bacterial_1000.tsv`,
   `selected_archaeal_1000.tsv`). No validation or test genome contributed to
   feature selection; this is audited in
   `results/revision/provenance/` (WS1.4).
2. **Single-copy core genes** were identified per domain with Prodigal + HMMER
   against `85_bcg.hmm` (bacteria) and `uacg.hmm` (archaea)
   (`scripts/09_identify_core_genes.py`). Feature selection is therefore
   annotation-dependent; **inference is not** (see R1-m5).
3. **Canonical 9-mer counting** over the core-gene nucleotide sequences
   (`scripts/10_count_9mers.py`), giving 4^9/2 = 131,072 canonical 9-mers per
   domain.
4. **Prevalence** = the number of the 1,000 domain reference genomes in which
   the k-mer occurs at least once (range 0–1000).
5. **Selection** (`scripts/11_select_kmers.py`): the prevalence tables were sorted
   descending and the top **9,000 bacterial** and top
   **1,000 archaeal** k-mers were taken (`head(N)`) and merged.
   **751** k-mers are in both lists, so the union is
   **9,249** unique canonical 9-mers.

## The cutoff is a RANK cutoff, not a prevalence-value threshold

This matters for exact reproduction and is stated explicitly rather than left
implicit. The selection takes a fixed *number* of k-mers, so k-mers tied at the
cutoff prevalence straddle the boundary and are separated only by their position
in the sorted table:

| Domain | k-mers taken (rank cutoff) | prevalence range | k-mers in the full table reaching the cutoff prevalence | excluded purely by tie-breaking |
|---|---|---|---|---|
| Bacteria | 9,000 | 529 – 992 | 9,008 | 8 |
| Archaea  | 1,000 | 791 – 998 | 1,000 | 0 |
| **Union (model input)** | **9,249** | — | — | — |

100 released k-mers sit exactly at the bacterial cutoff prevalence and
12 exactly at the archaeal cutoff (columns
`at_bacterial_rank_cutoff_prevalence` / `at_archaeal_rank_cutoff_prevalence`).
Reproducing the selection therefore requires the released prevalence tables, not
just the two threshold numbers. Domain assignment in `selected_9mers.tsv` comes
from the recorded selection provenance (`source` column of the original
`selected_kmers_annotated.tsv`), never from re-applying a value threshold.

Composition of the union: 8,249 bacterial-only,
249 archaeal-only, 751 shared.

Note that **no 9-mer is present in all 1,000 genomes of either domain**
(`kmers_in_all_genomes = 0` in both `bacterial_kmer_stats.json` and
`archaeal_kmer_stats.json`); the highest bacterial prevalence is
992/1000 and the highest archaeal
998/1000. The features are therefore
*high-prevalence*, not universal.

## Transform applied before the model sees them

`log10(count + 1)`, then z-scoring with the stored per-feature mean/std in
`data/features/normalization_params.json` (released here under
`normalization/`). Both steps are in `magicc/normalization.py`.

## Files

| File | Content |
|---|---|
| `selected_9mers.tsv` | the released list: index, sequence, domain assignment, both prevalences, both threshold flags, selection reason |
| `bacterial_kmer_prevalence.tsv` | prevalence of **all** 131,072 canonical 9-mers in the 1,000 bacterial genomes |
| `archaeal_kmer_prevalence.tsv` | the same for archaea |
| `kmer_selection_stats.json` | the machine-readable selection record |
| `selected_bacterial_1000.tsv`, `selected_archaeal_1000.tsv` | the 2,000 feature-selection genomes, with accessions and taxonomy |

The two full prevalence tables are included so that the selection can be
**independently recomputed**, not merely inspected.
