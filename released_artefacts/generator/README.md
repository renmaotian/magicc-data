# Synthetic benchmark generator — seeds and per-sample metadata  (WS7.7, R1-m15)

Reviewer 1 asked for the generator to be released "with all random seeds and
per-sample metadata (dominant ID, contaminant IDs, target/observed completeness
and contamination, fragmentation and dropout parameters)". This directory states
exactly what exists, for which set, and how complete it is. It does not
overstate coverage.

## Generator source

| Component | Module |
|---|---|
| Fragmentation / dropout simulation | `magicc/fragmentation.py` |
| Contamination event construction | `magicc/contamination.py` |
| Clean Sets C/D driver (fully instrumented) | `scripts/73_generate_clean_cd_benchmarks.py` |
| Finished-genome sets A_v2 / B_v2 / E driver | `scripts/34_generate_finished_benchmarks.py` |
| Original Sets A–D driver | `scripts/25_benchmark_generate.py` |
| Motivating sets driver | `scripts/31_motivating_benchmark_generate.py`, `scripts/41_generate_motivating_set_c.py` |

## Two tiers of seed provenance — stated plainly

**Tier 1 — fully recorded per-sample seeds (2 sets: `set_C_clean`, `set_D_clean`).**
`generation_metadata.tsv` has 43 columns and stores the literal integer seed of
every sample together with the complete parameter set that produced it:
dominant accession/phylum/domain/taxonomy/source-split/reference bp/retained bp,
contaminant accessions + phyla + reference bp + bp + source split, requested and
selected contaminant counts, target **and** observed completeness and
contamination, quality tier with its contig-count / N50 / minimum-contig /
log-normal-sigma ranges, all six dropout parameters (coverage mean, coverage
sigma, coverage threshold, GC-loss strength, repeat-loss probability,
low-complexity threshold), the constraint-guard flag, and per-source contig
counts. Regeneration was **tested**: three FASTA files were deleted, their
checkpoint lines removed, and the rerun reproduced them byte-identically.

**Tier 2 — derivable per-sample seeds (all other sets).**
The earlier drivers did not persist a `seed` column, but they are deterministic:
each sample's RNG is `np.random.default_rng(BASE + row_index + OFFSET)` with the
constants given in `seed_provenance.tsv`. Any sample can therefore be
regenerated from the released `metadata.tsv` row index without guessing. What is
**not** recoverable from the released files alone for these sets is the drawn
fragmentation/dropout parameters, because they were not written out — they are
reproduced by rerunning the generator with the stated seed, not read from a
table.

`per_sample_metadata_index.tsv` lists, for every set and file, the exact columns
available, so no reader has to assume.

## Files

| File | Content |
|---|---|
| `seed_provenance.tsv` | one row per set: generator, base seed, design RNG, per-sample RNG construction, whether seeds are recorded or derivable |
| `per_sample_metadata_index.tsv` | which per-sample columns exist in which file for which set |
