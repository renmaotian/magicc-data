# CAMI II marine source-scope audit

The 354 marine bins available only in the pairwise comparator cohort have
circular-element dominant references. They do not provide whole prokaryotic
genome truth. This conclusion follows original source identities and official
dataset-generation provenance, independently of prediction values, bin length
or whether a comparator returned a score.

## Original source provenance

The [CAMI II primary paper](https://www.nature.com/articles/s41592-022-01431-4)
describes 777 marine microbial genomes (155 newly sequenced isolates and 622
MarRef genomes), plus 200 newly sequenced circular elements including plasmids
and viruses. Its methods describe circular-element sequencing from an enriched
wastewater plasmid sample. These are biological source sequences subsequently
used for read simulation, not evidence of arbitrary synthetic decoy genomes.

The [official data-generation README](https://github.com/CAMI-challenge/second_challenge_evaluation/blob/501b543f65d62e5c1d6c3813be0badcac5e079ca/scripts/data_generation/README.md)
identifies the addition of 200 marine plasmids/circular elements and explicitly
connects these source names to the RNODE prefix. Its `add_plasmids.py` reads
the circular-element metadata, selects the requested number, copies the
sequences and appends their identities and existing category labels to the
simulation metadata. The exact README and script are preserved under
`upstream/`, with commit and SHA-256 values in `provenance.tsv`.

The original local marine setup metadata contains exactly the expected 977
entries:

| Source identity | Category | Count |
|---|---|---:|
| Otu-prefixed | Microbial genome novelty categories | 777 |
| RNODE-prefixed | plasmid | 108 |
| RNODE-prefixed | virus | 4 |
| RNODE-prefixed | unknown | 88 |

All 88 unknown circular elements carry NCBI taxid 32644. The [NCBI taxonomy
record](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=32644) names
this catchall entry “unidentified” and places it under unclassified sequences;
it is not a resolved prokaryotic lineage. Although its database rank is
species, sharing this identifier does not establish biological species
relatedness. The project's historical taxdump contains the same name, rank
and parent, recorded in `local_taxid_32644_records.json`.

## Missed scope filter and downstream effects

`scripts/196_ws38_cami2_truth_and_bins.py:146–151` correctly explains that
plasmid/virus reference lengths are not full microbial genome denominators,
but `EXCLUDED_NOVELTY` contains only the labels `plasmid` and `virus`.
Lines 557–577 consequently retain `unknown` RNODE entries in scoreable gold
bins and candidate source pairs. The resulting source audit has 864 entries,
including all 88 unknown circular elements. The other 776 entries are
observed microbial sources; 777 is the original setup count, not the observed
source-audit denominator.

Script 197 assesses reference overlap but does not establish biological source
scope. Unknown RNODEs receive no assembly/name overlap match, so they pass its
overlap screen. Script 196's `lineage`/`lca_rank` logic also treats their shared
32644 catchall identifier as species identity. All 206 affected mixed bins
are therefore labelled species-distance mixtures without valid species-level
evidence. Script 201's domain/scoreability/overlap filters do not independently
correct this source-scope omission.

The current pairwise membership audit joins each bin to its original dominant
and contaminant source identities:

| Bin type | Four-tool common, microbial sources | Pairwise-exclusive, circular-element dominant |
|---|---:|---:|
| Gold | 339 | 148 |
| Mixed | 864 | 206 |

No common-cohort bin contains an RNODE dominant **or** donor. Thus the current
common-cohort results are unaffected by a complete circular-element scope
filter. The extra pairwise cohort should not be represented as evidence of
genome-quality accuracy or score availability on valid prokaryotic genomes.

These sources remain legitimate inputs for CAMI's broader assembly/binning
challenges, and recovered-source fractions can be calculated for them. That
does not make their short circular-element lengths full microbial genome
denominators. They may be retained as explicitly out-of-scope diagnostic
evidence. A source-based exclusion of **all 200 known circular-element IDs**,
applied consistently to dominant and donor roles, implements the intended
genome-quality scope without selecting on any tool's outcome. The unknown
category should not be globally equated with a non-genome category in
unrelated datasets; the inference here uses the documented RNODE source set.

## Reproduction and preserved evidence

Run `python scripts/267_audit_cami_marine_source_scope.py`. This reads existing
metadata, truth and cohort tables, and downloads only the small official
generation sources if their saved snapshots are absent. It changes no
benchmark input, truth, prediction or figure. Outputs include the exact
200-element manifest, 88 historically retained elements, per-bin source-scope
join, counts, local taxonomy records, completion summary and SHA-256
provenance. Sequence-validity checks are independently owned by the figures
agent; they are not prerequisites for this metadata-based scope conclusion.
