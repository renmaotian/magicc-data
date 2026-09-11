# CAMI construction records and current genome-quality scope

The `truth/` files preserve the original construction rows and their historical
flags. In particular, their old `excluded_non_genome` flag did not exclude all
88 unknown-category RNODE circular elements; it must not be used as the current
microbial-genome eligibility rule by itself. These construction records are
retained for audit and are not extra validation genomes.

Current analyses require a microbial dominant and, for mixed bins, a microbial
donor. Native marine metadata identifies 777 microbial sources and 200 circular
elements (108 plasmid, 4 viral, 88 unknown). All 200 RNODE elements are outside
genome-quality validation. The exact IDs and source categories are in
[`marine_circular_element_manifest.tsv`](../updates/2026-09/snapshot/results/revision/holdout_resubmission5/cami_source_audit/marine_circular_element_manifest.tsv).
Script 270 applies this rule before statistical summaries, in addition to the
separate domain, overlap and tool-scoring criteria. Corrected analyses and their
source map are in [the September deposit](../updates/2026-09/README.md).

The corrected primary four-tool cohorts contain 339 marine gold / 864 marine
mixed bins and 700 strain-madness gold / 2,250 mixed bins. Their membership and
results were unchanged by the added source-unit restriction because no circular
elements were present. The previously apparent additional MAGICC–CheckM2 bins
were circular-element constructions and are not valid genome-quality evidence.
No third-party CAMI sequences are redistributed here.
