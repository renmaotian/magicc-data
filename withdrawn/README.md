# WITHDRAWN datasets — do not use, do not cite

The two sets in this directory were published as `benchmark/set_C` and
`benchmark/set_D`. **They are withdrawn.** Their dominant reference genomes
were drawn from `train + val + test` rather than from the held-out test split,
so the model being evaluated had already seen them during training.

| Withdrawn set | samples | dominants in TRAIN | in VAL | in TEST | leakage (train+val) | dominants also in the 9-mer feature-selection set |
|---|---|---|---|---|---|---|
| `set_C` (Patescibacteriota / CPR) | 1,000 | **1,000** | 0 | 0 | **100.0 %** | 23 |
| `set_D` (Archaea) | 1,000 | 796 | 107 | 97 | **90.3 %** | 511 |

Counts are identical under raw GTDB-string matching, under
prefix-and-version-stripped matching and under a GCA/GCF-aware cross-map, so
they are not artefacts of accession normalisation. The root cause is visible in
the generator's own module docstring
(`../data_generating_scripts/25_benchmark_generate.py`), which states that the
dominants come from "ALL … from train+val+test".

A second, independent defect: 141 of set C's and 213 of set D's samples have
contamination % greater than completeness %, outside the region the model was
trained on, because the training-domain constraint post-dates their
construction.

## Replacements

Use [`../benchmark/set_C_clean`](../benchmark/set_C_clean) and
[`../benchmark/set_D_clean`](../benchmark/set_D_clean). Each uses 100 dominant
references drawn from the **test split only**, with 10 independent simulations
per reference (1,000 samples), the same design otherwise. Audited: 0 samples in
train, 0 in validation, 0 in the k-mer feature-selection set.

## Why these files were kept rather than deleted

So that the withdrawn numbers remain checkable and the size of the leakage
effect can be measured against the *same frozen model*. Each subdirectory's
`WITHDRAWN.md` carries the audited counts, the mechanism and the
withdrawn-versus-clean accuracy comparison. See also
[`../provenance/withdrawn_vs_clean_cd_metrics.tsv`](../provenance/withdrawn_vs_clean_cd_metrics.tsv)
and
[`../provenance/leakage_specificity_control.md`](../provenance/leakage_specificity_control.md),
the control showing that the effect is attributable to leakage and not to the
lineages these sets happen to cover.

**No number derived from this directory appears in the manuscript, and none
should appear anywhere else.**
