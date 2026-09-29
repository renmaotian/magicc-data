# MAGICC benchmark data and scientific analyses

Benchmark metadata, predictions, statistics and analysis code for
[MAGICC](https://github.com/renmaotian/magicc). Production V5 is unchanged, with
ONNX SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`.

## Current results and reproduction

Use the [ordered reproduction workflow](reproduction/README.md). It verifies
inputs, rebuilds numerical results, renders scientific figures and supplies the
full genus-exclusion training recipe. [DEPENDENCIES.tsv](reproduction/DEPENDENCIES.tsv)
states why each retained script/module is needed; obsolete exploratory,
duplicate and editorial scripts remain in Git history.

The primary taxonomic experiment jointly excludes ten named bacterial genera
while preserving their parent families and phyla. Both arms share vocabulary
selection from panel-free training representatives and fit their own
training-only normalizers. A matched-full model
provides the control; the earlier family/phylum results remain sensitivity
records. This panel does not represent all archaeal or reduced-genome lineages.

The completed experiment evaluates 5,770 assemblies from 577 references.
Control-adjusted holdout–matched-full MAE differences span 1.53–9.23 percentage
points for completeness and 0.72–8.12 for contamination; all 20 primary contrasts
have positive 95% intervals and BH-adjusted q < 0.005. These are joint training-
pool, normalization and validation-selection effects from one seed per arm.
Different genus and historical family panels do not establish a paired rank trend.
See the [primary contrasts](reproduction/inputs/results/revision/holdout_resubmission7/did.tsv),
[three prespecified sensitivities](reproduction/inputs/results/revision/holdout_resubmission7/sensitivity_did.tsv)
and [independent audit](reproduction/inputs/results/revision/holdout_resubmission7/independent_numerical_audit.json).

The CheckM2 dose analysis separates contamination at or below 35% from above 35%.
The five-set CheckM2 contamination MAE is 22.14 percentage points: 5.69 within the
lower band and 40.72 above 35%. The higher band contains 46.98% of samples and
86.38% of total CheckM2 absolute contamination error. MAGICC's corresponding
MAEs are 5.31 overall, 2.54 within the lower band and 8.43 above 35%. This is descriptive
accounting; dose and contamination construction were not independently varied.
See the [recomputed table](reproduction/inputs/results/revision/checkm2_domain_resubmission7/dose_decomposition.tsv).

Corrected comparator inputs preserve CoCoPyE's official stage selection,
reference-cluster uncertainty and microbial-only CAMI source eligibility.
[September scientific records](updates/2026-09/README.md) retain the full
provenance. Three historical model-prediction tables are required solely for the
original primary multiple-test family and are identified as statistical context.

The 2,000 representative core-gene FASTAs are now available as one checksum-bound
archive in two Git-sized chunks. Restoration verifies all 2,003 members and
reassembles them automatically. Research ONNX models remain outside this data
repository; the full training recipe regenerates them. Production weights and
inference code remain unchanged. [DEPOSITION_STATUS.md](DEPOSITION_STATUS.md)
states the exact scope.

## Benchmark identity and downloads

The current display names **Set C** and **Set D** refer to the established
test-reference datasets `set_C_clean` and `set_D_clean`. Their internal IDs,
assembly bytes, metadata identifiers, checksums and asset names are unchanged.
The [display/data map](updates/2026-09/DISPLAY_DATASET_MAP.tsv) distinguishes
them from the withdrawn historical datasets.

| Display | Stable dataset ID | Assemblies | Design |
|---|---|---:|---|
| Set A | `set_A` (workspace `set_A_v2`) | 1,000 | Completeness gradient |
| Set B | `set_B` (workspace `set_B_v2`) | 1,000 | Contamination gradient |
| Set C | `set_C_clean` | 1,000 | 100 test-reference Patescibacteriota genomes × 10 simulations |
| Set D | `set_D_clean` | 1,000 | 100 test-reference archaeal genomes × 10 simulations |
| Set E | `set_E` | 1,000 | Realistic mixed quality profiles |
| Set F | `set_F` | 1,900 | Contamination type × donor relatedness |
| Set G | `set_G` | 1,920 | Sequencing/assembly stress tests |
| Set H | `set_H` (workspace `set_H_ncbi`) | 4,000 | Reference-selection sensitivity |

The 12,820 assemblies remain on
[v1.0.0](https://github.com/renmaotian/magicc-data/releases/tag/v1.0.0), with
existing multipart archives and SHA256 checksums:

```bash
git clone https://github.com/renmaotian/magicc-data.git
cd magicc-data
bash download_benchmarks.sh set_C_clean set_D_clean
# All eight sets, approximately 14.75 GB of archives:
bash download_benchmarks.sh --full-verify
```

Original training-overlapping C/D data remain under `withdrawn/` and as
`WITHDRAWN_benchmark_set_C.tar.gz` / `WITHDRAWN_benchmark_set_D.tar.gz` on
[v0.1.0](https://github.com/renmaotian/magicc-data/releases/tag/v0.1.0). They must
not be used as the current independent-reference benchmark. Historical motivating
A/B/C assemblies also remain on v0.1.0; they use a separate naming namespace.

## Scope and reproduction

Both quality percentages use the full dominant-reference genome length.
Completeness is retained dominant-origin bp divided by that length; contamination
is contaminant-origin bp divided by the same length. These sequence definitions
do not establish exact equivalence to protein/marker definitions. Analysis-specific
domain and tool-scoring restrictions remain with each result.

The primary benchmark uses independent dominant references. Production V5
inherited normalization fitted on unlabelled V4 training, validation and test
features, so it is not fully independent at preprocessing level. The new matched
holdout fits normalization exclusively on each arm's training data. Fixed-weight
normalization sensitivities do not correct training-time preprocessing overlap.

See the [restoration instructions](reproduction/README.md), `splits/`,
`provenance/` and release manifests. Sets A/B/E lack complete saved per-sample
seed/fragmentation records; their generation rules do not promise byte-identical
replay. The new holdout saves seeds and source identities. Third-party genomes,
tool databases and large intermediates are obtained separately.

## License and citation

Repository code and derived tables use the [MIT license](LICENSE); third-party
data retain source terms. Cite the MAGICC manuscript and original external-data
providers. Private reviewer comments, correspondence, journal forms and editorial
documents are excluded from the September public snapshot.
