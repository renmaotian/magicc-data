# MAGICC benchmark data and scientific analyses

Benchmark metadata, predictions, statistics and analysis code for
[MAGICC](https://github.com/renmaotian/magicc). Production V5 is unchanged, with
ONNX SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`.

## Current scientific deposit: September 2026

The [September update](updates/2026-09/README.md) supplies completed scientific
results and an explicit [file/SHA256 manifest](updates/2026-09/SCIENTIFIC_FILE_MANIFEST.tsv):

* CoCoPyE predictions follow its selected API/CSV stage: marker estimates at
  stage 2, neural estimates at stage 3, and quantitatively unscored at stage 1.
  Unscored inputs remain counted. No valid estimate is clipped.
* Pooled intervals cluster the 1,648 physical reference accessions shared across
  the five sets, retaining equal set weights. Set F tests use reference-level
  paired units. Threshold and comparison rules accompany component statistics.
* CAMI marine genome-quality truth excludes all 200 circular source elements
  from dominant and donor roles, including 88 previously missed unknown-category
  elements. They are not treated as prokaryotic genome truth.
* One new jointly excluded ten-family model is compared with one matched-full
  retrain. Both use actual feature reselection and separate training-only
  normalizers. Full trainings finished at 49 epochs/selected 29 and 90/selected
  70. Evaluation covers 8,720 family assemblies from 872 references plus 1,000
  controls from 100 references. Reference-bootstrap intervals do not cover
  model-seed uncertainty; this intervention does not isolate novelty alone.
* Sequence-level duplication verification, fixed-weight normalization
  sensitivities and the corresponding scientific source audits are supplied.

[UPDATED_PUBLIC_PATHS.tsv](updates/2026-09/UPDATED_PUBLIC_PATHS.tsv) identifies
existing files replaced by verified corrections. Their previous versions remain
at commit `c051b3d8383b560f34942e24ea4cbfe154f079c4`. The versioned snapshot is
the authority for new results. Historical reports may describe earlier extraction
or statistics; consult the corrected tables rather than combining versions.

The scientific tables and scripts are public in the committed snapshot. Research
ONNX models and the core-gene input archive are prepared with the submission
materials outside this repository; **no public download is provided here** and
this update does not require their upload. [DEPOSITION_STATUS.md](DEPOSITION_STATUS.md)
and [RELEASE_ASSETS.json](updates/2026-09/RELEASE_ASSETS.json) record this scope and
the prepared artifact identities. A separate scientific ZIP is optional because
its tables and code are already committed. No production weights or inference
code were retrained or replaced by this update.

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

See the [restoration instructions](updates/2026-09/README.md), `splits/`,
`provenance/` and release manifests. Sets A/B/E lack complete saved per-sample
seed/fragmentation records; their generation rules do not promise byte-identical
replay. The new holdout saves seeds and source identities. Third-party genomes,
tool databases and large intermediates are obtained separately.

## License and citation

Repository code and derived tables use the [MIT license](LICENSE); third-party
data retain source terms. Cite the MAGICC manuscript and original external-data
providers. Private reviewer comments, correspondence, journal forms and editorial
documents are excluded from the September public snapshot.
