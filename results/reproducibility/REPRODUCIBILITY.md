# MAGICC — reproducibility and software distribution (WS7)

Single index for everything WS7 produced. Addresses reviewer comments
**R1-M6, R1-M8, R1-m14, R1-m15, R1-m17, R1-m18, R2-o5** and editorial
requirements **E2, E5, E6**. Prepared 2026-07-26.

> **Publication status (2026-08-27).** This was written on 2026-07-26, when nothing
> had yet been published. Since then the software has been released
> (github.com/renmaotian/magicc tag `v0.3.1`, PyPI `magicc` 0.3.1) and the data,
> results and analysis code deposited in this repository. The Bioconda recipe is
> shipped but **not** submitted to bioconda-recipes. Paths below of the form
> `results/revision/...` are workspace paths; in this repository that material is
> under `results/...` and the release bundle under `released_artefacts/`.

---

## 1. Index

| Item | Deliverable | Where |
|---|---|---|
| 7.1 Docker | pinned, hash-verified, model baked in | `docker/Dockerfile`, `docker/requirements-lock.txt`, `docker/requirements.in`, `docker/regenerate_lock.sh` |
| 7.2 Apptainer | definition + built SIF | `containers/magicc.def`, `containers/magicc_0.3.0.sif`, `containers/build_containers.sh` |
| 7.3 conda | environment + full lock | `conda/environment.yml`, `conda/conda-lock.yml`, `conda/conda-linux-64.lock`, `conda/conda-osx-64.lock`, `conda/conda-osx-arm64.lock` |
| 7.4 / 7.11 Workflow | Snakemake + one-command reproduction | `workflow/` (see `workflow/README.md`) |
| 7.5 9-mer list | list + prevalence + criteria | `released_artefacts/kmers/` |
| 7.6 Splits / metadata / hashes | release bundle | `released_artefacts/` |
| 7.7 Generator + seeds | seed provenance for every set | `released_artefacts/generator/` |
| 7.8 / 7.9 `.gz` + `--input-list` | shipped in the CLI | `magicc/cli.py`, `tests/` (77 tests) |
| 7.10 Bioconda | recipe, built and tested locally | `conda-recipe/magicc/meta.yaml`, `conda-recipe/README.md` |
| 7.12 Determinism | report + rerunnable check | `results/revision/reproducibility/DETERMINISM.md`, `determinism_report.json`, `scripts/115_determinism_check.py` |

Builders: `scripts/112_release_artifacts.py` (7.5/7.6/7.7),
`scripts/115_determinism_check.py` (7.12), `containers/build_containers.sh`
(7.1/7.2).

---

## 2. Three ways to install, all pinned

```bash
# (a) Container -- most hermetic. Model baked in and checksummed at build time;
#     the build FAILS if a run-time download is attempted.
docker build -f docker/Dockerfile -t magicc:0.3.0 .
docker run --rm --network none -v "$PWD":/data magicc:0.3.0 \
    predict --input /data/genomes --output /data/predictions.tsv --threads 8

# (b) Apptainer / Singularity, for HPC without Docker
apptainer build containers/magicc_0.3.0.sif docker-daemon://magicc:0.3.0
apptainer run --containall --bind "$PWD":/data containers/magicc_0.3.0.sif \
    predict --input /data/genomes --output /data/predictions.tsv --threads 8

# (c) Locked conda environment
conda-lock install --name magicc conda/conda-lock.yml
conda activate magicc && pip install --no-deps .
```

All three were verified to produce **byte-identical predictions** — see §4.

---

## 3. Pinning, concretely

* **Base image pinned by digest**, not tag:
  `python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`.
* **Every Python dependency pinned to an exact version and verified against a
  recorded SHA256** (`pip install --require-hashes --no-deps`), transitive
  closure included — 13 packages. Regenerate with `bash docker/regenerate_lock.sh`.
* **The model is bundled and checksummed twice** during the Docker build: once
  against the literal expected hash and once against
  `results/revision/model_card.json`. Either mismatch fails the build.
* **conda-lock** resolves `conda/environment.yml` for linux-64, osx-64 and
  osx-arm64: 58 conda packages + 7 PyPI packages per platform, each with a hash.
* Direct runtime pins match the environment in which every manuscript number was
  produced: `numpy 1.26.4`, `numba 0.63.1`, `scipy 1.17.0`, `h5py 3.15.1`,
  `onnxruntime 1.23.2`.

`onnxruntime` comes from PyPI rather than conda-forge because conda-forge does
not package 1.23.2 for CPython 3.11 (its 1.23.x builds are py312+). The wheel is
the same upstream CPU build and is hash-pinned in both lock files.

---

## 4. Verification actually performed

| Check | Result |
|---|---|
| Test suite through the installed console script | **77/77 pass** |
| `.gz` input (7.8) | `X.fasta` and `X.fasta.gz` give byte-identical predictions |
| `--input-list` (7.9) | mixed plain/gz list gives byte-identical predictions |
| Determinism suite (7.12) | **7/7 pass** — threads, gzip, input mode, repeat, batch size, model hash, fixed-input probe |
| Docker build smoke test | passes, and asserts **no model download occurred** |
| Docker end-to-end, `--network none`, non-root | output byte-identical to host |
| conda-lock environment end-to-end | output byte-identical to host |
| Bioconda recipe | `conda build` succeeded, package tests passed, nothing uploaded |
| Snakemake workflow (7.4/7.11) | 14/14 rules execute; **full 5,000-genome run PASSES** |

Cross-environment agreement digest (5 Set E genomes, host / Docker / conda-lock):
`3a612d0b25cacf57908ddeb7427070a74d366c51bd689e48d8a29f96a762be2c`.

### The headline reproduction, in numbers

`bash workflow/run_reproduction.sh` regenerated the pooled leakage-free
five-set statistics from raw FASTA and matched the recorded values **exactly**:

| Metric | Reproduced MAE (pp) | 95 % CI (cluster bootstrap) | Recorded | Δ |
|---|---|---|---|---|
| Completeness | **4.6179** | [4.332, 4.927] | 4.6179 | +0.00000 |
| Contamination | **5.3086** | [5.013, 5.601] | 5.3086 | +0.00000 |

R² (coefficient of determination): completeness **0.7853**, contamination
**0.9149** — both identical to the recorded values. 5,000 genomes, 2,586
clusters. Per-set numbers also match `ws5.5_table_S2_rebuilt.tsv` (e.g.
`set_A_v2` completeness MAE 2.1766). Report:
`results/revision/reproducibility/workflow/HEADLINE_REPRODUCTION.md`.

---

## 5. Limitations, stated plainly

1. **The released weights are not bit-reproducible.** V5 training was never
   seeded (protocol §4.4b). The artefact is pinned by SHA256
   (`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`) and
   *inference* is deterministic. All retraining performed during this revision
   **is** seeded. See `DETERMINISM.md` §1.

2. **Apptainer builds here but does not run here.** The SIF
   (`containers/magicc_0.3.0.sif`, 367 MB, SHA256 `287e83d43549…`) was built
   rootless from the Docker image and inspects correctly, carrying the model-hash
   label. It **cannot be executed on this host**: Ubuntu's
   `kernel.apparmor_restrict_unprivileged_userns=1` blocks unprivileged user
   namespaces, the conda-forge build has no setuid-root `starter-suid`, and no
   passwordless sudo is available. The image is therefore **built and validated
   structurally but not executed**. On any host with a normal system Apptainer
   install, or with unprivileged user namespaces permitted, the documented
   `apptainer run` command applies unchanged. This is a property of this machine,
   not of the image.

3. **The PyPI package is `magicc`, not `magicc-genome`.**
   the internal project log records the latter; `pypi.org/pypi/
   magicc-genome/json` returns **404** while `magicc` returns 200 and is the
   author's package (0.3.0, same repository URLs, sdist SHA256
   `0c744ed5d797f861ad2e90d41864d22301e6915f270a8e704f504bb9f9a80520`, byte-identical
   to `dist/magicc-0.3.0.tar.gz`). **The Code Availability section must say
   `pip install magicc`.**

4. **The conda/PyPI package does not contain the model.** It is 169.7 MB and is
   downloaded to `~/.magicc` on first prediction. Reviewers wanting a hermetic,
   offline install should use a container, where the model is baked in and
   checksummed.

5. **The workflow reproduces the MAGICC arm only.** Competitor tools need
   separate environments and >20 GB of reference databases; their predictions are
   released as files and the metrics framework recomputes every published
   statistic from them. `workflow/README.md` states the boundary.

6. **Per-sample seeds are recorded for two benchmark sets and derivable for the
   rest.** `set_C_clean` and `set_D_clean` store the literal seed and all 43
   generation parameters per sample. The earlier sets are deterministic
   (`default_rng(BASE + row_index + OFFSET)`, constants released) but did not
   persist the drawn fragmentation/dropout parameters, so those are reproduced by
   rerunning the generator rather than read from a table. See
   `release/generator/README.md`.

---

## 6. Environments created during WS7

Scratch environments, created so that `magicc2` was never disturbed:

| Env | Contents | Purpose |
|---|---|---|
| `magicc_ws7_tools` | conda-lock 4.0.2, snakemake 9.23.1, conda-build | locking, workflow, recipe validation |
| `magicc_ws7_apptainer` | apptainer 1.4.5 | SIF build |
| `magicc_ws7_locktest` | installed from `conda/conda-lock.yml` | proving the lock reproduces predictions |

`magicc2` itself was changed in exactly one way: the package was reinstalled
**editable** (`pip install -e .`), replacing a stale non-editable copy in
`site-packages` that still ran pre-revision code. A leftover empty
`site-packages/magicc/` directory from the old install was also removed — it was
shadowing the package as a namespace portion and breaking
`importlib.resources.files('magicc')`.
