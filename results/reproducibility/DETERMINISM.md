# MAGICC determinism and reproducibility statement (WS7.12)

Addresses **R1-M6, R1-M8, R1-m17** and editorial requirements **E2/E5/E6**.
Generated 2026-07-26. Machine-readable companion:
`results/revision/reproducibility/determinism_report.json`.

> **Note for readers of the public repository.** This file and its JSON
> companion are shipped here because they document a property of the released
> software. The checker that produces them
> (`scripts/115_determinism_check.py`), like every other analysis script cited
> below, is part of the separately released analysis code — see the paper's
> Code availability statement. Paths of the form `scripts/…`,
> `results/revision/…` and `data/…` in the text below refer to the full
> analysis workspace, not to this repository.

---

## 1. The headline, stated without hedging

**Inference is deterministic. Training was not seeded.**

MAGICC's released weights (`models/magicc_v5.onnx`) **cannot be re-derived
bit-exactly** from the training data, because no random seed was set anywhere in
`magicc/trainer.py` or `scripts/53_train_v5_run3.py`. This was established by
direct code inspection during the revision and is recorded in
`results/revision/model_card.json` (`training.random_seed: null`) and in protocol
§4.4b. Retraining with the same script and the same data would produce a
different, similarly-performing model — not this one.

The released artefact is therefore pinned by **content hash**, not by seed:

```
models/magicc_v5.onnx
SHA256  b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096
size    169,658,949 bytes   opset 17   IR 8   FP32   producer pytorch 2.5.1
```

Every container build, the release bundle builder, and the determinism check all
verify this hash and fail if it does not match. All retraining performed *during*
this revision (WS1.6–1.10) **is** seeded (`scripts/holdout_lib/seeding.py`), so
the new experiments are reproducible even though the released model is not.

This limitation is disclosed rather than worked around. It does not affect any
reported number: every benchmark in the revised manuscript was produced by the
one frozen artefact above.

---

## 2. What *is* guaranteed: inference determinism

All checks below were run through the **installed `magicc` console script** — what
a reviewer actually invokes — not through an internal Python API.

`python scripts/115_determinism_check.py --n-genomes 24 --max-threads 4`
→ **7/7 PASS**, 24 real genomes from Set E.

| # | Check | Result |
|---|---|---|
| 1 | `models/magicc_v5.onnx` SHA256 matches `model_card.json` | PASS |
| 2 | Fixed-input probe from the model card reproduces **bit-exactly** (max abs deviation 0.0) | PASS |
| 3 | **Thread-count invariance**: 1 / 2 / 4 threads → one output digest | PASS |
| 4 | **Gzip invariance**: `X.fasta` vs `X.fasta.gz` → byte-identical | PASS |
| 5 | **Input-mode invariance**: `--input DIR` vs `--input-list` (mixed plain/gz) → identical per genome | PASS |
| 6 | **Repeat invariance**: same command twice → byte-identical | PASS |
| 7 | **Batch-size invariance**: `--batch-size` 1 / 64 / 512 → one output digest | PASS |

Reference output digest for the 24-genome fixture:
`7a5d34003c420a3fd7f7a676c6e503618b97eb95a7cd44b950811b87bcaacee4`.

The fixed-input probe is defined in `model_card.json`
(`numpy.default_rng(20260726)` standard normals of shape (4, 9249) and (4, 7),
float32, `CPUExecutionProvider`, 1 intra-/inter-op thread; probe input SHA256
`34d0a3ed…`) and reproduced exactly by onnxruntime 1.23.2.

### Why there is no CPU-vs-GPU row

MAGICC inference is **CPU-only by design**: the CLI creates the onnxruntime
session with `CPUExecutionProvider`, and the shipped model is a small MLP whose
inference cost is negligible against feature extraction. There is no GPU code
path to compare, so the CPU/GPU axis of R1-m17 does not arise. The GPU is used
only for *training*, which is separately addressed in §1.

---

## 3. Cross-environment reproduction (the strongest evidence)

The same 5 Set E genomes were scored in three **independently built**
environments. All three produced the identical prediction file:

```
3a612d0b25cacf57908ddeb7427070a74d366c51bd689e48d8a29f96a762be2c
```

| Environment | Python | How built |
|---|---|---|
| Host conda env `magicc2` | 3.11.0 | the environment in which every manuscript number was produced |
| **Docker** `magicc:0.3.0` (`sha256:d13e68ea…`) | 3.11.9 | `docker/Dockerfile`, pinned base digest, hash-verified deps, run with `--network none` as a non-root user |
| **conda-lock** env from `conda/conda-lock.yml` | 3.11.9 | solver-free install from the lock file, independent of the host env |

Byte-identical output across three separately-resolved dependency stacks and two
different CPython patch levels is a stronger statement than repeat-run
determinism within one environment.

Additional evidence banked earlier in the revision and independently
re-confirmed here: predictions from `X.fasta` and `X.fasta.gz` are byte-identical
across 100 real Set E genomes; 1-thread and 8-thread runs are byte-identical;
fresh V5 predictions reproduce the recorded `results/ncbi_comparison/` values to
5×10⁻⁵ pp.

---

## 4. Sources of nondeterminism that were considered

| Candidate | Status |
|---|---|
| **Numba** JIT | The package contains exactly one Numba kernel, `magicc/kmer_counter.py::_count_kmers_single`, declared `@nb.njit(cache=True)` — **no `parallel=True`, no `prange`**. It accumulates integer counts into a per-sequence `int64` array in a fixed serial loop, so it has no thread-dependent reduction order and no floating-point accumulation at all. `magicc/assembly_stats.py` contains **no Numba code**: the 7 summary features are pure NumPy over the already-computed integer count vector. |
| **Multiprocessing worker order** | Genome-level parallelism only, via `Pool.imap` (order-preserving), so output rows follow input order rather than completion order regardless of thread count. Confirmed byte-stable by checks 3 and 6. |
| **Float precision of the feature vector** | Counts are integers; the float32 cast happens once per genome after counting, before normalisation, and is independent of thread count or batching. |
| **ONNX Runtime** thread pools / batch shape | Verified invariant by checks 3 and 7. The model contains no reduction whose result depends on partitioning. |
| **BLAS threading** (`OMP_NUM_THREADS` etc.) | Pinned to 1 inside the determinism check; unpinned host runs (4 threads) gave the same digest, so it does not matter in practice. |
| **Python hash randomisation** | No output ordering depends on `set`/`dict` iteration. |
| **gzip compression level** | Irrelevant: decompressed bytes are compared, and compression is detected from magic bytes rather than filename. |
| **Model auto-download** | Eliminated in containers — the frozen model is baked in and checksummed at build time, and the Docker build **fails** if a run-time download is attempted (`~/.magicc` must not exist after the smoke test). |

No nondeterminism was found at inference. Nothing had to be suppressed or
special-cased to obtain the results above.

---

## 5. How to reproduce these checks

```bash
# 1. determinism suite (needs the package installed and the repo data)
python scripts/115_determinism_check.py --n-genomes 24 --max-threads 4

# 2. unit / CLI test suite (92 tests)
python -m pytest tests/ -q

# 3. container equivalence
docker build -f docker/Dockerfile -t magicc:0.3.0 .
docker run --rm --network none -v "$PWD":/data magicc:0.3.0 \
    predict --input /data/genomes --output /data/pred.tsv --threads 4

# 4. locked conda environment
conda-lock install --name magicc conda/conda-lock.yml
```
