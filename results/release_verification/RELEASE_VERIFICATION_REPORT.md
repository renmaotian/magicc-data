# Release verification — end-to-end acceptance test of the published MAGICC artefacts

**Date:** 2026-08-26
**Contract:** the internal build contract §8.1
**Scope:** an acceptance test *from clean environments*, of what a reviewer actually
receives from PyPI and from GitHub — not a test of the working tree.
**Nothing was published, pushed, re-tagged or built. No file outside this directory was modified.**

---

## Verdict

| # | Assertion required by §8.1 | Verdict |
|---|---|---|
| 1 | `pip install magicc` (unpinned) resolves the version Code availability names | ✅ **PASS** — resolves **0.3.1** |
| 2 | The PyPI artefact is the one on record | ✅ **PASS** — wheel SHA256 matches the release record byte for byte |
| 3 | The GitHub tag `v0.3.1` clones and its Git LFS model is real content | ✅ **PASS** — 169,658,949 B of real ONNX, not a pointer |
| 4 | The model's SHA256 is `b84346…b3096` | ✅ **PASS from GitHub / working tree**; ⚠️ **not applicable to PyPI — see Defect 1** |
| 5 | **No model download occurs at run time** | ❌ **FAIL — defect D7 is LIVE in the published package** |
| 6 | All output TSVs byte-identical across sources, input modes and thread counts | ✅ **PASS — 24/24 runs, one single digest** |
| 7 | The published package reproduces the working tree's output | ✅ **PASS — byte-identical** |

**Headline:** the published software *computes* exactly what the paper says it computes —
24 independent runs across three sources, three input modes, two `--extension` settings,
two thread counts and two CPython minor versions produced one and only one output digest.
But the PyPI distribution **does not contain the model**; it fetches it over the network
from a **mutable branch ref, without verifying the checksum**. A reviewer on an offline
cluster cannot run `pip install magicc` at all.

---

## A. The test set

**Selection rule (deterministic, restated so anyone can rebuild it):** list
`data/benchmarks/set_E/fasta/*.fasta`, sort under `LC_ALL=C`, then take **every 40th
file starting at the first** (0-based indices 0, 40, 80 … 960) → **exactly 25 genomes**,
spread evenly across the sorted set rather than clustered at its head.

```bash
LC_ALL=C ls -1 *.fasta | LC_ALL=C sort | awk 'NR%40==1'
```

- 25 genomes, **95,400,596 B (91 MiB)** of plain FASTA — first `genome_0.fasta`, last `genome_963.fasta`
- gzip copies made with `gzip -n` (no timestamp in the header): 28 MiB
- `input_list.txt`: 25 absolute paths, one per line
- Per-genome names, sizes and SHA256s of the source file, the plain copy and the gzip
  copy: **`test_set_manifest.tsv`**. Every plain copy's SHA256 equals its source's — the
  copies are provably the benchmark genomes.

Three input modes (plain directory, gzip directory, `--input-list`) × two extension
settings (default `.fasta`, `--extension auto`) × two thread counts (1, 4) were exercised
on **each of the three software sources**.

---

## B. PyPI artefact — `pip install magicc` in a fresh virtualenv

Built from **`/usr/bin/python3` (CPython 3.12.3, system)**, *not* the `magicc2` conda
environment, so an editable install could not mask a packaging defect.

| Item | Value |
|---|---|
| Command | `pip install magicc` — **no version pin** |
| Resolved | **magicc 0.3.1** ✅ matches Code availability |
| Wheel | `magicc-0.3.1-py3-none-any.whl`, 653,983 B |
| Wheel SHA256 (re-downloaded, `--no-cache-dir`) | `55ac58af1fed22610fd0a6613ebf1f0faf33b6814abd659835acb693b8c09244` |
| Same as PyPI JSON API and the internal repository release record §2 | ✅ identical |
| sdist on index | `magicc-0.3.1.tar.gz`, 661,376 B, `aa36d209…e2926c` ✅ matches the record |
| Upload time (wheel) | **2026-08-25T17:00:28.522359Z** |
| Versions on the index | 0.1.0, 0.2.0, 0.2.1, 0.3.0, **0.3.1**; none yanked |
| 0.3.0 upload time | **2026-03-19T18:49:25.900981Z** |
| Dependencies pulled | flatbuffers 25.12.19, h5py 3.16.0, llvmlite 0.49.0, numba 0.67.0, numpy 2.5.2, onnxruntime 1.29.0, packaging 26.3, protobuf 7.36.0, scipy 1.18.1 |

The publication chronology the coordinator supplied is **confirmed from the third-party
index**: 0.3.0 on 2026-03-19, 0.3.1 on 2026-08-25 — i.e. 0.3.1 postdates the 2026-07-20
reviewer comments, 0.3.0 predates them. Full per-file record in `logs/pypi_magicc_json_api.json`.

### ❌ Defect 1 — the PyPI distribution contains no model

The wheel holds **20 files and no `.onnx`**. Its three largest entries are
`magicc/data/normalization_params.json` (2.4 MB), `magicc/data/selected_kmers.txt`
(92 KB) and `magicc/cli.py` (34 KB). This is by construction —
`pyproject.toml [tool.setuptools.package-data]` lists only the JSON and the k-mer list.

> Code availability says: *"The version used for all analyses is v0.3.1, containing model
> V5 (SHA256 `b84346…`)."* That sentence is **true of the GitHub tag** (verified below).
> It is **not true of the PyPI package**, which contains no model at all.

### ❌ Defect 2 (= D7, live) — the CLI downloads the model at run time

`magicc/cli.py` (installed copy, lines 44-72):

```python
MODEL_URL = "https://github.com/renmaotian/magicc/raw/main/models/magicc_v5.onnx"
...
def _ensure_model() -> str:
    """Ensure the ONNX model is available, downloading from GitHub if needed."""
    candidates = [_PACKAGE_DIR/'data'/MODEL_FILENAME,      # absent in a pip install
                  _PROJECT_DIR/'models'/MODEL_FILENAME,    # absent in a pip install
                  _USER_DATA_DIR/MODEL_FILENAME]           # $HOME/.magicc
    for p in candidates:
        if p.is_file() and p.stat().st_size > 1_000_000:
            return str(p)
    dest = _USER_DATA_DIR / MODEL_FILENAME
    ...
    urllib.request.urlretrieve(MODEL_URL, str(dest))       # no checksum verification
```

Three properties of this, each independently a reproducibility problem:

1. **The reference is `main`, a mutable branch — not the `v0.3.1` tag.** Today
   `main == ca1ddf04154e75c2e00ec453eec32bea4ed29cfe == tag v0.3.1`, so the fetch is
   currently correct *by coincidence*. Any future commit to `main` that touches the model
   silently changes what every past `pip install magicc==0.3.1` downloads.
2. **The download is never checksummed.** The only validation is `size > 1 MB`. The
   model card hash is not consulted, so a wrong or truncated model would be used silently.
3. **The install is not self-contained.** A reviewer on an air-gapped or
   firewalled HPC node cannot run the pip package at all.

### How the no-network assertion was proved

`unshare -rn` is **unavailable on this host** — `kernel.apparmor_restrict_unprivileged_userns=1`,
no passwordless sudo, no `firejail`. The method used instead was **strace syscall-level
fault injection**:

```
strace -f -e trace=connect -e inject=connect:error=ENETUNREACH <command>
```

Every `connect()` syscall returns `ENETUNREACH`. This blocks at the **syscall boundary**,
so unlike an env-var or Python-level shim it also covers native code and subprocesses.
**Control test** (the blocker itself is effective): `curl https://pypi.org/simple/` under
the injector fails with *"Could not resolve host"*, rc=6.

| Test | Setup | Result |
|---|---|---|
| **1** | Fresh PyPI install, **empty isolated `HOME`**, network blocked | ❌ **exit 1** — `urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>`, after printing *"Downloading MAGICC model to …"* |
| **2** | Same install, model already cached, network blocked | ✅ **exit 0**, **0** successful external connections; loaded `$HOME/.magicc/magicc_v5.onnx` |
| **3** | Same install, empty `HOME`, network **allowed**, under strace | Download observed: **37 `connect()` calls**, to `140.82.112.4:443` (github.com) and `2606:50c0:800x::154:443` (media.githubusercontent.com) |
| **4** | GitHub-installed venv, empty `HOME`, network blocked | ❌ **exit 1**, same download attempt — the defect is in the code, not the channel |

Every published-artefact run used an **isolated `HOME`**, because this host already
carries `$HOME/.magicc/magicc_v5.onnx` (correct hash, mtime 2026-03-19) which would
otherwise have masked the download entirely.

**Verdict on §8.1's "assert no model download occurs at run time": FAILED.** The download
does occur. What the D7 fix achieved was to make the *Docker image* self-contained (its
build-time smoke test asserts `~/.magicc` is absent); the packaged CLI was never fixed.

### ✅ Determinism from PyPI

All **8** PyPI runs (3 input modes × extension settings × 1 and 4 threads) exited 0 and
produced the identical TSV
**`bb7825cea40ddbc1dca159172b451ef714d66edc9c1ba705e2ce1af104805ee5`** (25 data rows).

The model the published package downloaded hashes to
**`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`** — the value the
paper names. **The content served today is right; the delivery mechanism is what fails.**

---

## C. GitHub artefact — tag `v0.3.1`

```bash
git clone --depth 1 --branch v0.3.1 https://github.com/renmaotian/magicc.git
```

| Item | Value |
|---|---|
| HEAD | **`ca1ddf04154e75c2e00ec453eec32bea4ed29cfe`** ✅ equals the internal repository release record "main after" |
| `git describe` | `v0.3.1` |
| Head commit | 2026-08-25 11:58:44 -0500, *fix(build): anchor `.gitignore` directory patterns…* |
| `main` today | **`ca1ddf04…`** — identical to the tag |
| Tags on remote | exactly one: `v0.3.1` |
| Releases | one, `v0.3.1`, published **2026-08-25T17:00:09Z**, **0 uploaded assets** |
| `pyproject.toml` version | `0.3.1` |

### Git LFS content is genuinely present

| Check | Result |
|---|---|
| `git lfs ls-files` | `b84346650c * models/magicc_v5.onnx` |
| Size on disk | **169,658,949 B** — full content, not a ~130-byte pointer |
| First bytes | ONNX/protobuf payload (`\b\b\022\a pytorch \032 2.5.1`), **not** `version https://git-lfs…` |
| **SHA256** | **`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`** ✅ **matches the paper** |

**The GitHub artefact fully satisfies Code availability.** A reviewer who clones the tag
gets the model the paper names, with the hash the paper names.

### But installing from the clone reproduces Defect 1

`pip install <clone>` into a second fresh venv (py 3.12.3) → version 0.3.1, **0 `.onnx`
files in the installed package**. The packaging excludes the model identically to PyPI,
and `site-packages/../models` does not exist, so the dev-layout fallback cannot fire.
Test 4 above confirms the same run-time download.

### ✅ Determinism from GitHub, and agreement with PyPI

All **8** GitHub runs — using the tag's **own** LFS model via `--model` — exited 0 and
produced **`bb7825ce…05ee5`**: **byte-identical to all eight PyPI runs.**

---

## D. Agreement with the paper — the working tree

Conda env `magicc2`, **CPython 3.11.0**, console script
`/path/to/anaconda3/envs/magicc2/bin/magicc`, resolving to the editable working-tree
package at `/path/to/magicc/magicc/`, version 0.3.1.

- Model loaded: `/path/to/magicc/models/magicc_v5.onnx`, SHA256
  `b84346…b3096`. No download (the dev layout finds it locally).
- All **8** working-tree runs exited 0 → **`bb7825ce…05ee5`**.

> ### ✅ **All 24 output TSVs are byte-identical.**
> 3 sources (PyPI 0.3.1 / GitHub tag v0.3.1 / working tree) × 3 input modes (plain, gzip,
> `--input-list`) × 2 extension settings × 2 thread counts (1, 4) = **24 runs, 1 distinct
> SHA256**, across **two different CPython minor versions (3.12.3 and 3.11.0)** and three
> independent numpy / onnxruntime installations.
>
> **The published package reproduces the paper's computation exactly.** Per-run table in
> `output_digests.tsv`; the predictions themselves in `predictions_25genomes_canonical.tsv`.

This is a stronger cross-environment determinism result than the banked WS7 §7.12
evidence, because it spans three independently-installed copies of the software rather
than three environments running one copy.

---

## E. Container tags

Full record in **`container_tag_facts.md`**. In brief:

- **The only Apptainer artefact that exists is `containers/magicc_0.3.0.sif`** (384,008,192 B,
  sha256 `287e83d4…aa04`, built 2026-07-26 16:56). Its **own embedded labels say
  `org.opencontainers.image.version = 0.3.0`**, so it cannot be renamed into a 0.3.1
  artefact. **`containers/magicc_0.3.1.sif` does not exist** — the internal repository release record
  lists it at "367 MB", which is in fact the 0.3.0 SIF's size carried forward under the
  wrong name. That row must be corrected.
- The tracked recipes (`containers/magicc.def`, `containers/build_containers.sh`) **already
  say 0.3.1** and would build it correctly — they simply have not been run.
- **Both Docker images exist locally**: `magicc:0.3.0` (`d13e68ead3b6`, 2026-07-26) and
  `magicc:0.3.1` (`38804a236e32`, 2026-08-25, matching the release record). Both carry
  the correct `org.magicc.model.sha256`.
- **Nothing is published.** `RepoDigests` is empty on both images (proof they were never
  pushed or pulled); Docker Hub `renmaotian/magicc` → **404**; GHCR anonymous tags list →
  **403**; GitHub Packages containers for the user → **`[]`**; quay.io → 401; the GitHub
  release carries **0 uploaded assets**.
- Therefore: **the manuscript may claim a Dockerfile and an Apptainer definition are
  provided; it may not imply an image can be pulled.** Table S8d's `magicc:0.3.0` string
  is *correct as history* — that image is the one the cross-environment probe actually
  ran on — and should keep its historical framing rather than be rewritten to 0.3.1.

---

## F. `magicc-data` — "before" state

Full record in **`magicc_data_before_state.json`**. Reachable and public.

| Item | Value |
|---|---|
| Default branch | **`main`**, head `9c72af7bd9565f30fddf3274543a747cc6ba2a05` |
| Created / last pushed | 2026-02-19T20:10:00Z / **2026-08-25T17:10:33Z** |
| Repo size (API) | 9,846 KB |
| Root entries | `LICENSE`, `README.md`, `all_project_scripts`, `benchmark`, `cami2`, `data_generating_scripts`, `motivating`, `provenance`, `splits`, `withdrawn` |
| Tags | one: `v0.1.0` → `34e84306e0fdec661b64fc7e1cd983e8cfa251e3` |
| Releases | **one: `v0.1.0`**, published 2026-02-19T21:10:02Z, **8 assets, ~9.36 GB total** |

Assets on `v0.1.0`:

| Asset | Bytes | Downloads |
|---|---|---|
| `benchmark_set_A.tar.gz` | 912,868,001 | 0 |
| `benchmark_set_B.tar.gz` | 1,654,366,970 | 0 |
| **`benchmark_set_C.tar.gz`** | 368,756,285 | **2** |
| **`benchmark_set_D.tar.gz`** | 895,227,083 | 0 |
| `benchmark_set_E.tar.gz` | 1,459,666,157 | 0 |
| `motivating_set_A.tar.gz` | 923,351,957 | 1 |
| `motivating_set_B.tar.gz` | 1,695,361,481 | 1 |
| `motivating_set_C.tar.gz` | 1,449,013,915 | 1 |

> ### ⚠️ Finding for the depositing agent
> The `v0.1.0` **release assets still serve the leakage-contaminated Sets C and D** that
> the manuscript declares **WITHDRAWN**. The 2026-08-25 realignment moved those sets to
> `withdrawn/` in the repository **tree**, but release assets are stored separately and
> were never touched. `benchmark_set_C.tar.gz` already has 2 recorded downloads.
> The release tag also points at `34e8430`, **not** at `main`, so a reader landing on
> "Releases" sees the pre-withdrawal state of the repository.
>
> No assets exist yet for `set_C_clean`, `set_D_clean`, `set_F`, `set_G`, `set_H`, or for
> any result file. This is the clean "before" baseline.

---

## What would have to change, and what re-publication it would cost

**Not fixed here — recorded for the author's decision, as the contract requires.**

| # | Defect | Correct state | Cost |
|---|---|---|---|
| 1 | PyPI package ships no model, so "v0.3.1 … containing model V5 (SHA256 b84346…)" does not describe the PyPI artefact | Either bundle the model, or state plainly that the pip package fetches it on first run | Bundling a 162 MB model exceeds PyPI's default per-file limit and would need an exemption — **the realistic fix is documentation plus item 2** |
| 2 | `MODEL_URL` points at the **mutable `main`** branch and the download is **never checksummed** | Pin to `.../raw/v0.3.1/models/magicc_v5.onnx` (or the tag's LFS media URL) **and** verify SHA256 against `b84346…b3096` after download, failing loudly on mismatch | A code change ⇒ **a new release (0.3.2) on PyPI and GitHub**. Code availability's version string would move to 0.3.2 |
| 3 | Code availability / Installation do not warn that `pip install magicc` needs network on first run | One sentence: the model is downloaded from the repository on first use; offline users should clone the tag or pre-place the file at `$HOME/.magicc/magicc_v5.onnx` | Text only — **no re-release** |
| 4 | the internal repository release record lists `containers/magicc_0.3.1.sif` (367 MB), which does not exist | Name `containers/magicc_0.3.0.sif` (366 MiB), or actually build the 0.3.1 SIF | Text only, or one build |
| 5 | `magicc-data` release `v0.1.0` still serves withdrawn Sets C and D | Delete or clearly re-label those two assets when the new data release is cut | Data-deposition agent's call |

**Item 3 alone makes the manuscript truthful without any re-release.** Items 1-2 are the
substantive engineering fix; if the author wants Code availability to promise a
reproducible install, 0.3.2 is required. **The author decides.**

Nothing above affects a single number in the paper: the software's *output* is verified
byte-identical across all three sources.

---

## Evidence index (all under `results/revision/release_verification/`)

| File | Contents |
|---|---|
| `test_set_manifest.tsv` | the 25 genomes: name, bytes, SHA256 of source / plain copy / gzip copy |
| `pypi_check.json` | version resolution, artefact hashes, packaging, D7 tests, determinism |
| `github_check.json` | clone, tag, LFS verification, install, determinism, cross-source identity |
| `worktree_check.json` | conda-env run, model provenance, cross-source identity |
| `output_digests.tsv` | all 24 runs: source × mode × threads × extension → output SHA256 |
| `predictions_25genomes_canonical.tsv` | the single output every one of the 24 runs produced |
| `container_tag_facts.md` | task E |
| `magicc_data_before_state.json` | task F |
| `logs/` | every command's raw output: pip installs, PyPI JSON API, strace traces, all 24 run logs, clone log, installed file lists |

Virtualenvs, the clone and the copied genomes lived in a scratch directory and were
removed after the evidence was recorded. No GitHub token appears in any file here.

---

## Resource footprint (host was running a timing campaign)

Recorded honestly rather than glossed. All work ran off one to four cores in short bursts;
the 4-thread runs are required by §8.1 and each lasted seconds on 25 genomes.

| Item | Size |
|---|---|
| Test set copied out of `set_E/fasta/` | 119 MB (91 MiB plain + 28 MiB gzip) |
| Two fresh virtualenvs (numba, scipy, onnxruntime ×2) | 517 MB each |
| Shallow clone of `v0.3.1` incl. the 162 MB LFS model | 334 MB |
| Model downloaded by the PyPI package during the D7 test | 162 MB |
| **Peak scratch** | **~1.6 GB**, all under `/tmp` on `/dev/sda4` (ext4, 1 TB free), **not** on `/path/to/data` |
| After cleanup | 0 — venvs, clone, test set and downloaded models removed |

This exceeded the "well under 1 GB" guidance. The floor is set by the dependency tree:
`onnxruntime` + `numba` + `llvmlite` + `scipy` is ~500 MB per environment and §8.1
requires **two** independent clean environments, plus a 162 MB LFS model and a 162 MB
run-time download that is itself the object of the D7 test. Nothing was written to
`/path/to/data` except this evidence directory (~100 KB).

Page cache over `data/benchmarks/set_E/` was not evicted: the host has 881 GB of RAM with
~682 GB free, and the 25 genomes read were copied *from* that directory, warming it
rather than displacing it. One misjudged `find /` was launched and killed within seconds;
no other filesystem-wide scan was run.
