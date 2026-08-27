# Table S4 (rewritten) — computational cost under a single, stated definition

## Definitions used everywhere in this table

**Wall clock** = elapsed real time from process start to the results file being written and the process exiting. It **includes** interpreter start-up, library imports, JIT warm-up, model and database loading, input parsing, computation and output writing. It is measured with `/usr/bin/time -v` on the tool's own top-level command, with no wrapper process. This is the only timing definition used; the submitted version of this table mixed this definition with tool-internal compute-phase timers (see Table S4g).

**Hardware (identical for every row):** single host, 2 sockets / 48 physical cores, 881 GiB RAM, ext4 on a local SATA disk (`/dev/sdb`). Linux 6.17. **No GPU was used by any tool**: MAGICC's released inference path is CPU ONNX Runtime, DeepCheck was run on CPU PyTorch, and CheckM2 and CoCoPyE are CPU-only.

**File I/O mode (identical for every row):** uncompressed `.fasta`, one genome per file, read from a directory on local disk; **warm page cache** (cold-cache results are Table S4e). Outputs written to the same filesystem.

**Idleness:** the host was otherwise idle. The campaign driver waited for the 1-minute load average to fall below 1.0 before each timed run, and the load average immediately before and after every run is recorded in `results/revision/speed/matched_thread_runs.tsv`.

**Repeats:** every cell is the **median** of *n* independent repeats with the full **[min–max]** range given. Tool × thread-count order was randomised within each repeat block (`results/revision/speed/run_plan.tsv`). **No run has been removed from any cell.** Where the range is wide, the reason is given below the table.

**Page-cache state, and the five wide ranges in this table.** The host was rebooted on 2026-07-30, so the campaign began with an empty page cache. Five runs — always the first run that touched a given input set or a given tool's database — therefore read that data from the block device while every later repeat read it from RAM. They are identified objectively, for all 92 runs, by `/usr/bin/time -v`'s **File system inputs** counter (`ru_inblock`, 512-byte blocks): a warm run reads 0 bytes. Exactly five runs read more than 50 MB, and they are exactly the five runs that produced an above-median wall clock; no run with zero disk reads is an outlier. They are **kept in the table** (they are inside every median and every range below) and are also reported separately as the cold-start measurement in Table S4e. Per-run evidence: `results/revision/speed/cold_start_forensics.tsv` and `results/revision/speed/outlier_investigation.json`.

---

## Table S4a — matched-thread wall clock, all four tools, identical input

**Denominator: 100 genomes, 478,109,066 bp (0.478 Gbp), mean 4.78 Mbp per genome.** This is a seeded (9.9% of Set E bp) random subsample of Set E, drawn with a CRC-32-seeded generator (`results/revision/speed/inputs/input_manifest.json` lists every member). A 100-genome subsample is used for the complete factorial because CheckM2 at 1 thread on the full 1,000-genome Set E costs an estimated 22 h per repeat (~66 h for three), which is not practical; the full-set numbers that were measured are Table S4b.

| Tool | Threads | Wall clock, median | Range [min–max] | *n* | Peak RSS (GB) | Genomes/min |
|---|---:|---:|---:|---:|---:|---:|
| **MAGICC v0.3.0 (model V5)** | 1 | **5.9 s** | 5.9 s–6.0 s | 3 | 0.39 | 1016.9 |
| **MAGICC v0.3.0 (model V5)** | 8 | **2.2 s** | 2.1 s–2.2 s | 3 | 0.36 | 2714.9 |
| **MAGICC v0.3.0 (model V5)** | 16 | **2.1 s** | 2.1 s–2.1 s | 3 | 0.36 | 2843.6 |
| **MAGICC v0.3.0 (model V5)** | 32 | **2.2 s** | 2.1 s–13.2 s | 3 | 0.36 | 2690.6 |
| CheckM2 1.0.1 | 1 | **2.20 h** | 2.19 h–2.20 h | 3 | 10.61 | 0.8 |
| CheckM2 1.0.1 | 8 | **20 m 08.7 s** | 19 m 55.2 s–20 m 33.5 s | 3 | 10.54 | 5.0 |
| CheckM2 1.0.1 | 16 | **11 m 27.9 s** | 11 m 19.7 s–12 m 03.8 s | 3 | 10.53 | 8.7 |
| CheckM2 1.0.1 | 32 | **9 m 19.7 s** | 9 m 13.0 s–9 m 23.6 s | 3 | 10.66 | 10.7 |
| CoCoPyE 0.5.0 | 1 | **23 m 59.1 s** | 23 m 37.3 s–26 m 32.5 s | 3 | 15.84 | 4.2 |
| CoCoPyE 0.5.0 | 8 | **4 m 59.1 s** | 4 m 57.0 s–5 m 25.1 s | 3 | 15.86 | 20.1 |
| CoCoPyE 0.5.0 | 16 | **3 m 36.1 s** | 3 m 33.4 s–3 m 36.4 s | 3 | 15.87 | 27.8 |
| CoCoPyE 0.5.0 | 32 | **5 m 53.9 s** | 5 m 39.4 s–8 m 15.1 s | 3 | 15.89 | 17.0 |
| DeepCheck (inference only) | 1 | **55.5 s** | 55.3 s–55.6 s | 3 | 0.67 | 108.0 |
| DeepCheck (inference only) | 8 | **9.8 s** | 9.7 s–9.8 s | 3 | 0.67 | 613.5 |
| DeepCheck (inference only) | 16 | **6.4 s** | 6.4 s–6.5 s | 3 | 0.71 | 939.0 |
| DeepCheck (inference only) | 32 | **5.0 s** | 5.0 s–5.0 s | 3 | 0.67 | 1204.8 |

DeepCheck cannot read FASTA. Its row is **inference only**, starting from CheckM2's pickled feature vectors; producing those vectors requires a complete CheckM2 run, so DeepCheck's end-to-end cost is its own row **plus** the CheckM2 row. This was footnoted in the submitted table and is restated here as a table row rather than a footnote because it changes the ranking.

## Table S4b — full Set E, the historical comparator

**Denominator: 1,000 genomes, 4,806,980,767 bp (4.807 Gbp), mean 4.81 Mbp per genome.** Same hardware, same wall-clock definition, same file I/O mode as Table S4a.

| Tool | Threads | Wall clock, median | Range [min–max] | *n* | Peak RSS (GB) | Genomes/min |
|---|---:|---:|---:|---:|---:|---:|
| **MAGICC v0.3.0 (model V5)** | 1 | **49.0 s** | 48.6 s–49.3 s | 6 | 0.65 | 1224.1 |
| **MAGICC v0.3.0 (model V5)** | 8 | **10.3 s** | 10.3 s–10.3 s | 3 | 0.50 | 5842.3 |
| **MAGICC v0.3.0 (model V5)** | 16 | **6.9 s** | 6.9 s–7.3 s | 3 | 0.50 | 8645.5 |
| **MAGICC v0.3.0 (model V5)** | 32 | **6.0 s** | 5.8 s–39.1 s | 3 | 0.50 | 9933.8 |
| CheckM2 1.0.1 | 32 | **1.17 h** | 1.17 h–1.17 h | 1 | 18.88 | 14.3 |
| CoCoPyE 0.5.0 | 32 | **52 m 57.5 s** | 52 m 57.5 s–52 m 57.5 s | 1 | 15.90 | 18.9 |
| DeepCheck (inference only) | 1 | **8 m 54.6 s** | 8 m 54.4 s–8 m 57.3 s | 3 | 1.29 | 112.2 |
| DeepCheck (inference only) | 8 | **1 m 18.1 s** | 1 m 18.0 s–1 m 18.7 s | 3 | 1.28 | 768.1 |
| DeepCheck (inference only) | 16 | **43.5 s** | 42.7 s–43.8 s | 3 | 1.28 | 1378.0 |
| DeepCheck (inference only) | 32 | **29.5 s** | 29.2 s–39.8 s | 3 | 1.28 | 2035.3 |

The MAGICC 32-thread range is wide (5.83 s – 39.06 s over 3 repeats) for the reason given above: the 39.06 s repeat was the third timed run of the whole campaign and fetched 4.388 GB — 90.1% of this input set — from disk, while both other repeats fetched 0 bytes. Its CPU utilisation was 341% against 1683% and 1752%, and MAGICC's own phase timers put the entire excess in feature extraction (36.7 s vs 3.5 s), the only phase that reads FASTA, with normalisation and ONNX inference unchanged. The 1-minute load average before that run was 0.83, *lower* than the two fast repeats, so contention is excluded. It is a cold-cache measurement, and it is reported as one in Table S4e.

## Table S4c — parallel scaling, stated as speed-up and efficiency

`speed-up(T) = wall(1 thread) / wall(T threads)`; `efficiency(T) = speed-up(T) / T`. Both use the same end-to-end wall clock as the rest of the table, so each tool's fixed start-up cost is inside the numerator and the denominator — which is what a user actually experiences, and which is why efficiency falls with thread count for **every** tool. **MAGICC's scaling is sub-linear:** on Set E it goes from 49.0 s at 1 thread to 6.04 s at 32 threads, a **8.1× speed-up at 32 threads, i.e. 25% parallel efficiency, not 32×.** Beyond 16 threads it gains almost nothing (6.94 s → 6.04 s) because the run is by then dominated by process start-up, ONNX model load and inference rather than by per-genome feature extraction.

| Tool | Input set (denominator) | 1 thr | 8 thr | 16 thr | 32 thr | Speed-up at 32 thr | Efficiency at 32 thr |
|---|---|---:|---:|---:|---:|---:|---:|
| CheckM2 1.0.1 | set_E_100 (100 genomes) | 2.20 h | 20 m 08.7 s | 11 m 27.9 s | 9 m 19.7 s | **14.1×** | **44%** |
| CoCoPyE 0.5.0 | set_E_100 (100 genomes) | 23 m 59.1 s | 4 m 59.1 s | 3 m 36.1 s | 5 m 53.9 s | **4.1×** | **13%** |
| DeepCheck (inference only) | set_E_100 (100 genomes) | 55.5 s | 9.8 s | 6.4 s | 5.0 s | **11.2×** | **35%** |
| **MAGICC v0.3.0 (model V5)** | set_E_100 (100 genomes) | 5.9 s | 2.2 s | 2.1 s | 2.2 s | **2.6×** | **8%** |
| DeepCheck (inference only) | set_E_full (1,000 genomes) | 8 m 54.6 s | 1 m 18.1 s | 43.5 s | 29.5 s | **18.1×** | **57%** |
| **MAGICC v0.3.0 (model V5)** | set_E_full (1,000 genomes) | 49.0 s | 10.3 s | 6.9 s | 6.0 s | **8.1×** | **25%** |

On the 100-genome subsample MAGICC's efficiency at 32 threads is only 8%, and CoCoPyE is **slower** at 32 threads (353.9 s) than at 16 (216.1 s). Both are consequences of a short run: the fixed cost cannot be parallelised, and for CoCoPyE oversubscription actively hurts. This is reported because it bounds how far any of these tools, MAGICC included, benefits from more cores.

## Table S4d — derived ratios, fully qualified (R1-m20)

A speed ratio is meaningless without the conditions under which it was obtained, so each ratio below carries them explicitly. **Ratios are quoted only between cells measured at the same thread count, on the same input set, on the same hardware, under the same wall-clock definition, in the same file I/O mode.** No "per-thread" ratio is quoted, because per-thread throughput is not a property of a tool: it falls as thread count rises (see the MAGICC rows of Table S4a). The last column gives the number of repeats behind each side of the ratio; the two full-Set-E competitor rows rest on a single competitor run each, because one repeat of CheckM2 on 1,000 genomes costs 70 minutes even at 32 threads.

| Comparison | Threads | Input set (denominator) | Wall-clock ratio | Repeats (num ÷ den) |
|---|---:|---|---:|---:|
| CheckM2 1.0.1 ÷ MAGICC | 1 | set_E_100 (100 genomes, 0.478 Gbp) | **1340×** | 3 ÷ 3 |
| CoCoPyE 0.5.0 ÷ MAGICC | 1 | set_E_100 (100 genomes, 0.478 Gbp) | **244×** | 3 ÷ 3 |
| CheckM2 1.0.1 ÷ MAGICC | 8 | set_E_100 (100 genomes, 0.478 Gbp) | **547×** | 3 ÷ 3 |
| CoCoPyE 0.5.0 ÷ MAGICC | 8 | set_E_100 (100 genomes, 0.478 Gbp) | **135×** | 3 ÷ 3 |
| CheckM2 1.0.1 ÷ MAGICC | 16 | set_E_100 (100 genomes, 0.478 Gbp) | **326×** | 3 ÷ 3 |
| CoCoPyE 0.5.0 ÷ MAGICC | 16 | set_E_100 (100 genomes, 0.478 Gbp) | **102×** | 3 ÷ 3 |
| CheckM2 1.0.1 ÷ MAGICC | 32 | set_E_100 (100 genomes, 0.478 Gbp) | **251×** | 3 ÷ 3 |
| CoCoPyE 0.5.0 ÷ MAGICC | 32 | set_E_100 (100 genomes, 0.478 Gbp) | **159×** | 3 ÷ 3 |
| CheckM2 1.0.1 ÷ MAGICC | 32 | Set E (1,000 genomes, 4.807 Gbp) | **697×** | 1 ÷ 3 |
| CoCoPyE 0.5.0 ÷ MAGICC | 32 | Set E (1,000 genomes, 4.807 Gbp) | **526×** | 1 ÷ 3 |

**Memory ratio**, same qualification (peak RSS, `/usr/bin/time -v`, set_E_100, 100 genomes):

| Comparison | Peak-RSS ratio |
|---|---:|
| CheckM2 1.0.1 ÷ MAGICC | **27.7×** |
| CoCoPyE 0.5.0 ÷ MAGICC | **41.2×** |
| DeepCheck (inference only) ÷ MAGICC | **1.8×** |

## Table S4e — cold versus warm page cache (R1-m12)

**A correction, stated first.** The campaign included a designed cold-cache arm using an unprivileged, file-scoped eviction (`posix_fadvise(POSIX_FADV_DONTNEED)` on every input FASTA and on the tool's model/database files, since `/proc/sys/vm/drop_caches` needs root and passwordless sudo is not configured on this host). **That arm did not work.** The evictor (`scripts/162_ws8_drop_file_cache.py`) raised `TypeError: underlying buffer is not writable` on the first file it inspected — it built a read-only mmap and then asked `ctypes` for a writable pointer into it — and because only `OSError` was caught, the process exited before a single `posix_fadvise` call was made. All 11 `*.evict.txt` logs contain that identical traceback, and every run labelled `cache=cold` recorded `File system inputs = 0`, i.e. it read nothing from disk. **The cells labelled "cold" were warm runs**, which is exactly why they agree with the warm cells to within 3% (MAGICC, Set E, 1 thread: 49.07 s "cold" vs 49.02 s warm). They are reported here as what they are — additional warm repeats — and are not presented as cold-cache results. The fixed evictor is `scripts/176_ws8_evict_fixed.py`, verified to evict 100% of 5.04 GB of resident pages, with a re-runner (`scripts/177_ws8_cold_verified.sh`). That re-run has **not** been executed: an unrelated single-core job appeared on the host, and running it under contention would have broken the campaign's own idleness rule. It is not needed for the numbers below, which come from a stricter cold condition.

**What the cold-start numbers below actually are, and why they are stronger than the arm that failed.** The host was rebooted on 2026-07-30 and the campaign started with a genuinely empty page cache, so the first run to touch each input set and each tool's database is a **true whole-system cold start** — colder than the file-scoped eviction that was intended, because kernel, libc and conda shared objects were cold too. Those runs are identified objectively by non-zero `File system inputs` (`ru_inblock` × 512 B) and each is a single observation (*n* = 1); the warm figure beside it is the median of the remaining repeats of the same cell, on the same hardware, at the same thread count, in the same file-I/O mode.

| Tool | Input set (denominator) | Threads | Warm (median, *n*) | Cold start (*n* = 1) | Cold − warm | Cold ÷ warm | Read from disk |
|---|---|---:|---:|---:|---:|---:|---:|
| MAGICC v0.3.0 (model V5) | set_E_full | 32 | 5.9 s (n=5) | 39.1 s | +33.2 s | 6.67× | 4.39 GB |
| MAGICC v0.3.0 (model V5) | set_E_100 | 32 | 2.2 s (n=2) | 13.2 s | +11.1 s | 6.05× | 0.96 GB |
| CoCoPyE 0.5.0 | set_E_100 | 32 | 5 m 53.9 s (n=3) | 8 m 15.1 s | +141.2 s | 1.40× | 17.09 GB |
| DeepCheck (inference only) | set_E_full | 32 | 29.4 s (n=2) | 39.8 s | +10.5 s | 1.36× | 1.02 GB |
| CheckM2 1.0.1 | set_E_100 | 8 | 20 m 02.0 s (n=2) | 20 m 33.5 s | +31.5 s | 1.03× | 4.26 GB |

**Reading of this table.** A cold cache costs MAGICC +33 s on 1,000 genomes at 32 threads — it is the tool whose warm run is short enough for I/O to dominate, so *proportionally* it is hit hardest (6.7×). Stated precisely rather than favourably: MAGICC's worst measured cold start on 1,000 genomes at 32 threads (39.1 s) is still 81× faster than CoCoPyE's **warm** run (3,177.5 s) and 108× faster than CheckM2's **warm** run (4,208.0 s) on the same input — but it is **slower than DeepCheck's warm inference-only run** (29.5 s). That DeepCheck figure excludes the mandatory CheckM2 run (4,208.0 s at the same thread count) that produces its input, so DeepCheck's true end-to-end cost is still the larger; the comparison is stated this way because the inference-only row on its own does beat a cold MAGICC. In the other direction, CoCoPyE pays +141 s on only 100 genomes purely to page in its 17.0 GB database. A cold cache is a once-per-boot cost for every tool here; in production all of them run warm after the first invocation.

## Table S4f — total cost of ownership (R1-m12)

Two different quantities are reported here and are never added together silently: the **reference data** a user must download before the tool can run at all, and the **install footprint** of the software itself. Sizes are apparent bytes (`du -sb`) measured on the benchmark host on 2026-08-01T00:59:45Z.

| Tool | Reference data (must download) | First-run download | Install footprint | Total before first run |
|---|---:|---:|---:|---:|
| MAGICC v0.3.0 (V5) | **0.172 GB** | 7 s | 0.75 GB | 0.92 GB |
| CheckM2 1.0.1 | **3.083 GB** | 2.0 min *(projected)* | 2.77 GB | 5.85 GB |
| CoCoPyE 0.5.0 | **17.045 GB** | 11.2 min *(projected)* | **not separately measurable** | — |
| DeepCheck | **0.090 GB** | 4 s *(projected)* | **not separately measurable** | — |

**Download times.** MAGICC's is **measured**: the released CLI's own model URL fetched three times, median 6.7 s at 25.3 MB/s. The competitor times are **projections** of their exact download volumes at that same measured throughput, not measurements: their databases were already installed and re-downloading them would have overwritten a working installation and measured this site's network rather than the tool.

**How the install-footprint figures were obtained, and what they include.**

- **MAGICC v0.3.0 (V5)** — MEASURED: throwaway conda env containing only python 3.11 + magicc + its 5 declared runtime dependencies (13 pip packages in total), built from the released package, `du -sb`, then deleted. 0.241 GB of that is the bare CPython interpreter; MAGICC and its dependencies add 0.508 GB. Verified runnable (`magicc --help` rc=0, `import magicc` -> 0.3.0).
- **CheckM2 1.0.1** — MEASURED: `checkm2_py39` is a dedicated environment for this tool alone (29 conda / 58 pip packages), `du -sb`.
- **CoCoPyE 0.5.0** — NOT SEPARATELY MEASURABLE on this host: installed into the shared analysis environment, so its software footprint cannot be isolated without a clean reinstall. No number is reported rather than a wrong one. Its reference data is exact and is reported.
- **DeepCheck** — NOT SEPARATELY MEASURABLE on this host: no installer; run from a git checkout inside the shared analysis environment. It additionally requires a complete CheckM2 installation (environment + 3.083 GB database) to produce its input, so its true footprint is at least CheckM2's.

**Footnote — the analysis environment is not the install footprint.** The conda environment used to *develop* this paper (`/path/to/conda/envs/magicc2`, 8.18 GB, 65 conda / 133 pip packages) carries the training and plotting stack — `cocopye==0.5.0`, `lightgbm==4.6.0`, `matplotlib==3.10.8`, `optuna==4.7.0`, `pandas==3.0.0`, `scikit-learn==1.3.1`, `torch==2.5.1+cu121`, `xgboost==3.1.3` — and MAGICC, CoCoPyE and DeepCheck were all run from it. It is therefore **one shared environment, not three tool footprints**, and an earlier version of this table wrongly charged its size to each of the three. None of those packages is required to run MAGICC: `pyproject.toml` declares five runtime dependencies (`numpy>=1.20`, `numba>=0.53`, `scipy>=1.7`, `h5py>=3.0`, `onnxruntime>=1.10`) and the released inference path is CPU ONNX Runtime. PyTorch is a training-time dependency only.

**Containers (WS7).** The most reproducible install figure is the pinned image: Docker `magicc:0.3.0` is **1.06 GB** and the Apptainer `.sif` is **0.38 GB**, built from 13 fully pinned packages with the ONNX model bundled. Container sizes include a base-OS layer and the bundled ONNX model, so they are NOT directly comparable with a bare conda environment; they are the most reproducible figure because every layer is pinned. The conda probe above resolves dependencies to their *current* versions, so it measures the install cost of the released package rather than the exact versions used for the timings; the container is the version-exact artefact.

**Cold versus warm cache** is Table S4e. **Reference-data sizes are unaffected by this correction** and stand as measured.

### The honest summary

MAGICC's reference data is **0.172 GB** — 99× smaller than CoCoPyE's and 18× smaller than CheckM2's — and it is a single ONNX model plus a k-mer list, not a sequence database. Its install footprint (0.75 GB, of which 0.24 GB is the Python interpreter itself) is smaller than CheckM2's dedicated environment (2.77 GB), but the two are not a like-for-like comparison of packaging quality: CheckM2's environment is a conda environment with 29 conda packages including its own Python, and MAGICC's was built by pip into a bare interpreter. CoCoPyE's and DeepCheck's software footprints are simply not known on this host and no number is invented for them.

## Table S4g — reconciliation of the previously published timings (R1-m11)

| Figure | Where it appeared | What it actually measured | Status |
|---|---|---|---|
| **40 s** for 1,000 genomes | Main text | **Not a measurement.** 1,000 ÷ 1,451 genomes min⁻¹ thread⁻¹ = 41.4 s, rounded to "40 seconds". No run of that duration exists in any log in the repository. | **Withdrawn** |
| **1,451** genomes min⁻¹ thread⁻¹ | Table S4, Table 1, Abstract | The arithmetic **mean of five per-benchmark-set rates**, each computed from MAGICC's internal compute-phase timer (feature extraction + ONNX inference only — excluding interpreter start, imports, JIT warm-up, model load, normalisation and output write), with magicc_v3.onnx at 1 thread. Averaging rates over sets of unequal per-genome cost inflates the value: pooled over the same five sets the same timer gives **1066** genomes min⁻¹ thread⁻¹, a factor of 1.36. | **Withdrawn** |
| **97.5 s** for 1,000 genomes | Table S4 | A genuine **end-to-end** wall clock (`/usr/bin/time -v`), 1 thread, Set E, magicc_v3.onnx, invoked through `conda run`. | **Correct as a V3 measurement**; superseded by the V5 re-measurement above |
| **74.4 s / 7.9 s** | `results/phase7_set_e_test/` | End-to-end wall clock of `python -m magicc predict` at 1 and 43 threads, Set E, magicc_v3.onnx, **without** the `conda run` wrapper. The 23 s difference from the 97.5 s figure cannot be attributed from the archive: neither run recorded load average or page-cache state. | **Correct as V3 measurements**; superseded |

**The internal inconsistency the reviewer detected is real and is in the table itself.** The submitted Table S4 row reads *MAGICC | 1 thread | 0.66 GB | 97.5 s | 1,451 genomes/min/thread*, but 1,000 genomes in 97.5 s is 615 genomes min⁻¹ thread⁻¹, not 1,451 — a factor of 2.36. The wall-clock column and the throughput column came from two different experiments with two different timers. The same defect affects the competitor rows: CheckM2's row states 32 threads and 86 min 37 s, which is 0.36 genomes min⁻¹ thread⁻¹, while the table prints 0.82. **The published 1,700× and 2,100× ratios therefore divide one tool's mean-of-rates by another tool's mean-of-rates, and neither is the wall clock printed beside it. They are withdrawn and replaced by Table S4d.**

