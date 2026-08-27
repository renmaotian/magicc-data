# WS8 — matched-hardware speed and memory (R1-M7, R1-m11, R1-m12, R1-m20)

Generated from `results/revision/speed/`. 92 timed runs; every run has its own JSON under `runs/`.

## Measurement contract

| | |
|---|---|
| **Wall clock** | process start → results file written and process exited; **includes** interpreter start, imports, JIT warm-up, model/database load, input parsing, computation, output write. `/usr/bin/time -v`, no wrapper process. |
| **Peak memory** | `/usr/bin/time -v` *Maximum resident set size*, whole process tree. |
| **Hardware** | one host, 48 cores (2 sockets), 881 GiB RAM, ext4 on local `/dev/sdb`, Linux 6.17. Identical for every row. |
| **Accelerator** | none. All four tools were run CPU-only; MAGICC's released inference path is CPU ONNX Runtime. |
| **File I/O** | uncompressed `.fasta`, one genome per file, local disk, warm page cache unless stated. |
| **Repeats** | ≥3 per cell; median reported with the full [min–max] range. |
| **Idleness** | the driver waits for the 1-minute load average to fall below 1.0 before every timed run; the load average immediately before and after each run is in `matched_thread_runs.tsv`, together with the summed %CPU of every process on the host immediately before the run (`sum_pcpu_all_processes_before`). No run was started with a 1-minute load average at or above 1.0. |
| **Run order** | tool × thread-count order randomised within each repeat block with a CRC-32 seed (`run_plan.tsv`), so drift in machine state cannot be confounded with tool identity. |

### What `--threads N` means for each tool

- **MAGICC**: `N` worker *processes* for k-mer feature extraction (`multiprocessing.Pool`); ONNX inference is hard-coded to 1 intra-op and 1 inter-op thread (`magicc/cli.py:567`). `OMP_NUM_THREADS=1` was set so the N workers do not each oversubscribe.
- **CheckM2**: `--threads N`, passed to Prodigal parallelisation and to DIAMOND.
- **CoCoPyE**: `-t N`, passed to UProC.
- **DeepCheck**: `torch.set_num_threads(N)`; there is no other parallelism.

---

## Input sets

| Set | Genomes | Total sequence bp | Mean genome | On disk |
|---|---:|---:|---:|---:|
| **set_E_100** (primary matched factorial) | 100 | 478,109,066 (0.478 Gbp) | 4.78 Mbp | 0.484 GB |
| **Set E** (historical comparator) | 1,000 | 4,806,980,767 (4.807 Gbp) | 4.81 Mbp | 4.871 GB |

set_E_100 is a seeded random subsample of Set E (CRC-32 seed 3017395416 from the string `magicc-ws8-speed-subsample-v1`), carrying 9.95 % of Set E's bases. Every member is listed in `inputs/input_manifest.json`.

## 8.1 Matched-thread wall clock — **denominator: 100 genomes, 0.478 Gbp**

| Tool | 1 thread | 8 threads | 16 threads | 32 threads | Peak RSS (GB) |
|---|---:|---:|---:|---:|---:|
| MAGICC v0.3.0 (V5) | **5.9 s** [5.9 s–6.0 s] n=3 | **2.2 s** [2.1 s–2.2 s] n=3 | **2.1 s** [2.1 s–2.1 s] n=3 | **2.2 s** [2.1 s–13.2 s] n=3 | 0.39 |
| CheckM2 1.0.1 | **2.20 h** [2.19 h–2.20 h] n=3 | **20 m 08.7 s** [19 m 55.2 s–20 m 33.5 s] n=3 | **11 m 27.9 s** [11 m 19.7 s–12 m 03.8 s] n=3 | **9 m 19.7 s** [9 m 13.0 s–9 m 23.6 s] n=3 | 10.66 |
| CoCoPyE 0.5.0 | **23 m 59.1 s** [23 m 37.3 s–26 m 32.5 s] n=3 | **4 m 59.1 s** [4 m 57.0 s–5 m 25.1 s] n=3 | **3 m 36.1 s** [3 m 33.4 s–3 m 36.4 s] n=3 | **5 m 53.9 s** [5 m 39.4 s–8 m 15.1 s] n=3 | 15.89 |
| DeepCheck (inference only) | **55.5 s** [55.3 s–55.6 s] n=3 | **9.8 s** [9.7 s–9.8 s] n=3 | **6.4 s** [6.4 s–6.5 s] n=3 | **5.0 s** [5.0 s–5.0 s] n=3 | 0.71 |

### Full Set E — **denominator: 1,000 genomes, 4.807 Gbp**

| Tool | Threads | Wall clock, median [min–max] | *n* | Peak RSS (GB) |
|---|---:|---:|---:|---:|
| MAGICC v0.3.0 (V5) | 1 | **49.0 s** [48.6 s–49.3 s] | 6 | 0.65 |
| MAGICC v0.3.0 (V5) | 8 | **10.3 s** [10.3 s–10.3 s] | 3 | 0.50 |
| MAGICC v0.3.0 (V5) | 16 | **6.9 s** [6.9 s–7.3 s] | 3 | 0.50 |
| MAGICC v0.3.0 (V5) | 32 | **6.0 s** [5.8 s–39.1 s] | 3 | 0.50 |
| CheckM2 1.0.1 | 32 | **1.17 h** [1.17 h–1.17 h] | 1 | 18.88 |
| CoCoPyE 0.5.0 | 32 | **52 m 57.5 s** [52 m 57.5 s–52 m 57.5 s] | 1 | 15.90 |
| DeepCheck (inference only) | 1 | **8 m 54.6 s** [8 m 54.4 s–8 m 57.3 s] | 3 | 1.29 |
| DeepCheck (inference only) | 8 | **1 m 18.1 s** [1 m 18.0 s–1 m 18.7 s] | 3 | 1.28 |
| DeepCheck (inference only) | 16 | **43.5 s** [42.7 s–43.8 s] | 3 | 1.28 |
| DeepCheck (inference only) | 32 | **29.5 s** [29.2 s–39.8 s] | 3 | 1.28 |

## 8.3 Reconciliation of 40 s vs 97.5 s (R1-m11)

The two numbers were never measurements of the same thing, and they were never even measurements of the same experiment. 97.5 s is a real end-to-end wall-clock measurement of one process. 40 s is not a measurement at all: it is 1,000 divided by the 1,451 genomes/min/thread figure printed in the same table, and 1,451 is the arithmetic mean of five per-benchmark-set throughput rates computed from MAGICC's own internal compute-phase timer. Table S4 therefore places, in a single row, a wall-clock column and a throughput column that disagree with each other by 2.36x.

| Figure | Status |
|---|---|
| **40 s** | **Withdrawn.** Not a measurement: 1,000 ÷ 1,451 = 41.4 s. |
| **1,451 genomes/min/thread** | **Withdrawn.** Arithmetic mean of five per-set rates from MAGICC's internal compute-phase timer (V3 model). Pooled correctly the same timer gives 1066. |
| **97.5 s** | **Survives** as a correct V3 end-to-end measurement; superseded. |
| **74.4 s / 7.9 s** | **Survive** as correct V3 end-to-end measurements; superseded. |

**How much of MAGICC's real cost the retired timer never saw** (`magicc_phase_breakdown.tsv`):

| Input set | Threads | End-to-end wall | Compute phase only | Hidden |
|---|---:|---:|---:|---:|
| set_E_100 | 1 | 5.9 s | 5.1 s | 14 % |
| set_E_100 | 8 | 2.2 s | 1.5 s | 33 % |
| set_E_100 | 16 | 2.1 s | 1.4 s | 36 % |
| set_E_100 | 32 | 2.2 s | 1.5 s | 33 % |
| set_E_full | 1 | 49.0 s | 47.9 s | 2 % |
| set_E_full | 8 | 10.3 s | 9.2 s | 11 % |
| set_E_full | 16 | 6.9 s | 5.9 s | 16 % |
| set_E_full | 32 | 6.0 s | 5.1 s | 16 % |

**Does the `conda run` wrapper explain the historical 74.4 s vs 97.5 s gap?** Same binary, same input, same thread count, interleaved repeats:

| Invocation | Wall clock, median [min–max] | *n* |
|---|---:|---:|
| `conda run -n magicc2 magicc predict` | **50.8 s** [50.1 s–50.8 s] | 3 |
| direct `magicc predict` | **49.0 s** [49.0 s–49.3 s] | 3 |

**No.** The wrapper costs 1.8 s (4 %), not 23 s. The historical 74.4 s vs 97.5 s gap therefore cannot be attributed to `conda run`, and neither archived run recorded load average or page-cache state, so it cannot be attributed from the archive at all. Both are superseded by the V5 re-measurement, which records both.

## 8.2 Total cost of ownership (R1-m12)

| Tool | Reference data on disk | First-run download | Notes |
|---|---:|---:|---|
| MAGICC v0.3.0 (V5) | 0.172 GB | 6.7 s (**measured**, n=3) | No sequence database of any kind. The 'database' is the ONNX model plus the selected 9-mer list; it ships with the package or auto-downloads once to ~/.magicc/. |
| CheckM2 1.0.1 | 3.083 GB | 2.0 min (projected) | uniref100.KO.1.dmnd, a DIAMOND-formatted UniRef100 subset. Downloaded once via `checkm2 database --download`. |
| CoCoPyE 0.5.0 | 17.045 GB | 11.2 min (projected) | UProC Pfam databases (v24 + v28), UProC model, and the CoCoPyE reference database, downloaded by `cocopye toolbox download-dependencies`. |
| DeepCheck | 0.090 GB | 4 s (projected) | DeepCheck ships only a model; it has no sequence database. It cannot read FASTA at all: its input is CheckM2's pickled feature vectors, so in practice its true installed footprint is its own 86 MB PLUS the whole CheckM2 install and the 2.9 GB CheckM2 database. |

MAGICC's first-run setup was **measured directly**: 6.7 s median over 3 downloads of the 169.7 MB ONNX model from the URL the released CLI uses, at 25.3 MB/s. That is the whole of MAGICC's setup — there is no database to fetch, build or index. Competitor download times are **projections** at that same measured throughput, because re-downloading their databases would overwrite working installations; the byte volumes are exact.

**Install footprint (corrected).** An earlier version of this table charged MAGICC, CoCoPyE and DeepCheck the same 8.18 GB, which is the size of the shared conda environment used to *develop* this paper (65 conda / 133 pip packages, containing `cocopye==0.5.0`, `lightgbm==4.6.0`, `matplotlib==3.10.8`, `optuna==4.7.0`, `pandas==3.0.0`, `scikit-learn==1.3.1`, `torch==2.5.1+cu121`, `xgboost==3.1.3`). One shared environment is not three tool footprints, and none of those packages is needed to run MAGICC. What a user actually installs was therefore measured directly: a throwaway conda environment with only python 3.11, magicc and its five declared runtime dependencies is **0.75 GB** (13 pip packages), of which 0.24 GB is the bare CPython interpreter; it was verified runnable and then deleted. CheckM2's `checkm2_py39` **is** a dedicated environment and its 2.77 GB stands. CoCoPyE's and DeepCheck's software footprints are **not separately measurable** on this host and no number is reported for them. The version-exact artefact is the pinned container: Docker 1.06 GB / Apptainer 0.38 GB, which include a base-OS layer and the bundled model and so are not comparable with a bare conda environment. Full detail: `env_footprint.json`, `tco_table.md`.

### Cold vs warm page cache

**The designed cold-cache arm failed and its cells are invalid.** `scripts/162_ws8_drop_file_cache.py` raised `TypeError: underlying buffer is not writable` on the first file it inspected and exited before any `posix_fadvise(DONTNEED)` call; all 11 `*.evict.txt` logs carry that identical traceback and every `cache=cold` run recorded `File system inputs = 0`. Those cells are warm runs with a cold label and are treated as extra warm repeats. The fixed evictor is `scripts/176_ws8_evict_fixed.py` (verified to evict 100 % of 5.04 GB of resident pages) with re-runner `scripts/177_ws8_cold_verified.sh`; the re-run was not executed because an unrelated job appeared on the host and the campaign's idleness rule would have been broken.

**The cold-start numbers reported instead are stronger.** The host was rebooted on 2026-07-30, so the first run to touch each input set and each tool's database read it from the block device — a true whole-system cold cache. Those runs are identified for all 92 runs by non-zero `/usr/bin/time -v` *File system inputs*; exactly five qualify, and they are exactly the five runs that produced an above-median wall clock. No run is discarded: each is inside its cell's median and range in the tables above, and is also reported here. Evidence: `cold_start_forensics.tsv`, `outlier_investigation.json`, `cold_vs_warm_corrected.tsv`.

| Tool | Input set | Threads | Warm median (*n*) | Cold start (*n*=1) | Cold − warm | Cold ÷ warm | Read from disk |
|---|---|---:|---:|---:|---:|---:|---:|
| MAGICC v0.3.0 (V5) | set_E_full | 32 | 5.9 s (5) | 39.1 s | +33.2 s | 6.67× | 4.39 GB |
| MAGICC v0.3.0 (V5) | set_E_100 | 32 | 2.2 s (2) | 13.2 s | +11.1 s | 6.05× | 0.96 GB |
| CoCoPyE 0.5.0 | set_E_100 | 32 | 5 m 53.9 s (3) | 8 m 15.1 s | +141.2 s | 1.40× | 17.09 GB |
| DeepCheck (inference only) | set_E_full | 32 | 29.4 s (2) | 39.8 s | +10.5 s | 1.36× | 1.02 GB |
| CheckM2 1.0.1 | set_E_100 | 8 | 20 m 02.0 s (2) | 20 m 33.5 s | +31.5 s | 1.03× | 4.26 GB |

## 8.1b Parallel scaling — sub-linear for every tool

`speed-up(T) = wall(1 thread) / wall(T threads)`, `efficiency = speed-up / T`, both on the same end-to-end wall clock, so each tool's unparallelisable start-up cost is included. **MAGICC gains 8.1× from 32 threads on 1,000 genomes (49.0 s → 6.04 s), i.e. 25 % parallel efficiency — not 32×** — and essentially nothing beyond 16 threads (6.94 s → 6.04 s).

| Tool | Input set | Speed-up at 8 / 16 / 32 threads | Efficiency at 32 threads |
|---|---|---:|---:|
| CheckM2 1.0.1 | set_E_100 | 6.5× / 11.5× / 14.1× | **44 %** |
| CoCoPyE 0.5.0 | set_E_100 | 4.8× / 6.7× / 4.1× | **13 %** |
| DeepCheck (inference only) | set_E_100 | 5.7× / 8.7× / 11.2× | **35 %** |
| MAGICC v0.3.0 (V5) | set_E_100 | 2.7× / 2.8× / 2.6× | **8 %** |
| DeepCheck (inference only) | set_E_full | 6.8× / 12.3× / 18.1× | **57 %** |
| MAGICC v0.3.0 (V5) | set_E_full | 4.8× / 7.1× / 8.1× | **25 %** |

The reason is a fixed cost that no thread count removes. Solving `wall(n) = a + b·n` from the two measured workload sizes at 32 threads gives MAGICC a fixed start-up of **1.81 s** and a marginal cost of 4.2 ms per genome, so start-up is 30 % of the 1,000-genome run and 81 % of the 100-genome run (`fixed_variable_costs.tsv`).

## Files

- `results/revision/speed/WS8_REPORT.md`
- `results/revision/speed/cold_start_forensics.tsv`
- `results/revision/speed/cold_vs_warm.tsv`
- `results/revision/speed/cold_vs_warm_corrected.tsv`
- `results/revision/speed/env_footprint.json`
- `results/revision/speed/fixed_variable_costs.tsv`
- `results/revision/speed/magicc_phase_breakdown.tsv`
- `results/revision/speed/matched_thread_runs.tsv`
- `results/revision/speed/matched_thread_summary.tsv`
- `results/revision/speed/matched_thread_table.md`
- `results/revision/speed/outlier_investigation.json`
- `results/revision/speed/reconciliation_40s_vs_97.5s.json`
- `results/revision/speed/reconciliation_per_set_internal_timers.tsv`
- `results/revision/speed/run_plan.tsv`
- `results/revision/speed/scaling_efficiency.tsv`
- `results/revision/speed/setup_cost.json`
- `results/revision/speed/table_S4_rewritten.md`
- `results/revision/speed/table_S4_rewritten.tsv`
- `results/revision/speed/tco.json`
- `results/revision/speed/tco_table.md`
- `results/revision/speed/tco_table.tsv`
- `results/revision/speed/ws8_summary.json`
- `results/revision/speed/runs/` — one JSON + `/usr/bin/time -v` file + stdout per run
- `results/revision/speed/figures/` — CVD-safe figures + `palette_cvd_report.tsv`

