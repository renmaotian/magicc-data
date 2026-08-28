# WS11.T / v0.3.3 — MAGICC re-measured on the released artefact

**Campaign 2026-08-28, 15:28–15:46 UTC. 48 timed cells, all rc = 0, all 1,000 / 100 output rows
present, ZERO flagged runs** (no cell started above load 3.79; every 1-thread cell started below
1.5; max `File system inputs` across all 48 runs = 8 blocks = 4 kB, i.e. fully warm).
Competitor cells were **not** re-run. Nothing outside this directory was written.

## What was measured, and why the arms are what they are

`magicc` **0.3.3 was installed from PyPI into a clean conda environment**
(`envs/magicc_v033_rel`, Python 3.11.16) and *that* binary was timed, so the table describes
what a reader gets. **Every `.py` file and both packaged data files in the released wheel are
byte-identical to the working tree** (checked file by file), so the working tree is the release
for timing purposes. The environments are not identical, however, and that matters:

| | WS11.T env (`magicc2`) | released env (`magicc_v033_rel`) |
|---|---|---|
| Python | 3.11.0 | 3.11.16 |
| numpy | 1.26.4 | **2.4.6** |
| numba | 0.63.1 | **0.67.0** |
| onnxruntime | 1.23.2 | **1.29.0** |
| scipy / h5py | 1.17.0 / 3.15.1 | 1.17.1 / 3.16.0 |

**A crucial detail about what v0.3.3 verifies.** `--help` states, and the code confirms, that
an explicit `--model` path "is used as given and is **NOT** checksum-verified, so that
alternative models can be evaluated deliberately". The WS11.T harness passed `--model`
explicitly. **Those rows are therefore not invalidated by the version bump — they measure the
developer invocation, which behaves identically under 0.3.3.** The +0.5 s applies to the
*default* resolution path, which is what a released user gets. Both were measured:

* **`magicc_v033`** — `magicc predict` with **no** `--model`. The CLI resolves the model itself
  (package data → project layout → `~/.magicc`) and verifies its SHA-256 **before every load**.
  It resolved to `~/.magicc/magicc_v5.onnx`, 169,658,949 B, SHA-256 `b843466…2b3096`.
  **This is the released user's cost and the headline row.**
* **`magicc_v033_xmodel`** — identical, plus `--model ~/.magicc/magicc_v5.onnx`, which bypasses
  verification. **The same file**, so the two arms differ by the verification pass and nothing
  else, and the A − B contrast is that pass alone.

This gives a clean three-way decomposition rather than one confounded delta.

## Revised medians (n = 3 per cell, warm cache, `/usr/bin/time -v`, same wall-clock definition)

**Released path (`magicc_v033`) — quote these.**

| Input (denominator) | 1 thr | 8 thr | 16 thr | 32 thr | Peak RSS |
|---|---:|---:|---:|---:|---:|
| **Set E — 1,000 genomes, 4,806,980,767 bp** | **49.91 s** | **10.69 s** | **7.29 s** | **6.26 s** | 0.515–0.718 GB |
| 100 genomes, 478,109,066 bp | 6.64 s | 2.73 s | 2.47 s | 2.50 s | 0.418–0.454 GB |

Explicit-`--model` arm, for comparison with WS11.T: 49.30 / 10.28 / 6.90 / **5.74** s (Set E) and
6.19 / 2.29 / 2.06 / **2.09** s (100 genomes). Full ranges in `cell_summary_v033.tsv`.

## The delta, decomposed (32 threads, 1,000 genomes)

| Step | Wall | Change | Cause |
|---|---:|---:|---|
| WS11.T (`magicc2` env, explicit `--model`) | 5.98 s | — | the archived campaign row |
| v0.3.3 released env, explicit `--model` | 5.74 s | **−0.24 s** | newer numpy / numba / onnxruntime |
| v0.3.3 released env, default path | **6.26 s** | **+0.52 s** | **model SHA-256 verification** |
| **net WS11.T → released** | | **+0.28 s (+4.7 %)** | |

The same at 100 genomes / 32 threads: 2.21 → 2.09 (−0.12 s, dependency stack) → **2.50 s**
(+0.41 s, verification); net +0.29 s (+13 %). **The net move is smaller than a bare +0.51 s
would imply, because the newer dependency stack gives roughly half of it back.**

## It is a fixed cost — three independent confirmations

**1. Direct measurement of the pass itself** (`model_verification_cost.tsv`, magicc's own
`_verify_model`, n = 7): **0.398 s** [0.397–0.399], i.e. 426.7 MB/s over 169,658,949 B.

**2. The end-to-end delta does not scale with input** (`verification_cost_by_cell.tsv`):

| Threads | 100 genomes | 1,000 genomes |
|---:|---:|---:|
| 1 | 0.45 s | 0.61 s |
| 8 | 0.44 s | 0.41 s |
| 16 | 0.41 s | 0.39 s |
| 32 | 0.41 s | 0.52 s |

A **10× larger input** changes the delta by nothing outside run-to-run noise; mean 0.43 s vs
0.48 s. A cost proportional to input would have grown tenfold.

**3. It lands entirely in the fixed term of the cost model** (`fixed_variable_costs_v033.tsv`),
which is the strongest form of the statement:

| Threads | Fixed start-up, unverified → verified | Marginal s per genome, unverified → verified |
|---:|---|---|
| 1 | 1.400 → 1.832 s (**+0.432**) | 0.047900 → 0.048078 (unchanged) |
| 8 | 1.402 → 1.846 s (**+0.443**) | 0.008878 → 0.008844 (unchanged) |
| 16 | 1.522 → 1.934 s (**+0.412**) | 0.005378 → 0.005356 (unchanged) |
| 32 | 1.684 → 2.082 s (**+0.398**) | 0.004056 → 0.004178 (unchanged) |

The fixed term rises by **+0.398 to +0.443 s (mean +0.421 s)** — matching the directly measured
0.398 s hash — while the marginal per-genome term is untouched at every thread count.

## Revised ratios, against the UNCHANGED competitor medians

CheckM2 4,286.00 s (n = 3) and CoCoPyE 2,971.68 s (n = 3) at 32 threads on Set E; CheckM2
559.66 s (n = 5) and CoCoPyE 346.67 s (n = 4) at 32 threads on 100 genomes — all from
`speed_v3/pooled_cell_summary.tsv`, not re-measured. (`updated_ratios_v033.tsv`)

| Comparison | Threads | Input | WS11.T | **v0.3.3 released** | Move |
|---|---:|---|---:|---:|---:|
| CheckM2 ÷ MAGICC | 32 | Set E, 1,000 genomes | 716.7× | **684.7×** | −4.5 % |
| CheckM2 ÷ MAGICC | 32 | 100 genomes | 253.2× | **223.9×** | −11.6 % |
| CoCoPyE ÷ MAGICC | 32 | Set E, 1,000 genomes | 496.9× | **474.7×** | −4.5 % |
| CoCoPyE ÷ MAGICC | 32 | 100 genomes | 156.9× | **138.7×** | −11.6 % |

**Peak RSS, matched thread count, released path** (`rss_ratios_by_cell_v033.tsv`):

| Cell (32 threads both sides) | MAGICC | CheckM2 | **Ratio** | CoCoPyE | Ratio |
|---|---:|---:|---:|---:|---:|
| 100 genomes, 0.478 Gbp | 0.4175 GB | 10.6102 GB | **25.4×** | 15.8878 GB | 38.1× |
| 1,000 genomes, 4.807 Gbp | 0.5152 GB | 18.8610 GB | **36.6×** | 15.9000 GB | 30.9× |

MAGICC's peak RSS rose from WS11.T's 0.4975 GB to 0.5152 GB at this cell; the verification pass
accounts for essentially none of it (verified 0.5152 vs unverified 0.5145 GB) — it is the newer
numpy/onnxruntime. The 1-thread Set E cell moved most, 0.6484 → 0.7178 GB, for the same reason.

## Parallel scaling, released path (`scaling_efficiency_v033.tsv`)

Set E 1 → 32 threads is now **7.97× (24.9 % efficiency)**, essentially unchanged from WS11.T's
8.12× / 25.4 %; on 100 genomes **2.66× (8.3 %)**. The larger fixed start-up makes scaling
marginally worse, as expected, and the sub-linear story is unaffected.

## One quotable sentence

> MAGICC v0.3.3 processes 1,000 genomes (4,806,980,767 bp) in **6.26 s** of end-to-end wall clock
> at 32 threads on one 48-core host, of which **0.52 s — 8.3 % — is the SHA-256 verification of
> the 169,658,949-byte model that the released CLI performs before every load.**

## Files

`matched_thread_runs_v033.tsv` (48 runs, per-cell host state) · `cell_summary_v033.tsv` ·
`verification_cost_by_cell.tsv` · `model_verification_cost.tsv` · `updated_ratios_v033.tsv` ·
`rss_ratios_by_cell_v033.tsv` · `fixed_variable_costs_v033.tsv` · `scaling_efficiency_v033.tsv` ·
`run_plan_v033.tsv` · `logs/campaign.log`, `logs/load_gate.tsv` · `runs/` (JSON + `time -v` +
stdout + result table per cell). Scripts `200`–`203`.
