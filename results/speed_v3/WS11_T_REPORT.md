# WS11.T — competitor repeats, re-derived ratios, and the V3-vs-V5 attribution

**Campaign 2026-08-26, 16:41–22:03 UTC. 28 new timed cells, all rc = 0, all outputs complete
(1,000 / 100 rows as appropriate). Nothing in `results/revision/speed/` was modified.**
Every number below is traceable to a file in this directory; every run has its own JSON,
`/usr/bin/time -v` file and stdout under `runs/`.

Harness: `scripts/190`–`198`. `191_ws11t_run_one.sh` reuses the WS8 dispatch blocks verbatim
(`scripts/161`), so the new repeats are produced by exactly the harness that produced the
archived rows; only the output root and the recorded host-state fields differ. The input set
is read from `results/revision/speed/inputs/`, i.e. the same files and the same manifest
(`set_E_full` = 1,000 genomes, 4,806,980,767 bp; `set_E_100` = 100 genomes, 478,109,066 bp).
Wall clock is unchanged: process start → results file written and process exited, including
interpreter start, imports, JIT warm-up, model/database load and output write; `/usr/bin/time -v`,
no wrapper, uncompressed FASTA on local ext4, warm page cache.

**Idleness.** `/proc/loadavg` (all three figures), `nproc`, `MemAvailable` and `vmstat` are
recorded immediately before and after every cell (`matched_thread_runs_v3.tsv`). A cell starts
when the pre-run 1-minute load average is below 4.0 (MAGICC 1-thread cells: below 1.5, polled
every 60 s for up to 20 min); the decision, the load at that moment, the wait and the
instantaneous all-process %CPU are logged in `logs/load_gate.tsv`. **No run was dropped.**

**One harness defect found and fixed mid-campaign.** The first gate implementation tested the
1-minute load average with no settle period. That figure has a ~60 s decay constant, so
immediately after one of *our own* 32-thread cells it reads 20–40 on a completely idle host.
The CheckM2 100-genome pilot therefore started 4 s after a 15.9 GB CoCoPyE process exited, at
a reported load of 39.96, and ran 6.8 % slow (597.56 s against an archived median of 559.66 s).
The gate now settles first (`logs/194_campaign_v1_pilot.sh.bak` is the original). That run is
**kept and flagged**, and an extra repeat was added as required (r5 = 539.44 s); the pooled
median of 5 is 559.66 s, unmoved. It is the only WS11.T run flagged. The four other entries in
`flagged_runs.tsv` are the archived WS8 first-touch cold-cache runs, all identified by non-zero
`File system inputs`, already documented in §3.15.

---

## T1 — competitor repeats on the 1,000-genome set, 32 threads

The n = 1 weakness is removed. CheckM2 and CoCoPyE now stand on three independent runs each.

| Tool | *n* before | *n* now | Median | [min–max] | Individual runs (s) |
|---|---:|---:|---:|---|---|
| **MAGICC v0.3.0 (V5)** | 3 | **5** | **5.98 s** | 5.83–39.06 s | 5.83, 5.83, 5.98, 6.04, 39.06 |
| CheckM2 1.0.1 | 1 | **3** | **4,286.00 s** | 4,208.00–4,288.00 s | 4208.00, 4286.00, 4288.00 |
| CoCoPyE 0.5.0 | 1 | **3** | **2,971.68 s** | 2,924.29–3,177.54 s | 2924.29, 2971.68, 3177.54 |
| DeepCheck (inference only) | 3 | **5** | **29.48 s** | 29.16–39.84 s | 29.16, 29.24, 29.48, 29.67, 39.84 |

The two MAGICC/DeepCheck maxima (39.06 s, 39.84 s) are the archived WS8 first-touch cold-cache
runs, retained per §3.15. Every new run recorded `File system inputs = 0` except CheckM2 r3
(48 blocks = 24 kB) — all warm.

**The replicates are tight.** CheckM2's three runs span 1.9 % of the median; CoCoPyE's span
8.5 %, and its widest value is the *archived* run. The single-run rows were not misleading:
the archived CheckM2 value (4,208 s) sits 1.8 % below the new median, the archived CoCoPyE
value (3,177.54 s) 6.9 % above it.

Also brought up on the 100-genome cell: CheckM2 n = 3 → **5** (median 559.66 s, 539.44–597.56),
CoCoPyE n = 3 → **4** (346.67 s, 324.44–495.09), MAGICC n = 3 → **4** (2.21 s, 2.15–13.25),
DeepCheck n = 3 → **4** (4.98 s). Full per-cell table: `pooled_cell_summary.tsv`.

## T2a — re-derived headline ratios (`updated_ratios.tsv`)

| Comparison | Threads | Input (denominator) | Published | **Re-derived** | Move | Repeats (num ÷ den) |
|---|---:|---|---:|---:|---:|---:|
| CheckM2 ÷ MAGICC | 32 | Set E, 1,000 genomes, 4.807 Gbp | 697× | **717×** | +2.8 % | **3 ÷ 5** (was 1 ÷ 3) |
| CheckM2 ÷ MAGICC | 32 | 100 genomes, 0.478 Gbp | 251× | **253×** | +0.9 % | **5 ÷ 4** (was 3 ÷ 3) |
| CoCoPyE ÷ MAGICC | 32 | Set E, 1,000 genomes, 4.807 Gbp | 526× | **497×** | −5.5 % | **3 ÷ 5** (was 1 ÷ 3) |
| CoCoPyE ÷ MAGICC | 32 | 100 genomes, 0.478 Gbp | 159× | **157×** | −1.3 % | **4 ÷ 4** (was 3 ÷ 3) |

**The point estimates barely move: 697× → 717× and 251× → 253×.** Tripling the competitor
repeat count does not change the conclusion, which is the useful thing to be able to say.
The honest recommendation is to quote **~717×** and **~253×**, or to round to 700× and 250×.

**Ranges.** The extreme min–max envelope of the 1,000-genome CheckM2 ratio is 108×–736× —
but its lower end is produced entirely by dividing by MAGICC's 39.06 s cold-cache first-touch
run. Excluding that one documented cold observation (and nothing else) the envelope is
697×–736× with a median ratio of 726×. Both are tabulated in
`ratio_sensitivity_cold_start.tsv`; nothing is dropped from the primary table.

## T2b — peak RSS, per cell, unambiguously (`rss_ratios_by_cell.tsv`)

**Matched thread count on both sides, median over repeats:**

| Cell (32 threads, both tools) | MAGICC | CheckM2 | **Ratio** | CoCoPyE | Ratio | DeepCheck | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 genomes, 0.478 Gbp | 0.358 GB (n=4) | 10.610 GB (n=5) | **29.6×** | 15.888 GB (n=4) | 44.3× | 0.667 GB (n=4) | 1.9× |
| 1,000 genomes, 4.807 Gbp | 0.498 GB (n=5) | 18.861 GB (n=3) | **37.9×** | 15.900 GB (n=3) | 32.0× | 1.274 GB (n=5) | 2.6× |

**Where the published 27.7× comes from, and why it should be replaced.** It is reproduced
exactly (27.68×) as CheckM2's maximum peak RSS over **all thread counts** of the 100-genome
set (10.663 GB, measured at 32 threads) divided by MAGICC's maximum over all thread counts of
that set (0.385 GB, measured **at 1 thread**). It is therefore a **cross-thread** comparison,
which contradicts the rule stated in the same paragraph of Table S4d — "ratios are quoted only
between cells measured at the same thread count". The companion 41.2× for CoCoPyE has the same
construction (15.889 GB @ 32 thr ÷ 0.385 GB @ 1 thr). Both provenance rows are in
`rss_ratios_by_cell.tsv`, labelled `PROVENANCE (published style)`.

Table 1's implied 18.88 / 0.50 = 37.8× **is** the matched 1,000-genome 32-thread cell and is
right; the re-derived value with n = 3 CheckM2 repeats is **37.9×**.

**Recommended wording.** Quote **37.9× on the 1,000-genome set at 32 threads**, the same cell
as the 717× wall-clock ratio, so the memory and speed figures share one denominator; or quote
**29.6×** and name the 100-genome cell explicitly. Do not quote 27.7×.

## T3 — the V3-vs-V5 single-thread gap: ATTRIBUTED

**It is the assembly statistics that V4/V5 deleted, and nothing else.** Full evidence,
ladder, ruled-out candidates and a manuscript paragraph: `v3_vs_v5_attribution.md`.

Headline: on the identical 1,000 genomes at 1 thread, the archived V3 package restored from
its release commit (`471eb28`) runs in **72.18 s** (n = 3), reproducing the archived 74.36 s to
2.9 %. Disabling only the 19 genuine assembly statistics — same V3 ONNX model, same reader,
same k-mer counter, one function short-circuited — gives **49.80 s** (n = 3). That places
**22.38 s of the ~24 s gap (91 %)** in those statistics, entirely inside the feature-extraction
phase (69.6 s → 47.2 s). Ruled out by measurement: input-discovery mode (+0.90 s), the code-root
launcher (−0.53 s), the ONNX model itself (+0.11 s of inference), the Numba JIT cache (+0.46 s
from a pristine tree), the page cache (0 disk-read blocks on every run) and the `conda run`
wrapper (+1.8 s, WS8). An isolated bottom-up timing of the two feature functions predicts
**22.04 s** for the same difference — 1.5 % from the end-to-end result.

The suggested test of substituting `magicc_v3.onnx` into the current code path **cannot be run**:
V3's graph declares `assembly_features [batch, 26]` and the current worker computes 7, so
onnxruntime rejects it (`current_code_cannot_run_v3.txt`). That is itself the proof that the
historic V3 measurement ran a different feature-extraction path. Note also that V3 consumed
**26** assembly features, not the 20 stated in the brief.

**Table S4g can now be rewritten.** Its concession that the gap "cannot be attributed from the
archive" is true of the archive but no longer true of the evidence: the cause has been
established by direct intervention on a restored code tree.

---

## Files

| File | Contents |
|---|---|
| `matched_thread_runs_v3.tsv` | every new run: pre/post load average (3 figures), MemAvailable, vmstat, nproc, cache state, rc, wall clock, peak RSS, %CPU, file-system inputs, code root, model, flags |
| `pooled_cell_summary.tsv` | archived + new pooled per cell: n, median, min, max, RSS, %CPU |
| `updated_ratios.tsv` | re-derived ratios with n, medians and min–max envelopes |
| `ratio_sensitivity_cold_start.tsv` | the same ratios with and without the archived cold-cache run |
| `rss_ratios_by_cell.tsv` | matched-thread peak-RSS ratios + provenance of the published 27.7× |
| `v3_vs_v5_attribution.{tsv,md}` | the five-rung ladder, phase breakdown and verdict |
| `v3_vs_v5_phase_runs.tsv` | per-run phase timers for all 15 ladder runs |
| `assembly_stats_microbench.tsv` | bottom-up cost of the one differing line |
| `flagged_runs.tsv` | every flagged run and why (5 total, 4 of them archived) |
| `current_code_cannot_run_v3.txt` | the onnxruntime shape rejection |
| `v3_code/`, `v3_code_nostats/` | the restored V3 package (git 471eb28) and its one-line intervention |
| `run_plan_v3.tsv`, `logs/` | plan, campaign log, load-gate log, JIT warm-up log |
| `runs/` | per-cell JSON + `/usr/bin/time -v` + stdout + result table |
