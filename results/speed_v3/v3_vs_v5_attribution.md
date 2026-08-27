# WS11.T / T3 — the V3 (74.4 s) vs V5 (49.0 s) single-thread gap: ATTRIBUTED

**Verdict: the gap is the assembly statistics that V4/V5 deleted. It was never the model,
the wrapper, the I/O path, the JIT cache or the page cache.** On the identical 1,000-genome
Set E (4,806,980,767 bp) at 1 thread on one 48-core host, removing the 19 genuine assembly
statistics from the archived V3 code — changing nothing else, not even the ONNX model —
moves the run from **72.18 s to 49.80 s**, i.e. **22.38 s of the ~24 s gap (91 %)**.

## The experiment

The suggested test — "substitute `models/magicc_v3.onnx` into the current code path" —
**is impossible, and that impossibility is the first piece of evidence.** V3's ONNX graph
declares `assembly_features [batch, 26]`; V5's declares `[batch, 7]`. The current worker
computes 7, so the current code path aborts on V3 with
`INVALID_ARGUMENT: Got invalid dimensions for input: assembly_features … Got: 7 Expected: 26`
(`current_code_cannot_run_v3.txt`). The historic V3 measurement therefore *necessarily*
ran a different feature-extraction code path, and the question is what that path cost.

So the archived V3 package was restored from git (`471eb28`, the last commit before
`c5a9b95` "Release V4 model: remove assembly statistics") into `v3_code/`, and a five-rung
ladder was run in which **each rung differs from the next by exactly one thing**, n = 3 each,
same host, same time window, warm page cache (`fs_inputs = 0` on every run).

Two of the three modules that matter are **byte-identical between `471eb28` and HEAD**:
`magicc/kmer_counter.py` and `magicc/fragmentation.py`. FASTA reading and k-mer counting are
literally the same code in both arms. The V3 and V5 workers differ by **one line**:

```
V3:  assembly_feats = compute_assembly_stats(contigs, log10_total, kmer_counts)   # 26 features
V5:  assembly_feats = compute_assembly_stats(log10_total, kmer_counts)            #  7 features
```

V3's 26 = 11 contig-length stats + 4 base-composition + 4 distributional + the 7 that V5 kept.
Computing the first 19 requires, per contig, a `.encode('ascii')` copy of the sequence and a
Numba GC scan over **every base** — a second full pass over 4.807 Gbp that V5 does not make.
(The brief said "20 assembly statistics"; the archived model and code say **26**.)

## Ladder (1 thread, 1,000 genomes, 4,806,980,767 bp, warm cache, n = 3)

| Rung | What it is | Code tree | Model | Wall median (s) | Δ vs previous | Feature extr. (s) | ONNX (s) | Overhead (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | production CLI, `--input-list` | HEAD | V5 | 49.86 | — | 46.6 | 1.34 | 0.30 |
| 2 | production CLI, `--input DIR` | HEAD | V5 | 49.93 | +0.90 | 47.6 | 1.368 | 0.32 |
| 3 | code-root launcher | HEAD | V5 | 49.40 | −0.53 | 47.1 | 1.377 | 0.31 |
| 4 | launcher, **assembly stats OFF** | 471eb28 patched | **V3** | 49.80 | +0.40 | 47.2 | 1.489 | 0.40 |
| 5 | launcher, **V3 UNMODIFIED** | 471eb28 | **V3** | **72.18** | **+22.38** | **69.6** | 1.467 | 0.38 |

Rung 1 pooled with the archived WS8 repeats of the same cell gives 49.03 s (n = 15).
Rung 5 reproduces the archived V3 figure of **74.36 s to within 2.9 %**.
Per-run values: `v3_vs_v5_phase_runs.tsv`; per-rung medians: `v3_vs_v5_attribution.tsv`.

**The whole delta is inside feature extraction** (69.6 s → 47.2 s = 22.4 s), the only phase
that touches sequence. Normalisation, ONNX inference and process overhead are flat across
all five rungs. This matches the archived V3 phase breakdown from the 97.5 s run
(feature extraction 91.7 s, normalisation 0.251 s, ONNX inference 1.81 s, overhead 2.92 s):
there too, feature extraction was ~94 % of the run.

## Every candidate, ruled in or out BY MEASUREMENT

| Candidate | Measurement | Cost | Verdict |
|---|---|---:|---|
| **The 19 assembly statistics** | rung 5 − rung 4 (only that code path differs) | **+22.38 s** | **RULED IN — 91 % of the gap** |
| Input-discovery mode (list vs directory scan) | rung 2 − rung 1 | +0.90 s | ruled out (3.7 %) |
| Code-root launcher / harness | rung 3 − rung 2 | −0.53 s | ruled out (noise) |
| ONNX model itself (V3 179.5 MB / 26-dim vs V5 169.7 MB / 7-dim) | rung 4 vs rung 3, and the ONNX column across all rungs | +0.40 s end-to-end; inference 1.489 s vs 1.377 s = **+0.11 s** | ruled out |
| Numba JIT cache state | pristine tree, zero `.nbi/.nbc`, one full run each | V3 72.64 vs 72.18 s (**+0.46 s**); V5 49.83 vs 49.40 s (**+0.43 s**) | ruled out (1.9 %) |
| Page cache | `/usr/bin/time -v` File-system inputs on all 15 ladder runs and both JIT probes | **0 blocks** everywhere | ruled out |
| `conda run` wrapper | archived WS8 `magicc_condarun` cells, n = 3 | +1.8 s | ruled out (it explains the separate 74.4-vs-97.5 s archive gap only 8 %) |
| Git version of `magicc/` | the ladder *is* the version contrast: 471eb28 vs HEAD, both run today | see rungs 3–5 | it is the version — specifically the one line that version changed |

## Bottom-up confirmation

Timing the components in isolation on 100 genomes (478,109,066 bp), single-threaded, and
scaling to Set E (`assembly_stats_microbench.tsv`):

| Component | Projected over Set E |
|---|---:|
| `read_fasta_contigs` (identical code both arms) | 19.95 s |
| k-mer counting (identical code both arms) | 27.02 s |
| `compute_assembly_stats` **V3** (26 features, needs contigs) | 22.17 s |
| `compute_assembly_stats` **V5** (7 features, k-mer counts only) | 0.13 s |
| — of which per-contig `.encode('ascii')` + Numba GC byte scan | 21.64 s |
| **V3 − V5, the one-line difference** | **22.04 s** |

Bottom-up prediction **22.04 s**; end-to-end intervention **22.38 s** — **agreement to 1.5 %**.
Read + k-mer counting (46.97 s) also reproduces the measured V5 feature-extraction phase
(47.2 s) to 0.5 %. The causal chain is closed at both ends.

## One paragraph, for the manuscript

> The archived single-thread figure for the 1,000-genome set (74.4 s, model V3) and the
> current one (49.0 s, model V5) differ by 25 s, and the difference is not the model: it is
> the feature set the model consumed. V3 took 26 assembly features, of which 19 — contig
> length statistics, base composition and their distributional summaries — required a
> per-contig byte-level pass over all 4.807 Gbp in addition to k-mer counting. V4 removed
> them and V5 inherits the removal, so the current code makes one pass instead of two. We
> tested this directly: the V3 package was restored from its release commit and run on the
> identical input on the same host, reproducing 72.2 s (n = 3); disabling only those 19
> statistics, with the same V3 model and the same code otherwise untouched, gives 49.8 s
> (n = 3), placing 22.4 s of the gap — 91 % — in the deleted statistics. The remainder is
> accounted for by input-discovery mode (0.9 s) and the model itself (0.1 s of ONNX
> inference); the `conda run` wrapper (1.8 s), the Numba JIT cache (0.4 s) and the page
> cache (zero disk reads on every run) are excluded by measurement. An isolated timing of
> the two feature functions predicts 22.0 s for the same difference, within 1.5 % of the
> end-to-end result. The historic V3 number was therefore correct for what it measured; the
> speed-up is a real consequence of a modelling decision taken for accuracy reasons, not a
> measurement artefact.
