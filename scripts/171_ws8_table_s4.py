#!/usr/bin/env python3
"""
WS8.4 (R1-m20, R1-M5) — rewrite Table S4.

Rules enforced here, mechanically:
  1. Every number in the table is an end-to-end wall clock under ONE definition.
  2. A derived ratio is emitted ONLY if hardware, thread count, file I/O mode and
     the wall-clock definition are all stated in the table or its caption; every
     ratio row therefore carries its own thread count and input set.
  3. Every caption states the denominator (genome count and total bp).
  4. Nothing is reported that was not measured; cells that were not run say so.

Writes results/revision/speed/table_S4_rewritten.md and table_S4_rewritten.tsv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/path/to/magicc")
SPEED = PROJECT / "results" / "revision" / "speed"

LABEL = {"magicc": "**MAGICC v0.3.0 (model V5)**", "checkm2": "CheckM2 1.0.1",
         "cocopye": "CoCoPyE 0.5.0", "deepcheck": "DeepCheck (inference only)"}
ORDER = ["magicc", "checkm2", "cocopye", "deepcheck"]


def fmt(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "not run"
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{int(s//60)} m {s%60:04.1f} s"
    return f"{s/3600:.2f} h"


def main() -> None:
    s = pd.read_csv(SPEED / "matched_thread_summary.tsv", sep="\t")
    manifest = json.loads((SPEED / "inputs" / "input_manifest.json").read_text())
    tco = json.loads((SPEED / "tco.json").read_text())
    rec = json.loads((SPEED / "reconciliation_40s_vs_97.5s.json").read_text())

    m100, mfull = manifest["set_E_100"], manifest["set_E_full"]
    warm = s[s.cache == "warm"]

    def cell(tool, iset, th, col="wall_median_s"):
        r = warm[(warm.tool == tool) & (warm.input_set == iset) & (warm.threads == th)]
        return None if r.empty else float(r.iloc[0][col])

    def nrep(tool, iset, th):
        r = warm[(warm.tool == tool) & (warm.input_set == iset) & (warm.threads == th)]
        return 0 if r.empty else int(r.iloc[0].n_repeats)

    def rng(tool, iset, th):
        r = warm[(warm.tool == tool) & (warm.input_set == iset) & (warm.threads == th)]
        if r.empty:
            return ""
        r = r.iloc[0]
        return f"{fmt(r.wall_min_s)}–{fmt(r.wall_max_s)}"

    L = []
    A = L.append

    A("# Table S4 (rewritten) — computational cost under a single, stated definition")
    A("")
    A("## Definitions used everywhere in this table")
    A("")
    A("**Wall clock** = elapsed real time from process start to the results file being "
      "written and the process exiting. It **includes** interpreter start-up, library "
      "imports, JIT warm-up, model and database loading, input parsing, computation and "
      "output writing. It is measured with `/usr/bin/time -v` on the tool's own top-level "
      "command, with no wrapper process. This is the only timing definition used; the "
      "submitted version of this table mixed this definition with tool-internal "
      "compute-phase timers (see Table S4g).")
    A("")
    A("**Hardware (identical for every row):** single host, 2 sockets / 48 physical cores, "
      "881 GiB RAM, ext4 on a local SATA disk (`/dev/sdb`). Linux 6.17. "
      "**No GPU was used by any tool**: MAGICC's released inference path is CPU ONNX "
      "Runtime, DeepCheck was run on CPU PyTorch, and CheckM2 and CoCoPyE are CPU-only.")
    A("")
    A("**File I/O mode (identical for every row):** uncompressed `.fasta`, one genome per "
      "file, read from a directory on local disk; **warm page cache** (cold-cache results "
      "are Table S4e). Outputs written to the same filesystem.")
    A("")
    A("**Idleness:** the host was otherwise idle. The campaign driver waited for the "
      "1-minute load average to fall below 1.0 before each timed run, and the load average "
      "immediately before and after every run is recorded in "
      "`results/revision/speed/matched_thread_runs.tsv`.")
    A("")
    A("**Repeats:** every cell is the **median** of *n* independent repeats with the full "
      "**[min–max]** range given. Tool × thread-count order was randomised within each "
      "repeat block (`results/revision/speed/run_plan.tsv`). **No run has been removed "
      "from any cell.** Where the range is wide, the reason is given below the table.")
    A("")
    A("**Page-cache state, and the five wide ranges in this table.** The host was rebooted "
      "on 2026-07-30, so the campaign began with an empty page cache. Five runs — always "
      "the first run that touched a given input set or a given tool's database — therefore "
      "read that data from the block device while every later repeat read it from RAM. "
      "They are identified objectively, for all 92 runs, by `/usr/bin/time -v`'s "
      "**File system inputs** counter (`ru_inblock`, 512-byte blocks): a warm run reads "
      "0 bytes. Exactly five runs read more than 50 MB, and they are exactly the five "
      "runs that produced an above-median wall clock; no run with zero disk reads is an "
      "outlier. They are **kept in the table** (they are inside every median and every "
      "range below) and are also reported separately as the cold-start measurement in "
      "Table S4e. Per-run evidence: "
      "`results/revision/speed/cold_start_forensics.tsv` and "
      "`results/revision/speed/outlier_investigation.json`.")
    A("")
    A("---")
    A("")

    # -------------------------------------------------------------- S4a
    A(f"## Table S4a — matched-thread wall clock, all four tools, identical input")
    A("")
    A(f"**Denominator: {m100['n_genomes']} genomes, "
      f"{m100['total_sequence_bp']:,} bp ({m100['total_sequence_bp']/1e9:.3f} Gbp), "
      f"mean {m100['mean_bp_per_genome']/1e6:.2f} Mbp per genome.** This is a seeded "
      f"({m100.get('fraction_of_full_bp', 0.0995)*100:.1f}% of Set E bp) random subsample "
      f"of Set E, drawn with a CRC-32-seeded generator "
      f"(`results/revision/speed/inputs/input_manifest.json` lists every member). "
      f"A 100-genome subsample is used for the complete factorial because CheckM2 at "
      f"1 thread on the full 1,000-genome Set E costs an estimated "
      f"{(cell('checkm2','set_E_100',1) or 0)*10/3600:.0f} h per repeat "
      f"(~{(cell('checkm2','set_E_100',1) or 0)*30/3600:.0f} h for three), which is not "
      f"practical; the full-set numbers that were measured are Table S4b.")
    A("")
    hdr = "| Tool | Threads | Wall clock, median | Range [min–max] | *n* | Peak RSS (GB) | Genomes/min |"
    A(hdr)
    A("|---|---:|---:|---:|---:|---:|---:|")
    for tool in ORDER:
        for th in [1, 8, 16, 32]:
            v = cell(tool, "set_E_100", th)
            if v is None:
                A(f"| {LABEL[tool]} | {th} | not run | | 0 | | |")
                continue
            r = warm[(warm.tool == tool) & (warm.input_set == "set_E_100")
                     & (warm.threads == th)].iloc[0]
            A(f"| {LABEL[tool]} | {th} | **{fmt(v)}** | {rng(tool,'set_E_100',th)} | "
              f"{int(r.n_repeats)} | {r.peak_rss_gb_max:.2f} | {r.genomes_per_min:.1f} |")
    A("")
    A("DeepCheck cannot read FASTA. Its row is **inference only**, starting from CheckM2's "
      "pickled feature vectors; producing those vectors requires a complete CheckM2 run, "
      "so DeepCheck's end-to-end cost is its own row **plus** the CheckM2 row. This was "
      "footnoted in the submitted table and is restated here as a table row rather than a "
      "footnote because it changes the ranking.")
    A("")

    # -------------------------------------------------------------- S4b
    A(f"## Table S4b — full Set E, the historical comparator")
    A("")
    A(f"**Denominator: {mfull['n_genomes']:,} genomes, "
      f"{mfull['total_sequence_bp']:,} bp ({mfull['total_sequence_bp']/1e9:.3f} Gbp), "
      f"mean {mfull['mean_bp_per_genome']/1e6:.2f} Mbp per genome.** Same hardware, same "
      f"wall-clock definition, same file I/O mode as Table S4a.")
    A("")
    A("| Tool | Threads | Wall clock, median | Range [min–max] | *n* | Peak RSS (GB) | Genomes/min |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for tool in ORDER:
        for th in [1, 8, 16, 32]:
            v = cell(tool, "set_E_full", th)
            if v is None:
                continue
            r = warm[(warm.tool == tool) & (warm.input_set == "set_E_full")
                     & (warm.threads == th)].iloc[0]
            A(f"| {LABEL[tool]} | {th} | **{fmt(v)}** | {rng(tool,'set_E_full',th)} | "
              f"{int(r.n_repeats)} | {r.peak_rss_gb_max:.2f} | {r.genomes_per_min:.1f} |")
    A("")
    A("The MAGICC 32-thread range is wide (5.83 s – 39.06 s over 3 repeats) for the reason "
      "given above: the 39.06 s repeat was the third timed run of the whole campaign and "
      "fetched 4.388 GB — 90.1% of this input set — from disk, while both other repeats "
      "fetched 0 bytes. Its CPU utilisation was 341% against 1683% and 1752%, and MAGICC's "
      "own phase timers put the entire excess in feature extraction (36.7 s vs 3.5 s), the "
      "only phase that reads FASTA, with normalisation and ONNX inference unchanged. The "
      "1-minute load average before that run was 0.83, *lower* than the two fast repeats, "
      "so contention is excluded. It is a cold-cache measurement, and it is reported as "
      "one in Table S4e.")
    A("")

    # ------------------------------------------------------------- scaling
    sc = SPEED / "scaling_efficiency.tsv"
    if sc.exists():
        d = pd.read_csv(sc, sep="\t")
        A("## Table S4c — parallel scaling, stated as speed-up and efficiency")
        A("")
        A("`speed-up(T) = wall(1 thread) / wall(T threads)`; "
          "`efficiency(T) = speed-up(T) / T`. Both use the same end-to-end wall clock as "
          "the rest of the table, so each tool's fixed start-up cost is inside the "
          "numerator and the denominator — which is what a user actually experiences, and "
          "which is why efficiency falls with thread count for **every** tool. "
          "**MAGICC's scaling is sub-linear:** on Set E it goes from 49.0 s at 1 thread to "
          "6.04 s at 32 threads, a **8.1× speed-up at 32 threads, i.e. 25% parallel "
          "efficiency, not 32×.** Beyond 16 threads it gains almost nothing (6.94 s → "
          "6.04 s) because the run is by then dominated by process start-up, ONNX model "
          "load and inference rather than by per-genome feature extraction.")
        A("")
        A("| Tool | Input set (denominator) | 1 thr | 8 thr | 16 thr | 32 thr | Speed-up at 32 thr | Efficiency at 32 thr |")
        A("|---|---|---:|---:|---:|---:|---:|---:|")
        for (tool, iset), g in d.groupby(["tool", "input_set"], sort=False):
            if tool not in ORDER:
                continue
            g = g.set_index("threads")
            if 1 not in g.index:
                continue
            n = int(g.n_genomes.iloc[0])
            cells = [fmt(float(g.wall_median_s[t])) if t in g.index else "not run"
                     for t in (1, 8, 16, 32)]
            if 32 in g.index:
                sp, ef = float(g.speedup_vs_1_thread[32]), float(g.parallel_efficiency_pct[32])
                tail = f"**{sp:.1f}×** | **{ef:.0f}%**"
            else:
                tail = "not run | not run"
            A(f"| {LABEL[tool]} | {iset} ({n:,} genomes) | " + " | ".join(cells) + f" | {tail} |")
        A("")
        A("On the 100-genome subsample MAGICC's efficiency at 32 threads is only 8%, and "
          "CoCoPyE is **slower** at 32 threads (353.9 s) than at 16 (216.1 s). Both are "
          "consequences of a short run: the fixed cost cannot be parallelised, and for "
          "CoCoPyE oversubscription actively hurts. This is reported because it bounds how "
          "far any of these tools, MAGICC included, benefits from more cores.")
        A("")

    # -------------------------------------------------------------- ratios
    A("## Table S4d — derived ratios, fully qualified (R1-m20)")
    A("")
    A("A speed ratio is meaningless without the conditions under which it was obtained, so "
      "each ratio below carries them explicitly. **Ratios are quoted only between cells "
      "measured at the same thread count, on the same input set, on the same hardware, "
      "under the same wall-clock definition, in the same file I/O mode.** No "
      "\"per-thread\" ratio is quoted, because per-thread throughput is not a property of "
      "a tool: it falls as thread count rises (see the MAGICC rows of Table S4a). "
      "The last column gives the number of repeats behind each side of the ratio; the "
      "two full-Set-E competitor rows rest on a single competitor run each, because one "
      "repeat of CheckM2 on 1,000 genomes costs 70 minutes even at 32 threads.")
    A("")
    A("| Comparison | Threads | Input set (denominator) | Wall-clock ratio | Repeats (num ÷ den) |")
    A("|---|---:|---|---:|---:|")
    for th in [1, 8, 16, 32]:
        mg = cell("magicc", "set_E_100", th)
        if mg is None:
            continue
        for other in ["checkm2", "cocopye"]:
            ov = cell(other, "set_E_100", th)
            if ov is None:
                continue
            A(f"| {LABEL[other].strip('*')} ÷ MAGICC | {th} | set_E_100 "
              f"({m100['n_genomes']} genomes, {m100['total_sequence_bp']/1e9:.3f} Gbp) | "
              f"**{ov/mg:.0f}×** | {nrep(other,'set_E_100',th)} ÷ {nrep('magicc','set_E_100',th)} |")
    mgf = cell("magicc", "set_E_full", 32)
    for other in ["checkm2", "cocopye"]:
        ov = cell(other, "set_E_full", 32)
        if ov is not None and mgf:
            A(f"| {LABEL[other].strip('*')} ÷ MAGICC | 32 | Set E "
              f"({mfull['n_genomes']:,} genomes, {mfull['total_sequence_bp']/1e9:.3f} Gbp) | "
              f"**{ov/mgf:.0f}×** | {nrep(other,'set_E_full',32)} ÷ {nrep('magicc','set_E_full',32)} |")
    A("")
    A("**Memory ratio**, same qualification (peak RSS, `/usr/bin/time -v`, "
      f"set_E_100, {m100['n_genomes']} genomes):")
    A("")
    A("| Comparison | Peak-RSS ratio |")
    A("|---|---:|")
    mrss = warm[(warm.tool == "magicc") & (warm.input_set == "set_E_100")].peak_rss_gb_max.max()
    for other in ["checkm2", "cocopye", "deepcheck"]:
        o = warm[(warm.tool == other) & (warm.input_set == "set_E_100")].peak_rss_gb_max.max()
        if not np.isnan(o) and mrss:
            A(f"| {LABEL[other].strip('*')} ÷ MAGICC | **{o/mrss:.1f}×** |")
    A("")

    # ------------------------------------------------------------ cold/warm
    A("## Table S4e — cold versus warm page cache (R1-m12)")
    A("")
    A("**A correction, stated first.** The campaign included a designed cold-cache arm "
      "using an unprivileged, file-scoped eviction "
      "(`posix_fadvise(POSIX_FADV_DONTNEED)` on every input FASTA and on the tool's "
      "model/database files, since `/proc/sys/vm/drop_caches` needs root and passwordless "
      "sudo is not configured on this host). **That arm did not work.** The evictor "
      "(`scripts/162_ws8_drop_file_cache.py`) raised `TypeError: underlying buffer is not "
      "writable` on the first file it inspected — it built a read-only mmap and then asked "
      "`ctypes` for a writable pointer into it — and because only `OSError` was caught, the "
      "process exited before a single `posix_fadvise` call was made. All 11 `*.evict.txt` "
      "logs contain that identical traceback, and every run labelled `cache=cold` recorded "
      "`File system inputs = 0`, i.e. it read nothing from disk. **The cells labelled "
      "\"cold\" were warm runs**, which is exactly why they agree with the warm cells to "
      "within 3% (MAGICC, Set E, 1 thread: 49.07 s \"cold\" vs 49.02 s warm). They are "
      "reported here as what they are — additional warm repeats — and are not presented as "
      "cold-cache results. The fixed evictor is `scripts/176_ws8_evict_fixed.py`, verified "
      "to evict 100% of 5.04 GB of resident pages, with a re-runner "
      "(`scripts/177_ws8_cold_verified.sh`). That re-run has **not** been executed: an "
      "unrelated single-core job appeared on the host, and running it under contention "
      "would have broken the campaign\'s own idleness rule. It is not needed for the "
      "numbers below, which come from a stricter cold condition.")
    A("")
    A("**What the cold-start numbers below actually are, and why they are stronger than the "
      "arm that failed.** The host was rebooted on 2026-07-30 and the campaign started with "
      "a genuinely empty page cache, so the first run to touch each input set and each "
      "tool's database is a **true whole-system cold start** — colder than the file-scoped "
      "eviction that was intended, because kernel, libc and conda shared objects were cold "
      "too. Those runs are identified objectively by non-zero `File system inputs` "
      "(`ru_inblock` × 512 B) and each is a single observation (*n* = 1); the warm figure "
      "beside it is the median of the remaining repeats of the same cell, on the same "
      "hardware, at the same thread count, in the same file-I/O mode.")
    A("")
    cwc = SPEED / "cold_vs_warm_corrected.tsv"
    if cwc.exists():
        d = pd.read_csv(cwc, sep="\t")
        d = d[d.wall_s_first_touch.notna()]
        if not d.empty:
            A("| Tool | Input set (denominator) | Threads | Warm (median, *n*) | Cold start (*n* = 1) | Cold − warm | Cold ÷ warm | Read from disk |")
            A("|---|---|---:|---:|---:|---:|---:|---:|")
            for _, r in d.sort_values("first_touch_over_warm", ascending=False).iterrows():
                A(f"| {LABEL.get(r.tool, r.tool).strip('*')} | {r.input_set} | "
                  f"{int(r.threads)} | {fmt(r.wall_median_s_warm)} (n={int(r.n_warm_resident)}) | "
                  f"{fmt(r.wall_s_first_touch)} | {r.first_touch_minus_warm_s:+.1f} s | "
                  f"{r.first_touch_over_warm:.2f}× | {r.first_touch_disk_read_GB:.2f} GB |")
        A("")
        A("**Reading of this table.** A cold cache costs MAGICC +33 s on 1,000 genomes at "
          "32 threads — it is the tool whose warm run is short enough for I/O to dominate, "
          "so *proportionally* it is hit hardest (6.7×). Stated precisely rather than "
          "favourably: MAGICC's worst measured cold start on 1,000 genomes at 32 threads "
          "(39.1 s) is still 81× faster than CoCoPyE's **warm** run (3,177.5 s) and 108× "
          "faster than CheckM2's **warm** run (4,208.0 s) on the same input — but it is "
          "**slower than DeepCheck's warm inference-only run** (29.5 s). That DeepCheck "
          "figure excludes the mandatory CheckM2 run (4,208.0 s at the same thread count) "
          "that produces its input, so DeepCheck's true end-to-end cost is still the "
          "larger; the comparison is stated this way because the inference-only row on its "
          "own does beat a cold MAGICC. In the other direction, CoCoPyE pays +141 s on only "
          "100 genomes purely to page in its 17.0 GB database. A cold cache is a "
          "once-per-boot cost for every tool here; in production all of them run warm after "
          "the first invocation.")
    A("")

    # ---------------------------------------------------------------- TCO
    A("## Table S4f — total cost of ownership (R1-m12)")
    A("")
    tco_md = SPEED / "tco_table.md"
    if tco_md.exists():
        body = tco_md.read_text().split("\n")
        A("\n".join(l for l in body if not l.startswith("# Table S4e")).strip())
    A("")

    # ------------------------------------------------------- reconciliation
    A("## Table S4g — reconciliation of the previously published timings (R1-m11)")
    A("")
    A("| Figure | Where it appeared | What it actually measured | Status |")
    A("|---|---|---|---|")
    A(f"| **40 s** for 1,000 genomes | Main text | **Not a measurement.** "
      f"1,000 ÷ 1,451 genomes min⁻¹ thread⁻¹ = "
      f"{rec['figure_40_s']['arithmetic_check_s']:.1f} s, rounded to \"40 seconds\". "
      f"No run of that duration exists in any log in the repository. | **Withdrawn** |")
    A(f"| **1,451** genomes min⁻¹ thread⁻¹ | Table S4, Table 1, Abstract | The arithmetic "
      f"**mean of five per-benchmark-set rates**, each computed from MAGICC's internal "
      f"compute-phase timer (feature extraction + ONNX inference only — excluding "
      f"interpreter start, imports, JIT warm-up, model load, normalisation and output "
      f"write), with magicc_v3.onnx at 1 thread. Averaging rates over sets of unequal "
      f"per-genome cost inflates the value: pooled over the same five sets the same timer "
      f"gives **{rec['figure_1451_genomes_per_min_per_thread']['defect_1_mean_of_ratios']['pooled_total_genomes_over_total_time']:.0f}** "
      f"genomes min⁻¹ thread⁻¹, a factor of "
      f"{rec['figure_1451_genomes_per_min_per_thread']['defect_1_mean_of_ratios']['inflation_factor']:.2f}. "
      f"| **Withdrawn** |")
    A(f"| **97.5 s** for 1,000 genomes | Table S4 | A genuine **end-to-end** wall clock "
      f"(`/usr/bin/time -v`), 1 thread, Set E, magicc_v3.onnx, invoked through "
      f"`conda run`. | **Correct as a V3 measurement**; superseded by the V5 "
      f"re-measurement above |")
    A(f"| **74.4 s / 7.9 s** | `results/phase7_set_e_test/` | End-to-end wall clock of "
      f"`python -m magicc predict` at 1 and 43 threads, Set E, magicc_v3.onnx, **without** "
      f"the `conda run` wrapper. The 23 s difference from the 97.5 s figure cannot be "
      f"attributed from the archive: neither run recorded load average or page-cache "
      f"state. | **Correct as V3 measurements**; superseded |")
    A("")
    A("**The internal inconsistency the reviewer detected is real and is in the table "
      "itself.** The submitted Table S4 row reads *MAGICC | 1 thread | 0.66 GB | 97.5 s | "
      "1,451 genomes/min/thread*, but 1,000 genomes in 97.5 s is 615 genomes min⁻¹ "
      "thread⁻¹, not 1,451 — a factor of "
      f"{rec['table_S4_internal_inconsistency']['ratio']:.2f}. The wall-clock column and the "
      "throughput column came from two different experiments with two different timers. "
      "The same defect affects the competitor rows: CheckM2's row states 32 threads and "
      "86 min 37 s, which is 0.36 genomes min⁻¹ thread⁻¹, while the table prints 0.82. "
      "**The published 1,700× and 2,100× ratios therefore divide one tool's mean-of-rates "
      "by another tool's mean-of-rates, and neither is the wall clock printed beside it. "
      "They are withdrawn and replaced by Table S4d.**")
    A("")

    (SPEED / "table_S4_rewritten.md").write_text("\n".join(L) + "\n")
    warm.round(4).to_csv(SPEED / "table_S4_rewritten.tsv", sep="\t", index=False)
    print(f"wrote {SPEED/'table_S4_rewritten.md'}")


if __name__ == "__main__":
    main()
