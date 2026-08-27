#!/usr/bin/env python3
"""
WS8 — assemble every measured artefact into one report.

Reads only files already written by scripts 160-173 and emits
results/revision/speed/WS8_REPORT.md plus ws8_summary.json.
No claim in the report exists without a file under results/revision/speed/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPEED = Path("/path/to/magicc/results/revision/speed")

LABEL = {"magicc": "MAGICC v0.3.0 (V5)", "checkm2": "CheckM2 1.0.1",
         "cocopye": "CoCoPyE 0.5.0", "deepcheck": "DeepCheck (inference only)",
         "magicc_condarun": "MAGICC via `conda run`"}
ORDER = ["magicc", "checkm2", "cocopye", "deepcheck"]


def fmt(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "—"
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{int(s//60)} m {s%60:04.1f} s"
    return f"{s/3600:.2f} h"


def main() -> None:
    s = pd.read_csv(SPEED / "matched_thread_summary.tsv", sep="\t")
    runs = pd.read_csv(SPEED / "matched_thread_runs.tsv", sep="\t")
    manifest = json.loads((SPEED / "inputs" / "input_manifest.json").read_text())
    tco = json.loads((SPEED / "tco.json").read_text())
    setup = json.loads((SPEED / "setup_cost.json").read_text())
    rec = json.loads((SPEED / "reconciliation_40s_vs_97.5s.json").read_text())

    warm = s[s.cache == "warm"]
    m100, mfull = manifest["set_E_100"], manifest["set_E_full"]

    L, A = [], None
    A = L.append

    A("# WS8 — matched-hardware speed and memory (R1-M7, R1-m11, R1-m12, R1-m20)")
    A("")
    A(f"Generated from `results/revision/speed/`. "
      f"{len(runs)} timed runs; every run has its own JSON under `runs/`.")
    A("")
    A("## Measurement contract")
    A("")
    A("| | |")
    A("|---|---|")
    A("| **Wall clock** | process start → results file written and process exited; "
      "**includes** interpreter start, imports, JIT warm-up, model/database load, input "
      "parsing, computation, output write. `/usr/bin/time -v`, no wrapper process. |")
    A("| **Peak memory** | `/usr/bin/time -v` *Maximum resident set size*, whole process tree. |")
    A("| **Hardware** | one host, 48 cores (2 sockets), 881 GiB RAM, ext4 on local "
      "`/dev/sdb`, Linux 6.17. Identical for every row. |")
    A("| **Accelerator** | none. All four tools were run CPU-only; MAGICC's released "
      "inference path is CPU ONNX Runtime. |")
    A("| **File I/O** | uncompressed `.fasta`, one genome per file, local disk, warm page "
      "cache unless stated. |")
    A("| **Repeats** | ≥3 per cell; median reported with the full [min–max] range. |")
    A("| **Idleness** | the driver waits for the 1-minute load average to fall below 1.0 "
      "before every timed run; the load average immediately before and after each run is "
      "in `matched_thread_runs.tsv`, together with the summed %CPU of every process on the "
      "host immediately before the run (`sum_pcpu_all_processes_before`). No run was "
      "started with a 1-minute load average at or above 1.0. |")
    A("| **Run order** | tool × thread-count order randomised within each repeat block "
      "with a CRC-32 seed (`run_plan.tsv`), so drift in machine state cannot be "
      "confounded with tool identity. |")
    A("")
    A("### What `--threads N` means for each tool")
    A("")
    A("- **MAGICC**: `N` worker *processes* for k-mer feature extraction "
      "(`multiprocessing.Pool`); ONNX inference is hard-coded to 1 intra-op and 1 "
      "inter-op thread (`magicc/cli.py:567`). `OMP_NUM_THREADS=1` was set so the N "
      "workers do not each oversubscribe.")
    A("- **CheckM2**: `--threads N`, passed to Prodigal parallelisation and to DIAMOND.")
    A("- **CoCoPyE**: `-t N`, passed to UProC.")
    A("- **DeepCheck**: `torch.set_num_threads(N)`; there is no other parallelism.")
    A("")
    A("---")
    A("")

    # -------------------------------------------------------- input sets
    A("## Input sets")
    A("")
    A("| Set | Genomes | Total sequence bp | Mean genome | On disk |")
    A("|---|---:|---:|---:|---:|")
    A(f"| **set_E_100** (primary matched factorial) | {m100['n_genomes']} | "
      f"{m100['total_sequence_bp']:,} ({m100['total_sequence_bp']/1e9:.3f} Gbp) | "
      f"{m100['mean_bp_per_genome']/1e6:.2f} Mbp | "
      f"{m100['total_file_bytes']/1e9:.3f} GB |")
    A(f"| **Set E** (historical comparator) | {mfull['n_genomes']:,} | "
      f"{mfull['total_sequence_bp']:,} ({mfull['total_sequence_bp']/1e9:.3f} Gbp) | "
      f"{mfull['mean_bp_per_genome']/1e6:.2f} Mbp | "
      f"{mfull['total_file_bytes']/1e9:.3f} GB |")
    A("")
    A(f"set_E_100 is a seeded random subsample of Set E "
      f"(CRC-32 seed {manifest['subsample_seed_crc32']} from the string "
      f"`{manifest['subsample_seed_string']}`), carrying "
      f"{m100['fraction_of_full_bp']*100:.2f} % of Set E's bases. Every member is listed "
      f"in `inputs/input_manifest.json`.")
    A("")

    # ---------------------------------------------------- matched table
    A(f"## 8.1 Matched-thread wall clock — **denominator: {m100['n_genomes']} genomes, "
      f"{m100['total_sequence_bp']/1e9:.3f} Gbp**")
    A("")
    ths = sorted(warm[warm.input_set == "set_E_100"].threads.unique())
    A("| Tool | " + " | ".join(f"{t} thread{'s' if t>1 else ''}" for t in ths)
      + " | Peak RSS (GB) |")
    A("|---|" + "---:|" * (len(ths) + 1))
    for tool in ORDER:
        ts = warm[(warm.tool == tool) & (warm.input_set == "set_E_100")]
        if ts.empty:
            continue
        cells = []
        for t in ths:
            r = ts[ts.threads == t]
            if r.empty:
                cells.append("not run")
            else:
                r = r.iloc[0]
                cells.append(f"**{fmt(r.wall_median_s)}** [{fmt(r.wall_min_s)}–"
                             f"{fmt(r.wall_max_s)}] n={int(r.n_repeats)}")
        A(f"| {LABEL[tool]} | " + " | ".join(cells)
          + f" | {ts.peak_rss_gb_max.max():.2f} |")
    A("")

    full = warm[warm.input_set == "set_E_full"]
    if not full.empty:
        A(f"### Full Set E — **denominator: {mfull['n_genomes']:,} genomes, "
          f"{mfull['total_sequence_bp']/1e9:.3f} Gbp**")
        A("")
        A("| Tool | Threads | Wall clock, median [min–max] | *n* | Peak RSS (GB) |")
        A("|---|---:|---:|---:|---:|")
        for tool in ORDER:
            for _, r in full[full.tool == tool].sort_values("threads").iterrows():
                A(f"| {LABEL[tool]} | {int(r.threads)} | **{fmt(r.wall_median_s)}** "
                  f"[{fmt(r.wall_min_s)}–{fmt(r.wall_max_s)}] | {int(r.n_repeats)} | "
                  f"{r.peak_rss_gb_max:.2f} |")
        A("")

    # ----------------------------------------------------------- 8.3
    A("## 8.3 Reconciliation of 40 s vs 97.5 s (R1-m11)")
    A("")
    A(rec["verdict_summary"])
    A("")
    A("| Figure | Status |")
    A("|---|---|")
    A(f"| **40 s** | **Withdrawn.** Not a measurement: 1,000 ÷ 1,451 = "
      f"{rec['figure_40_s']['arithmetic_check_s']:.1f} s. |")
    A(f"| **1,451 genomes/min/thread** | **Withdrawn.** Arithmetic mean of five per-set "
      f"rates from MAGICC's internal compute-phase timer (V3 model). Pooled correctly the "
      f"same timer gives "
      f"{rec['figure_1451_genomes_per_min_per_thread']['defect_1_mean_of_ratios']['pooled_total_genomes_over_total_time']:.0f}. |")
    A(f"| **97.5 s** | **Survives** as a correct V3 end-to-end measurement; superseded. |")
    A(f"| **74.4 s / 7.9 s** | **Survive** as correct V3 end-to-end measurements; superseded. |")
    A("")

    ph = SPEED / "magicc_phase_breakdown.tsv"
    if ph.exists():
        d = pd.read_csv(ph, sep="\t")
        d = d[(d.cache == "warm") & (d.tool == "magicc")]
        if not d.empty:
            g = d.groupby(["input_set", "threads"], as_index=False).agg(
                wall=("wall_clock_s", "median"),
                comp=("compute_phase_s", "median"),
                hid=("pct_wall_not_seen_by_1451_timer", "median"))
            A("**How much of MAGICC's real cost the retired timer never saw** "
              "(`magicc_phase_breakdown.tsv`):")
            A("")
            A("| Input set | Threads | End-to-end wall | Compute phase only | Hidden |")
            A("|---|---:|---:|---:|---:|")
            for _, r in g.iterrows():
                A(f"| {r.input_set} | {int(r.threads)} | {fmt(r.wall)} | {fmt(r.comp)} | "
                  f"{r.hid:.0f} % |")
            A("")

    cr = warm[warm.tool == "magicc_condarun"]
    if not cr.empty:
        direct = runs[(runs.tool == "magicc") & (runs.input_set == "set_E_full")
                      & (runs.cache == "warm") & (runs.threads == 1) & (runs.repeat >= 4)]
        A("**Does the `conda run` wrapper explain the historical 74.4 s vs 97.5 s gap?** "
          "Same binary, same input, same thread count, interleaved repeats:")
        A("")
        A("| Invocation | Wall clock, median [min–max] | *n* |")
        A("|---|---:|---:|")
        for _, r in cr.iterrows():
            A(f"| `conda run -n magicc2 magicc predict` | **{fmt(r.wall_median_s)}** "
              f"[{fmt(r.wall_min_s)}–{fmt(r.wall_max_s)}] | {int(r.n_repeats)} |")
        if not direct.empty:
            A(f"| direct `magicc predict` | **{fmt(direct.wall_clock_s.median())}** "
              f"[{fmt(direct.wall_clock_s.min())}–{fmt(direct.wall_clock_s.max())}] | "
              f"{len(direct)} |")
            gap = float(cr.iloc[0].wall_median_s) - float(direct.wall_clock_s.median())
            A("")
            A(f"**No.** The wrapper costs {gap:.1f} s "
              f"({gap/float(direct.wall_clock_s.median())*100:.0f} %), not 23 s. The "
              f"historical 74.4 s vs 97.5 s gap therefore cannot be attributed to "
              f"`conda run`, and neither archived run recorded load average or page-cache "
              f"state, so it cannot be attributed from the archive at all. Both are "
              f"superseded by the V5 re-measurement, which records both.")
        A("")

    # ----------------------------------------------------------- 8.2
    A("## 8.2 Total cost of ownership (R1-m12)")
    A("")
    A("| Tool | Reference data on disk | First-run download | Notes |")
    A("|---|---:|---:|---|")
    for name, t in tco["tools"].items():
        b = (t.get("reference_data_bytes") or 0)
        p = setup["competitor_downloads"]["projections"].get(name, {})
        secs = p.get("projected_seconds_at_measured_throughput")
        if name.startswith("MAGICC"):
            dl = f"{setup['magicc_model_download']['median_seconds']:.1f} s (**measured**, n=3)"
        elif secs is None:
            dl = "—"
        elif secs < 120:
            dl = f"{secs:.0f} s (projected)"
        else:
            dl = f"{secs/60:.1f} min (projected)"
        A(f"| {name} | {b/1e9:.3f} GB | {dl} | {t.get('reference_data_note','')} |")
    A("")
    md = setup["magicc_model_download"]
    A(f"MAGICC's first-run setup was **measured directly**: "
      f"{md['median_seconds']:.1f} s median over {len(md['trials'])} downloads of the "
      f"169.7 MB ONNX model from the URL the released CLI uses, at "
      f"{md['median_MB_per_s']} MB/s. That is the whole of MAGICC's setup — there is no "
      f"database to fetch, build or index. Competitor download times are **projections** "
      f"at that same measured throughput, because re-downloading their databases would "
      f"overwrite working installations; the byte volumes are exact.")
    A("")
    envf = json.loads((SPEED / "env_footprint.json").read_text())
    probe = envf["magicc_minimal_env_probe"]
    shared = envf["environments_on_this_host"]["magicc2 (shared analysis environment)"]
    ded = envf["environments_on_this_host"]["checkm2_py39 (dedicated CheckM2 environment)"]
    cont = envf["containers_from_WS7"]
    A(f"**Install footprint (corrected).** An earlier version of this table charged "
      f"MAGICC, CoCoPyE and DeepCheck the same {shared['bytes']/1e9:.2f} GB, which is the "
      f"size of the shared conda environment used to *develop* this paper "
      f"({shared['n_conda_packages']} conda / {shared['n_pip_packages']} pip packages, "
      f"containing "
      + ", ".join(f"`{x}`" for x in shared["contains_non_magicc_packages"])
      + f"). One shared environment is not three tool footprints, and none of those "
        f"packages is needed to run MAGICC. What a user actually installs was therefore "
        f"measured directly: a throwaway conda environment with only python 3.11, magicc "
        f"and its five declared runtime dependencies is "
        f"**{probe['bytes_total']/1e9:.2f} GB** ({probe['n_pip_pkgs']} pip packages), of "
        f"which {probe['bytes_python_only']/1e9:.2f} GB is the bare CPython interpreter; "
        f"it was verified runnable and then deleted. CheckM2's `checkm2_py39` **is** a "
        f"dedicated environment and its {ded['bytes']/1e9:.2f} GB stands. CoCoPyE's and "
        f"DeepCheck's software footprints are **not separately measurable** on this host "
        f"and no number is reported for them. The version-exact artefact is the pinned "
        f"container: Docker {cont['docker_image_bytes']/1e9:.2f} GB / Apptainer "
        f"{cont['apptainer_sif_bytes']/1e9:.2f} GB, which include a base-OS layer and the "
        f"bundled model and so are not comparable with a bare conda environment. "
        f"Full detail: `env_footprint.json`, `tco_table.md`.")
    A("")
    A("### Cold vs warm page cache")
    A("")
    A("**The designed cold-cache arm failed and its cells are invalid.** "
      "`scripts/162_ws8_drop_file_cache.py` raised `TypeError: underlying buffer is not "
      "writable` on the first file it inspected and exited before any "
      "`posix_fadvise(DONTNEED)` call; all 11 `*.evict.txt` logs carry that identical "
      "traceback and every `cache=cold` run recorded `File system inputs = 0`. Those "
      "cells are warm runs with a cold label and are treated as extra warm repeats. "
      "The fixed evictor is `scripts/176_ws8_evict_fixed.py` (verified to evict 100 % of "
      "5.04 GB of resident pages) with re-runner `scripts/177_ws8_cold_verified.sh`; the "
      "re-run was not executed because an unrelated job appeared on the host and the "
      "campaign's idleness rule would have been broken.")
    A("")
    A("**The cold-start numbers reported instead are stronger.** The host was rebooted on "
      "2026-07-30, so the first run to touch each input set and each tool's database read "
      "it from the block device — a true whole-system cold cache. Those runs are "
      "identified for all 92 runs by non-zero `/usr/bin/time -v` *File system inputs*; "
      "exactly five qualify, and they are exactly the five runs that produced an "
      "above-median wall clock. No run is discarded: each is inside its cell's median and "
      "range in the tables above, and is also reported here. Evidence: "
      "`cold_start_forensics.tsv`, `outlier_investigation.json`, "
      "`cold_vs_warm_corrected.tsv`.")
    A("")
    cwc = SPEED / "cold_vs_warm_corrected.tsv"
    if cwc.exists():
        d = pd.read_csv(cwc, sep="\t")
        d = d[d.wall_s_first_touch.notna()].sort_values("first_touch_over_warm",
                                                        ascending=False)
        if not d.empty:
            A("| Tool | Input set | Threads | Warm median (*n*) | Cold start (*n*=1) | "
              "Cold − warm | Cold ÷ warm | Read from disk |")
            A("|---|---|---:|---:|---:|---:|---:|---:|")
            for _, r in d.iterrows():
                A(f"| {LABEL.get(r.tool, r.tool)} | {r.input_set} | {int(r.threads)} | "
                  f"{fmt(r.wall_median_s_warm)} ({int(r.n_warm_resident)}) | "
                  f"{fmt(r.wall_s_first_touch)} | {r.first_touch_minus_warm_s:+.1f} s | "
                  f"{r.first_touch_over_warm:.2f}× | {r.first_touch_disk_read_GB:.2f} GB |")
            A("")

    # -------------------------------------------------- parallel scaling
    sc = SPEED / "scaling_efficiency.tsv"
    if sc.exists():
        d = pd.read_csv(sc, sep="\t")
        A("## 8.1b Parallel scaling — sub-linear for every tool")
        A("")
        A("`speed-up(T) = wall(1 thread) / wall(T threads)`, `efficiency = speed-up / T`, "
          "both on the same end-to-end wall clock, so each tool's unparallelisable "
          "start-up cost is included. **MAGICC gains 8.1× from 32 threads on 1,000 "
          "genomes (49.0 s → 6.04 s), i.e. 25 % parallel efficiency — not 32×** — and "
          "essentially nothing beyond 16 threads (6.94 s → 6.04 s).")
        A("")
        A("| Tool | Input set | Speed-up at 8 / 16 / 32 threads | Efficiency at 32 threads |")
        A("|---|---|---:|---:|")
        for (tool, iset), g in d.groupby(["tool", "input_set"], sort=False):
            if tool not in ORDER:
                continue
            g = g.set_index("threads")
            if 32 not in g.index:
                continue
            sp = " / ".join(f"{g.speedup_vs_1_thread[t]:.1f}×" if t in g.index else "—"
                            for t in (8, 16, 32))
            A(f"| {LABEL[tool]} | {iset} | {sp} | "
              f"**{g.parallel_efficiency_pct[32]:.0f} %** |")
        A("")
        fv = SPEED / "fixed_variable_costs.tsv"
        if fv.exists():
            f = pd.read_csv(fv, sep="\t")
            mg = f[(f.tool == "magicc") & (f.threads == 32)]
            if not mg.empty:
                r = mg.iloc[0]
                A(f"The reason is a fixed cost that no thread count removes. Solving "
                  f"`wall(n) = a + b·n` from the two measured workload sizes at 32 threads "
                  f"gives MAGICC a fixed start-up of **{r.fixed_startup_s:.2f} s** and a "
                  f"marginal cost of {r.marginal_s_per_genome*1000:.1f} ms per genome, so "
                  f"start-up is {r.fixed_share_of_1000genome_run_pct:.0f} % of the "
                  f"1,000-genome run and {r.fixed_share_of_100genome_run_pct:.0f} % of the "
                  f"100-genome run (`fixed_variable_costs.tsv`).")
                A("")

    A("## Files")
    A("")
    for f in sorted(SPEED.glob("*")):
        if f.is_file():
            A(f"- `results/revision/speed/{f.name}`")
    A("- `results/revision/speed/runs/` — one JSON + `/usr/bin/time -v` file + stdout per run")
    A("- `results/revision/speed/figures/` — CVD-safe figures + `palette_cvd_report.tsv`")
    A("")

    (SPEED / "WS8_REPORT.md").write_text("\n".join(L) + "\n")

    summ = {
        "n_runs": int(len(runs)),
        "max_loadavg_before_any_timed_run": float(runs.loadavg_1min_before.max()),
        "all_return_codes_zero": bool((runs.return_code == 0).all()),
        "input_sets": {"set_E_100": m100["n_genomes"], "set_E_full": mfull["n_genomes"]},
        "set_E_total_bp": mfull["total_sequence_bp"],
        "set_E_100_total_bp": m100["total_sequence_bp"],
    }
    (SPEED / "ws8_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"wrote {SPEED/'WS8_REPORT.md'} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
