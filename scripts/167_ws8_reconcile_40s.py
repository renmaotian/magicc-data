#!/usr/bin/env python3
"""
WS8.3 (R1-m11) — reconcile the conflicting MAGICC timings for 1,000 genomes.

This script does NOT re-run anything. It reconstructs, from the archived code
and logs still present in the repository, exactly what each published or
recorded figure measured, and reproduces the arithmetic that produced the
derived ones. Fresh measurements under controlled conditions are made by
scripts/161/166 and merged by scripts/168.

Figures in the record:
   97.5 s   supplementary Table S4 "Wall-clock time (Set E)"
   1,451    supplementary Table S4 "Genomes/min/thread", also the abstract
     40 s   main text "Processing 1,000 genomes would only take 40 seconds"
   74.4 s   scripts/40_test_cli_set_e.py, 1 thread
    7.9 s   scripts/40_test_cli_set_e.py, 43 threads

Writes results/revision/speed/reconciliation_40s_vs_97.5s.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROJECT = Path("/path/to/magicc")
OUT = PROJECT / "results" / "revision" / "speed"
BENCH = PROJECT / "data" / "benchmarks"
SETS = ["A_v2", "B_v2", "C", "D", "E"]


def per_set_internal_timers() -> pd.DataFrame:
    """The `wall_clock_s` column stored in each set's magicc_predictions.tsv.

    scripts/26_benchmark_run_magicc.py:180 stores
        total_time = feat_time + infer_time
    i.e. feature extraction + ONNX inference ONLY. It therefore excludes
    interpreter start, imports, Numba JIT warm-up, ONNX model load,
    normalization (computed between the two timers) and output write.
    """
    rows = []
    for s in SETS:
        f = BENCH / f"set_{s}" / "magicc_predictions.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        w = float(df["wall_clock_s"].iloc[0])
        t = float(df["n_threads"].iloc[0])
        rows.append({
            "set": s,
            "n_genomes": len(df),
            "internal_timer_s": w,
            "threads": t,
            "genomes_per_min_per_thread": len(df) / w * 60 / t,
        })
    return pd.DataFrame(rows)


def parse_time_v(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    txt = path.read_text()
    m = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", txt)
    if m:
        p = [float(x) for x in m.group(1).split(":")]
        out["wall_clock_s"] = (p[0] * 60 + p[1]) if len(p) == 2 else \
            (p[0] * 3600 + p[1] * 60 + p[2]) if len(p) == 3 else p[0]
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", txt)
    if m:
        out["peak_rss_kb"] = int(m.group(1))
        out["peak_rss_gb"] = round(int(m.group(1)) / 1024 / 1024, 4)
    m = re.search(r'Command being timed: "(.*)"', txt)
    if m:
        out["command"] = m.group(1)
    m = re.search(r"Percent of CPU this job got:\s*(\d+)%", txt)
    if m:
        out["pct_cpu"] = int(m.group(1))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1,451
    per_set = per_set_internal_timers()
    mean_of_rates = float(per_set["genomes_per_min_per_thread"].mean())
    pooled = float(per_set["n_genomes"].sum() / per_set["internal_timer_s"].sum() * 60)
    derived_40s = 1000.0 / mean_of_rates * 60.0

    # ------------------------------------------------- 97.5 s memory benchmark
    memlog = PROJECT / "results" / "memory_benchmarks" / "magicc_time.log"
    mem = parse_time_v(memlog)
    internal_total = None
    model_used = None
    breakdown = {}
    if memlog.exists():
        txt = memlog.read_text()
        m = re.search(r"Total time:\s*([0-9.]+)s", txt)
        if m:
            internal_total = float(m.group(1))
        m = re.search(r"Loading ONNX model: (\S+)", txt)
        if m:
            model_used = m.group(1)
        for key, label in [("Feature extraction", "feature_extraction_s"),
                           ("Normalization", "normalization_s"),
                           ("ONNX inference", "onnx_inference_s")]:
            m = re.search(rf"{key}:\s*([0-9.]+)s", txt)
            if m:
                breakdown[label] = float(m.group(1))

    # -------------------------------------------------------- 74.4 s / 7.9 s
    p7 = PROJECT / "results" / "phase7_set_e_test" / "summary.json"
    phase7 = json.loads(p7.read_text()) if p7.exists() else {}

    # ------------------------------------------------------------- verdicts
    rec = {
        "question": "R1-m11: reconcile '1,000 genomes in 40 seconds' (main text) with "
                    "'97.5 s for 1,000 genomes' (Table S4).",
        "verdict_summary": (
            "The two numbers were never measurements of the same thing, and they were "
            "never even measurements of the same experiment. 97.5 s is a real end-to-end "
            "wall-clock measurement of one process. 40 s is not a measurement at all: it "
            "is 1,000 divided by the 1,451 genomes/min/thread figure printed in the same "
            "table, and 1,451 is the arithmetic mean of five per-benchmark-set throughput "
            "rates computed from MAGICC's own internal compute-phase timer. Table S4 "
            "therefore places, in a single row, a wall-clock column and a throughput "
            "column that disagree with each other by 2.36x."
        ),

        "figure_97_5_s": {
            "value_s": mem.get("wall_clock_s"),
            "what_it_measured": "END-TO-END wall clock of one OS process: interpreter "
                                "start, imports, Numba JIT warm-up, ONNX model load, "
                                "FASTA read, k-mer counting, normalization, inference, "
                                "output write, process exit.",
            "instrument": "/usr/bin/time -v",
            "threads": 1,
            "input": "Set E, 1,000 genomes",
            "model": model_used,
            "batch_size": 64,
            "file_io": "uncompressed .fasta read from a directory on /dev/sdb (ext4); "
                       "page-cache state at the time not recorded",
            "command": mem.get("command"),
            "peak_rss_gb": mem.get("peak_rss_gb"),
            "pct_cpu": mem.get("pct_cpu"),
            "date": "2026-02-15",
            "magicc_internal_total_time_s": internal_total,
            "magicc_internal_breakdown_s": breakdown,
            "process_overhead_outside_internal_timer_s":
                None if (internal_total is None or mem.get("wall_clock_s") is None)
                else round(mem["wall_clock_s"] - internal_total, 2),
            "status": "SURVIVES as a correctly-labelled end-to-end measurement, but it "
                      "was made with magicc_v3.onnx, which is neither the model the "
                      "manuscript reports (V4) nor the released model (V5). Superseded by "
                      "the WS8.1 V5 re-measurement.",
        },

        "figure_1451_genomes_per_min_per_thread": {
            "value": round(mean_of_rates, 1),
            "how_it_was_computed": "scripts/30_benchmark_analysis.py:623 and :834 -- "
                                   "speed_per_thread = (n / wall_clock_s * 60) / n_threads, "
                                   "evaluated once per benchmark set and then arithmetically "
                                   "AVERAGED over the five sets.",
            "wall_clock_s_source": "the `wall_clock_s` column of each set's "
                                   "magicc_predictions.tsv, written by "
                                   "scripts/26_benchmark_run_magicc.py:180 as "
                                   "total_time = feat_time + infer_time",
            "what_that_timer_excludes": [
                "interpreter start and imports (numpy, onnxruntime, numba)",
                "Numba JIT compilation of the k-mer counter",
                "ONNX model load / session creation",
                "feature normalization (computed between the two timers, so counted nowhere)",
                "reading the benchmark metadata table",
                "writing the predictions TSV",
            ],
            "model": "magicc_v3.onnx (scripts/26 line 37 and scripts/35 line 36)",
            "threads": 1,
            "per_set": per_set.round(4).to_dict(orient="records"),
            "defect_1_mean_of_ratios": {
                "explanation": "Averaging five throughput RATES gives each set equal weight "
                               "regardless of how much work it represents. The five sets "
                               "differ ~5-fold in per-genome cost (Set C is Patescibacteriota, "
                               "median genome 0.88 Mbp, and reaches 3,088 genomes/min; Set E "
                               "is finished genomes and reaches 614). The correct pooled "
                               "figure is total genomes / total time.",
                "mean_of_rates": round(mean_of_rates, 1),
                "pooled_total_genomes_over_total_time": round(pooled, 1),
                "inflation_factor": round(mean_of_rates / pooled, 3),
            },
            "defect_2_wrong_timer": "Even the pooled 1,066 genomes/min/thread is a "
                                    "compute-phase figure, not wall clock.",
            "status": "RETIRED. Not reported in the revision in this form.",
        },

        "figure_40_s": {
            "value_s": 40,
            "provenance": "NOT A MEASUREMENT. No run of any length near 40 s for 1,000 "
                          "genomes exists anywhere in the repository's logs or result files. "
                          "The number is 1,000 genomes / 1,451 genomes-per-minute = "
                          f"{derived_40s:.1f} s, rounded down to '40 seconds'.",
            "arithmetic_check_s": round(derived_40s, 2),
            "status": "RETIRED. It inherits both defects of the 1,451 figure (mean of "
                      "ratios; compute-phase timer rather than wall clock) and was never "
                      "measured. It is withdrawn rather than re-explained.",
        },

        "figure_74_4_s_and_7_9_s": {
            "source": "scripts/40_test_cli_set_e.py -> results/phase7_set_e_test/",
            "what_it_measured": "END-TO-END wall clock of `python -m magicc predict` as a "
                                "subprocess (time.time() around subprocess.run), i.e. the "
                                "same definition as the 97.5 s figure but WITHOUT the "
                                "`conda run` wrapper.",
            "one_thread_s": phase7.get("speed_single_thread", {}).get("wall_clock_s"),
            "forty_three_thread_s": phase7.get("speed_multi_thread", {}).get("wall_clock_s"),
            "threads": [1, 43],
            "model": phase7.get("model"),
            "input": "Set E, 1,000 genomes, directory input, .fasta, batch size 64",
            "status": "SURVIVES as an end-to-end measurement, but it is a V3 measurement "
                      "and it disagrees with the 97.5 s figure by 23 s for nominally the "
                      "same configuration on the same day. The two differ in wrapper "
                      "(`conda run` vs direct) and in unrecorded machine state; neither run "
                      "recorded load average or page-cache state, so the 23 s cannot be "
                      "attributed from the archive. Both are superseded by the WS8.1 "
                      "re-measurement, which records load average and repeats 3 times.",
            "note_43_threads": "7.9 s at 43 threads corresponds to 176 genomes/min/thread, "
                               "not 1,451 -- per-thread throughput FALLS with thread count "
                               "because the fixed startup cost is amortised over a shorter "
                               "run. Any per-thread ratio must therefore carry its thread "
                               "count, which is exactly R1-m20's complaint.",
        },

        "table_S4_internal_inconsistency": {
            "row": "MAGICC | 1 thread | 0.66 GB | 97.5 s | 1,451 genomes/min/thread",
            "problem": "1,000 genomes in 97.5 s is 615 genomes/min/thread, not 1,451. The "
                       "wall-clock column and the throughput column of the same row come "
                       "from two different experiments run with two different timers.",
            "ratio": round(mean_of_rates / (1000 / 97.52 * 60), 3),
            "same_defect_in_competitor_rows": (
                "The competitor throughput column is derived the same way. CheckM2's row "
                "states 32 threads and 86 min 37 s, which is 0.361 genomes/min/thread, but "
                "the table prints 0.82. So the published 1,700x and 2,100x speed ratios "
                "divide one tool's mean-of-ratios by another tool's mean-of-ratios, and "
                "neither numerator nor denominator is the wall clock printed beside it."
            ),
        },

        "what_the_revision_reports_instead": (
            "One uniformly defined metric: end-to-end wall clock from process start to "
            "results written, including model and database load, measured with "
            "/usr/bin/time -v, repeated 3 times per cell on a verified-idle host with the "
            "load average recorded alongside every run, at 1/8/16/32 threads, on an "
            "identical input set for every tool. Per-thread throughput is reported only as "
            "a secondary quantity, always with its thread count."
        ),
    }

    (OUT / "reconciliation_40s_vs_97.5s.json").write_text(json.dumps(rec, indent=2))
    per_set.to_csv(OUT / "reconciliation_per_set_internal_timers.tsv", sep="\t", index=False)

    print("WS8.3 reconciliation")
    print("-" * 72)
    print(per_set.to_string(index=False))
    print(f"\nmean of the five per-set rates = {mean_of_rates:.1f} g/min/thread  -> published as 1,451")
    print(f"pooled  total/total           = {pooled:.1f} g/min/thread  (inflation {mean_of_rates/pooled:.3f}x)")
    print(f"1000 genomes / {mean_of_rates:.1f} per min = {derived_40s:.1f} s -> published as '40 seconds'")
    print(f"\n97.5 s end-to-end  = {1000/97.52*60:.0f} g/min/thread")
    print(f"ratio 1,451 / 615  = {mean_of_rates/(1000/97.52*60):.2f}x  <- the reviewer's discrepancy")
    print(f"\nwrote {OUT/'reconciliation_40s_vs_97.5s.json'}")


if __name__ == "__main__":
    main()
