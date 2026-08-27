#!/usr/bin/env python3
"""
WS11.T / v0.3.3 — aggregate the re-measured MAGICC cells, isolate the cost of
the per-run model SHA-256 verification, and re-derive every figure that depends
on MAGICC's wall clock.

Competitor medians are NOT re-measured (nothing about them changed); they are
read from results/revision/speed_v3/pooled_cell_summary.tsv so the ratios stay
traceable to the WS11.T campaign rather than being retyped.

Writes, under results/revision/speed_v033/:
  matched_thread_runs_v033.tsv     every new run, one row
  cell_summary_v033.tsv            per arm x cell: n, median, min, max, RSS
  verification_cost_by_cell.tsv    arm A - arm B, per cell (the +0.5 s)
  updated_ratios_v033.tsv          re-derived ratios vs unchanged competitors
  rss_ratios_by_cell_v033.tsv      matched-thread peak-RSS ratios
  fixed_variable_costs_v033.tsv    extends results/revision/speed/fixed_variable_costs.tsv
  scaling_efficiency_v033.tsv      extends results/revision/speed/scaling_efficiency.tsv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT = Path("/path/to/magicc")
V3 = PROJECT / "results/revision/speed_v3"
OLD = PROJECT / "results/revision/speed"
NEW = PROJECT / "results/revision/speed_v033"

LABEL = {
    "magicc_v033": "MAGICC v0.3.3 (released path, model SHA256 verified each run)",
    "magicc_v033_xmodel": "MAGICC v0.3.3 (explicit --model, verification bypassed)",
}


def load_runs() -> pd.DataFrame:
    recs = []
    for f in sorted((NEW / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        if "wall_clock_s" not in r:
            continue
        r["cell_file"] = str(f)
        recs.append(r)
    df = pd.DataFrame(recs)
    df["loadavg_1min_before"] = df.loadavg_before.apply(lambda x: x[0])
    df["loadavg_5min_before"] = df.loadavg_before.apply(lambda x: x[1])
    df["loadavg_15min_before"] = df.loadavg_before.apply(lambda x: x[2])
    df["loadavg_1min_after"] = df.loadavg_after.apply(lambda x: x[0])
    df["tool_label"] = df.tool.map(LABEL)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["tool", "tool_label", "input_set", "threads"], as_index=False)
    s = g.agg(n=("wall_clock_s", "size"),
              wall_median_s=("wall_clock_s", "median"),
              wall_min_s=("wall_clock_s", "min"),
              wall_max_s=("wall_clock_s", "max"),
              rss_median_gb=("peak_rss_gb", "median"),
              rss_max_gb=("peak_rss_gb", "max"),
              pct_cpu_median=("pct_cpu", "median"),
              load1_before_max=("loadavg_1min_before", "max"),
              fs_inputs_max=("fs_inputs", "max"),
              rc_max=("return_code", "max"),
              n_genomes=("n_genomes", "max"),
              n_out_min=("n_output_rows", "min"))
    return s.sort_values(["input_set", "threads", "tool"])


def main() -> None:
    df = load_runs()
    if df.empty:
        print("no v0.3.3 runs yet"); return

    cols = ["campaign", "tool", "tool_label", "threads", "repeat", "input_set", "cache",
            "n_genomes", "n_output_rows", "wall_clock_s", "peak_rss_gb", "max_rss_kb",
            "pct_cpu", "user_s", "sys_s", "fs_inputs", "fs_outputs",
            "major_faults", "minor_faults", "loadavg_1min_before", "loadavg_5min_before",
            "loadavg_15min_before", "loadavg_1min_after",
            "sum_pcpu_all_processes_before", "nproc",
            "mem_available_kb_before", "mem_available_kb_after",
            "vmstat_before", "vmstat_after", "code_root", "model",
            "t_start_utc", "t_end_utc", "return_code", "cell_file"]
    df[[c for c in cols if c in df.columns]].sort_values(
        ["input_set", "threads", "tool", "repeat"]).round(4).to_csv(
        NEW / "matched_thread_runs_v033.tsv", sep="\t", index=False)

    s = summarise(df)
    s.round(4).to_csv(NEW / "cell_summary_v033.tsv", sep="\t", index=False)

    def cell(tool, iset, thr):
        r = s[(s.tool == tool) & (s.input_set == iset) & (s.threads == thr)]
        return None if r.empty else r.iloc[0]

    # ------------------------------------------- verification cost per cell
    vrows = []
    for iset in ["set_E_100", "set_E_full"]:
        for thr in [1, 8, 16, 32]:
            a, b = cell("magicc_v033", iset, thr), cell("magicc_v033_xmodel", iset, thr)
            if a is None or b is None:
                continue
            d = a.wall_median_s - b.wall_median_s
            vrows.append(dict(
                input_set=iset, n_genomes=int(a.n_genomes), threads=thr,
                verified_median_s=round(a.wall_median_s, 3), n_verified=int(a.n),
                unverified_median_s=round(b.wall_median_s, 3), n_unverified=int(b.n),
                verification_cost_s=round(d, 3),
                verification_share_of_run_pct=round(d / a.wall_median_s * 100, 2),
                verified_min_s=round(a.wall_min_s, 3), verified_max_s=round(a.wall_max_s, 3),
                unverified_min_s=round(b.wall_min_s, 3), unverified_max_s=round(b.wall_max_s, 3),
            ))
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(NEW / "verification_cost_by_cell.tsv", sep="\t", index=False)

    # ------------------------------------------------------- ratios vs comp
    comp = pd.read_csv(V3 / "pooled_cell_summary.tsv", sep="\t")

    def compcell(tool, iset, thr=32):
        r = comp[(comp.tool == tool) & (comp.input_set_canon == iset)
                 & (comp.threads == thr) & (comp.cache == "warm")]
        return None if r.empty else r.iloc[0]

    PUB = {("set_E_full", "checkm2"): 716.7, ("set_E_100", "checkm2"): 253.2,
           ("set_E_full", "cocopye"): 496.9, ("set_E_100", "cocopye"): 156.9}
    rrows = []
    for iset in ["set_E_full", "set_E_100"]:
        for ctool, clab in [("checkm2", "CheckM2 1.0.1"), ("cocopye", "CoCoPyE 0.5.0")]:
            c = compcell(ctool, iset)
            if c is None:
                continue
            for arm in ["magicc_v033", "magicc_v033_xmodel"]:
                d = cell(arm, iset, 32)
                if d is None:
                    continue
                rrows.append(dict(
                    cell=f"{iset} @ 32 threads", input_set=iset,
                    n_genomes=int(c.n_genomes), threads=32,
                    numerator=clab, n_numerator=int(c.n),
                    num_median_s=round(c.wall_median_s, 2),
                    num_min_s=round(c.wall_min_s, 2), num_max_s=round(c.wall_max_s, 2),
                    denominator=LABEL[arm], n_denominator=int(d.n),
                    den_median_s=round(d.wall_median_s, 3),
                    den_min_s=round(d.wall_min_s, 3), den_max_s=round(d.wall_max_s, 3),
                    ratio_of_medians=round(c.wall_median_s / d.wall_median_s, 1),
                    ratio_min_envelope=round(c.wall_min_s / d.wall_max_s, 1),
                    ratio_max_envelope=round(c.wall_max_s / d.wall_min_s, 1),
                    ws11t_ratio=PUB[(iset, ctool)],
                    change_pct=round((c.wall_median_s / d.wall_median_s
                                      - PUB[(iset, ctool)]) / PUB[(iset, ctool)] * 100, 2),
                ))
    pd.DataFrame(rrows).to_csv(NEW / "updated_ratios_v033.tsv", sep="\t", index=False)

    # ------------------------------------------------------------ RSS ratio
    srows = []
    for iset in ["set_E_100", "set_E_full"]:
        for arm in ["magicc_v033", "magicc_v033_xmodel"]:
            d = cell(arm, iset, 32)
            if d is None:
                continue
            for ctool, clab in [("checkm2", "CheckM2 1.0.1"), ("cocopye", "CoCoPyE 0.5.0"),
                                ("deepcheck", "DeepCheck (inference only)")]:
                c = compcell(ctool, iset)
                if c is None:
                    continue
                srows.append(dict(
                    cell=f"{iset} @ 32 threads", input_set=iset,
                    n_genomes=int(d.n_genomes), threads=32,
                    magicc_arm=LABEL[arm], n_magicc=int(d.n),
                    magicc_rss_gb_median=round(d.rss_median_gb, 4),
                    magicc_rss_gb_max=round(d.rss_max_gb, 4),
                    competitor=clab, n_competitor=int(c.n),
                    competitor_rss_gb_median=round(c.rss_median_gb, 4),
                    rss_ratio_median_over_median=round(
                        c.rss_median_gb / d.rss_median_gb, 2)))
    pd.DataFrame(srows).to_csv(NEW / "rss_ratios_by_cell_v033.tsv", sep="\t", index=False)

    # ------------------------------------------- fixed / variable, extended
    frows = []
    for arm in ["magicc_v033", "magicc_v033_xmodel"]:
        for thr in [1, 8, 16, 32]:
            a, b = cell(arm, "set_E_100", thr), cell(arm, "set_E_full", thr)
            if a is None or b is None:
                continue
            marg = (b.wall_median_s - a.wall_median_s) / 900.0
            fixed = a.wall_median_s - marg * 100
            frows.append(dict(
                tool=arm, tool_label=LABEL[arm], threads=thr,
                wall_100_s=round(a.wall_median_s, 3),
                wall_1000_s=round(b.wall_median_s, 3),
                fixed_startup_s=round(fixed, 4),
                marginal_s_per_genome=round(marg, 6),
                fixed_share_of_100genome_run_pct=round(fixed / a.wall_median_s * 100, 1),
                fixed_share_of_1000genome_run_pct=round(fixed / b.wall_median_s * 100, 1)))
    pd.DataFrame(frows).to_csv(NEW / "fixed_variable_costs_v033.tsv", sep="\t", index=False)

    # -------------------------------------------------- scaling, extended
    krows = []
    for arm in ["magicc_v033", "magicc_v033_xmodel"]:
        for iset in ["set_E_100", "set_E_full"]:
            base = cell(arm, iset, 1)
            if base is None:
                continue
            for thr in [1, 8, 16, 32]:
                d = cell(arm, iset, thr)
                if d is None:
                    continue
                sp = base.wall_median_s / d.wall_median_s
                gpm = d.n_genomes / d.wall_median_s * 60
                krows.append(dict(
                    tool=arm, tool_label=LABEL[arm], input_set=iset,
                    n_genomes=int(d.n_genomes), threads=thr,
                    wall_median_s=round(d.wall_median_s, 3), n_repeats=int(d.n),
                    speedup_vs_1_thread=round(sp, 2),
                    parallel_efficiency_pct=round(sp / thr * 100, 1),
                    genomes_per_min=round(gpm, 1),
                    genomes_per_min_per_thread=round(gpm / thr, 1)))
    pd.DataFrame(krows).to_csv(NEW / "scaling_efficiency_v033.tsv", sep="\t", index=False)

    print("== cells ==")
    print(s[["tool", "input_set", "threads", "n", "wall_median_s", "wall_min_s",
             "wall_max_s", "rss_median_gb", "rc_max"]].to_string(index=False))
    print("\n== verification cost ==")
    print(vdf.to_string(index=False))
    print("\n== ratios ==")
    rd = pd.DataFrame(rrows)
    print(rd[["cell", "numerator", "denominator", "den_median_s",
              "ratio_of_medians", "ws11t_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
