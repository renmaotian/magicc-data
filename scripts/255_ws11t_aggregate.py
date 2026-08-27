#!/usr/bin/env python3
"""
WS11.T (T1 + T2) — aggregate the new repeats, pool them with the archived WS8
runs, and re-derive the headline ratios and the peak-RSS ratios.

Reads  results/revision/speed/runs/*.json       (archived WS8, READ-ONLY)
       results/revision/speed_v3/runs/*.json    (new WS11.T repeats)

Writes results/revision/speed_v3/
       matched_thread_runs_v3.tsv   every NEW run, one row
       pooled_cell_summary.tsv      archived + new, per cell: n, median, min, max
       updated_ratios.tsv           re-derived headline ratios
       rss_ratios_by_cell.tsv       the unambiguous peak-RSS ratio table (T2)
       flagged_runs.tsv             cells needing an extra repeat, and why

FLAGGING RULE (coordinator, 2026-08-26): flag, never drop, a run whose pre-run
1-minute load average was >= 4.0, or whose wall clock deviates more than 20 %
from its cell's median.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/path/to/magicc")
OLD = PROJECT / "results" / "revision" / "speed"
NEW = PROJECT / "results" / "revision" / "speed_v3"

LOAD_FLAG = 4.0
DEV_FLAG_PCT = 20.0

TOOL_LABEL = {
    "magicc": "MAGICC v0.3.0 (V5)",
    "magicc_dir": "MAGICC v0.3.0 (V5), directory input",
    "magicc_v5code": "MAGICC V5 code tree via code-root launcher",
    "magicc_v3code": "MAGICC V3 code tree (git 471eb28) + magicc_v3.onnx",
    "magicc_v3code_nostats": "MAGICC V3 code tree, 19 assembly statistics disabled",
    "magicc_condarun": "MAGICC v0.3.0 (V5) via `conda run`",
    "checkm2": "CheckM2 1.0.1",
    "checkm2_vectors": "CheckM2 1.0.1 (+feature dump)",
    "cocopye": "CoCoPyE 0.5.0",
    "deepcheck": "DeepCheck (inference only)",
}

# cells whose input set is the same 1,000 genomes supplied a different way
INPUT_SET_CANON = {"set_E_full_dir": "set_E_full"}


def load(dirpath: Path, campaign: str) -> pd.DataFrame:
    recs = []
    for f in sorted((dirpath / "runs").glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "wall_clock_s" not in r:
            continue
        r["cell_file"] = str(f)
        r.setdefault("campaign", campaign)
        recs.append(r)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["loadavg_1min_before"] = df["loadavg_before"].apply(lambda x: x[0])
    df["loadavg_5min_before"] = df["loadavg_before"].apply(lambda x: x[1])
    df["loadavg_15min_before"] = df["loadavg_before"].apply(lambda x: x[2])
    df["loadavg_1min_after"] = df["loadavg_after"].apply(lambda x: x[0])
    df["tool_label"] = df["tool"].map(TOOL_LABEL).fillna(df["tool"])
    df["input_set_canon"] = df["input_set"].replace(INPUT_SET_CANON)
    return df


def cell_key(df: pd.DataFrame) -> pd.Series:
    return (df.tool + "|t" + df.threads.astype(str) + "|"
            + df.input_set_canon + "|" + df.cache)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["tool", "tool_label", "input_set_canon", "cache", "threads"],
                   as_index=False)
    out = g.agg(
        n=("wall_clock_s", "size"),
        wall_median_s=("wall_clock_s", "median"),
        wall_min_s=("wall_clock_s", "min"),
        wall_max_s=("wall_clock_s", "max"),
        wall_mean_s=("wall_clock_s", "mean"),
        rss_median_gb=("peak_rss_gb", "median"),
        rss_min_gb=("peak_rss_gb", "min"),
        rss_max_gb=("peak_rss_gb", "max"),
        pct_cpu_median=("pct_cpu", "median"),
        load1_before_max=("loadavg_1min_before", "max"),
        fs_inputs_max=("fs_inputs", "max"),
        busy_pct_before_max=("sum_pcpu_all_processes_before", "max"),
        rc_max=("return_code", "max"),
        n_genomes=("n_genomes", "max"),
        n_out_min=("n_output_rows", "min"),
        campaigns=("campaign", lambda s: "+".join(sorted(set(s)))),
    )
    out["wall_range_pct"] = (out.wall_max_s - out.wall_min_s) / out.wall_median_s * 100
    return out.sort_values(["input_set_canon", "threads", "tool"])


def pick(s: pd.DataFrame, tool: str, iset: str, threads: int, cache="warm"):
    r = s[(s.tool == tool) & (s.input_set_canon == iset)
          & (s.threads == threads) & (s.cache == cache)]
    return None if r.empty else r.iloc[0]


def main() -> None:
    old = load(OLD, "WS8")
    new = load(NEW, "WS11.T")
    if new.empty:
        print("no WS11.T runs yet")
        return
    both = pd.concat([old, new], ignore_index=True)

    # ---------------------------------------------------- per-run flagging
    med = both.groupby(cell_key(both))["wall_clock_s"].transform("median")
    both["cell_median_s"] = med
    both["dev_from_cell_median_pct"] = (both.wall_clock_s - med).abs() / med * 100
    both["flag_load"] = both.loadavg_1min_before >= LOAD_FLAG
    both["flag_deviation"] = both.dev_from_cell_median_pct > DEV_FLAG_PCT
    both["flagged"] = both.flag_load | both.flag_deviation

    runcols = ["campaign", "tool", "tool_label", "threads", "repeat", "input_set",
               "input_set_canon", "cache", "n_genomes", "n_output_rows",
               "wall_clock_s", "peak_rss_gb", "max_rss_kb", "pct_cpu",
               "user_s", "sys_s", "fs_inputs", "fs_outputs",
               "major_faults", "minor_faults",
               "loadavg_1min_before", "loadavg_5min_before", "loadavg_15min_before",
               "loadavg_1min_after", "sum_pcpu_all_processes_before", "nproc",
               "mem_available_kb_before", "mem_available_kb_after",
               "vmstat_before", "vmstat_after", "code_root", "model",
               "t_start_utc", "t_end_utc", "return_code",
               "cell_median_s", "dev_from_cell_median_pct",
               "flag_load", "flag_deviation", "flagged", "cell_file"]
    nrows = both[both.campaign == "WS11.T"][
        [c for c in runcols if c in both.columns]].sort_values(
        ["input_set_canon", "threads", "tool", "repeat"])
    nrows.round(4).to_csv(NEW / "matched_thread_runs_v3.tsv", sep="\t", index=False)

    both[both.flagged][[c for c in runcols if c in both.columns]].round(4).to_csv(
        NEW / "flagged_runs.tsv", sep="\t", index=False)

    s = summarise(both)
    s.round(4).to_csv(NEW / "pooled_cell_summary.tsv", sep="\t", index=False)

    manifest = json.loads((OLD / "inputs" / "input_manifest.json").read_text())
    bp = {"set_E_full": manifest["set_E_full"]["total_sequence_bp"],
          "set_E_100": manifest["set_E_100"]["total_sequence_bp"]}

    # -------------------------------------------------------- T2: ratios
    PUBLISHED = {("set_E_full", 32): 697.0, ("set_E_100", 32): 251.0}
    rows = []
    for iset, thr in [("set_E_full", 32), ("set_E_100", 32)]:
        num = pick(s, "checkm2", iset, thr)
        den = pick(s, "magicc", iset, thr)
        if num is None or den is None:
            continue
        r_med = num.wall_median_s / den.wall_median_s
        pub = PUBLISHED[(iset, thr)]
        rows.append(dict(
            cell=f"{iset} @ {thr} threads",
            input_set=iset,
            n_genomes=int(num.n_genomes),
            total_bp=bp[iset],
            threads=thr,
            numerator="CheckM2 1.0.1",
            denominator="MAGICC v0.3.0 (V5)",
            n_numerator=int(num.n),
            num_median_s=round(num.wall_median_s, 3),
            num_min_s=round(num.wall_min_s, 3),
            num_max_s=round(num.wall_max_s, 3),
            n_denominator=int(den.n),
            den_median_s=round(den.wall_median_s, 3),
            den_min_s=round(den.wall_min_s, 3),
            den_max_s=round(den.wall_max_s, 3),
            ratio_of_medians=round(r_med, 1),
            ratio_min_envelope=round(num.wall_min_s / den.wall_max_s, 1),
            ratio_max_envelope=round(num.wall_max_s / den.wall_min_s, 1),
            published_ratio=pub,
            change_abs=round(r_med - pub, 1),
            change_pct=round((r_med - pub) / pub * 100, 2),
            campaigns=f"CheckM2 {num.campaigns}; MAGICC {den.campaigns}",
        ))
        # CoCoPyE for completeness
        num2 = pick(s, "cocopye", iset, thr)
        if num2 is not None:
            rows.append(dict(
                cell=f"{iset} @ {thr} threads",
                input_set=iset, n_genomes=int(num2.n_genomes), total_bp=bp[iset],
                threads=thr, numerator="CoCoPyE 0.5.0",
                denominator="MAGICC v0.3.0 (V5)",
                n_numerator=int(num2.n),
                num_median_s=round(num2.wall_median_s, 3),
                num_min_s=round(num2.wall_min_s, 3),
                num_max_s=round(num2.wall_max_s, 3),
                n_denominator=int(den.n),
                den_median_s=round(den.wall_median_s, 3),
                den_min_s=round(den.wall_min_s, 3),
                den_max_s=round(den.wall_max_s, 3),
                ratio_of_medians=round(num2.wall_median_s / den.wall_median_s, 1),
                ratio_min_envelope=round(num2.wall_min_s / den.wall_max_s, 1),
                ratio_max_envelope=round(num2.wall_max_s / den.wall_min_s, 1),
                published_ratio=np.nan, change_abs=np.nan, change_pct=np.nan,
                campaigns=f"CoCoPyE {num2.campaigns}; MAGICC {den.campaigns}",
            ))
    pd.DataFrame(rows).to_csv(NEW / "updated_ratios.tsv", sep="\t", index=False)

    # ---------------------------------------------------- T2: RSS by cell
    rrows = []
    for iset, thr in [("set_E_100", 32), ("set_E_full", 32)]:
        den = pick(s, "magicc", iset, thr)
        for tool in ["checkm2", "cocopye", "deepcheck"]:
            num = pick(s, tool, iset, thr)
            if num is None or den is None:
                continue
            rrows.append(dict(
                cell=f"{iset} @ {thr} threads",
                input_set=iset, n_genomes=int(den.n_genomes), total_bp=bp[iset],
                threads=thr,
                magicc_rss_gb_median=round(den.rss_median_gb, 4),
                magicc_rss_gb_max=round(den.rss_max_gb, 4),
                n_magicc=int(den.n),
                competitor=TOOL_LABEL[tool],
                competitor_rss_gb_median=round(num.rss_median_gb, 4),
                competitor_rss_gb_max=round(num.rss_max_gb, 4),
                n_competitor=int(num.n),
                rss_ratio_median_over_median=round(
                    num.rss_median_gb / den.rss_median_gb, 2),
                rss_ratio_max_over_max=round(num.rss_max_gb / den.rss_max_gb, 2),
            ))
    # provenance rows: reproduce the published-style "Peak RSS" column, which
    # takes each tool's MAXIMUM peak RSS over ALL thread counts of an input set.
    # That is where the manuscript's 27.7x comes from, and it silently compares
    # CheckM2 at 32 threads against MAGICC at 1 thread.
    for iset in ["set_E_100", "set_E_full"]:
        sub = s[(s.input_set_canon == iset)]
        mg = sub[sub.tool == "magicc"]
        for tool in ["checkm2", "cocopye", "deepcheck"]:
            cp = sub[sub.tool == tool]
            if mg.empty or cp.empty:
                continue
            mg_i = mg.loc[mg.rss_max_gb.idxmax()]
            cp_i = cp.loc[cp.rss_max_gb.idxmax()]
            rrows.append(dict(
                cell=f"PROVENANCE (published style): {iset}, MAX over ALL thread counts",
                input_set=iset, n_genomes=int(mg_i.n_genomes), total_bp=bp[iset],
                threads=f"MAGICC@{int(mg_i.threads)} vs {tool}@{int(cp_i.threads)}",
                magicc_rss_gb_median=float("nan"),
                magicc_rss_gb_max=round(mg_i.rss_max_gb, 4),
                n_magicc=int(mg_i.n),
                competitor=TOOL_LABEL[tool],
                competitor_rss_gb_median=float("nan"),
                competitor_rss_gb_max=round(cp_i.rss_max_gb, 4),
                n_competitor=int(cp_i.n),
                rss_ratio_median_over_median=float("nan"),
                rss_ratio_max_over_max=round(cp_i.rss_max_gb / mg_i.rss_max_gb, 2),
            ))

    pd.DataFrame(rrows).to_csv(NEW / "rss_ratios_by_cell.tsv", sep="\t", index=False)

    # ------------------------------------------------------------- console
    key = s[(s.threads == 32) & (s.cache == "warm")
            & s.tool.isin(["magicc", "checkm2", "cocopye", "deepcheck"])]
    print("== pooled 32-thread warm cells ==")
    print(key[["tool", "input_set_canon", "n", "wall_median_s", "wall_min_s",
               "wall_max_s", "rss_median_gb", "rss_max_gb", "campaigns"]]
          .to_string(index=False))
    print("\n== updated ratios ==")
    if rows:
        print(pd.DataFrame(rows)[["cell", "numerator", "n_numerator", "num_median_s",
                                  "n_denominator", "den_median_s",
                                  "ratio_of_medians", "published_ratio"]]
              .to_string(index=False))
    print("\n== RSS ratios ==")
    print(pd.DataFrame(rrows)[["cell", "magicc_rss_gb_median", "competitor",
                               "competitor_rss_gb_median",
                               "rss_ratio_median_over_median"]].to_string(index=False))
    nflag = int(both.flagged.sum())
    print(f"\n{len(nrows)} new runs; {nflag} flagged run(s) across old+new "
          f"-> {NEW/'flagged_runs.tsv'}")


if __name__ == "__main__":
    main()
