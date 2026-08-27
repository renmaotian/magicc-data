#!/usr/bin/env python3
"""WS8.1 — parse a `/usr/bin/time -v` file into one machine-readable run record."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELDS = {
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "elapsed_raw",
    "Maximum resident set size (kbytes)": "max_rss_kb",
    "User time (seconds)": "user_s",
    "System time (seconds)": "sys_s",
    "Percent of CPU this job got": "pct_cpu",
    "Major (requiring I/O) page faults": "major_faults",
    "Minor (reclaiming a frame) page faults": "minor_faults",
    "File system inputs": "fs_inputs",
    "File system outputs": "fs_outputs",
    "Voluntary context switches": "vol_ctx",
    "Involuntary context switches": "invol_ctx",
    "Exit status": "exit_status",
}


def parse_elapsed(s: str) -> float:
    parts = s.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ["time-file", "json", "tool", "threads", "repeat", "input-set", "cache",
              "n-genomes", "n-output", "rc", "t-start", "t-end",
              "load-before", "load-after", "busy-before"]:
        ap.add_argument(f"--{a}", required=True)
    args = ap.parse_args()

    rec: dict = {
        "tool": args.tool,
        "threads": int(args.threads),
        "repeat": int(args.repeat),
        "input_set": args.input_set,
        "cache": args.cache,
        "n_genomes": int(args.n_genomes),
        "n_output_rows": int(args.n_output),
        "return_code": int(args.rc),
        "t_start_utc": args.t_start,
        "t_end_utc": args.t_end,
        "loadavg_before": [float(x) for x in args.load_before.split()],
        "loadavg_after": [float(x) for x in args.load_after.split()],
        "sum_pcpu_all_processes_before": float(args.busy_before),
    }

    tf = Path(args.time_file)
    if tf.exists():
        raw = {}
        for line in tf.read_text().splitlines():
            for key, name in FIELDS.items():
                if key in line:
                    val = line.split(":", 1)[1].strip() if name != "elapsed_raw" \
                        else line.split(": ", 1)[1].strip()
                    raw[name] = val
        rec["wall_clock_s"] = parse_elapsed(raw.get("elapsed_raw", "0"))
        rec["elapsed_raw"] = raw.get("elapsed_raw")
        for k in ["max_rss_kb", "major_faults", "minor_faults", "fs_inputs",
                  "fs_outputs", "vol_ctx", "invol_ctx", "exit_status"]:
            if k in raw:
                rec[k] = int(re.sub(r"[^0-9-]", "", raw[k]) or 0)
        for k in ["user_s", "sys_s"]:
            if k in raw:
                rec[k] = float(raw[k])
        if "pct_cpu" in raw:
            rec["pct_cpu"] = float(raw["pct_cpu"].rstrip("%"))
        if "max_rss_kb" in rec:
            rec["peak_rss_gb"] = round(rec["max_rss_kb"] / 1024 / 1024, 4)
        if rec.get("wall_clock_s"):
            rec["genomes_per_min"] = round(rec["n_genomes"] / rec["wall_clock_s"] * 60, 2)

    Path(args.json).write_text(json.dumps(rec, indent=2))
    print(f"[time] {args.tool} t={args.threads} r={args.repeat} "
          f"{args.input_set}/{args.cache}: wall={rec.get('wall_clock_s')}s "
          f"rss={rec.get('peak_rss_gb')}GB load_before={rec['loadavg_before'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
