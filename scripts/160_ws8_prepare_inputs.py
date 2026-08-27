#!/usr/bin/env python3
"""
WS8.1 input preparation — build the matched input sets for the speed benchmark.

Two nested input sets, both drawn from Set E (the historical comparator):

  set_E_full  : all 1,000 genomes            -> used where every tool can afford it
  set_E_100   : 100-genome seeded subsample  -> the FULL 4-tool x 4-thread factorial
                                                (CheckM2 at 1 thread on 1,000 genomes
                                                 is ~25-30 h per repeat; see 8.1 notes)

The subsample is drawn with a CRC-32 stable_hash-derived seed (NOT abs(hash()),
which is salted per interpreter process) so it is reproducible across sessions.

Writes:
  results/revision/speed/inputs/set_E_full.txt   (one absolute path per line)
  results/revision/speed/inputs/set_E_100.txt
  results/revision/speed/inputs/input_manifest.json
  results/revision/speed/inputs/set_E_100/       (symlink farm, for dir-input tools)

Usage:
  PYTHONHASHSEED=0 python scripts/160_ws8_prepare_inputs.py
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

PROJECT = Path("/path/to/magicc")
SET_E = PROJECT / "data" / "benchmarks" / "set_E" / "fasta"
OUT = PROJECT / "results" / "revision" / "speed" / "inputs"
SUBSAMPLE_N = 100


def stable_hash(s: str) -> int:
    """CRC-32 based, process-independent hash (see scripts/101_metrics_framework.py)."""
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def fasta_stats(path: Path) -> tuple[int, int]:
    """Return (sequence_bp, n_contigs) for a plain FASTA."""
    bp = 0
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                n += 1
            else:
                bp += len(line.strip())
    return bp, n


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(SET_E.glob("*.fasta"))
    if not files:
        print(f"ERROR: no FASTA found under {SET_E}", file=sys.stderr)
        return 1

    print(f"[prep] Set E: {len(files)} genomes at {SET_E}")

    # --- full-set stats -----------------------------------------------------
    print("[prep] measuring sequence bp (this reads every file once) ...")
    per_file = {}
    for f in files:
        bp, nc = fasta_stats(f)
        per_file[f.name] = {"bytes": f.stat().st_size, "bp": bp, "n_contigs": nc}

    tot_bp = sum(v["bp"] for v in per_file.values())
    tot_bytes = sum(v["bytes"] for v in per_file.values())
    bps = np.array([v["bp"] for v in per_file.values()])

    # --- seeded subsample ---------------------------------------------------
    seed = stable_hash("magicc-ws8-speed-subsample-v1")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(files), size=SUBSAMPLE_N, replace=False))
    sub = [files[i] for i in idx]
    sub_bp = sum(per_file[f.name]["bp"] for f in sub)
    sub_bytes = sum(per_file[f.name]["bytes"] for f in sub)

    # --- write path lists ---------------------------------------------------
    (OUT / "set_E_full.txt").write_text("\n".join(str(f) for f in files) + "\n")
    (OUT / "set_E_100.txt").write_text("\n".join(str(f) for f in sub) + "\n")

    # --- symlink farm for directory-input tools (CheckM2, CoCoPyE) ----------
    farm = OUT / "set_E_100"
    farm.mkdir(exist_ok=True)
    for old in farm.glob("*"):
        old.unlink()
    for f in sub:
        (farm / f.name).symlink_to(f)

    manifest = {
        "generated_by": "scripts/160_ws8_prepare_inputs.py",
        "source": str(SET_E),
        "subsample_seed_string": "magicc-ws8-speed-subsample-v1",
        "subsample_seed_crc32": int(seed),
        "subsample_rng": "numpy.random.default_rng(crc32_seed).choice(n, size=100, replace=False)",
        "set_E_full": {
            "n_genomes": len(files),
            "total_sequence_bp": int(tot_bp),
            "total_file_bytes": int(tot_bytes),
            "mean_bp_per_genome": float(tot_bp / len(files)),
            "median_bp_per_genome": float(np.median(bps)),
            "min_bp": int(bps.min()),
            "max_bp": int(bps.max()),
            "path_list": str(OUT / "set_E_full.txt"),
            "directory": str(SET_E),
        },
        "set_E_100": {
            "n_genomes": len(sub),
            "total_sequence_bp": int(sub_bp),
            "total_file_bytes": int(sub_bytes),
            "mean_bp_per_genome": float(sub_bp / len(sub)),
            "fraction_of_full_bp": float(sub_bp / tot_bp),
            "path_list": str(OUT / "set_E_100.txt"),
            "directory": str(farm),
            "members": [f.name for f in sub],
        },
        "per_file": per_file,
    }
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[prep] set_E_full : {len(files):>5} genomes  {tot_bp/1e9:.3f} Gbp  "
          f"{tot_bytes/1e9:.3f} GB on disk")
    print(f"[prep] set_E_100  : {len(sub):>5} genomes  {sub_bp/1e9:.3f} Gbp  "
          f"{sub_bytes/1e9:.3f} GB on disk  "
          f"({100*sub_bp/tot_bp:.2f}% of full-set bp)")
    print(f"[prep] wrote {OUT/'input_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
