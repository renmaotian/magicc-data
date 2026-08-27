#!/usr/bin/env python3
"""
WS8.2 — first-run setup cost.

What is measured directly: MAGICC's actual first-run model download from the
URL the released CLI uses (magicc/cli.py:44), to a scratch path, three times.
This is a real measurement of the whole first-run setup for MAGICC, because
MAGICC has no other reference data.

What is NOT re-measured, and why: CheckM2's 3.08 GB DIAMOND database and
CoCoPyE's 17.0 GB UProC/Pfam bundle are already installed. Re-downloading them
would overwrite working installations (the brief forbids modifying the
environments) and the resulting seconds would be a property of this site's
network rather than of the tool. Instead the download VOLUME is reported (which
is network-independent) together with the throughput measured here, and the
projected time is labelled as a projection.

Writes results/revision/speed/setup_cost.json
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

SPEED = Path("/path/to/magicc/results/revision/speed")
SCRATCH = SPEED / "scratch"
MODEL_URL = "https://github.com/renmaotian/magicc/raw/main/models/magicc_v5.onnx"
REPEATS = 3


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    dest = SCRATCH / "setupcost_magicc_v5.onnx"

    trials = []
    for i in range(REPEATS):
        if dest.exists():
            dest.unlink()
        t0 = time.time()
        r = subprocess.run(["curl", "-sSL", "-o", str(dest), MODEL_URL],
                           capture_output=True, text=True)
        dt = time.time() - t0
        ok = r.returncode == 0 and dest.exists()
        size = dest.stat().st_size if ok else 0
        trials.append({"repeat": i + 1, "seconds": round(dt, 2),
                       "bytes": size, "ok": ok,
                       "MB_per_s": round(size / 1e6 / dt, 1) if dt and size else None,
                       "stderr": r.stderr[:300] if r.returncode else ""})
        print(f"  repeat {i+1}: {dt:.2f} s, {size/1e6:.1f} MB, "
              f"{size/1e6/dt:.1f} MB/s" if size else f"  repeat {i+1}: FAILED")

    good = [t for t in trials if t["ok"]]
    thr = sorted(t["MB_per_s"] for t in good) if good else []
    median_thr = thr[len(thr) // 2] if thr else None

    tco = json.loads((SPEED / "tco.json").read_text())
    proj = {}
    for name, t in tco["tools"].items():
        b = t.get("reference_data_bytes") or 0
        proj[name] = {
            "download_bytes": b,
            "download_GB": round(b / 1e9, 3),
            "projected_seconds_at_measured_throughput":
                round(b / 1e6 / median_thr, 1) if median_thr else None,
            "projected_minutes":
                round(b / 1e6 / median_thr / 60, 1) if median_thr else None,
        }

    rec = {
        "magicc_model_download": {
            "url": MODEL_URL,
            "measured": True,
            "trials": trials,
            "median_seconds": sorted(t["seconds"] for t in good)[len(good) // 2] if good else None,
            "median_MB_per_s": median_thr,
            "note": "This is MAGICC's ENTIRE first-run setup: the CLI auto-downloads the "
                    "ONNX model once to ~/.magicc/ and there is no database to fetch, "
                    "build or index.",
        },
        "competitor_downloads": {
            "measured": False,
            "reason": "Databases are already installed; re-downloading would overwrite a "
                      "working installation (the environments must not be modified) and "
                      "the seconds would measure this site's network, not the tool. "
                      "Volumes are exact; times are projections at the throughput measured "
                      "above and are labelled as such.",
            "projections": proj,
        },
        "one_time_local_setup_beyond_download": {
            "MAGICC": "none",
            "CheckM2": "none beyond the download (the .dmnd is shipped pre-formatted)",
            "CoCoPyE": "`cocopye toolbox download-dependencies` fetches and unpacks the "
                       "UProC binaries, two Pfam databases and the CoCoPyE database; "
                       "unpacking cost not separately measured",
            "DeepCheck": "no installer; the repository is cloned and the model is used "
                         "in place, but a complete CheckM2 install is a hard prerequisite",
        },
    }
    (SPEED / "setup_cost.json").write_text(json.dumps(rec, indent=2))
    if dest.exists():
        dest.unlink()
    print(f"\nmedian throughput {median_thr} MB/s")
    for k, v in proj.items():
        print(f"  {k:<22} {v['download_GB']:>7.3f} GB  "
              f"-> ~{v['projected_minutes']} min (projected)")
    print(f"wrote {SPEED/'setup_cost.json'}")


if __name__ == "__main__":
    main()
