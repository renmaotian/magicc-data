#!/usr/bin/env python3
"""
WS8.2 (R1-m12) — total cost of ownership: database size, installed environment
size, first-run setup/download time, and cold vs warm page cache.

Everything here is measured on this host, from the installed artefacts. Where a
number could not be measured it is recorded as null with the reason, never
estimated silently.

Writes results/revision/speed/tco.json and tco_table.tsv
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROJECT = Path("/path/to/magicc")
OUT = PROJECT / "results" / "revision" / "speed"
ENVS = Path("/path/to/conda/envs")


def du_bytes(path: Path) -> int | None:
    """Apparent size on disk in bytes (du -sb), or None if absent."""
    if not path.exists():
        return None
    r = subprocess.run(["du", "-sb", "--", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return int(r.stdout.split()[0])


def gb(n: int | None) -> float | None:
    return None if n is None else round(n / 1e9, 3)


def conda_pkg_count(env: Path) -> int | None:
    meta = env / "conda-meta"
    if not meta.exists():
        return None
    return len(list(meta.glob("*.json")))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    checkm2_db = PROJECT / "tools" / "checkm2_db" / "CheckM2_database"
    cocopye_share = Path("/path/to/home/.local/share/cocopye")
    magicc_model = PROJECT / "models" / "magicc_v5.onnx"
    magicc_kmers = PROJECT / "magicc" / "data"
    deepcheck_dir = PROJECT / "tools" / "DeepCheck"

    tools: dict[str, dict] = {}

    # ------------------------------------------------------------- MAGICC
    tools["MAGICC v0.3.0 (V5)"] = {
        "reference_data_bytes": (du_bytes(magicc_model) or 0) + (du_bytes(magicc_kmers) or 0),
        "reference_data_detail": {
            "magicc_v5.onnx": du_bytes(magicc_model),
            "magicc/data (9-mer list + normalization params)": du_bytes(magicc_kmers),
        },
        "reference_data_note": "No sequence database of any kind. The 'database' is the "
                               "ONNX model plus the selected 9-mer list; it ships with the "
                               "package or auto-downloads once to ~/.magicc/.",
        "env_path": str(ENVS / "magicc2"),
        "env_bytes": du_bytes(ENVS / "magicc2"),
        "env_packages": conda_pkg_count(ENVS / "magicc2"),
        "env_note": "The magicc2 conda env on this host is a shared analysis environment "
                    "(it also carries PyTorch+CUDA for training and CoCoPyE). The size a "
                    "user actually installs is the pinned runtime environment, measured "
                    "separately below as 'minimal_runtime_env'.",
        "setup": {
            "database_download": "none",
            "model_download_bytes": du_bytes(magicc_model),
        },
    }

    # ------------------------------------------------------------ CheckM2
    tools["CheckM2 1.0.1"] = {
        "reference_data_bytes": du_bytes(checkm2_db),
        "reference_data_detail": {
            str(p.name): p.stat().st_size
            for p in sorted(checkm2_db.rglob("*")) if p.is_file()
        } if checkm2_db.exists() else {},
        "reference_data_note": "uniref100.KO.1.dmnd, a DIAMOND-formatted UniRef100 subset. "
                               "Downloaded once via `checkm2 database --download`.",
        "env_path": str(ENVS / "checkm2_py39"),
        "env_bytes": du_bytes(ENVS / "checkm2_py39"),
        "env_packages": conda_pkg_count(ENVS / "checkm2_py39"),
    }

    # ------------------------------------------------------------ CoCoPyE
    cocopye_detail = {}
    if cocopye_share.exists():
        for sub in sorted(p for p in cocopye_share.iterdir() if p.is_dir()):
            cocopye_detail[sub.name] = du_bytes(sub)
    tools["CoCoPyE 0.5.0"] = {
        "reference_data_bytes": du_bytes(cocopye_share),
        "reference_data_detail": cocopye_detail,
        "reference_data_note": "UProC Pfam databases (v24 + v28), UProC model, and the "
                               "CoCoPyE reference database, downloaded by "
                               "`cocopye toolbox download-dependencies`.",
        "env_path": str(ENVS / "magicc2"),
        "env_bytes": du_bytes(ENVS / "magicc2"),
        "env_packages": conda_pkg_count(ENVS / "magicc2"),
        "env_note": "Installed into the same conda env as MAGICC on this host, so the env "
                    "size is not separable; the reference data is.",
    }

    # ---------------------------------------------------------- DeepCheck
    tools["DeepCheck"] = {
        "reference_data_bytes": du_bytes(deepcheck_dir),
        "reference_data_detail": {
            "models/best_model.pt": du_bytes(deepcheck_dir / "models" / "best_model.pt"),
            "scaler_params.npz": du_bytes(deepcheck_dir / "scaler_params.npz"),
            "feature_names.pkl": du_bytes(deepcheck_dir / "feature_names.pkl"),
        },
        "reference_data_note": "DeepCheck ships only a model; it has no sequence database. "
                               "It cannot read FASTA at all: its input is CheckM2's pickled "
                               "feature vectors, so in practice its true installed footprint "
                               "is its own 86 MB PLUS the whole CheckM2 install and the "
                               "2.9 GB CheckM2 database.",
        "env_path": str(ENVS / "magicc2"),
        "env_bytes": du_bytes(ENVS / "magicc2"),
        "requires": "CheckM2 (environment + database + a full CheckM2 run per input set)",
    }

    # --------------------------------------- minimal pinned runtime for MAGICC
    lock = PROJECT / "docker" / "requirements-lock.txt"
    dockerfile = PROJECT / "docker" / "Dockerfile"
    sif = PROJECT / "containers" / "magicc_0.3.0.sif"
    minimal = {
        "docker_image_tag": "magicc:0.3.0",
        "docker_image_size_bytes": None,
        "apptainer_sif_bytes": du_bytes(sif),
        "pip_lock_file": str(lock) if lock.exists() else None,
        "n_pinned_pip_packages": (
            len([l for l in lock.read_text().splitlines()
                 if l.strip() and not l.startswith("#") and "==" in l])
            if lock.exists() else None),
        "dockerfile": str(dockerfile) if dockerfile.exists() else None,
    }
    try:
        r = subprocess.run(["docker", "image", "inspect", "magicc:0.3.0",
                            "--format", "{{.Size}}"], capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            minimal["docker_image_size_bytes"] = int(r.stdout.strip())
    except Exception:
        pass

    rec = {
        "measured_on": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                      capture_output=True, text=True).stdout.strip(),
        "host": os.uname().nodename,
        "filesystem": "/dev/sdb ext4 mounted at /path/to/data (3.6 TB); "
                      "conda envs and CoCoPyE data on the root filesystem",
        "tools": tools,
        "magicc_minimal_runtime": minimal,
        "cold_vs_warm_cache": {
            "method": "System-wide `echo 3 > /proc/sys/vm/drop_caches` requires root and "
                      "passwordless sudo is NOT configured on this host, so a true "
                      "system-wide cold cache could not be produced. THIS IS A STATED "
                      "LIMITATION. What was done instead is an unprivileged, file-scoped "
                      "eviction: posix_fadvise(POSIX_FADV_DONTNEED) applied to every input "
                      "FASTA and to the tool's model/database files, with residency "
                      "verified before and after by mincore(2) "
                      "(scripts/162_ws8_drop_file_cache.py). Everything the run reads from "
                      "those files therefore comes from disk again; kernel/libc pages and "
                      "the conda environment's shared objects remain cached.",
            "results": "see cold_vs_warm.tsv (filled by scripts/169)",
        },
        "first_run_setup_time": {
            "method": "Download and index times were NOT re-measured: re-downloading the "
                      "CheckM2 and CoCoPyE databases would consume bandwidth and disk for "
                      "artefacts already present, and the measured value would be a "
                      "property of this site's network rather than of the tool. What is "
                      "reported instead is the number of bytes that must be transferred "
                      "before first use, which is network-independent, together with the "
                      "wall-clock time to first result on an already-installed system "
                      "(the 'cold cache' column).",
        },
    }

    # ------------------------------------------------------------- flat table
    rows = []
    for name, t in tools.items():
        rows.append({
            "tool": name,
            "reference_data_GB": gb(t.get("reference_data_bytes")),
            "must_download_before_first_use_GB": gb(t.get("reference_data_bytes")),
            "conda_env_GB": gb(t.get("env_bytes")),
            "conda_env_packages": t.get("env_packages"),
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tco_table.tsv", sep="\t", index=False)
    (OUT / "tco.json").write_text(json.dumps(rec, indent=2))

    print(df.to_string(index=False))
    print(f"\nMAGICC minimal pinned runtime: docker image "
          f"{gb(minimal['docker_image_size_bytes'])} GB, "
          f"apptainer SIF {gb(minimal['apptainer_sif_bytes'])} GB, "
          f"{minimal['n_pinned_pip_packages']} pinned pip packages")
    print(f"wrote {OUT/'tco.json'}")


if __name__ == "__main__":
    main()
