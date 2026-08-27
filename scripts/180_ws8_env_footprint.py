#!/usr/bin/env python3
"""
WS8.2 (R1-m12) CORRECTION — measure the install footprint of the RELEASED tool,
not the footprint of the analysis environment used to write the paper.

WHAT WAS WRONG
--------------
tco.json charged MAGICC, CoCoPyE and DeepCheck the same 8.182 GB / 65-package
figure. That number is `/path/to/anaconda3/envs/magicc2`, a shared
data-analysis environment that contains torch+cu121, lightgbm, xgboost, optuna,
scikit-learn, pandas, matplotlib, seaborn -- and `cocopye 0.5.0`, a competitor
tool. Three tools were being charged one shared environment, so the per-tool
number was meaningless. MAGICC's released inference path needs none of it:
pyproject.toml declares five runtime dependencies (numpy, numba, scipy, h5py,
onnxruntime) and the inference path is CPU ONNX Runtime; PyTorch is a
training-time dependency only.

WHAT THIS SCRIPT MEASURES
-------------------------
1. A throwaway conda environment containing ONLY python + magicc + its five
   declared runtime dependencies, built from the released package, measured with
   `du -sb` (apparent bytes) and then deleted. This is the honest install cost of
   the released tool.
2. The existing environments as they are on this host, reported for exactly what
   they are (an analysis environment; a dedicated CheckM2 environment).
3. The pinned container images from WS7, noting that they include a base-OS
   layer and so are not directly comparable to a bare conda environment.

For CoCoPyE and DeepCheck the environment size is NOT separable on this host --
both are installed inside the shared analysis environment. That is reported as
"not separately measurable" rather than as a number known to be wrong.

Writes results/revision/speed/env_footprint.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path("/path/to/magicc")
SPEED = PROJECT / "results" / "revision" / "speed"
CONDA = Path("/path/to/anaconda3/bin/conda")
PROBE = Path("/path/to/magicc/.tmp_env_probe/magicc_minimal")


def du_bytes(p: Path) -> int | None:
    if not p.exists():
        return None
    out = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return int(out.stdout.split()[0])


def n_conda_pkgs(env: Path) -> int | None:
    d = env / "conda-meta"
    return len(list(d.glob("*.json"))) if d.is_dir() else None


def n_pip_pkgs(env: Path) -> int | None:
    pip = env / "bin" / "pip"
    if not pip.exists():
        return None
    out = subprocess.run([str(pip), "list", "--format=json"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return len(json.loads(out.stdout))


def pip_list(env: Path) -> list[dict]:
    pip = env / "bin" / "pip"
    out = subprocess.run([str(pip), "list", "--format=json"], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else []


def build_probe() -> dict:
    if PROBE.exists():
        shutil.rmtree(PROBE, ignore_errors=True)
    PROBE.parent.mkdir(parents=True, exist_ok=True)
    log = {}
    t0 = time.time()
    r = subprocess.run([str(CONDA), "create", "-y", "-p", str(PROBE),
                        "python=3.11", "pip", "-q"],
                       capture_output=True, text=True)
    log["conda_create_rc"] = r.returncode
    log["conda_create_s"] = round(time.time() - t0, 1)
    if r.returncode != 0:
        log["conda_create_stderr"] = r.stderr[-2000:]
        return log
    log["bytes_python_only"] = du_bytes(PROBE)
    log["n_conda_pkgs_python_only"] = n_conda_pkgs(PROBE)

    t1 = time.time()
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1")
    r = subprocess.run([str(PROBE / "bin" / "pip"), "install", "--no-cache-dir",
                        "-q", str(PROJECT)],
                       capture_output=True, text=True, env=env)
    log["pip_install_rc"] = r.returncode
    log["pip_install_s"] = round(time.time() - t1, 1)
    if r.returncode != 0:
        log["pip_install_stderr"] = r.stderr[-3000:]
        return log

    log["bytes_total"] = du_bytes(PROBE)
    log["bytes_added_by_magicc_and_deps"] = (
        log["bytes_total"] - log["bytes_python_only"]
        if log.get("bytes_python_only") else None)
    log["n_pip_pkgs"] = n_pip_pkgs(PROBE)
    log["pip_packages"] = sorted(f"{p['name']}=={p['version']}" for p in pip_list(PROBE))

    sp = next((PROBE / "lib").glob("python3.*/site-packages"), None)
    if sp:
        parts = {}
        for name in ["numpy", "numba", "scipy", "h5py", "onnxruntime", "magicc",
                     "llvmlite", "pip", "setuptools"]:
            b = du_bytes(sp / name)
            if b:
                parts[name] = b
        log["site_packages_breakdown_bytes"] = dict(
            sorted(parts.items(), key=lambda kv: -kv[1]))

    # prove it actually runs
    r = subprocess.run([str(PROBE / "bin" / "magicc"), "--help"],
                       capture_output=True, text=True)
    log["magicc_help_rc"] = r.returncode
    r = subprocess.run([str(PROBE / "bin" / "python"), "-c",
                        "import magicc, onnxruntime, numba, h5py, scipy, numpy; "
                        "print(magicc.__version__ if hasattr(magicc,'__version__') else 'ok')"],
                       capture_output=True, text=True)
    log["import_check_rc"] = r.returncode
    log["import_check_stdout"] = r.stdout.strip()
    return log


def main() -> None:
    keep = "--keep-probe" in sys.argv
    probe = build_probe()

    envs = {}
    for name, path in [("magicc2 (shared analysis environment)",
                        Path("/path/to/anaconda3/envs/magicc2")),
                       ("checkm2_py39 (dedicated CheckM2 environment)",
                        Path("/path/to/anaconda3/envs/checkm2_py39"))]:
        envs[name] = {
            "path": str(path),
            "bytes": du_bytes(path),
            "n_conda_packages": n_conda_pkgs(path),
            "n_pip_packages": n_pip_pkgs(path),
        }
    shared = pip_list(Path("/path/to/anaconda3/envs/magicc2"))
    envs["magicc2 (shared analysis environment)"]["contains_non_magicc_packages"] = [
        f"{p['name']}=={p['version']}" for p in shared
        if p["name"].lower() in {"cocopye", "torch", "lightgbm", "xgboost", "optuna",
                                 "scikit-learn", "pandas", "matplotlib", "seaborn",
                                 "tensorflow", "keras"}]

    tco = json.loads((SPEED / "tco.json").read_text())
    mm = tco.get("magicc_minimal_runtime", {})

    rep = {
        "measured_on": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why": "The environment-size column of tco.json charged MAGICC, CoCoPyE and "
               "DeepCheck the same shared analysis environment. This file replaces it.",
        "distinction": {
            "analysis_environment": "The conda environment used to DEVELOP this paper "
                                    "(training, plotting, running competitors). It is not "
                                    "what a user installs and must never be reported as a "
                                    "tool's install cost.",
            "install_footprint": "What a user installs to RUN the released tool: the "
                                 "package plus its declared runtime dependencies.",
        },
        "magicc_declared_runtime_dependencies": [
            "numpy>=1.20", "numba>=0.53", "scipy>=1.7", "h5py>=3.0", "onnxruntime>=1.10"],
        "magicc_minimal_env_probe": probe,
        "environments_on_this_host": envs,
        "containers_from_WS7": {
            "docker_image_tag": mm.get("docker_image_tag"),
            "docker_image_bytes": mm.get("docker_image_size_bytes"),
            "apptainer_sif_bytes": mm.get("apptainer_sif_bytes"),
            "n_pinned_pip_packages": mm.get("n_pinned_pip_packages"),
            "caveat": "Container sizes include a base-OS layer and the bundled ONNX model, "
                      "so they are NOT directly comparable with a bare conda environment; "
                      "they are the most reproducible figure because every layer is pinned.",
        },
        "not_separately_measurable": {
            "CoCoPyE 0.5.0": "installed into the shared analysis environment on this host; "
                             "its environment size cannot be isolated without a clean "
                             "reinstall, so no number is reported. Its reference data "
                             "(17.045 GB) is exact and is reported.",
            "DeepCheck": "no installer; run from a git checkout inside the shared analysis "
                         "environment, and it additionally requires a complete CheckM2 "
                         "installation to produce its input. No isolated environment size "
                         "is reported. Its model files (0.090 GB) are exact.",
        },
    }
    (SPEED / "env_footprint.json").write_text(json.dumps(rep, indent=1) + "\n")

    if not keep:
        shutil.rmtree(PROBE.parent, ignore_errors=True)
        rep["magicc_minimal_env_probe"]["probe_deleted_after_measurement"] = True
        (SPEED / "env_footprint.json").write_text(json.dumps(rep, indent=1) + "\n")

    b = probe.get("bytes_total")
    print(f"minimal MAGICC env: {b/1e9:.3f} GB "
          f"({probe.get('n_pip_pkgs')} pip packages, install rc="
          f"{probe.get('pip_install_rc')}, magicc --help rc={probe.get('magicc_help_rc')})"
          if b else f"probe FAILED: {probe}")
    print(f"wrote {SPEED/'env_footprint.json'}")


if __name__ == "__main__":
    main()
