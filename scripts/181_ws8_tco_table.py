#!/usr/bin/env python3
"""
WS8.2 (R1-m12) — total cost of ownership table, with the environment-size column
corrected.

Supersedes the `conda_env_GB` column written by scripts/168_ws8_tco.py, which
charged MAGICC, CoCoPyE and DeepCheck the same 8.182 GB: that is the shared
analysis environment used to develop this paper (it contains torch+cu121,
lightgbm, xgboost, optuna, scikit-learn, pandas, matplotlib -- and cocopye
0.5.0, a competitor tool). One shared environment being triple-counted is not a
per-tool figure. scripts/180_ws8_env_footprint.py measures instead what a user
installs to run the released tool, and this script tabulates it.

Two different quantities are reported and are never mixed:
  * REFERENCE DATA  -- what must be downloaded before first use (exact for all four)
  * INSTALL FOOTPRINT -- the software environment (isolated where measurable;
    "not separately measurable" where it is not)

Rewrites results/revision/speed/tco_table.tsv, writes tco_table.md, and amends
tco.json with the correction so the superseded field cannot be read naively.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SPEED = Path("/path/to/magicc/results/revision/speed")

ORDER = ["MAGICC v0.3.0 (V5)", "CheckM2 1.0.1", "CoCoPyE 0.5.0", "DeepCheck"]


def gb(b):
    return None if b is None else round(b / 1e9, 3)


def main() -> None:
    tco = json.loads((SPEED / "tco.json").read_text())
    env = json.loads((SPEED / "env_footprint.json").read_text())
    setup = json.loads((SPEED / "setup_cost.json").read_text())

    probe = env["magicc_minimal_env_probe"]
    envs = env["environments_on_this_host"]
    proj = setup["competitor_downloads"]["projections"]
    dl_mbs = setup["magicc_model_download"]["median_MB_per_s"]

    install = {
        "MAGICC v0.3.0 (V5)": {
            "install_footprint_GB": gb(probe.get("bytes_total")),
            "install_footprint_basis":
                f"MEASURED: throwaway conda env containing only python 3.11 + magicc + "
                f"its {len(env['magicc_declared_runtime_dependencies'])} declared runtime "
                f"dependencies ({probe.get('n_pip_pkgs')} pip packages in total), built "
                f"from the released package, `du -sb`, then deleted. "
                f"{gb(probe.get('bytes_python_only'))} GB of that is the bare CPython "
                f"interpreter; MAGICC and its dependencies add "
                f"{gb(probe.get('bytes_added_by_magicc_and_deps'))} GB. "
                f"Verified runnable (`magicc --help` rc=0, `import magicc` -> "
                f"{probe.get('import_check_stdout')}).",
        },
        "CheckM2 1.0.1": {
            "install_footprint_GB": gb(envs["checkm2_py39 (dedicated CheckM2 environment)"]["bytes"]),
            "install_footprint_basis":
                f"MEASURED: `checkm2_py39` is a dedicated environment for this tool alone "
                f"({envs['checkm2_py39 (dedicated CheckM2 environment)']['n_conda_packages']} "
                f"conda / {envs['checkm2_py39 (dedicated CheckM2 environment)']['n_pip_packages']} "
                f"pip packages), `du -sb`.",
        },
        "CoCoPyE 0.5.0": {
            "install_footprint_GB": None,
            "install_footprint_basis":
                "NOT SEPARATELY MEASURABLE on this host: installed into the shared "
                "analysis environment, so its software footprint cannot be isolated "
                "without a clean reinstall. No number is reported rather than a wrong one. "
                "Its reference data is exact and is reported.",
        },
        "DeepCheck": {
            "install_footprint_GB": None,
            "install_footprint_basis":
                "NOT SEPARATELY MEASURABLE on this host: no installer; run from a git "
                "checkout inside the shared analysis environment. It additionally requires "
                "a complete CheckM2 installation (environment + 3.083 GB database) to "
                "produce its input, so its true footprint is at least CheckM2's.",
        },
    }

    rows = []
    for name in ORDER:
        t = tco["tools"][name]
        rdb = t.get("reference_data_bytes")
        p = proj.get(name, {})
        rows.append({
            "tool": name,
            "reference_data_GB": gb(rdb),
            "must_download_before_first_use_GB": gb(rdb),
            "first_run_download_s_at_25.3_MB_s": p.get("projected_seconds_at_measured_throughput"),
            "first_run_download_measured": name.startswith("MAGICC"),
            "install_footprint_GB": install[name]["install_footprint_GB"],
            "install_footprint_basis": install[name]["install_footprint_basis"],
            "total_first_use_GB": (
                None if install[name]["install_footprint_GB"] is None
                else round(gb(rdb) + install[name]["install_footprint_GB"], 3)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(SPEED / "tco_table.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ markdown
    L, A = [], None
    A = L.append
    A("# Table S4e — total cost of ownership (R1-m12)")
    A("")
    A("Two different quantities are reported here and are never added together "
      "silently: the **reference data** a user must download before the tool can run at "
      "all, and the **install footprint** of the software itself. Sizes are apparent "
      "bytes (`du -sb`) measured on the benchmark host on "
      f"{env['measured_on']}.")
    A("")
    A("| Tool | Reference data (must download) | First-run download | Install footprint | Total before first run |")
    A("|---|---:|---:|---:|---:|")
    for r in rows:
        d = r["first_run_download_s_at_25.3_MB_s"]
        dtxt = ("—" if d is None else
                (f"{d:.0f} s" if d < 120 else f"{d/60:.1f} min"))
        if not r["first_run_download_measured"] and d is not None:
            dtxt += " *(projected)*"
        ifp = ("**not separately measurable**" if r["install_footprint_GB"] is None
               else f"{r['install_footprint_GB']:.2f} GB")
        tot = ("—" if r["total_first_use_GB"] is None else f"{r['total_first_use_GB']:.2f} GB")
        A(f"| {r['tool']} | **{r['reference_data_GB']:.3f} GB** | {dtxt} | {ifp} | {tot} |")
    A("")
    A(f"**Download times.** MAGICC's is **measured**: the released CLI's own model URL "
      f"fetched three times, median {setup['magicc_model_download']['median_seconds']:.1f} s "
      f"at {dl_mbs:.1f} MB/s. The competitor times are **projections** of their exact "
      f"download volumes at that same measured throughput, not measurements: their "
      f"databases were already installed and re-downloading them would have overwritten a "
      f"working installation and measured this site's network rather than the tool.")
    A("")
    A("**How the install-footprint figures were obtained, and what they include.**")
    A("")
    for name in ORDER:
        A(f"- **{name}** — {install[name]['install_footprint_basis']}")
    A("")
    A("**Footnote — the analysis environment is not the install footprint.** The conda "
      "environment used to *develop* this paper "
      f"(`{envs['magicc2 (shared analysis environment)']['path']}`, "
      f"{gb(envs['magicc2 (shared analysis environment)']['bytes']):.2f} GB, "
      f"{envs['magicc2 (shared analysis environment)']['n_conda_packages']} conda / "
      f"{envs['magicc2 (shared analysis environment)']['n_pip_packages']} pip packages) "
      "carries the training and plotting stack — "
      + ", ".join(f"`{p}`" for p in envs['magicc2 (shared analysis environment)']
                  ['contains_non_magicc_packages'])
      + " — and MAGICC, CoCoPyE and DeepCheck were all run from it. It is therefore **one "
        "shared environment, not three tool footprints**, and an earlier version of this "
        "table wrongly charged its size to each of the three. None of those packages is "
        "required to run MAGICC: `pyproject.toml` declares five runtime dependencies ("
      + ", ".join(f"`{d}`" for d in env["magicc_declared_runtime_dependencies"])
      + ") and the released inference path is CPU ONNX Runtime. PyTorch is a training-time "
        "dependency only.")
    A("")
    c = env["containers_from_WS7"]
    A(f"**Containers (WS7).** The most reproducible install figure is the pinned image: "
      f"Docker `{c['docker_image_tag']}` is **{gb(c['docker_image_bytes']):.2f} GB** and the "
      f"Apptainer `.sif` is **{gb(c['apptainer_sif_bytes']):.2f} GB**, built from "
      f"{c['n_pinned_pip_packages']} fully pinned packages with the ONNX model bundled. "
      f"{c['caveat']} The conda probe above resolves dependencies to their *current* "
      f"versions, so it measures the install cost of the released package rather than the "
      f"exact versions used for the timings; the container is the version-exact artefact.")
    A("")
    A("**Cold versus warm cache** is Table S4e. **Reference-data sizes are unaffected by "
      "this correction** and stand as measured.")
    A("")
    A("### The honest summary")
    A("")
    A(f"MAGICC's reference data is **{gb(tco['tools']['MAGICC v0.3.0 (V5)']['reference_data_bytes']):.3f} GB** "
      f"— {gb(tco['tools']['CoCoPyE 0.5.0']['reference_data_bytes'])/gb(tco['tools']['MAGICC v0.3.0 (V5)']['reference_data_bytes']):.0f}× "
      f"smaller than CoCoPyE's and "
      f"{gb(tco['tools']['CheckM2 1.0.1']['reference_data_bytes'])/gb(tco['tools']['MAGICC v0.3.0 (V5)']['reference_data_bytes']):.0f}× "
      f"smaller than CheckM2's — and it is a single ONNX model plus a k-mer list, not a "
      f"sequence database. Its install footprint "
      f"({install['MAGICC v0.3.0 (V5)']['install_footprint_GB']:.2f} GB, of which "
      f"{gb(probe.get('bytes_python_only')):.2f} GB is the Python interpreter itself) is "
      f"smaller than CheckM2's dedicated environment "
      f"({install['CheckM2 1.0.1']['install_footprint_GB']:.2f} GB), but the two are not a "
      f"like-for-like comparison of packaging quality: CheckM2's environment is a conda "
      f"environment with {envs['checkm2_py39 (dedicated CheckM2 environment)']['n_conda_packages']} "
      f"conda packages including its own Python, and MAGICC's was built by pip into a bare "
      f"interpreter. CoCoPyE's and DeepCheck's software footprints are simply not known on "
      f"this host and no number is invented for them.")
    (SPEED / "tco_table.md").write_text("\n".join(L) + "\n")

    # ------------------------------------------------------- amend tco.json
    for name, t in tco["tools"].items():
        if "env_bytes" in t:
            t["env_bytes_SUPERSEDED"] = t.pop("env_bytes")
            t["env_packages_SUPERSEDED"] = t.pop("env_packages", None)
            t["env_size_note"] = ("SUPERSEDED 2026-08-01. This was the size of the conda "
                                  "environment this tool happened to be installed in on "
                                  "this host; for MAGICC, CoCoPyE and DeepCheck that was "
                                  "one shared analysis environment, so the same number was "
                                  "charged to three tools. Use env_footprint.json / "
                                  "tco_table.tsv instead.")
    tco["env_size_correction"] = {
        "date": "2026-08-01",
        "problem": "conda_env_GB charged the 8.182 GB shared analysis environment to "
                   "MAGICC, CoCoPyE and DeepCheck alike. That environment contains "
                   "torch+cu121, lightgbm, xgboost, optuna, scikit-learn, pandas, "
                   "matplotlib and cocopye 0.5.0 (a competitor tool); none of it is "
                   "required to run MAGICC.",
        "replacement": "results/revision/speed/env_footprint.json (scripts/180) and the "
                       "install_footprint_GB column of tco_table.tsv (scripts/181).",
        "magicc_minimal_install_GB": gb(probe.get("bytes_total")),
        "direction_of_the_error": "The correction REDUCES MAGICC's reported footprint from "
                                  "8.18 GB to 0.75 GB, i.e. it favours MAGICC. It is made "
                                  "because the original figure measured the wrong object, "
                                  "and the method is stated in full so the claim can be "
                                  "checked.",
    }
    (SPEED / "tco.json").write_text(json.dumps(tco, indent=1) + "\n")

    print(df.to_string(index=False))
    print(f"\nwrote {SPEED/'tco_table.tsv'}, {SPEED/'tco_table.md'}; amended tco.json")


if __name__ == "__main__":
    main()
