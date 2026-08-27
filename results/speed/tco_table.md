# Table S4e — total cost of ownership (R1-m12)

Two different quantities are reported here and are never added together silently: the **reference data** a user must download before the tool can run at all, and the **install footprint** of the software itself. Sizes are apparent bytes (`du -sb`) measured on the benchmark host on 2026-08-01T00:59:45Z.

| Tool | Reference data (must download) | First-run download | Install footprint | Total before first run |
|---|---:|---:|---:|---:|
| MAGICC v0.3.0 (V5) | **0.172 GB** | 7 s | 0.75 GB | 0.92 GB |
| CheckM2 1.0.1 | **3.083 GB** | 2.0 min *(projected)* | 2.77 GB | 5.85 GB |
| CoCoPyE 0.5.0 | **17.045 GB** | 11.2 min *(projected)* | **not separately measurable** | — |
| DeepCheck | **0.090 GB** | 4 s *(projected)* | **not separately measurable** | — |

**Download times.** MAGICC's is **measured**: the released CLI's own model URL fetched three times, median 6.7 s at 25.3 MB/s. The competitor times are **projections** of their exact download volumes at that same measured throughput, not measurements: their databases were already installed and re-downloading them would have overwritten a working installation and measured this site's network rather than the tool.

**How the install-footprint figures were obtained, and what they include.**

- **MAGICC v0.3.0 (V5)** — MEASURED: throwaway conda env containing only python 3.11 + magicc + its 5 declared runtime dependencies (13 pip packages in total), built from the released package, `du -sb`, then deleted. 0.241 GB of that is the bare CPython interpreter; MAGICC and its dependencies add 0.508 GB. Verified runnable (`magicc --help` rc=0, `import magicc` -> 0.3.0).
- **CheckM2 1.0.1** — MEASURED: `checkm2_py39` is a dedicated environment for this tool alone (29 conda / 58 pip packages), `du -sb`.
- **CoCoPyE 0.5.0** — NOT SEPARATELY MEASURABLE on this host: installed into the shared analysis environment, so its software footprint cannot be isolated without a clean reinstall. No number is reported rather than a wrong one. Its reference data is exact and is reported.
- **DeepCheck** — NOT SEPARATELY MEASURABLE on this host: no installer; run from a git checkout inside the shared analysis environment. It additionally requires a complete CheckM2 installation (environment + 3.083 GB database) to produce its input, so its true footprint is at least CheckM2's.

**Footnote — the analysis environment is not the install footprint.** The conda environment used to *develop* this paper (`/path/to/conda/envs/magicc2`, 8.18 GB, 65 conda / 133 pip packages) carries the training and plotting stack — `cocopye==0.5.0`, `lightgbm==4.6.0`, `matplotlib==3.10.8`, `optuna==4.7.0`, `pandas==3.0.0`, `scikit-learn==1.3.1`, `torch==2.5.1+cu121`, `xgboost==3.1.3` — and MAGICC, CoCoPyE and DeepCheck were all run from it. It is therefore **one shared environment, not three tool footprints**, and an earlier version of this table wrongly charged its size to each of the three. None of those packages is required to run MAGICC: `pyproject.toml` declares five runtime dependencies (`numpy>=1.20`, `numba>=0.53`, `scipy>=1.7`, `h5py>=3.0`, `onnxruntime>=1.10`) and the released inference path is CPU ONNX Runtime. PyTorch is a training-time dependency only.

**Containers (WS7).** The most reproducible install figure is the pinned image: Docker `magicc:0.3.0` is **1.06 GB** and the Apptainer `.sif` is **0.38 GB**, built from 13 fully pinned packages with the ONNX model bundled. Container sizes include a base-OS layer and the bundled ONNX model, so they are NOT directly comparable with a bare conda environment; they are the most reproducible figure because every layer is pinned. The conda probe above resolves dependencies to their *current* versions, so it measures the install cost of the released package rather than the exact versions used for the timings; the container is the version-exact artefact.

**Cold versus warm cache** is Table S4e. **Reference-data sizes are unaffected by this correction** and stand as measured.

### The honest summary

MAGICC's reference data is **0.172 GB** — 99× smaller than CoCoPyE's and 18× smaller than CheckM2's — and it is a single ONNX model plus a k-mer list, not a sequence database. Its install footprint (0.75 GB, of which 0.24 GB is the Python interpreter itself) is smaller than CheckM2's dedicated environment (2.77 GB), but the two are not a like-for-like comparison of packaging quality: CheckM2's environment is a conda environment with 29 conda packages including its own Python, and MAGICC's was built by pip into a bare interpreter. CoCoPyE's and DeepCheck's software footprints are simply not known on this host and no number is invented for them.
