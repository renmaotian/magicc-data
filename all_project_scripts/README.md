# Legacy project scripts (snapshot, scripts 01–44)

A snapshot of the MAGICC pipeline as it stood **before** the 2026 revision:
reference curation, k-mer selection, synthesis, training, ONNX export, the
original benchmark generation and the original benchmarking runs.

**This directory is retained for continuity, not as the current analysis
code.** It predates the revision and therefore contains none of the work that
followed: the clean set C/D rebuild, the provenance audit, the taxonomic
holdout retraining, sets F, G and H, the CAMI II benchmark, the real-data
cohorts, the GUNC comparator or the matched-hardware timing campaign. It also
contains `25_benchmark_generate.py`, the generator whose train+val+test
sampling produced the **withdrawn** sets C and D — see [`../withdrawn/`](../withdrawn/).

For the scripts that generated the datasets actually shipped here, see
[`../data_generating_scripts/`](../data_generating_scripts/). For the complete
current analysis code, see the paper's Code availability statement.

Paths inside these scripts appear as `/path/to/magicc/...` and must be
repointed at a local checkout before they will run.
