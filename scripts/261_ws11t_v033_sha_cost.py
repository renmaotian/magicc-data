#!/usr/bin/env python3
"""
WS11.T / v0.3.3 — measure the SHA-256 pass over the frozen V5 model directly,
so the end-to-end delta between the two arms can be checked against a bottom-up
cost rather than merely asserted.

Uses magicc.cli's OWN _sha256_file / _verify_model, not a re-implementation.
Writes results/revision/speed_v033/model_verification_cost.tsv
"""
from __future__ import annotations

import statistics as st
import time
from pathlib import Path

from magicc import cli

OUT = Path("/path/to/magicc/results/revision/speed_v033")
N = 7

model = Path("/path/to/magicc/models/magicc_v5.onnx")
size = model.stat().st_size

# warm the page cache; the campaign runs warm too
cli._sha256_file(model)

hash_t, verify_t = [], []
for _ in range(N):
    t0 = time.perf_counter(); digest = cli._sha256_file(model); t1 = time.perf_counter()
    hash_t.append(t1 - t0)
    t0 = time.perf_counter(); cli._verify_model(model, "timing probe"); t1 = time.perf_counter()
    verify_t.append(t1 - t0)

assert digest == cli.MODEL_SHA256, "digest mismatch"

rows = [
    "measurement\tn\tmedian_s\tmin_s\tmax_s\tthroughput_MB_s\tnote",
    f"_sha256_file over the model\t{N}\t{st.median(hash_t):.4f}\t{min(hash_t):.4f}\t"
    f"{max(hash_t):.4f}\t{size/1e6/st.median(hash_t):.1f}\thashlib.sha256, 1 MiB chunks, warm page cache",
    f"_verify_model (hash + compare)\t{N}\t{st.median(verify_t):.4f}\t{min(verify_t):.4f}\t"
    f"{max(verify_t):.4f}\t{size/1e6/st.median(verify_t):.1f}\tthe exact call _ensure_model makes on every run",
    f"# model={model} size_bytes={size} sha256={cli.MODEL_SHA256}",
    f"# magicc version={cli.__dict__.get('__version__', 'see magicc.__version__')} "
    f"MODEL_URL={cli.MODEL_URL}",
]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "model_verification_cost.tsv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
