#!/usr/bin/env python3
"""
WS3 Track A — run the frozen MAGICC V5 model over an arbitrary directory of FASTA files.

Inference is direct ONNX Runtime and follows scripts/075_run_magicc_clean_cd.py exactly:
    FASTA -> selected-9-mer counts -> 7 k-mer-summary features -> normalisation
          -> models/magicc_v5.onnx
The `magicc` CLI is deliberately not used (it is edited concurrently by other agents).

Resumable: an existing output TSV covering all inputs is reused unless --force.

Usage:
    python scripts/138_run_magicc_v5_dir.py --input-dir DIR --output OUT.tsv \
        [--extension .fasta] [--workers 12]
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime as ort  # noqa: E402
from magicc.assembly_stats import compute_assembly_stats  # noqa: E402
from magicc.kmer_counter import KmerCounter  # noqa: E402
from magicc.normalization import FeatureNormalizer  # noqa: E402

SELECTED_KMERS = str(ROOT / "data" / "kmer_selection" / "selected_kmers.txt")
NORM_PARAMS = str(ROOT / "data" / "features" / "normalization_params.json")
ONNX_MODEL = str(ROOT / "models" / "magicc_v5.onnx")
BATCH = 64

_kc = None


def _init(kmers_path):
    global _kc
    _kc = KmerCounter(kmers_path)
    _kc.count_contigs(["ACGTACGTACGTACGTACGT" * 50])


def _read_contigs(path):
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    contigs, cur = [], []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    contigs.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        contigs.append("".join(cur).upper())
    return [c for c in contigs if c]


def _feat(args):
    idx, path = args
    try:
        contigs = _read_contigs(path)
        if not contigs:
            return idx, None, None, "no_contigs"
        kc = _kc.count_contigs(contigs)
        asm = compute_assembly_stats(_kc.total_kmer_count(kc), kc)
        return idx, kc.astype(np.float32), asm.astype(np.float32), None
    except Exception as e:  # noqa: BLE001
        return idx, None, None, str(e)


def run_dir(paths, ids, workers=12, quiet=False):
    """paths: list of FASTA paths; ids: matching genome ids. Returns DataFrame."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL, so, providers=["CPUExecutionProvider"])
    normalizer = FeatureNormalizer.load(NORM_PARAMS)

    work = list(enumerate(map(str, paths)))
    t0 = time.time()
    got, errs = {}, []
    with Pool(processes=workers, initializer=_init, initargs=(SELECTED_KMERS,)) as pool:
        for idx, kmer, asm, err in pool.imap_unordered(_feat, work, chunksize=4):
            if err:
                errs.append((ids[idx], err))
            else:
                got[idx] = (kmer, asm)
    feat_t = time.time() - t0
    idxs = sorted(got)
    n = len(idxs)
    if n == 0:
        raise SystemExit("no genomes produced features")
    kn = normalizer.normalize_kmer(np.stack([got[i][0] for i in idxs])).astype(np.float32)
    an = normalizer.normalize_assembly(
        np.stack([got[i][1] for i in idxs])).astype(np.float32)
    in_names = [i.name for i in session.get_inputs()]
    out_name = session.get_outputs()[0].name
    preds = np.zeros((n, 2), dtype=np.float32)
    t1 = time.time()
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        preds[s:e] = session.run([out_name],
                                 {in_names[0]: kn[s:e], in_names[1]: an[s:e]})[0]
    infer_t = time.time() - t1
    if not quiet:
        print(f"    features {feat_t:.1f}s ({feat_t/max(1,n)*1000:.0f} ms/genome, "
              f"{workers} workers); inference {infer_t:.2f}s; {len(errs)} errors")
        for gid, e in errs[:5]:
            print(f"      ERROR {gid}: {e}")
    return pd.DataFrame({
        "genome_id": [ids[i] for i in idxs],
        "magicc_completeness": preds[:, 0],
        "magicc_contamination": preds[:, 1],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--extension", default=".fasta")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    d = Path(args.input_dir)
    paths = sorted(d.glob(f"*{args.extension}"))
    ids = [p.name[: -len(args.extension)] for p in paths]
    print(f"MAGICC V5 on {len(paths)} genomes in {d}")
    out = Path(args.output)
    if out.exists() and not args.force:
        prev = pd.read_csv(out, sep="\t")
        if set(prev["genome_id"]) >= set(ids):
            print(f"  predictions already complete ({len(prev)}) — reuse")
            return
    out.parent.mkdir(parents=True, exist_ok=True)
    df = run_dir(paths, ids, workers=args.workers)
    df["model"] = "magicc_v5.onnx"
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, out)
    print(f"  wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
