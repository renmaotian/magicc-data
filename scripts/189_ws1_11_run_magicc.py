#!/usr/bin/env python3
"""
WS1.11 -- run the FROZEN MAGICC V5 model on ``set_H_ncbi``.

Direct ONNX Runtime inference, following scripts/075_run_magicc_clean_cd.py (itself
following scripts/056_benchmark_v5.py) exactly:

    FASTA -> selected-9-mer counts -> 7 k-mer-summary features -> normalisation
          -> models/magicc_v5.onnx

The model is READ-ONLY.  WS1.11 is an evaluation-only workstream (author decision): a
retrain would also invalidate the V5-anchored WS1.6 / WS1.9 holdout experiments.  The
model's SHA256 is verified against the frozen value before inference and again after.

Output:
    data/benchmarks/set_H_ncbi/magicc_v5_predictions.tsv

Usage:
    python scripts/189_ws1_11_run_magicc.py [--workers 24] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path('/path/to/magicc')
sys.path.insert(0, str(PROJECT_DIR))

import onnxruntime as ort                                        # noqa: E402
from magicc.assembly_stats import compute_assembly_stats         # noqa: E402
from magicc.kmer_counter import KmerCounter                      # noqa: E402
from magicc.normalization import FeatureNormalizer               # noqa: E402

DATA_DIR = PROJECT_DIR / 'data'
SET_DIR = DATA_DIR / 'benchmarks' / 'set_H_ncbi'
SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')
FROZEN_SHA = 'b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096'

BATCH_SIZE = 64
ONNX_THREADS = 1


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_fasta_contigs(path):
    contigs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if cur:
                    contigs.append(''.join(cur).upper())
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        contigs.append(''.join(cur).upper())
    return [c for c in contigs if c]


_kc = None


def _init_worker(kmers_path):
    global _kc
    _kc = KmerCounter(kmers_path)
    _kc.count_contigs(['ACGTACGTACGTACGTACGT' * 50])


def _feat_worker(args):
    idx, path = args
    try:
        contigs = read_fasta_contigs(path)
        if not contigs:
            return idx, None, None, 'no_contigs'
        kc = _kc.count_contigs(contigs)
        asm = compute_assembly_stats(_kc.total_kmer_count(kc), kc)
        return idx, kc.astype(np.float32), asm.astype(np.float32), None
    except Exception as e:                                       # noqa: BLE001
        return idx, None, None, str(e)


def main() -> int:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    print('=' * 84)
    print('WS1.11 — MAGICC V5 on set_H_ncbi (evaluation only; model is read-only)')
    print('=' * 84)
    sha_before = sha256(ONNX_MODEL_PATH)
    print(f'  model : {ONNX_MODEL_PATH}')
    print(f'  sha256: {sha_before}')
    if sha_before != FROZEN_SHA:
        raise SystemExit('FATAL: models/magicc_v5.onnx does not match the frozen hash')
    print('  matches the frozen V5 hash: True')

    meta = pd.read_csv(SET_DIR / 'metadata.tsv', sep='\t')
    out = SET_DIR / 'magicc_v5_predictions.tsv'
    if out.exists() and not args.force:
        pred = pd.read_csv(out, sep='\t')
        if len(pred) == len(meta) and set(pred['genome_id']) == set(meta['genome_id']):
            print(f'  predictions already complete ({len(pred)}) — reusing')
            return 0

    so = ort.SessionOptions()
    so.intra_op_num_threads = ONNX_THREADS
    so.inter_op_num_threads = ONNX_THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, so,
                                   providers=['CPUExecutionProvider'])
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)

    work = [(i, str(SET_DIR / 'fasta' / f'{g}.fasta'))
            for i, g in enumerate(meta['genome_id'])
            if (SET_DIR / 'fasta' / f'{g}.fasta').exists()]
    if len(work) != len(meta):
        print(f'    WARNING: {len(meta) - len(work)} FASTA files missing')

    t0 = time.time()
    got = {}
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        for idx, kmer, asm, err in pool.imap_unordered(_feat_worker, work, chunksize=10):
            if err:
                print(f'    WARNING idx={idx}: {err}')
            else:
                got[idx] = (kmer, asm)
    feat_t = time.time() - t0
    idxs = sorted(got)
    n = len(idxs)
    print(f'    features : {feat_t:.1f}s ({feat_t / max(1, n) * 1000:.2f} ms/genome, '
          f'{args.workers} workers)')

    kn = normalizer.normalize_kmer(
        np.stack([got[i][0] for i in idxs])).astype(np.float32)
    an = normalizer.normalize_assembly(
        np.stack([got[i][1] for i in idxs])).astype(np.float32)
    in_names = [i.name for i in session.get_inputs()]
    out_name = session.get_outputs()[0].name
    preds = np.zeros((n, 2), dtype=np.float32)
    t1 = time.time()
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        preds[s:e] = session.run([out_name],
                                 {in_names[0]: kn[s:e], in_names[1]: an[s:e]})[0]
    infer_t = time.time() - t1
    print(f'    inference: {infer_t:.2f}s ({infer_t / max(1, n) * 1000:.2f} ms/genome)')

    res = meta.iloc[idxs].copy().reset_index(drop=True)
    res['pred_completeness'] = preds[:, 0]
    res['pred_contamination'] = preds[:, 1]
    res['wall_clock_s'] = feat_t + infer_t
    res['n_threads'] = args.workers
    res['model'] = 'magicc_v5.onnx'
    res['model_sha256'] = sha_before
    tmp = out.with_suffix('.tsv.tmp')
    res.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, out)
    print(f'    wrote {out} ({len(res)} rows)')

    sha_after = sha256(ONNX_MODEL_PATH)
    print(f'  model sha256 unchanged after the run: {sha_after == sha_before}')
    return 0 if sha_after == sha_before else 1


if __name__ == '__main__':
    sys.exit(main())
