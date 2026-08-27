#!/usr/bin/env python3
"""
WS3.8 (R1-M3) — run the frozen MAGICC V5 model on the CAMI II bin sets.

Inference is direct ONNX Runtime, following scripts/75_run_magicc_clean_cd.py (which in
turn follows scripts/56_benchmark_v5.py) exactly:

    FASTA -> selected-9-mer counts -> 7 k-mer-summary features -> normalisation
          -> models/magicc_v5.onnx

The `magicc` console script is deliberately NOT used: the installed package in env
`magicc2` is a pre-revision non-editable build (see the Stage-A follow-up note), so the
frozen ONNX graph is driven directly. The model file is never written to; its SHA256 is
recorded in the output.

Resumable: already-scored bins are skipped unless --force.

Usage:
    python scripts/198_ws38_run_magicc_cami2.py --dataset strain_madness --binset gold
    python scripts/198_ws38_run_magicc_cami2.py --dataset marine --binset mixed
"""

import argparse
import hashlib
import os
import signal
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import onnxruntime as ort  # noqa: E402
from magicc.assembly_stats import compute_assembly_stats  # noqa: E402
from magicc.kmer_counter import KmerCounter  # noqa: E402
from magicc.normalization import FeatureNormalizer  # noqa: E402

DATA_DIR = PROJECT_DIR / 'data'
CAMI_BINS = DATA_DIR / 'real_data' / 'cami2' / 'bins'
RES_DIR = PROJECT_DIR / 'results' / 'revision' / 'cami2'
PRED_DIR = RES_DIR / 'predictions'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')
FROZEN_SHA256 = 'b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096'

N_WORKERS = 24
BATCH_SIZE = 64
ONNX_THREADS = 1


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
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
    except Exception as e:                                            # noqa: BLE001
        return idx, None, None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--binset', required=True, choices=['gold', 'mixed'])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--workers', type=int, default=N_WORKERS)
    args = ap.parse_args()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    bdir = CAMI_BINS / args.dataset / args.binset
    files = sorted(bdir.glob('*.fna'))
    if not files:
        raise SystemExit(f'no FASTAs in {bdir}')
    out = PRED_DIR / f'{args.dataset}_{args.binset}_magicc_v5.tsv'

    got_sha = sha256_file(ONNX_MODEL_PATH)
    if got_sha != FROZEN_SHA256:
        raise SystemExit(f'MODEL SHA256 MISMATCH: {got_sha} != {FROZEN_SHA256}')
    print(f'== MAGICC V5 on CAMI II {args.dataset}/{args.binset}: {len(files)} bins ==',
          flush=True)
    print(f'   model sha256 verified {got_sha}', flush=True)

    done = set()
    if out.exists() and not args.force:
        prev = pd.read_csv(out, sep='\t')
        done = set(prev['bin_id'])
        print(f'   resuming: {len(done)} already scored', flush=True)
    work = [(i, str(p)) for i, p in enumerate(files) if p.stem not in done]
    if not work:
        print('   nothing to do', flush=True)
        return
    names = [Path(p).stem for _, p in work]

    so = ort.SessionOptions()
    so.intra_op_num_threads = ONNX_THREADS
    so.inter_op_num_threads = ONNX_THREADS
    session = ort.InferenceSession(ONNX_MODEL_PATH, so, providers=['CPUExecutionProvider'])
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)   # classmethod: sets finalized

    t0 = time.time()
    store = {}
    errs = []
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        for n_done, (idx, kmer, asm, err) in enumerate(
                pool.imap_unordered(_feat_worker, work, chunksize=8), 1):
            if err:
                errs.append((idx, err))
            else:
                store[idx] = (kmer, asm)
            if n_done % 500 == 0:
                print(f'   features {n_done}/{len(work)} '
                      f'({time.time()-t0:.0f}s)', flush=True)
    feat_t = time.time() - t0
    idxs = sorted(store)
    n = len(idxs)
    print(f'   features: {feat_t:.1f}s for {n} bins '
          f'({feat_t/max(1,n)*1000:.1f} ms/bin, {args.workers} workers); '
          f'{len(errs)} errors', flush=True)

    kn = normalizer.normalize_kmer(np.stack([store[i][0] for i in idxs])).astype(np.float32)
    an = normalizer.normalize_assembly(
        np.stack([store[i][1] for i in idxs])).astype(np.float32)
    in_names = [i.name for i in session.get_inputs()]
    out_name = session.get_outputs()[0].name
    preds = np.zeros((n, 2), dtype=np.float32)
    t1 = time.time()
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        preds[s:e] = session.run([out_name],
                                 {in_names[0]: kn[s:e], in_names[1]: an[s:e]})[0]
    infer_t = time.time() - t1
    print(f'   inference: {infer_t:.2f}s ({infer_t/max(1,n)*1000:.2f} ms/bin)', flush=True)

    idx2path = dict(work)
    res = pd.DataFrame({
        'bin_id': [Path(idx2path[i]).stem for i in idxs],
        'pred_completeness': preds[:, 0],
        'pred_contamination': preds[:, 1],
    })
    res['dataset'] = args.dataset
    res['binset'] = args.binset
    res['tool'] = 'MAGICC_V5'
    res['model_sha256'] = got_sha
    res['run_utc'] = datetime.now(timezone.utc).isoformat()
    if done and out.exists():
        res = pd.concat([pd.read_csv(out, sep='\t'), res], ignore_index=True)
    tmp = out.with_suffix('.tsv.tmp')
    res.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, out)
    print(f'   wrote {out} ({len(res)} rows)', flush=True)
    if errs:
        (PRED_DIR / f'{args.dataset}_{args.binset}_magicc_errors.txt').write_text(
            '\n'.join(f'{i}\t{e}' for i, e in errs) + '\n')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
