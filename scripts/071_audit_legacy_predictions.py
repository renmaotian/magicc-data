#!/usr/bin/env python3
"""
WS0.2 — Legacy MAGICC-V5 prediction audit (+ fill-in of anything missing).

Every benchmark set under data/benchmarks/ that has a metadata.tsv is audited on:
  * n rows in metadata.tsv vs n FASTA files on disk
  * presence of magicc_v5_predictions.tsv
  * n rows in the V5 prediction file vs n rows in metadata.tsv
  * exact genome_id set equality between prediction file and metadata
  * label agreement: true_completeness / true_contamination identical (1e-6) in both
  * staleness: prediction file mtime older than metadata.tsv or than the model
  * which non-MAGICC competitor predictions exist (reported, NEVER regenerated)

Sets whose V5 predictions are missing / incomplete / stale are then (re)generated with
exactly the pipeline of scripts/56_benchmark_v5.py:
    FASTA -> selected-9-mer counts -> 7 k-mer-summary features -> normalisation
          -> models/magicc_v5.onnx  (direct ONNX Runtime inference; the CLI is NOT used)

Resumable: each set is written atomically and skipped on re-run once it passes the audit.

Outputs:
  results/revision/legacy_v5_prediction_audit.tsv
  results/revision/legacy_v5_metrics.tsv
  data/benchmarks/<set>/magicc_v5_predictions.tsv   (for sets that lacked them)

Usage:
    python scripts/71_audit_legacy_predictions.py            # audit + generate missing
    python scripts/71_audit_legacy_predictions.py --audit-only
"""

import argparse
import json
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
BENCHMARK_DIR = DATA_DIR / 'benchmarks'
RESULTS_DIR = PROJECT_DIR / 'results' / 'revision'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')

N_WORKERS = 20          # capped: other agents share this machine
BATCH_SIZE = 64
ONNX_THREADS = 1
PRED_FILE = 'magicc_v5_predictions.tsv'
COMPETITORS = ['checkm2_predictions.tsv', 'cocopye_predictions.tsv',
               'deepcheck_predictions.tsv']


# ---------------------------------------------------------------- FASTA I/O
def read_fasta_contigs(fasta_path):
    contigs, cur = [], []
    with open(fasta_path) as f:
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


_worker_kmer_counter = None


def _init_worker(kmers_path):
    global _worker_kmer_counter
    _worker_kmer_counter = KmerCounter(kmers_path)
    _worker_kmer_counter.count_contigs(["ACGTACGTACGTACGTACGT" * 50])


def _extract_features_worker(args):
    idx, fasta_path = args
    try:
        contigs = read_fasta_contigs(fasta_path)
        if not contigs:
            return idx, None, None, 'no_contigs'
        kc = _worker_kmer_counter.count_contigs(contigs)
        log10_total = _worker_kmer_counter.total_kmer_count(kc)
        asm = compute_assembly_stats(log10_total, kc)
        return idx, kc.astype(np.float32), asm.astype(np.float32), None
    except Exception as e:                                    # noqa: BLE001
        return idx, None, None, str(e)


# ------------------------------------------------------------------- audit
def discover_sets():
    """Every directory under data/benchmarks with a metadata.tsv, relative name."""
    out = []
    for meta in sorted(BENCHMARK_DIR.rglob('metadata.tsv')):
        d = meta.parent
        out.append((str(d.relative_to(BENCHMARK_DIR)), d))
    return out


def audit_set(name, d, model_mtime):
    meta_path = d / 'metadata.tsv'
    pred_path = d / PRED_FILE
    fasta_dir = d / 'fasta'

    meta = pd.read_csv(meta_path, sep='\t')
    n_meta = len(meta)
    n_fasta = len(list(fasta_dir.glob('*.fasta'))) if fasta_dir.is_dir() else 0
    n_fasta_expected_present = sum(
        1 for g in meta['genome_id'] if (fasta_dir / f'{g}.fasta').exists()
    ) if fasta_dir.is_dir() else 0

    row = {
        'set': name,
        'n_metadata': n_meta,
        'n_fasta_files': n_fasta,
        'n_metadata_fastas_present': n_fasta_expected_present,
        'v5_pred_exists': pred_path.exists(),
        'n_v5_rows': '',
        'genome_ids_match': '',
        'labels_match': '',
        'pred_newer_than_metadata': '',
        'pred_newer_than_model': '',
        'status': '',
        'competitors_present': ','.join(c.split('_')[0] for c in COMPETITORS
                                        if (d / c).exists()),
    }

    if not pred_path.exists():
        row['status'] = 'MISSING'
        return row, meta

    pred = pd.read_csv(pred_path, sep='\t')
    row['n_v5_rows'] = len(pred)
    ids_match = (set(pred['genome_id']) == set(meta['genome_id'])
                 and len(pred) == n_meta)
    row['genome_ids_match'] = bool(ids_match)

    labels_match = ''
    if ids_match and {'true_completeness', 'true_contamination'} <= set(pred.columns):
        a = meta.set_index('genome_id')[['true_completeness', 'true_contamination']]
        b = pred.set_index('genome_id')[['true_completeness', 'true_contamination']]
        labels_match = bool(np.allclose(a.loc[b.index].values, b.values, atol=1e-6))
    row['labels_match'] = labels_match

    pm, mm = pred_path.stat().st_mtime, meta_path.stat().st_mtime
    row['pred_newer_than_metadata'] = bool(pm >= mm)
    row['pred_newer_than_model'] = bool(pm >= model_mtime)

    if len(pred) != n_meta or not ids_match:
        row['status'] = 'INCOMPLETE'
    elif labels_match is False:
        row['status'] = 'LABEL_MISMATCH'
    elif not row['pred_newer_than_model']:
        row['status'] = 'STALE_VS_MODEL'
    else:
        row['status'] = 'OK'
    return row, meta


# -------------------------------------------------------------- generation
def run_v5_on_set(name, d, meta, session, normalizer):
    fasta_dir = d / 'fasta'
    out_path = d / PRED_FILE
    print(f'\n{"=" * 74}\n  Generating V5 predictions: {name} ({len(meta)} genomes)\n'
          f'{"=" * 74}')

    if not fasta_dir.is_dir():
        print('    SKIP: no fasta/ directory')
        return None

    work = []
    for i, g in enumerate(meta['genome_id']):
        p = fasta_dir / f'{g}.fasta'
        if p.exists():
            work.append((i, str(p)))
    if not work:
        print('    SKIP: no FASTA files found')
        return None
    if len(work) != len(meta):
        print(f'    WARNING: {len(meta) - len(work)} FASTA files missing')

    t0 = time.time()
    got = {}
    with Pool(processes=N_WORKERS, initializer=_init_worker,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        for idx, kmer, asm, err in pool.imap_unordered(
                _extract_features_worker, work, chunksize=10):
            if err:
                print(f'    WARNING idx={idx}: {err}')
            else:
                got[idx] = (kmer, asm)
    feat_t = time.time() - t0
    idxs = sorted(got)
    n = len(idxs)
    print(f'    features: {feat_t:.1f}s for {n} genomes '
          f'({feat_t / max(1, n) * 1000:.1f} ms/genome, {N_WORKERS} workers)')

    kmer_norm = normalizer.normalize_kmer(
        np.stack([got[i][0] for i in idxs])).astype(np.float32)
    asm_norm = normalizer.normalize_assembly(
        np.stack([got[i][1] for i in idxs])).astype(np.float32)

    in_names = [i.name for i in session.get_inputs()]
    out_name = session.get_outputs()[0].name
    preds = np.zeros((n, 2), dtype=np.float32)
    t1 = time.time()
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        preds[s:e] = session.run(
            [out_name], {in_names[0]: kmer_norm[s:e], in_names[1]: asm_norm[s:e]})[0]
    infer_t = time.time() - t1
    total_t = feat_t + infer_t
    print(f'    inference: {infer_t:.2f}s ({infer_t / max(1, n) * 1000:.2f} ms/genome)')

    res = meta.iloc[idxs].copy().reset_index(drop=True)
    res['pred_completeness'] = preds[:, 0]
    res['pred_contamination'] = preds[:, 1]
    res['wall_clock_s'] = total_t
    res['n_threads'] = N_WORKERS
    tmp = out_path.with_suffix('.tsv.tmp')
    res.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, out_path)
    print(f'    wrote {out_path}')
    return res


def metrics_for(name, pred):
    tc = pred['true_completeness'].values.astype(float)
    tx = pred['true_contamination'].values.astype(float)
    pc = pred['pred_completeness'].values.astype(float)
    px = pred['pred_contamination'].values.astype(float)

    def r2(t, p):
        if np.std(t) == 0 or np.std(p) == 0:
            return np.nan
        return float(np.corrcoef(t, p)[0, 1] ** 2)

    return {
        'set': name,
        'n': len(pred),
        'comp_mae': round(float(np.mean(np.abs(tc - pc))), 4),
        'comp_rmse': round(float(np.sqrt(np.mean((tc - pc) ** 2))), 4),
        'comp_bias': round(float(np.mean(pc - tc)), 4),
        'comp_r2': (round(r2(tc, pc), 4) if not np.isnan(r2(tc, pc)) else ''),
        'cont_mae': round(float(np.mean(np.abs(tx - px))), 4),
        'cont_rmse': round(float(np.sqrt(np.mean((tx - px) ** 2))), 4),
        'cont_bias': round(float(np.mean(px - tx)), 4),
        'cont_r2': (round(r2(tx, px), 4) if not np.isnan(r2(tx, px)) else ''),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_mtime = os.path.getmtime(ONNX_MODEL_PATH)
    print('=' * 78)
    print('WS0.2 — legacy MAGICC V5 prediction audit')
    print(f'model: {ONNX_MODEL_PATH}  (mtime '
          f'{datetime.fromtimestamp(model_mtime, timezone.utc).isoformat()})')
    print('=' * 78)

    sets = discover_sets()
    rows, metas = [], {}
    for name, d in sets:
        row, meta = audit_set(name, d, model_mtime)
        rows.append(row)
        metas[name] = (d, meta)

    audit = pd.DataFrame(rows)
    print('\nPASS 1 — audit before any generation')
    print(audit[['set', 'n_metadata', 'n_fasta_files', 'v5_pred_exists',
                 'n_v5_rows', 'status', 'competitors_present']].to_string(index=False))

    todo = [r['set'] for r in rows if r['status'] != 'OK']
    print(f'\nSets needing V5 predictions: {todo if todo else "none"}')

    if todo and not args.audit_only:
        so = ort.SessionOptions()
        so.intra_op_num_threads = ONNX_THREADS
        so.inter_op_num_threads = ONNX_THREADS
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(ONNX_MODEL_PATH, so,
                                       providers=['CPUExecutionProvider'])
        normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)
        for name in todo:
            d, meta = metas[name]
            run_v5_on_set(name, d, meta, session, normalizer)

        # PASS 2 — re-audit
        rows = []
        for name, d in sets:
            row, _ = audit_set(name, d, model_mtime)
            rows.append(row)
        audit = pd.DataFrame(rows)
        print('\nPASS 2 — audit after generation')
        print(audit[['set', 'n_metadata', 'v5_pred_exists', 'n_v5_rows',
                     'genome_ids_match', 'labels_match', 'status']].to_string(index=False))

    audit_path = RESULTS_DIR / 'legacy_v5_prediction_audit.tsv'
    audit.to_csv(audit_path, sep='\t', index=False)
    print(f'\nwrote {audit_path}')

    # metrics for every set that now has usable V5 predictions
    mrows = []
    for name, d in sets:
        p = d / PRED_FILE
        if p.exists():
            pred = pd.read_csv(p, sep='\t')
            if {'true_completeness', 'pred_completeness'} <= set(pred.columns):
                mrows.append(metrics_for(name, pred))
    mdf = pd.DataFrame(mrows)
    mpath = RESULTS_DIR / 'legacy_v5_metrics.tsv'
    mdf.to_csv(mpath, sep='\t', index=False)
    print(f'wrote {mpath}')
    print('\nV5 metrics on every benchmark set with a metadata.tsv:')
    print(mdf.to_string(index=False))

    with open(RESULTS_DIR / 'legacy_v5_prediction_audit.json', 'w') as f:
        json.dump({'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'model': ONNX_MODEL_PATH,
                   'n_workers': N_WORKERS,
                   'audit': rows,
                   'metrics': mrows}, f, indent=2)
    print('\nDONE')


if __name__ == '__main__':
    main()
