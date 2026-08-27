#!/usr/bin/env python3
"""
WS1.5 (MAGICC part) — Run the frozen MAGICC V5 model on Sets C_clean / D_clean and
quantify the leakage effect against the superseded Sets C / D.

Inference is direct ONNX Runtime, following scripts/56_benchmark_v5.py exactly:
    FASTA -> selected-9-mer counts -> 7 k-mer-summary features -> normalisation
          -> models/magicc_v5.onnx
The `magicc` CLI is deliberately NOT used (it is being edited concurrently).

Because each clean set reuses 100 reference genomes for 10 simulations each, every
statistic is also reported with a **cluster bootstrap 95 % CI, resampling reference
genomes** (protocol WS5.4 / reviewer R1-m16). The superseded sets have 1,000 distinct
dominants, so for them the cluster bootstrap degenerates to an ordinary bootstrap; both
are reported so the intervals are comparable.

The leaked -> clean delta mixes two defects of the old sets:
  (1) training-set leakage (WS1.4: Set C 1000/1000 dominants in TRAIN, Set D 796 TRAIN +
      107 VAL), and
  (2) 141 (C) / 213 (D) samples whose contamination % exceeds their completeness %, which
      the V5 training distribution never contains (the cap entered
      magicc/contamination.py only after those sets were generated).
The comparison table therefore also reports the superseded sets restricted to their
constraint-satisfying subsets, which isolates (1).

Outputs:
  data/benchmarks/set_{C,D}_clean/magicc_v5_predictions.tsv
  results/revision/ws1_5_clean_cd_metrics.tsv
  results/revision/ws1_5_clean_vs_leaked_cd.tsv
  results/revision/ws1_5_clean_cd_summary.json

Usage:
    python scripts/75_run_magicc_clean_cd.py
    python scripts/75_run_magicc_clean_cd.py --force     # ignore existing predictions
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

N_WORKERS = 20
BATCH_SIZE = 64
ONNX_THREADS = 1
N_BOOT = 2000
BOOT_SEED = 7500

CLEAN_SETS = {'C_clean': 'set_C_clean', 'D_clean': 'set_D_clean'}
OLD_SETS = {'C': 'set_C', 'D': 'set_D'}


# ------------------------------------------------------------------ inference
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
    _kc.count_contigs(["ACGTACGTACGTACGTACGT" * 50])


def _feat_worker(args):
    idx, path = args
    try:
        contigs = read_fasta_contigs(path)
        if not contigs:
            return idx, None, None, 'no_contigs'
        kc = _kc.count_contigs(contigs)
        asm = compute_assembly_stats(_kc.total_kmer_count(kc), kc)
        return idx, kc.astype(np.float32), asm.astype(np.float32), None
    except Exception as e:                                          # noqa: BLE001
        return idx, None, None, str(e)


def run_v5(set_label, subdir, session, normalizer, force=False):
    d = BENCHMARK_DIR / subdir
    meta = pd.read_csv(d / 'metadata.tsv', sep='\t')
    out = d / 'magicc_v5_predictions.tsv'

    if out.exists() and not force:
        pred = pd.read_csv(out, sep='\t')
        if len(pred) == len(meta) and set(pred['genome_id']) == set(meta['genome_id']):
            print(f'  {set_label}: predictions already complete ({len(pred)}) — reusing')
            return pred

    print(f'\n{"=" * 74}\n  MAGICC V5 on {subdir} ({len(meta)} genomes)\n{"=" * 74}')
    work = [(i, str(d / 'fasta' / f'{g}.fasta'))
            for i, g in enumerate(meta['genome_id'])
            if (d / 'fasta' / f'{g}.fasta').exists()]
    if len(work) != len(meta):
        print(f'    WARNING: {len(meta) - len(work)} FASTA files missing')

    t0 = time.time()
    got = {}
    with Pool(processes=N_WORKERS, initializer=_init_worker,
              initargs=(SELECTED_KMERS_PATH,)) as pool:
        for idx, kmer, asm, err in pool.imap_unordered(_feat_worker, work, chunksize=10):
            if err:
                print(f'    WARNING idx={idx}: {err}')
            else:
                got[idx] = (kmer, asm)
    feat_t = time.time() - t0
    idxs = sorted(got)
    n = len(idxs)
    print(f'    features: {feat_t:.1f}s ({feat_t / max(1, n) * 1000:.1f} ms/genome, '
          f'{N_WORKERS} workers)')

    kn = normalizer.normalize_kmer(np.stack([got[i][0] for i in idxs])).astype(np.float32)
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
    res['n_threads'] = N_WORKERS
    res['model'] = 'magicc_v5.onnx'
    tmp = out.with_suffix('.tsv.tmp')
    res.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, out)
    print(f'    wrote {out}')
    return res


# -------------------------------------------------------------------- metrics
def _core(t, p):
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    r2 = (float(np.corrcoef(t, p)[0, 1] ** 2)
          if np.std(t) > 0 and np.std(p) > 0 else float('nan'))
    return mae, rmse, bias, r2


def cluster_bootstrap_ci(t, p, clusters, stat='mae', n_boot=N_BOOT, seed=BOOT_SEED):
    """95 % CI by resampling clusters (reference genomes) with replacement."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    uniq = np.unique(clusters)
    idx_by_cluster = [np.where(clusters == c)[0] for c in uniq]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        sel = np.concatenate([idx_by_cluster[i] for i in pick])
        e = p[sel] - t[sel]
        if stat == 'mae':
            vals[b] = np.mean(np.abs(e))
        elif stat == 'rmse':
            vals[b] = np.sqrt(np.mean(e ** 2))
        else:
            vals[b] = np.mean(e)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics_block(label, pred, cluster_col=None, note=''):
    tc = pred['true_completeness'].values.astype(float)
    tx = pred['true_contamination'].values.astype(float)
    pc = pred['pred_completeness'].values.astype(float)
    px = pred['pred_contamination'].values.astype(float)
    clusters = (pred[cluster_col].values if cluster_col and cluster_col in pred.columns
                else np.arange(len(pred)))

    cm, cr, cb, cr2 = _core(tc, pc)
    xm, xr, xb, xr2 = _core(tx, px)
    cm_lo, cm_hi = cluster_bootstrap_ci(tc, pc, clusters, 'mae')
    xm_lo, xm_hi = cluster_bootstrap_ci(tx, px, clusters, 'mae')
    cb_lo, cb_hi = cluster_bootstrap_ci(tc, pc, clusters, 'bias')
    xb_lo, xb_hi = cluster_bootstrap_ci(tx, px, clusters, 'bias')

    return {
        'set': label,
        'n': int(len(pred)),
        'n_clusters': int(len(np.unique(clusters))),
        'cluster_unit': (cluster_col or 'sample (no clustering)'),
        'comp_mae': round(cm, 4), 'comp_mae_ci_lo': round(cm_lo, 4),
        'comp_mae_ci_hi': round(cm_hi, 4),
        'comp_rmse': round(cr, 4), 'comp_bias': round(cb, 4),
        'comp_bias_ci_lo': round(cb_lo, 4), 'comp_bias_ci_hi': round(cb_hi, 4),
        'comp_r2': (round(cr2, 4) if not np.isnan(cr2) else ''),
        'cont_mae': round(xm, 4), 'cont_mae_ci_lo': round(xm_lo, 4),
        'cont_mae_ci_hi': round(xm_hi, 4),
        'cont_rmse': round(xr, 4), 'cont_bias': round(xb, 4),
        'cont_bias_ci_lo': round(xb_lo, 4), 'cont_bias_ci_hi': round(xb_hi, 4),
        'cont_r2': (round(xr2, 4) if not np.isnan(xr2) else ''),
        'note': note,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('WS1.5 — MAGICC V5 on Sets C_clean / D_clean')
    print(f'model: {ONNX_MODEL_PATH}')
    print('=' * 78)

    so = ort.SessionOptions()
    so.intra_op_num_threads = ONNX_THREADS
    so.inter_op_num_threads = ONNX_THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, so,
                                   providers=['CPUExecutionProvider'])
    print(f'  inputs : {[(i.name, i.shape) for i in session.get_inputs()]}')
    print(f'  outputs: {[(o.name, o.shape) for o in session.get_outputs()]}')
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)

    rows = []
    preds = {}
    for label, subdir in CLEAN_SETS.items():
        p = run_v5(label, subdir, session, normalizer, force=args.force)
        preds[label] = p
        rows.append(metrics_block(f'set_{subdir[4:]}', p, cluster_col='ref_index',
                                  note='clean: dominants from held-out test split only, '
                                       '100 refs x 10 sims'))

    # superseded sets, recomputed from their stored V5 predictions
    for label, subdir in OLD_SETS.items():
        pp = BENCHMARK_DIR / subdir / 'magicc_v5_predictions.tsv'
        if not pp.exists():
            print(f'  WARNING: {pp} missing, cannot compare')
            continue
        old = pd.read_csv(pp, sep='\t')
        preds[label] = old
        rows.append(metrics_block(f'set_{label} (SUPERSEDED, leaked)', old,
                                  note='dominants from train+val+test'))
        ok = old['true_contamination'] <= old['true_completeness'] + 1e-6
        rows.append(metrics_block(
            f'set_{label} (SUPERSEDED, leaked, constraint-satisfying subset)',
            old[ok].reset_index(drop=True),
            note=f'{int((~ok).sum())} of {len(old)} samples removed because '
                 f'contamination% > completeness%, a state absent from V5 training data'))

    mdf = pd.DataFrame(rows)
    mpath = RESULTS_DIR / 'ws1_5_clean_cd_metrics.tsv'
    mdf.to_csv(mpath, sep='\t', index=False)
    print(f'\nwrote {mpath}')
    print()
    print(mdf[['set', 'n', 'n_clusters', 'comp_mae', 'comp_mae_ci_lo', 'comp_mae_ci_hi',
               'comp_bias', 'comp_r2', 'cont_mae', 'cont_mae_ci_lo', 'cont_mae_ci_hi',
               'cont_bias', 'cont_r2']].to_string(index=False))

    # ------------------------------------------------------ delta table
    def row_for(name):
        m = mdf[mdf['set'] == name]
        return m.iloc[0] if len(m) else None

    comp_rows = []
    for old_lab, clean_lab, lineage in (
            ('C', 'set_C_clean', 'Patescibacteriota (CPR)'),
            ('D', 'set_D_clean', 'Archaea')):
        leaked = row_for(f'set_{old_lab} (SUPERSEDED, leaked)')
        leaked_sub = row_for(f'set_{old_lab} (SUPERSEDED, leaked, '
                             f'constraint-satisfying subset)')
        clean = row_for(clean_lab)
        if leaked is None or clean is None:
            continue
        comp_rows.append({
            'lineage': lineage,
            'superseded_set': f'set_{old_lab}',
            'clean_set': clean_lab,
            'n_superseded': leaked['n'],
            'n_clean': clean['n'],
            'leaked_comp_mae': leaked['comp_mae'],
            'clean_comp_mae': clean['comp_mae'],
            'delta_comp_mae': round(clean['comp_mae'] - leaked['comp_mae'], 4),
            'delta_comp_mae_pct': round(
                100 * (clean['comp_mae'] - leaked['comp_mae']) / leaked['comp_mae'], 1),
            'leaked_cont_mae': leaked['cont_mae'],
            'clean_cont_mae': clean['cont_mae'],
            'delta_cont_mae': round(clean['cont_mae'] - leaked['cont_mae'], 4),
            'delta_cont_mae_pct': round(
                100 * (clean['cont_mae'] - leaked['cont_mae']) / leaked['cont_mae'], 1),
            'leaked_comp_bias': leaked['comp_bias'],
            'clean_comp_bias': clean['comp_bias'],
            'leaked_cont_bias': leaked['cont_bias'],
            'clean_cont_bias': clean['cont_bias'],
            'leaked_comp_r2': leaked['comp_r2'],
            'clean_comp_r2': clean['comp_r2'],
            'leaked_cont_r2': leaked['cont_r2'],
            'clean_cont_r2': clean['cont_r2'],
            'leaked_subset_n': (leaked_sub['n'] if leaked_sub is not None else ''),
            'leaked_subset_comp_mae': (leaked_sub['comp_mae']
                                       if leaked_sub is not None else ''),
            'leaked_subset_cont_mae': (leaked_sub['cont_mae']
                                       if leaked_sub is not None else ''),
            'delta_comp_mae_vs_leaked_subset': (
                round(clean['comp_mae'] - leaked_sub['comp_mae'], 4)
                if leaked_sub is not None else ''),
            'delta_cont_mae_vs_leaked_subset': (
                round(clean['cont_mae'] - leaked_sub['cont_mae'], 4)
                if leaked_sub is not None else ''),
            'clean_comp_mae_ci': f"[{clean['comp_mae_ci_lo']}, {clean['comp_mae_ci_hi']}]",
            'clean_cont_mae_ci': f"[{clean['cont_mae_ci_lo']}, {clean['cont_mae_ci_hi']}]",
        })
    cdf = pd.DataFrame(comp_rows)
    cpath = RESULTS_DIR / 'ws1_5_clean_vs_leaked_cd.tsv'
    cdf.to_csv(cpath, sep='\t', index=False)
    print(f'\nwrote {cpath}')
    print('\nLEAKED -> CLEAN COMPARISON (MAGICC V5, same frozen model)')
    print(cdf[['lineage', 'leaked_comp_mae', 'clean_comp_mae', 'delta_comp_mae',
               'leaked_cont_mae', 'clean_cont_mae', 'delta_cont_mae',
               'leaked_subset_comp_mae', 'leaked_subset_cont_mae']].to_string(index=False))

    # ------------------------------- stratified comparison (removes the residual
    # difference in the realised label distributions between old and clean sets)
    strat_rows = []
    comp_bins = [50, 60, 70, 80, 90, 100.0001]
    cont_bins = [0, 20, 40, 60, 80, 100.0001]
    for lineage, clean_lab, old_lab in (('C', 'C_clean', 'C'), ('D', 'D_clean', 'D')):
        variants = [('clean', preds[clean_lab])]
        if old_lab in preds:
            old = preds[old_lab]
            variants.append(('leaked_all', old))
            ok = old['true_contamination'] <= old['true_completeness'] + 1e-6
            variants.append(('leaked_constraint_ok', old[ok].reset_index(drop=True)))
        for axis, bins, col in (('true_completeness', comp_bins, 'true_completeness'),
                                ('true_contamination', cont_bins, 'true_contamination')):
            for lo, hi in zip(bins[:-1], bins[1:]):
                for vname, dfv in variants:
                    m = (dfv[col] >= lo) & (dfv[col] < hi)
                    if m.sum() == 0:
                        continue
                    s = dfv[m]
                    strat_rows.append({
                        'lineage': lineage, 'variant': vname, 'stratify_by': axis,
                        'bin': f'[{lo:.0f},{min(hi,100):.0f})', 'n': int(m.sum()),
                        'comp_mae': round(float(np.mean(np.abs(
                            s['pred_completeness'] - s['true_completeness']))), 4),
                        'cont_mae': round(float(np.mean(np.abs(
                            s['pred_contamination'] - s['true_contamination']))), 4),
                    })
    sdf = pd.DataFrame(strat_rows)
    spath = RESULTS_DIR / 'ws1_5_stratified_cd.tsv'
    sdf.to_csv(spath, sep='\t', index=False)
    print(f'\nwrote {spath}')
    print('\nSTRATIFIED comp/cont MAE (clean vs superseded), by true-value band:')
    piv = sdf.pivot_table(index=['lineage', 'stratify_by', 'bin'], columns='variant',
                          values=['comp_mae', 'cont_mae'])
    print(piv.to_string())

    # ----------------------------------------- per-reference spread (clean sets)
    per_ref = {}
    for label in CLEAN_SETS:
        p = preds[label]
        gg = p.groupby('ref_index').apply(
            lambda x: pd.Series({
                'comp_mae': np.mean(np.abs(x['pred_completeness']
                                           - x['true_completeness'])),
                'cont_mae': np.mean(np.abs(x['pred_contamination']
                                           - x['true_contamination'])),
            }), include_groups=False)
        per_ref[label] = {
            'comp_mae_per_reference': {'min': round(float(gg['comp_mae'].min()), 3),
                                       'median': round(float(gg['comp_mae'].median()), 3),
                                       'max': round(float(gg['comp_mae'].max()), 3),
                                       'iqr': [round(float(gg['comp_mae'].quantile(.25)), 3),
                                               round(float(gg['comp_mae'].quantile(.75)), 3)]},
            'cont_mae_per_reference': {'min': round(float(gg['cont_mae'].min()), 3),
                                       'median': round(float(gg['cont_mae'].median()), 3),
                                       'max': round(float(gg['cont_mae'].max()), 3),
                                       'iqr': [round(float(gg['cont_mae'].quantile(.25)), 3),
                                               round(float(gg['cont_mae'].quantile(.75)), 3)]},
        }
        gg.to_csv(BENCHMARK_DIR / CLEAN_SETS[label] / 'per_reference_v5_mae.tsv', sep='\t')
        print(f'\n  {label} per-reference MAE spread: {per_ref[label]}')

    with open(RESULTS_DIR / 'ws1_5_clean_cd_summary.json', 'w') as f:
        json.dump({'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'model': ONNX_MODEL_PATH,
                   'n_workers': N_WORKERS,
                   'bootstrap': {'n_boot': N_BOOT, 'seed': BOOT_SEED,
                                 'unit': 'reference genome (ref_index) for clean sets, '
                                         'sample for superseded sets'},
                   'metrics': rows,
                   'clean_vs_leaked': comp_rows,
                   'per_reference_spread': per_ref}, f, indent=2,
                  default=lambda o: (int(o) if isinstance(o, np.integer)
                                     else float(o) if isinstance(o, np.floating)
                                     else str(o)))
    print(f"\nwrote {RESULTS_DIR / 'ws1_5_clean_cd_summary.json'}\nDONE")


if __name__ == '__main__':
    main()
