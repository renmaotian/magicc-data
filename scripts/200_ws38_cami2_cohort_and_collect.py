#!/usr/bin/env python3
"""
WS3.8 (R1-M3) — two jobs around the competitor runs on CAMI II.

(a) --select   Build the competitor cohort.
    CheckM2 runs at roughly 0.8 genomes/min/thread, so the full CAMI II bin sets are
    not free. Every mixed bin is kept (they ARE the designed grid and each cell needs
    its n), while the gold bins are SUBSAMPLED DELIBERATELY, stratified by completeness
    decile so the completeness gradient is preserved. The selection is seeded with
    fw.stable_hash (CRC-32) so it is reproducible, and the exact membership plus the
    reason for it is written to a TSV that the report cites.
    Output is a symlink farm with .fasta names (CheckM2 and CoCoPyE both take a
    directory + extension).

(b) --collect  Parse CheckM2 / CoCoPyE output and, where CheckM2's --dbg_vectors PKL
    files exist, DeepCheck, into one tidy predictions table per (dataset, binset).

Usage:
    python scripts/200_ws38_cami2_cohort_and_collect.py --select  --dataset marine
    python scripts/200_ws38_cami2_cohort_and_collect.py --collect --dataset marine
"""

import argparse
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import importlib.util as _ilu  # noqa: E402


def _load(path, name):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load(PROJECT_DIR / 'scripts' / '101_metrics_framework.py',
           'magicc_metrics_framework')

BIN_DIR = PROJECT_DIR / 'data' / 'real_data' / 'cami2' / 'bins'
RES_DIR = PROJECT_DIR / 'results' / 'revision' / 'cami2'
TRUTH_DIR = RES_DIR / 'truth'
PRED_DIR = RES_DIR / 'predictions'
COMP_DIR = RES_DIR / 'competitors'

GOLD_COHORT_N = 700          # gold bins per dataset handed to CheckM2/CoCoPyE
SELECT_SEED = 38200


# ------------------------------------------------------------------ selection
def select_cohort(ds, binset, n_target):
    src = BIN_DIR / ds / binset
    dst = BIN_DIR / ds / f'{binset}_competitor_subset'
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob('*.fna'))
    if not files:
        raise SystemExit(f'no bins in {src}')

    truth_p = TRUTH_DIR / f'{ds}_{"gold" if binset == "gold" else "mixed"}_truth.tsv'
    truth = pd.read_csv(truth_p, sep='\t')
    key = 'bin_id'
    tmap = truth.set_index(key)

    avail = pd.DataFrame({'bin_id': [p.stem for p in files],
                          'path': [str(p) for p in files]})
    avail = avail[avail['bin_id'].isin(tmap.index)]
    avail['completeness_pct'] = tmap.loc[avail['bin_id'], 'completeness_pct'].values
    avail['scoreable'] = avail['completeness_pct'] >= 50.0

    if binset == 'mixed' or len(avail) <= n_target:
        chosen = avail.copy()
        chosen['selection_reason'] = 'all_bins_kept'
    else:
        # stratify by completeness decile so the gradient survives subsampling
        pool = avail[avail['scoreable']].copy()
        probe = avail[~avail['scoreable']].copy()
        pool['stratum'] = pd.qcut(pool['completeness_pct'], 10,
                                  labels=False, duplicates='drop')
        rng = np.random.default_rng(SELECT_SEED + fw.stable_hash(f'{ds}|{binset}'))
        take_per = max(1, n_target // max(1, pool['stratum'].nunique()))
        parts = []
        for s, sub in pool.groupby('stratum'):
            sub = sub.sort_values('bin_id')
            k = min(len(sub), take_per)
            idx = rng.choice(len(sub), size=k, replace=False)
            parts.append(sub.iloc[np.sort(idx)])
        chosen = pd.concat(parts, ignore_index=True)
        chosen['selection_reason'] = 'stratified_by_completeness_decile'
        # bounded below-floor probe cohort, explicitly labelled
        if len(probe):
            pr = probe.sort_values('bin_id')
            k = min(len(pr), 100)
            idx = rng.choice(len(pr), size=k, replace=False)
            pr = pr.iloc[np.sort(idx)].copy()
            pr['selection_reason'] = 'below_50pct_floor_probe'
            chosen = pd.concat([chosen, pr], ignore_index=True)

    for old in dst.glob('*.fasta'):
        old.unlink()
    for r in chosen.itertuples(index=False):
        link = dst / f'{r.bin_id}.fasta'
        if not link.exists():
            os.symlink(r.path, link)

    chosen['dataset'] = ds
    chosen['binset'] = binset
    out = RES_DIR / 'provenance' / f'{ds}_{binset}_competitor_cohort.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(out, sep='\t', index=False)
    print(f'  {ds}/{binset}: {len(files)} bins available -> {len(chosen)} selected '
          f'({chosen["selection_reason"].value_counts().to_dict()})', flush=True)
    print(f'  symlinks -> {dst}', flush=True)
    return chosen


# ------------------------------------------------------------------ DeepCheck
_DC_CTX = None


def deepcheck_ctx():
    """Load the DeepCheck ResNet exactly as scripts/91 does (which itself imports
    scripts/038_run_deepcheck_v2.py verbatim)."""
    global _DC_CTX
    if _DC_CTX is not None:
        return _DC_CTX
    try:
        import torch
        dc_dir = PROJECT_DIR / 'tools' / 'DeepCheck'
        m38 = _load(PROJECT_DIR / 'scripts' / '038_run_deepcheck_v2.py', 'dc38')
        sp = np.load(dc_dir / 'scaler_params.npz')
        model = m38.ResNetDualOutput(m38.ResidualBlock, [2, 2, 2, 2])
        sd = torch.load(dc_dir / 'models' / 'best_model.pt', map_location='cpu',
                        weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        _DC_CTX = {'scale': sp['scale'], 'min': sp['min_val'], 'model': model,
                   'Dataset': m38.DeepCheckDataset, 'torch': torch}
        return _DC_CTX
    except Exception as e:                                           # noqa: BLE001
        print(f'  DeepCheck unavailable: {e}', flush=True)
        return None


def deepcheck_from_pkls(ck_dir, torch_threads=8):
    pkls = sorted(ck_dir.glob('*.pkl')) if ck_dir.exists() else []
    if not pkls:
        return None
    ctx = deepcheck_ctx()
    if ctx is None:
        return None
    torch = ctx['torch']
    from torch.utils.data import DataLoader
    feats = pd.concat([pd.read_pickle(p) for p in pkls], ignore_index=True)
    names = feats['Name'].astype(str).values
    mat = feats.iloc[:, 1:].values.astype(float)
    scaled = (mat * ctx['scale'] + ctx['min'])[:, :20021]
    padded = np.zeros((scaled.shape[0], 20164), dtype=np.float32)
    padded[:, :20021] = scaled
    dl = DataLoader(ctx['Dataset'](padded), batch_size=64, shuffle=False, num_workers=0)
    comp, cont = [], []
    torch.set_num_threads(torch_threads)
    with torch.no_grad():
        for batch in dl:
            cp, cx = ctx['model'](batch)
            comp.extend((cp.squeeze() * 100).cpu().numpy().tolist())
            cont.extend((cx.squeeze() * 100).cpu().numpy().tolist())
    out = pd.DataFrame({'bin_id': names, 'pred_completeness': comp,
                        'pred_contamination': cont})
    out['tool'] = 'DeepCheck'
    return out


# ------------------------------------------------------------------ collection
def collect(ds, binset):
    tag = f'{ds}_{binset}'
    work = COMP_DIR / tag
    frames = []

    qr = work / 'checkm2_output' / 'quality_report.tsv'
    if qr.exists():
        c = pd.read_csv(qr, sep='\t')
        c = c.rename(columns={'Name': 'bin_id', 'Completeness': 'pred_completeness',
                              'Contamination': 'pred_contamination'})
        c['tool'] = 'CheckM2'
        frames.append(c[['bin_id', 'pred_completeness', 'pred_contamination', 'tool']])
        print(f'  CheckM2: {len(c)} rows', flush=True)

    raw = work / 'cocopye_raw_output.csv'
    if raw.exists():
        try:
            # Same rule as scripts/091_parse_competitor_clean_cd.py::do_cocopye, so the
            # CAMI II numbers are directly comparable with every other set in this
            # revision: stage-3 (markers + neural network) with a stage-2 fallback,
            # rescaled from CoCoPyE's 0-1 fractions to percentages.
            k = pd.read_csv(raw)
            if '3_completeness' in k.columns:
                comp = k['3_completeness'].fillna(k.get('2_completeness')) * 100.0
                cont = k['3_contamination'].fillna(k.get('2_contamination')) * 100.0
                stage = 'stage3 (stage2 fallback)'
            elif '2_completeness' in k.columns:
                comp = k['2_completeness'] * 100.0
                cont = k['2_contamination'] * 100.0
                stage = 'stage2 only'
            else:
                raise ValueError('no completeness columns')
            kk = pd.DataFrame({'bin_id': k['bin'].astype(str),
                               'pred_completeness': comp,
                               'pred_contamination': cont})
            kk['bin_id'] = kk['bin_id'].str.replace(r'\.fasta$', '', regex=True)
            kk['tool'] = 'CoCoPyE'
            frames.append(kk)
            print(f'  CoCoPyE: {len(kk)} rows ({stage}, '
                  f'{int(kk["pred_completeness"].isna().sum())} NaN)', flush=True)
        except Exception as e:                                       # noqa: BLE001
            print(f'  CoCoPyE parse failed: {e}', flush=True)

    # DeepCheck: a pure tensor transform of CheckM2's --dbg_vectors PKL feature
    # vectors, so it costs nothing extra once CheckM2 has run. The scaler transform,
    # the 20,021 -> 20,164 zero-pad, the 142x142 reshape and the work-around for
    # DeepCheck's upstream forward() bug are imported VERBATIM from
    # scripts/038_run_deepcheck_v2.py via scripts/91, so no behaviour is re-invented.
    dcp = deepcheck_from_pkls(work / 'checkm2_output')
    if dcp is not None:
        frames.append(dcp)
        print(f'  DeepCheck: {len(dcp)} rows', flush=True)

    mg = PRED_DIR / f'{ds}_{binset}_magicc_v5.tsv'
    if mg.exists():
        m = pd.read_csv(mg, sep='\t')
        frames.append(m[['bin_id', 'pred_completeness', 'pred_contamination', 'tool']])
        print(f'  MAGICC_V5: {len(m)} rows', flush=True)

    if not frames:
        print(f'  nothing to collect for {tag}', flush=True)
        return None
    allp = pd.concat(frames, ignore_index=True)
    allp['bin_id'] = allp['bin_id'].astype(str).str.replace(r'\.(fasta|fna)$', '',
                                                            regex=True)
    allp['dataset'] = ds
    allp['binset'] = binset
    out = PRED_DIR / f'{tag}_all_tools.tsv'
    allp.to_csv(out, sep='\t', index=False)
    print(f'  wrote {out} ({len(allp)} rows, tools='
          f'{sorted(allp["tool"].unique())})', flush=True)
    return allp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--binsets', default='gold,mixed')
    ap.add_argument('--select', action='store_true')
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--gold-n', type=int, default=GOLD_COHORT_N)
    args = ap.parse_args()

    for bs in args.binsets.split(','):
        if args.select:
            select_cohort(args.dataset, bs, args.gold_n)
        if args.collect:
            collect(args.dataset, bs)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
