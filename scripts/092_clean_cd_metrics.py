#!/usr/bin/env python3
"""
WS1.5 (competitor part) -- descriptive accuracy metrics for MAGICC V5, CheckM2,
CoCoPyE and DeepCheck on the clean replacement sets ``set_C_clean`` and
``set_D_clean``.

For each (set, tool, target) it reports MAE, RMSE, R² and the mean signed error
(bias), each with a 95 % CI from a **cluster bootstrap that resamples the 100
reference genomes**, not the 1,000 individual genomes: every reference
contributes 10 simulations that are not independent, so a naive per-sample
bootstrap would understate the interval.

    n_boot = 2000, seed = 7600, percentile method, clusters = dominant_accession

R² CONVENTION (standardised for the revision). "R²" means one thing only, the
**coefficient of determination**, identical to ``sklearn.metrics.r2_score``:

    <target>_r2         = 1 - SS_res / SS_tot     <-- THE R² reported everywhere

The Pearson correlation is reported in separate, differently named columns and is
never called R²:

    <target>_pearson_r  = corr(true, pred)
    <target>_r2_pearson_sq = corr(true, pred) ** 2   (legacy traceability only:
                          what scripts 29/30/56/71/75 printed as "R²")

The two diverge exactly where a predictor is biased - squared Pearson ignores bias
and scale error and is always >= the coefficient of determination - which is why the
same set_C_clean completeness predictions gave 0.656 under the legacy convention and
0.610 under the correct one (the clean set carries a -2.32 pp completeness bias).
The coefficient of determination can be negative, meaning worse than predicting the
mean; such values are reported plainly.

R² is omitted (empty cell, with a printed reason) whenever the true value has zero
variance: Set A_v2 contamination is 0 % throughout and Set B_v2 completeness is
100 % throughout, so R² is undefined there (WS5.5, Reviewer 1 minor 19).

This module is also imported by ``93_leakage_specificity_control.py`` and
``94_revised_benchmark_table.py`` so that all three use one implementation.

Outputs
-------
    results/revision/benchmark/clean_cd_metrics.tsv
    results/revision/benchmark/clean_cd_metrics.json
    results/revision/benchmark/clean_cd_alignment_check.json

Usage
-----
    conda run -n magicc2 python scripts/92_clean_cd_metrics.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_DIR / 'data' / 'benchmarks'
OUT_DIR = PROJECT_DIR / 'results' / 'revision' / 'benchmark'

N_BOOT = 2000
BOOT_SEED = 7600
CLUSTER_COL = 'dominant_accession'

#: display label -> prediction filename
TOOLS: Dict[str, str] = {
    'MAGICC V5': 'magicc_v5_predictions.tsv',
    'CheckM2 1.0.1': 'checkm2_predictions.tsv',
    'CoCoPyE 0.5.0': 'cocopye_predictions.tsv',
    'DeepCheck': 'deepcheck_predictions.tsv',
}

CLEAN_SETS = ['set_C_clean', 'set_D_clean']


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_pred(set_name: str, filename: str,
              subdir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load one prediction TSV; returns None when the file does not exist."""
    base = BENCHMARK_DIR / (subdir or '') / set_name if subdir else BENCHMARK_DIR / set_name
    path = base / filename
    if not path.is_file():
        return None
    df = pd.read_csv(path, sep='\t')
    df['genome_id'] = df['genome_id'].astype(str)
    need = {'genome_id', 'true_completeness', 'true_contamination',
            'pred_completeness', 'pred_contamination'}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f'{path}: missing columns {sorted(missing)}')
    if CLUSTER_COL not in df.columns:
        meta = pd.read_csv(base / 'metadata.tsv', sep='\t')
        meta['genome_id'] = meta['genome_id'].astype(str)
        df = df.merge(meta[['genome_id', CLUSTER_COL]], on='genome_id', how='left')
    df.attrs['path'] = str(path)
    return df


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _stats(t: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    err = p - t
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    out = {
        'mae': float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'bias': float(np.mean(err)),
        'r2_cod': (1.0 - float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0 else np.nan,
    }
    if np.std(t) > 0 and np.std(p) > 0:
        out['pearson_r'] = float(np.corrcoef(t, p)[0, 1])
    else:
        out['pearson_r'] = np.nan
    out['pearson_r2'] = (out['pearson_r'] ** 2
                         if np.isfinite(out['pearson_r']) else np.nan)
    return out


def cluster_bootstrap(t: Sequence[float], p: Sequence[float],
                      clusters: Sequence, n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED) -> Dict[str, tuple]:
    """95 % percentile CIs for mae/rmse/bias/r2_cod, resampling *clusters*."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by_cluster = [np.where(clusters == c)[0] for c in uniq]
    rng = np.random.default_rng(seed)
    keys = ['mae', 'rmse', 'bias', 'r2_cod']
    acc = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        sel = np.concatenate([idx_by_cluster[i] for i in pick])
        s = _stats(t[sel], p[sel])
        for k in keys:
            acc[k][b] = s[k]
    ci = {}
    for k in keys:
        v = acc[k][np.isfinite(acc[k])]
        ci[k] = ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                 if v.size else (np.nan, np.nan))
    return ci


def metrics_row(df: pd.DataFrame, set_label: str, tool_label: str,
                subset_label: str = 'all', n_boot: int = N_BOOT,
                seed: int = BOOT_SEED) -> Dict[str, object]:
    """One row of the metrics table for a (set, tool, subset).

    Genomes a tool failed to score (NaN prediction) are dropped and counted in
    ``n_unscored``; they are never silently averaged over.
    """
    n_in = int(len(df))
    df = df.dropna(subset=['pred_completeness', 'pred_contamination'])
    clusters = df[CLUSTER_COL].values
    row: Dict[str, object] = {
        'set': set_label, 'tool': tool_label, 'subset': subset_label,
        'n': int(len(df)), 'n_unscored': n_in - int(len(df)),
        'n_clusters': int(pd.unique(clusters).size),
        'cluster_unit': CLUSTER_COL, 'n_boot': n_boot, 'boot_seed': seed,
    }
    for target, prefix in (('completeness', 'comp'), ('contamination', 'cont')):
        t = df[f'true_{target}'].values.astype(float)
        p = df[f'pred_{target}'].values.astype(float)
        s = _stats(t, p)
        ci = cluster_bootstrap(t, p, clusters, n_boot=n_boot, seed=seed)
        const_true = bool(np.std(t) == 0)
        row[f'{prefix}_true_constant'] = const_true
        for k in ('mae', 'rmse', 'bias'):
            row[f'{prefix}_{k}'] = round(s[k], 4)
            row[f'{prefix}_{k}_ci_lo'] = round(ci[k][0], 4)
            row[f'{prefix}_{k}_ci_hi'] = round(ci[k][1], 4)
        if const_true:
            for suffix in ('r2', 'r2_ci_lo', 'r2_ci_hi', 'pearson_r', 'pearson_r2'):
                row[f'{prefix}_{suffix}'] = ''
            row[f'{prefix}_r2_omitted_reason'] = (
                f'true {target} has zero variance in this subset '
                f'(constant {t[0]:.6g} %), so SS_tot = 0 and R² is undefined')
            print(f'  R² omitted for {set_label}/{tool_label}/{subset_label} '
                  f'{target}: true value is constant ({t[0]:.6g} %), SS_tot = 0.')
        else:
            row[f'{prefix}_r2'] = round(s['r2_cod'], 4)
            row[f'{prefix}_r2_ci_lo'] = round(ci['r2_cod'][0], 4)
            row[f'{prefix}_r2_ci_hi'] = round(ci['r2_cod'][1], 4)
            row[f'{prefix}_pearson_r'] = (round(s['pearson_r'], 4)
                                         if np.isfinite(s['pearson_r']) else '')
            row[f'{prefix}_r2_pearson_sq'] = (round(s['pearson_r2'], 4)
                                          if np.isfinite(s['pearson_r2']) else '')
            row[f'{prefix}_r2_omitted_reason'] = ''
    return row


def fmt_ci(row: Dict[str, object], prefix: str, stat: str, nd: int = 2) -> str:
    v, lo, hi = (row[f'{prefix}_{stat}'], row[f'{prefix}_{stat}_ci_lo'],
                 row[f'{prefix}_{stat}_ci_hi'])
    if v == '' or v is None:
        return '-'
    return f'{float(v):.{nd}f} ({float(lo):.{nd}f}-{float(hi):.{nd}f})'


# --------------------------------------------------------------------------
# alignment across tools
# --------------------------------------------------------------------------
def check_alignment(set_name: str, frames: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Every tool must cover the same genomes in the same order with the same truth."""
    meta = pd.read_csv(BENCHMARK_DIR / set_name / 'metadata.tsv', sep='\t')
    meta['genome_id'] = meta['genome_id'].astype(str)
    rep: Dict[str, object] = {'n_metadata': int(len(meta)), 'tools': {}, 'ok': True}
    ref_ids = meta['genome_id'].tolist()
    for tool, df in frames.items():
        ids_equal = df['genome_id'].tolist() == ref_ids
        comp_equal = np.array_equal(df['true_completeness'].values.astype(float),
                                    meta['true_completeness'].values.astype(float))
        cont_equal = np.array_equal(df['true_contamination'].values.astype(float),
                                   meta['true_contamination'].values.astype(float))
        nan_pred = int(df[['pred_completeness', 'pred_contamination']].isna().sum().sum())
        entry = {'n_rows': int(len(df)), 'genome_ids_match_metadata_in_order': bool(ids_equal),
                 'true_completeness_matches': bool(comp_equal),
                 'true_contamination_matches': bool(cont_equal),
                 'n_nan_predictions': nan_pred, 'path': df.attrs.get('path')}
        entry['ok'] = bool(ids_equal and comp_equal and cont_equal
                           and len(df) == len(meta))
        entry['fully_scored'] = bool(nan_pred == 0)
        rep['tools'][tool] = entry
        rep['ok'] = bool(rep['ok'] and entry['ok'])
        if nan_pred:
            print(f'  WARNING {set_name}/{tool}: {nan_pred} NaN prediction value(s); '
                  f'those genomes are excluded from that tool\'s metrics and counted '
                  f'in n_unscored', file=sys.stderr)
    return rep


# --------------------------------------------------------------------------
def selftest_r2_matches_sklearn() -> str:
    """Verify _stats()['r2_cod'] == sklearn.metrics.r2_score on random data."""
    rng = np.random.default_rng(0)
    t = rng.uniform(50, 100, 500)
    p = t + rng.normal(-3, 5, 500)          # deliberately biased, where the
    mine = _stats(t, p)['r2_cod']           # two conventions must diverge
    legacy = _stats(t, p)['pearson_r2']
    try:
        from sklearn.metrics import r2_score
        ref = float(r2_score(t, p))
        assert abs(mine - ref) < 1e-12, f'R2 mismatch: {mine} vs sklearn {ref}'
        msg = (f'R2 self-test PASS: coefficient of determination {mine:.6f} == '
               f'sklearn.metrics.r2_score {ref:.6f}; squared Pearson (legacy, NOT R2) '
               f'would have reported {legacy:.6f}')
    except ImportError:
        msg = (f'R2 self-test: sklearn unavailable; coefficient of determination '
               f'{mine:.6f} vs squared Pearson (legacy, NOT R2) {legacy:.6f}')
    print(msg)
    return msg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selftest = selftest_r2_matches_sklearn()
    rows: List[Dict[str, object]] = []
    alignment: Dict[str, object] = {}
    missing: List[str] = []

    for set_name in CLEAN_SETS:
        frames: Dict[str, pd.DataFrame] = {}
        for tool, fn in TOOLS.items():
            df = load_pred(set_name, fn)
            if df is None:
                missing.append(f'{set_name}/{fn}')
                continue
            frames[tool] = df
        alignment[set_name] = check_alignment(set_name, frames)
        if not alignment[set_name]['ok']:
            print(f'ALIGNMENT FAILURE in {set_name}: '
                  f'{json.dumps(alignment[set_name], indent=2, default=str)}',
                  file=sys.stderr)
            return 1
        for tool, df in frames.items():
            row = metrics_row(df, set_name, tool)
            assert row['n_clusters'] == 100, \
                f'{set_name}/{tool}: expected 100 reference clusters, got {row["n_clusters"]}'
            assert row['n'] + row['n_unscored'] == 1000, \
                f'{set_name}/{tool}: n + n_unscored = {row["n"]} + {row["n_unscored"]} != 1000'
            rows.append(row)
            print(f'{set_name:14s} {tool:16s} '
                  f'comp MAE {fmt_ci(row, "comp", "mae")}  bias {fmt_ci(row, "comp", "bias"):>22s}  '
                  f'| cont MAE {fmt_ci(row, "cont", "mae")}  bias {fmt_ci(row, "cont", "bias")}')

    if missing:
        print('\nMISSING prediction files:\n  ' + '\n  '.join(missing), file=sys.stderr)

    tbl = pd.DataFrame(rows)
    tsv = OUT_DIR / 'clean_cd_metrics.tsv'
    tbl.to_csv(tsv, sep='\t', index=False)
    (OUT_DIR / 'clean_cd_metrics.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'bootstrap': {'n_boot': N_BOOT, 'seed': BOOT_SEED, 'method': 'percentile',
                      'cluster_unit': CLUSTER_COL,
                      'rationale': '10 simulations per reference genome are not '
                                   'independent; clusters are the 100 held-out '
                                   'test-split reference genomes'},
        'r2_convention': {
            'r2': 'THE R2 reported: coefficient of determination 1 - SS_res/SS_tot, '
                  'identical to sklearn.metrics.r2_score; may be negative',
            'pearson_r': 'Pearson correlation coefficient r (separate statistic, never '
                         'called R2)',
            'r2_pearson_sq': 'squared Pearson r - legacy traceability only; this is what '
                          'scripts 29/30/56/71/75 printed as "R2"',
            'omitted_when': 'true value has zero variance (Set A_v2 contamination, '
                            'Set B_v2 completeness); see *_r2_omitted_reason'},
        'r2_selftest': selftest,
        'missing_prediction_files': missing,
        'alignment': alignment,
        'rows': rows,
    }, indent=2, default=str))
    (OUT_DIR / 'clean_cd_alignment_check.json').write_text(
        json.dumps(alignment, indent=2, default=str))
    print(f'\n-> {tsv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
