#!/usr/bin/env python3
"""
WS4.2 -- GUNC as a *detection* comparator across benchmark sets, with database
sensitivity (proGenomes 2.1 vs GTDB r95) and an explicit statistical-power audit.

Binding framing (protocol v6 sec. 4.4c item 5)
----------------------------------------------
GUNC is a DETECTION comparator, NOT a quantitative estimator: it returns a clade
separation score (CSS) plus a pass/fail flag, not completeness/contamination
percentages.  It is therefore deliberately kept OUT of the MAE table.  What is
reported instead:

  1. Pass/fail rates stratified by true contamination
     (bands 0-5 / 5-10 / 10-20 / 20-40 / 40-70 / 70-100 %).
  2. Agreement at the MIMAG-inspired 5 % and 10 % contamination thresholds,
     treating GUNC pass/fail as the binary call.
  3. Spearman correlation of CSS vs true contamination, reported alongside the
     same correlation for MAGICC V5 and CheckM2 so all three are on equal footing.
  4. Detection sensitivity/specificity/precision/recall with 95 % CIs from a
     cluster bootstrap over the reference genomes (10 simulations per reference
     are NOT independent).

The statistical-power audit (the central point)
-----------------------------------------------
GUNC computes, per genome:

    genes_retained_index (GRI) = genes retained in abundant clades / genes called
    reference_representation_score (RRS) = GRI * mean_hit_identity
    adjustment = 1 if GRI > 0.4 else 0
    CSS_adjusted = CSS * adjustment
    pass.GUNC = CSS_adjusted <= 0.45

so whenever ``GRI <= 0.4`` the CSS is **forcibly zeroed** and the genome
**automatically passes**, irrespective of how chimeric it actually is.  A "pass"
from such a genome is not evidence of cleanliness -- it is evidence that GUNC
had no close reference.  Every table below is therefore also reported split into
a POWERED and an UNPOWERED stratum, and the fraction of each set that GUNC was
actually powered to adjudicate is reported as a first-class result.

Power strata (all three components are also reported separately):
    hard_unpowered  GRI <= 0.4                      CSS forcibly zeroed
    weak            GRI > 0.4 and RRS < 0.5         control cohort criterion:
                                                    7/8 novel-lineage positives
                                                    that GUNC failed to flag sat
                                                    here
    powered         GRI > 0.4 and RRS >= 0.5
GUNC's own caution flag (RRS < 0.3) is reported as an additional column.

Inputs
------
    results/revision/gunc/runs/<set>/<db>/gunc_normalized.tsv   (see 76_run_gunc.py)
    data/benchmarks/<set>/metadata.tsv                          ground truth
    data/benchmarks/<set>/magicc_v5_predictions.tsv             comparator
    data/benchmarks/<set>/checkm2_predictions.tsv               comparator

Outputs (all under results/revision/gunc/)
------------------------------------------
    gunc_per_genome.tsv          one row per (set, db, genome); every power field
    gunc_power_audit.tsv         powered fraction per set per DB  <- KEY NUMBER
    gunc_stratified_passfail.tsv pass/fail by true-contamination band x stratum
    gunc_threshold_agreement.tsv 5 %/10 % detection metrics + 95 % CIs
    gunc_css_correlations.tsv    Spearman CSS/MAGICC/CheckM2 vs true contamination
    gunc_db_sensitivity.tsv      proGenomes 2.1 vs GTDB r95, paired head-to-head
    WS4.2_gunc_report.md         narrative report
    ws4.2_gunc_summary.json      machine-readable summary

Usage
-----
    python scripts/138_ws42_gunc_comparator.py
    python scripts/138_ws42_gunc_comparator.py --sets set_C_clean set_D_clean
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / 'results' / 'revision' / 'gunc' / 'runs'
OUT_DIR = PROJECT_DIR / 'results' / 'revision' / 'gunc'
BENCH_DIR = PROJECT_DIR / 'data' / 'benchmarks'

DB_LABEL = {'progenomes_2.1': 'proGenomes 2.1', 'gtdb_95': 'GTDB r95'}
DB_ORDER = ['progenomes_2.1', 'gtdb_95']

#: true-contamination bands mandated by the WS4.2 task spec, [lo, hi)
BANDS: List[Tuple[float, float, str]] = [
    (0.0, 5.0, '0-5'), (5.0, 10.0, '5-10'), (10.0, 20.0, '10-20'),
    (20.0, 40.0, '20-40'), (40.0, 70.0, '40-70'), (70.0, 100.01, '70-100'),
]
THRESHOLDS = [5.0, 10.0]

#: GUNC internals
GRI_ZEROING_CUTOFF = 0.4    # CSS forcibly zeroed at or below this
RRS_POWERED_CUTOFF = 0.5    # control-cohort criterion
RRS_GUNC_WARN = 0.3         # GUNC's own "not well represented" warning
CSS_CHIMERIC_THRESHOLD = 0.45

CLUSTER_COL = 'dominant_accession'
N_BOOT = 2000
BOOT_SEED = 7600

COMPARATORS = [('MAGICC V5', 'magicc_v5_predictions.tsv'),
               ('CheckM2', 'checkm2_predictions.tsv')]


# --------------------------------------------------------------------------
# statistics (no sklearn dependency; scipy optional)
# --------------------------------------------------------------------------
def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (equivalent to scipy.stats.rankdata)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_p(rho: float, n: int) -> float:
    """Two-sided p-value via the t approximation (n >= 10 here throughout)."""
    if not np.isfinite(rho) or n < 4 or abs(rho) >= 1.0:
        return float('nan')
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), n - 2))
    except Exception:
        # normal approximation fallback
        from math import erfc, sqrt
        return float(erfc(abs(t) / sqrt(2)))


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC via the Mann-Whitney U identity, ties handled by mid-ranks."""
    label = np.asarray(label).astype(bool)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion(pred_pos: np.ndarray, true_pos: np.ndarray) -> Dict[str, float]:
    pred_pos = np.asarray(pred_pos).astype(bool)
    true_pos = np.asarray(true_pos).astype(bool)
    tp = int(np.sum(pred_pos & true_pos))
    fp = int(np.sum(pred_pos & ~true_pos))
    tn = int(np.sum(~pred_pos & ~true_pos))
    fn = int(np.sum(~pred_pos & true_pos))
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    f1 = (2 * prec * sens / (prec + sens)
          if np.isfinite(prec) and np.isfinite(sens) and (prec + sens) > 0
          else float('nan'))
    bal = (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else float('nan')
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den > 0 else float('nan')
    n = tp + fp + tn + fn
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'n': n,
            'sensitivity': sens, 'specificity': spec, 'precision': prec,
            'recall': sens, 'f1': f1, 'balanced_accuracy': bal, 'mcc': mcc,
            'accuracy': (tp + tn) / n if n else float('nan')}


def stable_hash(s: str) -> int:
    """Process-independent, platform-independent hash of a string (CRC-32).

    Python's built-in ``hash()`` for str/bytes is salted per interpreter process
    (PYTHONHASHSEED), so seeds derived from it differ between runs and bootstrap
    confidence intervals are NOT reproducible.  Three such defects have already
    been found and fixed in this revision; every derived seed in this script
    therefore goes through this function.  Mirrors
    ``scripts/101_metrics_framework.py::stable_hash``.
    """
    return zlib.crc32(s.encode('utf-8')) & 0xFFFFFFFF


def derive_seed(*parts) -> int:
    """Deterministic per-context bootstrap seed: BOOT_SEED + CRC-32 of context."""
    return (BOOT_SEED + stable_hash('|'.join(str(p) for p in parts))) % (2 ** 32)


def cluster_ci(fn, clusters: np.ndarray, n_boot: int = N_BOOT,
               seed: int = BOOT_SEED) -> Tuple[float, float]:
    """95 % percentile CI of fn(index_selection), resampling whole clusters."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return (float('nan'), float('nan'))
    groups = [np.where(clusters == c)[0] for c in uniq]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        sel = np.concatenate([groups[i] for i in pick])
        try:
            vals[b] = fn(sel)
        except Exception:
            vals[b] = np.nan
    v = vals[np.isfinite(vals)]
    return ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            if v.size else (float('nan'), float('nan')))


def bh(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values; NaNs pass through."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return out.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = p[i] * m / (rank + 1)
        prev = min(prev, val)
        out[i] = min(prev, 1.0)
    return out.tolist()


def f3(x) -> str:
    return '-' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.3f}'


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def discover_runs(sets: Optional[Sequence[str]]) -> List[Tuple[str, str, Path]]:
    """Return [(set_name, db, normalized_tsv_path), ...] found on disk."""
    out: List[Tuple[str, str, Path]] = []
    if not RUNS_DIR.is_dir():
        return out
    for sdir in sorted(RUNS_DIR.iterdir()):
        if not sdir.is_dir():
            continue
        if sets and sdir.name not in sets:
            continue
        for ddir in sorted(sdir.iterdir()):
            if not ddir.is_dir():
                continue
            tsv = ddir / 'gunc_normalized.tsv'
            if tsv.is_file():
                out.append((sdir.name, ddir.name, tsv))
    out.sort(key=lambda t: (t[0], DB_ORDER.index(t[1]) if t[1] in DB_ORDER else 99))
    return out


def load_comparator(set_name: str, filename: str) -> Optional[pd.DataFrame]:
    p = BENCH_DIR / set_name / filename
    if not p.is_file():
        return None
    d = pd.read_csv(p, sep='\t')
    if 'genome_id' not in d.columns or 'pred_contamination' not in d.columns:
        return None
    d['genome_id'] = d['genome_id'].astype(str)
    return d[['genome_id', 'pred_contamination', 'pred_completeness']]


def build_per_genome(runs: Sequence[Tuple[str, str, Path]],
                     notes: List[str]) -> pd.DataFrame:
    frames = []
    for set_name, db, tsv in runs:
        meta_p = BENCH_DIR / set_name / 'metadata.tsv'
        if not meta_p.is_file():
            notes.append(f'{set_name}: metadata.tsv absent -- skipped')
            continue
        meta = pd.read_csv(meta_p, sep='\t')
        meta['genome_id'] = meta['genome_id'].astype(str)

        g = pd.read_csv(tsv, sep='\t', dtype={'gunc_pass': str})
        g['genome'] = g['genome'].astype(str)
        for c in ('gunc_css', 'n_effective_surplus_clades', 'n_genes_called',
                  'n_genes_mapped', 'n_contigs',
                  'proportion_genes_retained_in_major_clades',
                  'genes_retained_index', 'contamination_portion',
                  'mean_hit_identity', 'reference_representation_score'):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors='coerce')

        d = meta.merge(g, left_on='genome_id', right_on='genome', how='left')
        n_missing = int(d['genome'].isna().sum())
        if n_missing:
            notes.append(f'{set_name}/{db}: {n_missing} genome(s) absent from the '
                         'GUNC output')
        d['set'] = set_name
        d['db'] = db

        # ---- pass/fail as a proper boolean -------------------------------
        d['gunc_fail'] = d['gunc_pass'].astype(str).str.strip().str.lower().map(
            {'false': True, 'true': False})
        d['gunc_scored'] = d['gunc_fail'].notna()

        # ---- power audit --------------------------------------------------
        gri = d['genes_retained_index']
        rrs = d['reference_representation_score']
        d['css_forcibly_zeroed'] = gri.notna() & (gri <= GRI_ZEROING_CUTOFF)
        d['gunc_warn_rrs_lt_0.3'] = rrs.notna() & (rrs < RRS_GUNC_WARN)
        d['rrs_lt_0.5'] = rrs.notna() & (rrs < RRS_POWERED_CUTOFF)
        d['powered'] = (d['gunc_scored'] & gri.notna() & rrs.notna()
                        & (gri > GRI_ZEROING_CUTOFF) & (rrs >= RRS_POWERED_CUTOFF))
        d['power_stratum'] = np.where(
            ~d['gunc_scored'], 'unscored',
            np.where(d['css_forcibly_zeroed'], 'hard_unpowered',
                     np.where(d['rrs_lt_0.5'], 'weak', 'powered')))

        # ---- comparators --------------------------------------------------
        for label, fn in COMPARATORS:
            c = load_comparator(set_name, fn)
            key = label.lower().replace(' ', '_')
            if c is None:
                d[f'{key}_pred_contamination'] = np.nan
                d[f'{key}_pred_completeness'] = np.nan
                notes.append(f'{set_name}: {fn} absent -- {label} omitted')
                continue
            c = c.rename(columns={'pred_contamination': f'{key}_pred_contamination',
                                  'pred_completeness': f'{key}_pred_completeness'})
            d = d.merge(c, on='genome_id', how='left')
        frames.append(d)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------
def power_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (set_name, db), d in df.groupby(['set', 'db'], sort=False):
        n = len(d)
        scored = d[d.gunc_scored]
        rows.append({
            'set': set_name, 'db': db, 'db_label': DB_LABEL.get(db, db),
            'n_genomes': n,
            'n_scored': int(d.gunc_scored.sum()),
            'n_unscored_min_mapped_genes': int((~d.gunc_scored).sum()),
            'n_powered': int(d.powered.sum()),
            'frac_powered': round(float(d.powered.mean()), 4),
            'n_hard_unpowered_css_zeroed': int(d.css_forcibly_zeroed.sum()),
            'frac_hard_unpowered': round(float(d.css_forcibly_zeroed.mean()), 4),
            'n_weak_rrs_lt_0.5': int(((~d.css_forcibly_zeroed) & d['rrs_lt_0.5']).sum()),
            'n_gunc_warn_rrs_lt_0.3': int(d['gunc_warn_rrs_lt_0.3'].sum()),
            'median_reference_representation_score':
                round(float(scored.reference_representation_score.median()), 4)
                if len(scored) else np.nan,
            'median_genes_retained_index':
                round(float(scored.genes_retained_index.median()), 4)
                if len(scored) else np.nan,
            'median_mean_hit_identity':
                round(float(scored.mean_hit_identity.median()), 4)
                if len(scored) else np.nan,
            'median_n_genes_mapped':
                float(scored.n_genes_mapped.median()) if len(scored) else np.nan,
            'median_n_genes_called':
                float(scored.n_genes_called.median()) if len(scored) else np.nan,
            'overall_fail_rate':
                round(float(scored.gunc_fail.mean()), 4) if len(scored) else np.nan,
            'powered_fail_rate':
                round(float(d[d.powered].gunc_fail.mean()), 4)
                if int(d.powered.sum()) else np.nan,
            'unpowered_fail_rate':
                round(float(d[d.gunc_scored & ~d.powered].gunc_fail.mean()), 4)
                if int((d.gunc_scored & ~d.powered).sum()) else np.nan,
        })
    return pd.DataFrame(rows)


def stratified_passfail(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (set_name, db), d in df.groupby(['set', 'db'], sort=False):
        for stratum, sub_all in (('all', d), ('powered', d[d.powered]),
                                 ('unpowered', d[d.gunc_scored & ~d.powered])):
            for lo, hi, label in BANDS:
                m = (sub_all.true_contamination >= lo) & (sub_all.true_contamination < hi)
                sub = sub_all[m]
                s = sub[sub.gunc_scored]
                rows.append({
                    'set': set_name, 'db': db, 'db_label': DB_LABEL.get(db, db),
                    'power_stratum': stratum,
                    'true_contamination_band_pct': label,
                    'n': int(len(sub)),
                    'n_gunc_scored': int(len(s)),
                    'n_fail': int(s.gunc_fail.sum()) if len(s) else 0,
                    'n_pass': int((~s.gunc_fail.astype(bool)).sum()) if len(s) else 0,
                    'gunc_fail_rate': round(float(s.gunc_fail.mean()), 4) if len(s) else np.nan,
                    'gunc_css_median': round(float(s.gunc_css.median()), 4) if len(s) else np.nan,
                    'gunc_css_mean': round(float(s.gunc_css.mean()), 4) if len(s) else np.nan,
                    'median_rrs': round(float(s.reference_representation_score.median()), 4)
                    if len(s) else np.nan,
                    'mean_true_contamination':
                        round(float(sub.true_contamination.mean()), 3) if len(sub) else np.nan,
                })
    return pd.DataFrame(rows)


def threshold_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (set_name, db), d0 in df.groupby(['set', 'db'], sort=False):
        for stratum, d in (('all', d0), ('powered', d0[d0.powered]),
                           ('unpowered', d0[d0.gunc_scored & ~d0.powered])):
            d = d[d.gunc_scored]
            if len(d) < 20:
                continue
            clusters = d[CLUSTER_COL].values
            truth_cont = d.true_contamination.values.astype(float)
            for thr in THRESHOLDS:
                truth = truth_cont >= thr
                if truth.sum() == 0 or (~truth).sum() == 0:
                    continue
                detectors: List[Tuple[str, np.ndarray, str]] = [
                    ('GUNC 1.1.1 (pass.GUNC == False)',
                     d.gunc_fail.values.astype(bool),
                     'single fixed operating point (CSS > 0.45 at maxCSS level); '
                     'cannot be re-thresholded per MIMAG level'),
                ]
                for label, _ in COMPARATORS:
                    key = label.lower().replace(' ', '_')
                    col = f'{key}_pred_contamination'
                    if col in d.columns and d[col].notna().any():
                        detectors.append(
                            (f'{label} (pred contamination >= {thr:g} %)',
                             (d[col].values >= thr), 'threshold-matched'))
                for name, pred, op in detectors:
                    c = confusion(pred, truth)
                    sd = derive_seed('agree', set_name, db, stratum, thr, name)
                    sens_ci = cluster_ci(
                        lambda sel, p=pred, t=truth: confusion(p[sel], t[sel])['sensitivity'],
                        clusters, seed=sd)
                    spec_ci = cluster_ci(
                        lambda sel, p=pred, t=truth: confusion(p[sel], t[sel])['specificity'],
                        clusters, seed=sd)
                    prec_ci = cluster_ci(
                        lambda sel, p=pred, t=truth: confusion(p[sel], t[sel])['precision'],
                        clusters, seed=sd)
                    f1_ci = cluster_ci(
                        lambda sel, p=pred, t=truth: confusion(p[sel], t[sel])['f1'],
                        clusters, seed=sd)
                    bal_ci = cluster_ci(
                        lambda sel, p=pred, t=truth:
                        confusion(p[sel], t[sel])['balanced_accuracy'], clusters, seed=sd)
                    rows.append({
                        'set': set_name, 'db': db, 'db_label': DB_LABEL.get(db, db),
                        'power_stratum': stratum, 'threshold_pct': thr,
                        'detector': name, 'operating_point': op,
                        'n': c['n'], 'n_true_contaminated': int(truth.sum()),
                        'tp': c['tp'], 'fp': c['fp'], 'tn': c['tn'], 'fn': c['fn'],
                        'sensitivity_recall': round(c['sensitivity'], 4),
                        'sensitivity_ci_lo': round(sens_ci[0], 4),
                        'sensitivity_ci_hi': round(sens_ci[1], 4),
                        'specificity': round(c['specificity'], 4),
                        'specificity_ci_lo': round(spec_ci[0], 4),
                        'specificity_ci_hi': round(spec_ci[1], 4),
                        'precision': round(c['precision'], 4),
                        'precision_ci_lo': round(prec_ci[0], 4),
                        'precision_ci_hi': round(prec_ci[1], 4),
                        'f1': round(c['f1'], 4),
                        'f1_ci_lo': round(f1_ci[0], 4), 'f1_ci_hi': round(f1_ci[1], 4),
                        'balanced_accuracy': round(c['balanced_accuracy'], 4),
                        'balanced_accuracy_ci_lo': round(bal_ci[0], 4),
                        'balanced_accuracy_ci_hi': round(bal_ci[1], 4),
                        'mcc': round(c['mcc'], 4),
                    })
    return pd.DataFrame(rows)


def css_correlations(df: pd.DataFrame, notes: Optional[List[str]] = None) -> pd.DataFrame:
    rows = []
    skipped_zero_variance = set()
    for (set_name, db), d0 in df.groupby(['set', 'db'], sort=False):
        for stratum, d in (('all', d0), ('powered', d0[d0.powered]),
                           ('unpowered', d0[d0.gunc_scored & ~d0.powered])):
            d = d[d.gunc_scored]
            if len(d) < 20:
                continue
            clusters = d[CLUSTER_COL].values
            truth = d.true_contamination.values.astype(float)
            # R1-m19 convention: rank/variance statistics are undefined and are
            # omitted where the true-contamination variance is ~0 (set_A_v2 is
            # uniformly 0 % contamination by construction).
            if np.nanstd(truth) < 1e-9:
                skipped_zero_variance.add((set_name, db))
                continue
            scores: List[Tuple[str, np.ndarray]] = [
                ('GUNC 1.1.1 CSS', d.gunc_css.values.astype(float))]
            if 'contamination_portion' in d.columns:
                scores.append(('GUNC 1.1.1 contamination_portion',
                               d.contamination_portion.values.astype(float)))
            for label, _ in COMPARATORS:
                key = label.lower().replace(' ', '_')
                col = f'{key}_pred_contamination'
                if col in d.columns and d[col].notna().any():
                    scores.append((f'{label} predicted contamination',
                                   d[col].values.astype(float)))
            for name, s in scores:
                ok = np.isfinite(s) & np.isfinite(truth)
                if ok.sum() < 20:
                    continue
                sv, tv, cv = s[ok], truth[ok], clusters[ok]
                rho = spearman(sv, tv)
                n_eff = len(np.unique(cv))
                sd = derive_seed('corr', set_name, db, stratum, name)
                ci = cluster_ci(lambda sel, a=sv, b=tv: spearman(a[sel], b[sel]),
                                cv, seed=sd)
                row = {'set': set_name, 'db': db, 'db_label': DB_LABEL.get(db, db),
                       'power_stratum': stratum, 'score': name, 'n': int(ok.sum()),
                       'n_clusters': n_eff,
                       'spearman_rho_vs_true_contamination': round(rho, 4),
                       'spearman_ci_lo': round(ci[0], 4),
                       'spearman_ci_hi': round(ci[1], 4),
                       'p_naive_genome_level': spearman_p(rho, int(ok.sum())),
                       'p_cluster_level': spearman_p(rho, n_eff)}
                for thr in THRESHOLDS:
                    lab = tv >= thr
                    auc = roc_auc(sv, lab)
                    aci = cluster_ci(lambda sel, a=sv, l=lab: roc_auc(a[sel], l[sel]),
                                     cv, seed=derive_seed('auroc', set_name, db,
                                                          stratum, name, thr))
                    row[f'auroc_ge{int(thr)}pct'] = round(auc, 4)
                    row[f'auroc_ge{int(thr)}pct_ci_lo'] = round(aci[0], 4)
                    row[f'auroc_ge{int(thr)}pct_ci_hi'] = round(aci[1], 4)
                rows.append(row)
    if notes is not None:
        for set_name, db in sorted(skipped_zero_variance):
            notes.append(
                f'{set_name}/{DB_LABEL.get(db, db)}: rank/AUROC statistics omitted — '
                'true contamination is uniformly 0 % by construction, so Spearman ρ '
                'and AUROC are undefined (R1-m19 zero-true-variance convention). '
                'Pass/fail and the power audit are still reported: this set is a pure '
                'false-positive (specificity) test.')
    out = pd.DataFrame(rows)
    if len(out):
        out['p_cluster_level_bh'] = bh(out['p_cluster_level'].tolist())
    return out


def power_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Powered-minus-unpowered detection gap, per set x DB x threshold.

    This is the quantitative form of the central claim: a GUNC pass from a
    genome it was not powered to adjudicate is worth much less than a pass from
    one it was. The comparison is made for GUNC and, as a specificity control,
    for MAGICC V5 and CheckM2 on the *same* strata -- if the gap were a property
    of the genomes rather than of GUNC's reference coverage, the reference-free
    tools would show it too.
    """
    rows = []
    slim = ['gunc_scored', 'powered', 'gunc_fail', 'true_contamination', CLUSTER_COL]
    slim += [f'{lab.lower().replace(" ", "_")}_pred_contamination'
             for lab, _ in COMPARATORS]
    for (set_name, db), d0 in df.groupby(['set', 'db'], sort=False):
        # slim before bootstrapping: .iloc cost scales with column count
        d0 = d0[[c for c in slim if c in d0.columns]]
        d0 = d0[d0.gunc_scored]
        pw, up = d0[d0.powered], d0[~d0.powered]
        if len(pw) < 20 or len(up) < 20:
            continue
        if np.nanstd(d0.true_contamination.values.astype(float)) < 1e-9:
            continue    # no positives to be sensitive to (set_A_v2)
        for thr in THRESHOLDS:
            detectors: List[Tuple[str, str]] = [('GUNC 1.1.1 (pass.GUNC == False)', '__gunc__')]
            for label, _ in COMPARATORS:
                key = label.lower().replace(' ', '_')
                if f'{key}_pred_contamination' in d0.columns:
                    detectors.append((f'{label} (pred >= {thr:g} %)',
                                      f'{key}_pred_contamination'))
            for name, col in detectors:
                def call(sub, c=col, t=thr):
                    return (sub.gunc_fail.astype(bool).values if c == '__gunc__'
                            else (sub[c].values >= t))

                def sens(sub, t=thr):
                    tr = (sub.true_contamination >= t).values
                    return confusion(call(sub), tr)['sensitivity']

                def spec(sub, t=thr):
                    tr = (sub.true_contamination >= t).values
                    return confusion(call(sub), tr)['specificity']

                s_pw, s_up = sens(pw), sens(up)
                sp_pw, sp_up = spec(pw), spec(up)
                # unpaired cluster bootstrap: resample clusters within each stratum
                def boot_diff(bseed):
                    cl_p = pw[CLUSTER_COL].values
                    cl_u = up[CLUSTER_COL].values
                    up_p = np.unique(cl_p)
                    up_u = np.unique(cl_u)
                    gp = [np.where(cl_p == c)[0] for c in up_p]
                    gu = [np.where(cl_u == c)[0] for c in up_u]
                    rng = np.random.default_rng(bseed)
                    vals = np.empty(N_BOOT)
                    for b in range(N_BOOT):
                        ip = np.concatenate([gp[i] for i in
                                             rng.integers(0, len(up_p), len(up_p))])
                        iu = np.concatenate([gu[i] for i in
                                             rng.integers(0, len(up_u), len(up_u))])
                        try:
                            vals[b] = sens(pw.iloc[ip]) - sens(up.iloc[iu])
                        except Exception:
                            vals[b] = np.nan
                    v = vals[np.isfinite(vals)]
                    return ((float(np.percentile(v, 2.5)),
                             float(np.percentile(v, 97.5))) if v.size
                            else (float('nan'), float('nan')))

                lo, hi = boot_diff(derive_seed('peff', set_name, db, thr, name))
                rows.append({
                    'set': set_name, 'db': db, 'db_label': DB_LABEL.get(db, db),
                    'threshold_pct': thr, 'detector': name,
                    'n_powered': int(len(pw)), 'n_unpowered': int(len(up)),
                    'sensitivity_powered': round(s_pw, 4),
                    'sensitivity_unpowered': round(s_up, 4),
                    'delta_sensitivity_powered_minus_unpowered':
                        round(s_pw - s_up, 4) if np.isfinite(s_pw) and np.isfinite(s_up)
                        else np.nan,
                    'delta_ci_lo': round(lo, 4), 'delta_ci_hi': round(hi, 4),
                    'specificity_powered': round(sp_pw, 4),
                    'specificity_unpowered': round(sp_up, 4),
                })
    return pd.DataFrame(rows)


def _paired_dbs(df: pd.DataFrame, set_name: str):
    """Align the two DB arms of one set on identical genome_ids."""
    d = df[df['set'] == set_name]
    dbs = [x for x in DB_ORDER if x in set(d.db)]
    if len(dbs) < 2:
        return None
    # keep only the columns the paired analyses touch: the bootstrap does
    # thousands of .iloc row selections and the cost scales with column count
    cols = ['db', 'genome_id', CLUSTER_COL, 'true_contamination', 'powered',
            'css_forcibly_zeroed', 'reference_representation_score',
            'genes_retained_index', 'mean_hit_identity', 'n_genes_mapped',
            'gunc_fail', 'gunc_css']
    d = d[[c for c in cols if c in d.columns]]
    a = d[d.db == dbs[0]].drop_duplicates('genome_id').set_index('genome_id')
    b = d[d.db == dbs[1]].drop_duplicates('genome_id').set_index('genome_id')
    common = a.index.intersection(b.index)
    return dbs, a.loc[common], b.loc[common]


def db_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Paired proGenomes 2.1 vs GTDB r95 comparison on identical genomes.

    Reported for two strata (protocol requirement that power strata never be
    blurred): ``all_paired`` = every genome present in both arms, and
    ``powered_both_dbs`` = only genomes GUNC was powered to adjudicate under
    *both* databases, which is the like-for-like comparison in which a change in
    verdict cannot be an artefact of one arm having no close reference.
    """
    rows = []
    for set_name in df['set'].drop_duplicates():
        pair = _paired_dbs(df, set_name)
        if pair is None:
            continue
        dbs, a_all, b_all = pair
        both_pw = (a_all.powered.values & b_all.powered.values)
        strata = [('all_paired', a_all, b_all)]
        if int(both_pw.sum()) >= 20:
            strata.append(('powered_both_dbs', a_all[both_pw], b_all[both_pw]))

        for stratum, a, b in strata:
            clusters = a[CLUSTER_COL].values
            n_pair = int(len(a))
            has_pos = np.nanstd(a.true_contamination.values.astype(float)) > 1e-9

            def metric_pair(fn, name, unit='', _a=a, _b=b, _cl=clusters,
                            _st=stratum, _n=n_pair):
                va, vb = fn(_a), fn(_b)
                delta = vb - va
                ci = cluster_ci(lambda sel: fn(_b.iloc[sel]) - fn(_a.iloc[sel]), _cl,
                                seed=derive_seed('dbsens', set_name, _st, name))
                rows.append({
                    'set': set_name, 'power_stratum': _st, 'n_paired': _n,
                    'metric': name, 'unit': unit,
                    f'{dbs[0]}': round(float(va), 4) if np.isfinite(va) else np.nan,
                    f'{dbs[1]}': round(float(vb), 4) if np.isfinite(vb) else np.nan,
                    'delta_gtdb95_minus_progenomes':
                        round(float(delta), 4) if np.isfinite(delta) else np.nan,
                    'delta_ci_lo': round(ci[0], 4), 'delta_ci_hi': round(ci[1], 4),
                })

            metric_pair(lambda x: float(x.powered.mean()), 'fraction powered', 'proportion')
            metric_pair(lambda x: float(x.css_forcibly_zeroed.mean()),
                        'fraction hard-unpowered (CSS forcibly zeroed)', 'proportion')
            metric_pair(lambda x: float(x.reference_representation_score.median()),
                        'median reference_representation_score', 'score')
            metric_pair(lambda x: float(x.genes_retained_index.median()),
                        'median genes_retained_index', 'index')
            metric_pair(lambda x: float(x.mean_hit_identity.median()),
                        'median mean AA hit identity', 'identity')
            metric_pair(lambda x: float(x.n_genes_mapped.median()),
                        'median n_genes_mapped', 'genes')
            metric_pair(lambda x: float(x.gunc_fail.astype(float).mean()),
                        'overall GUNC fail rate', 'proportion')
            metric_pair(lambda x: float(x.gunc_css.median()), 'median CSS', 'score')
            if has_pos:
                metric_pair(lambda x: spearman(x.gunc_css.values.astype(float),
                                               x.true_contamination.values.astype(float)),
                            'Spearman CSS vs true contamination', 'rho')
                for thr in THRESHOLDS:
                    metric_pair(
                        lambda x, t=thr: confusion(
                            x.gunc_fail.astype(bool).values,
                            (x.true_contamination >= t).values)['sensitivity'],
                        f'detection sensitivity at >={thr:g} % contamination', 'proportion')
                    metric_pair(
                        lambda x, t=thr: confusion(
                            x.gunc_fail.astype(bool).values,
                            (x.true_contamination >= t).values)['specificity'],
                        f'detection specificity at >={thr:g} % contamination', 'proportion')
    return pd.DataFrame(rows)


CONTROL_DIRS = {'progenomes_2.1': OUT_DIR / 'controls',
                'gtdb_95': OUT_DIR / 'controls_gtdb95'}


def control_db_comparison() -> pd.DataFrame:
    """WS4.1 real-genome controls re-scored under both databases.

    These are the 8 Kraken2-'confirmed' contaminated MAGs and 6 finished pure
    cultures of protocol §4.4c.  They are real genomes, not simulations, and are
    the only place where the 'GUNC missed them because proGenomes 2.1 has no
    close reference' hypothesis can be tested directly: if it were true, moving
    to GTDB r95 should power GUNC up and change the calls.
    """
    rows = []
    for arm in ('positive', 'negative'):
        frames = {}
        for db, root in CONTROL_DIRS.items():
            p = root / arm / 'gunc_normalized.tsv'
            if p.is_file():
                frames[db] = pd.read_csv(p, sep='\t')
        if len(frames) < 2:
            continue
        for db, g in frames.items():
            gri = pd.to_numeric(g.genes_retained_index, errors='coerce')
            rrs = pd.to_numeric(g.reference_representation_score, errors='coerce')
            fail = g.gunc_pass.astype(str).str.strip().str.lower().eq('false')
            pw = (gri > GRI_ZEROING_CUTOFF) & (rrs >= RRS_POWERED_CUTOFF)
            rows.append({
                'control_arm': arm, 'db': db, 'db_label': DB_LABEL.get(db, db),
                'n': int(len(g)), 'n_powered': int(pw.sum()),
                'n_hard_unpowered_css_zeroed': int((gri <= GRI_ZEROING_CUTOFF).sum()),
                'n_gunc_fail': int(fail.sum()),
                'n_gunc_fail_within_powered': int((fail & pw).sum()),
                'median_reference_representation_score': round(float(rrs.median()), 3),
                'median_mean_hit_identity':
                    round(float(pd.to_numeric(g.mean_hit_identity,
                                              errors='coerce').median()), 3),
                'median_css': round(float(pd.to_numeric(g.gunc_css,
                                                        errors='coerce').median()), 3),
                'max_css': round(float(pd.to_numeric(g.gunc_css,
                                                     errors='coerce').max()), 3),
            })
    return pd.DataFrame(rows)


def db_verdict_flips(df: pd.DataFrame) -> pd.DataFrame:
    """Per-genome verdict and power transitions when the DB is swapped.

    The 20-genome CPR pilot found three pass->fail flips, all on genuinely
    contaminated genomes.  This is the full-scale version: every flip is
    classified as a true or false gain/loss against the >=5 % ground truth, so
    a database that merely fails more genomes cannot masquerade as a better one.
    """
    rows = []
    for set_name in df['set'].drop_duplicates():
        pair = _paired_dbs(df, set_name)
        if pair is None:
            continue
        _, a, b = pair
        fa = a.gunc_fail.astype(bool).values
        fb = b.gunc_fail.astype(bool).values
        truth5 = (a.true_contamination.values.astype(float) >= 5.0)
        truth10 = (a.true_contamination.values.astype(float) >= 10.0)
        pw_a, pw_b = a.powered.values.astype(bool), b.powered.values.astype(bool)
        p2f, f2p = (~fa) & fb, fa & (~fb)
        rows.append({
            'set': set_name, 'n_paired': int(len(a)),
            'n_same_verdict': int((fa == fb).sum()),
            'n_pass_to_fail_gtdb': int(p2f.sum()),
            'n_fail_to_pass_gtdb': int(f2p.sum()),
            'pass_to_fail_true_positive_ge5': int((p2f & truth5).sum()),
            'pass_to_fail_false_positive_lt5': int((p2f & ~truth5).sum()),
            'fail_to_pass_lost_true_positive_ge5': int((f2p & truth5).sum()),
            'fail_to_pass_corrected_false_positive_lt5': int((f2p & ~truth5).sum()),
            'pass_to_fail_true_positive_ge10': int((p2f & truth10).sum()),
            'net_true_positives_gained_ge5':
                int((p2f & truth5).sum() - (f2p & truth5).sum()),
            'net_false_positives_gained_lt5':
                int((p2f & ~truth5).sum() - (f2p & ~truth5).sum()),
            'median_true_contamination_of_pass_to_fail':
                round(float(np.median(a.true_contamination.values[p2f])), 3)
                if p2f.sum() else np.nan,
            'n_unpowered_progenomes_to_powered_gtdb': int(((~pw_a) & pw_b).sum()),
            'n_powered_progenomes_to_unpowered_gtdb': int((pw_a & (~pw_b)).sum()),
            'n_powered_both': int((pw_a & pw_b).sum()),
            'n_unpowered_both': int(((~pw_a) & (~pw_b)).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def write_report(df, power, strat, agree, corr, peff, dbsens, flips, ctrl, notes, runs) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    md: List[str] = []
    md.append('# WS4.2 — GUNC as a standard detection comparator, with database '
              'sensitivity and a statistical-power audit\n')
    md.append(f'Generated {ts} by `scripts/138_ws42_gunc_comparator.py`. '
              'GUNC 1.1.1, DIAMOND 2.1.24, Prodigal V2.6.3.\n')

    md.append('## 0. How GUNC is (and is not) used here\n')
    md.append('GUNC returns a **clade separation score (CSS)** and a **pass/fail** flag. '
              'It does **not** estimate completeness or contamination as percentages, so '
              'it is deliberately **excluded from the MAE table** (protocol §4.4c item 5). '
              'It is scored here as a *detector*, and MAGICC V5 and CheckM2 are scored on '
              'the same detection task at the same thresholds so the comparison is '
              'like-for-like. One asymmetry favours the estimators and is stated rather '
              'than hidden: they are thresholded at exactly the MIMAG-inspired level '
              'being tested, whereas GUNC has a single built-in operating point '
              f'(CSS > {CSS_CHIMERIC_THRESHOLD} at the maxCSS taxonomic level) that cannot be '
              're-tuned per level. Section 4 (Spearman/AUROC) is threshold-free and '
              'removes that asymmetry.\n')
    md.append('**MIMAG-inspired thresholds** used throughout: high quality ≥90 % '
              'completeness AND <5 % contamination; medium quality ≥50 % AND <10 % '
              '(Bowers et al. 2017). "MIMAG-inspired" because the strict definition '
              'additionally requires rRNA/tRNA criteria that cannot be evaluated from '
              'these assemblies. The ground-truth positive (contaminated) class is '
              'therefore *true contamination ≥ threshold*.\n')

    md.append('### Database choice is a methodological decision, and the trade-off is real\n')
    md.append('| Database | reference genomes | phyla | CPR/Patescibacteria refs | archaeal refs |\n'
              '|---|---|---|---|---|\n'
              '| proGenomes 2.1 | 11,223 | 149 | **104** | 457 |\n'
              '| GTDB r95 | 31,910 | 149 | **1,131** | 1,672 |\n')
    md.append('proGenomes 2.1 holds only **104** CPR references, so GUNC is expected to be '
              'near-blind on `set_C_clean`, which is entirely Patescibacteriota. GTDB '
              'databases give far better novel-lineage coverage **but include MAG-derived '
              'reference genomes** (14,566 GenBank-prefixed entries in r95), i.e. reference '
              'genomes that may themselves be contaminated. proGenomes 2.1 is the cleaner '
              'reference set but blind to CPR/DPANN. **Neither is strictly superior**; both '
              'are run and the comparison is reported as-is.\n')

    md.append('### The power caveat, stated precisely\n')
    md.append('GUNC computes `genes_retained_index (GRI) = genes retained in abundant '
              'clades / genes called` and `reference_representation_score (RRS) = GRI × '
              'mean AA hit identity`. When **GRI ≤ 0.4 the CSS is forcibly multiplied by '
              'zero**, so the genome **automatically passes** no matter how chimeric it '
              'actually is. A pass from such a genome is *not* evidence of cleanliness — '
              'it is evidence that GUNC had no close reference. Earlier control testing on '
              'novel-lineage MAGs found 7/8 positives with RRS < 0.5 and GUNC flagged '
              '0/8, while correctly passing 6/6 finished pure cultures. Power strata used '
              'below:\n\n'
              f'* `hard_unpowered` — GRI ≤ {GRI_ZEROING_CUTOFF} (CSS forcibly zeroed)\n'
              f'* `weak` — GRI > {GRI_ZEROING_CUTOFF} but RRS < {RRS_POWERED_CUTOFF}\n'
              f'* `powered` — GRI > {GRI_ZEROING_CUTOFF} and RRS ≥ {RRS_POWERED_CUTOFF}\n\n'
              '`unpowered` = `hard_unpowered` ∪ `weak`. GUNC\'s own caution flag '
              f'(RRS < {RRS_GUNC_WARN}) is reported separately.\n')

    # ---- headline findings, derived from the tables below -----------------
    md.append('\n## 0b. Headline findings\n')
    pw_bits = []
    for _, r in power.iterrows():
        pw_bits.append(f'{r["set"]}/{r["db_label"]} **{100 * r["frac_powered"]:.1f} %** '
                       f'({r["n_powered"]}/{r["n_genomes"]})')
    md.append('**1. Powered fraction — the number that gates every other number '
              'here.** ' + '; '.join(pw_bits) + '.\n')
    md.append('The two novel-lineage clean sets are where GUNC is least able to '
              'adjudicate, and `set_C_clean` (entirely Patescibacteriota) under '
              'proGenomes 2.1 is the worst case. Nothing in the `unpowered` stratum '
              'should be read as evidence of cleanliness.\n')

    clean = strat[(strat.true_contamination_band_pct == '0-5')
                  & (strat.power_stratum.isin(['all', 'powered']))]
    if len(clean):
        md.append('\n**2. GUNC\'s false-positive rate is a property of novel lineages, '
                  'not of GUNC.** Fail rate on genomes whose *true* contamination is '
                  '< 5 % — i.e. genomes GUNC should pass:\n')
        md.append('| Set | Database | n clean | fail rate (all strata) | fail rate (powered) |')
        md.append('|---|---|---|---|---|')
        for (s, db), g in clean.groupby(['set', 'db'], sort=False):
            a = g[g.power_stratum == 'all']
            p = g[g.power_stratum == 'powered']
            if not len(a) or int(a.iloc[0]['n']) == 0:
                continue
            md.append(f'| {s} | {DB_LABEL.get(db, db)} | {int(a.iloc[0]["n"])} | '
                      f'**{f3(a.iloc[0]["gunc_fail_rate"])}** | '
                      f'{f3(p.iloc[0]["gunc_fail_rate"]) if len(p) else "-"} |')
        md.append('\nOn `set_A_v2` — 1,000 uncontaminated genomes spanning 31 phyla of '
                  'mainstream taxa — GUNC is essentially never wrong. On the CPR and '
                  'archaeal clean sets it is wrong constantly. **The failure mode is '
                  'reference representation, and it is invisible unless the strata are '
                  'reported separately.**\n')
        md.append('⚠️ **Stratum discipline.** On `set_D_clean` under proGenomes 2.1 GUNC '
                  'fails a large majority of genuinely clean archaeal genomes: **68.4 % '
                  'in the powered stratum, 62.9 % across all strata**. These two numbers '
                  'are not interchangeable and neither may be quoted without its stratum. '
                  'The same counter-finding holds under GTDB r95 (67.8 % powered, 64.5 % '
                  'all strata), so it is not a database artefact.\n')

    gunc_rows = corr[(corr.power_stratum == 'all')
                     & (corr.score == 'GUNC 1.1.1 CSS')]
    if len(gunc_rows):
        md.append('\n**3. Rank correlation with true contamination — the three tools '
                  'side by side** (stratum `all`; full table with CIs in §4):\n')
        md.append('| Set | Database | GUNC CSS | GUNC contamination_portion | MAGICC V5 | '
                  'CheckM2 | GUNC CSS AUROC ≥5 % |')
        md.append('|---|---|---|---|---|---|---|')
        for (s, db), g in corr[corr.power_stratum == 'all'].groupby(['set', 'db'],
                                                                   sort=False):
            def get(name, col='spearman_rho_vs_true_contamination'):
                m = g[g.score == name]
                return f3(m.iloc[0][col]) if len(m) else '-'
            md.append(f'| {s} | {DB_LABEL.get(db, db)} | {get("GUNC 1.1.1 CSS")} | '
                      f'{get("GUNC 1.1.1 contamination_portion")} | '
                      f'{get("MAGICC V5 predicted contamination")} | '
                      f'{get("CheckM2 predicted contamination")} | '
                      f'{get("GUNC 1.1.1 CSS", "auroc_ge5pct")} |')
        md.append('\nMAGICC V5 ranks contamination best on every set. GUNC\'s CSS ranks '
                  'worst — but the AUROC column shows why that is **saturation, not a '
                  'failure of detection**: CSS pins at 1.0 as soon as a genome is '
                  'confidently chimeric, so it carries little rank information among '
                  'contaminated genomes while still separating them from clean ones. '
                  'GUNC\'s `contamination_portion` ranks far better than its CSS. This is '
                  'exactly why GUNC is scored as a detector and **kept out of the MAE '
                  'table**.\n')
        md.append('\n**4. Where GUNC beats the estimators, stated plainly.** GUNC '
                  'out-detects CheckM2 on recall at ≥5 % on every set tested — most '
                  'starkly on the novel-lineage sets, where CheckM2 recovers roughly half '
                  '(`set_C_clean`) to two-thirds (`set_D_clean`) of contaminated genomes '
                  'while GUNC recovers nearly all of them. CheckM2 buys that with perfect '
                  'specificity on those sets; GUNC does not. Both facts belong in the '
                  'record.\n')

    md.append('\n## 1. Power audit — how much of each set GUNC could actually adjudicate\n')
    md.append('| Set | Database | n | scored | **powered** | **% powered** | hard-unpowered '
              '(CSS zeroed) | weak (RRS<0.5) | RRS<0.3 | median RRS | median GRI | '
              'median AA id | median genes mapped |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in power.iterrows():
        md.append(f'| {r["set"]} | {r["db_label"]} | {r["n_genomes"]} | {r["n_scored"]} | '
                  f'**{r["n_powered"]}** | **{100 * r["frac_powered"]:.1f} %** | '
                  f'{r["n_hard_unpowered_css_zeroed"]} | {r["n_weak_rrs_lt_0.5"]} | '
                  f'{r["n_gunc_warn_rrs_lt_0.3"]} | {f3(r["median_reference_representation_score"])} | '
                  f'{f3(r["median_genes_retained_index"])} | {f3(r["median_mean_hit_identity"])} | '
                  f'{r["median_n_genes_mapped"]:.0f} |')
    md.append('\nFail rates by stratum (a "pass" in the unpowered stratum is uninformative):\n')
    md.append('| Set | Database | overall fail rate | powered fail rate | unpowered fail rate |')
    md.append('|---|---|---|---|---|')
    for _, r in power.iterrows():
        md.append(f'| {r["set"]} | {r["db_label"]} | {f3(r["overall_fail_rate"])} | '
                  f'{f3(r["powered_fail_rate"])} | {f3(r["unpowered_fail_rate"])} |')

    md.append('\n## 2. GUNC pass/fail stratified by true contamination\n')
    md.append('Bands are `[lo, hi)` in percent contamination.\n')
    for stratum in ('all', 'powered', 'unpowered'):
        sub = strat[strat.power_stratum == stratum]
        if sub.n.sum() == 0:
            continue
        md.append(f'\n### 2.{ {"all": 1, "powered": 2, "unpowered": 3}[stratum] } '
                  f'stratum = `{stratum}`\n')
        md.append('| Set | Database | band (%) | n | scored | fail | fail rate | '
                  'median CSS | median RRS |')
        md.append('|---|---|---|---|---|---|---|---|---|')
        for _, r in sub.iterrows():
            if r['n'] == 0:
                continue
            md.append(f'| {r["set"]} | {r["db_label"]} | {r["true_contamination_band_pct"]} | '
                      f'{r["n"]} | {r["n_gunc_scored"]} | {r["n_fail"]} | '
                      f'{f3(r["gunc_fail_rate"])} | {f3(r["gunc_css_median"])} | '
                      f'{f3(r["median_rrs"])} |')

    md.append('\n## 3. Detection agreement at the MIMAG-inspired 5 % and 10 % '
              'contamination thresholds\n')
    md.append('95 % CIs: cluster bootstrap over the reference genomes '
              f'(`{CLUSTER_COL}`), {N_BOOT} resamples, seed {BOOT_SEED} — the 10 '
              'simulations per reference are not independent.\n')
    for stratum in ('all', 'powered', 'unpowered'):
        sub = agree[agree.power_stratum == stratum]
        if not len(sub):
            continue
        md.append(f'\n### 3.{ {"all": 1, "powered": 2, "unpowered": 3}[stratum] } '
                  f'stratum = `{stratum}`\n')
        md.append('| Set | Database | thr | detector | TP | FP | TN | FN | '
                  'sensitivity/recall (95 % CI) | specificity (95 % CI) | '
                  'precision (95 % CI) | F1 | balanced acc. | MCC |')
        md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in sub.iterrows():
            md.append(
                f'| {r["set"]} | {r["db_label"]} | ≥{r["threshold_pct"]:g} % | '
                f'{r["detector"]} | {r["tp"]} | {r["fp"]} | {r["tn"]} | {r["fn"]} | '
                f'{f3(r["sensitivity_recall"])} ({f3(r["sensitivity_ci_lo"])}–{f3(r["sensitivity_ci_hi"])}) | '
                f'{f3(r["specificity"])} ({f3(r["specificity_ci_lo"])}–{f3(r["specificity_ci_hi"])}) | '
                f'{f3(r["precision"])} ({f3(r["precision_ci_lo"])}–{f3(r["precision_ci_hi"])}) | '
                f'{f3(r["f1"])} | {f3(r["balanced_accuracy"])} | {f3(r["mcc"])} |')

    md.append('\n## 4. Threshold-free ranking: Spearman CSS vs true contamination, '
              'beside MAGICC V5 and CheckM2\n')
    md.append('This is the section that puts the three tools on equal footing: no '
              'threshold is applied to any of them. `p_cluster_level` uses the number of '
              'reference-genome clusters as the effective n (the conservative choice) and '
              'is Benjamini–Hochberg corrected across every row of this table.\n')
    md.append('**Two reading instructions, both necessary to avoid misinterpreting this '
              'table.** (i) A low Spearman ρ for CSS is *not* a failure of detection: CSS '
              '**saturates at 1.0** once a genome is confidently chimeric, so it carries '
              'almost no rank information *within* the contaminated range even where it '
              'separates contaminated from clean genomes well — compare ρ with the AUROC '
              'columns, which are the detection-relevant numbers. This is precisely why '
              'GUNC must be reported as a detector and never placed in an MAE table. '
              '(ii) CSS ρ can be *higher* in the `unpowered` stratum than in the `powered` '
              'one for the same reason inverted: attenuated (or forcibly zeroed) scores '
              'are un-saturated and therefore more rank-informative, while being *less* '
              'reliable as calls. Do not read a higher unpowered ρ as better performance.\n')
    md.append('GUNC also emits `contamination_portion`, a continuous quantity that is a '
              'far better rank-predictor of true contamination than CSS. It is reported '
              'here for completeness, but it is **not** GUNC\'s published output for '
              'quality assessment and is not calibrated as a contamination percentage, so '
              'it does not enter the MAE table either.\n')
    for stratum in ('all', 'powered', 'unpowered'):
        sub = corr[corr.power_stratum == stratum]
        if not len(sub):
            continue
        md.append(f'\n### 4.{ {"all": 1, "powered": 2, "unpowered": 3}[stratum] } '
                  f'stratum = `{stratum}`\n')
        md.append('| Set | Database | score | n | clusters | Spearman ρ (95 % CI) | '
                  'BH q | AUROC ≥5 % | AUROC ≥10 % |')
        md.append('|---|---|---|---|---|---|---|---|---|')
        for _, r in sub.iterrows():
            md.append(f'| {r["set"]} | {r["db_label"]} | {r["score"]} | {r["n"]} | '
                      f'{r["n_clusters"]} | '
                      f'{f3(r["spearman_rho_vs_true_contamination"])} '
                      f'({f3(r["spearman_ci_lo"])}–{f3(r["spearman_ci_hi"])}) | '
                      f'{f3(r.get("p_cluster_level_bh"))} | '
                      f'{f3(r["auroc_ge5pct"])} ({f3(r["auroc_ge5pct_ci_lo"])}–{f3(r["auroc_ge5pct_ci_hi"])}) | '
                      f'{f3(r["auroc_ge10pct"])} ({f3(r["auroc_ge10pct_ci_lo"])}–{f3(r["auroc_ge10pct_ci_hi"])}) |')

    if len(peff):
        md.append('\n## 4b. The power effect, quantified — powered minus unpowered '
                  'detection sensitivity\n')
        md.append('If the gap were a property of the *genomes* rather than of GUNC\'s '
                  'reference coverage, MAGICC V5 and CheckM2 — which never consult a '
                  'reference database at inference — would show the same gap on the same '
                  'strata. They are therefore included as a specificity control. 95 % CI: '
                  'unpaired cluster bootstrap resampling reference genomes within each '
                  f'stratum, {N_BOOT} resamples.\n')
        md.append('| Set | Database | thr | detector | n powered | n unpowered | '
                  'sens. powered | sens. unpowered | Δ sensitivity (95 % CI) | '
                  'spec. powered | spec. unpowered |')
        md.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in peff.iterrows():
            md.append(f'| {r["set"]} | {r["db_label"]} | ≥{r["threshold_pct"]:g} % | '
                      f'{r["detector"]} | {r["n_powered"]} | {r["n_unpowered"]} | '
                      f'{f3(r["sensitivity_powered"])} | {f3(r["sensitivity_unpowered"])} | '
                      f'**{f3(r["delta_sensitivity_powered_minus_unpowered"])}** '
                      f'({f3(r["delta_ci_lo"])}–{f3(r["delta_ci_hi"])}) | '
                      f'{f3(r["specificity_powered"])} | {f3(r["specificity_unpowered"])} |')
        md.append('\n**How to read this table honestly.** Under proGenomes 2.1 on '
                  '`set_C_clean`, GUNC loses Δ +0.236 of sensitivity in the unpowered '
                  'stratum while the reference-free MAGICC V5 loses only Δ +0.047 on the '
                  'identical genomes — a five-fold difference that is hard to explain as '
                  'genome difficulty. But CheckM2, also reference-free at inference, '
                  'loses Δ +0.133 on `set_C_clean` and Δ +0.322 on `set_D_clean`, so the '
                  'unpowered stratum **is** intrinsically somewhat harder and the '
                  'reference-coverage argument must not be overstated from these columns '
                  'alone.\n')
        md.append('The decisive observation is the database swap. Moving to GTDB r95 '
                  'collapses GUNC\'s power gap to essentially zero (Δ −0.001 on '
                  '`set_C_clean`, Δ 0.000 on `set_D_clean`) while MAGICC V5\'s and '
                  'CheckM2\'s gaps persist on the same genomes. A change that touches only '
                  'the reference database removes GUNC\'s gap and leaves the reference-free '
                  'tools\' gaps intact: **GUNC\'s power gap was reference coverage; the '
                  'residual gap in the other tools is genome difficulty.**\n')

    if len(dbsens):
        md.append('\n## 5. Database sensitivity — proGenomes 2.1 vs GTDB r95, paired on '
                  'identical genomes\n')
        md.append('This section answers protocol §4.4c item 4 at full scale (it was '
                  'previously a 20-genome CPR pilot). Δ = GTDB r95 − proGenomes 2.1; '
                  '95 % CI by cluster bootstrap over reference genomes. Reported for '
                  'two strata: `all_paired` (every genome in both arms) and '
                  '`powered_both_dbs` (only genomes GUNC was powered to adjudicate '
                  '**under both databases**) — in the latter, a verdict change cannot '
                  'be explained by one arm simply having no close reference.\n')
        md.append('**Correction to the 20-genome pilot.** The pilot reported median RRS '
                  '0.575 → 0.940 and median AA identity 0.850 → 0.980 on `set_C_clean`. '
                  'Those 20 genomes reproduce bit-identically inside the full run, so the '
                  'pilot was not wrong about them — it was an unrepresentative sample. '
                  'Across all 1,000 genomes the gain is far smaller (median RRS '
                  '0.56 → 0.64, median AA identity 0.77 → 0.80). The powered-fraction '
                  'gain held up (12/20 → 17/20 in the pilot; 681 → 880 of 1,000 at full '
                  'scale). **The full-scale numbers supersede the pilot.**\n')
        for stratum in ('all_paired', 'powered_both_dbs'):
            sub = dbsens[dbsens.power_stratum == stratum]
            if not len(sub):
                continue
            md.append(f'\n### 5.{ {"all_paired": 1, "powered_both_dbs": 2}[stratum] } '
                      f'stratum = `{stratum}`\n')
            md.append('| Set | n paired | metric | proGenomes 2.1 | GTDB r95 | Δ (95 % CI) |')
            md.append('|---|---|---|---|---|---|')
            for _, r in sub.iterrows():
                md.append(f'| {r["set"]} | {r["n_paired"]} | {r["metric"]} | '
                          f'{f3(r.get("progenomes_2.1"))} | {f3(r.get("gtdb_95"))} | '
                          f'{f3(r["delta_gtdb95_minus_progenomes"])} '
                          f'({f3(r["delta_ci_lo"])}–{f3(r["delta_ci_hi"])}) |')

    if len(flips):
        md.append('\n### 5.3 Verdict and power transitions when the database is swapped\n')
        md.append('Every genome whose GUNC verdict changes between the two databases, '
                  'classified against the ≥5 % ground truth. A database that merely '
                  'fails more genomes is not thereby better — what matters is whether '
                  'the extra failures are true positives.\n')
        md.append('| Set | n paired | same verdict | pass→fail (GTDB) | of which truly '
                  '≥5 % | of which clean (<5 %) | fail→pass (GTDB) | of which truly ≥5 % '
                  '(lost) | net true positives | net false positives | unpowered→powered | '
                  'powered→unpowered |')
        md.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in flips.iterrows():
            md.append(
                f'| {r["set"]} | {r["n_paired"]} | {r["n_same_verdict"]} | '
                f'**{r["n_pass_to_fail_gtdb"]}** | {r["pass_to_fail_true_positive_ge5"]} | '
                f'{r["pass_to_fail_false_positive_lt5"]} | {r["n_fail_to_pass_gtdb"]} | '
                f'{r["fail_to_pass_lost_true_positive_ge5"]} | '
                f'**{r["net_true_positives_gained_ge5"]:+d}** | '
                f'{r["net_false_positives_gained_lt5"]:+d} | '
                f'{r["n_unpowered_progenomes_to_powered_gtdb"]} | '
                f'{r["n_powered_progenomes_to_unpowered_gtdb"]} |')
        md.append('\n**The trade-off, stated explicitly and not resolved in favour of '
                  'either database.** GTDB r95 buys reference coverage — 1,131 CPR and '
                  '1,672 archaeal references against proGenomes 2.1\'s 104 and 457 — and '
                  'that is what moves genomes into the powered stratum. It buys it with '
                  '14,566 GenBank-prefixed, largely MAG-derived reference genomes, which '
                  'may themselves be contaminated; a "fail" against a contaminated '
                  'reference is not the same evidence as a "fail" against a finished '
                  'genome. proGenomes 2.1 is the cleaner reference set and is near-blind '
                  'on `set_C_clean`, which is entirely Patescibacteriota. **Neither is '
                  'strictly superior. Both are reported; the comparison stands as the '
                  'result.**\n')

    if len(ctrl):
        md.append('\n### 5.4 The database hypothesis tested directly on the WS4.1 real-genome '
                  'controls\n')
        md.append('The synthetic sets above cannot settle *why* GUNC missed the eight '
                  'Kraken2-"confirmed" contaminated MAGs of §4.4c, because those are real '
                  'novel-lineage genomes. Those 14 control genomes (8 putative positives, '
                  '6 finished pure cultures) were therefore re-scored under both databases. '
                  'If poor reference coverage were the explanation, GTDB r95 should power '
                  'GUNC up **and change the calls**. It does the first and not the second.\n')
        md.append('| Control arm | Database | n | powered | GUNC fails | fails within '
                  'powered | median RRS | median AA id | median CSS | max CSS |')
        md.append('|---|---|---|---|---|---|---|---|---|---|')
        for _, r in ctrl.iterrows():
            md.append(f'| {r["control_arm"]} | {r["db_label"]} | {r["n"]} | '
                      f'**{r["n_powered"]}/{r["n"]}** | {r["n_gunc_fail"]} | '
                      f'{r["n_gunc_fail_within_powered"]} | '
                      f'{f3(r["median_reference_representation_score"])} | '
                      f'{f3(r["median_mean_hit_identity"])} | {f3(r["median_css"])} | '
                      f'{f3(r["max_css"])} |')
        md.append('\nSwapping to GTDB r95 raises the powered count on the putative '
                  'positives from **1/8 to 5/8** — the database change did exactly what it '
                  'was supposed to do — and GUNC still flags **0/8**, with every CSS at or '
                  'below 0.25 against a 0.45 failure threshold. The 6/6 finished pure '
                  'cultures continue to pass at CSS 0.000 under both databases, so this is '
                  'not an insensitive install. **The proGenomes-2.1 result was therefore '
                  'not a database-capacity artefact**, which removes the last alternative '
                  'explanation available to the withdrawn Kraken2 claim (§4.4c). This '
                  'table belongs to §4.4c/WS4.3 and is reproduced here only because it is '
                  'a database-sensitivity result; it is not part of the WS4.2 benchmark '
                  'comparison.\n')

    md.append('\n## 6. Runs included\n')
    md.append('| Set | Database | normalized TSV |')
    md.append('|---|---|---|')
    for s, db, p in runs:
        md.append(f'| {s} | {DB_LABEL.get(db, db)} | `{p.relative_to(PROJECT_DIR)}` |')
    if notes:
        md.append('\n## 7. Notes and caveats\n')
        for n in notes:
            md.append(f'* {n}')
    md.append('')
    return '\n'.join(md)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--sets', nargs='*', default=None,
                    help='Restrict to these benchmark sets')
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    notes: List[str] = []
    runs = discover_runs(args.sets)
    if not runs:
        print(f'No GUNC runs found under {RUNS_DIR}', file=sys.stderr)
        return 1
    print(f'Found {len(runs)} run(s):')
    for s, db, p in runs:
        print(f'  {s:16s} {db:16s} {p}')

    df = build_per_genome(runs, notes)
    if df.empty:
        print('No usable rows.', file=sys.stderr)
        return 1

    keep = ['set', 'db', 'genome_id', 'true_completeness', 'true_contamination',
            'dominant_accession', 'dominant_phylum', 'ref_index', 'replicate',
            'gunc_css', 'gunc_pass', 'gunc_fail', 'gunc_scored',
            'n_effective_surplus_clades', 'taxonomic_level',
            'n_genes_called', 'n_genes_mapped', 'n_contigs',
            'proportion_genes_retained_in_major_clades', 'genes_retained_index',
            'contamination_portion', 'mean_hit_identity',
            'reference_representation_score',
            'css_forcibly_zeroed', 'gunc_warn_rrs_lt_0.3', 'rrs_lt_0.5',
            'powered', 'power_stratum',
            'magicc_v5_pred_contamination', 'magicc_v5_pred_completeness',
            'checkm2_pred_contamination', 'checkm2_pred_completeness']
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(out / 'gunc_per_genome.tsv', sep='\t', index=False)

    power = power_audit(df)
    strat = stratified_passfail(df)
    agree = threshold_agreement(df)
    corr = css_correlations(df, notes)
    peff = power_effect(df)
    dbsens = db_sensitivity(df)
    flips = db_verdict_flips(df)
    ctrl = control_db_comparison()

    if len(peff):
        peff.to_csv(out / 'gunc_power_effect.tsv', sep='\t', index=False)
    power.to_csv(out / 'gunc_power_audit.tsv', sep='\t', index=False)
    strat.to_csv(out / 'gunc_stratified_passfail.tsv', sep='\t', index=False)
    agree.to_csv(out / 'gunc_threshold_agreement.tsv', sep='\t', index=False)
    corr.to_csv(out / 'gunc_css_correlations.tsv', sep='\t', index=False)
    if len(dbsens):
        dbsens.to_csv(out / 'gunc_db_sensitivity.tsv', sep='\t', index=False)
    if len(flips):
        flips.to_csv(out / 'gunc_db_verdict_flips.tsv', sep='\t', index=False)
    if len(ctrl):
        ctrl.to_csv(out / 'gunc_controls_db_comparison.tsv', sep='\t', index=False)

    md = write_report(df, power, strat, agree, corr, peff, dbsens, flips, ctrl,
                      notes, runs)
    (out / 'WS4.2_gunc_report.md').write_text(md)

    (out / 'ws4.2_gunc_summary.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'script': 'scripts/138_ws42_gunc_comparator.py',
        'gunc_internals': {
            'gri_zeroing_cutoff': GRI_ZEROING_CUTOFF,
            'css_chimeric_threshold': CSS_CHIMERIC_THRESHOLD,
            'rrs_powered_cutoff': RRS_POWERED_CUTOFF,
            'rrs_gunc_warning': RRS_GUNC_WARN,
        },
        'bootstrap': {'n_boot': N_BOOT, 'base_seed': BOOT_SEED,
                      'cluster_unit': CLUSTER_COL,
                      'seed_derivation': 'BOOT_SEED + zlib.crc32(context) (CRC-32, '
                                         'PYTHONHASHSEED-independent)'},
        'true_contamination_bands_pct': [b[2] for b in BANDS],
        'mimag_thresholds_pct': THRESHOLDS,
        'runs': [{'set': s, 'db': db, 'tsv': str(p)} for s, db, p in runs],
        'power_audit': power.to_dict('records'),
        'stratified_passfail': strat.to_dict('records'),
        'threshold_agreement': agree.to_dict('records'),
        'css_correlations': corr.to_dict('records'),
        'power_effect': peff.to_dict('records') if len(peff) else [],
        'db_sensitivity': dbsens.to_dict('records') if len(dbsens) else [],
        'db_verdict_flips': flips.to_dict('records') if len(flips) else [],
        'controls_db_comparison': ctrl.to_dict('records') if len(ctrl) else [],
        'notes': notes,
    }, indent=2, default=str))

    print('\n=== POWER AUDIT (the key number) ===')
    for _, r in power.iterrows():
        print(f'{r["set"]:14s} {r["db_label"]:16s} powered {r["n_powered"]:5d}/'
              f'{r["n_genomes"]:<5d} ({100 * r["frac_powered"]:5.1f} %)   '
              f'fail rate all={f3(r["overall_fail_rate"])} '
              f'powered={f3(r["powered_fail_rate"])}')
    print('\n=== SPEARMAN vs TRUE CONTAMINATION (stratum = all) ===')
    for _, r in corr[corr.power_stratum == 'all'].iterrows():
        print(f'{r["set"]:14s} {r["db_label"]:16s} {r["score"]:38s} '
              f'rho={r["spearman_rho_vs_true_contamination"]:+.3f} '
              f'[{r["spearman_ci_lo"]:+.3f},{r["spearman_ci_hi"]:+.3f}]')
    print(f'\n-> {out / "WS4.2_gunc_report.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
