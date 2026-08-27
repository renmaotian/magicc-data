#!/usr/bin/env python3
"""
WS1.5 / WS4.2 -- GUNC as a contamination *detector* on the clean Sets C/D.

GUNC reports a clade separation score (CSS) and a pass/fail flag; it does not
estimate contamination as a percentage. Forcing it into an MAE table would be
both unfair to GUNC and wrong, so it is evaluated the way it is meant to be
used - as a detector - and the quantitative tools are evaluated on the *same*
detection task so the comparison is like for like. This is the direct answer to
Reviewer 2's major comment 4.

Three analyses per clean set
---------------------------
1. **Pass/fail stratified by true contamination.** GUNC fail rate and mean CSS in
   true-contamination bands 0, (0-1], (1-5], (5-10], (10-20], (20-50], (50-100] %.

2. **Detection agreement at the MIMAG-inspired thresholds 5 % and 10 %.**
   The MIMAG-inspired criteria used throughout this revision are high quality
   >= 90 % completeness AND < 5 % contamination, medium quality >= 50 % AND
   < 10 % contamination (Bowers et al. 2017); "MIMAG-inspired" because the strict
   definition also requires rRNA/tRNA criteria that cannot be evaluated from these
   assemblies. Because the pass criterion is "< threshold", the ground-truth
   positive (contaminated) class is **true contamination >= threshold**. Decision
   rules:
       GUNC       pass.GUNC == False                (ONE fixed operating point;
                                                     GUNC cannot be re-thresholded
                                                     per MIMAG level - stated as a
                                                     caveat, not hidden)
       MAGICC V5 / CheckM2 / CoCoPyE / DeepCheck
                  predicted contamination >= threshold (threshold-matched)
   Reported: TP/FP/TN/FN, sensitivity, specificity, precision, F1, balanced
   accuracy, MCC; F1 and balanced accuracy with cluster-bootstrap CIs.

3. **Ranking ability, which is threshold-free and therefore the fairest single
   number.** Spearman rho against true contamination, and ROC AUC for
   true contamination >= 5 % and >= 10 %, using each tool's continuous score:
   GUNC CSS (and GUNC's ``contamination_portion``) versus the four estimators'
   predicted contamination.

CIs: cluster bootstrap over the 100 held-out reference genomes
(``dominant_accession``), 2,000 resamples, seed 7600 - the 10 simulations per
reference are not independent.

Outputs
-------
    results/revision/benchmark/gunc_detection_stratified.tsv
    results/revision/benchmark/gunc_detection_agreement.tsv
    results/revision/benchmark/gunc_detection_ranking.tsv
    results/revision/benchmark/gunc_detection_comparison.md
    results/revision/benchmark/gunc_detection_comparison.json

Usage
-----
    conda run -n magicc2 python scripts/94b_gunc_detection_comparison.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M92 = _load('m92', PROJECT_DIR / 'scripts' / '92_clean_cd_metrics.py')
OUT_DIR = M92.OUT_DIR
TOOLS = M92.TOOLS
CLUSTER_COL = M92.CLUSTER_COL
N_BOOT = M92.N_BOOT
BOOT_SEED = M92.BOOT_SEED
CLEAN_SETS = M92.CLEAN_SETS

BANDS = [(0, 0, '0 (pure)'), (0, 1, '(0, 1]'), (1, 5, '(1, 5]'), (5, 10, '(5, 10]'),
         (10, 20, '(10, 20]'), (20, 50, '(20, 50]'), (50, 100.01, '(50, 100]')]
THRESHOLDS = [5.0, 10.0]


# --------------------------------------------------------------------------
# statistics without sklearn/scipy dependencies
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
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC via the Mann-Whitney U identity, ties handled by mid-ranks."""
    label = np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    r = rankdata(np.asarray(score, float))
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
          if np.isfinite(prec) and np.isfinite(sens) and (prec + sens) > 0 else float('nan'))
    bal = (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else float('nan')
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den > 0 else float('nan')
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'n': tp + fp + tn + fn,
            'sensitivity': sens, 'specificity': spec, 'precision': prec,
            'f1': f1, 'balanced_accuracy': bal, 'mcc': mcc,
            'accuracy': (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else float('nan')}


def cluster_ci(fn, clusters: np.ndarray, n_boot: int = N_BOOT,
               seed: int = BOOT_SEED) -> Tuple[float, float]:
    """95 % percentile CI of fn(index_selection), resampling clusters."""
    uniq = np.unique(clusters)
    groups = [np.where(clusters == c)[0] for c in uniq]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        sel = np.concatenate([groups[i] for i in pick])
        vals[b] = fn(sel)
    v = vals[np.isfinite(vals)]
    return ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            if v.size else (float('nan'), float('nan')))


def f2(x) -> str:
    return '-' if x is None or not np.isfinite(x) else f'{x:.3f}'


# --------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strat_rows: List[Dict[str, object]] = []
    agree_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []
    notes: List[str] = []

    for set_name in CLEAN_SETS:
        gpath = (PROJECT_DIR / 'data' / 'benchmarks' / set_name / 'gunc_predictions.tsv')
        if not gpath.is_file():
            notes.append(f'{set_name}: {gpath} absent -- GUNC not run for this set')
            continue
        g = pd.read_csv(gpath, sep='\t')
        g['genome_id'] = g['genome_id'].astype(str)

        # assemble one frame: truth + GUNC + every estimator's predicted contamination
        base = g[['genome_id', 'true_completeness', 'true_contamination',
                  CLUSTER_COL, 'gunc_css', 'gunc_pass']].copy()
        if 'contamination_portion' in g.columns:
            base['gunc_contamination_portion'] = pd.to_numeric(
                g['contamination_portion'], errors='coerce')
        base['gunc_css'] = pd.to_numeric(base['gunc_css'], errors='coerce')
        base['gunc_fail'] = base['gunc_pass'].astype(str).str.lower().map(
            {'false': True, 'true': False})

        for tool, fn in TOOLS.items():
            d = M92.load_pred(set_name, fn)
            if d is None:
                notes.append(f'{set_name}/{tool}: prediction file missing')
                continue
            base = base.merge(
                d[['genome_id', 'pred_contamination']].rename(
                    columns={'pred_contamination': f'pred_{tool}'}),
                on='genome_id', how='left')
            if base[f'pred_{tool}'].isna().any():
                notes.append(f'{set_name}/{tool}: {int(base[f"pred_{tool}"].isna().sum())} '
                             'genome(s) without a prediction after the merge')

        n_unscored = int(base['gunc_fail'].isna().sum())
        scored = base.dropna(subset=['gunc_fail', 'gunc_css']).copy()
        notes.append(f'{set_name}: GUNC scored {len(scored)}/{len(base)} genomes '
                     f'({n_unscored} below the min_mapped_genes floor)')

        # ---------------- 1. stratified pass/fail ----------------
        for lo, hi, label in BANDS:
            if lo == hi == 0:
                m = base.true_contamination == 0
            else:
                m = (base.true_contamination > lo) & (base.true_contamination <= hi)
            sub = base[m]
            subs = sub.dropna(subset=['gunc_fail'])
            strat_rows.append({
                'set': set_name, 'true_contamination_band': label,
                'n': int(len(sub)), 'n_gunc_scored': int(len(subs)),
                'n_gunc_unscored': int(len(sub) - len(subs)),
                'gunc_fail_rate': (round(float(subs.gunc_fail.mean()), 4)
                                   if len(subs) else ''),
                'gunc_n_fail': int(subs.gunc_fail.sum()) if len(subs) else 0,
                'gunc_css_mean': (round(float(subs.gunc_css.mean()), 4)
                                  if len(subs) else ''),
                'gunc_css_median': (round(float(subs.gunc_css.median()), 4)
                                    if len(subs) else ''),
                'mean_true_contamination': round(float(sub.true_contamination.mean()), 3)
                if len(sub) else '',
            })

        # ---------------- 2. detection agreement ----------------
        clusters = scored[CLUSTER_COL].values
        for thr in THRESHOLDS:
            truth = (scored.true_contamination >= thr).values
            detectors: List[Tuple[str, np.ndarray, str]] = [
                ('GUNC 1.1.1 (pass.GUNC == False)', scored.gunc_fail.values.astype(bool),
                 'single fixed operating point; not re-thresholdable per MIMAG level')]
            for tool in TOOLS:
                col = f'pred_{tool}'
                if col in scored.columns:
                    detectors.append((f'{tool} (pred contamination >= {thr:g} %)',
                                      (scored[col] >= thr).values, 'threshold-matched'))
            for name, pred, caveat in detectors:
                c = confusion(pred, truth)
                f1_ci = cluster_ci(lambda sel, p=pred, t=truth: confusion(p[sel], t[sel])['f1'],
                                   clusters)
                bal_ci = cluster_ci(
                    lambda sel, p=pred, t=truth:
                    confusion(p[sel], t[sel])['balanced_accuracy'], clusters)
                agree_rows.append({
                    'set': set_name, 'threshold_pct': thr, 'detector': name,
                    'operating_point': caveat,
                    'n': c['n'], 'n_true_positive_genomes': int(truth.sum()),
                    'tp': c['tp'], 'fp': c['fp'], 'tn': c['tn'], 'fn': c['fn'],
                    'sensitivity': round(c['sensitivity'], 4),
                    'specificity': round(c['specificity'], 4),
                    'precision': round(c['precision'], 4),
                    'f1': round(c['f1'], 4),
                    'f1_ci_lo': round(f1_ci[0], 4), 'f1_ci_hi': round(f1_ci[1], 4),
                    'balanced_accuracy': round(c['balanced_accuracy'], 4),
                    'balanced_accuracy_ci_lo': round(bal_ci[0], 4),
                    'balanced_accuracy_ci_hi': round(bal_ci[1], 4),
                    'mcc': round(c['mcc'], 4), 'accuracy': round(c['accuracy'], 4),
                })

        # ---------------- 3. ranking ability ----------------
        score_cols: List[Tuple[str, str]] = [('GUNC 1.1.1 CSS', 'gunc_css')]
        if 'gunc_contamination_portion' in scored.columns:
            score_cols.append(('GUNC 1.1.1 contamination_portion',
                               'gunc_contamination_portion'))
        score_cols += [(f'{t} predicted contamination', f'pred_{t}')
                       for t in TOOLS if f'pred_{t}' in scored.columns]
        truth_cont = scored.true_contamination.values.astype(float)
        for name, col in score_cols:
            s = pd.to_numeric(scored[col], errors='coerce').values.astype(float)
            ok = np.isfinite(s)
            if ok.sum() < 10:
                continue
            sv, tv, cv = s[ok], truth_cont[ok], clusters[ok]
            rho = spearman(sv, tv)
            rho_ci = cluster_ci(lambda sel, a=sv, b=tv: spearman(a[sel], b[sel]), cv)
            row: Dict[str, object] = {
                'set': set_name, 'score': name, 'n': int(ok.sum()),
                'spearman_rho_vs_true_contamination': round(rho, 4),
                'spearman_ci_lo': round(rho_ci[0], 4),
                'spearman_ci_hi': round(rho_ci[1], 4),
            }
            for thr in THRESHOLDS:
                lab = tv >= thr
                auc = roc_auc(sv, lab)
                auc_ci = cluster_ci(
                    lambda sel, a=sv, l=lab: roc_auc(a[sel], l[sel]), cv)
                row[f'auroc_gt{int(thr)}pct'] = round(auc, 4)
                row[f'auroc_gt{int(thr)}pct_ci_lo'] = round(auc_ci[0], 4)
                row[f'auroc_gt{int(thr)}pct_ci_hi'] = round(auc_ci[1], 4)
            rank_rows.append(row)

    if not strat_rows:
        print('GUNC predictions absent for every clean set; nothing to do.\n  ' +
              '\n  '.join(notes), file=sys.stderr)
        (OUT_DIR / 'gunc_detection_comparison.json').write_text(json.dumps(
            {'generated_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'GUNC predictions absent', 'notes': notes}, indent=2))
        return 0

    pd.DataFrame(strat_rows).to_csv(OUT_DIR / 'gunc_detection_stratified.tsv',
                                    sep='\t', index=False)
    pd.DataFrame(agree_rows).to_csv(OUT_DIR / 'gunc_detection_agreement.tsv',
                                    sep='\t', index=False)
    pd.DataFrame(rank_rows).to_csv(OUT_DIR / 'gunc_detection_ranking.tsv',
                                   sep='\t', index=False)

    md: List[str] = ['# GUNC as a contamination detector on the clean Sets C/D (WS1.5 / WS4.2)\n']
    md.append(f'Generated {datetime.now(timezone.utc).isoformat()} by '
              '`scripts/94b_gunc_detection_comparison.py`.\n')
    md.append('GUNC returns a clade separation score and a pass/fail flag, **not** a '
              'contamination percentage, so it is deliberately kept out of the MAE table '
              'and evaluated as a detector. The four quantitative tools are scored on the '
              'same detection task at the same thresholds, which makes the comparison '
              'like-for-like. Note the asymmetry that favours the estimators: they are '
              'thresholded at exactly the MIMAG level being tested, whereas GUNC has a '
              'single built-in operating point (CSS and surplus-clade cutoffs at the '
              'maxCSS taxonomic level) that cannot be re-tuned per level. The '
              'threshold-free AUROC/Spearman section removes that asymmetry.\n')

    md.append('\n## 1. GUNC pass/fail stratified by true contamination\n')
    md.append('| Set | true contamination band (%) | n | GUNC scored | GUNC fail | '
              'fail rate | mean CSS | median CSS |')
    md.append('|---|---|---|---|---|---|---|---|')
    for r in strat_rows:
        md.append(f'| {r["set"]} | {r["true_contamination_band"]} | {r["n"]} | '
                  f'{r["n_gunc_scored"]} | {r["gunc_n_fail"]} | '
                  f'{r["gunc_fail_rate"]} | {r["gunc_css_mean"]} | {r["gunc_css_median"]} |')

    md.append('\n## 2. Detection agreement with ground truth at the MIMAG-inspired '
              'contamination thresholds\n')
    md.append('Ground-truth positive (contaminated) = true contamination **≥** threshold, '
              'because the MIMAG-inspired pass criterion is "< 5 %" for high quality and '
              '"< 10 %" for medium quality (Bowers et al. 2017). '
              '"MIMAG-inspired" because rRNA/tRNA criteria cannot be evaluated here.\n')
    md.append('| Set | threshold | detector | TP | FP | TN | FN | sensitivity | '
              'specificity | precision | F1 (95% CI) | balanced acc. (95% CI) | MCC |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in agree_rows:
        md.append(f'| {r["set"]} | >{r["threshold_pct"]:g} % | {r["detector"]} | '
                  f'{r["tp"]} | {r["fp"]} | {r["tn"]} | {r["fn"]} | '
                  f'{f2(r["sensitivity"])} | {f2(r["specificity"])} | '
                  f'{f2(r["precision"])} | '
                  f'{f2(r["f1"])} ({f2(r["f1_ci_lo"])}–{f2(r["f1_ci_hi"])}) | '
                  f'{f2(r["balanced_accuracy"])} '
                  f'({f2(r["balanced_accuracy_ci_lo"])}–{f2(r["balanced_accuracy_ci_hi"])}) | '
                  f'{f2(r["mcc"])} |')

    md.append('\n## 3. Threshold-free ranking ability against true contamination\n')
    md.append('| Set | score | n | Spearman ρ (95% CI) | AUROC ≥5 % (95% CI) | '
              'AUROC ≥10 % (95% CI) |')
    md.append('|---|---|---|---|---|---|')
    for r in rank_rows:
        md.append(f'| {r["set"]} | {r["score"]} | {r["n"]} | '
                  f'{f2(r["spearman_rho_vs_true_contamination"])} '
                  f'({f2(r["spearman_ci_lo"])}–{f2(r["spearman_ci_hi"])}) | '
                  f'{f2(r["auroc_gt5pct"])} '
                  f'({f2(r["auroc_gt5pct_ci_lo"])}–{f2(r["auroc_gt5pct_ci_hi"])}) | '
                  f'{f2(r["auroc_gt10pct"])} '
                  f'({f2(r["auroc_gt10pct_ci_lo"])}–{f2(r["auroc_gt10pct_ci_hi"])}) |')

    md.append(f'\nCIs: cluster bootstrap over the 100 held-out reference genomes '
              f'(`{CLUSTER_COL}`), {N_BOOT} resamples, seed {BOOT_SEED}.\n')
    if notes:
        md.append('\n### Notes\n')
        for n in notes:
            md.append(f'* {n}')
        md.append('')
    (OUT_DIR / 'gunc_detection_comparison.md').write_text('\n'.join(md))
    (OUT_DIR / 'gunc_detection_comparison.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'bootstrap': {'n_boot': N_BOOT, 'seed': BOOT_SEED, 'cluster_unit': CLUSTER_COL},
        'thresholds_pct': THRESHOLDS, 'notes': notes,
        'stratified': strat_rows, 'agreement': agree_rows, 'ranking': rank_rows,
    }, indent=2, default=str))

    for r in rank_rows:
        print(f'{r["set"]:14s} {r["score"]:42s} rho={r["spearman_rho_vs_true_contamination"]:+.3f}  '
              f'AUROC>5%={r["auroc_gt5pct"]:.3f}  AUROC>10%={r["auroc_gt10pct"]:.3f}')
    print(f'\n-> {OUT_DIR / "gunc_detection_comparison.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
