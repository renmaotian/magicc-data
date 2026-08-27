#!/usr/bin/env python3
"""
Correct the R² convention in ``results/revision/legacy_v5_metrics.tsv``.

``scripts/071_audit_legacy_predictions.py`` wrote ``np.corrcoef(true, pred)[0,1]**2``
into columns named ``comp_r2`` / ``cont_r2``. That is the squared Pearson
correlation, not R². Squared Pearson ignores bias and scale error and is always
>= the coefficient of determination, so the two diverge exactly where a predictor
is biased - and it is the coefficient of determination that a
prediction-accuracy table means by R² (``sklearn.metrics.r2_score``). Left
uncorrected, the wrong statistic would propagate into the manuscript.

This script recomputes every legacy V5 metric straight from the per-set
``magicc_v5_predictions.tsv`` files and rewrites the table with unambiguous
column names:

    comp_r2 / cont_r2            coefficient of determination, 1 - SS_res/SS_tot
    comp_pearson_r / cont_...    Pearson correlation coefficient r
    comp_r2_pearson_sq / cont_...   squared Pearson r - legacy traceability only,
                                 i.e. the number script 71 mislabelled as R²
    *_r2_omitted_reason          why R² is blank (true value has zero variance)

MAE / RMSE / bias are recomputed too and asserted to reproduce the existing file,
which proves the rewrite is reading the same predictions and is not silently
substituting different numbers. The original file is copied to
``results/revision/benchmark/legacy_v5_metrics_ORIGINAL_pearson_r2.tsv`` first.

Usage
-----
    conda run -n magicc2 python scripts/092b_fix_legacy_r2_convention.py
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
LEGACY = PROJECT_DIR / 'results' / 'revision' / 'legacy_v5_metrics.tsv'
BENCH = PROJECT_DIR / 'data' / 'benchmarks'
BACKUP_DIR = PROJECT_DIR / 'results' / 'revision' / 'benchmark'


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M92 = _load('m92', PROJECT_DIR / 'scripts' / '092_clean_cd_metrics.py')


def main() -> int:
    if not LEGACY.is_file():
        print(f'{LEGACY} not present; nothing to fix.')
        return 0
    old = pd.read_csv(LEGACY, sep='\t')
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / 'legacy_v5_metrics_ORIGINAL_pearson_r2.tsv'
    if not backup.exists():
        shutil.copy2(LEGACY, backup)
        print(f'original preserved -> {backup}')

    rows, problems = [], []
    for _, r in old.iterrows():
        set_name = str(r['set'])
        pred_path = BENCH / set_name / 'magicc_v5_predictions.tsv'
        if not pred_path.is_file():
            problems.append(f'{set_name}: {pred_path} missing, row copied unchanged')
            rows.append(r.to_dict())
            continue
        d = pd.read_csv(pred_path, sep='\t')
        row = {'set': set_name, 'n': int(len(d))}
        for target, prefix in (('completeness', 'comp'), ('contamination', 'cont')):
            t = d[f'true_{target}'].values.astype(float)
            p = d[f'pred_{target}'].values.astype(float)
            s = M92._stats(t, p)
            row[f'{prefix}_mae'] = round(s['mae'], 4)
            row[f'{prefix}_rmse'] = round(s['rmse'], 4)
            row[f'{prefix}_bias'] = round(s['bias'], 4)
            if np.std(t) == 0:
                row[f'{prefix}_r2'] = ''
                row[f'{prefix}_pearson_r'] = ''
                row[f'{prefix}_r2_pearson_sq'] = ''
                row[f'{prefix}_r2_omitted_reason'] = (
                    f'true {target} has zero variance (constant {t[0]:.6g} %), '
                    'SS_tot = 0, R2 undefined')
                print(f'  R2 omitted for {set_name} {target}: true value constant '
                      f'({t[0]:.6g} %), SS_tot = 0.')
            else:
                row[f'{prefix}_r2'] = round(s['r2_cod'], 4)
                row[f'{prefix}_pearson_r'] = round(s['pearson_r'], 4)
                row[f'{prefix}_r2_pearson_sq'] = round(s['pearson_r2'], 4)
                row[f'{prefix}_r2_omitted_reason'] = ''
            # prove we are reading the same predictions script 71 read
            for k in ('mae', 'rmse', 'bias'):
                oldv = r.get(f'{prefix}_{k}')
                if pd.notna(oldv) and oldv != '':
                    if abs(float(oldv) - row[f'{prefix}_{k}']) > 5e-4:
                        problems.append(
                            f'{set_name}/{prefix}_{k}: recomputed '
                            f'{row[f"{prefix}_{k}"]} != stored {oldv}')
            oldr2 = r.get(f'{prefix}_r2')
            if pd.notna(oldr2) and oldr2 != '' and row[f'{prefix}_r2_pearson_sq'] != '':
                if abs(float(oldr2) - row[f'{prefix}_r2_pearson_sq']) > 5e-4:
                    problems.append(
                        f'{set_name}/{prefix}_r2: stored {oldr2} is not the squared '
                        f'Pearson r either ({row[f"{prefix}_r2_pearson_sq"]})')
        rows.append(row)

    new = pd.DataFrame(rows)
    cols = ['set', 'n']
    for prefix in ('comp', 'cont'):
        cols += [f'{prefix}_mae', f'{prefix}_rmse', f'{prefix}_bias', f'{prefix}_r2',
                 f'{prefix}_pearson_r', f'{prefix}_r2_pearson_sq',
                 f'{prefix}_r2_omitted_reason']
    new = new[[c for c in cols if c in new.columns]]
    new.to_csv(LEGACY, sep='\t', index=False)

    note = LEGACY.with_name('legacy_v5_metrics_R2_CONVENTION.txt')
    note.write_text(
        f'Rewritten {datetime.now(timezone.utc).isoformat()} by '
        'scripts/092b_fix_legacy_r2_convention.py.\n\n'
        'comp_r2 / cont_r2      = coefficient of determination, 1 - SS_res/SS_tot,\n'
        '                         identical to sklearn.metrics.r2_score. THIS is R2.\n'
        '                         It may be negative (worse than predicting the mean).\n'
        'comp_pearson_r         = Pearson correlation coefficient r (a different\n'
        '                         statistic; never call it R2).\n'
        'comp_r2_pearson_sq      = r**2. Legacy traceability only: this is the number\n'
        '                         scripts 29/30/56/71/75 printed under the name "R2".\n'
        '*_r2_omitted_reason    = why R2 is blank (true value has zero variance:\n'
        '                         set_*_v2 A contamination is 0 % throughout,\n'
        '                         B completeness is 100 % throughout).\n\n'
        'The pre-correction file is preserved at\n'
        f'{backup}\n')

    print('\ncorrected legacy V5 metrics:')
    show = ['set', 'n', 'comp_mae', 'comp_r2', 'comp_r2_pearson_sq',
            'cont_mae', 'cont_r2', 'cont_r2_pearson_sq']
    print(new[[c for c in show if c in new.columns]].to_string(index=False))
    if problems:
        print('\nPROBLEMS:\n  ' + '\n  '.join(problems), file=sys.stderr)
        return 1
    print(f'\nverified: MAE/RMSE/bias reproduce the stored values, and every stored '
          f'"r2" equals the squared Pearson r\n-> {LEGACY}\n-> {note}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
