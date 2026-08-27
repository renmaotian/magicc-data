#!/usr/bin/env python3
"""
WS1.5 (competitor part) -- leakage-specificity control.

Logic of the control
--------------------
MAGICC V5 was fitted on the genomes that made up the superseded Sets C and D
(1,000/1,000 of Set C's dominants and 903/1,000 of Set D's are train+val
genomes). CheckM2, CoCoPyE and DeepCheck were never trained on them. If the
accuracy drop from the superseded sets to the clean, strictly-held-out
replacements is caused by training-set memorisation, then MAGICC must degrade
while the three competitors stay flat. If instead the clean sets are simply
*harder* for every method, all four tools degrade together and the drop cannot
be attributed to leakage.

Confounder
----------
The superseded sets were generated before the constraint
``contamination % <= completeness %`` entered ``magicc/contamination.py``
(commit c5a9b95), so set_C holds 141/1,000 and set_D 213/1,000 samples outside
the region any model was trained for; the clean sets hold none. Every
comparison is therefore run twice:

    comparison = "full"           superseded set as-is        (CONFOUNDED)
    comparison = "constraint_ok"  superseded set restricted to
                                  contamination % <= completeness %
                                                                (CONFOUNDER-FREE
                                                                 for this defect)

The ``constraint_ok`` comparison is the one to quote. Residual, unavoidable
differences remain (different reference genomes and a slightly different label
distribution), but the *identified* confounder is removed and the label
distributions of the two generations were verified to match
(``results/revision/ws1_23_generation_validation.json``).

CIs and deltas
--------------
Each set's MAE/bias CI is a cluster bootstrap over ``dominant_accession``
(2,000 resamples, seed 7600; the superseded sets have 1,000 distinct dominants,
so their cluster bootstrap coincides with an ordinary bootstrap). The
clean-minus-leaked delta CI resamples the two sets independently -- they contain
disjoint genomes, so the difference is an independent-sample contrast, not a
paired one.

Outputs
-------
    results/revision/benchmark/leakage_specificity_control.tsv
    results/revision/benchmark/leakage_specificity_control.json
    results/revision/benchmark/leakage_specificity_control.md

Usage
-----
    conda run -n magicc2 python scripts/93_leakage_specificity_control.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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

PAIRS = [
    ('Patescibacteriota (CPR)', 'set_C', 'set_C_clean'),
    ('Archaea', 'set_D', 'set_D_clean'),
]

TRAINED_ON_THESE_GENOMES = {
    'MAGICC V5': True,
    'CheckM2 1.0.1': False,
    'CoCoPyE 0.5.0': False,
    'DeepCheck': False,
}


def constraint_ok(df: pd.DataFrame) -> pd.DataFrame:
    """Samples that satisfy the V5 training constraint contamination% <= completeness%."""
    return df[df['true_contamination'] <= df['true_completeness']].copy()


def _cluster_index(clusters: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    uniq = np.unique(clusters)
    return uniq, [np.where(clusters == c)[0] for c in uniq]


def delta_ci(df_leaked: pd.DataFrame, df_clean: pd.DataFrame, target: str,
             stat: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED
             ) -> Tuple[float, float]:
    """95 % CI for stat(clean) - stat(leaked), resampling the two sets independently."""
    def prep(df):
        t = df[f'true_{target}'].values.astype(float)
        p = df[f'pred_{target}'].values.astype(float)
        uniq, idx = _cluster_index(df[CLUSTER_COL].values)
        return t, p, uniq, idx

    tl, pl, ul, il = prep(df_leaked)
    tc, pc, uc, ic = prep(df_clean)
    rng_l = np.random.default_rng(seed)
    rng_c = np.random.default_rng(seed + 1)
    out = np.empty(n_boot)
    for b in range(n_boot):
        sl = np.concatenate([il[i] for i in rng_l.integers(0, len(ul), size=len(ul))])
        sc = np.concatenate([ic[i] for i in rng_c.integers(0, len(uc), size=len(uc))])
        out[b] = M92._stats(tc[sc], pc[sc])[stat] - M92._stats(tl[sl], pl[sl])[stat]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    notes: List[str] = []

    for lineage, leaked_set, clean_set in PAIRS:
        for tool, fn in TOOLS.items():
            dl_full = M92.load_pred(leaked_set, fn)
            dc = M92.load_pred(clean_set, fn)
            if dl_full is None or dc is None:
                notes.append(f'{lineage}/{tool}: missing '
                             f'{leaked_set if dl_full is None else clean_set}/{fn}')
                continue

            n_viol = int((dl_full['true_contamination'] > dl_full['true_completeness']).sum())
            for comparison, dl in (('full', dl_full),
                                   ('constraint_ok', constraint_ok(dl_full))):
                # the same restriction is applied to the clean set for symmetry; it is
                # a no-op there (0 violations by construction) and is asserted to be so
                dc_use = constraint_ok(dc) if comparison == 'constraint_ok' else dc
                assert len(dc_use) == len(dc), (
                    f'{clean_set} unexpectedly contains '
                    f'{len(dc) - len(dc_use)} out-of-constraint samples')
                r_l = M92.metrics_row(dl, leaked_set, tool, comparison)
                r_c = M92.metrics_row(dc_use, clean_set, tool, comparison)
                row: Dict[str, object] = {
                    'lineage': lineage, 'tool': tool,
                    'trained_on_these_genomes': TRAINED_ON_THESE_GENOMES.get(tool),
                    'comparison': comparison,
                    'confounder_free': comparison == 'constraint_ok',
                    'leaked_set': leaked_set, 'clean_set': clean_set,
                    'n_leaked': r_l['n'], 'n_clean': r_c['n'],
                    'n_out_of_constraint_in_leaked_set': n_viol,
                }
                for prefix in ('comp', 'cont'):
                    for stat in ('mae', 'bias'):
                        row[f'leaked_{prefix}_{stat}'] = r_l[f'{prefix}_{stat}']
                        row[f'leaked_{prefix}_{stat}_ci'] = (
                            f'[{r_l[f"{prefix}_{stat}_ci_lo"]}, {r_l[f"{prefix}_{stat}_ci_hi"]}]')
                        row[f'clean_{prefix}_{stat}'] = r_c[f'{prefix}_{stat}']
                        row[f'clean_{prefix}_{stat}_ci'] = (
                            f'[{r_c[f"{prefix}_{stat}_ci_lo"]}, {r_c[f"{prefix}_{stat}_ci_hi"]}]')
                        d = float(r_c[f'{prefix}_{stat}']) - float(r_l[f'{prefix}_{stat}'])
                        row[f'delta_{prefix}_{stat}'] = round(d, 4)
                        lo, hi = delta_ci(dl, dc_use, 'completeness' if prefix == 'comp'
                                          else 'contamination', stat)
                        row[f'delta_{prefix}_{stat}_ci_lo'] = round(lo, 4)
                        row[f'delta_{prefix}_{stat}_ci_hi'] = round(hi, 4)
                        row[f'delta_{prefix}_{stat}_ci_excludes_zero'] = bool(lo > 0 or hi < 0)
                    base = float(r_l[f'{prefix}_mae'])
                    row[f'delta_{prefix}_mae_pct'] = (
                        round(100.0 * (float(r_c[f'{prefix}_mae']) - base) / base, 1)
                        if base > 0 else '')
                    row[f'leaked_{prefix}_r2'] = r_l[f'{prefix}_r2']
                    row[f'clean_{prefix}_r2'] = r_c[f'{prefix}_r2']
                    row[f'leaked_{prefix}_pearson_r2_legacy'] = r_l[f'{prefix}_r2_pearson_sq']
                    row[f'clean_{prefix}_pearson_r2_legacy'] = r_c[f'{prefix}_r2_pearson_sq']
                rows.append(row)
                print(f'{lineage:24s} {tool:16s} {comparison:14s} '
                      f'comp MAE {row["leaked_comp_mae"]:>7} -> {row["clean_comp_mae"]:>7} '
                      f'(D {row["delta_comp_mae"]:+7.3f}) | '
                      f'cont MAE {row["leaked_cont_mae"]:>7} -> {row["clean_cont_mae"]:>7} '
                      f'(D {row["delta_cont_mae"]:+7.3f})')

    tbl = pd.DataFrame(rows)
    tsv = OUT_DIR / 'leakage_specificity_control.tsv'
    tbl.to_csv(tsv, sep='\t', index=False)

    # ------------------------------------------------------------------
    # specificity summary: MAGICC's degradation relative to the competitors'
    # ------------------------------------------------------------------
    summary: List[Dict[str, object]] = []
    for lineage, leaked_set, clean_set in PAIRS:
        for comparison in ('full', 'constraint_ok'):
            sub = tbl[(tbl.lineage == lineage) & (tbl.comparison == comparison)]
            if sub.empty:
                continue
            magicc = sub[sub.tool == 'MAGICC V5']
            comps = sub[sub.tool != 'MAGICC V5']
            if magicc.empty or comps.empty:
                continue
            for prefix, tname in (('comp', 'completeness'), ('cont', 'contamination')):
                m_d = float(magicc.iloc[0][f'delta_{prefix}_mae'])
                c_d = comps[f'delta_{prefix}_mae'].astype(float)
                summary.append({
                    'lineage': lineage, 'comparison': comparison, 'target': tname,
                    'magicc_delta_mae': round(m_d, 4),
                    'competitor_delta_mae_mean': round(float(c_d.mean()), 4),
                    'competitor_delta_mae_min': round(float(c_d.min()), 4),
                    'competitor_delta_mae_max': round(float(c_d.max()), 4),
                    'competitor_delta_mae_abs_max': round(float(c_d.abs().max()), 4),
                    'n_competitors_with_ci_excluding_zero': int(
                        comps[f'delta_{prefix}_mae_ci_excludes_zero'].sum()),
                    'magicc_ci_excludes_zero': bool(
                        magicc.iloc[0][f'delta_{prefix}_mae_ci_excludes_zero']),
                    'magicc_minus_mean_competitor_delta': round(
                        m_d - float(c_d.mean()), 4),
                })

    md: List[str] = []
    md.append('# Leakage-specificity control (WS1.5)\n')
    md.append(f'Generated {datetime.now(timezone.utc).isoformat()} by '
              '`scripts/93_leakage_specificity_control.py`.\n')
    md.append('MAGICC V5 was fitted on the superseded Sets C/D dominants; CheckM2, CoCoPyE '
              'and DeepCheck were not. A drop confined to MAGICC isolates the effect to '
              'training-set memorisation.\n')
    md.append('The superseded sets contain samples that violate the training constraint '
              '`contamination % <= completeness %` (set_C 141/1,000, set_D 213/1,000) while '
              'the clean sets contain none, so each comparison is given twice. **The '
              '`constraint_ok` rows are the confounder-free comparison and the ones to '
              'quote.**\n')
    for comparison, title in (('constraint_ok',
                               'Confounder-free: superseded set restricted to '
                               'contamination % <= completeness %'),
                              ('full', 'Confounded: superseded set as-is')):
        md.append(f'\n## {title}\n')
        md.append('| Lineage | Tool | Trained on these genomes | n leaked | n clean | '
                  'comp MAE leaked | comp MAE clean | Δ comp MAE (95% CI) | '
                  'cont MAE leaked | cont MAE clean | Δ cont MAE (95% CI) |')
        md.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in tbl[tbl.comparison == comparison].iterrows():
            md.append(
                f'| {r.lineage} | {r.tool} | '
                f'{"**yes**" if r.trained_on_these_genomes else "no"} | '
                f'{r.n_leaked} | {r.n_clean} | '
                f'{float(r.leaked_comp_mae):.2f} | {float(r.clean_comp_mae):.2f} | '
                f'{float(r.delta_comp_mae):+.2f} '
                f'({float(r.delta_comp_mae_ci_lo):+.2f}, {float(r.delta_comp_mae_ci_hi):+.2f}) | '
                f'{float(r.leaked_cont_mae):.2f} | {float(r.clean_cont_mae):.2f} | '
                f'{float(r.delta_cont_mae):+.2f} '
                f'({float(r.delta_cont_mae_ci_lo):+.2f}, {float(r.delta_cont_mae_ci_hi):+.2f}) |')
    md.append('\n## Specificity summary (Δ = clean − superseded, percentage points)\n')
    md.append('| Lineage | Comparison | Target | MAGICC Δ MAE | competitor Δ MAE '
              '(mean / min / max) | MAGICC − mean competitor | competitors whose Δ CI '
              'excludes 0 |')
    md.append('|---|---|---|---|---|---|---|')
    for s in summary:
        md.append(f'| {s["lineage"]} | {s["comparison"]} | {s["target"]} | '
                  f'{s["magicc_delta_mae"]:+.2f} | '
                  f'{s["competitor_delta_mae_mean"]:+.2f} / '
                  f'{s["competitor_delta_mae_min"]:+.2f} / '
                  f'{s["competitor_delta_mae_max"]:+.2f} | '
                  f'{s["magicc_minus_mean_competitor_delta"]:+.2f} | '
                  f'{s["n_competitors_with_ci_excluding_zero"]}/3 |')
    md.append('\nR² columns are the coefficient of determination '
              '(1 − SS_res/SS_tot, = `sklearn.metrics.r2_score`); the squared Pearson '
              'correlation is kept separately as `*_pearson_r2_legacy` and is never '
              'called R².\n')
    md.append('\nCIs: cluster bootstrap over `dominant_accession`, '
              f'{N_BOOT} resamples, seed {BOOT_SEED}; the clean−leaked delta resamples the '
              'two (disjoint) sets independently. Descriptive only — the paired two-sided '
              'clustered tests with BH correction and effect sizes are produced by the '
              'statistics framework (`scripts/101`–`105`, `results/revision/metrics/`).\n')
    (OUT_DIR / 'leakage_specificity_control.md').write_text('\n'.join(md))

    (OUT_DIR / 'leakage_specificity_control.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'bootstrap': {'n_boot': N_BOOT, 'seed': BOOT_SEED, 'cluster_unit': CLUSTER_COL},
        'notes': notes,
        'rows': rows,
        'specificity_summary': summary,
    }, indent=2, default=str))

    print('\n--- specificity summary ---')
    for s in summary:
        print(f'{s["lineage"]:24s} {s["comparison"]:14s} {s["target"]:14s} '
              f'MAGICC {s["magicc_delta_mae"]:+7.3f}  '
              f'competitors mean {s["competitor_delta_mae_mean"]:+7.3f} '
              f'(min {s["competitor_delta_mae_min"]:+7.3f}, '
              f'max {s["competitor_delta_mae_max"]:+7.3f})')
    if notes:
        print('\nNOTES:\n  ' + '\n  '.join(notes))
    print(f'\n-> {tsv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
