#!/usr/bin/env python3
"""
WS1.5 (competitor part) -- the revised five-set benchmark table.

Replaces the submitted manuscript's Sets C and D (built from training genomes)
with the clean, strictly held-out replacements, and reports all four tools on
one frozen model version (MAGICC V5):

    set_A_v2     1,000   completeness gradient, finished test-split dominants
    set_B_v2     1,000   contamination gradient, finished test-split dominants
    set_C_clean  1,000   Patescibacteriota, 100 held-out test refs x 10 sims
    set_D_clean  1,000   Archaea,            100 held-out test refs x 10 sims
    set_E        1,000   mixed, finished test-split dominants

A_v2 / B_v2 / E reuse the existing V5 and competitor predictions unchanged
(nothing is re-run for them); C_clean / D_clean come from
``90_run_competitors_clean_cd.sh`` + ``91_parse_competitor_clean_cd.py``.

Two caveats are made explicit in the output rather than left in a footnote:

1. **Set E is partly out of training domain.** 132/1,000 Set E samples have
   contamination % > completeness %, which the V5 training distribution never
   contains (the cap entered ``magicc/contamination.py`` in commit c5a9b95,
   after Set E was generated). Set E is therefore reported twice: as-is, and
   restricted to its 868 in-domain samples. The overall row is likewise given
   in both variants so the manuscript can present one consistent primary
   analysis.
2. **Undefined R².** Set A_v2 has constant true contamination (0 %) and Set B_v2
   constant true completeness (100 %), so R² does not exist for those cells
   (WS5.5); they are printed as '-'.

R² CONVENTION: coefficient of determination, 1 - SS_res/SS_tot, identical to
``sklearn.metrics.r2_score``, and nothing else is ever labelled R². The squared
Pearson correlation used by the project's older scripts is retained only as
``*_r2_pearson_sq`` for traceability.

MIMAG-inspired thresholds used throughout the revision: high quality >= 90 %
completeness and < 5 % contamination, medium quality >= 50 % and < 10 %
(Bowers et al. 2017). "MIMAG-inspired" because the strict definition also
requires rRNA/tRNA criteria that cannot be evaluated from these assemblies.

CIs: cluster bootstrap over ``dominant_accession`` (2,000 resamples, seed 7600).
Per-set clusters: A_v2 798, B_v2 803, C_clean 100, D_clean 100, E 785. The
overall row resamples clusters **within each set** (stratified), which keeps the
1,000-genome-per-set composition of the pooled estimate fixed.

Descriptive statistics only. The paired two-sided clustered significance tests
with Benjamini-Hochberg correction and effect sizes are the statistics
framework's job (``scripts/101``-``105``, ``results/revision/metrics/``).

Outputs
-------
    results/revision/benchmark/revised_benchmark_table.tsv
    results/revision/benchmark/revised_benchmark_table.md
    results/revision/benchmark/revised_benchmark_table.json

Usage
-----
    conda run -n magicc2 python scripts/94_revised_benchmark_table.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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

SETS = [
    ('set_A_v2', 'A_v2', 'completeness gradient 50-100 %, 0 % contamination; '
                         'finished test-split dominants'),
    ('set_B_v2', 'B_v2', '100 % completeness, contamination 0-80 % cross-phylum; '
                         'finished test-split dominants'),
    ('set_C_clean', 'C_clean', 'Patescibacteriota; 100 held-out test-split refs x 10 sims'),
    ('set_D_clean', 'D_clean', 'Archaea; 100 held-out test-split refs x 10 sims'),
    ('set_E', 'E', 'mixed: 200 pure + 200 complete + 600 contaminated; '
                   'finished test-split dominants'),
]


# --------------------------------------------------------------------------
def stratified_cluster_bootstrap(pooled: pd.DataFrame, target: str,
                                 n_boot: int = N_BOOT, seed: int = BOOT_SEED
                                 ) -> Dict[str, tuple]:
    """CIs for a pooled estimate, resampling clusters *within* each set."""
    t = pooled[f'true_{target}'].values.astype(float)
    p = pooled[f'pred_{target}'].values.astype(float)
    strata: List[List[np.ndarray]] = []
    for _, sub in pooled.groupby('set_label', sort=True):
        pos = sub.index.values
        clusters = pooled.loc[pos, CLUSTER_COL].values
        uniq = np.unique(clusters)
        strata.append([pos[np.where(clusters == c)[0]] for c in uniq])
    rng = np.random.default_rng(seed)
    keys = ['mae', 'rmse', 'bias', 'r2_cod']
    acc = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        sel_parts = []
        for groups in strata:
            pick = rng.integers(0, len(groups), size=len(groups))
            sel_parts.append(np.concatenate([groups[i] for i in pick]))
        sel = np.concatenate(sel_parts)
        s = M92._stats(t[sel], p[sel])
        for k in keys:
            acc[k][b] = s[k]
    ci = {}
    for k in keys:
        v = acc[k][np.isfinite(acc[k])]
        ci[k] = ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                 if v.size else (np.nan, np.nan))
    return ci


def pooled_row(pooled: pd.DataFrame, set_label: str, tool: str,
               subset: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        'set': set_label, 'tool': tool, 'subset': subset,
        'n': int(len(pooled)),
        'n_clusters': int(pooled.groupby('set_label')[CLUSTER_COL].nunique().sum()),
        'cluster_unit': f'{CLUSTER_COL} within set (stratified)',
        'n_boot': N_BOOT, 'boot_seed': BOOT_SEED,
    }
    for target, prefix in (('completeness', 'comp'), ('contamination', 'cont')):
        t = pooled[f'true_{target}'].values.astype(float)
        p = pooled[f'pred_{target}'].values.astype(float)
        s = M92._stats(t, p)
        ci = stratified_cluster_bootstrap(pooled, target)
        row[f'{prefix}_true_constant'] = bool(np.std(t) == 0)
        for k in ('mae', 'rmse', 'bias'):
            row[f'{prefix}_{k}'] = round(s[k], 4)
            row[f'{prefix}_{k}_ci_lo'] = round(ci[k][0], 4)
            row[f'{prefix}_{k}_ci_hi'] = round(ci[k][1], 4)
        row[f'{prefix}_r2'] = round(s['r2_cod'], 4)
        row[f'{prefix}_r2_ci_lo'] = round(ci['r2_cod'][0], 4)
        row[f'{prefix}_r2_ci_hi'] = round(ci['r2_cod'][1], 4)
        row[f'{prefix}_pearson_r'] = (round(s['pearson_r'], 4)
                                     if np.isfinite(s['pearson_r']) else '')
        row[f'{prefix}_r2_pearson_sq'] = (round(s['pearson_r2'], 4)
                                      if np.isfinite(s['pearson_r2']) else '')
        row[f'{prefix}_r2_omitted_reason'] = ''
    return row


def cell(row: Optional[Dict[str, object]], prefix: str, stat: str = 'mae') -> str:
    if row is None:
        return 'n/a'
    v = row.get(f'{prefix}_{stat}')
    if v == '' or v is None:
        return '-'
    return (f'{float(v):.2f} ({float(row[f"{prefix}_{stat}_ci_lo"]):.2f}–'
            f'{float(row[f"{prefix}_{stat}_ci_hi"]):.2f})')


def r2cell(row: Optional[Dict[str, object]], prefix: str) -> str:
    if row is None:
        return 'n/a'
    v = row.get(f'{prefix}_r2')
    return '-' if v == '' or v is None else f'{float(v):.3f}'


# --------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    pooled_store: Dict[str, List[pd.DataFrame]] = {t: [] for t in TOOLS}
    set_facts: Dict[str, Dict[str, object]] = {}

    for set_name, label, design in SETS:
        meta = pd.read_csv(PROJECT_DIR / 'data' / 'benchmarks' / set_name / 'metadata.tsv',
                           sep='\t')
        n_viol = int((meta.true_contamination > meta.true_completeness).sum())
        set_facts[label] = {
            'set_dir': set_name, 'n': int(len(meta)), 'design': design,
            'n_unique_dominants': int(meta.dominant_accession.nunique()),
            'n_out_of_training_domain': n_viol,
            'comp_constant': bool(meta.true_completeness.nunique() == 1),
            'cont_constant': bool(meta.true_contamination.nunique() == 1),
        }

        for tool, fn in TOOLS.items():
            df = M92.load_pred(set_name, fn)
            if df is None:
                missing.append(f'{set_name}/{fn}')
                continue
            rows.append(M92.metrics_row(df, label, tool, 'all'))
            df2 = df.copy()
            df2['set_label'] = label
            pooled_store[tool].append(df2)

            if n_viol > 0:                      # in-domain variant for Set E
                sub = df[df.true_contamination <= df.true_completeness].copy()
                rows.append(M92.metrics_row(sub, label, tool, 'in_domain'))

    # ---------------- overall rows ----------------
    for tool in TOOLS:
        if not pooled_store[tool]:
            continue
        pooled = pd.concat(pooled_store[tool], ignore_index=True)
        rows.append(pooled_row(pooled, 'OVERALL', tool, 'all'))
        mask = ~((pooled.set_label == 'E') &
                 (pooled.true_contamination > pooled.true_completeness))
        rows.append(pooled_row(pooled[mask].reset_index(drop=True),
                               'OVERALL', tool, 'E_in_domain'))

    tbl = pd.DataFrame(rows)
    tsv = OUT_DIR / 'revised_benchmark_table.tsv'
    tbl.to_csv(tsv, sep='\t', index=False)

    def get(set_label: str, tool: str, subset: str) -> Optional[Dict[str, object]]:
        sub = tbl[(tbl.set == set_label) & (tbl.tool == tool) & (tbl.subset == subset)]
        return None if sub.empty else sub.iloc[0].to_dict()

    display_rows = [(lbl, 'all') for _, lbl, _ in SETS]
    display_rows += [(lbl, 'in_domain') for _, lbl, _ in SETS
                     if set_facts[lbl]['n_out_of_training_domain'] > 0]
    display_rows += [('OVERALL', 'all'), ('OVERALL', 'E_in_domain')]

    md: List[str] = []
    md.append('# Revised five-set benchmark table (WS1.5)\n')
    md.append(f'Generated {datetime.now(timezone.utc).isoformat()} by '
              '`scripts/94_revised_benchmark_table.py`.\n')
    md.append('All numbers come from one frozen model version, **MAGICC V5** '
              '(`models/magicc_v5.onnx`, SHA256 b843466…). Sets C and D of the submitted '
              'manuscript are **withdrawn** and replaced by `C_clean` / `D_clean`, built '
              'from 100 strictly held-out test-split references × 10 simulations each '
              '(0 train/val overlap, 0 overlap with the 2,000 k-mer feature-selection '
              'genomes; `results/revision/provenance/`).\n')
    md.append('Denominator for both metrics: the dominant genome\'s **full reference '
              'length**. Completeness = retained dominant bp / reference bp × 100; '
              'contamination = total contaminant bp / reference bp × 100.\n')
    md.append('MAE, percentage points, with 95 % CI from a cluster bootstrap over '
              f'`dominant_accession` ({N_BOOT} resamples, seed {BOOT_SEED}); the OVERALL '
              'row resamples clusters within each set (stratified), keeping the '
              '1,000-per-set composition fixed.\n')

    md.append('\n## Completeness MAE (95 % CI)\n')
    hdr = '| Set | subset | n | clusters | ' + ' | '.join(TOOLS) + ' |'
    md.append(hdr)
    md.append('|' + '---|' * (4 + len(TOOLS)))
    for lbl, subset in display_rows:
        any_row = next((get(lbl, t, subset) for t in TOOLS if get(lbl, t, subset)), None)
        if any_row is None:
            continue
        cells = ' | '.join(cell(get(lbl, t, subset), 'comp') for t in TOOLS)
        md.append(f'| {lbl} | {subset} | {any_row["n"]} | {any_row["n_clusters"]} | {cells} |')

    md.append('\n## Contamination MAE (95 % CI)\n')
    md.append(hdr)
    md.append('|' + '---|' * (4 + len(TOOLS)))
    for lbl, subset in display_rows:
        any_row = next((get(lbl, t, subset) for t in TOOLS if get(lbl, t, subset)), None)
        if any_row is None:
            continue
        cells = ' | '.join(cell(get(lbl, t, subset), 'cont') for t in TOOLS)
        md.append(f'| {lbl} | {subset} | {any_row["n"]} | {any_row["n_clusters"]} | {cells} |')

    md.append('\n## Mean signed error (bias = predicted − true, 95 % CI)\n')
    md.append('| Set | subset | Target | ' + ' | '.join(TOOLS) + ' |')
    md.append('|' + '---|' * (3 + len(TOOLS)))
    for lbl, subset in display_rows:
        for prefix, tname in (('comp', 'completeness'), ('cont', 'contamination')):
            any_row = next((get(lbl, t, subset) for t in TOOLS if get(lbl, t, subset)), None)
            if any_row is None:
                continue
            cells = ' | '.join(cell(get(lbl, t, subset), prefix, 'bias') for t in TOOLS)
            md.append(f'| {lbl} | {subset} | {tname} | {cells} |')

    md.append('\n## R² = coefficient of determination (1 − SS_res/SS_tot, '
              '`sklearn.metrics.r2_score`)\n')
    md.append('Negative values mean the predictor is worse than always predicting the '
              'mean of the true values; they are correct and reported plainly. The '
              'squared Pearson correlation - which ignores bias and scale error, is '
              'always ≥ this quantity, and is what the submitted Table S2c mixed in for '
              'some tools - is kept in the TSV as `*_r2_pearson_sq` and is never called R². '
              "'-' marks cells where the true value has zero variance (Set A_v2 "
              'contamination is 0 % throughout, Set B_v2 completeness is 100 % '
              'throughout), so R² is undefined (WS5.5).\n')
    md.append('| Set | subset | Target | ' + ' | '.join(TOOLS) + ' |')
    md.append('|' + '---|' * (3 + len(TOOLS)))
    for lbl, subset in display_rows:
        for prefix, tname in (('comp', 'completeness'), ('cont', 'contamination')):
            any_row = next((get(lbl, t, subset) for t in TOOLS if get(lbl, t, subset)), None)
            if any_row is None:
                continue
            cells = ' | '.join(r2cell(get(lbl, t, subset), prefix) for t in TOOLS)
            md.append(f'| {lbl} | {subset} | {tname} | {cells} |')

    md.append('\n## Caveats that must travel with this table\n')
    md.append('| Set | n | unique dominants | samples with contamination % > completeness % '
              '(outside the V5 training domain) | R² undefined |')
    md.append('|---|---|---|---|---|')
    for _, lbl, _ in SETS:
        f = set_facts[lbl]
        undef = []
        if f['cont_constant']:
            undef.append('contamination (true value constant)')
        if f['comp_constant']:
            undef.append('completeness (true value constant)')
        md.append(f'| {lbl} | {f["n"]} | {f["n_unique_dominants"]} | '
                  f'{f["n_out_of_training_domain"]} | {", ".join(undef) or "-"} |')
    md.append('\n* **Set E carries 132/1,000 out-of-domain samples.** The `in_domain` row '
              'restricts Set E to its 868 constraint-satisfying samples, and '
              '`OVERALL / E_in_domain` restricts only Set E inside the pooled estimate '
              '(4,868 genomes). Use one variant consistently throughout the manuscript.\n')
    md.append('* Sets A_v2, B_v2, C_clean and D_clean contain **no** out-of-domain '
              'samples, so their rows are unaffected.\n')
    md.append('* Descriptive statistics only. Paired two-sided clustered tests with '
              'Benjamini–Hochberg correction and effect sizes come from the statistics '
              'framework (`scripts/101`–`105`, `results/revision/metrics/`).\n')
    if missing:
        md.append('* **Missing prediction files:** ' + ', '.join(f'`{m}`' for m in missing) + '\n')
    (OUT_DIR / 'revised_benchmark_table.md').write_text('\n'.join(md))

    (OUT_DIR / 'revised_benchmark_table.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'model': 'MAGICC V5 (models/magicc_v5.onnx)',
        'bootstrap': {'n_boot': N_BOOT, 'seed': BOOT_SEED,
                      'per_set_cluster_unit': CLUSTER_COL,
                      'overall': 'clusters resampled within each set (stratified)'},
        'set_facts': set_facts,
        'missing_prediction_files': missing,
        'rows': rows,
    }, indent=2, default=str))

    print('\nCompleteness MAE (95 % CI)')
    for lbl, subset in display_rows:
        any_row = next((get(lbl, t, subset) for t in TOOLS if get(lbl, t, subset)), None)
        if any_row is None:
            continue
        print(f'  {lbl:9s} {subset:12s} n={any_row["n"]:>5} ' +
              '  '.join(f'{t}: {cell(get(lbl, t, subset), "comp")}' for t in TOOLS))
    print('\nContamination MAE (95 % CI)')
    for lbl, subset in display_rows:
        any_row = next((get(lbl, t, subset) for t in TOOLS if get(lbl, t, subset)), None)
        if any_row is None:
            continue
        print(f'  {lbl:9s} {subset:12s} n={any_row["n"]:>5} ' +
              '  '.join(f'{t}: {cell(get(lbl, t, subset), "cont")}' for t in TOOLS))
    if missing:
        print('\nMISSING: ' + ', '.join(missing), file=sys.stderr)
    print(f'\n-> {tsv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
