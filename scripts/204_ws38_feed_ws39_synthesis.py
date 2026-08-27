#!/usr/bin/env python3
"""
WS3.9 feed — add the CAMI II cohorts to the cross-dataset real-data synthesis view.

`scripts/144_realdata_synthesis.py` built `results/revision/real_data/synthesis_table.tsv`
for the three cohorts with genuine ground truth (Meslier MOCK1, Zymo isolates, NCBI
Tier-1 pairs). WS3.8 adds a fourth, procedurally independent one: CAMI II. This script
emits CAMI II rows in exactly that schema and writes the merged view, so WS3.9 has all
real-data evidence in one place.

Two conventions differ from the original table and are enforced here, both required by
the revision:
  * R2 is the coefficient of determination and is set to NaN, with the reason recorded,
    wherever the truth has (near-)zero variance. The original table carries values such
    as -669.73 and -154.06 for contamination on near-pure mocks; those are artefacts of
    SS_tot ~ 0, not findings (R1-m19), and are blanked in the merged view with the
    reason stated in `r2_omitted_reason`.
  * MIMAG thresholds are labelled MIMAG-inspired.

Outputs:
  results/revision/real_data/synthesis_table_with_cami2.tsv
  results/revision/real_data/synthesis_table_with_cami2.md
  results/revision/cami2/analysis/cami2_ws39_rows.tsv

Usage: PYTHONHASHSEED=0 python scripts/204_ws38_feed_ws39_synthesis.py
"""

import signal
import sys
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

RD = PROJECT_DIR / 'results' / 'revision' / 'real_data'
CAMI = PROJECT_DIR / 'results' / 'revision' / 'cami2'
AN = CAMI / 'analysis'

TOOL_LABEL = {'MAGICC_V5': 'MAGICC V5', 'CheckM2': 'CheckM2 1.0.1',
              'CoCoPyE': 'CoCoPyE 0.5.0', 'DeepCheck': 'DeepCheck'}

COHORTS = [
    ('marine', 'gold', 'CAMI II marine, gold-standard pure bins (leakage-free, >=50 %)'),
    ('marine', 'mixed', 'CAMI II marine, constructed mixed bins (leakage-free, in-domain)'),
    ('strain_madness', 'gold',
     'CAMI II strain-madness, gold-standard pure bins (leakage-free, >=50 %)'),
    ('strain_madness', 'mixed',
     'CAMI II strain-madness, constructed mixed bins (leakage-free, in-domain)'),
]


def main():
    long = pd.read_csv(AN / 'cami2_long_predictions.tsv', sep='\t')
    acc = pd.read_csv(AN / 'cami2_accuracy_by_cohort.tsv', sep='\t')
    mim = pd.read_csv(AN / 'cami2_mimag.tsv', sep='\t')

    rows = []
    for ds, bset, label in COHORTS:
        sub = long[(long['dataset'] == ds) & (long['binset'] == bset)
                   & long['in_domain'] & long['scoreable'] & long['leakage_free']]
        if sub.empty:
            continue
        for tool, tlabel in TOOL_LABEL.items():
            ts = sub[sub['tool'] == tool]
            if len(ts) < 8:
                continue
            a = acc[(acc['dataset'] == ds) & (acc['binset'] == bset)
                    & (acc['cohort'] == 'primary_in_domain_scoreable_LEAKAGE_FREE')
                    & (acc['tool'] == tool)]
            ac = a[a['metric'] == 'completeness']
            ak = a[a['metric'] == 'contamination']
            if ac.empty or ak.empty:
                continue
            ac, ak = ac.iloc[0], ak.iloc[0]
            m = mim[(mim['dataset'] == ds) & (mim['binset'] == bset)
                    & (mim['tool'] == tool)]
            m = m.iloc[0] if len(m) else None

            true_cont_sd = float(ts['true_contamination'].std())
            reason = ''
            cont_r2 = ak['r2']
            if not np.isfinite(cont_r2) or true_cont_sd < 1e-8:
                cont_r2 = np.nan
                reason = ('contamination R2 omitted: true contamination has '
                          '(near-)zero variance in this cohort, so SS_tot ~ 0 (R1-m19)')
            rows.append({
                'dataset': label, 'track': f'cami2_{ds}', 'cohort':
                    f'{bset}_primary_in_domain_scoreable_LEAKAGE_FREE',
                'tool': tlabel, 'n': int(ac['n']), 'n_clusters': int(ac['n_clusters']),
                'cluster_unit': 'CAMI II source genome (dominant)',
                'true_comp_mean': round(float(ts['true_completeness'].mean()), 4),
                'true_cont_mean': round(float(ts['true_contamination'].mean()), 4),
                'comp_MAE': round(float(ac['mae']), 4),
                'comp_MAE_95CI': f"[{ac['mae_lo']:.4f}, {ac['mae_hi']:.4f}]",
                'comp_bias': round(float(ac['bias']), 4),
                'comp_bias_95CI': f"[{ac['bias_lo']:.4f}, {ac['bias_hi']:.4f}]",
                'comp_R2_CoD': round(float(ac['r2']), 4)
                if np.isfinite(ac['r2']) else np.nan,
                'cont_MAE': round(float(ak['mae']), 4),
                'cont_MAE_95CI': f"[{ak['mae_lo']:.4f}, {ak['mae_hi']:.4f}]",
                'cont_bias': round(float(ak['bias']), 4),
                'cont_bias_95CI': f"[{ak['bias_lo']:.4f}, {ak['bias_hi']:.4f}]",
                'cont_R2_CoD': round(float(cont_r2), 4)
                if np.isfinite(cont_r2) else np.nan,
                'true_HQ': int(m['n_true_HQ']) if m is not None else np.nan,
                'pred_HQ': int(m['n_pred_HQ']) if m is not None else np.nan,
                'MIMAG_class_agreement_pct': round(100.0 * float(m['HQ_agreement']), 2)
                if m is not None else np.nan,
                'false_fail_5pct_rate': round(100.0 * float(m['false_fail_rate_at_5pct']), 2)
                if m is not None and np.isfinite(m['false_fail_rate_at_5pct']) else np.nan,
                'r2_omitted_reason': reason,
                'procedural_independence':
                    'genome selection, read simulation, assembly and gold standard all '
                    'produced by a third party (CAMI II); only the grouping of contigs '
                    'into bins is ours',
            })

    cami_rows = pd.DataFrame(rows)
    cami_rows.to_csv(AN / 'cami2_ws39_rows.tsv', sep='\t', index=False)
    print(f'  CAMI II synthesis rows: {len(cami_rows)}', flush=True)

    old_p = RD / 'synthesis_table.tsv'
    if old_p.exists():
        old = pd.read_csv(old_p, sep='\t')
        old['procedural_independence'] = ''
        # blank the meaningless near-zero-variance contamination R2 values (R1-m19)
        old['r2_omitted_reason'] = ''
        bad = old['true_cont_mean'] < 1.0
        old.loc[bad, 'r2_omitted_reason'] = (
            'contamination R2 omitted: true contamination is near-constant in this '
            'cohort, so SS_tot ~ 0 (R1-m19)')
        old.loc[bad, 'cont_R2_CoD'] = np.nan
        merged = pd.concat([old, cami_rows], ignore_index=True)
    else:
        merged = cami_rows

    merged.to_csv(RD / 'synthesis_table_with_cami2.tsv', sep='\t', index=False)
    cols = ['dataset', 'tool', 'n', 'n_clusters', 'comp_MAE', 'comp_MAE_95CI',
            'comp_bias', 'cont_MAE', 'cont_MAE_95CI', 'cont_bias', 'comp_R2_CoD',
            'cont_R2_CoD']
    md = ['# WS3.9 — cross-dataset real-data synthesis, including CAMI II (WS3.8)\n',
          '**Denominator (identical for truth and every tool, R1-M5):** completeness = '
          'retained dominant bp / FULL reference length of the dominant genome x 100; '
          'contamination = total contaminant bp / the SAME denominator x 100.\n',
          '**R2 = coefficient of determination** (1 - SS_res/SS_tot), omitted where the '
          'true value has (near-)zero variance (R1-m19). MIMAG thresholds are '
          'MIMAG-inspired throughout.\n',
          '**CAMI II is the only cohort here with full procedural independence**: genome '
          'selection, read simulation, assembly and gold standard were all produced by a '
          'third party.\n',
          fw.md_table(merged[cols])]
    (RD / 'synthesis_table_with_cami2.md').write_text('\n'.join(md))
    print(f'  wrote {RD / "synthesis_table_with_cami2.tsv"} ({len(merged)} rows)',
          flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
