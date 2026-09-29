#!/usr/bin/env python3
'Set H unmodified-reference scores from cached predictions'

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SET_DIR = ROOT / 'data' / 'benchmarks' / 'set_H_ncbi'
OUT = ROOT / 'results' / 'revision' / 'circularity'
LOGS = ROOT / 'logs' / 'revision'


def _load_framework():
    spec = importlib.util.spec_from_file_location(
        'magicc_metrics_framework', ROOT / 'scripts' / '101_metrics_framework.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['magicc_metrics_framework'] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()


def run_magicc(sel: pd.DataFrame, workers: int) -> pd.DataFrame:
    return pd.read_csv(SET_DIR / 'reference_level_magicc_v5.tsv', sep='\t')


def run_checkm2(threads: int) -> pd.DataFrame:
    q = pd.read_csv(SET_DIR / 'reference_level_checkm2_output/quality_report.tsv', sep='\t')
    q = q.rename(columns={'Name': 'primary_accession',
                          'Completeness': 'checkm2_local_completeness',
                          'Contamination': 'checkm2_local_contamination'})
    return q[['primary_accession', 'checkm2_local_completeness', 'checkm2_local_contamination']]


def run_cocopye(threads: int) -> pd.DataFrame:
    selected = pd.read_csv(SET_DIR / 'reference_level_cocopye_selected.tsv', sep='\t')
    assert selected.genome_id.is_unique and selected.tool_scored.all()
    return pd.DataFrame({'primary_accession': selected.genome_id.astype(str).str.replace(r'\.fasta$', '', regex=True),
                         'cocopye_completeness': selected.pred_completeness,
                         'cocopye_contamination': selected.pred_contamination})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=24)
    ap.add_argument('--tools', default='magicc,checkm2,cocopye')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    tools = set(args.tools.split(','))
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print('=' * 88)
    print('WS1.11 — tool scores on the 400 UNMODIFIED reference genomes')
    print('=' * 88)
    sel = pd.read_csv(SET_DIR / 'reference_selection_final.tsv', sep='\t')
    print(f'  references: {len(sel)}  ({int((sel.arm == "H_fail").sum())} H_fail / '
          f'{int((sel.arm == "H_pass").sum())} H_pass)')
    assert (SET_DIR / 'reference_level_magicc_v5.tsv').is_file(), 'Cached MAGICC scores required'
    assert (SET_DIR / 'reference_level_checkm2_output/quality_report.tsv').is_file(), 'Cached CheckM2 scores required'
    assert (SET_DIR / 'reference_level_cocopye_selected.tsv').is_file(), 'Cached CoCoPyE scores required'
    print('Using existing reference scores; no FASTA links or inference are regenerated')

    df = sel[['primary_accession', 'accession', 'arm', 'pair_id', 'match_level',
              'severity', 'gtdb_phylum', 'gtdb_genus', 'gtdb_species', 'genome_size',
              'contig_count', 'n50_contigs', 'checkm2_completeness',
              'checkm2_contamination']].copy()
    df = df.rename(columns={'checkm2_completeness': 'gtdb_checkm2_completeness',
                            'checkm2_contamination': 'gtdb_checkm2_contamination'})

    if 'magicc' in tools:
        print('\n[MAGICC V5]')
        df = df.merge(run_magicc(sel, args.threads), on='primary_accession', how='left')
    if 'checkm2' in tools:
        print('\n[CheckM2 1.0.1 (local)]')
        c = run_checkm2(args.threads)
        if c is not None:
            df = df.merge(c, on='primary_accession', how='left')
    if 'cocopye' in tools:
        print('\n[CoCoPyE 0.5.0]')
        c = run_cocopye(args.threads)
        if c is not None:
            df = df.merge(c, on='primary_accession', how='left')

    df.to_csv(SET_DIR / 'reference_level_predictions.tsv', sep='\t', index=False)
    print(f'\nwrote {SET_DIR / "reference_level_predictions.tsv"}')

    # ---------------------------------------------------- arm comparison
    est_cols = [c for c in df.columns
                if c.endswith(('_completeness', '_contamination'))]
    rows = []
    piv_key = df.set_index(['pair_id', 'arm'])
    for col in est_cols:
        p = df.pivot_table(index='pair_id', columns='arm', values=col)
        if 'H_fail' not in p or 'H_pass' not in p:
            continue
        p = p.dropna()
        d = (p['H_fail'] - p['H_pass']).to_numpy(float)
        bs = fw.Bootstrapper(n_rows=d.size, n_iter=args.n_boot, ci_level=0.95,
                             seed=fw.stable_hash('reflevel|' + col) % (2 ** 31))
        ci = bs.ci(lambda i: float(np.mean(d[i])))
        pv, _ = bs.p_two_sided(lambda i: float(np.mean(d[i])), null=0.0)
        rows.append({
            'estimate': col, 'n_pairs': int(d.size),
            'mean_H_pass': round(float(p['H_pass'].mean()), 4),
            'mean_H_fail': round(float(p['H_fail'].mean()), 4),
            'median_H_pass': round(float(p['H_pass'].median()), 4),
            'median_H_fail': round(float(p['H_fail'].median()), 4),
            'D_fail_minus_pass': round(ci['estimate'], 4),
            'ci_lo': round(ci['ci_lo'], 4), 'ci_hi': round(ci['ci_hi'], 4),
            'p_two_sided': pv,
            'uses_checkm2': col.startswith(('gtdb_checkm2', 'checkm2')),
        })
    comp = pd.DataFrame(rows)
    if len(comp):
        comp['q_bh'] = fw.bh_correct(comp['p_two_sided'].to_numpy())
        comp['significant_bh_0.05'] = comp['q_bh'] < 0.05
    comp.to_csv(OUT / 'ws1_11_reference_level_scores.tsv', sep='\t', index=False)
    print('\nARM DIFFERENCE ON THE RAW REFERENCE GENOMES '
          '(H_fail − H_pass, 200 matched pairs)')
    print(comp.to_string(index=False))

    # CheckM2 local-vs-GTDB reproducibility
    repro = {}
    if 'checkm2_local_completeness' in df.columns:
        a = df['gtdb_checkm2_completeness'].to_numpy(float)
        b = df['checkm2_local_completeness'].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        repro = {
            'n': int(ok.sum()),
            'completeness_mean_abs_diff': round(float(np.mean(np.abs(a[ok] - b[ok]))), 4),
            'completeness_median_abs_diff': round(
                float(np.median(np.abs(a[ok] - b[ok]))), 4),
            'completeness_pearson_r': round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 4),
            'contamination_mean_abs_diff': round(float(np.mean(np.abs(
                df['gtdb_checkm2_contamination'].to_numpy(float)[ok]
                - df['checkm2_local_contamination'].to_numpy(float)[ok]))), 4),
            'n_refs_local_would_now_pass_filter': int(np.sum(
                (b[ok] >= 98)
                & (df['checkm2_local_contamination'].to_numpy(float)[ok] <= 2)
                & (df['arm'].to_numpy()[ok] == 'H_fail'))),
            'n_H_fail': int((df['arm'] == 'H_fail').sum()),
        }
        print(f'\nCheckM2 local-vs-GTDB reproducibility on the same 400 assemblies: '
              f'{repro}')

    with open(OUT / 'ws1_11_reference_level_summary.json', 'w') as f:
        json.dump({'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'generated_by': 'scripts/192_ws1_11_reference_level_scores.py',
                   'n_references': int(len(df)),
                   'arm_difference': rows,
                   'checkm2_local_vs_gtdb': repro,
                   'note': 'Scores on the unmodified deposited assemblies. No '
                           'simulation, therefore no ground truth: this table shows '
                           'whether independent estimators reproduce the CheckM2 '
                           'deficit that caused the H_fail genomes to be excluded, '
                           'not which estimator is correct.'},
                  f, indent=2, default=str)
    print(f'wrote {OUT / "ws1_11_reference_level_summary.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
