#!/usr/bin/env python3
"""
WS1.11 -- score the 400 UNMODIFIED reference genomes themselves.

WHY THIS IS NEEDED
------------------
The simulation defines completeness relative to the *deposited* reference assembly and
treats that assembly as complete and clean. For the H_fail arm -- genomes the project's
CheckM2-based curation would have rejected -- that assumption is exactly what is in
question. If those references really are incomplete or contaminated, then any well
calibrated estimator should read low/high on them and will be scored as "biased" by a
benchmark whose truth says otherwise. If instead the CheckM2 scores that excluded them
are themselves the outlier, the curation removed genomes for an artefact.

Only orthogonal evidence separates the two, so this script runs every available tool on
the 400 deposited assemblies with no simulation at all, and asks whether tools that
never see a CheckM2 score reproduce the deficit that caused exclusion.

Reported per tool, per arm:  completeness / contamination estimate on the raw reference,
and the H_fail - H_pass difference with a bootstrap CI over the 200 matched pairs.
Also a reproducibility check: does local CheckM2 1.0.1 reproduce the CheckM2 score GTDB
recorded (the number the curation filter actually used)?

Outputs
  data/benchmarks/set_H_ncbi/references_fasta/           symlinks with a .fasta suffix
  data/benchmarks/set_H_ncbi/reference_level_predictions.tsv
  results/revision/circularity/ws1_11_reference_level_scores.tsv
  results/revision/circularity/ws1_11_reference_level_summary.json

Usage:
    python scripts/192_ws1_11_reference_level_scores.py [--threads 24] [--tools ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path('/path/to/magicc')
sys.path.insert(0, str(ROOT))
SET_DIR = ROOT / 'data' / 'benchmarks' / 'set_H_ncbi'
REF_DIR = SET_DIR / 'references'
LINK_DIR = SET_DIR / 'references_fasta'
OUT = ROOT / 'results' / 'revision' / 'circularity'
LOGS = ROOT / 'logs' / 'revision'
CHECKM2_DB = ROOT / 'tools' / 'checkm2_db' / 'CheckM2_database' / 'uniref100.KO.1.dmnd'


def _load_framework():
    spec = importlib.util.spec_from_file_location(
        'magicc_metrics_framework', ROOT / 'scripts' / '101_metrics_framework.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['magicc_metrics_framework'] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()


def build_links(sel: pd.DataFrame) -> int:
    LINK_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for acc in sel['primary_accession']:
        src = REF_DIR / f'{acc}.fna'
        dst = LINK_DIR / f'{acc}.fasta'
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(os.path.relpath(src, LINK_DIR), dst)
        n += 1
    return n


def run_magicc(sel: pd.DataFrame, workers: int) -> pd.DataFrame:
    out = SET_DIR / 'reference_level_magicc_v5.tsv'
    if out.exists():
        return pd.read_csv(out, sep='\t')
    from multiprocessing import Pool
    import onnxruntime as ort
    m189 = importlib.util.spec_from_file_location(
        'ws1_11_magicc', ROOT / 'scripts' / '189_ws1_11_run_magicc.py')
    mod = importlib.util.module_from_spec(m189)
    sys.modules['ws1_11_magicc'] = mod
    m189.loader.exec_module(mod)
    from magicc.normalization import FeatureNormalizer

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(ROOT / 'models' / 'magicc_v5.onnx'), so,
                                   providers=['CPUExecutionProvider'])
    normalizer = FeatureNormalizer.load(
        str(ROOT / 'data' / 'features' / 'normalization_params.json'))
    work = [(i, str(REF_DIR / f'{a}.fna'))
            for i, a in enumerate(sel['primary_accession'])]
    got = {}
    with Pool(processes=workers, initializer=mod._init_worker,
              initargs=(str(ROOT / 'data' / 'kmer_selection' / 'selected_kmers.txt'),)
              ) as pool:
        for idx, kmer, asm, err in pool.imap_unordered(mod._feat_worker, work,
                                                       chunksize=4):
            if err:
                print(f'    WARNING idx={idx}: {err}')
            else:
                got[idx] = (kmer, asm)
    idxs = sorted(got)
    kn = normalizer.normalize_kmer(
        np.stack([got[i][0] for i in idxs])).astype(np.float32)
    an = normalizer.normalize_assembly(
        np.stack([got[i][1] for i in idxs])).astype(np.float32)
    in_names = [i.name for i in session.get_inputs()]
    on = session.get_outputs()[0].name
    preds = np.zeros((len(idxs), 2), dtype=np.float32)
    for s in range(0, len(idxs), 64):
        e = min(s + 64, len(idxs))
        preds[s:e] = session.run([on], {in_names[0]: kn[s:e],
                                        in_names[1]: an[s:e]})[0]
    df = pd.DataFrame({'primary_accession': sel['primary_accession'].values[idxs],
                       'magicc_v5_completeness': preds[:, 0],
                       'magicc_v5_contamination': preds[:, 1]})
    df.to_csv(out, sep='\t', index=False)
    print(f'    wrote {out}')
    return df


def run_checkm2(threads: int) -> pd.DataFrame | None:
    odir = SET_DIR / 'reference_level_checkm2_output'
    qr = odir / 'quality_report.tsv'
    if not qr.exists():
        print(f'    running CheckM2 on {LINK_DIR} ({threads} threads) ...')
        t0 = time.time()
        cmd = ['conda', 'run', '-n', 'checkm2_py39', 'env',
               f'CHECKM2DB={CHECKM2_DB}', 'checkm2', 'predict', '--threads',
               str(threads), '-x', '.fasta', '--input', str(LINK_DIR),
               '--output-directory', str(odir), '--force']
        log = LOGS / 'ws1.11_checkm2_references.log'
        with open(log, 'w') as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f'    CheckM2 rc={rc}, {time.time() - t0:.0f}s')
        if rc != 0 or not qr.exists():
            print(f'    CheckM2 FAILED — see {log}')
            return None
    q = pd.read_csv(qr, sep='\t')
    q = q.rename(columns={'Name': 'primary_accession',
                          'Completeness': 'checkm2_local_completeness',
                          'Contamination': 'checkm2_local_contamination'})
    return q[['primary_accession', 'checkm2_local_completeness',
              'checkm2_local_contamination']]


def run_cocopye(threads: int) -> pd.DataFrame | None:
    raw = SET_DIR / 'reference_level_cocopye.csv'
    if not raw.exists():
        print(f'    running CoCoPyE on {LINK_DIR} ({threads} threads) ...')
        t0 = time.time()
        cmd = ['conda', 'run', '-n', 'magicc2', 'cocopye', 'run', '-i', str(LINK_DIR),
               '-o', str(raw), '-t', str(threads), '-v', 'full']
        log = LOGS / 'ws1.11_cocopye_references.log'
        with open(log, 'w') as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f'    CoCoPyE rc={rc}, {time.time() - t0:.0f}s')
        if rc != 0 or not raw.exists():
            print(f'    CoCoPyE FAILED — see {log}')
            return None
    # Stage-3 (marker + neural network), stage-2 fallback, 0-1 -> 0-100 %.
    # Identical convention to scripts/91_parse_competitor_clean_cd.py::do_cocopye,
    # so these values are comparable with every other CoCoPyE number in the revision.
    c = pd.read_csv(raw)
    if '3_completeness' in c.columns:
        comp_v = c['3_completeness'].fillna(
            c.get('2_completeness', pd.Series(np.nan, index=c.index))) * 100.0
        cont_v = c['3_contamination'].fillna(
            c.get('2_contamination', pd.Series(np.nan, index=c.index))) * 100.0
    elif '2_completeness' in c.columns:
        comp_v = c['2_completeness'] * 100.0
        cont_v = c['2_contamination'] * 100.0
    else:
        print(f'    CoCoPyE columns not recognised: {list(c.columns)}')
        return None
    return pd.DataFrame({
        'primary_accession': c['bin'].astype(str).str.replace(r'\.fasta$', '',
                                                              regex=True),
        'cocopye_completeness': comp_v.to_numpy(float),
        'cocopye_contamination': cont_v.to_numpy(float)})


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
    print(f'  created {build_links(sel)} new .fasta symlinks in {LINK_DIR}')

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
