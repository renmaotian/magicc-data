#!/usr/bin/env python3
"""
WS1.11 (R1-m13) -- does the CheckM2-based reference curation bias the conclusions?

THE COMPARISON THAT ANSWERS THE OBJECTION
-----------------------------------------
``set_H_ncbi`` holds two arms of reference genomes chosen by NCBI's ``assembly_level ==
"Complete Genome"`` annotation with NO CheckM2 filter, matched 1:1 at the deepest
available taxonomic rank and by genome size:

    H_pass : would have PASSED the project's CheckM2-based curation filter
    H_fail : would have FAILED it -- the genomes the curation removed, which therefore
             appear in no MAGICC training set and in no previous MAGICC benchmark

Both arms of a matched pair receive the *same* target completeness / contamination
draws at each replicate, so every comparison below is paired at the sample level and
clustered at the reference-pair level.

Primary estimator (per tool, per metric)

    d_i = |error on the H_fail sample|  -  |error on its matched H_pass sample|
    D   = mean(d_i)  =  MAE_H_fail - MAE_H_pass

with a cluster bootstrap over the 200 reference pairs, a two-sided bootstrap p-value
against D = 0, Benjamini-Hochberg correction over the whole test family, and
Hodges-Lehmann / rank-biserial effect sizes with CIs.

Second estimator -- difference-in-differences against CheckM2

    DiD = D(MAGICC) - D(CheckM2)

Both tools score the identical samples, so DiD is immune to any property of the H_fail
genomes that makes them intrinsically harder for every method. A DiD near zero means
MAGICC is no more affected by the curation boundary than the tool that defined it.

CheckM2's own error on H_fail is reported in its own right: if CheckM2 is wrong on the
genomes its own scores excluded, that is precisely the blind spot the curation created.

MECHANISM CHECK
---------------
A reference the curation rejected for low CheckM2 completeness may genuinely be missing
sequence. The benchmark defines completeness as *retained bp / bp of the deposited
reference*, so if a reference is really 94 % complete, a tool that estimates biological
completeness will read ~6 pp low and be scored as biased. Per-reference signed bias is
therefore regressed on the reference's CheckM2 completeness deficit. For MAGICC this is
an independent test (MAGICC never saw the CheckM2 score); for CheckM2 it is circular by
construction and is labelled as such.

CONVENTIONS
-----------
R2 = coefficient of determination (never squared Pearson), omitted where the truth has
~zero variance (R1-m19). Bootstrap seeds come from ``fw.stable_hash`` (CRC-32) with
PYTHONHASHSEED=0. Figures use the CVD-safe palette of scripts/101_metrics_framework.py
(no red/green discrimination, E4). Every percentage states its denominator (R1-M5).

Usage:
    PYTHONHASHSEED=0 python scripts/191_ws1_11_analysis.py [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/path/to/magicc')
_HERE = ROOT / 'scripts'


def _load_framework():
    spec = importlib.util.spec_from_file_location(
        'magicc_metrics_framework', _HERE / '101_metrics_framework.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['magicc_metrics_framework'] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()
plt = fw.init_matplotlib()

SET_DIR = ROOT / 'data' / 'benchmarks' / 'set_H_ncbi'
OUT = ROOT / 'results' / 'revision' / 'circularity'
FIG = OUT / 'figures'

TOOLS = [('magicc_v5', 'magicc_v5_predictions.tsv', 'MAGICC V5'),
         ('checkm2', 'checkm2_predictions.tsv', 'CheckM2 1.0.1'),
         ('cocopye', 'cocopye_predictions.tsv', 'CoCoPyE 0.5.0'),
         ('deepcheck', 'deepcheck_predictions.tsv', 'DeepCheck')]
METRICS = [('completeness', 'true_completeness', 'pred_completeness'),
           ('contamination', 'true_contamination', 'pred_contamination')]
SEVERITY_ORDER = ['mild', 'moderate', 'severe']
# ordered lightness ramp (inherently CVD-safe; verified below with palette_cvd_report)
SEV_COLOURS = {'pass': '#C6DBEF', 'mild': '#6BAED6',
               'moderate': '#2171B5', 'severe': '#08306B'}
ARM_HATCH = {'H_pass': '', 'H_fail': '///'}


def seed_for(*parts) -> int:
    return fw.stable_hash('|'.join(str(p) for p in parts)) % (2 ** 31)


# --------------------------------------------------------------------- loading
def load_long() -> pd.DataFrame:
    meta = pd.read_csv(SET_DIR / 'metadata.tsv', sep='\t')
    gen = pd.read_csv(SET_DIR / 'generation_metadata.tsv', sep='\t')
    base = meta.merge(
        gen[['genome_id', 'match_level', 'ref_checkm2_completeness',
             'ref_checkm2_contamination', 'ref_contig_count', 'ref_n50_contigs',
             'dominant_taxonomy', 'dominant_reference_bp', 'target_completeness',
             'target_contamination']], on='genome_id', how='left', validate='1:1')
    frames = []
    available = []
    for tool, fn, label in TOOLS:
        p = SET_DIR / fn
        if not p.exists():
            print(f'  [{tool}] MISSING {p.name} — skipped')
            continue
        pr = pd.read_csv(p, sep='\t')
        need = {'genome_id', 'pred_completeness', 'pred_contamination'}
        if not need <= set(pr.columns):
            print(f'  [{tool}] missing columns — skipped')
            continue
        d = base.merge(pr[['genome_id', 'pred_completeness', 'pred_contamination']],
                       on='genome_id', how='inner', validate='1:1')
        if len(d) != len(base):
            print(f'  [{tool}] WARNING: {len(base) - len(d)} genomes missing from '
                  f'predictions')
        d['tool'] = tool
        d['tool_label'] = label
        frames.append(d)
        available.append(tool)
        print(f'  [{tool}] {len(d)} rows')
    if not frames:
        raise SystemExit('FATAL: no tool predictions found')
    long = pd.concat(frames, ignore_index=True)
    long['abs_err_completeness'] = (long['pred_completeness']
                                    - long['true_completeness']).abs()
    long['abs_err_contamination'] = (long['pred_contamination']
                                     - long['true_contamination']).abs()
    long['err_completeness'] = long['pred_completeness'] - long['true_completeness']
    long['err_contamination'] = long['pred_contamination'] - long['true_contamination']
    long['checkm2_completeness_deficit'] = (
        98.0 - long['ref_checkm2_completeness']).clip(lower=0)
    long['checkm2_contamination_excess'] = (
        long['ref_checkm2_contamination'] - 2.0).clip(lower=0)
    add_adjusted_truth(long)
    return long, available


def add_adjusted_truth(long: pd.DataFrame) -> None:
    """SENSITIVITY ONLY -- truth corrected for the reference genome's own imperfection.

    The benchmark defines completeness and contamination relative to the *deposited*
    reference assembly, which it treats as a complete, clean genome. For the
    would-have-failed arm that assumption is by construction questionable. Under a
    first-order model in which the deposited reference of length L is a fraction
    phi = r_cont / (r_comp + r_cont) foreign sequence and represents r_comp % of the
    organism's true genome g, so that g = 100 L (1 - phi) / r_comp, and in which
    fragmentation removes bp uniformly:

        adjusted completeness  = observed completeness * r_comp / 100
        adjusted contamination = (observed completeness * phi + observed contamination)
                                 / 100 * r_comp / (1 - phi)

    Both r_comp and r_cont are CheckM2's own scores for the reference, so this
    correction structurally favours CheckM2 and CANNOT be a primary result. Its only
    use is to ask whether MAGICC's excess error on the would-have-failed arm is
    consistent with the references themselves being imperfect rather than with the
    model failing.
    """
    r_comp = long['ref_checkm2_completeness'].to_numpy(float)
    r_cont = long['ref_checkm2_contamination'].to_numpy(float)
    denom = np.where((r_comp + r_cont) > 0, r_comp + r_cont, np.nan)
    phi = np.nan_to_num(r_cont / denom, nan=0.0)
    comp = long['true_completeness'].to_numpy(float)
    cont = long['true_contamination'].to_numpy(float)
    long['adj_true_completeness'] = comp * r_comp / 100.0
    long['adj_true_contamination'] = ((comp * phi + cont) / 100.0
                                      * r_comp / np.maximum(1e-9, 1.0 - phi))
    long['reference_foreign_fraction_phi'] = phi
    long['abs_err_adj_completeness'] = (long['pred_completeness']
                                        - long['adj_true_completeness']).abs()
    long['abs_err_adj_contamination'] = (long['pred_contamination']
                                         - long['adj_true_contamination']).abs()
    long['err_adj_completeness'] = (long['pred_completeness']
                                    - long['adj_true_completeness'])
    long['err_adj_contamination'] = (long['pred_contamination']
                                     - long['adj_true_contamination'])


# --------------------------------------------------------------------- metrics
def block(d: pd.DataFrame, n_boot: int, seed: int, cluster_col='dominant_accession'):
    """MAE / bias / RMSE / R2(CoD) for both metrics with a cluster bootstrap CI."""
    bs = fw.Bootstrapper(clusters=d[cluster_col].to_numpy(), n_iter=n_boot,
                         ci_level=0.95, seed=seed)
    tc = d['true_completeness'].to_numpy(float)
    pc = d['pred_completeness'].to_numpy(float)
    tx = d['true_contamination'].to_numpy(float)
    px = d['pred_contamination'].to_numpy(float)

    def stats(idx):
        return {
            'comp_mae': fw.mae(tc[idx], pc[idx]),
            'comp_bias': fw.bias(tc[idx], pc[idx]),
            'comp_rmse': fw.rmse(tc[idx], pc[idx]),
            'cont_mae': fw.mae(tx[idx], px[idx]),
            'cont_bias': fw.bias(tx[idx], px[idx]),
            'cont_rmse': fw.rmse(tx[idx], px[idx]),
        }

    res = bs.ci_multi(stats)
    out = {'n': int(len(d)), 'n_clusters': int(bs.n_clusters)}
    for k, v in res.items():
        out[k] = round(v['estimate'], 4)
        out[f'{k}_ci_lo'] = round(v['ci_lo'], 4)
        out[f'{k}_ci_hi'] = round(v['ci_hi'], 4)
    # R2 = coefficient of determination; omitted where the truth has ~zero variance
    for name, t, p in (('comp', tc, pc), ('cont', tx, px)):
        out[f'{name}_true_sd'] = round(float(np.std(t)), 4)
        out[f'{name}_r2_cod'] = (round(fw.r2_coefficient_of_determination(t, p), 4)
                                 if np.std(t) > 1e-6 else '')
    return out


def paired_diff(long: pd.DataFrame, tool: str, metric_col: str, n_boot: int,
                subset_mask=None):
    """d_i = value(H_fail) - value(matched H_pass), clustered on pair_id."""
    d = long[long['tool'] == tool]
    if subset_mask is not None:
        keep_pairs = set(d.loc[subset_mask(d), 'pair_id'])
        d = d[d['pair_id'].isin(keep_pairs)]
    piv = d.pivot_table(index=['pair_id', 'replicate'], columns='arm',
                        values=metric_col)
    piv = piv.dropna()
    if 'H_fail' not in piv or 'H_pass' not in piv or piv.empty:
        return None
    diff = (piv['H_fail'] - piv['H_pass']).to_numpy(float)
    pairs = piv.index.get_level_values('pair_id').to_numpy()
    bs = fw.Bootstrapper(clusters=pairs, n_iter=n_boot, ci_level=0.95,
                         seed=seed_for('paired', tool, metric_col))
    ci = bs.ci(lambda idx: float(np.mean(diff[idx])))
    p, _ = bs.p_two_sided(lambda idx: float(np.mean(diff[idx])), null=0.0)
    hl = fw.hodges_lehmann_paired(diff)
    rb = fw.rank_biserial_paired(diff)
    return {'tool': tool, 'metric': metric_col, 'n_paired_samples': int(diff.size),
            'n_pairs': int(len(np.unique(pairs))),
            'mean_H_fail': round(float(piv['H_fail'].mean()), 4),
            'mean_H_pass': round(float(piv['H_pass'].mean()), 4),
            'D_fail_minus_pass': round(ci['estimate'], 4),
            'ci_lo': round(ci['ci_lo'], 4), 'ci_hi': round(ci['ci_hi'], 4),
            'p_two_sided': p,
            'hodges_lehmann': round(float(hl), 4),
            'rank_biserial': round(float(rb), 4)}


def did_vs(long: pd.DataFrame, tool_a: str, tool_b: str, metric_col: str, n_boot: int):
    """DiD = D(tool_a) - D(tool_b) on the identical matched samples."""
    def arm_pivot(tool):
        d = long[long['tool'] == tool]
        return d.pivot_table(index=['pair_id', 'replicate'], columns='arm',
                             values=metric_col)
    pa, pb = arm_pivot(tool_a), arm_pivot(tool_b)
    idx = pa.dropna().index.intersection(pb.dropna().index)
    if len(idx) == 0:
        return None
    da = (pa.loc[idx, 'H_fail'] - pa.loc[idx, 'H_pass']).to_numpy(float)
    db = (pb.loc[idx, 'H_fail'] - pb.loc[idx, 'H_pass']).to_numpy(float)
    did = da - db
    pairs = idx.get_level_values('pair_id').to_numpy()
    bs = fw.Bootstrapper(clusters=pairs, n_iter=n_boot, ci_level=0.95,
                         seed=seed_for('did', tool_a, tool_b, metric_col))
    ci = bs.ci(lambda i: float(np.mean(did[i])))
    p, _ = bs.p_two_sided(lambda i: float(np.mean(did[i])), null=0.0)
    return {'tool_a': tool_a, 'tool_b': tool_b, 'metric': metric_col,
            'n_paired_samples': int(did.size), 'n_pairs': int(len(np.unique(pairs))),
            'D_tool_a': round(float(da.mean()), 4),
            'D_tool_b': round(float(db.mean()), 4),
            'DiD': round(ci['estimate'], 4), 'ci_lo': round(ci['ci_lo'], 4),
            'ci_hi': round(ci['ci_hi'], 4), 'p_two_sided': p}


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = fw.load_config()
    NB = args.n_boot

    print('=' * 92)
    print('WS1.11 (R1-m13) — circularity safeguard analysis on set_H_ncbi')
    print('=' * 92)
    print('\n[0] loading predictions')
    long, tools = load_long()
    denom = fw.caption_denominator(cfg)

    # palette check (E4)
    rep = fw.palette_cvd_report(SEV_COLOURS)
    min_de = float(rep['min_delta_e'].min())
    print(f'  severity ramp min CIE76 Delta-E over normal/deutan/protan/tritan: '
          f'{min_de:.1f}')
    rep.to_csv(OUT / 'ws1_11_palette_cvd_report.tsv', sep='\t', index=False)

    n_pairs = long['pair_id'].nunique()
    print(f'  {len(long) // max(1, len(tools))} samples x {len(tools)} tools; '
          f'{n_pairs} matched reference pairs')

    # ------------------------------------------------- 1. arm-level metrics
    print('\n[1] arm-level metrics (cluster bootstrap over reference genomes)')
    rows = []
    for tool in tools:
        for arm in ('H_pass', 'H_fail'):
            d = long[(long['tool'] == tool) & (long['arm'] == arm)]
            r = {'tool': tool, 'group': arm,
                 'group_description': ('references that WOULD HAVE PASSED the '
                                       'CheckM2-based curation filter'
                                       if arm == 'H_pass' else
                                       'references that WOULD HAVE FAILED it')}
            r.update(block(d, NB, seed_for('arm', tool, arm)))
            rows.append(r)
        # severity strata inside H_fail, plus their matched controls
        for sev in SEVERITY_ORDER:
            fail = long[(long['tool'] == tool) & (long['arm'] == 'H_fail')
                        & (long['checkm2_severity'] == sev)]
            if fail.empty:
                continue
            pids = set(fail['pair_id'])
            ctrl = long[(long['tool'] == tool) & (long['arm'] == 'H_pass')
                        & (long['pair_id'].isin(pids))]
            for lbl, d in ((f'H_fail:{sev}', fail),
                           (f'H_pass:matched_to_{sev}', ctrl)):
                r = {'tool': tool, 'group': lbl,
                     'group_description': f'CheckM2-deficit severity stratum "{sev}"'}
                r.update(block(d, NB, seed_for('sev', tool, lbl)))
                rows.append(r)
    arm_metrics = pd.DataFrame(rows)
    arm_metrics.to_csv(OUT / 'ws1_11_arm_metrics.tsv', sep='\t', index=False)
    print(arm_metrics[arm_metrics['group'].isin(['H_pass', 'H_fail'])][
        ['tool', 'group', 'n', 'n_clusters', 'comp_mae', 'comp_mae_ci_lo',
         'comp_mae_ci_hi', 'comp_bias', 'comp_r2_cod', 'cont_mae', 'cont_mae_ci_lo',
         'cont_mae_ci_hi', 'cont_bias', 'cont_r2_cod']].to_string(index=False))

    # ------------------------------------- 2. paired arm test (the primary result)
    print('\n[2] paired H_fail - H_pass tests (primary estimator)')
    prows = []
    for tool in tools:
        for mname, tcol, pcol in METRICS:
            for stat, col in (('abs_error', f'abs_err_{mname}'),
                              ('signed_error', f'err_{mname}')):
                r = paired_diff(long, tool, col, NB)
                if r is None:
                    continue
                r['statistic'] = stat
                r['metric_name'] = mname
                r['subset'] = 'all pairs'
                prows.append(r)
        # severity-restricted (absolute error only)
        for sev in SEVERITY_ORDER:
            for mname, _, _ in METRICS:
                r = paired_diff(
                    long, tool, f'abs_err_{mname}', NB,
                    subset_mask=lambda d, s=sev: (d['arm'] == 'H_fail')
                    & (d['checkm2_severity'] == s))
                if r is None:
                    continue
                r['statistic'] = 'abs_error'
                r['metric_name'] = mname
                r['subset'] = f'severity={sev}'
                prows.append(r)
        # residual-confounding control: pairs matched at genus level or better
        for mname, _, _ in METRICS:
            r = paired_diff(
                long, tool, f'abs_err_{mname}', NB,
                subset_mask=lambda d: d['match_level'].isin(['species', 'genus']))
            if r is None:
                continue
            r['statistic'] = 'abs_error'
            r['metric_name'] = mname
            r['subset'] = 'match_level in {species, genus}'
            prows.append(r)
        # sensitivity: truth corrected for the reference genome's own imperfection
        for mname, _, _ in METRICS:
            r = paired_diff(long, tool, f'abs_err_adj_{mname}', NB)
            if r is None:
                continue
            r['statistic'] = 'abs_error_vs_adjusted_truth'
            r['metric_name'] = mname
            r['subset'] = 'all pairs'
            prows.append(r)
    paired = pd.DataFrame(prows)
    # BH over the whole family produced in this run
    paired['q_bh'] = fw.bh_correct(paired['p_two_sided'].to_numpy())
    paired['significant_bh_0.05'] = paired['q_bh'] < 0.05
    paired = paired[['tool', 'metric_name', 'statistic', 'subset', 'n_paired_samples',
                     'n_pairs', 'mean_H_pass', 'mean_H_fail', 'D_fail_minus_pass',
                     'ci_lo', 'ci_hi', 'hodges_lehmann', 'rank_biserial',
                     'p_two_sided', 'q_bh', 'significant_bh_0.05']]
    paired.to_csv(OUT / 'ws1_11_paired_arm_tests.tsv', sep='\t', index=False)
    print(paired[(paired['subset'] == 'all pairs')
                 & (paired['statistic'] == 'abs_error')].to_string(index=False))

    # ------------------------- 2a. stratified within matched true-value bands
    print('\n[2a] error stratified by true-value band (composition-controlled)')
    srows = []
    comp_bins = [50, 60, 70, 80, 90, 100.0001]
    cont_bins = [0, 5, 10, 20, 40, 70, 100.0001]
    for tool in tools:
        for axis, bins in (('completeness', comp_bins), ('contamination', cont_bins)):
            for lo, hi in zip(bins[:-1], bins[1:]):
                for armv in ('H_pass', 'H_fail'):
                    d = long[(long['tool'] == tool) & (long['arm'] == armv)]
                    m = (d[f'true_{axis}'] >= lo) & (d[f'true_{axis}'] < hi)
                    if m.sum() == 0:
                        continue
                    s = d[m]
                    srows.append({
                        'tool': tool, 'arm': armv, 'stratify_by': f'true_{axis}',
                        'bin': f'[{lo:g},{min(hi, 100):g})', 'n': int(m.sum()),
                        'comp_mae': round(float(s['abs_err_completeness'].mean()), 4),
                        'comp_bias': round(float(s['err_completeness'].mean()), 4),
                        'cont_mae': round(float(s['abs_err_contamination'].mean()), 4),
                        'cont_bias': round(float(s['err_contamination'].mean()), 4)})
    strat = pd.DataFrame(srows)
    strat.to_csv(OUT / 'ws1_11_stratified_by_true_value.tsv', sep='\t', index=False)
    print(strat[(strat.tool == 'magicc_v5')
                & (strat.stratify_by == 'true_contamination')][
        ['arm', 'bin', 'n', 'cont_bias', 'cont_mae']].to_string(index=False))

    # ---------------------------- 2b. adjusted-truth sensitivity (arm-level MAE)
    print('\n[2b] sensitivity: truth corrected for the reference genome itself')
    arows = []
    for tool in tools:
        for arm in ('H_pass', 'H_fail'):
            d = long[(long['tool'] == tool) & (long['arm'] == arm)]
            arows.append({
                'tool': tool, 'arm': arm, 'n': int(len(d)),
                'comp_mae_raw_truth': round(float(d['abs_err_completeness'].mean()), 4),
                'comp_mae_adjusted_truth': round(
                    float(d['abs_err_adj_completeness'].mean()), 4),
                'cont_mae_raw_truth': round(float(d['abs_err_contamination'].mean()), 4),
                'cont_mae_adjusted_truth': round(
                    float(d['abs_err_adj_contamination'].mean()), 4),
                'mean_reference_foreign_fraction_phi': round(
                    float(d['reference_foreign_fraction_phi'].mean()), 5),
                'mean_ref_checkm2_completeness': round(
                    float(d['ref_checkm2_completeness'].mean()), 3),
            })
    adj = pd.DataFrame(arows)
    adj.to_csv(OUT / 'ws1_11_adjusted_truth_sensitivity.tsv', sep='\t', index=False)
    print(adj.to_string(index=False))

    # --------------------------------------------- 3. difference-in-differences
    print('\n[3] difference-in-differences, MAGICC vs each competitor')
    drows = []
    for other in [t for t in tools if t != 'magicc_v5']:
        for mname, _, _ in METRICS:
            r = did_vs(long, 'magicc_v5', other, f'abs_err_{mname}', NB)
            if r:
                r['metric_name'] = mname
                r['statistic'] = 'abs_error'
                drows.append(r)
            r = did_vs(long, 'magicc_v5', other, f'err_{mname}', NB)
            if r:
                r['metric_name'] = mname
                r['statistic'] = 'signed_error'
                drows.append(r)
    did = pd.DataFrame(drows)
    if len(did):
        did['q_bh'] = fw.bh_correct(did['p_two_sided'].to_numpy())
        did['significant_bh_0.05'] = did['q_bh'] < 0.05
        did.to_csv(OUT / 'ws1_11_did_vs_competitors.tsv', sep='\t', index=False)
        print(did.to_string(index=False))

    # --------------------------------------- 4. MIMAG-inspired threshold behaviour
    print('\n[4] MIMAG-inspired classification and decision thresholds')
    mrows, trows = [], []
    for tool in tools:
        for arm in ('H_pass', 'H_fail'):
            d = long[(long['tool'] == tool) & (long['arm'] == arm)]
            t_cls = fw.mimag_classify(d['true_completeness'], d['true_contamination'],
                                      cfg)
            p_cls = fw.mimag_classify(d['pred_completeness'], d['pred_contamination'],
                                      cfg)
            cm = fw.classification_metrics(t_cls, p_cls, cfg.mimag_classes)
            clusters = d['dominant_accession'].to_numpy()
            bs = fw.Bootstrapper(clusters=clusters, n_iter=NB, ci_level=0.95,
                                 seed=seed_for('mimag', tool, arm))
            tc = fw.encode_labels(t_cls, cfg.mimag_classes)
            pc = fw.encode_labels(p_cls, cfg.mimag_classes)

            # observed-class mask fixed on the FULL group, so the bootstrap
            # estimator keeps one definition across replicates (framework note)
            obs_mask = cm['support'] > 0

            def f1(idx, tc=tc, pc=pc, obs=obs_mask):
                m = fw.metrics_from_cm(
                    fw.cm_from_codes(tc[idx], pc[idx], len(cfg.mimag_classes)),
                    cfg.mimag_classes, observed_mask=obs)
                return float(m['macro_f1'])
            ci = bs.ci(f1)
            row = {'tool': tool, 'arm': arm, 'n': int(len(d)),
                   'n_clusters': int(bs.n_clusters),
                   'macro_f1': round(float(cm['macro_f1']), 4),
                   'macro_f1_ci_lo': round(ci['ci_lo'], 4),
                   'macro_f1_ci_hi': round(ci['ci_hi'], 4),
                   'macro_f1_observed': round(float(cm['macro_f1_observed']), 4),
                   'cohen_kappa': round(float(cm['cohen_kappa']), 4),
                   'n_true_high': int(np.sum(t_cls == 'high')),
                   'n_true_medium': int(np.sum(t_cls == 'medium')),
                   'n_true_low': int(np.sum(t_cls == 'low'))}
            for ci_, cls in enumerate(cfg.mimag_classes):
                row[f'{cls}_precision'] = round(float(cm['precision'][ci_]), 4)
                row[f'{cls}_recall'] = round(float(cm['recall'][ci_]), 4)
                row[f'{cls}_f1'] = round(float(cm['f1'][ci_]), 4)
            mrows.append(row)

            for crit, taus in (('contamination', cfg.raw['thresholds']['contamination']),
                               ('completeness', cfg.raw['thresholds']['completeness'])):
                t = d[f'true_{crit}'].to_numpy(float)
                p = d[f'pred_{crit}'].to_numpy(float)
                for tau in taus:
                    tm = fw.threshold_metrics(t, p, float(tau), crit)
                    bs2 = fw.Bootstrapper(clusters=clusters, n_iter=NB, ci_level=0.95,
                                          seed=seed_for('thr', tool, arm, crit, tau))

                    def st(idx, t=t, p=p, tau=float(tau), crit=crit):
                        m = fw.threshold_metrics(t[idx], p[idx], tau, crit)
                        return {'false_fail_rate': m['false_fail_rate'],
                                'false_pass_rate': m['false_pass_rate'],
                                'balanced_accuracy': m['balanced_accuracy']}
                    cis = bs2.ci_multi(st)
                    trows.append({
                        'tool': tool, 'arm': arm, 'criterion': crit, 'tau': float(tau),
                        'n': tm['n'], 'n_true_pass': tm['n_true_pass'],
                        'n_true_fail': tm['n_true_fail'],
                        'n_false_fail': tm['n_false_fail'],
                        'n_false_pass': tm['n_false_pass'],
                        'false_fail_rate': round(tm['false_fail_rate'], 4),
                        'false_fail_ci_lo': round(cis['false_fail_rate']['ci_lo'], 4),
                        'false_fail_ci_hi': round(cis['false_fail_rate']['ci_hi'], 4),
                        'false_pass_rate': round(tm['false_pass_rate'], 4),
                        'false_pass_ci_lo': round(cis['false_pass_rate']['ci_lo'], 4),
                        'false_pass_ci_hi': round(cis['false_pass_rate']['ci_hi'], 4),
                        'balanced_accuracy': round(tm['balanced_accuracy'], 4),
                        'bal_acc_ci_lo': round(cis['balanced_accuracy']['ci_lo'], 4),
                        'bal_acc_ci_hi': round(cis['balanced_accuracy']['ci_hi'], 4),
                    })
    mimag = pd.DataFrame(mrows)
    thr = pd.DataFrame(trows)
    mimag.to_csv(OUT / 'ws1_11_mimag.tsv', sep='\t', index=False)
    thr.to_csv(OUT / 'ws1_11_thresholds.tsv', sep='\t', index=False)
    print(mimag[['tool', 'arm', 'n', 'macro_f1', 'macro_f1_ci_lo',
                 'macro_f1_ci_hi']].to_string(index=False))
    print(thr[(thr.criterion == 'contamination') & (thr.tau == 5.0)][
        ['tool', 'arm', 'n_true_pass', 'n_true_fail', 'false_fail_rate',
         'false_pass_rate', 'balanced_accuracy']].to_string(index=False))

    # ------------------------------- 5. mechanism: reference incompleteness
    print('\n[5] mechanism — per-reference bias vs the reference CheckM2 deficit')
    from scipy import stats as sstats
    mech_rows, per_ref_rows = [], []
    for tool in tools:
        d = long[long['tool'] == tool]
        g = d.groupby(['dominant_accession', 'arm', 'pair_id', 'checkm2_severity',
                       'ref_checkm2_completeness', 'ref_checkm2_contamination',
                       'checkm2_completeness_deficit', 'checkm2_contamination_excess',
                       'dominant_phylum'], as_index=False).agg(
            comp_bias=('err_completeness', 'mean'),
            cont_bias=('err_contamination', 'mean'),
            comp_mae=('abs_err_completeness', 'mean'),
            cont_mae=('abs_err_contamination', 'mean'), n=('genome_id', 'size'))
        g['tool'] = tool
        per_ref_rows.append(g)
        for yname, xname in (('comp_bias', 'checkm2_completeness_deficit'),
                             ('cont_bias', 'checkm2_contamination_excess')):
            x = g[xname].to_numpy(float)
            y = g[yname].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10 or np.std(x[ok]) == 0:
                continue
            rho, prho = sstats.spearmanr(x[ok], y[ok])
            sl, ic, rv, pv, se = sstats.linregress(x[ok], y[ok])
            mech_rows.append({
                'tool': tool, 'y': yname, 'x': xname, 'n_references': int(ok.sum()),
                'spearman_rho': round(float(rho), 4),
                'spearman_p': float(prho),
                'ols_slope': round(float(sl), 4),
                'ols_slope_se': round(float(se), 4),
                'ols_intercept': round(float(ic), 4),
                'ols_p': float(pv),
                'independent_of_the_x_variable': (tool != 'checkm2'),
                'note': ('CIRCULAR: the x variable IS this tool\'s own estimate'
                         if tool == 'checkm2' else
                         'independent: this tool never sees the CheckM2 score')})
    mech = pd.DataFrame(mech_rows)
    per_ref = pd.concat(per_ref_rows, ignore_index=True)
    mech.to_csv(OUT / 'ws1_11_reference_incompleteness_mechanism.tsv', sep='\t',
                index=False)
    per_ref.to_csv(OUT / 'ws1_11_per_reference_errors.tsv', sep='\t', index=False)
    if len(mech):
        print(mech.to_string(index=False))

    # ------------------------------------------------- 6. context: other sets
    print('\n[6] context — MAGICC V5 on the CheckM2-filtered benchmark sets')
    ctx = []
    for name in ('set_A_v2', 'set_B_v2', 'set_C_clean', 'set_D_clean', 'set_E'):
        d = ROOT / 'data' / 'benchmarks' / name
        pf, mf = d / 'magicc_v5_predictions.tsv', d / 'metadata.tsv'
        if not (pf.exists() and mf.exists()):
            continue
        p = pd.read_csv(pf, sep='\t')
        if 'true_completeness' not in p.columns:
            m = pd.read_csv(mf, sep='\t')
            p = m.merge(p[['genome_id', 'pred_completeness', 'pred_contamination']],
                        on='genome_id')
        cl = ('dominant_accession' if 'dominant_accession' in p.columns
              else 'genome_id')
        r = {'tool': 'magicc_v5', 'group': name,
             'group_description': 'references selected under the CheckM2-based filter'}
        r.update(block(p, NB, seed_for('ctx', name), cluster_col=cl))
        ctx.append(r)
    for arm in ('H_pass', 'H_fail'):
        r = arm_metrics[(arm_metrics.tool == 'magicc_v5')
                        & (arm_metrics.group == arm)].iloc[0].to_dict()
        ctx.append(r)
    ctxdf = pd.DataFrame(ctx)
    ctxdf.to_csv(OUT / 'ws1_11_context_other_sets.tsv', sep='\t', index=False)
    print(ctxdf[['group', 'n', 'n_clusters', 'comp_mae', 'comp_mae_ci_lo',
                 'comp_mae_ci_hi', 'cont_mae', 'cont_mae_ci_lo',
                 'cont_mae_ci_hi']].to_string(index=False))

    # ------------------------------------------------------------- 7. figures
    print('\n[7] figures (CVD-safe, no red/green discrimination)')
    captions = []
    make_figures(long, arm_metrics, paired, thr, per_ref, cfg, denom, captions, tools)
    (OUT / 'ws1_11_figure_captions.md').write_text('\n'.join(captions))

    # -------------------------------------------------------------- summary
    def get(tool, arm, col):
        s = arm_metrics[(arm_metrics.tool == tool) & (arm_metrics.group == arm)]
        return float(s.iloc[0][col]) if len(s) else float('nan')

    def getp(tool, metric, stat='abs_error', subset='all pairs'):
        s = paired[(paired.tool == tool) & (paired.metric_name == metric)
                   & (paired.statistic == stat) & (paired.subset == subset)]
        return s.iloc[0].to_dict() if len(s) else {}

    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'generated_by': 'scripts/191_ws1_11_analysis.py',
        'set': 'set_H_ncbi',
        'design': {
            'reference_selector': 'NCBI assembly_level == "Complete Genome" '
                                  '(assembly_summary_{genbank,refseq}.txt), '
                                  'version_status == latest; NO CheckM2 filter',
            'n_reference_pairs': int(n_pairs),
            'n_samples_per_tool': int(len(long) // max(1, len(tools))),
            'simulations_per_reference': 10,
            'arms': {'H_fail': 'would have FAILED the project CheckM2 curation filter',
                     'H_pass': 'matched controls that would have passed'},
            'generator': 'scripts/073_generate_clean_cd_benchmarks.py logic verbatim',
        },
        'bootstrap': {'n_iter': NB, 'cluster_unit': 'reference genome (arm metrics) / '
                                                    'matched reference pair (paired '
                                                    'tests)',
                      'seeding': 'fw.stable_hash (CRC-32), PYTHONHASHSEED=0'},
        'palette_min_delta_e': min_de,
        'headline': {
            'magicc_comp_mae_H_pass': get('magicc_v5', 'H_pass', 'comp_mae'),
            'magicc_comp_mae_H_fail': get('magicc_v5', 'H_fail', 'comp_mae'),
            'magicc_cont_mae_H_pass': get('magicc_v5', 'H_pass', 'cont_mae'),
            'magicc_cont_mae_H_fail': get('magicc_v5', 'H_fail', 'cont_mae'),
            'checkm2_comp_mae_H_pass': get('checkm2', 'H_pass', 'comp_mae'),
            'checkm2_comp_mae_H_fail': get('checkm2', 'H_fail', 'comp_mae'),
            'checkm2_cont_mae_H_pass': get('checkm2', 'H_pass', 'cont_mae'),
            'checkm2_cont_mae_H_fail': get('checkm2', 'H_fail', 'cont_mae'),
            'paired_magicc_completeness': getp('magicc_v5', 'completeness'),
            'paired_magicc_contamination': getp('magicc_v5', 'contamination'),
            'paired_checkm2_completeness': getp('checkm2', 'completeness'),
            'paired_checkm2_contamination': getp('checkm2', 'contamination'),
        },
        'files': sorted(p.name for p in OUT.glob('ws1_11_*')),
    }
    with open(OUT / 'ws1_11_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {OUT / "ws1_11_summary.json"}')
    print('DONE')
    return 0


# ------------------------------------------------------------------- figures
def _save(fig, name, caption, captions):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight')
    captions.append(f'### {name}\n\n{caption}\n')
    plt.close(fig)
    print(f'    {FIG / (name + ".png")}')


def make_figures(long, arm_metrics, paired, thr, per_ref, cfg, denom, captions, tools):
    order = [t for t in ('magicc_v5', 'checkm2', 'cocopye', 'deepcheck') if t in tools]

    # --- Fig 1: signed error distributions by tool x arm ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (mname, _, _) in zip(axes, METRICS):
        pos = 0
        ticks, labels = [], []
        for tool in order:
            col = cfg.tool_colour(tool)
            for arm in ('H_pass', 'H_fail'):
                v = long[(long.tool == tool) & (long.arm == arm)][
                    f'err_{mname}'].to_numpy(float)
                parts = ax.violinplot([v], positions=[pos], widths=0.78,
                                      showextrema=False, showmedians=False)
                for b in parts['bodies']:
                    b.set_facecolor(col)
                    b.set_edgecolor('#222222')
                    b.set_alpha(0.30 if arm == 'H_pass' else 0.65)
                    b.set_linewidth(0.5)
                    b.set_hatch(ARM_HATCH[arm])
                bp = ax.boxplot([v], positions=[pos], widths=0.26, whis=1.5,
                                showfliers=False, patch_artist=True,
                                manage_ticks=False)
                for box in bp['boxes']:
                    box.set(facecolor='white', edgecolor='#222222', linewidth=0.6)
                for k in ('whiskers', 'caps'):
                    for ln in bp[k]:
                        ln.set(color='#222222', linewidth=0.6)
                for md in bp['medians']:
                    md.set(color='#222222', linewidth=1.1)
                ax.plot([pos], [v.mean()], marker=cfg.tool_marker(tool), ms=3.4,
                        mfc=col, mec='#222222', mew=0.5, ls='none', zorder=5)
                ticks.append(pos)
                labels.append('pass' if arm == 'H_pass' else 'FAIL')
                pos += 1
            pos += 0.7
        ax.axhline(0, color='#555555', lw=0.6, ls=':')
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(f'signed error, {mname} (pp)\npredicted − true')
        ax.set_title(f'{mname.capitalize()}', pad=3)
        # tool name as a second row of x labels, below the arm labels
        for i, tool in enumerate(order):
            ax.text(i * 2.7 + 0.5, -0.085, cfg.tool_short(tool),
                    transform=ax.get_xaxis_transform(), ha='center', va='top',
                    fontsize=7)
        ax.tick_params(axis='x', length=0)
    n_per = int((long.tool == order[0]).sum())
    _save(fig, 'ws1_11_fig1_signed_error_by_arm',
          f'Signed error (predicted − true, percentage points) of each tool on '
          f'set_H_ncbi, split by whether the reference genome would have PASSED '
          f'("pass", n={n_per // 2} simulations from 200 references) or FAILED '
          f'("FAIL", hatched, n={n_per // 2} from 200 references) the project\'s '
          f'CheckM2-based curation filter. Violin = kernel density, box = median and '
          f'IQR with 1.5xIQR whiskers, marker = mean. Arms are matched 1:1 by taxonomy '
          f'and genome size and share identical target draws. Colour is CVD-safe and '
          f'always redundant with position, marker and hatch. {denom}', captions)

    # --- Fig 2: paired arm difference with CI -------------------------------
    sub = paired[(paired.subset == 'all pairs') & (paired.statistic == 'abs_error')]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    y = 0
    ticks, labels = [], []
    for mname in ('completeness', 'contamination'):
        for tool in order:
            r = sub[(sub.tool == tool) & (sub.metric_name == mname)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar([r['D_fail_minus_pass']], [y],
                        xerr=[[r['D_fail_minus_pass'] - r['ci_lo']],
                              [r['ci_hi'] - r['D_fail_minus_pass']]],
                        fmt=cfg.tool_marker(tool), ms=5, mfc=cfg.tool_colour(tool),
                        mec='#222222', mew=0.6, ecolor='#222222', elinewidth=1.0,
                        capsize=2.5)
            ticks.append(y)
            labels.append(f'{cfg.tool_short(tool)} — {mname}')
            y += 1
        y += 0.6
    ax.axvline(0, color='#555555', lw=0.8, ls=':')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('D = MAE(would-have-FAILED refs) − MAE(matched would-have-PASSED refs)'
                  '  (pp)')
    _save(fig, 'ws1_11_fig2_paired_arm_difference',
          f'Primary WS1.11 estimator. D is the mean per-sample difference in absolute '
          f'error between a reference the CheckM2-based curation would have rejected '
          f'and its taxonomically matched, size-matched accepted control, on '
          f'simulations that share identical target completeness/contamination draws. '
          f'Bars are 95 % cluster-bootstrap CIs resampling the 200 matched reference '
          f'pairs ({int(sub.iloc[0]["n_paired_samples"])} paired simulations per tool '
          f'per metric, 2,000 replicates). D > 0 means the tool is worse on the '
          f'genomes the curation removed. {denom}', captions)

    # --- Fig 3: MAE by CheckM2-deficit severity -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, mname in zip(axes, ('comp', 'cont')):
        width = 0.8 / max(1, len(order))
        for i, tool in enumerate(order):
            xs, ys, lo, hi = [], [], [], []
            for j, sev in enumerate(['pass'] + SEVERITY_ORDER):
                grp = ('H_pass' if sev == 'pass' else f'H_fail:{sev}')
                r = arm_metrics[(arm_metrics.tool == tool)
                                & (arm_metrics.group == grp)]
                if r.empty:
                    continue
                r = r.iloc[0]
                xs.append(j + (i - (len(order) - 1) / 2) * width)
                ys.append(r[f'{mname}_mae'])
                lo.append(r[f'{mname}_mae'] - r[f'{mname}_mae_ci_lo'])
                hi.append(r[f'{mname}_mae_ci_hi'] - r[f'{mname}_mae'])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=cfg.tool_marker(tool), ms=4.6,
                        mfc=cfg.tool_colour(tool), mec='#222222', mew=0.6,
                        ecolor='#222222', elinewidth=0.9, capsize=2, ls='none',
                        label=cfg.tool_short(tool))
        ax.set_xticks(range(4))
        ax.set_xticklabels(['would pass\n(control)', 'mild', 'moderate', 'severe'])
        ax.set_ylabel(f'{"completeness" if mname == "comp" else "contamination"} '
                      f'MAE (pp)')
        ax.set_xlabel('CheckM2 deficit of the reference genome')
    axes[0].legend(frameon=False, ncol=2, fontsize=6.5)
    one = long[long.tool == order[0]]
    n_sev = {s: int(((one.arm == 'H_fail') & (one.checkm2_severity == s)).sum())
             for s in SEVERITY_ORDER}
    r_sev = {s: int(one.loc[(one.arm == 'H_fail') & (one.checkm2_severity == s),
                            'dominant_accession'].nunique()) for s in SEVERITY_ORDER}
    n_ctl = int((one.arm == 'H_pass').sum())
    r_ctl = int(one.loc[one.arm == 'H_pass', 'dominant_accession'].nunique())
    _save(fig, 'ws1_11_fig3_mae_by_checkm2_deficit',
          f'MAE against the CheckM2 deficit of the reference genome. "mild" = CheckM2 '
          f'completeness ≥96 % and contamination ≤4 % but outside the ≥98 %/≤2 % '
          f'curation filter (n={n_sev["mild"]} simulations from {r_sev["mild"]} '
          f'references); "moderate" = completeness ≥90 % or contamination ≤10 % '
          f'(n={n_sev["moderate"]} from {r_sev["moderate"]}); "severe" = completeness '
          f'<90 % or contamination >10 % (n={n_sev["severe"]} from {r_sev["severe"]}); '
          f'the control column is the matched would-have-passed references '
          f'(n={n_ctl} simulations from {r_ctl}). Bars are 95 % cluster-bootstrap CIs '
          f'over reference genomes. {denom}', captions)

    # --- Fig 4: mechanism scatter -------------------------------------------
    have = [t for t in ('magicc_v5', 'checkm2') if t in tools]
    fig, axes = plt.subplots(1, len(have), figsize=(3.3 * len(have), 2.9),
                             squeeze=False)
    for ax, tool in zip(axes[0], have):
        g = per_ref[per_ref.tool == tool]
        for armv, fc, lab in (('H_pass', 'white', 'would pass'),
                              ('H_fail', cfg.tool_colour(tool), 'would FAIL')):
            gg = g[g.arm == armv]
            ax.scatter(gg['checkm2_completeness_deficit'], gg['comp_bias'], s=12,
                       facecolors=fc, edgecolors=cfg.tool_colour(tool)
                       if armv == 'H_pass' else '#222222', linewidths=0.5,
                       marker=cfg.tool_marker(tool), alpha=0.85, label=lab)
        x = g['checkm2_completeness_deficit'].to_numpy(float)
        y = g['comp_bias'].to_numpy(float)
        if len(x) > 2 and np.std(x) > 0:
            sl, ic = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, sl * xs + ic, color='#222222', lw=1.0, ls='--')
            ax.text(0.03, 0.05, f'slope {sl:+.2f} pp per pp', transform=ax.transAxes,
                    fontsize=6.5)
        ax.axhline(0, color='#555555', lw=0.6, ls=':')
        ax.set_xlabel('reference CheckM2 completeness deficit\n(98 % − CheckM2 %, pp)')
        ax.set_ylabel('per-reference mean signed\ncompleteness error (pp)')
        ax.set_title(cfg.tool_short(tool)
                     + ('  (independent)' if tool != 'checkm2' else '  (circular)'),
                     fontsize=7.5)
        ax.legend(frameon=False, fontsize=6.5, loc='upper right')
    _save(fig, 'ws1_11_fig4_reference_incompleteness_mechanism',
          f'Per-reference mean signed completeness error against how far the reference '
          f'genome fell below the curation filter\'s 98 % CheckM2-completeness '
          f'criterion, over all 400 references (10 simulations each; filled = the 200 '
          f'would-have-FAILED references, open = their 200 matched controls, which sit '
          f'at deficit 0 by construction). '
          f'The benchmark defines completeness relative to the deposited assembly, so a '
          f'reference that is genuinely incomplete makes any biologically calibrated '
          f'estimator read low; a negative slope therefore indicates the truth '
          f'definition moving, not the estimator failing. For MAGICC the x axis is '
          f'independent evidence (MAGICC never sees a CheckM2 score); for CheckM2 the x '
          f'axis is its own estimate, so its slope is circular by construction and is '
          f'shown only for contrast. {denom}', captions)

    # --- Fig 5: threshold behaviour at the MIMAG-inspired 5 % boundary ------
    t5 = thr[(thr.criterion == 'contamination') & (thr.tau == 5.0)]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    width = 0.36
    xs = np.arange(len(order))
    for k, (arm, alpha) in enumerate((('H_pass', 0.35), ('H_fail', 0.8))):
        vals, lo, hi = [], [], []
        for tool in order:
            r = t5[(t5.tool == tool) & (t5.arm == arm)]
            v = float(r.iloc[0]['false_fail_rate']) if len(r) else np.nan
            vals.append(v)
            lo.append(v - float(r.iloc[0]['false_fail_ci_lo']) if len(r) else 0)
            hi.append(float(r.iloc[0]['false_fail_ci_hi']) - v if len(r) else 0)
        ax.bar(xs + (k - 0.5) * width, vals, width * 0.92,
               color=[cfg.tool_colour(t) for t in order], alpha=alpha,
               edgecolor='#222222', linewidth=0.6, hatch=ARM_HATCH[arm],
               yerr=[lo, hi], ecolor='#222222', capsize=2.5,
               label=('would pass' if arm == 'H_pass' else 'would FAIL'))
    ax.set_xticks(xs)
    ax.set_xticklabels([cfg.tool_short(t) for t in order])
    ax.set_ylabel('false-fail rate at 5 % contamination\nP(called ≥5 % | truly <5 %)')
    ax.legend(frameon=False, fontsize=6.5)
    npass = t5[t5.arm == 'H_pass']['n_true_pass'].max()
    nfail = t5[t5.arm == 'H_fail']['n_true_pass'].max()
    _save(fig, 'ws1_11_fig5_threshold_5pct_false_fail',
          f'False-fail rate at the MIMAG-inspired 5 % contamination boundary: the '
          f'fraction of truly clean simulations (<5 % contamination) that a tool calls '
          f'contaminated. Denominators are the truly clean simulations only — '
          f'n={int(npass)} in the would-have-passed arm and n={int(nfail)} in the '
          f'would-have-FAILED arm (hatched). Bars are 95 % cluster-bootstrap CIs over '
          f'reference genomes. False-fail rate must be read together with the '
          f'false-pass rate in ws1_11_thresholds.tsv: a tool can buy a low false-fail '
          f'rate by barely calling contamination at all. {denom}', captions)


if __name__ == '__main__':
    sys.exit(main())
