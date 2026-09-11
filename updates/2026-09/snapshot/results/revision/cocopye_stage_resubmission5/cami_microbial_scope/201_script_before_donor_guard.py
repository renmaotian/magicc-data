#!/usr/bin/env python3
"""
WS3.8 (R1-M3) — CAMI II analysis: accuracy, MIMAG-inspired thresholds, and the
head-to-head test of the WS2 close-contaminant limitation on third-party data.

WHAT THIS ANSWERS
-----------------
1. Procedural independence (what R1-M3 literally asks). Genome selection, read
   simulation, assembly and the gold standard were all produced by a third party; only
   the grouping of CAMI's own contigs into bins is ours. Accuracy is therefore measured
   outside our simulation assumptions.

2. THE HEADLINE. WS2 (Set F) established that MAGICC is near-blind to close
   contaminants: signed contamination bias -7.76 pp at species distance, with per-type
   OLS detection slopes at species of -0.014 (replaced) / 0.135 (single) / 0.178
   (redundant) versus ~1.0 at phylum. CAMI II strain_madness is built from many closely
   related strains by an independent group. This script tests whether that limitation
   reproduces, and reports the numbers side by side.

CONVENTIONS (binding)
---------------------
* R2 = coefficient of determination, 1 - SS_res/SS_tot (protocol 4.4d), NEVER squared
  Pearson. It is OMITTED (NaN, with the reason recorded) wherever the true value has
  (near-)zero variance -- which is the case for contamination on the pure gold-standard
  bins, where every truth value is exactly 0 (R1-m19).
* MIMAG-inspired: high >=90 % completeness AND <5 % contamination; medium >=50 % AND
  <10 %. Always labelled "MIMAG-inspired" because the strict definition additionally
  requires rRNA/tRNA criteria that cannot be evaluated from these assemblies.
* Primary analysis is IN-DOMAIN (contamination % <= completeness %, protocol 4.4a);
  out-of-domain bins are reported separately.
* MAGICC's 50 % completeness floor: bins below it are censored and counted explicitly;
  a bounded below-floor probe cohort is reported separately and never pooled in.
* All CIs are cluster bootstraps clustered by SOURCE GENOME (the dominant), 2,000
  iterations, seeded through fw.stable_hash (CRC-32) with PYTHONHASHSEED=0.
* Two-sided paired tests, BH correction, Hodges-Lehmann and Cliff's delta with CIs.
* Every table states its denominator (R1-M5).

OUTPUTS (results/revision/cami2/analysis/)
    cami2_headline.json
    cami2_accuracy_by_cohort.tsv
    cami2_mixed_by_distance.tsv            <- the head-to-head with Set F
    cami2_detection_slopes.tsv
    cami2_setF_comparison.tsv
    cami2_mimag.tsv
    cami2_paired_tests.tsv
    cami2_censoring.tsv
    cami2_gold_by_completeness_decile.tsv
    cami2_domain_restriction.tsv
    cami2_long_predictions.tsv

Usage:
    PYTHONHASHSEED=0 python scripts/201_ws38_cami2_analysis.py
"""

import argparse
import json
import signal
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

signal.signal(signal.SIGHUP, signal.SIG_IGN)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

RES_DIR = PROJECT_DIR / 'results' / 'revision' / 'cami2'
TRUTH_DIR = RES_DIR / 'truth'
PRED_DIR = RES_DIR / 'predictions'
PROV_DIR = RES_DIR / 'provenance'
OUT_DIR = RES_DIR / 'analysis'

N_BOOT = 2000
BASE_SEED = 38300
FLOOR = 50.0
RANKS = ['species', 'genus', 'family', 'order', 'class', 'phylum']
TOOL_ORDER = ['MAGICC_V5', 'CheckM2', 'CoCoPyE', 'DeepCheck']

DENOM = ('completeness = retained dominant bp / FULL reference length of the dominant '
         'source genome x 100; contamination = total contaminant bp / the SAME '
         'denominator x 100 (MAGICC convention, magicc/contamination.py). Truth from '
         "CAMI II's own gsa_mapping.tsv contig->source-genome assignment.")

# WS2 Set F reference numbers (results/revision/set_F/), for the head-to-head
SETF_CONT_BIAS = {'species': -7.76, 'genus': -7.73, 'family': -5.98,
                  'order': -3.60, 'class': -1.89, 'phylum': -0.90}
SETF_SPECIES_SLOPES = {'replaced': -0.014, 'single': 0.135, 'redundant': 0.178}
SETF_PHYLUM_SLOPE_RANGE = (0.94, 1.08)


# ------------------------------------------------------------------ helpers
def r2_cod(t, p, min_sd=1e-8):
    """Coefficient of determination. NaN where the TRUTH has (near-)zero variance,
    because R2 is undefined there and would otherwise print as a large negative
    number (R1-m19)."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    if t.size < 3 or np.std(t) < min_sd:
        return np.nan
    return float(fw.r2_coefficient_of_determination(t, p))


def ols_slope(t, p):
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    if t.size < 3 or np.std(t) < 1e-8:
        return np.nan
    return float(np.polyfit(t, p, 1)[0])


def boot(clusters, seed_key, n_iter=N_BOOT):
    return fw.Bootstrapper(clusters=clusters, n_iter=n_iter,
                           seed=BASE_SEED + fw.stable_hash(seed_key))


def core_stats(true, pred):
    e = pred - true
    return {'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e)),
            'rmse': float(np.sqrt(np.mean(e ** 2))), 'r2': r2_cod(true, pred),
            'slope': ols_slope(true, pred)}


def cohort_row(df, tool, metric, seed_key, cluster_col='cluster'):
    """MAE / bias / R2 / slope with cluster-bootstrap CIs for one tool+metric."""
    t = df[f'true_{metric}'].to_numpy(float)
    p = df[f'pred_{metric}'].to_numpy(float)
    cl = df[cluster_col].to_numpy()
    bs = boot(cl, seed_key)

    def fn(idx):
        return core_stats(t[idx], p[idx])

    ci = bs.ci_multi(fn)
    out = {'tool': tool, 'metric': metric, 'n': int(len(df)),
           'n_clusters': int(bs.n_clusters)}
    for k in ('mae', 'bias', 'rmse', 'r2', 'slope'):
        out[k] = ci[k]['estimate']
        out[f'{k}_lo'] = ci[k]['ci_lo']
        out[f'{k}_hi'] = ci[k]['ci_hi']
    if np.isnan(out['r2']):
        out['r2_omitted_reason'] = 'true value has (near-)zero variance (R1-m19)'
    else:
        out['r2_omitted_reason'] = ''
    return out


def paired_test(df, tool_a, tool_b, metric, cluster_col='cluster'):
    """Paired |error| comparison between two tools on identical bins."""
    a = df[df['tool'] == tool_a].set_index('bin_id')
    b = df[df['tool'] == tool_b].set_index('bin_id')
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 8:
        return None
    ea = (a.loc[common, f'pred_{metric}'] - a.loc[common, f'true_{metric}']).abs()
    eb = (b.loc[common, f'pred_{metric}'] - b.loc[common, f'true_{metric}']).abs()
    d = (ea - eb).to_numpy(float)                       # <0 means tool_a better
    try:
        w = stats.wilcoxon(ea, eb, alternative='two-sided', zero_method='wilcox')
        pval = float(w.pvalue)
    except ValueError:
        pval = np.nan
    hl = float(fw.hodges_lehmann_paired(d))
    delta = float(fw.cliffs_delta(ea.to_numpy(float), eb.to_numpy(float)))
    cl = a.loc[common, cluster_col].to_numpy()
    bs = boot(cl, f'paired|{tool_a}|{tool_b}|{metric}')

    def fn(idx):
        return {'hl': float(fw.hodges_lehmann_paired(d[idx])),
                'delta': float(fw.cliffs_delta(ea.to_numpy(float)[idx],
                                               eb.to_numpy(float)[idx]))}

    ci = bs.ci_multi(fn)
    return {'tool_a': tool_a, 'tool_b': tool_b, 'metric': metric, 'n_pairs': len(common),
            'mean_abs_err_a': float(ea.mean()), 'mean_abs_err_b': float(eb.mean()),
            'hodges_lehmann': hl, 'hl_lo': ci['hl']['ci_lo'], 'hl_hi': ci['hl']['ci_hi'],
            'cliffs_delta': delta, 'delta_lo': ci['delta']['ci_lo'],
            'delta_hi': ci['delta']['ci_hi'], 'wilcoxon_p': pval,
            'favours': tool_a if hl < 0 else tool_b}


# ------------------------------------------------------------------ loading
def load_long():
    """One tidy frame: bin_id x tool x truth x prediction, for every dataset/binset."""
    frames = []
    for ds in ('marine', 'strain_madness'):
        for bs_name in ('gold', 'mixed'):
            tp = TRUTH_DIR / f'{ds}_{bs_name}_truth.tsv'
            pp = PRED_DIR / f'{ds}_{bs_name}_all_tools.tsv'
            if not tp.exists() or not pp.exists():
                continue
            truth = pd.read_csv(tp, sep='\t')
            pred = pd.read_csv(pp, sep='\t')
            # both tables carry dataset/binset; keep the truth table's copy so the
            # merge cannot produce dataset_x / dataset_y
            pred = pred.drop(columns=['dataset', 'binset'], errors='ignore')
            truth = truth.rename(columns={'completeness_pct': 'true_completeness',
                                          'contamination_pct': 'true_contamination'})
            if bs_name == 'gold':
                truth['cluster'] = truth['genome']
                truth['distance_rank'] = 'none_pure_bin'
                truth['dominant'] = truth['genome']
                truth['contaminant'] = ''
            else:
                truth['cluster'] = truth['dominant']
            keep = ['bin_id', 'dataset', 'sample', 'cluster', 'dominant', 'contaminant',
                    'distance_rank', 'true_completeness', 'true_contamination',
                    'ref_len']
            for c in ('target_completeness', 'target_contamination',
                      'cami_novelty_category'):
                if c in truth.columns:
                    keep.append(c)
            m = pred.merge(truth[keep], on='bin_id', how='inner')
            m['binset'] = bs_name
            frames.append(m)
    if not frames:
        raise SystemExit('no predictions found -- run 198/199/200 first')
    long = pd.concat(frames, ignore_index=True)
    long['tool_scored'] = np.isfinite(long['pred_completeness']) & np.isfinite(long['pred_contamination'])
    long['in_domain'] = long['true_contamination'] <= long['true_completeness']
    long['scoreable'] = long['true_completeness'] >= FLOOR
    long['comp_err'] = long['pred_completeness'] - long['true_completeness']
    long['cont_err'] = long['pred_contamination'] - long['true_contamination']

    # MAGICC is cheap enough to run on EVERY bin, while CheckM2/CoCoPyE/DeepCheck run
    # only on the deliberately subsampled competitor cohort. Comparing tools on
    # different bin sets would be meaningless, so mark the subset every available tool
    # scored; the primary tool-vs-tool tables use only that subset, and MAGICC's
    # full-coverage numbers are reported separately as their own cohort.
    flags = []
    for (ds, bset), sub in long.groupby(['dataset', 'binset']):
        assert set(sub['tool']) == set(TOOL_ORDER)
        n_tools = len(TOOL_ORDER)
        cnt = sub[sub.tool_scored].groupby('bin_id')['tool'].nunique()
        common = set(cnt[cnt == n_tools].index)
        flags.append(pd.Series(sub['bin_id'].isin(common).values, index=sub.index))
    long['all_tools_scored'] = pd.concat(flags).reindex(long.index).fillna(False)
    return long


def attach_leakage(long):
    """Mark bins whose DOMINANT source genome is leaked into train/val/9-mer sel."""
    leak = {}
    for ds in ('marine', 'strain_madness'):
        p = PROV_DIR / f'{ds}_source_genome_audit.tsv'
        if p.exists():
            a = pd.read_csv(p, sep='\t')
            for g, v in zip(a['genome'], a['leaked']):
                leak[(ds, g)] = bool(v)
    long['dominant_leaked'] = [leak.get((d, g), False)
                               for d, g in zip(long['dataset'], long['dominant'])]
    # a mixed bin is leakage-free only if BOTH members are clean
    cont_leak = [leak.get((d, c), False) if isinstance(c, str) and c else False
                 for d, c in zip(long['dataset'], long['contaminant'])]
    long['contaminant_leaked'] = cont_leak
    long['leakage_free'] = ~(long['dominant_leaked'] | long['contaminant_leaked'])
    return long


def microbial_source_ids(dataset):
    setup = 'simulation_short_read' if dataset == 'marine' else 'short_read'
    path = PROJECT_DIR / 'data/real_data/cami2' / dataset / 'setup' / setup / 'metadata.tsv'
    meta = pd.read_csv(path, sep='\t')
    assert meta.genome_ID.is_unique
    microbial = set(meta.loc[meta.genome_ID.str.startswith('Otu'), 'genome_ID']) if dataset == 'marine' else set(meta.genome_ID)
    assert len(microbial) == (777 if dataset == 'marine' else 408)
    if dataset == 'marine': assert meta.genome_ID.str.startswith(('Otu', 'RNODE_')).all()
    return microbial


def apply_microbial_scope(long):
    valid = {dataset: microbial_source_ids(dataset) for dataset in ('marine', 'strain_madness')}
    mask = [dom in valid[dataset] and (not isinstance(donor, str) or not donor or donor in valid[dataset])
            for dataset, dom, donor in zip(long.dataset, long.dominant, long.contaminant)]
    return long.loc[mask].copy()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long = apply_microbial_scope(attach_leakage(load_long()))
    coverage = []
    for (ds, bset), sub in long.groupby(['dataset', 'binset']):
        for cohort, mask in [('all_prediction_rows_microbial_scope', np.ones(len(sub), dtype=bool)), ('primary_in_domain_scoreable_training_overlap_screened', sub.in_domain & sub.scoreable & sub.leakage_free)]:
            z = sub.loc[mask]
            old_counts = z.groupby('bin_id').tool.nunique()
            old_common = set(old_counts[old_counts == len(TOOL_ORDER)].index)
            for tool, zz in z.groupby('tool'):
                coverage.append(dict(dataset=ds, binset=bset, cohort=cohort, tool=tool, n_prediction_rows=len(zz), n_scored=int(zz.tool_scored.sum()), n_unscored=int((~zz.tool_scored).sum()), original_common_n=len(old_common), corrected_common_n=int(z.loc[z.all_tools_scored, 'bin_id'].nunique())))
    pd.DataFrame(coverage).to_csv(OUT_DIR / 'cami2_tool_scoring_coverage.tsv', sep='\t', index=False)
    long.to_csv(OUT_DIR / 'cami2_long_predictions.tsv', sep='\t', index=False)
    long = long[long.tool_scored].copy()
    print(f'== WS3.8 analysis: {len(long)} tool-bin rows, tools='
          f'{sorted(long["tool"].unique())} ==', flush=True)

    # -------------------------------------------------- censoring & domain
    cens = []
    for (ds, bset), sub in long.groupby(['dataset', 'binset']):
        u = sub.drop_duplicates('bin_id')
        cens.append({
            'dataset': ds, 'binset': bset, 'n_bins_scored': int(len(u)),
            'n_below_50pct_floor': int((~u['scoreable']).sum()),
            'pct_below_floor': round(100.0 * (~u['scoreable']).mean(), 2),
            'n_out_of_domain': int((~u['in_domain']).sum()),
            'n_dominant_leaked': int(u['dominant_leaked'].sum()),
            'n_leakage_free': int(u['leakage_free'].sum()),
        })
    # full truth-table censoring (includes bins never written to FASTA)
    for ds in ('marine', 'strain_madness'):
        p = TRUTH_DIR / f'{ds}_gold_truth.tsv'
        if p.exists():
            g = pd.read_csv(p, sep='\t')
            g = g[g.genome.isin(microbial_source_ids(ds))].copy()
            cens.append({
                'dataset': ds, 'binset': 'gold_ALL_TRUTH_ROWS',
                'n_bins_scored': int(len(g)),
                'n_below_50pct_floor': int((g['completeness_pct'] < FLOOR).sum()),
                'pct_below_floor': round(100.0 * (g['completeness_pct'] < FLOOR).mean(), 2),
                'n_out_of_domain': 0, 'n_dominant_leaked': np.nan,
                'n_leakage_free': np.nan})
    censd = pd.DataFrame(cens)
    censd['denominator'] = DENOM
    censd.to_csv(OUT_DIR / 'cami2_censoring.tsv', sep='\t', index=False)
    print(censd.to_string(index=False), flush=True)

    # -------------------------------------------------- accuracy by cohort
    rows = []
    cohorts = []
    for ds in sorted(long['dataset'].unique()):
        for bset in sorted(long['binset'].unique()):
            base = long[(long['dataset'] == ds) & (long['binset'] == bset)]
            if not len(base):
                continue
            ok = base['in_domain'] & base['scoreable']
            cohorts += [
                # PRIMARY: identical bins for every tool, leakage-free, in-domain
                (ds, bset, 'primary_in_domain_scoreable_LEAKAGE_FREE',
                 base[ok & base['leakage_free'] & base['all_tools_scored']]),
                (ds, bset, 'all_in_domain_scoreable_common_subset',
                 base[ok & base['all_tools_scored']]),
                (ds, bset, 'leaked_only',
                 base[ok & ~base['leakage_free'] & base['all_tools_scored']]),
                # MAGICC's full coverage (all bins, not only the competitor subsample);
                # NOT comparable tool-to-tool, reported for coverage only
                (ds, bset, 'MAGICC_full_coverage_not_tool_comparable',
                 base[ok & base['leakage_free'] & (base['tool'] == 'MAGICC_V5')]),
                (ds, bset, 'below_floor_probe_NOT_POOLED', base[~base['scoreable']]),
                (ds, bset, 'out_of_domain', base[~base['in_domain']]),
            ]
    for ds, bset, cname, sub in cohorts:
        if len(sub) == 0:
            continue
        for tool in TOOL_ORDER:
            ts = sub[sub['tool'] == tool]
            if len(ts) < 8:
                continue
            for metric in ('completeness', 'contamination'):
                r = cohort_row(ts, tool, metric, f'{ds}|{bset}|{cname}|{tool}|{metric}')
                r.update({'dataset': ds, 'binset': bset, 'cohort': cname})
                rows.append(r)
    acc = pd.DataFrame(rows)
    if len(acc):
        acc['denominator'] = DENOM
        acc = acc[['dataset', 'binset', 'cohort', 'tool', 'metric', 'n', 'n_clusters',
                   'mae', 'mae_lo', 'mae_hi', 'bias', 'bias_lo', 'bias_hi',
                   'rmse', 'r2', 'r2_lo', 'r2_hi', 'r2_omitted_reason',
                   'slope', 'slope_lo', 'slope_hi', 'denominator']]
    acc.to_csv(OUT_DIR / 'cami2_accuracy_by_cohort.tsv', sep='\t', index=False)
    print(f'  accuracy rows: {len(acc)}', flush=True)

    # -------------------------------------------------- THE HEADLINE
    # mixed bins, by taxonomic distance, leakage-free & in-domain & scoreable
    mixed = long[(long['binset'] == 'mixed') & long['in_domain'] & long['scoreable']
                 & long['leakage_free'] & long['all_tools_scored']]
    drows, srows = [], []
    for ds in sorted(mixed['dataset'].unique()):
        for tool in TOOL_ORDER:
            for rank in RANKS:
                sub = mixed[(mixed['dataset'] == ds) & (mixed['tool'] == tool)
                            & (mixed['distance_rank'] == rank)]
                if len(sub) < 8:
                    continue
                t = sub['true_contamination'].to_numpy(float)
                p = sub['pred_contamination'].to_numpy(float)
                tc = sub['true_completeness'].to_numpy(float)
                pc = sub['pred_completeness'].to_numpy(float)
                bs = boot(sub['cluster'].to_numpy(), f'dist|{ds}|{tool}|{rank}')

                def fn(idx, t=t, p=p, tc=tc, pc=pc):
                    return {'cont_bias': float(np.mean(p[idx] - t[idx])),
                            'cont_mae': float(np.mean(np.abs(p[idx] - t[idx]))),
                            'comp_bias': float(np.mean(pc[idx] - tc[idx])),
                            'comp_mae': float(np.mean(np.abs(pc[idx] - tc[idx]))),
                            'slope': ols_slope(t[idx], p[idx])}

                ci = bs.ci_multi(fn)
                row = {'dataset': ds, 'tool': tool, 'distance_rank': rank,
                       'n': int(len(sub)), 'n_clusters': int(bs.n_clusters)}
                for k in ('cont_bias', 'cont_mae', 'comp_bias', 'comp_mae', 'slope'):
                    row[k] = ci[k]['estimate']
                    row[f'{k}_lo'] = ci[k]['ci_lo']
                    row[f'{k}_hi'] = ci[k]['ci_hi']
                row['cont_r2'] = r2_cod(t, p)
                drows.append(row)
                if tool == 'MAGICC_V5':
                    srows.append({'dataset': ds, 'distance_rank': rank,
                                  'detection_slope': row['slope'],
                                  'slope_lo': row['slope_lo'],
                                  'slope_hi': row['slope_hi'],
                                  'n': row['n']})
    dist = pd.DataFrame(drows)
    if len(dist):
        dist['distance_rank'] = pd.Categorical(dist['distance_rank'], RANKS, ordered=True)
        dist = dist.sort_values(['dataset', 'tool', 'distance_rank'])
        dist['denominator'] = DENOM
    dist.to_csv(OUT_DIR / 'cami2_mixed_by_distance.tsv', sep='\t', index=False)
    pd.DataFrame(srows).to_csv(OUT_DIR / 'cami2_detection_slopes.tsv',
                               sep='\t', index=False)

    # head-to-head against Set F
    cmp_rows = []
    for ds in sorted(mixed['dataset'].unique()):
        for rank in RANKS:
            r = dist[(dist['dataset'] == ds) & (dist['tool'] == 'MAGICC_V5')
                     & (dist['distance_rank'] == rank)]
            if not len(r):
                continue
            r = r.iloc[0]
            cmp_rows.append({
                'distance_rank': rank, 'dataset': f'CAMI_II_{ds}',
                'cami2_cont_bias_pp': round(float(r['cont_bias']), 3),
                'cami2_bias_ci': f"[{r['cont_bias_lo']:.2f}, {r['cont_bias_hi']:.2f}]",
                'cami2_detection_slope': round(float(r['slope']), 3),
                'cami2_slope_ci': f"[{r['slope_lo']:.3f}, {r['slope_hi']:.3f}]",
                'cami2_n': int(r['n']), 'cami2_n_clusters': int(r['n_clusters']),
                'ws2_setF_cont_bias_pp': SETF_CONT_BIAS[rank],
                'delta_cami2_minus_setF': round(float(r['cont_bias'])
                                                - SETF_CONT_BIAS[rank], 3),
            })
    cmpdf = pd.DataFrame(cmp_rows)
    if len(cmpdf):
        cmpdf['setF_species_slopes_per_type'] = json.dumps(SETF_SPECIES_SLOPES)
        cmpdf['denominator'] = DENOM
    cmpdf.to_csv(OUT_DIR / 'cami2_setF_comparison.tsv', sep='\t', index=False)
    if len(cmpdf):
        print('\n== HEAD-TO-HEAD vs WS2 Set F (MAGICC V5 contamination bias, pp) ==',
              flush=True)
        print(cmpdf[['dataset', 'distance_rank', 'cami2_cont_bias_pp',
                     'ws2_setF_cont_bias_pp', 'delta_cami2_minus_setF',
                     'cami2_detection_slope', 'cami2_n']].to_string(index=False),
              flush=True)

    # -------------------------------------------------- sensitivity: is the distance
    # gradient an artefact of how much of the dominant we subsampled? At
    # target_completeness = 90 the dominant is left almost intact; at 60 it is heavily
    # subsampled. If the gradient is the same at both, the grouping choice is not
    # driving it. Free from data already computed.
    srows2 = []
    if 'target_completeness' in mixed.columns:
        for ds in sorted(mixed['dataset'].unique()):
            for ct in sorted(mixed['target_completeness'].dropna().unique()):
                for rank in RANKS:
                    sub = mixed[(mixed['dataset'] == ds)
                                & (mixed['tool'] == 'MAGICC_V5')
                                & (mixed['target_completeness'] == ct)
                                & (mixed['distance_rank'] == rank)]
                    if len(sub) < 8:
                        continue
                    t = sub['true_contamination'].to_numpy(float)
                    p = sub['pred_contamination'].to_numpy(float)
                    srows2.append({
                        'dataset': ds, 'target_completeness': float(ct),
                        'distance_rank': rank, 'n': int(len(sub)),
                        'cont_bias': float(np.mean(p - t)),
                        'detection_slope': ols_slope(t, p),
                        'note': 'higher target completeness = less of the dominant '
                                'subsampled away; a stable gradient across rows shows '
                                'the distance effect is not an artefact of subsampling'})
    pd.DataFrame(srows2).to_csv(
        OUT_DIR / 'cami2_sensitivity_subsampling.tsv', sep='\t', index=False)

    # -------------------------------------------------- MIMAG-inspired
    mrows = []
    for (ds, bset), sub in long[long['in_domain'] & long['scoreable']
                                & long['leakage_free']
                                & long['all_tools_scored']].groupby(['dataset', 'binset']):
        for tool in TOOL_ORDER:
            ts = sub[sub['tool'] == tool]
            if len(ts) < 8:
                continue
            tc, tk = ts['true_completeness'], ts['true_contamination']
            pc, pk = ts['pred_completeness'], ts['pred_contamination']
            true_hq = (tc >= 90) & (tk < 5)
            pred_hq = (pc >= 90) & (pk < 5)
            true_mq = (tc >= 50) & (tk < 10)
            pred_mq = (pc >= 50) & (pk < 10)
            true_dirty = tk >= 5
            row = {'dataset': ds, 'binset': bset, 'tool': tool, 'n': int(len(ts)),
                   'n_true_HQ': int(true_hq.sum()), 'n_pred_HQ': int(pred_hq.sum()),
                   'HQ_agreement': float((true_hq == pred_hq).mean()),
                   'HQ_sensitivity': float(pred_hq[true_hq].mean())
                   if true_hq.sum() else np.nan,
                   'HQ_precision': float(true_hq[pred_hq].mean())
                   if pred_hq.sum() else np.nan,
                   'MQ_agreement': float((true_mq == pred_mq).mean()),
                   'n_true_contaminated_ge5pct': int(true_dirty.sum()),
                   'false_clean_rate_at_5pct':
                       float((pk[true_dirty] < 5).mean()) if true_dirty.sum() else np.nan,
                   'false_fail_rate_at_5pct':
                       float((pk[~true_dirty] >= 5).mean())
                       if (~true_dirty).sum() else np.nan,
                   'threshold_note': 'MIMAG-inspired: HQ >=90% comp AND <5% cont; '
                                     'MQ >=50% AND <10%. rRNA/tRNA criteria of the '
                                     'strict definition are not evaluable here.'}
            mrows.append(row)
    mim = pd.DataFrame(mrows)
    if len(mim):
        mim['denominator'] = DENOM
    mim.to_csv(OUT_DIR / 'cami2_mimag.tsv', sep='\t', index=False)

    # -------------------------------------------------- paired tests
    prows = []
    for (ds, bset), sub in long[long['in_domain'] & long['scoreable']
                                & long['leakage_free']
                                & long['all_tools_scored']].groupby(['dataset', 'binset']):
        tools = [t for t in TOOL_ORDER if t in set(sub['tool'])]
        for other in tools:
            if other == 'MAGICC_V5':
                continue
            for metric in ('completeness', 'contamination'):
                r = paired_test(sub, 'MAGICC_V5', other, metric)
                if r:
                    r.update({'dataset': ds, 'binset': bset})
                    prows.append(r)
        # the key stratified test: species/genus vs order/class/phylum
        if bset == 'mixed':
            for other in tools:
                if other == 'MAGICC_V5':
                    continue
                for grp, ranks in (('close_species_genus', ['species', 'genus']),
                                   ('distant_order_class_phylum',
                                    ['order', 'class', 'phylum'])):
                    ss = sub[sub['distance_rank'].isin(ranks)]
                    r = paired_test(ss, 'MAGICC_V5', other, 'contamination')
                    if r:
                        r.update({'dataset': ds, 'binset': f'mixed::{grp}'})
                        prows.append(r)
    pt = pd.DataFrame(prows)
    if len(pt):
        pt['p_bh'] = fw.bh_correct(pt['wilcoxon_p'].fillna(1.0).tolist())
        pt['denominator'] = DENOM
    pt.to_csv(OUT_DIR / 'cami2_paired_tests.tsv', sep='\t', index=False)

    # -------------------------------------------------- gold completeness gradient
    grows = []
    gold = long[(long['binset'] == 'gold') & long['scoreable'] & long['leakage_free']
                & long['all_tools_scored']]
    if len(gold):
        gold = gold.copy()
        gold['decile'] = pd.cut(gold['true_completeness'],
                                [50, 60, 70, 80, 90, 95, 100.01],
                                labels=['50-60', '60-70', '70-80', '80-90',
                                        '90-95', '95-100'], include_lowest=True)
        for (ds, dec, tool), sub in gold.groupby(['dataset', 'decile', 'tool'],
                                                 observed=True):
            if len(sub) < 5:
                continue
            grows.append({
                'dataset': ds, 'completeness_band': str(dec), 'tool': tool,
                'n': int(len(sub)),
                'comp_mae': float(sub['comp_err'].abs().mean()),
                'comp_bias': float(sub['comp_err'].mean()),
                'cont_mae': float(sub['cont_err'].abs().mean()),
                'cont_bias': float(sub['cont_err'].mean()),
                'r2_note': 'contamination R2 omitted: truth is exactly 0 for every '
                           'pure gold-standard bin, so SS_tot = 0 (R1-m19)'})
    gd = pd.DataFrame(grows)
    if len(gd):
        gd['denominator'] = DENOM
    gd.to_csv(OUT_DIR / 'cami2_gold_by_completeness_decile.tsv', sep='\t', index=False)

    # -------------------------------------------------- domain restriction
    drrows = []
    for (ds, bset), sub in long.groupby(['dataset', 'binset']):
        for tool in TOOL_ORDER:
            for dom, label in ((True, 'in_domain'), (False, 'out_of_domain')):
                ts = sub[(sub['tool'] == tool) & (sub['in_domain'] == dom)
                         & sub['scoreable']]
                if len(ts) < 8:
                    continue
                drrows.append({
                    'dataset': ds, 'binset': bset, 'tool': tool, 'domain': label,
                    'n': int(len(ts)),
                    'comp_mae': float(ts['comp_err'].abs().mean()),
                    'cont_mae': float(ts['cont_err'].abs().mean()),
                    'comp_bias': float(ts['comp_err'].mean()),
                    'cont_bias': float(ts['cont_err'].mean())})
    pd.DataFrame(drrows).to_csv(OUT_DIR / 'cami2_domain_restriction.tsv',
                                sep='\t', index=False)

    # -------------------------------------------------- headline JSON
    def pick(dsname, bset, cohort, tool, metric, field):
        r = acc[(acc['dataset'] == dsname) & (acc['binset'] == bset)
                & (acc['cohort'] == cohort) & (acc['tool'] == tool)
                & (acc['metric'] == metric)]
        return None if not len(r) else float(r.iloc[0][field])

    headline = {
        'workstream': 'WS3.8 CAMI II external benchmark (R1-M3)',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'denominator': DENOM,
        'r2_convention': 'coefficient of determination (1 - SS_res/SS_tot); omitted '
                         'where the true value has (near-)zero variance (R1-m19)',
        'mimag_note': 'MIMAG-inspired thresholds throughout',
        'bootstrap': {'n_iter': args.n_boot, 'clustered_by': 'dominant source genome',
                      'seeding': 'fw.stable_hash CRC-32 + PYTHONHASHSEED=0'},
        'tools': sorted(long['tool'].unique().tolist()),
        'n_tool_bin_rows': int(len(long)),
        'censoring': censd.to_dict('records'),
        'setF_reference': {'contamination_bias_pp': SETF_CONT_BIAS,
                           'species_detection_slopes_per_type': SETF_SPECIES_SLOPES,
                           'phylum_slope_range': list(SETF_PHYLUM_SLOPE_RANGE)},
    }
    if len(cmpdf):
        headline['setF_head_to_head'] = cmpdf.drop(
            columns=['setF_species_slopes_per_type', 'denominator'],
            errors='ignore').to_dict('records')
    (OUT_DIR / 'cami2_headline.json').write_text(json.dumps(headline, indent=2,
                                                           default=str))
    print(f'\n  wrote {OUT_DIR}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
