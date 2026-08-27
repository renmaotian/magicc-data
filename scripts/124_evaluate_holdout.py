#!/usr/bin/env python3
"""
WS1.6 / WS1.9 / WS11.G step 4b (+ WS1.8) - Evaluate leave-TAXON-out generalization.

Level is selected by MAGICC_HOLDOUT_LEVEL (phylum | family | genus). Runs BOTH models on
the SAME evaluation genomes:
  * holdout model  models/magicc_holdout_{level}.onnx  (never saw the panel lineages)
  * production V5  models/magicc_v5.onnx               (trained on all lineages)

THE HOLDOUT MODELS ARE VALIDATION ARTIFACTS ONLY. The released MAGICC model
remains trained on all data.

Each model normalizes the shared RAW k-mer counts with ITS OWN normalization
parameters (holdout: data/holdout/holdout_normalization_params.json;
V5: data/features/normalization_params.json), which is what each model was
trained against. The difference between the two therefore isolates exactly one
thing: the value of having seen the lineage during training.

Metrics per evaluation group, for completeness and contamination:
  MAE, RMSE, R^2, mean signed error (bias, predicted - true), and cluster
  bootstrap 95% CIs resampling REFERENCE GENOMES (not samples), because each
  reference contributes multiple simulations (protocol WS5.4 / R1-m16).

Extra analyses
  * split-stratified breakdown for any group whose dominants could not all come
    from the test split (WS1.6 DPANN only; every WS1.9 group uses test refs).
  * per-taxon breakdown inside pooled groups and inside the control.
  * WS1.8 novelty-vs-error: per-group and per-reference error tables.
  * MIMAG-inspired threshold behaviour and error stratified by true-completeness
    and true-contamination band.
  * Optional: data/benchmarks/set_C_clean / set_D_clean if present.

Outputs (results/revision/holdout/ at phylum level,
         results/revision/holdout_family/ at family level,
         results/revision/holdout_genus/ at genus level)
  lineage_novelty_effect_did.tsv   PRIMARY: difference-in-differences
  head_to_head_by_group.tsv        raw MAE/bias/R2, both models
  stratified_error_by_band.tsv     completeness- and contamination-band strata
  mimag_threshold_by_group.tsv     HQ precision/recall, false-clean/false-dirty
  mimag_confusion_by_group.tsv
  sub_phylum_breakdown.tsv         per-taxon inside pooled groups
  per_sample_predictions.tsv.gz    every prediction from both models
  per_reference_errors.tsv         WS1.8 plotting input
  degradation_vs_in_distribution.tsv  the old raw-MAE rule, flagged as confounded
  metrics_full.json, ws1.*_consolidated.json
  clean_sets_evaluation.tsv        (if set_C_clean / set_D_clean exist)

Usage
  python scripts/124_evaluate_holdout.py
  python scripts/124_evaluate_holdout.py --holdout-onnx <path>
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_lib import config as C
from holdout_lib.normalization import FeatureNormalizer

N_BOOT = 2000
BOOT_SEED = 20260726

# Taxonomic novelty ordering for the WS1.8 curve (0 = in distribution)
NOVELTY = {g: (0 if g == C.CONTROL_GROUP else 1) for g in C.EVAL_DESIGN}


def load_model(onnx_path, norm_path):
    nrm = FeatureNormalizer.load(str(norm_path))
    assert nrm.finalized, f'{norm_path} not finalized'
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    return sess, nrm


def predict(sess, nrm, kmer_raw, asm_raw, batch=2000):
    out = np.zeros((len(kmer_raw), 2), dtype=np.float64)
    for s in range(0, len(kmer_raw), batch):
        e = min(s + batch, len(kmer_raw))
        k = nrm.normalize_kmer(kmer_raw[s:e].astype(np.float64)).astype(np.float32)
        a = nrm.normalize_assembly(asm_raw[s:e].astype(np.float64)).astype(np.float32)
        out[s:e] = sess.run(None, {'kmer_features': k, 'assembly_features': a})[0]
    return out


def cluster_boot_ci(err_abs, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% CI for MAE, resampling clusters (reference genomes) with replacement."""
    uc, inv = np.unique(clusters, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uc))]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        stats[b] = err_abs[idx].mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_cluster_boot(ae_a, ae_b, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired cluster bootstrap on the MAE difference (model A - model B).

    Both models are evaluated on the SAME samples, so the comparison must be
    paired; and each reference genome contributes many simulations, so the
    resampling unit is the REFERENCE GENOME, not the sample (protocol WS5.4 /
    R1-m16). Returns observed difference, 95% CI and a two-sided bootstrap
    p-value for H0: difference = 0.
    """
    uc, inv = np.unique(clusters, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uc))]
    obs = float(ae_a.mean() - ae_b.mean())
    if len(groups) < 3:
        return {'delta_mae': round(obs, 4), 'ci95': None, 'p_value': None,
                'n_clusters': len(groups)}
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        d[b] = ae_a[idx].mean() - ae_b[idx].mean()
    # two-sided bootstrap p-value, centred on the observed effect
    shifted = d - obs
    p = 2.0 * min(np.mean(shifted >= obs), np.mean(shifted <= obs))
    p = float(min(1.0, max(p, 1.0 / n_boot)))
    return {'delta_mae': round(obs, 4),
            'ci95': [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            'p_value': p, 'n_clusters': len(groups)}


def did_cluster_boot(ae_ho_g, ae_v5_g, cl_g, ae_ho_c, ae_v5_c, cl_c,
                     n_boot=N_BOOT, seed=BOOT_SEED):
    """Difference-in-differences: (MAE_HO - MAE_V5)_group - (MAE_HO - MAE_V5)_control.

    WHY THIS IS THE PRIMARY ESTIMATOR OF LINEAGE NOVELTY.
    Raw MAEs are not comparable across evaluation groups, because each held-out
    group is a single phylum while the in-distribution control is a
    sqrt-proportional mixture of 42 phyla dominated by rare, intrinsically hard,
    single-reference lineages (e.g. Margulisbacteria, Myxococcota). Comparing
    group MAE against control MAE therefore mixes lineage novelty with group
    composition, and in the flattering direction.

    The within-group difference Delta = MAE_holdout - MAE_V5 is immune to this,
    because both models score the SAME samples. Subtracting the control Delta
    removes the remaining offset that is common to both (the cost of training on
    a smaller genome pool: 19.64% fewer genomes at phylum level, 6.95% at family
    level), leaving the effect attributable specifically to the lineage being
    absent from training. It also cancels error that is shared by both models on
    the same genomes - notably the genome-SIZE-driven completeness bias
    established in WS3.10 - so a small-genome group's DiD is a novelty effect,
    not a size effect. Clusters are resampled independently in group and control.
    """
    def _grp(cl):
        uc, inv = np.unique(cl, return_inverse=True)
        return [np.where(inv == i)[0] for i in range(len(uc))]
    gg, gc = _grp(cl_g), _grp(cl_c)
    obs = float((ae_ho_g.mean() - ae_v5_g.mean()) -
                (ae_ho_c.mean() - ae_v5_c.mean()))
    if len(gg) < 3 or len(gc) < 3:
        return {'did': round(obs, 4), 'ci95': None, 'p_value': None}
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ig = np.concatenate([gg[i] for i in rng.integers(0, len(gg), len(gg))])
        ic = np.concatenate([gc[i] for i in rng.integers(0, len(gc), len(gc))])
        d[b] = ((ae_ho_g[ig].mean() - ae_v5_g[ig].mean()) -
                (ae_ho_c[ic].mean() - ae_v5_c[ic].mean()))
    shifted = d - obs
    p = 2.0 * min(np.mean(shifted >= obs), np.mean(shifted <= obs))
    return {'did': round(obs, 4),
            'ci95': [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            'p_value': float(min(1.0, max(p, 1.0 / n_boot)))}


def benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values, preserving input order."""
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = p[i] * m / (rank + 1)
        prev = min(prev, val)
        q[i] = min(1.0, prev)
    return q


def mimag_class(comp, cont):
    """MIMAG quality class from completeness/contamination percentages.

    MIMAG-INSPIRED (the strict definition also requires rRNA/tRNA criteria, which
    cannot be evaluated from these assemblies).

    high    : completeness >= 90 and contamination < 5
    medium  : completeness >= 50 and contamination < 10
    low     : everything else

    NOTE (2026-07-27): the HQ completeness bound was '> 90' here while the
    comp90_false_high / comp90_false_low counters below already used '>= 90'.
    Protocol section 4.4d standardizes on '>= 90', so this line was corrected.
    The two differ only on an exact tie at 90.000, which has probability ~0 for
    continuous true and predicted completeness, so no WS1.6 number changes.
    """
    out = np.full(len(comp), 'low', dtype=object)
    med = (comp >= 50) & (cont < 10)
    out[med] = 'medium'
    hq = (comp >= 90) & (cont < 5)
    out[hq] = 'high'
    return out


def metrics(true, pred, clusters, ci=True):
    err = pred - true
    ae = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    # PROJECT STANDARD (agreed 2026-07-26): every reported R2 is the
    # COEFFICIENT OF DETERMINATION, r2 = 1 - SS_res/SS_tot, which penalises
    # systematic bias. Squared Pearson correlation is reported alongside it under
    # the distinct name `r2_pearson` and must never be labelled R2, because it
    # ignores bias and scale and reads ~0.05 higher on biased predictions (this
    # discrepancy was a real defect in the submitted manuscript).
    with np.errstate(invalid='ignore'):
        rp = (float(np.corrcoef(true, pred)[0, 1]) ** 2
              if true.std() > 0 and pred.std() > 0 else float('nan'))
    m = {'n': int(len(true)), 'n_refs': int(len(np.unique(clusters))),
         'mae': float(ae.mean()), 'rmse': float(np.sqrt((err ** 2).mean())),
         'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
         'r2_pearson': rp,
         'bias': float(err.mean()), 'median_signed': float(np.median(err)),
         'p90_abs': float(np.percentile(ae, 90)),
         'true_mean': float(true.mean()), 'pred_mean': float(pred.mean())}
    if ci and len(np.unique(clusters)) >= 3:
        lo, hi = cluster_boot_ci(ae, clusters)
        m['mae_ci95'] = [round(lo, 4), round(hi, 4)]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout-onnx', default=str(C.HOLDOUT_ONNX))
    ap.add_argument('--holdout-norm', default=None,
                    help='override the holdout normalization params (dry-run/testing)')
    ap.add_argument('--eval-dir', default=str(C.EVAL_DIR))
    ap.add_argument('--out-dir', default=str(C.RESULTS_DIR))
    a = ap.parse_args()
    C.RESULTS_DIR = Path(a.out_dir)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print(f'{C.WS.upper()} / WS1.8 - leave-{C.PANEL_LEVEL}-out generalization evaluation')
    if C.PANEL_LEVEL == 'family':
        print('Every held-out family\'s PARENT PHYLUM remained in training for both '
              'models.')
    elif C.PANEL_LEVEL == 'genus':
        print('Every held-out genus\'s PARENT FAMILY and PARENT PHYLUM remained in '
              'training for both models.')
        print('Degradation is NOT assumed to be monotone across ranks - that is '
              'exactly what is being measured.')
    print('=' * 78)
    models = {}
    ho_norm = Path(a.holdout_norm) if a.holdout_norm else C.HOLDOUT_NORM_PARAMS
    print(f'holdout : {a.holdout_onnx}\n          {ho_norm}')
    models['holdout'] = load_model(a.holdout_onnx, ho_norm)
    print(f'V5 prod : {C.V5_ONNX}\n          {C.V5_NORM_PARAMS}')
    models['V5'] = load_model(C.V5_ONNX, C.V5_NORM_PARAMS)

    eval_dir = Path(a.eval_dir)
    manifest = json.loads((eval_dir / 'manifest.json').read_text())

    # ------------------------------------------------------------------
    # Provenance audit: prove the holdout model never saw the panel lineages
    # and never saw the exact evaluation dominants.
    # ------------------------------------------------------------------
    audit = {}
    try:
        with h5py.File(C.HOLDOUT_H5, 'r') as f:
            tr_acc, tr_phy = set(), set()
            for split in ['train', 'val']:
                n = f[split]['labels'].shape[0]
                for s in range(0, n, 100_000):
                    e = min(s + 100_000, n)
                    md_ = f[split]['metadata'][s:e]
                    tr_acc |= set(np.char.decode(md_['dominant_accession']))
                    tr_phy |= set(np.char.decode(md_['dominant_phylum']))
        panel_in_train = sorted(tr_phy & set(C.PANEL_PHYLA))
        # METADATA_DTYPE has no dominant_family field, so at family level the
        # leak test runs on ACCESSIONS (strictly stronger than a name test).
        panel_acc = set()
        if C.PANEL_LEVEL != 'phylum':
            for tsv in [C.TRAIN_TSV, C.VAL_TSV, C.TEST_TSV]:
                _d = pd.read_csv(tsv, sep='\t',
                                 usecols=['ncbi_accession', 'gtdb_taxonomy'])
                _d['_t'] = _d.gtdb_taxonomy.map(
                    lambda t: C.rank_from_taxonomy(t, C.TAXON_COL))
                panel_acc |= set(_d.loc[_d._t.isin(C.PANEL_TAXA), 'ncbi_accession'])
        panel_taxon_leak = sorted(tr_acc & panel_acc)
        audit['panel_level'] = C.PANEL_LEVEL
        audit['n_panel_taxa'] = len(C.PANEL_TAXA)
        audit['n_panel_taxon_genomes'] = len(panel_acc)
        audit['panel_taxon_genomes_in_holdout_train_val'] = len(panel_taxon_leak)
        audit['holdout_train_val_dominant_phyla'] = len(tr_phy)
        audit['panel_phyla_in_holdout_train_val'] = panel_in_train
        audit['holdout_train_val_dominant_accessions'] = len(tr_acc)
        ev_acc = set()
        for g in C.EVAL_DESIGN:
            mp2 = eval_dir / g / 'metadata.tsv'
            if mp2.exists():
                ev_acc |= set(pd.read_csv(mp2, sep='\t').dominant_accession)
        # accessions in the split TSVs use ncbi_accession, same as metadata
        overlap = sorted(ev_acc & tr_acc)
        audit['eval_dominant_accessions'] = len(ev_acc)
        audit['eval_dominants_also_in_holdout_training'] = len(overlap)
        audit['overlap_examples'] = overlap[:10]
        print('\nPROVENANCE AUDIT')
        print(f'  holdout train+val dominant phyla: {len(tr_phy)}; '
              f'panel phyla present: {panel_in_train or "NONE"}')
        print(f'  holdout train+val distinct dominant accessions: {len(tr_acc):,}')
        print(f'  evaluation dominant accessions: {len(ev_acc):,}; '
              f'also used in holdout training: {len(overlap)}')
        print(f'  panel-{C.PANEL_LEVEL} genomes appearing as holdout-training '
              f'dominants: {len(panel_taxon_leak)}')
        assert not panel_in_train, f'PANEL LEAK in holdout training data: {panel_in_train}'
        assert not panel_taxon_leak, (
            f'PANEL-TAXON LEAK in holdout training data: {panel_taxon_leak[:10]}')
    except FileNotFoundError:
        print('\n(holdout HDF5 not found - provenance audit skipped)')
    groups = [g for g in C.EVAL_DESIGN if (eval_dir / g / 'features.h5').exists()]
    print(f'\nevaluation groups found: {groups}')

    allrows, group_metrics = [], {}
    for g in groups:
        md = pd.read_csv(eval_dir / g / 'metadata.tsv', sep='\t')
        with h5py.File(eval_dir / g / 'features.h5', 'r') as f:
            kr = f['kmer_counts_raw'][:]
            ar = f['summary_features_raw'][:]
        assert len(md) == len(kr), f'{g}: metadata/feature length mismatch'
        true_c = md.true_completeness.values.astype(np.float64)
        true_x = md.true_contamination.values.astype(np.float64)
        clusters = md.dominant_accession.values

        gm = {}
        for name, (sess, nrm) in models.items():
            p = predict(sess, nrm, kr, ar)
            md[f'{name}_completeness'] = p[:, 0]
            md[f'{name}_contamination'] = p[:, 1]
            gm[name] = {
                'completeness': metrics(true_c, p[:, 0], clusters),
                'contamination': metrics(true_x, p[:, 1], clusters)}
        # paired holdout-vs-V5 significance, clustered by reference genome
        gm['paired'] = {
            'completeness': paired_cluster_boot(
                np.abs(md.holdout_completeness.values - true_c),
                np.abs(md.V5_completeness.values - true_c), clusters),
            'contamination': paired_cluster_boot(
                np.abs(md.holdout_contamination.values - true_x),
                np.abs(md.V5_contamination.values - true_x), clusters)}
        group_metrics[g] = gm
        md['group'] = g
        md['novelty'] = NOVELTY.get(g, 1)
        allrows.append(md)

        print(f'\n--- {g}  (n={len(md)}, refs={md.dominant_accession.nunique()}) ---')
        for name in models:
            c, x = gm[name]['completeness'], gm[name]['contamination']
            print(f'  {name:<8} comp MAE {c["mae"]:6.3f} bias {c["bias"]:+6.3f} '
                  f'R2 {c["r2"]:6.3f} | cont MAE {x["mae"]:6.3f} '
                  f'bias {x["bias"]:+6.3f} R2 {x["r2"]:6.3f}')

    full = pd.concat(allrows, ignore_index=True)
    pcols = ['genome_id', 'group', 'novelty', 'dominant_accession', 'dominant_phylum',
             'dominant_taxon', 'dominant_domain', 'dominant_v5_split',
             'dominant_genome_size',
             'target_completeness', 'target_contamination', 'true_completeness',
             'true_contamination', 'n_contaminants', 'contaminant_phyla', 'seed',
             'n_contigs', 'total_length',
             'holdout_completeness', 'holdout_contamination',
             'V5_completeness', 'V5_contamination']
    with gzip.open(C.RESULTS_DIR / 'per_sample_predictions.tsv.gz', 'wt') as f:
        full[[c for c in pcols if c in full.columns]].to_csv(f, sep='\t', index=False)
    print(f'\nwrote {C.RESULTS_DIR / "per_sample_predictions.tsv.gz"} ({len(full)} rows)')

    # ---------------- head-to-head table ---------------------------------
    rows = []
    for g in groups:
        h, v = group_metrics[g]['holdout'], group_metrics[g]['V5']
        rows.append({
            'group': g,
            'lineage_seen_by_holdout': 'NO' if g != 'in_distribution' else 'yes',
            'n': h['completeness']['n'], 'n_refs': h['completeness']['n_refs'],
            'true_comp_mean': round(h['completeness']['true_mean'], 2),
            'true_cont_mean': round(h['contamination']['true_mean'], 2),
            'HO_comp_mae': round(h['completeness']['mae'], 3),
            'V5_comp_mae': round(v['completeness']['mae'], 3),
            'd_comp_mae': round(h['completeness']['mae'] - v['completeness']['mae'], 3),
            'HO_comp_bias': round(h['completeness']['bias'], 3),
            'V5_comp_bias': round(v['completeness']['bias'], 3),
            'HO_comp_rmse': round(h['completeness']['rmse'], 3),
            'V5_comp_rmse': round(v['completeness']['rmse'], 3),
            'HO_comp_r2': round(h['completeness']['r2'], 4),
            'V5_comp_r2': round(v['completeness']['r2'], 4),
            'HO_cont_mae': round(h['contamination']['mae'], 3),
            'V5_cont_mae': round(v['contamination']['mae'], 3),
            'd_cont_mae': round(h['contamination']['mae'] - v['contamination']['mae'], 3),
            'HO_cont_bias': round(h['contamination']['bias'], 3),
            'V5_cont_bias': round(v['contamination']['bias'], 3),
            'HO_cont_rmse': round(h['contamination']['rmse'], 3),
            'V5_cont_rmse': round(v['contamination']['rmse'], 3),
            'HO_cont_r2': round(h['contamination']['r2'], 4),
            'V5_cont_r2': round(v['contamination']['r2'], 4),
            'HO_comp_r2_pearson': round(h['completeness']['r2_pearson'], 4),
            'V5_comp_r2_pearson': round(v['completeness']['r2_pearson'], 4),
            'HO_cont_r2_pearson': round(h['contamination']['r2_pearson'], 4),
            'V5_cont_r2_pearson': round(v['contamination']['r2_pearson'], 4),
            'HO_comp_mae_ci95': h['completeness'].get('mae_ci95'),
            'HO_cont_mae_ci95': h['contamination'].get('mae_ci95'),
            'V5_comp_mae_ci95': v['completeness'].get('mae_ci95'),
            'V5_cont_mae_ci95': v['contamination'].get('mae_ci95'),
            'paired_d_comp_ci95': group_metrics[g]['paired']['completeness']['ci95'],
            'paired_d_comp_p': group_metrics[g]['paired']['completeness']['p_value'],
            'paired_d_cont_ci95': group_metrics[g]['paired']['contamination']['ci95'],
            'paired_d_cont_p': group_metrics[g]['paired']['contamination']['p_value'],
        })
    h2h = pd.DataFrame(rows)
    # Benjamini-Hochberg across all groups x both metrics
    allp = list(h2h.paired_d_comp_p) + list(h2h.paired_d_cont_p)
    q = benjamini_hochberg(allp)
    h2h['paired_d_comp_q_bh'] = np.round(q[:len(h2h)], 5)
    h2h['paired_d_cont_q_bh'] = np.round(q[len(h2h):], 5)
    h2h.to_csv(C.RESULTS_DIR / 'head_to_head_by_group.tsv', sep='\t', index=False)

    print('\n' + '=' * 78)
    print('HEAD-TO-HEAD: holdout model (HO) vs production V5 on identical genomes')
    print('=' * 78)
    print('All R2 values are the coefficient of determination (1 - SSres/SStot).')
    print(h2h[['group', 'n', 'n_refs', 'HO_comp_mae', 'V5_comp_mae', 'd_comp_mae',
               'HO_comp_bias', 'V5_comp_bias', 'HO_comp_r2', 'V5_comp_r2']]
          .to_string(index=False))
    print()
    print(h2h[['group', 'HO_cont_mae', 'V5_cont_mae', 'd_cont_mae',
               'HO_cont_bias', 'V5_cont_bias', 'HO_cont_r2', 'V5_cont_r2']]
          .to_string(index=False))
    print('\nPaired holdout-vs-V5 test on identical samples, cluster bootstrap by '
          'reference genome, BH-corrected across 7 groups x 2 metrics:')
    print(h2h[['group', 'd_comp_mae', 'paired_d_comp_ci95', 'paired_d_comp_p',
               'paired_d_comp_q_bh', 'd_cont_mae', 'paired_d_cont_ci95',
               'paired_d_cont_p', 'paired_d_cont_q_bh']].to_string(index=False))

    # ---------------- PRIMARY: lineage-novelty effect (difference-in-differences)
    did_rows = []
    if 'in_distribution' in groups:
        ctl = full[full.group == 'in_distribution']
        ae_ho_c_c = np.abs(ctl.holdout_completeness.values - ctl.true_completeness.values)
        ae_v5_c_c = np.abs(ctl.V5_completeness.values - ctl.true_completeness.values)
        ae_ho_c_x = np.abs(ctl.holdout_contamination.values
                           - ctl.true_contamination.values)
        ae_v5_c_x = np.abs(ctl.V5_contamination.values - ctl.true_contamination.values)
        cl_c = ctl.dominant_accession.values

        ctl_d_c = float(ae_ho_c_c.mean() - ae_v5_c_c.mean())
        ctl_d_x = float(ae_ho_c_x.mean() - ae_v5_c_x.mean())
        print('\n' + '=' * 78)
        print('CONTROL - in_distribution (both models saw these lineages)')
        print('=' * 78)
        print('The holdout-minus-V5 gap here is the cost of training on a smaller\n'
              'genome pool ALONE, with lineage novelty held constant:')
        print(f'  completeness  Delta = {ctl_d_c:+.3f} pp MAE')
        print(f'  contamination Delta = {ctl_d_x:+.3f} pp MAE')

        for g in groups:
            if g == 'in_distribution':
                continue
            sub = full[full.group == g]
            cl_g = sub.dominant_accession.values
            r = {'group': g}
            for metric, tcol, pcol in [('comp', 'true_completeness', '_completeness'),
                                       ('cont', 'true_contamination', '_contamination')]:
                ho = np.abs(sub['holdout' + pcol].values - sub[tcol].values)
                v5 = np.abs(sub['V5' + pcol].values - sub[tcol].values)
                cc_ho = ae_ho_c_c if metric == 'comp' else ae_ho_c_x
                cc_v5 = ae_v5_c_c if metric == 'comp' else ae_v5_c_x
                d = did_cluster_boot(ho, v5, cl_g, cc_ho, cc_v5, cl_c)
                r[f'{metric}_delta'] = round(float(ho.mean() - v5.mean()), 3)
                r[f'{metric}_did'] = d['did']
                r[f'{metric}_did_ci95'] = d['ci95']
                r[f'{metric}_did_p'] = d['p_value']
            did_rows.append(r)
        diddf = pd.DataFrame(did_rows)
        q = benjamini_hochberg(list(diddf.comp_did_p) + list(diddf.cont_did_p))
        diddf['comp_did_q_bh'] = np.round(q[:len(diddf)], 5)
        diddf['cont_did_q_bh'] = np.round(q[len(diddf):], 5)
        diddf.to_csv(C.RESULTS_DIR / 'lineage_novelty_effect_did.tsv',
                     sep='\t', index=False)
        print('\n' + '=' * 78)
        print('PRIMARY RESULT - lineage-novelty effect, difference-in-differences')
        print(f'  DiD = (MAE_HO - MAE_V5)_heldout_{C.PANEL_LEVEL} '
              f'- (MAE_HO - MAE_V5)_control')
        print('  Positive = the holdout model is worse specifically BECAUSE the '
              'lineage was absent.')
        print('  Immune to group composition; both models score identical samples.')
        print('=' * 78)
        print(diddf.to_string(index=False))

        # Secondary, and explicitly flagged as composition-confounded
        base = group_metrics['in_distribution']['holdout']
        deg = []
        for g in groups:
            if g == 'in_distribution':
                continue
            hh = group_metrics[g]['holdout']
            deg.append({'group': g,
                        'HO_comp_mae': round(hh['completeness']['mae'], 3),
                        'comp_mae_pp_vs_control': round(hh['completeness']['mae']
                                                        - base['completeness']['mae'], 3),
                        'HO_cont_mae': round(hh['contamination']['mae'], 3),
                        'cont_mae_ratio_vs_control': round(hh['contamination']['mae']
                                                           / base['contamination']['mae'],
                                                           3)})
        degdf = pd.DataFrame(deg)
        degdf.to_csv(C.RESULTS_DIR / 'degradation_vs_in_distribution.tsv',
                     sep='\t', index=False)
        print('\nProtocol decision rule as literally written (WS1.6 acceptance): '
              'descend to family\nlevel if held-out contamination MAE > ~2x '
              'in-distribution, OR completeness MAE\ndegrades > 5 pp. Evaluated '
              'below, but note this rule compares RAW MAEs across\ngroups of '
              'different phylum composition and is therefore confounded; the DiD '
              'table\nabove is the defensible estimator.')
        print(degdf.to_string(index=False))
        worst_c = degdf.comp_mae_pp_vs_control.max()
        worst_x = degdf.cont_mae_ratio_vs_control.max()
        print(f'  worst completeness MAE degradation: {worst_c:+.3f} pp '
              f'(threshold +5 pp) -> {"EXCEEDED" if worst_c > 5 else "within"}')
        print(f'  worst contamination MAE ratio: {worst_x:.3f}x '
              f'(threshold 2.0x) -> {"EXCEEDED" if worst_x > 2 else "within"}')
        worst_did_c = max(r['comp_did'] for r in did_rows)
        worst_did_x = max(r['cont_did'] for r in did_rows)
        print(f'\n  DiD-based reading: worst lineage-novelty penalty is '
              f'{worst_did_c:+.3f} pp completeness MAE and {worst_did_x:+.3f} pp '
              f'contamination MAE.')

    # ---------------- split stratification for 'all'-source groups -------
    # Only needed where dominants could not all come from the test split (WS1.6
    # DPANN). Every WS1.9 family group uses test-split dominants, so this block
    # is a no-op at family level.
    for _g in [g for g in groups
               if C.EVAL_DESIGN.get(g, {}).get('ref_source') == 'all']:
        d = full[full.group == _g]
        rows = []
        for sp, sub in d.groupby('dominant_v5_split'):
            cl = sub.dominant_accession.values
            for name in models:
                rows.append({
                    'v5_split_of_dominant': sp, 'model': name, 'n': len(sub),
                    'n_refs': sub.dominant_accession.nunique(),
                    'comp_mae': round(float(np.abs(sub[f'{name}_completeness']
                                                   - sub.true_completeness).mean()), 3),
                    'cont_mae': round(float(np.abs(sub[f'{name}_contamination']
                                                   - sub.true_contamination).mean()), 3),
                    'comp_bias': round(float((sub[f'{name}_completeness']
                                              - sub.true_completeness).mean()), 3),
                    'cont_bias': round(float((sub[f'{name}_contamination']
                                              - sub.true_contamination).mean()), 3)})
        dp = pd.DataFrame(rows).sort_values(['v5_split_of_dominant', 'model'])
        dp.to_csv(C.RESULTS_DIR / 'dpann_split_stratified.tsv', sep='\t', index=False)
        print(f'\n{_g} stratified by which V5 split the dominant reference came from.')
        print('V5 TRAINED on the train-split references; the holdout model saw none.')
        print(dp.to_string(index=False))

    # ---------------- MIMAG-threshold classification (R2-M2) -------------
    # Point estimates can hide the failure mode that actually matters to users:
    # whether a genome lands in the right MIMAG bin. Report per group, per model.
    mimag_rows, conf_rows = [], []
    for g in groups:
        sub = full[full.group == g]
        tcls = mimag_class(sub.true_completeness.values, sub.true_contamination.values)
        for name in models:
            pcls = mimag_class(sub[f'{name}_completeness'].values,
                               sub[f'{name}_contamination'].values)
            agree = float((tcls == pcls).mean())
            # threshold-crossing errors that matter most
            t_hq = tcls == 'high'
            p_hq = pcls == 'high'
            mimag_rows.append({
                'group': g, 'model': name, 'n': len(sub),
                'mimag_agreement': round(agree, 4),
                'true_HQ': int(t_hq.sum()), 'pred_HQ': int(p_hq.sum()),
                'HQ_recall': round(float(p_hq[t_hq].mean()), 4) if t_hq.sum() else None,
                'HQ_precision': round(float(t_hq[p_hq].mean()), 4) if p_hq.sum() else None,
                # contamination threshold errors
                'cont5_false_clean': int(((sub.true_contamination >= 5) &
                                          (sub[f'{name}_contamination'] < 5)).sum()),
                'cont5_false_dirty': int(((sub.true_contamination < 5) &
                                          (sub[f'{name}_contamination'] >= 5)).sum()),
                'cont10_false_clean': int(((sub.true_contamination >= 10) &
                                           (sub[f'{name}_contamination'] < 10)).sum()),
                'cont10_false_dirty': int(((sub.true_contamination < 10) &
                                           (sub[f'{name}_contamination'] >= 10)).sum()),
                'comp90_false_high': int(((sub.true_completeness < 90) &
                                          (sub[f'{name}_completeness'] >= 90)).sum()),
                'comp90_false_low': int(((sub.true_completeness >= 90) &
                                         (sub[f'{name}_completeness'] < 90)).sum()),
            })
            for tc in ['high', 'medium', 'low']:
                for pc in ['high', 'medium', 'low']:
                    n_ = int(((tcls == tc) & (pcls == pc)).sum())
                    if n_:
                        conf_rows.append({'group': g, 'model': name, 'true': tc,
                                          'pred': pc, 'n': n_})
    mimag_df = pd.DataFrame(mimag_rows)
    mimag_df.to_csv(C.RESULTS_DIR / 'mimag_threshold_by_group.tsv', sep='\t', index=False)
    pd.DataFrame(conf_rows).to_csv(C.RESULTS_DIR / 'mimag_confusion_by_group.tsv',
                                   sep='\t', index=False)
    print('\n' + '=' * 78)
    print('MIMAG-INSPIRED THRESHOLDS (HQ: comp>=90 & cont<5; MQ: comp>=50 & cont<10)')
    print('=' * 78)
    print(mimag_df.to_string(index=False))

    # ---------------- stratified error: where does it break? -------------
    strat_rows = []
    cbands = [(50, 70), (70, 90), (90, 100.01)]
    xbands = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 70), (70, 100.01)]
    for g in groups:
        sub = full[full.group == g]
        for lo, hi in cbands:
            m_ = (sub.true_completeness >= lo) & (sub.true_completeness < hi)
            if m_.sum() < 10:
                continue
            for name in models:
                s2 = sub[m_]
                strat_rows.append({
                    'group': g, 'stratify_by': 'true_completeness',
                    'band': f'{lo}-{hi if hi<=100 else 100}', 'model': name,
                    'n': int(m_.sum()),
                    'comp_mae': round(float(np.abs(s2[f'{name}_completeness']
                                                   - s2.true_completeness).mean()), 3),
                    'comp_bias': round(float((s2[f'{name}_completeness']
                                              - s2.true_completeness).mean()), 3),
                    'cont_mae': round(float(np.abs(s2[f'{name}_contamination']
                                                   - s2.true_contamination).mean()), 3),
                    'cont_bias': round(float((s2[f'{name}_contamination']
                                              - s2.true_contamination).mean()), 3)})
        for lo, hi in xbands:
            m_ = (sub.true_contamination >= lo) & (sub.true_contamination < hi)
            if m_.sum() < 10:
                continue
            for name in models:
                s2 = sub[m_]
                strat_rows.append({
                    'group': g, 'stratify_by': 'true_contamination',
                    'band': f'{lo}-{hi if hi<=100 else 100}', 'model': name,
                    'n': int(m_.sum()),
                    'comp_mae': round(float(np.abs(s2[f'{name}_completeness']
                                                   - s2.true_completeness).mean()), 3),
                    'comp_bias': round(float((s2[f'{name}_completeness']
                                              - s2.true_completeness).mean()), 3),
                    'cont_mae': round(float(np.abs(s2[f'{name}_contamination']
                                                   - s2.true_contamination).mean()), 3),
                    'cont_bias': round(float((s2[f'{name}_contamination']
                                              - s2.true_contamination).mean()), 3)})
    pd.DataFrame(strat_rows).to_csv(C.RESULTS_DIR / 'stratified_error_by_band.tsv',
                                    sep='\t', index=False)
    print(f'\nwrote {C.RESULTS_DIR / "stratified_error_by_band.tsv"} '
          f'({len(strat_rows)} rows: each group x completeness/contamination band '
          f'x model)')
    print('Low-contamination band (0-5%), the MIMAG-relevant regime:')
    sdf = pd.DataFrame(strat_rows)
    lowc = (sdf[(sdf.stratify_by == 'true_contamination') & (sdf.band == '0-5')]
            if len(sdf) else sdf)
    if len(lowc):
        print(lowc[['group', 'model', 'n', 'cont_mae', 'cont_bias',
                    'comp_mae', 'comp_bias']].to_string(index=False))

    # ---------------- sub-phylum breakdown (DPANN + in_distribution) -----
    # Reported for the control (which pools many phyla) and for every panel
    # group that pools more than one taxon - at family level that is the
    # Patescibacteriota and Halobacteriota groups, where the per-family numbers
    # matter because the pooled headline can hide a single bad family.
    sub_rows = []
    _pooled = [g for g in groups
               if g == C.CONTROL_GROUP or len(C.PANEL_GROUPS.get(g, {}).get('taxa', [])) > 1]
    for g in _pooled:
        sub = full[full.group == g]
        _key = 'dominant_phylum' if g == C.CONTROL_GROUP else 'dominant_taxon'
        if _key not in sub.columns:
            _key = 'dominant_phylum'
        for ph, s2 in sub.groupby(_key):
            if len(s2) < 10:
                continue
            for name in models:
                sub_rows.append({
                    'group': g, 'stratum_level': ('phylum' if _key == 'dominant_phylum'
                                                  else C.PANEL_LEVEL),
                    'dominant_phylum': ph, 'model': name, 'n': len(s2),
                    'n_refs': int(s2.dominant_accession.nunique()),
                    'comp_mae': round(float(np.abs(s2[f'{name}_completeness']
                                                   - s2.true_completeness).mean()), 3),
                    'comp_bias': round(float((s2[f'{name}_completeness']
                                              - s2.true_completeness).mean()), 3),
                    'cont_mae': round(float(np.abs(s2[f'{name}_contamination']
                                                   - s2.true_contamination).mean()), 3),
                    'cont_bias': round(float((s2[f'{name}_contamination']
                                              - s2.true_contamination).mean()), 3)})
    if sub_rows:
        sbdf = pd.DataFrame(sub_rows)
        sbdf.to_csv(C.RESULTS_DIR / 'sub_phylum_breakdown.tsv', sep='\t', index=False)
        print('\nPER-TAXON BREAKDOWN inside pooled evaluation groups')
        print(sbdf.to_string(index=False))

    # ---------------- WS1.8: per-reference errors ------------------------
    per_ref = []
    for (g, acc), sub in full.groupby(['group', 'dominant_accession']):
        r = {'group': g, 'novelty': int(sub.novelty.iloc[0]),
             'dominant_accession': acc, 'dominant_phylum': sub.dominant_phylum.iloc[0],
             'dominant_taxon': (sub.dominant_taxon.iloc[0]
                                if 'dominant_taxon' in sub.columns else ''),
             'dominant_domain': sub.dominant_domain.iloc[0],
             'dominant_v5_split': sub.dominant_v5_split.iloc[0],
             'genome_size_Mbp': round(sub.dominant_genome_size.iloc[0] / 1e6, 4),
             'n_sims': len(sub)}
        for name in models:
            r[f'{name}_comp_mae'] = round(float(np.abs(
                sub[f'{name}_completeness'] - sub.true_completeness).mean()), 4)
            r[f'{name}_cont_mae'] = round(float(np.abs(
                sub[f'{name}_contamination'] - sub.true_contamination).mean()), 4)
            r[f'{name}_comp_bias'] = round(float(
                (sub[f'{name}_completeness'] - sub.true_completeness).mean()), 4)
            r[f'{name}_cont_bias'] = round(float(
                (sub[f'{name}_contamination'] - sub.true_contamination).mean()), 4)
        per_ref.append(r)
    prdf = pd.DataFrame(per_ref)
    prdf.to_csv(C.RESULTS_DIR / 'per_reference_errors.tsv', sep='\t', index=False)
    print(f'\nwrote {C.RESULTS_DIR / "per_reference_errors.tsv"} '
          f'({len(prdf)} reference genomes) - WS1.8 plotting input')

    # ---------------- optional clean Sets C/D ----------------------------
    clean_rows = []
    for setname in ['set_C_clean', 'set_D_clean']:
        sd = C.PROJECT_ROOT / 'data/benchmarks' / setname
        mp_ = sd / 'metadata.tsv'
        if not mp_.exists():
            print(f'\n{setname}: not present yet (another agent is generating it) - skipped')
            continue
        print(f'\n{setname}: found, evaluating both models')
        cm = pd.read_csv(mp_, sep='\t')
        try:
            from holdout_lib.kmer_counter import (load_selected_kmers, build_kmer_index,
                                                  _count_kmers_single, K)
            from holdout_lib.assembly_stats import compute_assembly_stats
            from holdout_lib.fragmentation import read_fasta
            ki = build_kmer_index(load_selected_kmers(str(C.SELECTED_KMERS)))
            nk = C.N_KMER_FEATURES
            fcol = ('fasta_path' if 'fasta_path' in cm.columns else
                    'fasta' if 'fasta' in cm.columns else None)
            kr = np.zeros((len(cm), nk), dtype=np.int64)
            ar = np.zeros((len(cm), C.N_SUMMARY_FEATURES))
            for i, row in cm.iterrows():
                fp = (row[fcol] if fcol else str(sd / 'fasta' / f'{row["genome_id"]}.fasta'))
                seq = read_fasta(fp)
                cnt = _count_kmers_single(
                    np.frombuffer(seq.encode('ascii'), dtype=np.uint8), ki, nk, K)
                kr[i] = cnt
                t = cnt.sum()
                ar[i] = compute_assembly_stats(np.log10(float(t)) if t > 0 else 0.0, cnt)
            tc = cm['true_completeness'].values.astype(float)
            tx = cm['true_contamination'].values.astype(float)
            cl = cm['dominant_accession'].values
            nviol = int((tx > tc + 1e-6).sum())
            print(f'  n={len(cm)} refs={cm.dominant_accession.nunique()} '
                  f'constraint violations={nviol}')
            print(f'  phyla: {dict(cm.dominant_phylum.value_counts())}')
            preds = {}
            for name, (sess, nrm) in models.items():
                preds[name] = predict(sess, nrm, kr, ar)
            # whole set, then stratified by dominant phylum (set_D_clean spans
            # 6 archaeal phyla, only 2 of which are in the holdout panel)
            strata = [('ALL', np.ones(len(cm), bool))]
            for ph in sorted(cm.dominant_phylum.unique()):
                strata.append((ph, (cm.dominant_phylum == ph).values))
            for label, msk in strata:
                if msk.sum() < 2:
                    continue
                # At family level no phylum is held out, so a phylum stratum is
                # only PARTIALLY held out (some of its families are in the panel).
                inpanel = ('n/a' if label == 'ALL'
                           else ('HELD OUT' if label in C.PANEL_PHYLA
                                 else ('partially held out (some families in panel)'
                                       if (C.PANEL_LEVEL in ('family', 'genus') and
                                           label in C.PANEL_PARENT_PHYLA)
                                       else 'in-dist')))
                for name in models:
                    mc = metrics(tc[msk], preds[name][msk, 0], cl[msk],
                                 ci=(label == 'ALL'))
                    mx = metrics(tx[msk], preds[name][msk, 1], cl[msk],
                                 ci=(label == 'ALL'))
                    clean_rows.append({
                        'set': setname, 'stratum': label,
                        'holdout_status': inpanel, 'model': name,
                        'n': int(msk.sum()), 'n_refs': mc['n_refs'],
                        'comp_mae': round(mc['mae'], 3),
                        'comp_bias': round(mc['bias'], 3),
                        'comp_rmse': round(mc['rmse'], 3),
                        'comp_r2': round(mc['r2'], 4),
                        'comp_r2_pearson': round(mc['r2_pearson'], 4),
                        'cont_mae': round(mx['mae'], 3),
                        'cont_bias': round(mx['bias'], 3),
                        'cont_rmse': round(mx['rmse'], 3),
                        'cont_r2': round(mx['r2'], 4),
                        'cont_r2_pearson': round(mx['r2_pearson'], 4),
                        'comp_mae_ci95': mc.get('mae_ci95'),
                        'cont_mae_ci95': mx.get('mae_ci95')})
            cm['group'] = setname
            for name in models:
                cm[f'{name}_completeness'] = preds[name][:, 0]
                cm[f'{name}_contamination'] = preds[name][:, 1]
            cm.to_csv(C.RESULTS_DIR / f'{setname}_predictions_both_models.tsv',
                      sep='\t', index=False)
        except Exception as e:
            print(f'  could not evaluate {setname}: {e}')
    if clean_rows:
        cdf = pd.DataFrame(clean_rows)
        cdf.to_csv(C.RESULTS_DIR / 'clean_sets_evaluation.tsv', sep='\t', index=False)
        print('\n' + '=' * 78)
        print('SET C_clean / D_clean (built independently by another agent from '
              'strictly held-out test-split references)')
        print('CAVEAT: these sets draw CONTAMINANTS from the whole test split, which '
              'includes\npanel phyla (e.g. Bacteroidota). For the holdout model both the '
              'dominant AND some\ncontaminant sequence can therefore be novel, so its '
              'numbers here are an upper bound\non degradation. The purpose-built '
              'eval_sets above use panel-free contaminants only\nand isolate dominant-'
              'lineage novelty cleanly.')
        print('=' * 78)
        print(cdf[cdf.stratum == 'ALL'].to_string(index=False))
        print('\nStratified by dominant phylum:')
        print(cdf[cdf.stratum != 'ALL'][
            ['set', 'stratum', 'holdout_status', 'model', 'n', 'n_refs',
             'comp_mae', 'comp_bias', 'cont_mae', 'cont_bias']].to_string(index=False))

    out = {'holdout_onnx': a.holdout_onnx, 'v5_onnx': str(C.V5_ONNX),
           'provenance_audit': audit,
           'eval_manifest': manifest, 'group_metrics': group_metrics,
           'n_bootstrap': N_BOOT, 'bootstrap': 'cluster bootstrap by reference genome'}
    (C.RESULTS_DIR / 'metrics_full.json').write_text(json.dumps(out, indent=2))
    print(f'\nwrote {C.RESULTS_DIR / "metrics_full.json"}')

    # ---- one consolidated file for the Methods/Results writer -----------
    def _read(p):
        p = Path(p)
        return json.loads(p.read_text()) if p.exists() else None
    consolidated = {
        'workstreams': {
            'phylum': ['WS1.6 leave-phylum-out retraining',
                       'WS1.7 k-mer feature-selection leakage control',
                       'WS1.8 accuracy vs taxonomic novelty'],
            'family': ['WS1.9 leave-family-out retraining (families held out from '
                       'phyla that REMAIN in training)'],
            'genus': ['WS11.G leave-genus-out retraining (genera held out from '
                      'FAMILIES and phyla that REMAIN in training)'],
        }[C.PANEL_LEVEL],
        'panel_level': C.PANEL_LEVEL,
        'panel_taxa': C.PANEL_TAXA,
        'holdout_model_status': ('VALIDATION ARTIFACT ONLY - the released MAGICC '
                                 'model remains trained on all data'),
        'generated': pd.Timestamp.now().isoformat(),
        'panel': _read(C.PANEL_JSON),
        'training': _read(C.RESULTS_DIR / 'training_summary.json'),
        'onnx_export': _read(C.RESULTS_DIR / 'onnx_export_verification.json'),
        'kmer_reselection': _read(C.RESULTS_DIR / 'kmer_reselection_summary.json'),
        'evaluation': out,
        'head_to_head_table': h2h.to_dict('records'),
        'clean_sets': clean_rows or None,
        'r2_convention': ('All fields named r2 are the coefficient of determination '
                          '1 - SSres/SStot. Fields named r2_pearson are squared '
                          'Pearson correlation and must not be labelled R2.'),
        'mimag_threshold': mimag_df.to_dict('records'),
        'sub_phylum_breakdown': sub_rows or None,
        'lineage_novelty_did': did_rows or None,
        'primary_estimator': ('difference-in-differences: (MAE_holdout - MAE_V5) on the '
                              'held-out phylum minus the same quantity on the '
                              'in_distribution control. Raw cross-group MAE comparisons '
                              'are confounded by phylum composition of the control.'),
        'files': {
            'head_to_head': str(C.RESULTS_DIR / 'head_to_head_by_group.tsv'),
            'lineage_novelty_did': str(C.RESULTS_DIR / 'lineage_novelty_effect_did.tsv'),
            'mimag_threshold': str(C.RESULTS_DIR / 'mimag_threshold_by_group.tsv'),
            'mimag_confusion': str(C.RESULTS_DIR / 'mimag_confusion_by_group.tsv'),
            'stratified_error': str(C.RESULTS_DIR / 'stratified_error_by_band.tsv'),
            'sub_phylum': str(C.RESULTS_DIR / 'sub_phylum_breakdown.tsv'),
            'per_sample': str(C.RESULTS_DIR / 'per_sample_predictions.tsv.gz'),
            'per_reference_ws1_8': str(C.RESULTS_DIR / 'per_reference_errors.tsv'),
            'degradation': str(C.RESULTS_DIR / 'degradation_vs_in_distribution.tsv'),
            'dpann': str(C.RESULTS_DIR / 'dpann_split_stratified.tsv'),
            'eval_sets': str(eval_dir),
            'holdout_features_h5': str(C.HOLDOUT_H5),
            'holdout_onnx': a.holdout_onnx,
            'holdout_norm_params': str(C.HOLDOUT_NORM_PARAMS),
            'training_history': str(C.HOLDOUT_HISTORY),
        }}
    _cons = {'phylum': 'ws1.6_1.7_1.8_consolidated.json',
             'family': 'ws1.9_consolidated.json',
             'genus': 'ws11g_consolidated.json'}[C.PANEL_LEVEL]
    (C.RESULTS_DIR / _cons).write_text(
        json.dumps(consolidated, indent=2, default=str))
    print(f'wrote {C.RESULTS_DIR / _cons}')
    print('\nEVALUATION COMPLETE')


if __name__ == '__main__':
    main()
