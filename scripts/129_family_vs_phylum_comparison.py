#!/usr/bin/env python3
"""
WS1.9 step 6 - DIRECT comparison of FAMILY-level against PHYLUM-level novelty.

THE DESIGN THAT MAKES THIS A CLEAN COMPARISON
  The WS1.6 phylum panel held out Bacteroidota, Campylobacterota, Halobacteriota
  and Patescibacteriota - which are exactly the four parent phyla of the WS1.9
  family panel. Running all THREE models on the SAME family evaluation genomes
  therefore gives a three-rung ladder of taxonomic novelty on identical samples:

    production V5            saw the family AND the phylum
    family-holdout model     never saw the family, DID see the phylum   <- WS1.9
    phylum-holdout model     never saw the family NOR the phylum        <- WS1.6

  so the two novelty effects are measured on the same genomes, with the same
  contaminants, at the same true completeness/contamination - a paired
  comparison, not two independent experiments glued together.

  DiD_family = (MAE_familyHO - MAE_V5)_group - (MAE_familyHO - MAE_V5)_control
  DiD_phylum = (MAE_phylumHO - MAE_V5)_group - (MAE_phylumHO - MAE_V5)_control
  ATTENUATION = DiD_phylum - DiD_family
      > 0  novelty costs LESS at family level than at phylum level, i.e. keeping
           the parent phylum in training rescues part of the loss;
      ~ 0  family-level novelty is as damaging as phylum-level novelty.

  All three models score identical samples, so the estimator is immune to group
  composition and to the genome-SIZE-driven completeness bias documented in
  WS3.10 (that bias is shared by all three models on the same genome and cancels).

CONTROL RESTRICTION (necessary, and reported)
  The WS1.9 in_distribution control is drawn from non-panel-FAMILY test genomes,
  some of which belong to phyla that the WS1.6 model never saw. Those samples are
  not in-distribution for the phylum-holdout model. The common control used here
  is therefore restricted to control samples whose dominant phylum is outside the
  WS1.6 phylum panel; the retained n is reported.

Significance: paired two-sided cluster bootstrap over REFERENCE GENOMES (2,000
iterations), Benjamini-Hochberg corrected across all groups x 2 metrics.
Bootstrap seeding uses CRC-32 (stable_hash), never Python's salted hash().

Outputs (results/revision/holdout_family/)
  family_vs_phylum_did.tsv            the deliverable: DiD ladder + attenuation
  three_model_head_to_head.tsv        raw MAE/bias/R2 for all three models
  cross_experiment_did_summary.tsv    the two experiments' own DiD tables, by phylum
  family_vs_phylum_per_sample.tsv.gz  every prediction from all three models
  family_vs_phylum_report.md          readable summary

Usage
  MAGICC_HOLDOUT_LEVEL=family python scripts/129_family_vs_phylum_comparison.py
"""

import gzip
import json
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_lib import config as C
from holdout_lib.normalization import FeatureNormalizer

N_BOOT = 2000
BOOT_SEED = 20260727

PHYLUM_RESULTS = C.PROJECT_ROOT / 'results/revision/holdout'
PHYLUM_ONNX = C.MODELS_DIR / 'magicc_holdout_phylum.onnx'
PHYLUM_NORM = C.PROJECT_ROOT / 'data/holdout/holdout_normalization_params.json'
# The WS1.6 panel, needed to restrict the common control
WS16_PANEL_PHYLA = ['Altiarchaeota', 'Bacteroidota', 'Bacteroidota_A',
                    'Campylobacterota', 'Halobacteriota', 'Iainarchaeota',
                    'Micrarchaeota', 'Nanobdellota', 'Nanohalarchaeota',
                    'Patescibacteriota']


def stable_hash(name: str) -> int:
    """CRC-32; process-stable, unlike Python's salted hash()."""
    return zlib.crc32(str(name).encode('utf-8')) & 0xFFFFFFFF


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


def _clusters(cl):
    uc, inv = np.unique(cl, return_inverse=True)
    return [np.where(inv == i)[0] for i in range(len(uc))]


def did_and_attenuation(ae_fam_g, ae_phy_g, ae_v5_g, cl_g,
                        ae_fam_c, ae_phy_c, ae_v5_c, cl_c,
                        n_boot=N_BOOT, seed=BOOT_SEED):
    """Both DiDs and their difference, from ONE cluster bootstrap.

    Because all three models score the same samples, the same resampled cluster
    indices are used for all three within each bootstrap iteration, so the
    attenuation is a properly PAIRED quantity.
    """
    gg, gc = _clusters(cl_g), _clusters(cl_c)

    def _did(ho_g, ho_c):
        return ((ho_g.mean() - ae_v5_g.mean()) - (ho_c.mean() - ae_v5_c.mean()))

    obs_f = float(_did(ae_fam_g, ae_fam_c))
    obs_p = float(_did(ae_phy_g, ae_phy_c))
    obs_a = obs_p - obs_f
    if len(gg) < 3 or len(gc) < 3:
        return {'did_family': round(obs_f, 4), 'did_phylum': round(obs_p, 4),
                'attenuation': round(obs_a, 4), 'ci95_family': None,
                'ci95_phylum': None, 'ci95_attenuation': None,
                'p_family': None, 'p_attenuation': None,
                'pct_of_phylum_effect': None}
    rng = np.random.default_rng(seed)
    bf = np.empty(n_boot)
    bp = np.empty(n_boot)
    for b in range(n_boot):
        ig = np.concatenate([gg[i] for i in rng.integers(0, len(gg), len(gg))])
        ic = np.concatenate([gc[i] for i in rng.integers(0, len(gc), len(gc))])
        dv_g = ae_v5_g[ig].mean()
        dv_c = ae_v5_c[ic].mean()
        bf[b] = (ae_fam_g[ig].mean() - dv_g) - (ae_fam_c[ic].mean() - dv_c)
        bp[b] = (ae_phy_g[ig].mean() - dv_g) - (ae_phy_c[ic].mean() - dv_c)
    ba = bp - bf

    def _p(draws, obs):
        sh = draws - obs
        v = 2.0 * min(np.mean(sh >= obs), np.mean(sh <= obs))
        return float(min(1.0, max(v, 1.0 / n_boot)))

    return {
        'did_family': round(obs_f, 4), 'did_phylum': round(obs_p, 4),
        'attenuation': round(obs_a, 4),
        'ci95_family': [round(float(np.percentile(bf, 2.5)), 4),
                        round(float(np.percentile(bf, 97.5)), 4)],
        'ci95_phylum': [round(float(np.percentile(bp, 2.5)), 4),
                        round(float(np.percentile(bp, 97.5)), 4)],
        'ci95_attenuation': [round(float(np.percentile(ba, 2.5)), 4),
                             round(float(np.percentile(ba, 97.5)), 4)],
        'p_family': _p(bf, obs_f),
        'p_attenuation': _p(ba, obs_a),
        'pct_of_phylum_effect': (round(100.0 * obs_f / obs_p, 1)
                                 if abs(obs_p) > 1e-9 else None)}


def benjamini_hochberg(pvals):
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m, prev = len(order), 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = min(1.0, prev)
    return q


def metrics(true, pred):
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {'mae': float(np.abs(err).mean()),
            'rmse': float(np.sqrt((err ** 2).mean())),
            # PROJECT STANDARD: R2 = coefficient of determination, never r^2
            'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
            'bias': float(err.mean())}


def main():
    assert C.PANEL_LEVEL == 'family', 'run with MAGICC_HOLDOUT_LEVEL=family'
    out_dir = C.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 78)
    print('WS1.9 vs WS1.6 - is novel-FAMILY degradation smaller than '
          'novel-PHYLUM degradation?')
    print('=' * 78)

    missing = [str(p) for p in [C.HOLDOUT_ONNX, C.HOLDOUT_NORM_PARAMS,
                                PHYLUM_ONNX, PHYLUM_NORM, C.V5_ONNX,
                                C.V5_NORM_PARAMS] if not Path(p).exists()]
    if missing:
        sys.exit(f'missing model artifacts: {missing}')

    models = {
        'V5': load_model(C.V5_ONNX, C.V5_NORM_PARAMS),
        'familyHO': load_model(C.HOLDOUT_ONNX, C.HOLDOUT_NORM_PARAMS),
        'phylumHO': load_model(PHYLUM_ONNX, PHYLUM_NORM),
    }
    print('models:')
    print(f'  V5        {C.V5_ONNX}')
    print(f'  familyHO  {C.HOLDOUT_ONNX}   (WS1.9; never saw the panel FAMILIES)')
    print(f'  phylumHO  {PHYLUM_ONNX}   (WS1.6; never saw those families\' PHYLA)')
    print('Both holdout models are VALIDATION ARTIFACTS ONLY; the released MAGICC '
          'model remains trained on all data.')

    eval_dir = C.EVAL_DIR
    groups = [g for g in C.EVAL_DESIGN if (eval_dir / g / 'features.h5').exists()]
    print(f'\nevaluation groups: {groups}')

    rows = []
    for g in groups:
        md = pd.read_csv(eval_dir / g / 'metadata.tsv', sep='\t')
        with h5py.File(eval_dir / g / 'features.h5', 'r') as f:
            kr = f['kmer_counts_raw'][:]
            ar = f['summary_features_raw'][:]
        assert len(md) == len(kr)
        for name, (sess, nrm) in models.items():
            p = predict(sess, nrm, kr, ar)
            md[f'{name}_completeness'] = p[:, 0]
            md[f'{name}_contamination'] = p[:, 1]
        md['group'] = g
        rows.append(md)
    full = pd.concat(rows, ignore_index=True)

    keep = ['genome_id', 'group', 'dominant_accession', 'dominant_phylum',
            'dominant_taxon', 'dominant_genome_size', 'true_completeness',
            'true_contamination'] + \
           [f'{m}_{k}' for m in models for k in ('completeness', 'contamination')]
    with gzip.open(out_dir / 'family_vs_phylum_per_sample.tsv.gz', 'wt') as f:
        full[[c for c in keep if c in full.columns]].to_csv(f, sep='\t', index=False)

    # ---------------- three-model head-to-head ---------------------------
    h2h = []
    for g in groups:
        sub = full[full.group == g]
        r = {'group': g, 'n': len(sub),
             'n_refs': int(sub.dominant_accession.nunique()),
             'true_comp_mean': round(float(sub.true_completeness.mean()), 2),
             'true_cont_mean': round(float(sub.true_contamination.mean()), 2)}
        for name in models:
            mc = metrics(sub.true_completeness.values,
                         sub[f'{name}_completeness'].values)
            mx = metrics(sub.true_contamination.values,
                         sub[f'{name}_contamination'].values)
            r.update({f'{name}_comp_mae': round(mc['mae'], 3),
                      f'{name}_comp_bias': round(mc['bias'], 3),
                      f'{name}_comp_r2': round(mc['r2'], 4),
                      f'{name}_cont_mae': round(mx['mae'], 3),
                      f'{name}_cont_bias': round(mx['bias'], 3),
                      f'{name}_cont_r2': round(mx['r2'], 4)})
        h2h.append(r)
    h2h = pd.DataFrame(h2h)
    h2h.to_csv(out_dir / 'three_model_head_to_head.tsv', sep='\t', index=False)
    print('\n' + '=' * 78)
    print('THREE MODELS ON IDENTICAL SAMPLES (R2 = coefficient of determination)')
    print('=' * 78)
    print(h2h[['group', 'n', 'n_refs', 'V5_comp_mae', 'familyHO_comp_mae',
               'phylumHO_comp_mae', 'V5_cont_mae', 'familyHO_cont_mae',
               'phylumHO_cont_mae']].to_string(index=False))

    # ---------------- common control -------------------------------------
    ctl_all = full[full.group == C.CONTROL_GROUP]
    ctl = ctl_all[~ctl_all.dominant_phylum.isin(WS16_PANEL_PHYLA)]
    print(f'\nCOMMON CONTROL: {len(ctl)}/{len(ctl_all)} in_distribution samples '
          f'({ctl.dominant_accession.nunique()} references) are in-distribution for '
          f'BOTH holdout models.')
    print(f'  dropped: {len(ctl_all) - len(ctl)} samples whose dominant phylum is in '
          f'the WS1.6 panel and is therefore NOT in-distribution for phylumHO.')
    if ctl.dominant_accession.nunique() < 3:
        sys.exit('common control has too few reference genomes')

    cl_c = ctl.dominant_accession.values
    ctl_ae = {}
    for name in models:
        ctl_ae[(name, 'comp')] = np.abs(ctl[f'{name}_completeness'].values
                                        - ctl.true_completeness.values)
        ctl_ae[(name, 'cont')] = np.abs(ctl[f'{name}_contamination'].values
                                        - ctl.true_contamination.values)
    print('  control deltas vs V5 (cost of the smaller training pool alone):')
    ctl_delta = {}
    for name in ['familyHO', 'phylumHO']:
        for m in ['comp', 'cont']:
            d = float(ctl_ae[(name, m)].mean() - ctl_ae[('V5', m)].mean())
            ctl_delta[f'{name}_{m}'] = round(d, 4)
            print(f'    {name:<9} {m}  Delta = {d:+.3f} pp MAE')

    # ---------------- the deliverable ------------------------------------
    res = []
    for g in groups:
        if g == C.CONTROL_GROUP:
            continue
        sub = full[full.group == g]
        cl_g = sub.dominant_accession.values
        row = {'group': g,
               'parent_phylum': C.PANEL_GROUPS[g]['parent_phylum'],
               'families_held_out': ','.join(sorted(C.PANEL_GROUPS[g]['taxa'])),
               'n': len(sub), 'n_refs': int(sub.dominant_accession.nunique()),
               'median_Mbp': round(float(sub.dominant_genome_size.median() / 1e6), 3)}
        for metric, col in [('comp', 'completeness'), ('cont', 'contamination')]:
            t = sub[f'true_{col}'].values
            ae = {n: np.abs(sub[f'{n}_{col}'].values - t) for n in models}
            d = did_and_attenuation(
                ae['familyHO'], ae['phylumHO'], ae['V5'], cl_g,
                ctl_ae[('familyHO', metric)], ctl_ae[('phylumHO', metric)],
                ctl_ae[('V5', metric)], cl_c,
                seed=BOOT_SEED + stable_hash(g + metric) % 100000)
            for k, v in d.items():
                row[f'{metric}_{k}'] = v
        res.append(row)
    rd = pd.DataFrame(res)
    q = benjamini_hochberg(list(rd.comp_p_family) + list(rd.cont_p_family))
    rd['comp_q_family_bh'] = np.round(q[:len(rd)], 5)
    rd['cont_q_family_bh'] = np.round(q[len(rd):], 5)
    q2 = benjamini_hochberg(list(rd.comp_p_attenuation) + list(rd.cont_p_attenuation))
    rd['comp_q_attenuation_bh'] = np.round(q2[:len(rd)], 5)
    rd['cont_q_attenuation_bh'] = np.round(q2[len(rd):], 5)
    rd.to_csv(out_dir / 'family_vs_phylum_did.tsv', sep='\t', index=False)

    print('\n' + '=' * 78)
    print('DELIVERABLE - family-level vs phylum-level lineage novelty, '
          'PAIRED on identical samples')
    print('  DiD > 0 : the model is worse specifically because the lineage was absent')
    print('  attenuation = DiD_phylum - DiD_family; > 0 means keeping the parent')
    print('                phylum in training rescues part of the loss')
    print('=' * 78)
    base = ['group', 'parent_phylum', 'n_refs', '{m}_did_family', '{m}_ci95_family',
            '{m}_q_family_bh', '{m}_did_phylum', '{m}_attenuation',
            '{m}_ci95_attenuation', '{m}_pct_of_phylum_effect',
            '{m}_q_attenuation_bh']
    for metric, label in [('comp', 'COMPLETENESS'), ('cont', 'CONTAMINATION')]:
        cols = [c.format(m=metric) for c in base]
        print(f'\n{label}')
        print(rd[[c for c in cols if c in rd.columns]].to_string(index=False))

    # ---------------- cross-experiment context table ---------------------
    pdid = PHYLUM_RESULTS / 'lineage_novelty_effect_did.tsv'
    fdid = out_dir / 'lineage_novelty_effect_did.tsv'
    if pdid.exists() and fdid.exists():
        pp = pd.read_csv(pdid, sep='\t').rename(columns={'group': 'phylum_group'})
        ff = pd.read_csv(fdid, sep='\t').rename(columns={'group': 'family_group'})
        ff['parent_phylum'] = ff.family_group.map(
            lambda g: C.PANEL_GROUPS[g]['parent_phylum'])
        pp['parent_phylum'] = pp.phylum_group
        cross = ff.merge(pp, on='parent_phylum', how='left',
                         suffixes=('_familyexp', '_phylumexp'))
        cross.to_csv(out_dir / 'cross_experiment_did_summary.tsv',
                     sep='\t', index=False)
        print('\nCROSS-EXPERIMENT CONTEXT (each experiment\'s OWN DiD, own eval sets '
              'and own control).\nThe paired table above is the primary comparison; '
              'this one is what each\nexperiment reports standalone.')
        cc = [c for c in ['family_group', 'parent_phylum', 'comp_did_familyexp',
                          'comp_did_phylumexp', 'cont_did_familyexp',
                          'cont_did_phylumexp'] if c in cross.columns]
        print(cross[cc].to_string(index=False))

    # ---------------- report ---------------------------------------------
    panel = json.loads(C.PANEL_JSON.read_text())
    comp_cols = [c.format(m='comp') for c in base]
    cont_cols = [c.format(m='cont') for c in base]
    lines = [
        '# WS1.9 - leave-FAMILY-out holdout: does MAGICC generalize to a novel '
        'family inside a seen phylum?', '',
        '**Holdout models are validation artifacts only. The released MAGICC model '
        '(`models/magicc_v5.onnx`) remains trained on all data.**', '',
        '## Design', '',
        f'- Panel: {len(C.PANEL_TAXA)} families in {len(C.PANEL_GROUPS)} evaluation '
        f'groups, drawn from {", ".join(C.PANEL_PARENT_PHYLA)}.',
        f'- **Every held-out family\'s parent phylum remains in training**: all '
        f'{panel["surviving"]["train_phyla"]} phyla survive; phyla eliminated: '
        f'{panel["surviving"]["phyla_eliminated"] or "none"}.',
        f'- Training pool removed: {panel["n_removed"]["train"]:,}/79,948 = '
        f'**{panel["pct_removed"]["train"]}%** (WS1.6 phylum panel removed 19.64%).',
        f'- Reduced-genome pool: V5 definition used verbatim, no adaptation needed '
        f'({panel["reduced_genome_pool"]["pct_surviving"]}% survives).',
        '- Training data matched to V5 exactly (1M/100k/100k; 15/15/30/30/5/5%), so '
        'there is no sample-size confound. Seed 42.',
        '- Panel families excluded from BOTH the dominant and the contaminant pool; '
        'evaluation dominants come from the held-out TEST split.', '',
        '## Primary comparison (paired, identical samples)', '',
        'All three models score the same genomes:', '',
        '| model | saw the family? | saw the phylum? |', '|---|---|---|',
        '| production V5 | yes | yes |',
        '| family-holdout (WS1.9) | **no** | yes |',
        '| phylum-holdout (WS1.6) | **no** | **no** |', '',
        '### Completeness', '', '```',
        rd[[c for c in comp_cols if c in rd.columns]].to_string(index=False),
        '```', '', '### Contamination', '', '```',
        rd[[c for c in cont_cols if c in rd.columns]].to_string(index=False),
        '```', '',
        f'Common control: {len(ctl)}/{len(ctl_all)} in_distribution samples '
        f'({ctl.dominant_accession.nunique()} references), restricted to dominant '
        f'phyla outside the WS1.6 panel so the control is valid for both holdout '
        f'models. Control deltas vs V5: {ctl_delta}.', '',
        '## Files', '',
        '- `family_vs_phylum_did.tsv` - the deliverable',
        '- `three_model_head_to_head.tsv`',
        '- `cross_experiment_did_summary.tsv`',
        '- `lineage_novelty_effect_did.tsv` - WS1.9 standalone DiD',
        '- `head_to_head_by_group.tsv`, `stratified_error_by_band.tsv`, '
        '`mimag_threshold_by_group.tsv`, `mimag_confusion_by_group.tsv`, '
        '`per_reference_errors.tsv`, `sub_phylum_breakdown.tsv`',
    ]
    (out_dir / 'family_vs_phylum_report.md').write_text('\n'.join(lines) + '\n')

    summary = {
        'generated': pd.Timestamp.now().isoformat(),
        'estimator': ('paired difference-in-differences on identical samples; '
                      'cluster bootstrap over reference genomes, 2000 iterations, '
                      'BH-corrected across groups x 2 metrics'),
        'r2_convention': 'coefficient of determination (1 - SSres/SStot)',
        'models': {'V5': str(C.V5_ONNX), 'familyHO': str(C.HOLDOUT_ONNX),
                   'phylumHO': str(PHYLUM_ONNX)},
        'holdout_model_status': ('VALIDATION ARTIFACTS ONLY - the released MAGICC '
                                 'model remains trained on all data'),
        'common_control': {'n_samples': int(len(ctl)),
                           'n_refs': int(ctl.dominant_accession.nunique()),
                           'n_dropped': int(len(ctl_all) - len(ctl)),
                           'restriction': 'dominant phylum outside the WS1.6 panel',
                           'deltas_vs_v5': ctl_delta},
        'family_vs_phylum': rd.to_dict('records'),
        'three_model_head_to_head': h2h.to_dict('records'),
    }
    (out_dir / 'family_vs_phylum_summary.json').write_text(
        json.dumps(summary, indent=2, default=str))
    print(f'\nwrote {out_dir}/family_vs_phylum_did.tsv')
    print(f'wrote {out_dir}/family_vs_phylum_report.md')
    print('COMPARISON COMPLETE')


if __name__ == '__main__':
    main()
