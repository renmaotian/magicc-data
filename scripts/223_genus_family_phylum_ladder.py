#!/usr/bin/env python3
"""
WS11.G step 6 - the GENUS -> FAMILY -> PHYLUM ladder on identical genomes.

WHY THIS IS THE MOST VALUABLE OUTPUT OF WS11.G
  Every genus in the WS11.G panel was drawn from one of the six families WS1.9
  held out, and each of those families sits inside one of the phyla WS1.6 held
  out (verified by script 220, section 7). So the SAME evaluation genome is

    production V5   models/magicc_v5.onnx              saw genus, family, phylum
    genusHO         models/magicc_holdout_genus.onnx   never saw the GENUS
    familyHO        models/magicc_holdout_family.onnx  never saw the FAMILY
    phylumHO        models/magicc_holdout_phylum.onnx  never saw the PHYLUM

  and three novelty effects are measured on identical samples, with identical
  contaminants, at identical true completeness/contamination - a paired ladder,
  not three experiments glued together.

    DiD_rank = (MAE_rankHO - MAE_V5)_group - (MAE_rankHO - MAE_V5)_control

  ATTENUATION between two rungs = DiD_deeper - DiD_shallower, e.g.
    attenuation_family_vs_genus  = DiD_family - DiD_genus
    attenuation_phylum_vs_family = DiD_phylum - DiD_family
  > 0  novelty costs LESS at the shallower rank (keeping the parent in training
       rescues part of the loss);
  ~ 0  the shallower novelty is as damaging as the deeper one;
  < 0  the shallower novelty is WORSE - which is not excluded a priori and is
       exactly why this experiment exists. MONOTONICITY ACROSS RANKS IS NOT
       ASSUMED ANYWHERE: WS1.9's Helicobacteraceae cell already showed
       family-level novelty costing as much as phylum-level novelty
       (attenuation -0.39 pp, p = 0.228).

  All four models score identical samples, so the estimator is immune to group
  composition and to the genome-SIZE-driven completeness bias of WS3.10 (shared
  by all four models on the same genome, therefore cancelled).

COMMON CONTROL (necessary, and reported)
  The WS11.G in_distribution control is drawn from non-panel-GENUS test genomes,
  some of which belong to families WS1.9 held out or phyla WS1.6 held out. Those
  samples are not in-distribution for the family/phylum holdout models. The
  common control is therefore restricted to control samples whose dominant
  phylum is outside the WS1.6 panel - which also puts them outside the WS1.9
  family panel and the WS11.G genus panel, since every panel family and genus
  lies inside a WS1.6 panel phylum. The retained n is reported.

Significance: paired two-sided cluster bootstrap over REFERENCE GENOMES (2,000
iterations), Benjamini-Hochberg corrected across all groups x 2 metrics.
Bootstrap seeding uses CRC-32 (stable_hash), never Python's salted hash().

Outputs (results/revision/holdout_genus/)
  genus_vs_family_vs_phylum_did.tsv   the deliverable: 3-rung DiD ladder
  four_model_head_to_head.tsv         raw MAE/bias/R2 for all four models
  ladder_per_sample.tsv.gz            every prediction from all four models
  ladder_report.md                    readable summary

Usage
  MAGICC_HOLDOUT_LEVEL=genus python scripts/223_genus_family_phylum_ladder.py
"""

import gzip
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
BOOT_SEED = 20260827
RANKS = ['genus', 'family', 'phylum']
MODEL_OF = {'genus': 'genusHO', 'family': 'familyHO', 'phylum': 'phylumHO'}


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


def ladder_boot(ae_g, ae_c, cl_g, cl_c, n_boot=N_BOOT, seed=BOOT_SEED):
    """Three DiDs and all pairwise attenuations from ONE cluster bootstrap.

    `ae_g` / `ae_c` map model name -> absolute-error vector on the group /
    control. Because all four models score the same samples, the SAME resampled
    cluster indices are used for every model within each bootstrap iteration, so
    the DiD differences (attenuations) are properly PAIRED quantities.
    """
    gg, gc = _clusters(cl_g), _clusters(cl_c)

    def _did(m):
        return ((ae_g[m].mean() - ae_g['V5'].mean()) -
                (ae_c[m].mean() - ae_c['V5'].mean()))

    obs = {r: float(_did(MODEL_OF[r])) for r in RANKS}
    out = {f'did_{r}': round(obs[r], 4) for r in RANKS}
    if len(gg) < 3 or len(gc) < 3:
        return {**out, 'n_clusters_group': len(gg), 'n_clusters_control': len(gc)}

    rng = np.random.default_rng(seed)
    draws = {r: np.empty(n_boot) for r in RANKS}
    for b in range(n_boot):
        ig = np.concatenate([gg[i] for i in rng.integers(0, len(gg), len(gg))])
        ic = np.concatenate([gc[i] for i in rng.integers(0, len(gc), len(gc))])
        v_g = ae_g['V5'][ig].mean()
        v_c = ae_c['V5'][ic].mean()
        for r in RANKS:
            m = MODEL_OF[r]
            draws[r][b] = (ae_g[m][ig].mean() - v_g) - (ae_c[m][ic].mean() - v_c)

    def _p(d, o):
        sh = d - o
        v = 2.0 * min(np.mean(sh >= o), np.mean(sh <= o))
        return float(min(1.0, max(v, 1.0 / n_boot)))

    for r in RANKS:
        d = draws[r]
        out[f'ci95_{r}'] = [round(float(np.percentile(d, 2.5)), 4),
                            round(float(np.percentile(d, 97.5)), 4)]
        out[f'p_{r}'] = _p(d, obs[r])

    for deep, shal in [('family', 'genus'), ('phylum', 'family'), ('phylum', 'genus')]:
        att = draws[deep] - draws[shal]
        o = obs[deep] - obs[shal]
        key = f'attenuation_{deep}_vs_{shal}'
        out[key] = round(o, 4)
        out[f'ci95_{key}'] = [round(float(np.percentile(att, 2.5)), 4),
                              round(float(np.percentile(att, 97.5)), 4)]
        out[f'p_{key}'] = _p(att, o)
        out[f'pct_of_{deep}_effect_remaining_at_{shal}'] = (
            round(100.0 * obs[shal] / obs[deep], 1)
            if abs(obs[deep]) > 1e-9 else None)
    out['n_clusters_group'] = len(gg)
    out['n_clusters_control'] = len(gc)
    return out


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
    assert C.PANEL_LEVEL == 'genus', 'run with MAGICC_HOLDOUT_LEVEL=genus'
    out_dir = C.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 78)
    print('WS11.G - GENUS vs FAMILY vs PHYLUM novelty on IDENTICAL genomes')
    print('=' * 78)

    need = [C.HOLDOUT_ONNX, C.HOLDOUT_NORM_PARAMS, C.FAMILY_ONNX,
            C.FAMILY_NORM_PARAMS, C.PHYLUM_ONNX, C.PHYLUM_NORM_PARAMS,
            C.V5_ONNX, C.V5_NORM_PARAMS]
    missing = [str(p) for p in need if not Path(p).exists()]
    if missing:
        sys.exit(f'missing model artifacts: {missing}')

    models = {
        'V5': load_model(C.V5_ONNX, C.V5_NORM_PARAMS),
        'genusHO': load_model(C.HOLDOUT_ONNX, C.HOLDOUT_NORM_PARAMS),
        'familyHO': load_model(C.FAMILY_ONNX, C.FAMILY_NORM_PARAMS),
        'phylumHO': load_model(C.PHYLUM_ONNX, C.PHYLUM_NORM_PARAMS),
    }
    print('models (each normalizes the shared RAW counts with ITS OWN parameters):')
    print(f'  V5        {C.V5_ONNX}')
    print(f'  genusHO   {C.HOLDOUT_ONNX}   (WS11.G; never saw the panel GENERA)')
    print(f'  familyHO  {C.FAMILY_ONNX}   (WS1.9; never saw those genera\' FAMILIES)')
    print(f'  phylumHO  {C.PHYLUM_ONNX}   (WS1.6; never saw those families\' PHYLA)')
    print('All three holdout models are VALIDATION ARTIFACTS ONLY; the released '
          'MAGICC model remains trained on all data.')

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
    with gzip.open(out_dir / 'ladder_per_sample.tsv.gz', 'wt') as f:
        full[[c for c in keep if c in full.columns]].to_csv(f, sep='\t', index=False)

    # ---------------- four-model head-to-head ----------------------------
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
    h2h.to_csv(out_dir / 'four_model_head_to_head.tsv', sep='\t', index=False)
    print('\n' + '=' * 78)
    print('FOUR MODELS ON IDENTICAL SAMPLES (R2 = coefficient of determination)')
    print('=' * 78)
    print(h2h[['group', 'n', 'n_refs', 'V5_comp_mae', 'genusHO_comp_mae',
               'familyHO_comp_mae', 'phylumHO_comp_mae']].to_string(index=False))
    print(h2h[['group', 'V5_cont_mae', 'genusHO_cont_mae', 'familyHO_cont_mae',
               'phylumHO_cont_mae']].to_string(index=False))

    # ---------------- common control -------------------------------------
    ctl_all = full[full.group == C.CONTROL_GROUP]
    ctl = ctl_all[~ctl_all.dominant_phylum.isin(C.WS16_PHYLUM_PANEL_TAXA)]
    print(f'\nCOMMON CONTROL: {len(ctl)}/{len(ctl_all)} in_distribution samples '
          f'({ctl.dominant_accession.nunique()} references) are in-distribution for '
          f'ALL THREE holdout models.')
    print(f'  dropped: {len(ctl_all) - len(ctl)} samples whose dominant phylum is in '
          f'the WS1.6 panel and is therefore NOT in-distribution for phylumHO.')
    if ctl.dominant_accession.nunique() < 3:
        sys.exit('common control has too few reference genomes')

    cl_c = ctl.dominant_accession.values
    ctl_ae = {}
    for metric, col in [('comp', 'completeness'), ('cont', 'contamination')]:
        ctl_ae[metric] = {n: np.abs(ctl[f'{n}_{col}'].values
                                    - ctl[f'true_{col}'].values) for n in models}
    print('  control deltas vs V5 (cost of the smaller training pool alone):')
    ctl_delta = {}
    for name in ['genusHO', 'familyHO', 'phylumHO']:
        for m in ['comp', 'cont']:
            d = float(ctl_ae[m][name].mean() - ctl_ae[m]['V5'].mean())
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
               'parent_family': C.PANEL_GROUPS[g]['parent_family'],
               'genera_held_out': ','.join(sorted(C.PANEL_GROUPS[g]['taxa'])),
               'n': len(sub), 'n_refs': int(sub.dominant_accession.nunique()),
               'median_Mbp': round(float(sub.dominant_genome_size.median() / 1e6), 3)}
        for metric, col in [('comp', 'completeness'), ('cont', 'contamination')]:
            t = sub[f'true_{col}'].values
            ae_g = {n: np.abs(sub[f'{n}_{col}'].values - t) for n in models}
            d = ladder_boot(ae_g, ctl_ae[metric], cl_g, cl_c,
                            seed=BOOT_SEED + stable_hash(g + metric) % 100000)
            for k, v in d.items():
                row[f'{metric}_{k}'] = v
        res.append(row)
    rd = pd.DataFrame(res)
    n = len(rd)
    for key in ['did_genus', 'attenuation_family_vs_genus',
                'attenuation_phylum_vs_genus']:
        q = benjamini_hochberg(list(rd[f'comp_p_{key}']) + list(rd[f'cont_p_{key}']))
        rd[f'comp_q_{key}_bh'] = np.round(q[:n], 5)
        rd[f'cont_q_{key}_bh'] = np.round(q[n:], 5)
    rd.to_csv(out_dir / 'genus_vs_family_vs_phylum_did.tsv', sep='\t', index=False)

    print('\n' + '=' * 78)
    print('DELIVERABLE - the three-rung novelty ladder, all four models on the')
    print('SAME evaluation genomes against a common control')
    print('=' * 78)
    for metric, label in [('comp', 'COMPLETENESS'), ('cont', 'CONTAMINATION')]:
        print(f'\n{label} difference-in-differences (pp of MAE)')
        t = rd[['group', 'n_refs', f'{metric}_did_genus', f'{metric}_ci95_genus',
                f'{metric}_did_family', f'{metric}_did_phylum',
                f'{metric}_attenuation_family_vs_genus',
                f'{metric}_attenuation_phylum_vs_genus',
                f'{metric}_q_did_genus_bh']].copy()
        t.columns = ['group', 'n_refs', 'DiD_genus', 'CI95_genus', 'DiD_family',
                     'DiD_phylum', 'att_fam_vs_gen', 'att_phy_vs_gen', 'q_genus']
        print(t.to_string(index=False))

    # ---------------- readable report ------------------------------------
    lines = [
        '# WS11.G - genus vs family vs phylum novelty ladder',
        '',
        f'Generated {pd.Timestamp.now().isoformat()}',
        '',
        'Four models score the SAME evaluation genomes: production V5, the WS11.G',
        'genus-holdout model, the WS1.9 family-holdout model and the WS1.6',
        'phylum-holdout model. Every panel genus lies inside a family WS1.9 held',
        'out and a phylum WS1.6 held out, so the ladder is paired on identical',
        'samples rather than assembled from separate experiments.',
        '',
        '**Monotonicity across ranks is not assumed.** It is measured. WS1.9 already',
        'produced a counterexample (Helicobacteraceae: family-level novelty as costly',
        'as phylum-level, attenuation -0.39 pp, p = 0.228).',
        '',
        f'Common control: {len(ctl)}/{len(ctl_all)} control samples '
        f'({ctl.dominant_accession.nunique()} references) are in-distribution for all',
        'three holdout models (dominant phylum outside the WS1.6 panel).',
        '',
        '## Control deltas vs V5 (cost of the smaller training pool alone)',
        '',
        '| model | completeness dMAE | contamination dMAE |',
        '|---|---|---|',
    ]
    for name in ['genusHO', 'familyHO', 'phylumHO']:
        lines.append(f'| {name} | {ctl_delta[f"{name}_comp"]:+.3f} | '
                     f'{ctl_delta[f"{name}_cont"]:+.3f} |')
    for metric, label in [('comp', 'Completeness'), ('cont', 'Contamination')]:
        lines += ['', f'## {label} DiD ladder (pp of MAE)', '',
                  '| group | n_refs | DiD genus [95% CI] | q(BH) | DiD family | '
                  'DiD phylum | attenuation family-vs-genus | attenuation '
                  'phylum-vs-genus |', '|---|---|---|---|---|---|---|---|']
        for _, r in rd.iterrows():
            ci = r[f'{metric}_ci95_genus']
            ci = f'[{ci[0]:.2f}, {ci[1]:.2f}]' if isinstance(ci, list) else 'n/a'
            lines.append(
                f'| {r["group"]} | {r["n_refs"]} | '
                f'{r[f"{metric}_did_genus"]:+.2f} {ci} | '
                f'{r[f"{metric}_q_did_genus_bh"]:.4f} | '
                f'{r[f"{metric}_did_family"]:+.2f} | '
                f'{r[f"{metric}_did_phylum"]:+.2f} | '
                f'{r[f"{metric}_attenuation_family_vs_genus"]:+.2f} | '
                f'{r[f"{metric}_attenuation_phylum_vs_genus"]:+.2f} |')
    (out_dir / 'ladder_report.md').write_text('\n'.join(lines) + '\n')
    print(f'\nwrote {out_dir / "genus_vs_family_vs_phylum_did.tsv"}')
    print(f'wrote {out_dir / "four_model_head_to_head.tsv"}')
    print(f'wrote {out_dir / "ladder_report.md"}')
    print('\nLADDER COMPLETE')


if __name__ == '__main__':
    main()
