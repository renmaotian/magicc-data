#!/usr/bin/env python3
"""
WS11.G step 7 - assemble results/revision/holdout_genus/WS11_G_REPORT.md from the
artifacts the pipeline produced. Reads only; computes nothing new.

Usage
  MAGICC_HOLDOUT_LEVEL=genus python scripts/225_ws11g_report.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_lib import config as C

R = C.RESULTS_DIR


def _tsv(name):
    p = R / name
    return pd.read_csv(p, sep='\t') if p.exists() else None


def _json(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_ci(v):
    if isinstance(v, str) and v.startswith('['):
        try:
            a, b = json.loads(v)
            return f'[{a:.2f}, {b:.2f}]'
        except Exception:
            return v
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f'[{v[0]:.2f}, {v[1]:.2f}]'
    return 'n/a'


def main():
    assert C.PANEL_LEVEL == 'genus', 'run with MAGICC_HOLDOUT_LEVEL=genus'
    panel = _json('eda_panel_summary.json')
    did = _tsv('lineage_novelty_effect_did.tsv')
    h2h = _tsv('head_to_head_by_group.tsv')
    ladder = _tsv('genus_vs_family_vs_phylum_did.tsv')
    four = _tsv('four_model_head_to_head.tsv')
    detail = _tsv('panel_genus_detail.tsv')
    kmer = _json('kmer_reselection_summary.json')
    train = _json('training_summary.json')
    dry = None
    p = R / 'dryrun_v5_vs_v5/lineage_novelty_effect_did.tsv'
    if p.exists():
        d = pd.read_csv(p, sep='\t')
        cols = [c for c in d.columns if c.endswith(('_did', '_delta'))]
        dry = {'n_cols': len(cols),
               'max_abs': float(d[cols].abs().to_numpy().max())}

    L = []
    A = L.append
    A('# WS11.G - leave-GENUS-out retraining')
    A('')
    A(f'Generated {pd.Timestamp.now().isoformat()}')
    A('')
    A('**Holdout models are validation artefacts only.** The released MAGICC model '
      '(`models/magicc_v5.onnx`) remains trained on all data and was not touched.')
    A('')
    A('## Why this experiment exists')
    A('')
    A('Reviewer 2 asked for taxonomic holdout at genus, family or phylum level and '
      'Reviewer 1 asked for held-out evaluation by genome, species, genus and '
      'phylum. Phylum (WS1.6) and family (WS1.9) were run; genus was declined with '
      'the argument that a genus-level holdout inside a represented family would be '
      'expected to fall below the family-level cost. That argument assumes '
      'degradation is monotone across ranks, and WS1.9\'s own Helicobacteraceae '
      'cell contradicts it (family-level novelty cost as much as phylum-level '
      'novelty: attenuation -0.39 pp, p = 0.228). This experiment measures the '
      'genus level instead of arguing about it. **Monotonicity across ranks is not '
      'assumed anywhere in this report.**')
    A('')

    # ---------------- panel -------------------------------------------------
    if panel:
        A('## The panel')
        A('')
        A(f'{len(panel["panel_taxa"])} genera in '
          f'{len(panel["panel_parent_families"])} families across '
          f'{len(panel["panel_parent_phyla"])} phyla. '
          f'{panel["n_removed"]["train"]:,} of 79,948 training genomes removed '
          f'({panel["pct_removed"]["train"]:.2f} %; WS1.9 family panel 6.95 %, '
          f'WS1.6 phylum panel 19.64 %).')
        A('')
        A('| group | genera | parent family | parent phylum | train removed | '
          'test refs | parent family retained | parent phylum retained |')
        A('|---|---|---|---|---|---|---|---|')
        for g, d in panel['panel_groups'].items():
            A(f'| {g} | {len(d["taxa"])} | {d["parent_family"]} | '
              f'{d["parent_phylum"]} | {d["train"]} | {d["test"]} | '
              f'{d["parent_family_retained_pct"]:.1f} % | '
              f'{d["parent_phylum_retained_pct"]:.1f} % |')
        A('')
        A('### Constraint verification (numerical, not asserted)')
        A('')
        s = panel['surviving']
        A(f'- Phyla with training genomes after removal: **{s["train_phyla"]}** of '
          f'110; eliminated: **{s["phyla_eliminated"] or "NONE"}**.')
        A(f'- Families with training genomes after removal: '
          f'**{s["train_families"]}** of 1,620; eliminated: '
          f'**{s["families_eliminated"] or "NONE"}**.')
        A(f'- Genera: 5,648 -> {s["train_genera"]}. Surviving training pool '
          f'{s["train"]:,} genomes.')
        rp = panel['reduced_genome_pool']
        A(f'- Reduced-genome pool under V5\'s own definition survives at '
          f'{rp["n_surviving"]}/{rp["n_v5_pool"]} = {rp["pct_surviving"]} % '
          f'(median {rp["median_Mbp_surviving"]} Mbp vs V5 '
          f'{rp["median_Mbp_v5"]} Mbp), so **no adaptation was applied** - '
          f'WS1.6 needed a redefined pool, WS1.9 kept 63.5 %.')
        A(f'- Ladder completeness: every panel genus lies inside a family WS1.9 '
          f'held out and a phylum WS1.6 held out '
          f'(`ladder_complete = {panel["ladder_complete"]}`).')
        A('')

    # ---------------- design validity ---------------------------------------
    A('## Design validity')
    A('')
    if dry:
        verdict = ('PASS - exactly zero' if dry['max_abs'] == 0.0
                   else f'FAIL - {dry["max_abs"]:.3e}')
        A(f'- **Dry run, production V5 substituted for both models:** max '
          f'|difference| over {dry["n_cols"]} numeric DiD/delta columns = '
          f'**{dry["max_abs"]:.3e}** -> **{verdict}**. '
          f'(`dryrun_v5_vs_v5/`)')
    if h2h is not None and 'group' in h2h.columns:
        c = h2h[h2h.group == C.CONTROL_GROUP]
        if len(c):
            r = c.iloc[0]
            for m, lab in [('comp', 'completeness'), ('cont', 'contamination')]:
                if f'd_{m}_mae' in h2h.columns:
                    A(f'- **`in_distribution` control, holdout vs V5 dMAE '
                      f'({lab}):** **{float(r[f"d_{m}_mae"]):+.3f} pp** '
                      f'(holdout {float(r[f"HO_{m}_mae"]):.3f} vs V5 '
                      f'{float(r[f"V5_{m}_mae"]):.3f}).')
            A('  WS1.6\'s completeness control delta was -0.047 and WS1.9\'s '
              '+0.029. A control delta of ~0 means removing the panel genomes had '
              'no measurable effect on retained lineages, so every measured '
              'degradation is attributable to lineage novelty alone. **The design '
              'is only valid if this holds.**')
    A('- **Level-switch reproduction:** adding `genus` to `MAGICC_HOLDOUT_LEVEL` '
      'changes nothing at phylum or family level - every value the pipeline reads '
      'out of `config.py`, including the SHA-256 of the reference-genome accession '
      'list `select_refs()` draws for every group, is identical under the '
      'pre-WS11.G config and the current one '
      '(`level_switch_reproduction.json`, script 224).')
    A('- **Frozen library:** the ten non-config files in `scripts/holdout_lib/` are '
      'byte-identical to the `FROZEN_SHA256_WS1.9.txt` record; only `config.py` '
      'changed (`FROZEN_SHA256_WS11.G.txt`).')
    if train:
        A(f'- **Training:** seed 42, '
          f'{train.get("best_epoch", "?")} best epoch, '
          f'{train.get("elapsed_hours", train.get("elapsed_h", "?"))} h.')
    A('')

    # ---------------- primary result ----------------------------------------
    if did is not None:
        A('## Primary result - difference-in-differences')
        A('')
        A('`DiD = (MAE_holdout - MAE_V5)_held-out-group - '
          '(MAE_holdout - MAE_V5)_in-distribution-control`, both models scoring '
          'identical samples; 95 % cluster-bootstrap CIs over reference genomes '
          '(2,000 resamples), two-sided paired tests, Benjamini-Hochberg corrected '
          'across groups x metrics.')
        A('')
        if 'comp_did' in did.columns:
            A('| group | comp DiD | 95 % CI | q(BH) | cont DiD | 95 % CI | q(BH) |')
            A('|---|---|---|---|---|---|---|')
            for _, r in did.sort_values('comp_did').iterrows():
                A(f'| {r["group"]} | {r["comp_did"]:+.2f} | '
                  f'{fmt_ci(r.get("comp_did_ci95"))} | '
                  f'{r.get("comp_did_q_bh", float("nan")):.4f} | '
                  f'{r["cont_did"]:+.2f} | {fmt_ci(r.get("cont_did_ci95"))} | '
                  f'{r.get("cont_did_q_bh", float("nan")):.4f} |')
            A('')
            A('`*_delta` columns in the TSV are the within-group '
              '(MAE_holdout - MAE_V5) differences before the control is '
              'subtracted; the DiD is the estimator.')
            A('')

    # ---------------- head to head ------------------------------------------
    if h2h is not None:
        A('## Head-to-head raw MAEs and signed biases')
        A('')
        A('Raw cross-group MAE comparisons are confounded (registry W17, trap T7) '
          'and are reported for completeness only; the DiD above is the estimator.')
        A('')
        A('```')
        A(h2h.to_string(index=False))
        A('```')
        A('')

    # ---------------- ladder -------------------------------------------------
    if ladder is not None:
        A('## The genus -> family -> phylum ladder on identical genomes')
        A('')
        A('All four models (V5, genusHO, familyHO, phylumHO) score the same '
          'evaluation genomes against a common control restricted to dominants '
          'outside the WS1.6 phylum panel. Attenuation = DiD_deeper - '
          'DiD_shallower; a value near zero means the shallower novelty is as '
          'damaging as the deeper one, and a negative value means it is worse.')
        A('')
        for metric, label in [('comp', 'Completeness'), ('cont', 'Contamination')]:
            A(f'### {label} (pp of MAE)')
            A('')
            A('| group | n refs | DiD genus | 95 % CI | q(BH) | DiD family | '
              'DiD phylum | att. family-vs-genus | att. phylum-vs-genus |')
            A('|---|---|---|---|---|---|---|---|---|')
            for _, r in ladder.iterrows():
                A(f'| {r["group"]} | {r["n_refs"]} | '
                  f'{r[f"{metric}_did_genus"]:+.2f} | '
                  f'{fmt_ci(r.get(f"{metric}_ci95_genus"))} | '
                  f'{r.get(f"{metric}_q_did_genus_bh", float("nan")):.4f} | '
                  f'{r[f"{metric}_did_family"]:+.2f} | '
                  f'{r[f"{metric}_did_phylum"]:+.2f} | '
                  f'{r[f"{metric}_attenuation_family_vs_genus"]:+.2f} | '
                  f'{r[f"{metric}_attenuation_phylum_vs_genus"]:+.2f} |')
            A('')
    if four is not None:
        A('### Four models, raw MAE on identical samples')
        A('')
        A('```')
        A(four.to_string(index=False))
        A('```')
        A('')

    # ---------------- k-mer control -----------------------------------------
    if kmer:
        A('## k-mer feature-selection leakage control')
        A('')
        for dom, d in (kmer.get('domains') or {}).items():
            A(f'- **{dom}**: {d.get("n_panel", "?")} of {d.get("n_reps", "?")} '
              f'feature-selection representatives belong to panel genera.')
        A('')
    elif panel:
        k = panel.get('kmer_selection_leakage', {})
        if k:
            A('## k-mer feature-selection leakage control')
            A('')
            for dom, d in k.items():
                A(f'- **{dom}**: {d["n_panel"]}/{d["n_reps"]} = {d["pct_panel"]} % '
                  f'of feature-selection representatives belong to panel genera '
                  f'(phylum panel: 18.7 % bacterial / 42.6 % archaeal).')
            A('')

    if detail is not None:
        A('## Per-genus panel detail')
        A('')
        A('```')
        A(detail.to_string(index=False))
        A('```')
        A('')

    A('## Determinism and provenance notes (for the supplementary note)')
    A('')
    A('1. **WS1.6 subsampled groups do not re-derive from the current code '
      '(defect D2).** WS1.6 drew evaluation references with `abs(hash(group))`, '
      'which Python salts per process; the defect was found and fixed in WS1.9 by '
      'switching to CRC-32 `stable_hash`. Re-running selection today reproduces '
      'WS1.6\'s three groups that use ALL available references exactly '
      '(Bacteroidota_A, Halobacteriota, DPANN) and not the three that subsample '
      '(overlap 8-67 of 100). **WS1.6 remains fully auditable because the '
      'references it actually drew are recorded verbatim in each group\'s '
      '`metadata.tsv`.** WS1.9 and WS11.G use CRC-32 and re-derive exactly: all '
      'six WS1.9 panel groups reproduce 100 %.')
    A('2. **`in_distribution` re-derives 92 of 100 references at family level.** '
      'The sqrt-proportional allocator breaks a largest-remainder tie between two '
      'equal-count single-genome phyla (Zhuqueibacterota vs Zixibacteria), and '
      'swapping one stratum reorders every downstream `.sample()` draw. The '
      'allocation differs by exactly one genome. Pre-existing; affects no reported '
      'number, because each run\'s own control is the one used in its own DiD.')
    A('3. **Synthetic training data is deterministic in content, not in row '
      'order.** Three independent 384-sample pilot builds at genus level produced '
      'an identical multiset of samples - same dominant genomes, same labels, same '
      'sample types, same composition - but different row order within a batch, '
      'because the top-up retry loop consumes `pool.imap_unordered(...)` and stops '
      'as soon as the batch is full, so which completed samples land in the batch '
      'depends on worker scheduling. **This was deliberately NOT "fixed".** Sorting '
      'the results would change what the phylum and family levels do, and WS1.6 and '
      'WS1.9 are completed experiments running on this same shared code path. It '
      'affects no reported number: the multiset defines the training distribution, '
      'and the training DataLoader shuffles it under its own seed.')
    A('')
    A('## Files')
    A('')
    for f in sorted(R.glob('*')):
        if f.is_file():
            A(f'- `{f.relative_to(C.PROJECT_ROOT)}`')
    A('')

    out = R / 'WS11_G_REPORT.md'
    out.write_text('\n'.join(L) + '\n')
    print(f'wrote {out} ({len(L)} lines)')


if __name__ == '__main__':
    main()
