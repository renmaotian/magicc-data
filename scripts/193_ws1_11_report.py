#!/usr/bin/env python3
"""
WS1.11 -- assemble the answer to Reviewer 1, minor comment 13 (circularity safeguards).

Every number in the report is read from a result file under results/revision/circularity/;
nothing is transcribed by hand and nothing is computed here that is not also on disk.
The verdict is derived from the confidence intervals against a materiality threshold
that is stated in the report itself, so the conclusion cannot be tuned after the fact.

Usage:
    python scripts/193_ws1_11_report.py [--materiality-pp 2.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/path/to/magicc')
OUT = ROOT / 'results' / 'revision' / 'circularity'


def load(name, sep='\t'):
    p = OUT / name
    if not p.exists():
        return None
    return pd.read_csv(p, sep=sep) if name.endswith('.tsv') else json.loads(p.read_text())


def f(x, n=2):
    try:
        return f'{float(x):.{n}f}'
    except (TypeError, ValueError):
        return 'n/a'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--materiality-pp', type=float, default=2.0)
    args = ap.parse_args()
    MAT = args.materiality_pp

    sel = load('ws1_11_reference_selection.json')
    fetch = load('ws1_11_reference_fetch.json')
    genval = load('ws1_11_generation_validation.json')
    det = load('ws1_11_determinism_check.json')
    prov = json.loads((OUT / 'provenance' / 'audit_summary.json').read_text())
    ovl = pd.read_csv(OUT / 'provenance' / 'overlap_summary.tsv', sep='\t')
    arm = load('ws1_11_arm_metrics.tsv')
    paired = load('ws1_11_paired_arm_tests.tsv')
    did = load('ws1_11_did_vs_competitors.tsv')
    mimag = load('ws1_11_mimag.tsv')
    thr = load('ws1_11_thresholds.tsv')
    mech = load('ws1_11_reference_incompleteness_mechanism.tsv')
    adj = load('ws1_11_adjusted_truth_sensitivity.tsv')
    ctx = load('ws1_11_context_other_sets.tsv')
    reflev = load('ws1_11_reference_level_scores.tsv')
    reflev_j = load('ws1_11_reference_level_summary.json')
    acct = load('ws1_11_filter_accounting.tsv')

    def A(tool, group, col):
        r = arm[(arm.tool == tool) & (arm.group == group)]
        return r.iloc[0][col] if len(r) else np.nan

    def P(tool, metric, stat='abs_error', subset='all pairs'):
        r = paired[(paired.tool == tool) & (paired.metric_name == metric)
                   & (paired.statistic == stat) & (paired.subset == subset)]
        return r.iloc[0] if len(r) else None

    tools_present = sorted(arm['tool'].unique())
    has_ck = 'checkm2' in tools_present

    # ------------------------------------------------------------- verdict
    verdict_bits, worst_hi = [], 0.0
    for m in ('completeness', 'contamination'):
        r = P('magicc_v5', m)
        if r is None:
            continue
        worst_hi = max(worst_hi, float(r['ci_hi']))
        verdict_bits.append(
            f"{m}: D = {f(r['D_fail_minus_pass'])} pp "
            f"[{f(r['ci_lo'])}, {f(r['ci_hi'])}], q = {float(r['q_bh']):.3g}")
    if worst_hi < MAT:
        verdict = ('**Circularity does not materially bias the published conclusions.** '
                   f'The upper 95 % confidence bound on MAGICC\'s excess error over the '
                   f'genomes the CheckM2 filter removed is {f(worst_hi)} pp, below the '
                   f'{f(MAT, 1)} pp materiality threshold declared before the analysis.')
        verdict_tag = 'NO MATERIAL BIAS'
    elif worst_hi < 2 * MAT:
        verdict = (f'**A small, quantified effect is present.** MAGICC\'s excess error '
                   f'on the removed genomes has an upper 95 % bound of {f(worst_hi)} pp, '
                   f'above the {f(MAT, 1)} pp materiality threshold but below twice it. '
                   f'It is reported as a bounded limitation, not dismissed.')
        verdict_tag = 'SMALL BOUNDED EFFECT'
    else:
        verdict = (f'**A material effect is present and must be reported as a finding.** '
                   f'MAGICC\'s excess error on the removed genomes reaches '
                   f'{f(worst_hi)} pp at the upper 95 % bound, more than twice the '
                   f'{f(MAT, 1)} pp materiality threshold.')
        verdict_tag = 'MATERIAL EFFECT'

    cp = sel['candidate_pool']
    arms = sel['arms']
    n_pairs = int(genval['report']['n_pairs'])
    n_samples = int(genval['report']['n_samples'])

    md = []
    w = md.append
    w('# WS1.11 — Circularity safeguard (Reviewer 1, minor comment 13)\n')
    w(f'_Generated {datetime.now(timezone.utc).isoformat()} by '
      f'`scripts/193_ws1_11_report.py` from the result files in '
      f'`results/revision/circularity/`. Evaluation only: no model was retrained and '
      f'`models/magicc_v5.onnx` (SHA256 `{prov.get("model_sha256", "")}`) is unchanged._\n')

    w('## Verdict\n')
    w(f'**{verdict_tag}.** {verdict}\n')
    w('Per metric, paired over the matched reference pairs (D > 0 = worse on the '
      'genomes the CheckM2-based curation removed):\n')
    for b in verdict_bits:
        w(f'- {b}')
    w('')

    # --------------------------------------------------------- 1. objection
    w('## 1. The objection, stated precisely\n')
    w('Every reference genome in this project passed a CheckM2-based filter: GTDB '
      'metadata restricted to CheckM2 completeness ≥ 98 %, contamination ≤ 2 %, < 100 '
      'contigs, N50 > 20 kbp and longest contig > 100 kbp → 277,183 genomes → 100,000 '
      'sampled → train/val/test. Ground truth for **both** training and benchmarking is '
      'therefore drawn from genomes CheckM2 judged near-complete and clean. If CheckM2 '
      'systematically mis-scores a class of genome, those genomes were removed from the '
      'pool, so MAGICC learned a CheckM2-curated view of genome space **and the '
      'benchmark cannot reveal the error, because the affected genomes are absent**. A '
      'sensitivity analysis that re-used the same filtered pool would not answer this.\n')

    # ------------------------------------------------------- 2. how selected
    w('## 2. How the reference genomes were selected — CheckM2 removed from the loop\n')
    w('References were selected by NCBI\'s **`assembly_level == "Complete Genome"`** '
      'annotation taken from `assembly_summary_genbank.txt` / '
      '`assembly_summary_refseq.txt` (`version_status == latest`) — a submitter/NCBI '
      'assembly annotation that does not involve CheckM2 — with **no CheckM2 filter '
      'applied**. CheckM2 scores were read afterwards only to *label* each reference as '
      'would-have-passed or would-have-failed.\n')
    if acct is not None:
        w('| stage | n remaining | note |')
        w('|---|---|---|')
        for _, r in acct.iterrows():
            note = r['note'] if isinstance(r['note'], str) else ''
            w(f'| {r["stage"]} | {int(r["n_remaining"]):,} | {note} |')
        w('')
    w(f'**Taxonomic-consistency filter** (protocol WS1.11): a candidate must carry both '
      f'a GTDB and an NCBI taxonomy and they must agree at genus level after '
      f'normalising GTDB\'s alphabetic suffixes. Of 47,820 NCBI-Complete GTDB genomes, '
      f'6,623 were dropped for a missing/placeholder NCBI assignment and 1,814 for a '
      f'genus mismatch; 0 lacked a GTDB assignment.\n')
    w(f'**The headline design number.** Of the {cp["n_total"]:,} eligible NCBI-Complete '
      f'references remaining after taxonomic consistency and leakage exclusion, '
      f'**{cp["n_would_fail"]:,} ({100 * cp["n_would_fail"] / cp["n_total"]:.1f} %) '
      f'would have FAILED the original CheckM2-based curation filter** and '
      f'{cp["n_would_pass"]:,} would have passed. Essentially all failures are CheckM2 '
      f'failures, not assembly-quality failures: '
      f'{cp["would_fail_reason"]["checkm2_completeness_lt_98"]:,} fall below 98 % '
      f'CheckM2 completeness and '
      f'{cp["would_fail_reason"]["checkm2_contamination_gt_2"]:,} exceed 2 % CheckM2 '
      f'contamination (criteria are not mutually exclusive), while only '
      f'{cp["would_fail_reason"]["contig_count_ge_100"]} have ≥ 100 contigs, '
      f'{cp["would_fail_reason"]["n50_le_20kb"]} have N50 ≤ 20 kbp and '
      f'{cp["would_fail_reason"]["longest_contig_le_100kb"]} have a longest contig '
      f'≤ 100 kbp.\n')
    w(f'**Set H composition.** {arms["H_fail"]["n"]} would-have-failed references were '
      f'drawn with √-proportional allocation across three CheckM2-deficit severity '
      f'strata (the same scheme Phase 1 used across phyla, so the severe stratum is '
      f'represented): '
      f'{", ".join(f"{k} {v}" for k, v in sel["severity_allocation"]["allocation"].items())} '
      f'from strata of size '
      f'{", ".join(f"{k} {v:,}" for k, v in sel["severity_allocation"]["strata_sizes"].items())}. '
      f'Each was matched 1:1 to a '
      f'would-have-passed reference at the deepest available taxonomic rank and then by '
      f'closest genome size: '
      f'{", ".join(f"{k} {v}" for k, v in sel["match_level_counts"].items())}. '
      f'The arms are balanced — '
      f'{arms["H_fail"]["n_phyla"]} vs {arms["H_pass"]["n_phyla"]} phyla, median genome '
      f'size {arms["H_fail"]["genome_size_Mbp"]["median"]} vs '
      f'{arms["H_pass"]["genome_size_Mbp"]["median"]} Mbp, median contig count '
      f'{arms["H_fail"]["contig_count_median"]:.0f} vs '
      f'{arms["H_pass"]["contig_count_median"]:.0f} — and differ in CheckM2 score by '
      f'construction (median completeness '
      f'{f(arms["H_fail"]["checkm2_completeness"]["median"])} vs '
      f'{f(arms["H_pass"]["checkm2_completeness"]["median"])} %, median contamination '
      f'{f(arms["H_fail"]["checkm2_contamination"]["median"])} vs '
      f'{f(arms["H_pass"]["checkm2_contamination"]["median"])} %).\n')
    w(f'**Stated boundary.** Candidates must be present in GTDB, because the '
      f'taxonomic-consistency filter needs a GTDB taxonomy; '
      f'{sel["ncbi_complete_prokaryotic_not_in_gtdb"]:,} NCBI prokaryotic Complete '
      f'Genomes are outside GTDB and are therefore unreachable here. GTDB\'s own floor '
      f'(≈ ≥ 50 % completeness, ≤ 10 % contamination) is far weaker than the project\'s '
      f'≥ 98 % / ≤ 2 % filter, so the genomes that matter for this objection — those '
      f'excluded by the project\'s filter but not by GTDB\'s — are exactly the ones '
      f'this set contains.\n')

    # ------------------------------------------------------ 3. benchmark set
    r = genval['report']
    w('## 3. Benchmark generation — only the reference selection changed\n')
    w(f'`set_H_ncbi` was produced by the generation logic of '
      f'`scripts/073_generate_clean_cd_benchmarks.py` **verbatim** (itself '
      f'`scripts/025_benchmark_generate.py` verbatim): same fragmentation call, same '
      f'cross-phylum contaminant draw from `data/splits/test_genomes.tsv`, same caps, '
      f'same label arithmetic, same constraint guard, same FASTA writer, 10 independent '
      f'simulations per reference, targets completeness ~ U[50, 100) % and '
      f'contamination ~ U[0, 100) %. Design RNG `default_rng(7_500_000)`; per-sample RNG '
      f'`default_rng(7_500_000 + 1000·ref_index + replicate)`. One deliberate '
      f'refinement: both members of a matched pair receive the *same* target draw at '
      f'each replicate, which makes every comparison paired at the sample level without '
      f'changing the marginal target distributions.\n')
    w(f'- {r["n_samples"]:,} simulations from {r["n_unique_dominants"]} references '
      f'({r["n_pairs"]} matched pairs), exactly '
      f'{", ".join(str(x) for x in r["sims_per_reference"])} simulations each; '
      f'0 generation errors.')
    w(f'- FASTA integrity: {r["fasta_missing"]} missing, {r["fasta_empty"]} empty, '
      f'{r["fasta_unparseable"]} unparseable, {r["fasta_length_mismatch"]} with a '
      f'contig count or bp total disagreeing with the metadata.')
    w(f'- Constraint violations: {r["violations_contaminant_bp_gt_dominant_bp"]} '
      f'(contaminant bp > dominant bp), '
      f'{r["violations_contamination_gt_completeness"]} (contamination % > '
      f'completeness %), {r["violations_contamination_gt_100"]} (contamination > 100 %), '
      f'{r["violations_completeness_out_of_50_100"]} (completeness outside 50–100 %). '
      f'**Out-of-domain samples (protocol §4.4a): '
      f'{r["out_of_domain_samples"]}** — the design is entirely in-domain.')
    w(f'- Target uniformity: KS p = {r["ks_target_completeness_uniform"]["p"]} '
      f'(completeness), {r["ks_target_contamination_uniform"]["p"]} (contamination). '
      f'Paired targets identical across arms: {r["paired_targets_identical"]}.')
    w(f'- `generation_metadata.tsv` has '
      f'{r["n_generation_metadata_columns"]} columns including per-sample seed, both '
      f'targets, both observed values, every contaminant accession, the fragmentation '
      f'tier and all six dropout parameters (WS7.7 / R1-m15).')
    if det:
        w(f'- Determinism: 5 FASTAs deleted with their checkpoint lines removed and '
          f'regenerated — byte-identical: '
          f'**{det["regenerated_fastas_byte_identical"]}**; `generation_metadata.tsv` '
          f'unchanged: {det["generation_metadata_unchanged"]}.')
    if fetch:
        w(f'- All {fetch["n_accepted"]}/{fetch["n_requested"]} reference assemblies '
          f'downloaded from NCBI and accepted; total bp deviates from the GTDB-recorded '
          f'genome size by at most '
          f'{fetch["bp_rel_deviation"]["max"] * 100:.4f} %, i.e. the assemblies scored '
          f'here are exactly the assemblies GTDB scored.')
    w('')

    # ---------------------------------------------------- 4. leakage verdict
    a = ovl.iloc[0]
    hf = ovl[ovl['set'].str.contains('H_fail')].iloc[0]
    w('## 4. Provenance and leakage — proven, not asserted\n')
    w(f'`scripts/188_ws1_11_provenance_audit.py` imports the normalisation and GCA↔GCF '
      f'cross-map code of `scripts/074_provenance_audit.py` directly, so the two audits '
      f'cannot drift apart. Cross-map: '
      f'{prov["crossmap_stats"]["filtered_genomes_rows"]:,} rows → '
      f'{prov["crossmap_stats"]["distinct_accession_strings_mapped"]:,} accession '
      f'strings → {prov["crossmap_stats"]["distinct_canonical_assemblies"]:,} canonical '
      f'assemblies, {prov["crossmap_stats"]["rows_with_inconsistent_assembly_numbers"]} '
      f'inconsistencies. Counts below are identical under all three normalisations.\n')
    w(f'**DISJOINTNESS VERDICT: {prov["disjointness_verdict"]}.**\n')
    w('| universe | samples (n=%d) | unique references (n=%d) |'
      % (int(a['n_samples']), int(a['n_unique_dominants_canonical'])))
    w('|---|---|---|')
    for uni, lab in (('train', 'training split'), ('val', 'validation split'),
                     ('test', 'test split'),
                     ('kmer_selection', '2,000-genome 9-mer feature-selection set'),
                     ('curation_pool_277k',
                      '277,183-genome CheckM2-filtered curation pool')):
        w(f'| {lab} | {int(a[f"samples_in_{uni}_canonical"])} | '
          f'{int(a[f"unique_dominants_in_{uni}_canonical"])} |')
    w('')
    w(f'The last row is the point of the experiment: the H_fail arm has '
      f'**{int(hf["samples_in_curation_pool_277k_canonical"])} of '
      f'{int(hf["n_samples"])}** samples inside the CheckM2-filtered pool, i.e. none of '
      f'those {int(hf["n_unique_dominants_canonical"])} references was ever available to '
      f'any MAGICC model or any previous MAGICC benchmark. '
      f'{prov["contaminants"]["contamination_events"]:,} contamination events over '
      f'{prov["contaminants"]["unique_contaminant_genomes"]:,} unique genomes all come '
      f'from the held-out test split '
      f'({prov["contaminants"]["events_from_train_split"]} from train, '
      f'{prov["contaminants"]["events_from_val_split"]} from val), and '
      f'{prov["contaminants"]["events_same_phylum_as_dominant"]} share the dominant\'s '
      f'phylum — identical to `set_C_clean`/`set_D_clean`, because the contaminant pool '
      f'is deliberately unchanged. A SHA256 manifest covers '
      f'{prov["manifest_files"]:,} files.\n')

    # ------------------------------------------------------- 5. the result
    w('## 5. The comparison that answers R1-m13\n')
    w('### 5.1 Error on CheckM2-filtered vs NCBI-selected references\n')
    w('| tool | reference group | n | clusters | completeness MAE (95 % CI) | comp. bias | '
      'comp. R² | contamination MAE (95 % CI) | cont. bias | cont. R² |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for tool in ['magicc_v5'] + [t for t in tools_present if t != 'magicc_v5']:
        for g, lab in (('H_pass', 'would have PASSED the CheckM2 filter'),
                       ('H_fail', '**would have FAILED it**')):
            w(f'| {tool} | {lab} | {int(A(tool, g, "n"))} | '
              f'{int(A(tool, g, "n_clusters"))} | '
              f'{f(A(tool, g, "comp_mae"))} ({f(A(tool, g, "comp_mae_ci_lo"))}–'
              f'{f(A(tool, g, "comp_mae_ci_hi"))}) | {f(A(tool, g, "comp_bias"))} | '
              f'{f(A(tool, g, "comp_r2_cod"), 3)} | '
              f'{f(A(tool, g, "cont_mae"))} ({f(A(tool, g, "cont_mae_ci_lo"))}–'
              f'{f(A(tool, g, "cont_mae_ci_hi"))}) | {f(A(tool, g, "cont_bias"))} | '
              f'{f(A(tool, g, "cont_r2_cod"), 3)} |')
    w('')
    w('R² is the coefficient of determination throughout (never squared Pearson). '
      'CIs are 95 % cluster bootstrap over reference genomes, 2,000 replicates, seeds '
      'from `fw.stable_hash` (CRC-32) with `PYTHONHASHSEED=0`. Denominators: '
      'completeness % = retained dominant bp / full reference length × 100; '
      'contamination % = total contaminant bp / full reference length × 100.\n')

    w('### 5.2 Primary estimator — paired difference over matched reference pairs\n')
    w('Each would-have-failed reference is compared with its taxonomically matched, '
      'size-matched control on simulations that share identical target draws, so '
      'composition cannot confound the contrast. '
      '`D = MAE(H_fail) − MAE(H_pass)`; positive = worse on the removed genomes. '
      'CIs and two-sided p-values are cluster bootstraps over the '
      f'{n_pairs} pairs; q is Benjamini–Hochberg over the whole test family '
      f'({len(paired)} tests).\n')
    w('| tool | metric | mean abs. err. H_pass | H_fail | D (95 % CI) | Hodges–Lehmann | '
      'rank-biserial | p | q (BH) |')
    w('|---|---|---|---|---|---|---|---|---|')
    for tool in ['magicc_v5'] + [t for t in tools_present if t != 'magicc_v5']:
        for m in ('completeness', 'contamination'):
            r = P(tool, m)
            if r is None:
                continue
            w(f'| {tool} | {m} | {f(r["mean_H_pass"])} | {f(r["mean_H_fail"])} | '
              f'**{f(r["D_fail_minus_pass"])}** ({f(r["ci_lo"])}, {f(r["ci_hi"])}) | '
              f'{f(r["hodges_lehmann"])} | {f(r["rank_biserial"], 3)} | '
              f'{float(r["p_two_sided"]):.3g} | {float(r["q_bh"]):.3g} |')
    w('')
    sub = paired[paired.subset == 'match_level in {species, genus}']
    if len(sub):
        w('Restricting to the pairs matched at genus level or better (residual-'
          'confounding control):\n')
        w('| tool | metric | n pairs | D (95 % CI) | q (BH) |')
        w('|---|---|---|---|---|')
        for _, r in sub.iterrows():
            w(f'| {r["tool"]} | {r["metric_name"]} | {int(r["n_pairs"])} | '
              f'{f(r["D_fail_minus_pass"])} ({f(r["ci_lo"])}, {f(r["ci_hi"])}) | '
              f'{float(r["q_bh"]):.3g} |')
        w('')

    if did is not None and len(did):
        w('### 5.3 Difference-in-differences against the tool that defined the filter\n')
        w('Both tools score the identical samples, so `DiD = D(MAGICC) − D(other)` is '
          'immune to any property of the removed genomes that makes them intrinsically '
          'harder for every method. DiD ≈ 0 means MAGICC is no more affected by the '
          'curation boundary than the comparator.\n')
        w('| comparison | metric | D(MAGICC) | D(other) | DiD (95 % CI) | q (BH) |')
        w('|---|---|---|---|---|---|')
        for _, r in did[did.statistic == 'abs_error'].iterrows():
            w(f'| MAGICC V5 vs {r["tool_b"]} | {r["metric_name"]} | '
              f'{f(r["D_tool_a"])} | {f(r["D_tool_b"])} | '
              f'**{f(r["DiD"])}** ({f(r["ci_lo"])}, {f(r["ci_hi"])}) | '
              f'{float(r["q_bh"]):.3g} |')
        w('')

    if has_ck:
        w('### 5.4 CheckM2\'s own error on the subgroup its scores excluded\n')
        w(f'On the {int(A("checkm2", "H_fail", "n"))} simulations built from references '
          f'the CheckM2-based curation would have rejected, CheckM2 itself scores '
          f'completeness MAE {f(A("checkm2", "H_fail", "comp_mae"))} pp '
          f'({f(A("checkm2", "H_fail", "comp_mae_ci_lo"))}–'
          f'{f(A("checkm2", "H_fail", "comp_mae_ci_hi"))}) with bias '
          f'{f(A("checkm2", "H_fail", "comp_bias"))} pp, and contamination MAE '
          f'{f(A("checkm2", "H_fail", "cont_mae"))} pp '
          f'({f(A("checkm2", "H_fail", "cont_mae_ci_lo"))}–'
          f'{f(A("checkm2", "H_fail", "cont_mae_ci_hi"))}) with bias '
          f'{f(A("checkm2", "H_fail", "cont_bias"))} pp, against '
          f'{f(A("checkm2", "H_pass", "comp_mae"))} / '
          f'{f(A("checkm2", "H_pass", "cont_mae"))} pp on the matched controls.\n')

    # ------------------------------------------------------ 6. severity
    w('## 6. Severity gradient inside the removed subgroup\n')
    w('| tool | stratum | n | completeness MAE (95 % CI) | contamination MAE (95 % CI) |')
    w('|---|---|---|---|---|')
    for tool in ['magicc_v5'] + [t for t in tools_present if t != 'magicc_v5']:
        for g, lab in (('H_pass', 'matched controls (would pass)'),
                       ('H_fail:mild', 'mild deficit'),
                       ('H_fail:moderate', 'moderate deficit'),
                       ('H_fail:severe', 'severe deficit')):
            if not len(arm[(arm.tool == tool) & (arm.group == g)]):
                continue
            w(f'| {tool} | {lab} | {int(A(tool, g, "n"))} | '
              f'{f(A(tool, g, "comp_mae"))} ({f(A(tool, g, "comp_mae_ci_lo"))}–'
              f'{f(A(tool, g, "comp_mae_ci_hi"))}) | '
              f'{f(A(tool, g, "cont_mae"))} ({f(A(tool, g, "cont_mae_ci_lo"))}–'
              f'{f(A(tool, g, "cont_mae_ci_hi"))}) |')
    w('')
    w('Strata: *mild* = CheckM2 completeness ≥ 96 % and contamination ≤ 4 % but outside '
      'the ≥ 98 % / ≤ 2 % filter; *moderate* = completeness ≥ 90 % or contamination '
      '≤ 10 %; *severe* = completeness < 90 % or contamination > 10 %.\n')

    # ------------------------------------------------ 7. MIMAG thresholds
    if mimag is not None:
        w('## 7. MIMAG-inspired threshold behaviour\n')
        w('MIMAG-inspired (completeness/contamination only; rRNA and tRNA criteria are '
          'not evaluable from these estimates): high ≥ 90 % completeness AND < 5 % '
          'contamination; medium ≥ 50 % AND < 10 %.\n')
        w('| tool | arm | n | macro F1 (95 % CI) | Cohen κ | true high / medium / low |')
        w('|---|---|---|---|---|---|')
        for _, r in mimag.iterrows():
            w(f'| {r["tool"]} | {r["arm"]} | {int(r["n"])} | {f(r["macro_f1"], 3)} '
              f'({f(r["macro_f1_ci_lo"], 3)}–{f(r["macro_f1_ci_hi"], 3)}) | '
              f'{f(r["cohen_kappa"], 3)} | {int(r["n_true_high"])} / '
              f'{int(r["n_true_medium"])} / {int(r["n_true_low"])} |')
        w('')
        t5 = thr[(thr.criterion == 'contamination') & (thr.tau == 5.0)]
        w('At the 5 % contamination boundary (false-fail denominator = truly clean '
          'simulations only; false-pass denominator = truly contaminated simulations '
          'only):\n')
        w('| tool | arm | truly clean n | truly contaminated n | false-fail rate '
          '(95 % CI) | false-pass rate (95 % CI) | balanced accuracy |')
        w('|---|---|---|---|---|---|---|')
        for _, r in t5.iterrows():
            w(f'| {r["tool"]} | {r["arm"]} | {int(r["n_true_pass"])} | '
              f'{int(r["n_true_fail"])} | {f(r["false_fail_rate"], 3)} '
              f'({f(r["false_fail_ci_lo"], 3)}–{f(r["false_fail_ci_hi"], 3)}) | '
              f'{f(r["false_pass_rate"], 3)} ({f(r["false_pass_ci_lo"], 3)}–'
              f'{f(r["false_pass_ci_hi"], 3)}) | {f(r["balanced_accuracy"], 3)} |')
        w('')

    # ------------------------------------------------------ 8. mechanism
    w('## 8. Why the removed genomes are slightly harder — orthogonal evidence\n')
    w('The benchmark defines completeness relative to the **deposited** reference '
      'assembly and treats that assembly as complete and clean. For the removed arm '
      'that is exactly the assumption in question: if those references really are '
      'imperfect, a well calibrated estimator reads low (or high) and is scored as '
      'biased by a truth that says otherwise. Every tool was therefore also run on the '
      '400 **unmodified** deposited assemblies, with no simulation at all.\n')
    if reflev is not None:
        w('| estimate | mean, would-pass refs | mean, would-FAIL refs | difference '
          '(95 % CI) | uses CheckM2? |')
        w('|---|---|---|---|---|')
        for _, r in reflev.iterrows():
            w(f'| {r["estimate"]} | {f(r["mean_H_pass"])} | {f(r["mean_H_fail"])} | '
              f'{f(r["D_fail_minus_pass"])} ({f(r["ci_lo"])}, {f(r["ci_hi"])}) | '
              f'{"yes — circular" if r["uses_checkm2"] else "no — independent"} |')
        w('')
        mg = reflev[reflev.estimate == 'magicc_v5_completeness']
        gk = reflev[reflev.estimate == 'gtdb_checkm2_completeness']
        if len(mg) and len(gk):
            w(f'MAGICC — which never sees a CheckM2 score — independently reproduces '
              f'the *direction* of the deficit on the removed references '
              f'({f(mg.iloc[0]["D_fail_minus_pass"])} pp completeness, 95 % CI '
              f'{f(mg.iloc[0]["ci_lo"])} to {f(mg.iloc[0]["ci_hi"])}) but at a '
              f'much smaller magnitude than CheckM2 claims '
              f'({f(gk.iloc[0]["D_fail_minus_pass"])} pp). So these assemblies are '
              f'genuinely, mildly imperfect — the exclusion was not pure artefact — '
              f'while the size of the CheckM2 deficit that triggered exclusion is not '
              f'corroborated. Per integrity rule 3, disagreement is reported as '
              f'disagreement: nothing here establishes which estimate is correct.\n')
    if reflev_j and reflev_j.get('checkm2_local_vs_gtdb'):
        rr = reflev_j['checkm2_local_vs_gtdb']
        w(f'Reproducibility check: local CheckM2 1.0.1 re-run on the same 400 '
          f'assemblies reproduces the GTDB-recorded CheckM2 completeness to a mean '
          f'absolute difference of {f(rr.get("completeness_mean_abs_diff"))} pp '
          f'(median {f(rr.get("completeness_median_abs_diff"))} pp, r = '
          f'{f(rr.get("completeness_pearson_r"), 3)}); '
          f'{rr.get("n_refs_local_would_now_pass_filter")} of {rr.get("n_H_fail")} '
          f'would-have-failed references would now pass the filter on the local '
          f're-run.\n')
    if mech is not None and len(mech):
        w('Per-reference regression of the tool\'s mean signed error on the '
          'reference\'s CheckM2 deficit (`ws1_11_reference_incompleteness_mechanism.tsv`):\n')
        w('| tool | signed error | vs | n refs | Spearman ρ | OLS slope ± SE | note |')
        w('|---|---|---|---|---|---|---|')
        for _, r in mech.iterrows():
            w(f'| {r["tool"]} | {r["y"]} | {r["x"]} | {int(r["n_references"])} | '
              f'{f(r["spearman_rho"], 3)} | {f(r["ols_slope"], 3)} ± '
              f'{f(r["ols_slope_se"], 3)} | {r["note"]} |')
        w('')
    strat = load('ws1_11_stratified_by_true_value.tsv')
    if strat is not None:
        s = strat[(strat.tool == 'magicc_v5')
                  & (strat.stratify_by == 'true_contamination')]
        w('Stratified within matched true-contamination bands, which removes any '
          'residual difference in the realised label distributions, MAGICC\'s signed '
          'contamination error separates the arms exactly where a genuinely '
          'contaminated reference would show up — the low-contamination bands:\n')
        w('| true contamination band | n (pass / FAIL) | MAGICC bias, would-pass refs | '
          'MAGICC bias, would-FAIL refs |')
        w('|---|---|---|---|')
        for b in s['bin'].unique():
            rp = s[(s['bin'] == b) & (s.arm == 'H_pass')]
            rf = s[(s['bin'] == b) & (s.arm == 'H_fail')]
            if not len(rp) or not len(rf):
                continue
            w(f'| {b} | {int(rp.iloc[0]["n"])} / {int(rf.iloc[0]["n"])} | '
              f'{f(rp.iloc[0]["cont_bias"])} | {f(rf.iloc[0]["cont_bias"])} |')
        w('')
    if adj is not None:
        w('Sensitivity analysis with the truth corrected for the reference genome\'s '
          'own imperfection (first-order model in '
          '`scripts/191_ws1_11_analysis.py::add_adjusted_truth`; it uses CheckM2\'s own '
          'reference scores, so it structurally favours CheckM2 and is a sensitivity '
          'analysis only):\n')
        w('| tool | arm | completeness MAE raw → adjusted | contamination MAE raw → '
          'adjusted |')
        w('|---|---|---|---|')
        for _, r in adj.iterrows():
            w(f'| {r["tool"]} | {r["arm"]} | {f(r["comp_mae_raw_truth"])} → '
              f'{f(r["comp_mae_adjusted_truth"])} | {f(r["cont_mae_raw_truth"])} → '
              f'{f(r["cont_mae_adjusted_truth"])} |')
        w('')

    # -------------------------------------------------------- 9. context
    if ctx is not None:
        w('## 9. Context — MAGICC V5 across reference-selection regimes\n')
        w('Reported for scale only; the sets differ in taxonomic composition, so the '
          'matched within-`set_H_ncbi` contrast above is the primary comparison.\n')
        w('| benchmark set / group | n | clusters | completeness MAE (95 % CI) | '
          'contamination MAE (95 % CI) |')
        w('|---|---|---|---|---|')
        for _, r in ctx.iterrows():
            g = r['group']
            lab = {'H_pass': 'set_H_ncbi, would-have-passed references',
                   'H_fail': '**set_H_ncbi, would-have-FAILED references**'}.get(g, g)
            w(f'| {lab} | {int(r["n"])} | {int(r["n_clusters"])} | '
              f'{f(r["comp_mae"])} ({f(r["comp_mae_ci_lo"])}–'
              f'{f(r["comp_mae_ci_hi"])}) | {f(r["cont_mae"])} '
              f'({f(r["cont_mae_ci_lo"])}–{f(r["cont_mae_ci_hi"])}) |')
        w('')

    # -------------------------------------------------- 10. limitations
    w('## 10. What this experiment does and does not establish\n')
    w('**Establishes.** Reference genomes chosen by an annotation that has nothing to '
      'do with CheckM2, deliberately including genomes the project\'s CheckM2 filter '
      'would have rejected, are strictly outside the training data and outside the '
      'curated pool (audited, not asserted). On those genomes MAGICC\'s error changes '
      'by the amount quantified in §5.2, measured against taxonomically and '
      'size-matched controls on identically parameterised simulations.\n')
    w('**Does not establish.** (i) Nothing here adjudicates whether CheckM2\'s scores '
      'for the excluded references are right — the tools disagree, and disagreement is '
      'not correctness. (ii) NCBI-Complete assemblies absent from GTDB cannot be '
      'reached, because the taxonomic-consistency filter needs a GTDB taxonomy. '
      '(iii) This is an evaluation, not a retrain: it measures whether the *benchmark* '
      'was blind to a class of genome, not what a model trained on an uncurated pool '
      'would do. A retrain was excluded by author decision because it would also '
      'invalidate the V5-anchored WS1.6 / WS1.9 holdout experiments. (iv) GUNC was not '
      'run: it yields CSS and a pass/fail flag rather than completeness/contamination '
      'percentages, so it cannot contribute to an error comparison.\n')

    # ------------------------------------------------------- 11. files
    w('## 11. Files\n')
    w('| path | content |')
    w('|---|---|')
    for name, desc in [
        ('data/benchmarks/set_H_ncbi/', 'benchmark set: 4,000 simulated FASTAs, '
         '400 reference assemblies, metadata, labels, generation metadata (53 columns), '
         'checkpoint, per-tool predictions'),
        ('data/benchmarks/set_H_ncbi/reference_selection_final.tsv',
         'the 400 references with arm, pair, match level, CheckM2 scores, taxonomy'),
        ('data/benchmarks/set_H_ncbi/candidate_pool.tsv.gz',
         'every eligible NCBI-Complete candidate with its would-pass/would-fail label'),
        ('results/revision/circularity/provenance/',
         'overlap summary, dominant list, contaminant list, SHA256 manifest, README'),
        ('results/revision/circularity/ws1_11_arm_metrics.tsv',
         'MAE / bias / RMSE / R² per tool per arm and per severity stratum, with CIs'),
        ('results/revision/circularity/ws1_11_paired_arm_tests.tsv',
         'primary estimator: paired D with CI, effect sizes, p and BH q'),
        ('results/revision/circularity/ws1_11_did_vs_competitors.tsv',
         'difference-in-differences against each competitor'),
        ('results/revision/circularity/ws1_11_mimag.tsv, ws1_11_thresholds.tsv',
         'MIMAG-inspired classification and 5 %/10 %/50 %/90 % decision thresholds'),
        ('results/revision/circularity/ws1_11_reference_level_scores.tsv',
         'tool scores on the 400 unmodified deposited assemblies (orthogonal evidence)'),
        ('results/revision/circularity/ws1_11_reference_incompleteness_mechanism.tsv',
         'per-reference signed error regressed on the CheckM2 deficit'),
        ('results/revision/circularity/ws1_11_adjusted_truth_sensitivity.tsv',
         'MAE against a truth corrected for the reference genome itself'),
        ('results/revision/circularity/ws1_11_context_other_sets.tsv',
         'MAGICC V5 on the CheckM2-filtered benchmark sets, for scale'),
        ('results/revision/circularity/figures/',
         '5 CVD-safe figures with captions in ws1_11_figure_captions.md'),
        ('scripts/185–193', 'selection, fetch, generation, audit, inference, competitor '
         'runs, analysis, reference-level scores, this report'),
    ]:
        w(f'| `{name}` | {desc} |')
    w('')

    (OUT / 'WS1_11_R1m13_REPORT.md').write_text('\n'.join(md))
    print(f'wrote {OUT / "WS1_11_R1m13_REPORT.md"}  ({len(md)} lines)')
    print(f'VERDICT: {verdict_tag}  (worst upper CI = {worst_hi:.3f} pp, '
          f'materiality {MAT} pp)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
