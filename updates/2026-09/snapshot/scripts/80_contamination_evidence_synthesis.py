#!/usr/bin/env python3
"""
WS4.3/4.5/4.6 -- synthesis: is the Kraken2 contamination claim real or a
novelty artifact, and is MAGICC V5's higher contamination corroborated by
database-independent evidence?

The manuscript claims (R2-M5):
    "Kraken2 independently confirms 109/141 (77.3%) have cross-phylum
     contamination >5%", with V5 correlating r=0.19 (p=0.024) and CheckM2 not
     correlating (r=-0.09, p=0.279).

Three questions, answered in order.

Q1  DECISIVE EXPERIMENT -- matched-novelty negative controls.
    The suspected confound: novel lineages get diffuse low-support LCA
    assignments, and the metric (1 - dominant-phylum fraction, full contig
    length attributed regardless of k-mer support) scores that as
    contamination.  We therefore match ``agree_clean`` MAGs (both MAGICC V5 and
    CheckM2 call them HQ) to the 141 disagreement MAGs on
        informative_kmer_density  (Kraken2-database representation = novelty)
        log10 genome size, log10 contig count, log10 N50, GC%
    by 1:1 nearest-neighbour on standardized covariates, then compare the SAME
    Kraken2 metric.  If the cohorts are indistinguishable after matching, the
    metric measures novelty, not contamination.
    Reported with Hodges-Lehmann shift, Cliff's delta, bootstrap CIs and
    Mann-Whitney U -- effect sizes, not just p-values.

Q2  Does the signal survive a defensible metric?  (strict metric from
    scripts/78, which requires per-contig confidence and excludes
    unclassified/low-confidence bp from the denominator.)

Q3  Is V5's contamination corroborated by DATABASE-INDEPENDENT evidence
    (divergent single-copy core-gene duplicates, scripts/79)?  And -- the
    hypothesis we must test rather than assume -- is V5's correlation with the
    Kraken2 metric merely a shared dependence on novelty?  V5 is known to
    degrade on held-out novel lineages (Patescibacteria completeness MAE
    3.15% -> 6.82% once training leakage was removed, contamination bias
    -3.55), so a weak positive correlation with a novelty-driven metric could
    reflect V5's error mode rather than contamination detection.
    Addressed with partial (rank) correlations controlling for novelty.

Usage
-----
    python scripts/80_contamination_evidence_synthesis.py

Outputs (``results/revision/contamination_evidence/``)
------------------------------------------------------
    matched_cohort_pairs.tsv        the 1:1 matches and their covariates
    evidence_correlations.tsv       correlation matrix across evidence lines
    contamination_evidence_table.tsv  per-genome consolidated evidence (WS4.6)
    synthesis_summary.json          all statistics, effect sizes and CIs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVID_DIR = PROJECT_DIR / 'results' / 'revision' / 'contamination_evidence'
GUNC_DIR = PROJECT_DIR / 'results' / 'revision' / 'gunc'

MATCH_COVARIATES = [
    'informative_kmer_density',   # Kraken2 DB representation == novelty proxy
    'log10_total_bp',
    'log10_n_contigs',
    'log10_n50',
    'gc_pct',
]
SEED = 42


# ---------------------------------------------------------------------------
# small stats helpers
# ---------------------------------------------------------------------------
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """P(a>b) - P(a<b); +1 means a strictly larger. Rank-based, O(n log n)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float('nan')
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(2.0 * u1 / (n1 * n2) - 1.0)


def hodges_lehmann(a: Sequence[float], b: Sequence[float],
                   max_pairs: int = 4_000_000) -> float:
    """Median of all pairwise differences a_i - b_j."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    if len(a) * len(b) <= max_pairs:
        return float(np.median(np.subtract.outer(a, b).ravel()))
    rng = np.random.default_rng(SEED)
    ia = rng.integers(0, len(a), max_pairs)
    ib = rng.integers(0, len(b), max_pairs)
    return float(np.median(a[ia] - b[ib]))


def boot_ci(func, *arrays, n_boot: int = 10000, alpha: float = 0.05
            ) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    arrays = [np.asarray(x, float) for x in arrays]
    vals = []
    for _ in range(n_boot):
        res = [x[rng.integers(0, len(x), len(x))] for x in arrays]
        try:
            vals.append(func(*res))
        except Exception:
            continue
    if not vals:
        return (float('nan'), float('nan'))
    lo, hi = np.percentile(vals, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)


def paired_boot_ci(diffs: Sequence[float], n_boot: int = 10000,
                   alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (float('nan'), float('nan'))
    meds = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    lo, hi = np.percentile(meds, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)


def spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return float('nan'), float('nan'), int(ok.sum())
    r, p = stats.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def pearson(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return float('nan'), float('nan'), int(ok.sum())
    r, p = stats.pearsonr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def partial_spearman(x, y, controls: List[Sequence[float]]
                     ) -> Tuple[float, float, int]:
    """
    Spearman partial correlation: rank-transform everything, linearly regress
    x and y on the control ranks, correlate the residuals.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    C = [np.asarray(c, float) for c in controls]
    ok = ~(np.isnan(x) | np.isnan(y))
    for c in C:
        ok &= ~np.isnan(c)
    n = int(ok.sum())
    if n < 6:
        return float('nan'), float('nan'), n
    xr = stats.rankdata(x[ok])
    yr = stats.rankdata(y[ok])
    Cr = np.column_stack([stats.rankdata(c[ok]) for c in C])
    Cr = np.column_stack([np.ones(n), Cr])
    bx, *_ = np.linalg.lstsq(Cr, xr, rcond=None)
    by, *_ = np.linalg.lstsq(Cr, yr, rcond=None)
    rx = xr - Cr @ bx
    ry = yr - Cr @ by
    r, _ = stats.pearsonr(rx, ry)
    # p-value with df reduced by the number of controls
    dof = n - 2 - len(C)
    if dof <= 0:
        return float(r), float('nan'), n
    t = r * math.sqrt(dof / max(1e-12, 1 - r * r))
    p = 2 * stats.t.sf(abs(t), dof)
    return float(r), float(p), n


def fnum(v) -> float:
    try:
        f = float(v)
        return f if not (math.isinf(f) or math.isnan(f)) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


# ---------------------------------------------------------------------------
def build_records() -> Dict[str, Dict[str, object]]:
    """Merge cohort stats + Kraken2 metrics + SCG duplication + GUNC."""
    rec: Dict[str, Dict[str, object]] = {}

    for r in read_tsv(EVID_DIR / 'cohort_genome_stats.tsv'):
        a = r['accession']
        rec[a] = {
            'accession': a, 'cohort': r['cohort'],
            'v5_completeness': fnum(r['v5_completeness']),
            'v5_contamination': fnum(r['v5_contamination']),
            'checkm2_completeness': fnum(r['checkm2_completeness']),
            'checkm2_contamination': fnum(r['checkm2_contamination']),
            'total_bp': fnum(r['total_bp']),
            'n_contigs': fnum(r['n_contigs']),
            'n50': fnum(r['n50']),
            'gc_pct': fnum(r['gc_pct']),
            'log10_total_bp': fnum(r['log10_total_bp']),
        }
        rec[a]['log10_n_contigs'] = math.log10(max(1.0, rec[a]['n_contigs']))
        rec[a]['log10_n50'] = math.log10(max(1.0, rec[a]['n50']))

    kp = EVID_DIR / 'kraken2_metrics.tsv'
    if kp.is_file():
        for r in read_tsv(kp):
            a = r['accession']
            if a not in rec:
                continue
            rec[a].update({
                'orig_phylum_contam_pct': fnum(r['orig_phylum_contam_pct']),
                'orig_phylum_dominant_pct': fnum(r['orig_phylum_dominant_pct']),
                'orig_n_phyla': fnum(r['orig_n_phyla']),
                'strict_phylum_contam_pct': fnum(r['strict_phylum_contam_pct']),
                'strict_qualifying_bp_frac': fnum(r['strict_qualifying_bp_frac']),
                'unclassified_pct': fnum(r['unclassified_pct']),
                'informative_kmer_density': fnum(r['informative_kmer_density']),
                'mean_contig_confidence_bpw': fnum(r['mean_contig_confidence_bpw']),
            })

    sp = EVID_DIR / 'scg_duplication.tsv'
    if sp.is_file():
        for r in read_tsv(sp):
            a = r['accession']
            if a not in rec:
                continue
            rec[a].update({
                'scg_domain': r['domain'],
                'frac_profiles_detected': fnum(r['frac_profiles_detected']),
                'n_families_multicopy': fnum(r['n_families_multicopy']),
                'total_extra_copies': fnum(r['total_extra_copies']),
                'max_copies': fnum(r['max_copies']),
                'n_dup_pairs': fnum(r['n_dup_pairs']),
                'n_dup_pairs_diff_contig': fnum(r['n_dup_pairs_diff_contig']),
                'median_dup_pair_identity': fnum(r['median_dup_pair_identity']),
                'n_dup_pairs_divergent': fnum(r['n_dup_pairs_divergent']),
                'divergent_dup_families_per_100_profiles': fnum(
                    r['divergent_dup_families_per_100_profiles']),
            })

    # GUNC (whatever runs exist)
    for sub in ('controls/negative', 'controls/positive',
                'controls_gtdb95/negative', 'controls_gtdb95/positive',
                'cohorts/disagreement_141', 'cohorts/agree_clean'):
        p = GUNC_DIR / sub / 'gunc_normalized.tsv'
        if not p.is_file():
            continue
        tag = 'gtdb95' if 'gtdb95' in sub else 'pg21'
        for r in read_tsv(p):
            a = r['genome']
            if a not in rec:
                continue
            rec[a][f'gunc_{tag}_css'] = fnum(r['gunc_css'])
            rec[a][f'gunc_{tag}_pass'] = r['gunc_pass']
            rec[a][f'gunc_{tag}_rrs'] = fnum(r.get('reference_representation_score', ''))
    return rec


# ---------------------------------------------------------------------------
def match_cohorts(rec: Dict[str, Dict[str, object]], caliper: float
                  ) -> List[Tuple[str, str, float]]:
    """1:1 nearest-neighbour matching on standardized covariates, greedy."""
    treat = [a for a, r in rec.items()
             if r['cohort'] == 'disagreement_141'
             and all(not math.isnan(fnum(r.get(c))) for c in MATCH_COVARIATES)]
    ctrl = [a for a, r in rec.items()
            if r['cohort'] == 'agree_clean'
            and all(not math.isnan(fnum(r.get(c))) for c in MATCH_COVARIATES)]
    if not treat or not ctrl:
        return []

    allids = treat + ctrl
    X = np.array([[fnum(rec[a][c]) for c in MATCH_COVARIATES] for a in allids])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    zof = {a: Z[i] for i, a in enumerate(allids)}

    # greedy nearest neighbour, treated ordered by how hard they are to match
    used = set()
    pairs = []
    order = sorted(treat, key=lambda a: min(
        float(np.linalg.norm(zof[a] - zof[c])) for c in ctrl))
    for t in order:
        best, bestd = None, float('inf')
        for c in ctrl:
            if c in used:
                continue
            d = float(np.linalg.norm(zof[t] - zof[c]))
            if d < bestd:
                best, bestd = c, d
        if best is not None and bestd <= caliper:
            used.add(best)
            pairs.append((t, best, bestd))
    return pairs


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(EVID_DIR))
    ap.add_argument('--caliper', type=float, default=1.0,
                    help='max standardized distance for a match (default 1.0)')
    ap.add_argument('--n-boot', type=int, default=10000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    random.seed(SEED)

    rec = build_records()
    have_k2 = [a for a, r in rec.items() if 'orig_phylum_contam_pct' in r]
    have_scg = [a for a, r in rec.items() if 'n_families_multicopy' in r]
    print(f"records: {len(rec)}   with Kraken2: {len(have_k2)}   with SCG: {len(have_scg)}")
    summary: Dict[str, object] = {'n_records': len(rec),
                                  'n_with_kraken2': len(have_k2),
                                  'n_with_scg': len(have_scg),
                                  'seed': SEED}

    d141 = [r for r in rec.values() if r['cohort'] == 'disagreement_141']
    clean = [r for r in rec.values() if r['cohort'] == 'agree_clean']

    # ================= Q1: matched-novelty comparison =====================
    print("\n" + "=" * 78)
    print("Q1  MATCHED-NOVELTY COMPARISON  (original Kraken2 metric)")
    print("=" * 78)
    have_both = (any('orig_phylum_contam_pct' in r for r in d141)
                 and any('orig_phylum_contam_pct' in r for r in clean))
    if not have_both:
        print("  SKIPPED: Kraken2 metrics not yet available for agree_clean.")
        print("  Run: python scripts/78_kraken2_strict_metric.py --run-missing")
        summary['Q1'] = {'status': 'skipped -- agree_clean Kraken2 missing'}
        pairs = []
    else:
        pairs = match_cohorts(rec, args.caliper)
        print(f"  matched pairs: {len(pairs)} (caliper {args.caliper} SD)")
        with open(out_dir / 'matched_cohort_pairs.tsv', 'w', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['treated_accession', 'control_accession', 'match_distance']
                        + [f'treated_{c}' for c in MATCH_COVARIATES]
                        + [f'control_{c}' for c in MATCH_COVARIATES]
                        + ['treated_orig_contam_pct', 'control_orig_contam_pct'])
            for t, c, d in pairs:
                w.writerow([t, c, round(d, 4)]
                           + [rec[t][k] for k in MATCH_COVARIATES]
                           + [rec[c][k] for k in MATCH_COVARIATES]
                           + [rec[t].get('orig_phylum_contam_pct', ''),
                              rec[c].get('orig_phylum_contam_pct', '')])

        # covariate balance
        print(f"\n  covariate balance after matching (standardized mean diff):")
        balance = {}
        for cv in MATCH_COVARIATES:
            tv = np.array([fnum(rec[t][cv]) for t, _, _ in pairs])
            cvv = np.array([fnum(rec[c][cv]) for _, c, _ in pairs])
            pooled = math.sqrt((tv.var() + cvv.var()) / 2) or 1.0
            smd = (tv.mean() - cvv.mean()) / pooled
            balance[cv] = {'treated_mean': float(tv.mean()),
                           'control_mean': float(cvv.mean()),
                           'std_mean_diff': float(smd)}
            print(f"    {cv:28s} treated {tv.mean():12.5f}  control {cvv.mean():12.5f}"
                  f"   SMD {smd:+.3f}")

        a = np.array([fnum(rec[t]['orig_phylum_contam_pct']) for t, _, _ in pairs])
        b = np.array([fnum(rec[c]['orig_phylum_contam_pct']) for _, c, _ in pairs])
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        diffs = a - b
        hl = hodges_lehmann(a, b)
        hl_lo, hl_hi = boot_ci(hodges_lehmann, a, b, n_boot=min(args.n_boot, 2000))
        cd = cliffs_delta(a, b)
        cd_lo, cd_hi = boot_ci(cliffs_delta, a, b, n_boot=args.n_boot)
        u, pu = stats.mannwhitneyu(a, b, alternative='two-sided')
        try:
            wstat, pw = stats.wilcoxon(diffs, alternative='two-sided')
        except ValueError:
            wstat, pw = float('nan'), float('nan')
        md_lo, md_hi = paired_boot_ci(diffs, n_boot=args.n_boot)

        print(f"\n  ORIGINAL Kraken2 phylum contamination, matched pairs (n={len(a)}):")
        print(f"    disagreement_141  median {np.median(a):7.2f}%   "
              f"IQR [{np.percentile(a,25):.2f}, {np.percentile(a,75):.2f}]")
        print(f"    matched clean     median {np.median(b):7.2f}%   "
              f"IQR [{np.percentile(b,25):.2f}, {np.percentile(b,75):.2f}]")
        print(f"    Hodges-Lehmann shift {hl:+.2f} pp   95% CI [{hl_lo:+.2f}, {hl_hi:+.2f}]")
        print(f"    paired median diff   {np.median(diffs):+.2f} pp  "
              f"95% CI [{md_lo:+.2f}, {md_hi:+.2f}]")
        print(f"    Cliff's delta        {cd:+.3f}   95% CI [{cd_lo:+.3f}, {cd_hi:+.3f}]")
        print(f"    Mann-Whitney U p = {pu:.4g}   Wilcoxon signed-rank p = {pw:.4g}")
        gt5_a = int((a > 5).sum()); gt5_b = int((b > 5).sum())
        print(f"    >5% contamination:  disagreement {gt5_a}/{len(a)} "
              f"({gt5_a/len(a)*100:.1f}%)   matched clean {gt5_b}/{len(b)} "
              f"({gt5_b/len(b)*100:.1f}%)")

        summary['Q1'] = {
            'n_matched_pairs': int(len(a)), 'caliper_sd': args.caliper,
            'covariates': MATCH_COVARIATES, 'balance': balance,
            'disagreement_median_pct': float(np.median(a)),
            'matched_clean_median_pct': float(np.median(b)),
            'hodges_lehmann_shift_pp': hl,
            'hodges_lehmann_ci95': [hl_lo, hl_hi],
            'paired_median_diff_pp': float(np.median(diffs)),
            'paired_median_diff_ci95': [md_lo, md_hi],
            'cliffs_delta': cd, 'cliffs_delta_ci95': [cd_lo, cd_hi],
            'mannwhitney_p': float(pu), 'wilcoxon_p': float(pw),
            'gt5pct_disagreement': gt5_a, 'gt5pct_matched_clean': gt5_b,
            'n_per_arm': int(len(a)),
        }

    # ============ Q2: does the signal survive a defensible metric? =========
    print("\n" + "=" * 78)
    print("Q2  STRICT (CONFIDENCE-AWARE) METRIC")
    print("=" * 78)
    q2 = {}
    for label, grp in (('disagreement_141', d141), ('agree_clean', clean)):
        o = [fnum(r.get('orig_phylum_contam_pct')) for r in grp]
        s = [fnum(r.get('strict_phylum_contam_pct')) for r in grp]
        o = [x for x in o if not math.isnan(x)]
        s = [x for x in s if not math.isnan(x)]
        if not o:
            continue
        print(f"  {label:20s} n={len(o):>4}  original median {np.median(o):6.2f}%"
              f"   strict scoreable {len(s):>4}"
              + (f"  strict median {np.median(s):6.2f}%" if s else "  strict: none scoreable"))
        q2[label] = {'n_original': len(o),
                     'median_original_pct': float(np.median(o)),
                     'n_strict_scoreable': len(s),
                     'median_strict_pct': float(np.median(s)) if s else None}
    summary['Q2'] = q2

    # ======== Q3: DB-independent evidence and the novelty confound =========
    print("\n" + "=" * 78)
    print("Q3  DATABASE-INDEPENDENT EVIDENCE (divergent SCG duplicates)")
    print("=" * 78)
    scg_field = 'divergent_dup_families_per_100_profiles'
    a_s = np.array([fnum(r.get(scg_field)) for r in d141])
    b_s = np.array([fnum(r.get(scg_field)) for r in clean])
    a_s = a_s[~np.isnan(a_s)]
    b_s = b_s[~np.isnan(b_s)]
    q3: Dict[str, object] = {}
    if len(a_s) and len(b_s):
        hl = hodges_lehmann(a_s, b_s)
        hl_lo, hl_hi = boot_ci(hodges_lehmann, a_s, b_s, n_boot=min(args.n_boot, 2000))
        cd = cliffs_delta(a_s, b_s)
        cd_lo, cd_hi = boot_ci(cliffs_delta, a_s, b_s, n_boot=args.n_boot)
        u, pu = stats.mannwhitneyu(a_s, b_s, alternative='two-sided')
        frac_a = float((a_s > 0).mean()); frac_b = float((b_s > 0).mean())
        print(f"  divergent duplicate SCG families per 100 profiles:")
        print(f"    disagreement_141  n={len(a_s):>4} median {np.median(a_s):6.3f}"
              f"   mean {a_s.mean():6.3f}   any>0: {frac_a*100:5.1f}%")
        print(f"    agree_clean       n={len(b_s):>4} median {np.median(b_s):6.3f}"
              f"   mean {b_s.mean():6.3f}   any>0: {frac_b*100:5.1f}%")
        print(f"    Hodges-Lehmann {hl:+.3f}  95% CI [{hl_lo:+.3f}, {hl_hi:+.3f}]")
        print(f"    Cliff's delta  {cd:+.3f}  95% CI [{cd_lo:+.3f}, {cd_hi:+.3f}]")
        print(f"    Mann-Whitney U p = {pu:.4g}")
        # proportion test on "any divergent duplicate"
        ct = np.array([[int((a_s > 0).sum()), int((a_s == 0).sum())],
                       [int((b_s > 0).sum()), int((b_s == 0).sum())]])
        try:
            _, pf = stats.fisher_exact(ct)
        except Exception:
            pf = float('nan')
        print(f"    any divergent duplicate, Fisher exact p = {pf:.4g}")
        q3['scg_between_cohorts'] = {
            'n_disagreement': int(len(a_s)), 'n_clean': int(len(b_s)),
            'median_disagreement': float(np.median(a_s)),
            'median_clean': float(np.median(b_s)),
            'mean_disagreement': float(a_s.mean()), 'mean_clean': float(b_s.mean()),
            'frac_any_divergent_disagreement': frac_a,
            'frac_any_divergent_clean': frac_b,
            'hodges_lehmann': hl, 'hodges_lehmann_ci95': [hl_lo, hl_hi],
            'cliffs_delta': cd, 'cliffs_delta_ci95': [cd_lo, cd_hi],
            'mannwhitney_p': float(pu), 'fisher_any_divergent_p': float(pf),
        }

    # ---- correlation matrix across evidence lines (within the 141) ----
    print("\n  Correlations within disagreement_141 "
          "(Spearman; novelty-controlled partial in brackets):")
    # Novelty controls must be available for essentially the whole cohort or the
    # partial correlation silently loses power. Candidates are ranked by
    # coverage; anything missing for >20% of the cohort is dropped.
    novelty_candidates = ['informative_kmer_density', 'mean_contig_confidence_bpw',
                          'frac_profiles_detected']
    novelty = []
    for c in novelty_candidates:
        cov = sum(1 for r in d141 if not math.isnan(fnum(r.get(c)))) / max(1, len(d141))
        if cov >= 0.8:
            novelty.append(c)
        else:
            print(f"    (dropping novelty control '{c}': only "
                  f"{cov*100:.0f}% coverage in cohort)")
    print(f"    novelty controls: {novelty}")
    targets = [
        ('v5_contamination', 'MAGICC V5 contamination'),
        ('checkm2_contamination', 'CheckM2 contamination'),
    ]
    evidence = [
        ('orig_phylum_contam_pct', 'Kraken2 ORIGINAL phylum contam'),
        ('strict_phylum_contam_pct', 'Kraken2 STRICT phylum contam'),
        (scg_field, 'divergent SCG dup families/100'),
        ('n_dup_pairs_diff_contig', 'SCG dup pairs on diff contigs'),
        ('gunc_pg21_css', 'GUNC CSS (proGenomes 2.1)'),
        ('gunc_gtdb95_css', 'GUNC CSS (GTDB r95)'),
    ]
    corr_rows = []
    for tkey, tlabel in targets:
        print(f"\n    {tlabel}:")
        for ekey, elabel in evidence:
            x = [fnum(r.get(tkey)) for r in d141]
            y = [fnum(r.get(ekey)) for r in d141]
            r_, p_, n_ = spearman(x, y)
            if n_ < 4:
                continue
            ctrls = [[fnum(r.get(c)) for r in d141] for c in novelty]
            pr, pp, pn = partial_spearman(x, y, ctrls)
            print(f"      {elabel:34s} rho={r_:+.3f} p={p_:7.4g} n={n_:>4}"
                  f"   [partial rho={pr:+.3f} p={pp:7.4g}]")
            corr_rows.append({'cohort': 'disagreement_141', 'target': tkey,
                              'evidence': ekey, 'spearman_rho': round(r_, 4),
                              'spearman_p': p_, 'n': n_,
                              'partial_rho_novelty_controlled': (
                                  '' if pr != pr else round(pr, 4)),
                              'partial_p': pp, 'partial_n': pn,
                              'controls': ';'.join(novelty)})

    # ---- the explicit confound test ----
    print("\n  NOVELTY CONFOUND TEST -- do the tool predictions themselves track novelty?")
    conf_rows = []
    for tkey, tlabel in targets + [('orig_phylum_contam_pct',
                                    'Kraken2 ORIGINAL phylum contam')]:
        nkeys = list(dict.fromkeys(
            novelty + ['unclassified_pct', 'mean_contig_confidence_bpw']))
        for nkey in nkeys:
            x = [fnum(r.get(tkey)) for r in d141]
            y = [fnum(r.get(nkey)) for r in d141]
            r_, p_, n_ = spearman(x, y)
            if n_ < 4:
                continue
            print(f"    {tlabel:32s} vs {nkey:30s} rho={r_:+.3f} p={p_:7.4g} n={n_}")
            conf_rows.append({'cohort': 'disagreement_141', 'target': tkey,
                              'evidence': nkey, 'spearman_rho': round(r_, 4),
                              'spearman_p': p_, 'n': n_,
                              'partial_rho_novelty_controlled': '',
                              'partial_p': '', 'partial_n': '',
                              'controls': 'NONE (novelty association test)'})

    # ---- headline verdict block ----
    print("\n  HEADLINE STATISTICS")
    x_v5 = [fnum(r.get('v5_contamination')) for r in d141]
    x_ck = [fnum(r.get('checkm2_contamination')) for r in d141]
    y_k2 = [fnum(r.get('orig_phylum_contam_pct')) for r in d141]
    y_scg = [fnum(r.get(scg_field)) for r in d141]
    ctrls = [[fnum(r.get(c)) for r in d141] for c in novelty]
    headline = {}
    for lbl, xx, yy, ylbl in (
            ('v5_vs_kraken2_original', x_v5, y_k2, 'Kraken2 ORIGINAL'),
            ('checkm2_vs_kraken2_original', x_ck, y_k2, 'Kraken2 ORIGINAL'),
            ('v5_vs_scg_divergent_dups', x_v5, y_scg, 'DB-independent SCG'),
            ('checkm2_vs_scg_divergent_dups', x_ck, y_scg, 'DB-independent SCG')):
        r_, p_, n_ = spearman(xx, yy)
        pr, pp, pn = partial_spearman(xx, yy, ctrls)
        tool = 'MAGICC V5' if lbl.startswith('v5') else 'CheckM2  '
        print(f"    {tool} vs {ylbl:20s} rho={r_:+.3f} (p={p_:8.4g}, n={n_:>3})"
              f"   novelty-controlled rho={pr:+.3f} (p={pp:8.4g}, n={pn:>3})")
        headline[lbl] = {'spearman_rho': None if r_ != r_ else round(r_, 4),
                         'spearman_p': None if p_ != p_ else p_, 'n': n_,
                         'partial_rho': None if pr != pr else round(pr, 4),
                         'partial_p': None if pp != pp else pp, 'partial_n': pn,
                         'controls': novelty}
    q3['headline'] = headline

    with open(out_dir / 'evidence_correlations.tsv', 'w', newline='') as f:
        cols = ['cohort', 'target', 'evidence', 'spearman_rho', 'spearman_p', 'n',
                'partial_rho_novelty_controlled', 'partial_p', 'partial_n', 'controls']
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
        w.writeheader()
        w.writerows(corr_rows + conf_rows)
    q3['correlations'] = corr_rows
    q3['novelty_association'] = conf_rows
    summary['Q3'] = q3

    # ---- consolidated per-genome evidence table (WS4.6) ----
    ev_cols = ['accession', 'cohort',
               'v5_completeness', 'v5_contamination',
               'checkm2_completeness', 'checkm2_contamination',
               'total_bp', 'n_contigs', 'n50', 'gc_pct',
               'unclassified_pct', 'informative_kmer_density',
               'mean_contig_confidence_bpw',
               'orig_phylum_contam_pct', 'orig_phylum_dominant_pct', 'orig_n_phyla',
               'strict_phylum_contam_pct', 'strict_qualifying_bp_frac',
               'scg_domain', 'frac_profiles_detected', 'n_families_multicopy',
               'total_extra_copies', 'max_copies', 'n_dup_pairs',
               'n_dup_pairs_diff_contig', 'median_dup_pair_identity',
               'n_dup_pairs_divergent', scg_field,
               'gunc_pg21_css', 'gunc_pg21_pass', 'gunc_pg21_rrs',
               'gunc_gtdb95_css', 'gunc_gtdb95_pass', 'gunc_gtdb95_rrs']
    with open(out_dir / 'contamination_evidence_table.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ev_cols, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        for a in sorted(rec, key=lambda x: (rec[x]['cohort'], x)):
            row = {k: rec[a].get(k, '') for k in ev_cols}
            for k, v in row.items():
                if isinstance(v, float) and math.isnan(v):
                    row[k] = ''
            w.writerow(row)

    (out_dir / 'synthesis_summary.json').write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_dir / 'evidence_correlations.tsv'}")
    print(f"Wrote {out_dir / 'contamination_evidence_table.tsv'}")
    print(f"Wrote {out_dir / 'synthesis_summary.json'}")
    if pairs:
        print(f"Wrote {out_dir / 'matched_cohort_pairs.tsv'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
