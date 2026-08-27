#!/usr/bin/env python3
"""
WS4.3 -- Kraken2 database-capacity test: capped vs uncapped reference database.

Why
---
WS4.3 showed the published cross-phylum contamination metric on the 141
disagreement MAGs is driven by taxonomic assignments supported by a median of 8
informative k-mers per contig (confidence ~5e-4), and that the metric correlates
strongly and negatively with database representation
(rho = -0.605, p = 2e-15) -- i.e. it measures novelty, not contamination.

The obvious objection is that ``tools/kraken2_db`` is the **capped**
``k2_standard_08gb`` build (hash.k2d exactly 8,000,000,032 bytes), whose
minimizer down-sampling alone could explain the low hit rate.  This script
closes that door by repeating the measurement on the same 141 MAGs with a
substantially larger database and reporting the results side by side.

The dichotomy this establishes, either way, is a reportable finding:

  * density rises substantially and the strict metric becomes evaluable
        -> the original failure was DATABASE CAPACITY; report the corrected
           analysis as the replacement for the withdrawn claim.
  * density stays low even with full reference coverage
        -> the failure is GENUINE TAXONOMIC NOVELTY: these MAGs have no adequate
           reference anywhere, and no Kraken2-based metric can adjudicate them.
           That is an important, quotable limit of taxonomy-based contamination
           verification on novel lineages.

Inputs
------
One ``kraken2_metrics{_tag}.tsv`` per database, produced by
``scripts/078_kraken2_strict_metric.py`` with matching ``--db-tag``.  Run 78 once
per database with the SAME ``--kraken2-bin`` so the comparison is not confounded
by Kraken2 version.

Usage
-----
    python scripts/081_kraken2_db_capacity_comparison.py \
        --dbs "k2_standard_08gb(capped)=capped8gb" \
              "k2_standard_20250714(full)=k2std"

Outputs (``results/revision/contamination_evidence/``)
------------------------------------------------------
    kraken2_db_capacity_comparison.tsv    side-by-side per-database statistics
    kraken2_db_density_deciles.tsv        density / confidence distributions
    kraken2_db_capacity_summary.json      all statistics plus the verdict
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVID_DIR = PROJECT_DIR / 'results' / 'revision' / 'contamination_evidence'
COHORT = 'disagreement_141'
NOVELTY = ['informative_kmer_density', 'mean_contig_confidence_bpw']

# ---------------------------------------------------------------------------
# What each database DOES and DOES NOT contain.
# This is load-bearing for the verdict: a low hit rate against a RefSeq-derived
# database shows only that no *RefSeq* reference exists. The cohort here is
# dominated by uncultivated, MAG-derived lineages -- exactly the material RefSeq
# lacks and GTDB has -- so the capacity-versus-novelty question is only
# decidable against a GTDB-level reference set.
# ---------------------------------------------------------------------------
DB_REGISTRY: Dict[str, Dict[str, object]] = {
    'capped8gb': {
        'contains': ('RefSeq archaea, bacteria, viral, plasmid, human, UniVec_Core '
                     '(the k2_standard content), MINIMIZER-DOWN-SAMPLED to an '
                     '8 GB hash via --max-db-size'),
        'lacks': ('MAG-derived / uncultivated lineages (RefSeq-derived), AND most '
                  'of its own source minimizers because of the 8 GB cap'),
        'is_refseq_derived': True,
        'is_gtdb_level': False,
        'is_capped': True,
    },
    'k2std': {
        'contains': ('full uncapped RefSeq archaea, bacteria, viral, plasmid, '
                     'human, UniVec_Core'),
        'lacks': ('MAG-derived / uncultivated lineages -- RefSeq contains almost '
                  'no MAGs, which is precisely the material this cohort is made of'),
        'is_refseq_derived': True,
        'is_gtdb_level': False,
        'is_capped': False,
    },
    'gtdbcustom': {
        'contains': ('the project GTDB reference set (data/genomes/), i.e. the '
                     'exact reference material MAGICC was trained on, including '
                     'MAG-derived and uncultivated lineages'),
        'lacks': ('lineages absent from GTDB / from the curated 100k selection'),
        'is_refseq_derived': False,
        'is_gtdb_level': True,
        'is_capped': False,
    },
}


def db_info(tag: str) -> Dict[str, object]:
    return DB_REGISTRY.get(tag, {'contains': 'unknown', 'lacks': 'unknown',
                                 'is_refseq_derived': None,
                                 'is_gtdb_level': False, 'is_capped': None})


def fnum(v) -> float:
    try:
        f = float(v)
        return f if not (math.isinf(f) or math.isnan(f)) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def read_tsv(p: Path) -> List[Dict[str, str]]:
    with open(p) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def partial_spearman(x, y, controls) -> Tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    C = [np.asarray(c, float) for c in controls]
    ok = ~(np.isnan(x) | np.isnan(y))
    for c in C:
        ok &= ~np.isnan(c)
    n = int(ok.sum())
    if n < 6:
        return float('nan'), float('nan'), n
    xr, yr = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    Cr = np.column_stack([np.ones(n)] + [stats.rankdata(c[ok]) for c in C])
    bx, *_ = np.linalg.lstsq(Cr, xr, rcond=None)
    by, *_ = np.linalg.lstsq(Cr, yr, rcond=None)
    r, _ = stats.pearsonr(xr - Cr @ bx, yr - Cr @ by)
    dof = n - 2 - len(C)
    if dof <= 0:
        return float(r), float('nan'), n
    t = r * math.sqrt(dof / max(1e-12, 1 - r * r))
    return float(r), float(2 * stats.t.sf(abs(t), dof)), n


def spearman(x, y) -> Tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return float('nan'), float('nan'), int(ok.sum())
    r, p = stats.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dbs', nargs='+', required=True,
                    help='LABEL=TAG entries; TAG matches '
                         'kraken2_metrics_TAG.tsv (empty TAG = no suffix)')
    ap.add_argument('--out-dir', default=str(EVID_DIR))
    ap.add_argument('--cohort', default=COHORT)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    loaded: List[Tuple[str, str, List[Dict[str, str]], Optional[dict]]] = []
    for spec in args.dbs:
        label, _, tag = spec.partition('=')
        suffix = f'_{tag}' if tag else ''
        p = out_dir / f'kraken2_metrics{suffix}.tsv'
        if not p.is_file():
            print(f"  SKIP {label}: {p.name} not found")
            continue
        sj = out_dir / f'kraken2_metrics_summary{suffix}.json'
        meta = json.loads(sj.read_text()) if sj.is_file() else None
        rows = [r for r in read_tsv(p) if r['cohort'] == args.cohort]
        loaded.append((label, tag, rows, meta))
        print(f"  loaded {label:38s} n={len(rows)}  ({p.name})")

    if not loaded:
        raise SystemExit("ERROR: no database metric files found")

    # ---------------- side-by-side statistics ----------------
    comp_rows = []
    dec_rows = []
    print(f"\n{'database':38}{'n':>5}{'medDensity':>12}{'meanDensity':>12}"
          f"{'medConf':>10}{'strictOK':>10}{'medStrict%':>12}"
          f"{'orig>5%':>9}{'strict>5%':>11}")
    for label, tag, rows, meta in loaded:
        dens = np.array([fnum(r['informative_kmer_density']) for r in rows])
        conf = np.array([fnum(r['mean_contig_confidence_bpw']) for r in rows])
        orig = np.array([fnum(r['orig_phylum_contam_pct']) for r in rows])
        strict = np.array([fnum(r['strict_phylum_contam_pct']) for r in rows])
        qf = np.array([fnum(r['strict_qualifying_bp_frac']) for r in rows])
        sok = strict[~np.isnan(strict)]
        o5 = int((orig[~np.isnan(orig)] > 5).sum())
        s5 = int((sok > 5).sum()) if len(sok) else 0
        print(f"{label:38}{len(rows):>5}{np.nanmedian(dens):>12.6f}"
              f"{np.nanmean(dens):>12.6f}{np.nanmedian(conf):>10.5f}"
              f"{len(sok):>10}"
              f"{(np.median(sok) if len(sok) else float('nan')):>12.2f}"
              f"{o5:>9}{s5:>11}")
        info = db_info(tag)
        comp_rows.append({
            'database': label, 'tag': tag, 'n_genomes': len(rows),
            'reference_contains': info['contains'],
            'reference_lacks': info['lacks'],
            'is_refseq_derived': info['is_refseq_derived'],
            'is_gtdb_level': info['is_gtdb_level'],
            'is_capped': info['is_capped'],
            'hash_k2d_gb': (meta or {}).get('kraken2_db_hash_gb', ''),
            'kraken2_version': (meta or {}).get('kraken2_version', ''),
            'median_informative_kmer_density': round(float(np.nanmedian(dens)), 8),
            'mean_informative_kmer_density': round(float(np.nanmean(dens)), 8),
            'median_mean_contig_confidence': round(float(np.nanmedian(conf)), 8),
            'median_unclassified_pct': round(float(np.nanmedian(
                [fnum(r['unclassified_pct']) for r in rows])), 4),
            'median_orig_phylum_contam_pct': round(float(np.nanmedian(orig)), 4),
            'n_strict_scoreable': len(sok),
            'median_strict_phylum_contam_pct': (round(float(np.median(sok)), 4)
                                                if len(sok) else ''),
            'median_strict_qualifying_bp_frac': round(float(np.nanmedian(qf)), 6),
            'n_orig_gt5pct': o5,
            'n_strict_gt5pct': s5,
        })
        for d in range(0, 101, 10):
            dec_rows.append({
                'database': label, 'percentile': d,
                'informative_kmer_density': round(float(
                    np.nanpercentile(dens, d)), 8),
                'mean_contig_confidence': round(float(
                    np.nanpercentile(conf, d)), 8),
                'orig_phylum_contam_pct': round(float(
                    np.nanpercentile(orig, d)), 4),
            })

    # ---------------- correlations per database ----------------
    print(f"\nNovelty-controlled correlations per database "
          f"(cohort {args.cohort}):")
    corr = {}
    for label, tag, rows, meta in loaded:
        v5 = [fnum(r['v5_contamination']) for r in rows]
        ck = [fnum(r['checkm2_contamination']) for r in rows]
        k2 = [fnum(r['orig_phylum_contam_pct']) for r in rows]
        ctrls = [[fnum(r[c]) for r in rows] for c in NOVELTY]
        rk, pk, nk = spearman(k2, [fnum(r['informative_kmer_density'])
                                   for r in rows])
        r1, p1, n1 = spearman(v5, k2)
        pr1, pp1, pn1 = partial_spearman(v5, k2, ctrls)
        r2, p2, n2 = spearman(ck, k2)
        pr2, pp2, pn2 = partial_spearman(ck, k2, ctrls)
        print(f"  {label}")
        print(f"    metric vs density        rho={rk:+.3f} p={pk:9.3g} n={nk}")
        print(f"    V5      vs metric        rho={r1:+.3f} p={p1:9.3g}"
              f"   novelty-controlled rho={pr1:+.3f} p={pp1:9.3g}")
        print(f"    CheckM2 vs metric        rho={r2:+.3f} p={p2:9.3g}"
              f"   novelty-controlled rho={pr2:+.3f} p={pp2:9.3g}")
        corr[label] = {
            'metric_vs_density': {'rho': rk, 'p': pk, 'n': nk},
            'v5_vs_metric': {'rho': r1, 'p': p1, 'n': n1,
                             'partial_rho': pr1, 'partial_p': pp1},
            'checkm2_vs_metric': {'rho': r2, 'p': p2, 'n': n2,
                                  'partial_rho': pr2, 'partial_p': pp2},
        }

    with open(out_dir / 'kraken2_db_capacity_comparison.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()), delimiter='\t')
        w.writeheader(); w.writerows(comp_rows)
    with open(out_dir / 'kraken2_db_density_deciles.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dec_rows[0].keys()), delimiter='\t')
        w.writeheader(); w.writerows(dec_rows)

    # ---------------- reference content of each database tested -------------
    print("\nReference content of each database tested "
          "(decides what a low hit rate can license):")
    for label, tag, rows, meta in loaded:
        i = db_info(tag)
        print(f"  {label}")
        print(f"    contains: {i['contains']}")
        print(f"    lacks   : {i['lacks']}")

    # ---------------- three-way verdict ----------------
    # (i)   database capacity was limiting
    # (ii)  RefSeq lacks these lineages but a GTDB-level set covers them
    # (iii) novel even relative to GTDB
    # (iii) may ONLY be concluded if a GTDB-level database was actually tested.
    verdict = None
    if len(comp_rows) >= 2:
        base = next((r for r in comp_rows if db_info(r['tag']).get('is_capped')),
                    comp_rows[0])
        others = [r for r in comp_rows if r is not base]
        big = max(others, key=lambda r: r['median_informative_kmer_density'])
        d0 = base['median_informative_kmer_density'] or 1e-12
        d1 = big['median_informative_kmer_density']
        fold = d1 / d0 if d0 else float('inf')
        evaluable = big['n_strict_scoreable'] >= 0.5 * big['n_genomes']

        gtdb_rows = [r for r in comp_rows if db_info(r['tag']).get('is_gtdb_level')]
        gtdb_tested = bool(gtdb_rows)
        gtdb_evaluable = (gtdb_rows and
                          max(gtdb_rows,
                              key=lambda r: r['n_strict_scoreable'])['n_strict_scoreable']
                          >= 0.5 * gtdb_rows[0]['n_genomes'])

        if evaluable and fold >= 5:
            code = 'i_database_capacity'
            v = ('(i) DATABASE CAPACITY was limiting: informative k-mer density '
                 f'rose {fold:.1f}-fold with {big["database"]} and the strict '
                 f'metric became evaluable for {big["n_strict_scoreable"]}/'
                 f'{big["n_genomes"]} genomes. The corrected analysis replaces the '
                 'withdrawn claim.')
        elif gtdb_tested and gtdb_evaluable:
            code = 'ii_refseq_gap_gtdb_covers'
            v = ('(ii) REFSEQ GAP: the cohort is poorly represented in '
                 'RefSeq-derived databases but adequately covered by the '
                 'GTDB-level reference set, where the strict metric becomes '
                 'evaluable. Taxonomy-based verification of these MAGs requires a '
                 'GTDB-level database; the RefSeq-based analysis in the manuscript '
                 'could not have adjudicated them.')
        elif gtdb_tested:
            code = 'iii_novel_even_vs_gtdb'
            v = ('(iii) NOVEL EVEN RELATIVE TO GTDB: informative k-mer density '
                 f'remained low ({d1:.2e}) and the strict metric stayed '
                 'unevaluable for most of the cohort even against the GTDB-level '
                 'reference set MAGICC was trained on. No Kraken2-based metric can '
                 'adjudicate these genomes. This is the strongest finding and also '
                 'the most constraining: it means the real-world section cannot '
                 'lean on taxonomic verification for this cohort, and that V5 is '
                 'being asked to estimate contamination for lineages outside its '
                 'training distribution.')
        else:
            code = 'undecided_needs_gtdb_level_db'
            v = ('UNDECIDED -- a GTDB-level database has NOT been tested. Density '
                 f'is {d1:.2e} ({fold:.2f}x the capped database) against '
                 f'{big["database"]}, which is RefSeq-derived and contains almost '
                 'no MAG-derived references. Because this cohort is dominated by '
                 'uncultivated MAG-derived lineages, a low hit rate here shows only '
                 'that no RefSeq reference exists -- it does NOT license concluding '
                 'genuine taxonomic novelty. Build/test a GTDB-level database '
                 '(scripts/082_build_gtdb_kraken2_db.sh) before concluding.')
        verdict = {'code': code,
                   'fold_change_median_density': round(fold, 4),
                   'baseline_database': base['database'],
                   'largest_database': big['database'],
                   'strict_metric_evaluable_majority': bool(evaluable),
                   'gtdb_level_database_tested': gtdb_tested,
                   'databases_tested': {r['tag']: db_info(r['tag']) for r in comp_rows},
                   'conclusion': v}
        print(f"\nVERDICT [{code}]:\n  {v}")

    (out_dir / 'kraken2_db_capacity_summary.json').write_text(json.dumps({
        'cohort': args.cohort, 'novelty_controls': NOVELTY,
        'databases': comp_rows, 'correlations': corr, 'verdict': verdict,
    }, indent=2, default=str))
    print(f"\nWrote {out_dir / 'kraken2_db_capacity_comparison.tsv'}")
    print(f"Wrote {out_dir / 'kraken2_db_density_deciles.tsv'}")
    print(f"Wrote {out_dir / 'kraken2_db_capacity_summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
