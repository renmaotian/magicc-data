#!/usr/bin/env python3
"""
WS4.1 -- GUNC control verification.

Verifies the GUNC installation against controls with *known* answers rather than
merely checking that the tool executes:

Negative controls
    Finished (1-2 contig) NCBI pure-culture genomes on which both MAGICC V5 and
    CheckM2 report >99.5% completeness and <1.5% contamination.
    Expectation: GUNC pass = True, clade separation score (CSS) near 0.

Positive controls
    MAGs from the 141-MAG MAGICC-vs-CheckM2 disagreement cohort that Kraken2
    independently shows to carry high cross-*phylum* contamination
    (`results/ncbi_comparison/kraken2_contamination_analysis.tsv`).
    Expectation: GUNC fail = contamination flagged.

Reads the normalized TSVs produced by ``scripts/076_run_gunc.py`` and emits a
joined verification table plus a pass/fail verdict per genome.

Usage
-----
    python scripts/076b_gunc_control_verification.py \
        --negative-dir results/revision/gunc/controls/negative \
        --positive-dir results/revision/gunc/controls/positive \
        --out-dir     results/revision/gunc/controls
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent
KRAKEN2_TSV = PROJECT_DIR / 'results' / 'ncbi_comparison' / 'kraken2_contamination_analysis.tsv'
PURE_TSV = PROJECT_DIR / 'results' / 'ncbi_comparison' / 'pure_culture_comparison.tsv'


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def fnum(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_normalized(run_dir: Path) -> Dict[str, Dict[str, str]]:
    tsv = run_dir / 'gunc_normalized.tsv'
    if not tsv.is_file():
        raise SystemExit(f"ERROR: missing {tsv}; run scripts/076_run_gunc.py first")
    return {r['genome']: r for r in read_tsv(tsv)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--negative-dir', required=True)
    ap.add_argument('--positive-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    neg = load_normalized(Path(args.negative_dir))
    pos = load_normalized(Path(args.positive_dir))

    kraken = {r['accession']: r for r in read_tsv(KRAKEN2_TSV)}
    pure = {r['accession']: r for r in read_tsv(PURE_TSV)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = [
        'control_type', 'genome',
        # --- Kraken2 evidence (and the diagnostics that qualify it) ---
        'kraken2_phylum_contam_pct', 'kraken2_phylum_dominant_pct',
        'kraken2_n_phyla', 'kraken2_unclassified_pct',
        'kraken2_n_contigs', 'kraken2_n_classified_contigs',
        # --- tool estimates ---
        'magicc_v5_completeness', 'magicc_v5_contamination',
        'checkm2_completeness', 'checkm2_contamination',
        # --- GUNC verdict ---
        'gunc_css', 'gunc_pass', 'n_effective_surplus_clades',
        'taxonomic_level', 'gunc_contamination_portion',
        'gunc_n_genes_called', 'gunc_n_genes_mapped',
        # --- GUNC reference-coverage diagnostics: these decide whether GUNC's
        #     verdict is informative at all (RRS < 0.5 => poorly represented) ---
        'gunc_genes_retained_index', 'gunc_mean_hit_identity',
        'gunc_reference_representation_score', 'gunc_well_powered',
        'expected', 'as_expected',
    ]
    rows = []
    counts = {'negative': {'ok': 0, 'n': 0}, 'positive': {'ok': 0, 'n': 0}}

    for kind, table, expected in [('negative', neg, 'pass'), ('positive', pos, 'fail')]:
        for genome in sorted(table):
            g = table[genome]
            verdict = g.get('gunc_pass', 'NA').strip()
            if kind == 'negative':
                as_expected = (verdict in ('True', 'true', 'TRUE'))
                k = {}
                m = pure.get(genome, {})
            else:
                as_expected = (verdict in ('False', 'false', 'FALSE'))
                k = kraken.get(genome, {})
                m = k
            counts[kind]['n'] += 1
            counts[kind]['ok'] += int(as_expected)
            rrs = fnum(g.get('reference_representation_score', ''))
            rows.append({
                'control_type': kind,
                'genome': genome,
                'kraken2_phylum_contam_pct': k.get('phylum_contam_pct', ''),
                'kraken2_phylum_dominant_pct': k.get('phylum_dominant_pct', ''),
                'kraken2_n_phyla': k.get('phylum_n_taxa', ''),
                'kraken2_unclassified_pct': k.get('unclassified_pct', ''),
                'kraken2_n_contigs': k.get('n_contigs', ''),
                'kraken2_n_classified_contigs': k.get('n_classified', ''),
                'magicc_v5_completeness': m.get('v5_completeness', ''),
                'magicc_v5_contamination': m.get('v5_contamination', ''),
                'checkm2_completeness': m.get('checkm2_completeness', ''),
                'checkm2_contamination': m.get('checkm2_contamination', ''),
                'gunc_css': g.get('gunc_css', ''),
                'gunc_pass': verdict,
                'n_effective_surplus_clades': g.get('n_effective_surplus_clades', ''),
                'taxonomic_level': g.get('taxonomic_level', ''),
                'gunc_contamination_portion': g.get('contamination_portion', ''),
                'gunc_n_genes_called': g.get('n_genes_called', ''),
                'gunc_n_genes_mapped': g.get('n_genes_mapped', ''),
                'gunc_genes_retained_index': g.get('genes_retained_index', ''),
                'gunc_mean_hit_identity': g.get('mean_hit_identity', ''),
                'gunc_reference_representation_score': g.get(
                    'reference_representation_score', ''),
                # GUNC's verdict is only informative when the genome is
                # adequately represented in the reference DB.
                'gunc_well_powered': ('NA' if rrs is None else str(rrs >= 0.5)),
                'expected': expected,
                'as_expected': str(as_expected),
            })

    out_tsv = out_dir / 'control_verification.tsv'
    with open(out_tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        w.writeheader()
        w.writerows(rows)

    # ---- console report ----
    def fmt(v, width=6, prec=2):
        x = fnum(v)
        return f'{x:{width}.{prec}f}' if x is not None else ' ' * (width - 2) + 'NA'

    for kind in ('negative', 'positive'):
        sel = [r for r in rows if r['control_type'] == kind]
        print(f"\n=== {kind.upper()} CONTROLS "
              f"(expected GUNC {'pass' if kind == 'negative' else 'fail'}) ===")
        print(f"{'genome':<20}{'K2_phyl%':>9}{'K2_dom%':>8}{'V5_cont':>9}{'CK2_cont':>9}"
              f"{'GUNC_CSS':>9}{'pass':>6}{'surplus':>8}{'taxlevel':>9}"
              f"{'meanID':>7}{'RRS':>6}{'powered':>8}  ok")
        for r in sel:
            print(f"{r['genome']:<20}{fmt(r['kraken2_phylum_contam_pct'], 9):>9}"
                  f"{fmt(r['kraken2_phylum_dominant_pct'], 8):>8}"
                  f"{fmt(r['magicc_v5_contamination'], 9):>9}"
                  f"{fmt(r['checkm2_contamination'], 9):>9}"
                  f"{fmt(r['gunc_css'], 9, 3):>9}"
                  f"{r['gunc_pass']:>6}"
                  f"{fmt(r['n_effective_surplus_clades'], 8, 2):>8}"
                  f"{r['taxonomic_level']:>9}"
                  f"{fmt(r['gunc_mean_hit_identity'], 7):>7}"
                  f"{fmt(r['gunc_reference_representation_score'], 6):>6}"
                  f"{r['gunc_well_powered']:>8}"
                  f"  {'YES' if r['as_expected'] == 'True' else 'NO'}")
        c = counts[kind]
        print(f"  -> {c['ok']}/{c['n']} as expected")
        powered = [r for r in sel if r['gunc_well_powered'] == 'True']
        print(f"  -> GUNC well-powered (reference_representation_score >= 0.5): "
              f"{len(powered)}/{len(sel)}")

    def powered_count(kind):
        sel = [r for r in rows if r['control_type'] == kind]
        return {'n': len(sel),
                'well_powered': sum(r['gunc_well_powered'] == 'True' for r in sel)}

    summary = {
        'negative_controls': counts['negative'],
        'positive_controls': counts['positive'],
        'negative_source': 'data/ncbi/pure_culture (finished, 1-2 contigs, both tools HQ)',
        'positive_source': ('data/ncbi/mags, 141-MAG disagreement cohort, '
                            'Kraken2 cross-phylum contamination >40%'),
        'gunc_reference_coverage': {
            'negative': powered_count('negative'),
            'positive': powered_count('positive'),
        },
        'gunc_pass_criterion': ('pass.GUNC = False iff adjusted clade separation '
                                'score (CSS) > 0.45 at the max-CSS taxonomic level; '
                                'CSS is forced to 0 when genes_retained_index <= 0.4'),
        'interpretation_caveat': (
            'GUNC CSS measures whether taxonomic discordance separates cleanly by '
            'contig. A pass on a genome with reference_representation_score < 0.5 is '
            'NOT evidence of cleanliness -- GUNC has insufficient reference coverage '
            'to adjudicate. Likewise, the Kraken2 phylum_contam_pct metric '
            '(1 - dominant phylum fraction) is inflated for novel lineages whose '
            'LCA assignments are diffuse rather than bimodal.'),
        'table': str(out_tsv),
    }
    (out_dir / 'control_verification_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_tsv}")
    print(f"Wrote {out_dir / 'control_verification_summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
