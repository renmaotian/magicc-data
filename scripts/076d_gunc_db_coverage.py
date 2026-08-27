#!/usr/bin/env python3
"""
WS4.1 -- reference coverage of the candidate GUNC databases.

GUNC's verdict is only informative when the query genome is represented in the
reference database (GUNC reports this itself as
``reference_representation_score`` = genes_retained_index x mean_hit_identity,
and forcibly zeroes the clade separation score when
``genes_retained_index <= 0.4``).

The control verification (``076b_gunc_control_verification.py``) found that 7 of 8
Kraken2-confirmed contaminated MAGs had ``reference_representation_score < 0.5``
against **proGenomes 2.1** (mean amino-acid hit identity ~0.5), i.e. GUNC had no
close reference and could not adjudicate. This script quantifies *why*, by
counting reference genomes per lineage in each database's bundled
genome-to-taxonomy table -- in particular for the two lineages that Sets
C_clean / D_clean are built from (Patescibacteria/CPR and Archaea).

Usage
-----
    python scripts/076d_gunc_db_coverage.py \
        --out-dir results/revision/gunc

Outputs
-------
    db_coverage_comparison.tsv    one row per database
    db_coverage_comparison.json   same, plus the lineage synonym lists used
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

# GUNC bundles one genome2taxonomy table per database.
DB_TAXONOMY_FILES = {
    'progenomes_2.1': 'genome2taxonomy_pg2.1ref.tsv',
    'progenomes_3': 'genome2taxonomy_pg3ref.tsv',
    'gtdb_95': 'genome2taxonomy_gtdb95ref.tsv',
    'gtdb_214': 'genome2taxonomy_gtdb214ref.tsv',
}

# CPR / Patescibacteria appear under many names across NCBI and GTDB taxonomies.
CPR_SYNONYMS = [
    'Patescibacteria', 'Parcubacteria', 'Microgenomates', 'Gracilibacteria',
    'Saccharibacteria', 'Absconditabacteria', 'Peregrinibacteria',
    'Dojkabacteria', 'Katanobacteria', 'Berkelbacteria', 'Candidate division',
]
ARCHAEA_SYNONYMS = ['Archaea']


def find_gunc_data_dir(explicit: str | None) -> Path:
    """Locate gunc/data/ inside the gunc_env conda environment."""
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise SystemExit(f"ERROR: --gunc-data-dir not found: {p}")
        return p
    candidates: List[Path] = []
    for base in (Path.home() / 'anaconda3', Path.home() / 'miniconda3',
                 Path.home() / 'mambaforge', Path.home() / 'miniforge3'):
        env = base / 'envs' / 'gunc_env' / 'lib'
        if env.is_dir():
            candidates.extend(sorted(env.glob('python3.*/site-packages/gunc/data')))
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit("ERROR: could not locate gunc/data/; pass --gunc-data-dir")


def count_matches(rows: List[Dict[str, str]], synonyms: List[str], field: str) -> int:
    lowered = [s.lower() for s in synonyms]
    return sum(1 for r in rows
               if any(s in r.get(field, '').lower() for s in lowered))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='results/revision/gunc')
    ap.add_argument('--gunc-data-dir', default=None)
    args = ap.parse_args()

    data_dir = find_gunc_data_dir(args.gunc_data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = ['database', 'taxonomy_file', 'n_reference_genomes', 'n_distinct_phyla',
              'n_cpr_patescibacteria', 'n_archaea',
              'n_genbank_prefixed', 'n_refseq_prefixed', 'available']
    rows = []
    for db, fname in DB_TAXONOMY_FILES.items():
        path = data_dir / fname
        if not path.is_file():
            rows.append({'database': db, 'taxonomy_file': fname,
                         'n_reference_genomes': '', 'n_distinct_phyla': '',
                         'n_cpr_patescibacteria': '', 'n_archaea': '',
                         'n_genbank_prefixed': '', 'n_refseq_prefixed': '',
                         'available': 'False'})
            continue
        with open(path) as f:
            tax = list(csv.DictReader(f, delimiter='\t'))
        rows.append({
            'database': db,
            'taxonomy_file': fname,
            'n_reference_genomes': len(tax),
            'n_distinct_phyla': len({r['phylum'] for r in tax}),
            'n_cpr_patescibacteria': count_matches(tax, CPR_SYNONYMS, 'phylum'),
            'n_archaea': count_matches(tax, ARCHAEA_SYNONYMS, 'kingdom'),
            'n_genbank_prefixed': sum(1 for r in tax if r['genome'].startswith('GB_')),
            'n_refseq_prefixed': sum(1 for r in tax if r['genome'].startswith('RS_')),
            'available': 'True',
        })

    out_tsv = out_dir / 'db_coverage_comparison.tsv'
    with open(out_tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        w.writeheader()
        w.writerows(rows)

    print(f"{'database':16}{'genomes':>10}{'phyla':>7}{'CPR':>7}{'Archaea':>9}"
          f"{'GB_':>8}{'RS_':>8}")
    for r in rows:
        if r['available'] != 'True':
            print(f"{r['database']:16}{'(taxonomy file not bundled)':>49}")
            continue
        print(f"{r['database']:16}{r['n_reference_genomes']:>10,}"
              f"{r['n_distinct_phyla']:>7,}{r['n_cpr_patescibacteria']:>7,}"
              f"{r['n_archaea']:>9,}{r['n_genbank_prefixed']:>8,}"
              f"{r['n_refseq_prefixed']:>8,}")

    (out_dir / 'db_coverage_comparison.json').write_text(json.dumps({
        'gunc_data_dir': str(data_dir),
        'cpr_synonyms': CPR_SYNONYMS,
        'archaea_synonyms': ARCHAEA_SYNONYMS,
        'note': ('GB_ = GenBank-derived GTDB entries (include MAGs from '
                 'uncultivated lineages); RS_ = RefSeq isolate genomes. '
                 'proGenomes 2.1 is isolate/specI-cluster based and therefore '
                 'thinly covers uncultivated lineages such as CPR.'),
        'caveat': ('Lineage counts use substring matching over two different '
                   'taxonomies (NCBI names in proGenomes, GTDB names in gtdb_*), '
                   'so per-lineage totals are not strictly commensurable. The '
                   'order-of-magnitude differences are robust, and the GB_/RS_ '
                   'split independently shows that only the GTDB databases '
                   'contain MAG-derived references.'),
        'rows': rows,
    }, indent=2))
    print(f"\nWrote {out_tsv}")
    print(f"Wrote {out_dir / 'db_coverage_comparison.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
