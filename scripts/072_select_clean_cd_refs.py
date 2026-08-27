#!/usr/bin/env python3
"""
WS1.1 — Select the clean (strictly held-out) reference genomes for Sets C_clean/D_clean.

The submitted Sets C and D drew their dominant reference genomes from train+val+test
(985/1000 of Set C dominants are TRAIN genomes). The held-out test split is the only
clean source of Patescibacteriota / archaeal references, because reference curation took
ALL 1,609 Patescibacteriota and ALL 1,976 archaeal genomes of the 277,183-genome
filtered pool into the 100,000-genome working set (zero unused).

  Set C_clean dominants: 100 of the 161 Patescibacteriota in data/splits/test_genomes.tsv
                         seed 7001
  Set D_clean dominants: 100 of the 198 Archaea         in data/splits/test_genomes.tsv
                         seed 7002

Sampling is done with numpy Generator.choice(replace=False) over the candidate table
sorted by gtdb_accession, so the selection depends only on (candidate set, seed) and not
on pandas/row-order behaviour.

`fasta_path` in the split files points at a historical project root
(/path/to/magicc/...). It is re-rooted to this checkout and the existence
of every file is verified.

Outputs:
  data/benchmarks/set_C_clean/reference_selection.tsv
  data/benchmarks/set_D_clean/reference_selection.tsv
  results/revision/ws1_1_reference_selection_summary.json

Usage:
    python scripts/072_select_clean_cd_refs.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'
RESULTS_DIR = PROJECT_DIR / 'results' / 'revision'

LEGACY_ROOTS = ['/path/to/magicc',
                '/path/to/magicc-legacy']

N_PER_SET = 100
SEEDS = {'C_clean': 7001, 'D_clean': 7002}

KEEP_COLS = ['gtdb_accession', 'ncbi_accession', 'gcf_accession', 'domain', 'phylum',
             'gtdb_taxonomy', 'genome_size', 'contig_count', 'n50_contigs',
             'longest_contig', 'checkm2_completeness', 'checkm2_contamination',
             'fasta_path', 'download_dir']


def resolve_path(p: str) -> str:
    """Re-root a historical absolute fasta_path onto this checkout."""
    for root in LEGACY_ROOTS:
        if p.startswith(root):
            return p.replace(root, str(PROJECT_DIR), 1)
    return p


def select(candidates: pd.DataFrame, n: int, seed: int, label: str) -> pd.DataFrame:
    cand = candidates.sort_values('gtdb_accession').reset_index(drop=True)
    if len(cand) < n:
        raise SystemExit(f'FATAL: only {len(cand)} {label} candidates, need {n}')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(cand), size=n, replace=False))
    sel = cand.iloc[idx].reset_index(drop=True)
    sel.insert(0, 'ref_index', np.arange(n))
    sel['selection_seed'] = seed
    sel['source_split'] = 'test'
    return sel


def main():
    print('=' * 78)
    print('WS1.1 — clean reference selection for Sets C_clean / D_clean')
    print('=' * 78)

    test = pd.read_csv(SPLITS_DIR / 'test_genomes.tsv', sep='\t')
    print(f'\ntest split: {len(test)} genomes')

    pool = {
        'C_clean': test[test['phylum'] == 'Patescibacteriota'].copy(),
        'D_clean': test[test['domain'] == 'Archaea'].copy(),
    }
    print(f'  Patescibacteriota in test split: {len(pool["C_clean"])}')
    print(f'  Archaea in test split:           {len(pool["D_clean"])}')

    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'generated_by': 'scripts/072_select_clean_cd_refs.py',
        'source': 'data/splits/test_genomes.tsv',
        'n_test_split_genomes': int(len(test)),
        'sampling': ('numpy.random.default_rng(seed).choice(n_candidates, size=100, '
                     'replace=False) over candidates sorted by gtdb_accession'),
        'sets': {},
    }

    for label, cand in pool.items():
        seed = SEEDS[label]
        sel = select(cand, N_PER_SET, seed, label)
        sel = sel[['ref_index'] + [c for c in KEEP_COLS if c in sel.columns]
                  + ['selection_seed', 'source_split']]
        sel['fasta_path_resolved'] = [resolve_path(p) for p in sel['fasta_path']]

        missing = [p for p in sel['fasta_path_resolved'] if not Path(p).exists()]
        if missing:
            raise SystemExit(f'FATAL: {len(missing)} reference FASTA files missing for '
                             f'{label}, e.g. {missing[:3]}')

        out_dir = BENCHMARK_DIR / f'set_{label}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'reference_selection.tsv'
        sel.to_csv(out_path, sep='\t', index=False)

        gs = sel['genome_size'].astype(float)
        cc = sel['contig_count'].astype(float)
        n50 = sel['n50_contigs'].astype(float)
        print(f'\nSet {label}: seed={seed}, {len(sel)} of {len(cand)} candidates')
        print(f'  genome size  Mbp: median {gs.median()/1e6:.3f}  '
              f'min {gs.min()/1e6:.3f}  max {gs.max()/1e6:.3f}')
        print(f'  contig count    : median {cc.median():.0f}  '
              f'min {cc.min():.0f}  max {cc.max():.0f}')
        print(f'  N50 kbp         : median {n50.median()/1e3:.1f}  '
              f'min {n50.min()/1e3:.1f}  max {n50.max()/1e3:.1f}')
        print(f'  phyla: {sel["phylum"].value_counts().to_dict()}')
        print(f'  classes: {sel["gtdb_taxonomy"].str.split(";").str[2].nunique()} unique')
        print(f'  families: {sel["gtdb_taxonomy"].str.split(";").str[4].nunique()} unique')
        print(f'  genera:   {sel["gtdb_taxonomy"].str.split(";").str[5].nunique()} unique')
        print(f'  species:  {sel["gtdb_taxonomy"].str.split(";").str[6].nunique()} unique')
        print(f'  wrote {out_path}')

        summary['sets'][f'set_{label}'] = {
            'seed': seed,
            'n_selected': int(len(sel)),
            'n_candidates_in_test_split': int(len(cand)),
            'selection_criterion': ('phylum == Patescibacteriota' if label == 'C_clean'
                                    else 'domain == Archaea'),
            'output': str(out_path.relative_to(PROJECT_DIR)),
            'genome_size_bp': {'median': float(gs.median()), 'min': float(gs.min()),
                               'max': float(gs.max()), 'mean': float(gs.mean())},
            'contig_count': {'median': float(cc.median()), 'min': int(cc.min()),
                             'max': int(cc.max())},
            'n50_contigs': {'median': float(n50.median()), 'min': int(n50.min()),
                            'max': int(n50.max())},
            'phylum_counts': sel['phylum'].value_counts().to_dict(),
            'n_unique_classes': int(sel['gtdb_taxonomy'].str.split(';').str[2].nunique()),
            'n_unique_orders': int(sel['gtdb_taxonomy'].str.split(';').str[3].nunique()),
            'n_unique_families': int(sel['gtdb_taxonomy'].str.split(';').str[4].nunique()),
            'n_unique_genera': int(sel['gtdb_taxonomy'].str.split(';').str[5].nunique()),
            'n_unique_species': int(sel['gtdb_taxonomy'].str.split(';').str[6].nunique()),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sp = RESULTS_DIR / 'ws1_1_reference_selection_summary.json'
    with open(sp, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {sp}\nDONE')


if __name__ == '__main__':
    main()
