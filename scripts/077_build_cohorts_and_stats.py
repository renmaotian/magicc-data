#!/usr/bin/env python3
"""
WS4.3/4.5 -- cohort definition and assembly statistics for the contamination
evidence analysis.

Context
-------
The manuscript claims (R2-M5) that Kraken2 "independently confirms" cross-phylum
contamination in 109/141 (77.3%) of the MAGs that CheckM2 calls HQ but MAGICC V5
downgrades.  WS4.1 control verification raised the concern that the Kraken2
metric used (``phylum_contam_pct`` = non-dominant-phylum bp / total bp, where
each contig contributes its *full length* to whatever its k-mer LCA resolves to)
is inflated for **novel lineages**, whose contigs receive diffuse, low-support
taxonomic assignments.

Testing that requires a matched negative-control cohort: MAGs that BOTH tools
agree are clean, matched to the 141 on taxonomic novelty and assembly
characteristics.  This script builds the cohorts and the matching covariates.

Cohorts (from ``results/ncbi_comparison/mag_comparison.tsv``, n = 1000 MAGs)
--------------------------------------------------------------------------
    disagreement_141  CheckM2 HQ and V5 not HQ   -> the cohort behind the claim
    agree_clean       both tools HQ              -> candidate matched controls
    v5hq_only         V5 HQ and CheckM2 not HQ   -> reported for completeness
    neither_hq        neither tool HQ            -> reported for completeness

MIMAG HQ is defined here exactly as in the original analysis:
completeness >= 90 and contamination <= 5.

Usage
-----
    python scripts/077_build_cohorts_and_stats.py --threads 16

Outputs (``results/revision/contamination_evidence/``)
------------------------------------------------------
    cohort_genome_stats.tsv   one row per MAG: cohort, tool calls, assembly stats
    cohort_summary.json       cohort sizes and per-cohort stat distributions
    kraken2_todo.txt          accessions in agree_clean that still need Kraken2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
MAG_DIR = PROJECT_DIR / 'data' / 'ncbi' / 'mags'
MAG_COMPARISON = PROJECT_DIR / 'results' / 'ncbi_comparison' / 'mag_comparison.tsv'
KRAKEN2_EXISTING = [
    PROJECT_DIR / 'data' / 'ncbi' / 'kraken2_results',
    PROJECT_DIR / 'data' / 'ncbi' / 'kraken2_results_v5hq',
]
OUT_DIR = PROJECT_DIR / 'results' / 'revision' / 'contamination_evidence'


# ---------------------------------------------------------------------------
# Assembly statistics
# ---------------------------------------------------------------------------
def contig_lengths_and_gc(fasta_path: Path) -> Tuple[List[int], int, int]:
    """Return (contig lengths, total GC count, total ACGT count)."""
    lengths: List[int] = []
    cur = 0
    gc = 0
    acgt = 0
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                if cur:
                    lengths.append(cur)
                cur = 0
            else:
                s = line.strip().upper()
                cur += len(s)
                gc += s.count('G') + s.count('C')
                acgt += s.count('A') + s.count('C') + s.count('G') + s.count('T')
    if cur:
        lengths.append(cur)
    return lengths, gc, acgt


def n50_of(lengths: List[int]) -> int:
    if not lengths:
        return 0
    total = sum(lengths)
    acc = 0
    for L in sorted(lengths, reverse=True):
        acc += L
        if acc >= total / 2:
            return L
    return 0


def stats_worker(args: Tuple[str, str]) -> Optional[Dict[str, object]]:
    acc, path = args
    try:
        lengths, gc, acgt = contig_lengths_and_gc(Path(path))
    except OSError as e:
        print(f"  WARN {acc}: {e}")
        return None
    if not lengths:
        return None
    total = sum(lengths)
    return {
        'accession': acc,
        'total_bp': total,
        'n_contigs': len(lengths),
        'n50': n50_of(lengths),
        'longest_contig': max(lengths),
        'mean_contig_bp': round(total / len(lengths), 1),
        'gc_pct': round(gc / acgt * 100, 3) if acgt else '',
        'log10_total_bp': round(__import__('math').log10(total), 4),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=16,
                    help='workers for assembly-stat computation (default 16)')
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(MAG_COMPARISON) as f:
        mags = list(csv.DictReader(f, delimiter='\t'))
    print(f"MAGs in comparison table: {len(mags)}")

    def is_hq(r, tool):
        return (float(r[f'{tool}_completeness']) >= 90.0
                and float(r[f'{tool}_contamination']) <= 5.0)

    for r in mags:
        c, v = is_hq(r, 'checkm2'), is_hq(r, 'v5')
        if c and not v:
            r['cohort'] = 'disagreement_141'
        elif c and v:
            r['cohort'] = 'agree_clean'
        elif v and not c:
            r['cohort'] = 'v5hq_only'
        else:
            r['cohort'] = 'neither_hq'

    counts: Dict[str, int] = {}
    for r in mags:
        counts[r['cohort']] = counts.get(r['cohort'], 0) + 1
    for k in ('disagreement_141', 'agree_clean', 'v5hq_only', 'neither_hq'):
        print(f"  {k:20s} {counts.get(k, 0):>5}")

    # ---- assembly statistics ----
    jobs = []
    for r in mags:
        p = MAG_DIR / f"{r['accession']}.fna"
        if p.is_file():
            jobs.append((r['accession'], str(p)))
    print(f"Computing assembly stats for {len(jobs)} MAG FASTAs "
          f"using {args.threads} worker(s) ...")
    with Pool(processes=max(1, args.threads)) as pool:
        results = pool.map(stats_worker, jobs, chunksize=8)
    stats = {d['accession']: d for d in results if d}
    print(f"  stats computed: {len(stats)}")

    # ---- which genomes already have Kraken2 output ----
    have_kraken = set()
    for d in KRAKEN2_EXISTING:
        if d.is_dir():
            for p in d.glob('*.kraken2.txt'):
                have_kraken.add(p.name.replace('.kraken2.txt', ''))
    print(f"  existing Kraken2 per-sequence outputs: {len(have_kraken)}")

    header = ['accession', 'cohort',
              'v5_completeness', 'v5_contamination', 'v5_mimag',
              'checkm2_completeness', 'checkm2_contamination', 'checkm2_mimag',
              'total_bp', 'log10_total_bp', 'n_contigs', 'n50',
              'longest_contig', 'mean_contig_bp', 'gc_pct',
              'has_kraken2_output']
    out_rows = []
    for r in mags:
        s = stats.get(r['accession'])
        if s is None:
            continue
        row = {k: r.get(k, '') for k in header if k in r}
        row.update({k: v for k, v in s.items() if k != 'accession'})
        row['accession'] = r['accession']
        row['cohort'] = r['cohort']
        row['has_kraken2_output'] = str(r['accession'] in have_kraken)
        out_rows.append(row)

    out_tsv = out_dir / 'cohort_genome_stats.tsv'
    with open(out_tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        w.writeheader()
        w.writerows(out_rows)

    todo = [r['accession'] for r in out_rows
            if r['cohort'] == 'agree_clean' and r['has_kraken2_output'] == 'False']
    (out_dir / 'kraken2_todo.txt').write_text('\n'.join(todo) + '\n')

    # ---- per-cohort stat distributions ----
    def dist(cohort, field):
        vals = [float(r[field]) for r in out_rows
                if r['cohort'] == cohort and r[field] not in ('', None)]
        if not vals:
            return None
        vals.sort()
        n = len(vals)
        return {'n': n, 'median': vals[n // 2],
                'q1': vals[n // 4], 'q3': vals[3 * n // 4],
                'min': vals[0], 'max': vals[-1],
                'mean': round(sum(vals) / n, 4)}

    summary = {
        'n_mags': len(out_rows),
        'cohort_counts': counts,
        'hq_definition': 'completeness >= 90 and contamination <= 5 (MIMAG HQ)',
        'distributions': {
            c: {f: dist(c, f) for f in
                ('total_bp', 'n_contigs', 'n50', 'gc_pct',
                 'v5_contamination', 'checkm2_contamination')}
            for c in ('disagreement_141', 'agree_clean', 'v5hq_only', 'neither_hq')
        },
        'kraken2_existing': len(have_kraken),
        'kraken2_todo_agree_clean': len(todo),
        'outputs': {'stats': str(out_tsv), 'todo': str(out_dir / 'kraken2_todo.txt')},
    }
    (out_dir / 'cohort_summary.json').write_text(json.dumps(summary, indent=2))

    print()
    print(f"{'cohort':20}{'n':>5}{'med_bp':>12}{'med_ctg':>9}{'med_N50':>10}{'med_GC':>8}")
    for c in ('disagreement_141', 'agree_clean', 'v5hq_only', 'neither_hq'):
        d = summary['distributions'][c]
        if not d['total_bp']:
            continue
        print(f"{c:20}{d['total_bp']['n']:>5}{d['total_bp']['median']:>12,.0f}"
              f"{d['n_contigs']['median']:>9,.0f}{d['n50']['median']:>10,.0f}"
              f"{d['gc_pct']['median']:>8.1f}")
    print(f"\nWrote {out_tsv}")
    print(f"Wrote {out_dir / 'cohort_summary.json'}")
    print(f"Wrote {out_dir / 'kraken2_todo.txt'} ({len(todo)} accessions need Kraken2)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
