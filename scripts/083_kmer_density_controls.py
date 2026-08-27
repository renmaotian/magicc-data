#!/usr/bin/env python3
"""
WS4.3 -- internal positive controls for informative-k-mer density.

Purpose
-------
"Informative-k-mer density is low" is only interpretable relative to what a
genome that the database *does* represent achieves. This script measures density
on reference genomes drawn so that the comparison is informative rather than
trivially easy, and reports their distribution alongside the 141-MAG cohort.

Control design
--------------
Controls are drawn from the project's own GTDB reference set
(``data/gtdb/selected_100k_genomes.tsv``) with two properties:

  1. **Not the genome used to build the database.** When the Kraken2 database is
     built with ``--one-per-species`` (scripts/82), the species representative is
     the first genome of each lineage in file order. Controls are drawn from the
     *non*-representatives, so a control's species is in the database but the
     control genome itself is not. That is a realistic positive control -- it
     measures the density a genome gets when its lineage is represented, without
     the self-hit that would make the test meaningless.
  2. **Stratified by how well its phylum is represented.** The whole question is
     novel-lineage sensitivity, so controls span:
        well_represented  -- phyla in the top decile by reference genome count
        sparse            -- phyla in the bottom tercile by count, and always
                             Patescibacteriota (CPR) and archaeal DPANN-like
                             phyla when present
     If density for the 141 MAGs falls below even the *sparse* controls, that is
     a clean result.

Density needs no taxonomy: it is computed directly from Kraken2's k-mer LCA
string as (k-mers with a non-zero taxid) / (all non-ambiguous k-mers), so this
script is self-contained and DB-agnostic.

Usage
-----
    python scripts/083_kmer_density_controls.py \
        --kraken2-db tools/kraken2_db_standard \
        --kraken2-bin /path/to/anaconda3/envs/kraken2_env/bin/kraken2 \
        --db-tag k2std --per-stratum 40 --threads 12

Outputs (``results/revision/contamination_evidence/``)
------------------------------------------------------
    density_controls_<tag>.tsv          per control genome
    density_controls_summary_<tag>.json strata distributions + cohort comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
GENOME_TSV = PROJECT_DIR / 'data' / 'gtdb' / 'selected_100k_genomes.tsv'
GENOME_DIR = PROJECT_DIR / 'data' / 'genomes'
EVID_DIR = PROJECT_DIR / 'results' / 'revision' / 'contamination_evidence'
SEED = 42

# Always include these sparsely-represented lineages: the question is precisely
# about novel-lineage sensitivity.
ALWAYS_SPARSE = ['Patescibacteriota', 'Nanoarchaeota', 'Micrarchaeota',
                 'Altarchaeota', 'Aenigmatarchaeota', 'Huberarchaeota']


def find_fasta(acc: str) -> Optional[Path]:
    d = GENOME_DIR / acc
    if not d.is_dir():
        return None
    for pat in ('*_genomic.fna', '*.fna', '*.fa', '*.fasta'):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def density_from_lca(lca: str) -> Tuple[int, int]:
    """Return (informative_kmers, total_non_ambiguous_kmers) from an LCA string."""
    inf = tot = 0
    for tok in lca.split():
        if ':' not in tok:
            continue
        left, right = tok.rsplit(':', 1)
        try:
            c = int(right)
        except ValueError:
            continue
        if left == 'A':
            continue
        tot += c
        if left != '0':
            inf += c
    return inf, tot


def run_kraken2(accs: List[str], paths: Dict[str, Path], db: Path, binpath: str,
                threads: int, workdir: Path) -> Dict[str, Tuple[int, int, int]]:
    """Batched Kraken2; returns accession -> (informative, total, n_contigs)."""
    workdir.mkdir(parents=True, exist_ok=True)
    combined = workdir / 'controls.fna'
    n_seq = 0
    with open(combined, 'w') as out:
        for a in accs:
            with open(paths[a]) as f:
                for line in f:
                    if line.startswith('>'):
                        cid = line[1:].split()[0] if line[1:].split() else 'c'
                        out.write(f'>{a}|{cid}\n')
                        n_seq += 1
                    else:
                        out.write(line)
    print(f"    combined FASTA: {len(accs)} genomes, {n_seq} sequences, "
          f"{combined.stat().st_size / 1e9:.2f} GB")
    out_f = workdir / 'controls.kraken2.txt'
    cmd = [binpath, '--db', str(db), '--threads', str(threads),
           '--output', str(out_f), '--report', str(workdir / 'controls.report'),
           '--use-names', str(combined)]
    print(f"    {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"ERROR: kraken2 failed:\n{p.stderr[-3000:]}")
    agg: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    with open(out_f) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5 or '|' not in parts[1]:
                continue
            acc = parts[1].split('|', 1)[0]
            inf, tot = density_from_lca(parts[4])
            agg[acc][0] += inf
            agg[acc][1] += tot
            agg[acc][2] += 1
    return {k: (v[0], v[1], v[2]) for k, v in agg.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--kraken2-db', required=True)
    ap.add_argument('--kraken2-bin',
                    default='/path/to/anaconda3/envs/kraken2_env/bin/kraken2')
    ap.add_argument('--db-tag', required=True)
    ap.add_argument('--per-stratum', type=int, default=40)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--out-dir', default=str(EVID_DIR))
    ap.add_argument('--keep-temp', action='store_true')
    args = ap.parse_args()

    random.seed(SEED)
    out_dir = Path(args.out_dir)
    db = Path(args.kraken2_db)
    if not (db / 'hash.k2d').is_file():
        sys.exit(f"ERROR: no hash.k2d in {db}")

    with open(GENOME_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    print(f"reference genomes: {len(rows):,}")

    # species representatives = first genome per lineage in file order,
    # matching scripts/082_build_gtdb_kraken2_db.py --one-per-species
    seen_lineage = set()
    is_rep: Dict[str, bool] = {}
    phylum_count: Dict[str, int] = defaultdict(int)
    for r in rows:
        lin = r.get('gtdb_taxonomy', '')
        acc = r.get('ncbi_accession') or r.get('gtdb_accession', '')
        rep = lin not in seen_lineage
        seen_lineage.add(lin)
        is_rep[acc] = rep
        phylum_count[r.get('phylum', '')] += 1

    counts = sorted(phylum_count.items(), key=lambda kv: -kv[1])
    n_ph = len(counts)
    top_decile = {p for p, _ in counts[:max(1, n_ph // 10)]}
    bottom_tercile = {p for p, _ in counts[-max(1, n_ph // 3):]}
    sparse_set = set(bottom_tercile) | {p for p in phylum_count
                                        if p in ALWAYS_SPARSE}
    print(f"phyla: {n_ph}   well-represented (top decile): {len(top_decile)}   "
          f"sparse: {len(sparse_set)}")
    for p in ALWAYS_SPARSE:
        if p in phylum_count:
            print(f"  always-sparse present: {p} "
                  f"({phylum_count[p]} reference genomes)")

    # candidate controls: NON-representatives (species in DB, genome not in DB)
    #
    # Stratum assignment order matters. Patescibacteriota has 1,609 genomes in
    # this reference set, so a purely count-based rule puts CPR in the TOP decile
    # and leaves the biologically-novel stratum almost empty. The lineages we
    # specifically care about therefore take precedence over the count rule.
    cand: Dict[str, List[Dict[str, str]]] = {
        'well_represented': [], 'cpr_dpann': [], 'rare_phylum': []}
    for r in rows:
        acc = r.get('ncbi_accession') or r.get('gtdb_accession', '')
        if is_rep.get(acc):
            continue
        ph = r.get('phylum', '')
        if ph in ALWAYS_SPARSE:
            cand['cpr_dpann'].append(r)
        elif ph in bottom_tercile:
            cand['rare_phylum'].append(r)
        elif ph in top_decile:
            cand['well_represented'].append(r)
    for k, v in cand.items():
        print(f"  candidate {k}: {len(v):,} non-representative genomes")

    selected: List[Tuple[str, str, Dict[str, str]]] = []
    paths: Dict[str, Path] = {}
    for stratum, pool in cand.items():
        random.shuffle(pool)
        taken = 0
        for r in pool:
            if taken >= args.per_stratum:
                break
            acc = r.get('ncbi_accession') or r.get('gtdb_accession', '')
            fa = find_fasta(acc)
            if fa is None:
                continue
            paths[acc] = fa
            selected.append((acc, stratum, r))
            taken += 1
        print(f"  selected {taken} for stratum {stratum}")
    if not selected:
        sys.exit("ERROR: no control genomes could be selected")

    tmp = Path(tempfile.mkdtemp(prefix=f'dens_{args.db_tag}_',
                                dir=str(out_dir)))
    try:
        print(f"Running Kraken2 on {len(selected)} control genomes "
              f"against {db.name} ...")
        agg = run_kraken2([a for a, _, _ in selected], paths, db,
                          args.kraken2_bin, args.threads, tmp)
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)

    ctrl_rows = []
    for acc, stratum, r in selected:
        inf, tot, nc = agg.get(acc, (0, 0, 0))
        ctrl_rows.append({
            'accession': acc, 'stratum': stratum,
            'phylum': r.get('phylum', ''), 'domain': r.get('domain', ''),
            'phylum_reference_genomes': phylum_count.get(r.get('phylum', ''), 0),
            'genome_size': r.get('genome_size', ''),
            'n_contigs_meta': r.get('contig_count', ''),
            'n_sequences_scored': nc,
            'informative_kmers': inf, 'total_kmers': tot,
            'informative_kmer_density': (round(inf / tot, 8) if tot else ''),
        })
    ctrl_tsv = out_dir / f'density_controls_{args.db_tag}.tsv'
    with open(ctrl_tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ctrl_rows[0].keys()), delimiter='\t')
        w.writeheader(); w.writerows(ctrl_rows)

    # ---- distributions, plus the cohort for comparison ----
    def pct(vals, q):
        return float(np.percentile(vals, q)) if len(vals) else float('nan')

    print(f"\nInformative-k-mer density against {db.name}:")
    print(f"  {'group':22}{'n':>5}{'p10':>12}{'p25':>12}{'median':>12}"
          f"{'p75':>12}{'p90':>12}")
    summary: Dict[str, object] = {}
    groups: List[Tuple[str, List[float]]] = []
    for stratum in ('well_represented', 'rare_phylum', 'cpr_dpann'):
        v = [float(r['informative_kmer_density']) for r in ctrl_rows
             if r['stratum'] == stratum and r['informative_kmer_density'] != '']
        groups.append((f'control_{stratum}', v))

    # the 141 cohort measured against the SAME database, if available
    suffix = f'_{args.db_tag}' if args.db_tag else ''
    mp = out_dir / f'kraken2_metrics{suffix}.tsv'
    cohort_v: List[float] = []
    if mp.is_file():
        with open(mp) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                if r['cohort'] == 'disagreement_141' and \
                        r['informative_kmer_density'] not in ('', None):
                    cohort_v.append(float(r['informative_kmer_density']))
        groups.append(('cohort_disagreement_141', cohort_v))
    else:
        print(f"  (note: {mp.name} not present -- run scripts/78 with "
              f"--db-tag {args.db_tag} to add the cohort row)")

    for name, v in groups:
        if not v:
            continue
        print(f"  {name:22}{len(v):>5}{pct(v,10):>12.6f}{pct(v,25):>12.6f}"
              f"{pct(v,50):>12.6f}{pct(v,75):>12.6f}{pct(v,90):>12.6f}")
        summary[name] = {'n': len(v), 'p10': pct(v, 10), 'p25': pct(v, 25),
                         'median': pct(v, 50), 'p75': pct(v, 75),
                         'p90': pct(v, 90), 'min': float(min(v)),
                         'max': float(max(v))}

    verdict = None
    novel_key = ('control_cpr_dpann' if summary.get('control_cpr_dpann')
                 else 'control_rare_phylum')
    if cohort_v and summary.get(novel_key):
        c_med = summary['cohort_disagreement_141']['median']
        s_p10 = summary[novel_key]['p10']
        s_med = summary[novel_key]['median']
        below = c_med < s_p10
        verdict = {
            'cohort_median': c_med,
            'novel_control_stratum': novel_key,
            'novel_control_median': s_med,
            'novel_control_p10': s_p10,
            'cohort_below_novel_control_p10': bool(below),
            'interpretation': (
                'Cohort density falls BELOW the 10th percentile of even the '
                'sparsely-represented (CPR/DPANN or rare-phylum) positive controls: these MAGs are less well '
                'represented than reference genomes from the thinnest lineages in '
                'the reference set. Clean result.'
                if below else
                'Cohort density is NOT below the novel-control floor, so low '
                'absolute density does not by itself establish novelty; interpret '
                'the strict-metric evaluability alongside.'),
        }
        print(f"\n  cohort median {c_med:.6f} vs sparse-control p10 {s_p10:.6f}"
              f"  -> {'BELOW' if below else 'NOT BELOW'}")

    (out_dir / f'density_controls_summary_{args.db_tag}.json').write_text(
        json.dumps({
            'kraken2_db': str(db),
            'kraken2_db_hash_bytes': (db / 'hash.k2d').stat().st_size,
            'control_design': (
                'non-representative reference genomes (species present in a '
                '--one-per-species database, genome itself absent), stratified by '
                'phylum representation'),
            'always_sparse_lineages': ALWAYS_SPARSE,
            'per_stratum_requested': args.per_stratum,
            'strata': summary, 'verdict': verdict,
            'controls_tsv': str(ctrl_tsv),
        }, indent=2, default=str))
    print(f"\nWrote {ctrl_tsv}")
    print(f"Wrote {out_dir / f'density_controls_summary_{args.db_tag}.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
