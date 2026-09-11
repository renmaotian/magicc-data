#!/usr/bin/env python3
"""
WS4.3 -- reproduce the original Kraken2 contamination metric and compute a
strict, confidence-aware alternative.

Why
---
The published claim (R2-M5) rests on ``phylum_contam_pct`` from
``scripts/61_kraken2_contamination_check.py``:

    phylum_contam_pct = (resolved bp not in the dominant phylum) / total bp x 100

Every contig contributes its **entire length** to whichever phylum its Kraken2
LCA label resolves to, with no requirement on how many k-mers supported that
label.  Inspection of the retained per-sequence output shows labels supported by
a single k-mer, e.g.

    C  JAGPDB010000013.1  Desulfoluna limicola (taxid 2810562)  49286  0:24624 2810562:1 0:24627

49,286 bp attributed to a different phylum on the evidence of 1 k-mer out of
49,252.  For genomes from lineages absent from the Kraken2 database this
produces diffuse, essentially arbitrary phylum assignments that the metric
scores as contamination.  That is a *novelty* signal, not a contamination
signal.

What this script does
---------------------
Kraken2's per-sequence output retains the full k-mer -> taxon mapping, so
per-contig confidence can be recomputed offline; the 141-MAG cohort therefore
needs no Kraken2 re-run.  For each contig we parse the LCA string
(``taxid:count`` runs, ``0`` = no DB match, ``A`` = ambiguous nucleotides) and
compute:

    total_kmers        all k-mers except ambiguous ('A') runs
    informative_kmers  k-mers with a non-zero taxid
    clade_kmers        k-mers mapping inside the clade rooted at the call
    confidence         clade_kmers / total_kmers   (Kraken2's own definition)

Two genome-level metrics are then emitted:

    ORIGINAL  faithful reproduction of scripts/61 (validated against its output)
    STRICT    a contig qualifies only if it is classified at or below phylum
              rank, confidence >= --min-confidence, informative_kmers >=
              --min-informative-kmers, and length >= --min-contig-bp.
              Unclassified and low-confidence bp are EXCLUDED from the
              denominator instead of being silently treated as clean.
              strict_phylum_contam_pct = qualifying bp outside the dominant
              phylum / qualifying bp x 100

It also emits the key novelty covariate, ``informative_kmer_density``
(informative k-mers / total k-mers over the whole genome), which measures how
well the genome is represented in the Kraken2 database and is independent of
*which* taxa the k-mers point to -- making it a legitimate matching variable.

Usage
-----
    # run Kraken2 for genomes that lack output, then score everything
    python scripts/78_kraken2_strict_metric.py --run-missing --threads 16

    # score only, using existing per-sequence outputs
    python scripts/78_kraken2_strict_metric.py

    # confirm that batched Kraken2 == per-genome Kraken2
    python scripts/78_kraken2_strict_metric.py --verify-batching 5 --threads 8

Outputs (``results/revision/contamination_evidence/``)
------------------------------------------------------
    kraken2_metrics.tsv          per genome: original + strict metrics, novelty
    kraken2_contig_detail.tsv.gz per contig: length, call, confidence, phylum
    kraken2_metrics_summary.json thresholds, validation, per-cohort summaries
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
# Database and per-genome-output locations are overridable so the same analysis
# can be repeated across Kraken2 databases of different capacity (see WS4.3:
# the original analysis used the CAPPED k2_standard_08gb build).
KRAKEN2_DB = PROJECT_DIR / 'tools' / 'kraken2_db'
KRAKEN2_BIN = 'kraken2'
# nodes.dmp/names.dmp location. Prebuilt Kraken2 index tarballs do not always
# ship the flat NCBI taxdump, so this is overridable; both databases use NCBI
# taxids, so a single taxdump can resolve lineages for either.
TAXONOMY_DIR = PROJECT_DIR / 'tools' / 'kraken2_db'
MAG_DIR = PROJECT_DIR / 'data' / 'ncbi' / 'mags'
EVID_DIR = PROJECT_DIR / 'results' / 'revision' / 'contamination_evidence'
KRAKEN2_DIRS = [
    PROJECT_DIR / 'data' / 'ncbi' / 'kraken2_results',
    PROJECT_DIR / 'data' / 'ncbi' / 'kraken2_results_v5hq',
    PROJECT_DIR / 'data' / 'ncbi' / 'kraken2_results_agree_clean',
]
NEW_KRAKEN2_DIR = PROJECT_DIR / 'data' / 'ncbi' / 'kraken2_results_agree_clean'

MAIN_RANKS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
TAXID_RE = re.compile(r'\(taxid\s+(\d+)\)')


# ---------------------------------------------------------------------------
# NCBI taxonomy
# ---------------------------------------------------------------------------
def _k2_version(binpath: str) -> str:
    try:
        p = subprocess.run([binpath, '--version'], capture_output=True, text=True, timeout=60)
        return (p.stdout + p.stderr).strip().splitlines()[0]
    except Exception as e:
        return f'unknown ({e})'


def load_taxonomy() -> Tuple[Dict[int, int], Dict[int, str], Dict[int, str]]:
    parent: Dict[int, int] = {}
    rank: Dict[int, str] = {}
    with open(TAXONOMY_DIR / 'nodes.dmp') as f:
        for line in f:
            p = [x.strip() for x in line.split('|')]
            if len(p) >= 3:
                tid, pid = int(p[0]), int(p[1])
                parent[tid] = pid
                rank[tid] = p[2]
    name: Dict[int, str] = {}
    with open(TAXONOMY_DIR / 'names.dmp') as f:
        for line in f:
            p = [x.strip() for x in line.split('|')]
            if len(p) >= 4 and p[3] == 'scientific name':
                name[int(p[0])] = p[1]
    return parent, rank, name


class Taxonomy:
    def __init__(self):
        self.parent, self.rank, self.name = load_taxonomy()
        self._lineage_cache: Dict[int, Dict[str, str]] = {}
        self._ancestors_cache: Dict[int, frozenset] = {}

    def lineage(self, taxid: int) -> Dict[str, str]:
        """{rank: name} for the main ranks."""
        if taxid in self._lineage_cache:
            return self._lineage_cache[taxid]
        out: Dict[str, str] = {}
        cur = taxid
        seen = set()
        while cur and cur not in seen and cur != 1:
            seen.add(cur)
            r = self.rank.get(cur, 'no rank')
            if r in MAIN_RANKS and r not in out:
                out[r] = self.name.get(cur, f'taxid_{cur}')
            cur = self.parent.get(cur)
        self._lineage_cache[taxid] = out
        return out

    def ancestors(self, taxid: int) -> frozenset:
        """Set of taxid and all its ancestors (for clade membership tests)."""
        if taxid in self._ancestors_cache:
            return self._ancestors_cache[taxid]
        out = set()
        cur = taxid
        while cur and cur not in out:
            out.add(cur)
            if cur == 1:
                break
            cur = self.parent.get(cur)
        fs = frozenset(out)
        self._ancestors_cache[taxid] = fs
        return fs

    def in_clade(self, kmer_taxid: int, call_taxid: int) -> bool:
        """True if kmer_taxid lies in the clade rooted at call_taxid."""
        if kmer_taxid == call_taxid:
            return True
        return call_taxid in self.ancestors(kmer_taxid)


# ---------------------------------------------------------------------------
# Per-sequence output parsing
# ---------------------------------------------------------------------------
def parse_lca_string(lca: str) -> List[Tuple[Optional[int], int]]:
    """Parse 'taxid:count' runs. Returns [(taxid|None for ambiguous, count)]."""
    out = []
    for tok in lca.split():
        if ':' not in tok:
            continue
        left, right = tok.rsplit(':', 1)
        try:
            cnt = int(right)
        except ValueError:
            continue
        if left == 'A':
            out.append((None, cnt))
        else:
            try:
                out.append((int(left), cnt))
            except ValueError:
                out.append((None, cnt))
    return out


def score_contig(tax: Taxonomy, status: str, call_taxid: int,
                 length: int, lca: str) -> Dict[str, object]:
    runs = parse_lca_string(lca)
    total_kmers = sum(c for t, c in runs if t is not None)
    ambiguous = sum(c for t, c in runs if t is None)
    informative = sum(c for t, c in runs if t is not None and t != 0)
    clade = 0
    if status == 'C' and call_taxid > 0:
        for t, c in runs:
            if t is None or t == 0:
                continue
            if tax.in_clade(t, call_taxid):
                clade += c
    conf = (clade / total_kmers) if total_kmers else 0.0
    lin = tax.lineage(call_taxid) if (status == 'C' and call_taxid > 0) else {}
    return {
        'length': length,
        'status': status,
        'call_taxid': call_taxid,
        'call_rank': tax.rank.get(call_taxid, 'no rank') if call_taxid else 'unclassified',
        'phylum': lin.get('phylum', ''),
        'kingdom': lin.get('kingdom', ''),
        'total_kmers': total_kmers,
        'ambiguous_kmers': ambiguous,
        'informative_kmers': informative,
        'clade_kmers': clade,
        'confidence': round(conf, 6),
    }


def read_kraken_output(path: Path) -> List[Tuple[str, str, int, int, str]]:
    """Yield (seqid, status, call_taxid, length, lca_string)."""
    rows = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 5:
                continue
            status, seqid, taxfield, lenfield, lca = parts[0], parts[1], parts[2], parts[3], parts[4]
            m = TAXID_RE.search(taxfield)
            taxid = int(m.group(1)) if m else 0
            try:
                length = int(str(lenfield).split('|')[0])
            except ValueError:
                continue
            rows.append((seqid, status, taxid, length, lca))
    return rows


def find_kraken_output(acc: str) -> Optional[Path]:
    for d in KRAKEN2_DIRS:
        p = d / f'{acc}.kraken2.txt'
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Running Kraken2 (batched: one DB load for many genomes)
# ---------------------------------------------------------------------------
def run_kraken2_batched(accessions: List[str], threads: int,
                        out_dir: Path, tag: str) -> None:
    """
    Concatenate genomes into one FASTA with '<acc>|<contig>' sequence IDs, run
    Kraken2 once (a single 8 GB database load), then split the per-sequence
    output back into per-genome files.

    Kraken2 classifies each sequence independently, so batching is equivalent to
    per-genome invocation (verified by --verify-batching).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [a for a in accessions if not (out_dir / f'{a}.kraken2.txt').is_file()]
    if not todo:
        print(f"  [{tag}] all {len(accessions)} already have Kraken2 output")
        return
    print(f"  [{tag}] running Kraken2 on {len(todo)} genome(s), {threads} threads")

    tmpdir = Path(tempfile.mkdtemp(prefix=f'k2_{tag}_', dir=str(out_dir)))
    combined = tmpdir / 'combined.fna'
    try:
        n_seq = 0
        with open(combined, 'w') as out:
            for acc in todo:
                fa = MAG_DIR / f'{acc}.fna'
                if not fa.is_file():
                    print(f"    WARN missing FASTA: {acc}")
                    continue
                with open(fa) as f:
                    for line in f:
                        if line.startswith('>'):
                            cid = line[1:].split()[0] if line[1:].split() else 'c'
                            out.write(f'>{acc}|{cid}\n')
                            n_seq += 1
                        else:
                            out.write(line)
        print(f"    combined FASTA: {n_seq} sequences, "
              f"{combined.stat().st_size / 1e9:.2f} GB")

        raw_out = tmpdir / 'combined.kraken2.txt'
        raw_rep = tmpdir / 'combined.kraken2.report'
        cmd = [KRAKEN2_BIN, '--db', str(KRAKEN2_DB), '--threads', str(threads),
               '--output', str(raw_out), '--report', str(raw_rep),
               '--use-names', str(combined)]
        print(f"    {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"ERROR: kraken2 failed:\n{proc.stderr[-3000:]}")
        print(f"    {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'done'}")

        # split back per genome
        handles: Dict[str, object] = {}
        try:
            with open(raw_out) as f:
                for line in f:
                    parts = line.split('\t')
                    if len(parts) < 5:
                        continue
                    seqid = parts[1]
                    if '|' not in seqid:
                        continue
                    acc, cid = seqid.split('|', 1)
                    if acc not in handles:
                        handles[acc] = open(out_dir / f'{acc}.kraken2.txt', 'w')
                    parts[1] = cid
                    handles[acc].write('\t'.join(parts))
        finally:
            for h in handles.values():
                h.close()
        print(f"    split into {len(handles)} per-genome files")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Genome-level metrics
# ---------------------------------------------------------------------------
def genome_metrics(tax: Taxonomy, contigs: List[Dict[str, object]],
                   min_conf: float, min_inf: int, min_bp: int) -> Dict[str, object]:
    total_bp = sum(c['length'] for c in contigs)
    classified_bp = sum(c['length'] for c in contigs if c['status'] == 'C')
    unclassified_bp = total_bp - classified_bp

    # ---- ORIGINAL metric (faithful to scripts/61) ----
    orig_by_phylum: Dict[str, int] = defaultdict(int)
    for c in contigs:
        if c['status'] == 'C' and c['phylum']:
            orig_by_phylum[c['phylum']] += c['length']
    orig_resolved = sum(orig_by_phylum.values())
    if orig_by_phylum:
        od, odbp = max(orig_by_phylum.items(), key=lambda kv: kv[1])
        orig_contam_pct = (orig_resolved - odbp) / total_bp * 100 if total_bp else 0.0
        orig_dom_pct = odbp / total_bp * 100 if total_bp else 0.0
    else:
        od, orig_contam_pct, orig_dom_pct = 'unresolved', 0.0, 0.0

    # ---- STRICT metric ----
    qual = [c for c in contigs
            if c['status'] == 'C' and c['phylum']
            and c['confidence'] >= min_conf
            and c['informative_kmers'] >= min_inf
            and c['length'] >= min_bp]
    strict_by_phylum: Dict[str, int] = defaultdict(int)
    for c in qual:
        strict_by_phylum[c['phylum']] += c['length']
    strict_bp = sum(strict_by_phylum.values())
    if strict_by_phylum:
        sd, sdbp = max(strict_by_phylum.items(), key=lambda kv: kv[1])
        strict_contam_pct = (strict_bp - sdbp) / strict_bp * 100 if strict_bp else 0.0
        strict_dom_pct = sdbp / strict_bp * 100 if strict_bp else 0.0
    else:
        sd, strict_contam_pct, strict_dom_pct = 'unresolved', float('nan'), float('nan')

    tot_k = sum(c['total_kmers'] for c in contigs)
    inf_k = sum(c['informative_kmers'] for c in contigs)
    # bp-weighted mean per-contig confidence over classified contigs
    cls = [c for c in contigs if c['status'] == 'C']
    wconf = (sum(c['confidence'] * c['length'] for c in cls) /
             sum(c['length'] for c in cls)) if cls else 0.0

    return {
        'total_bp': total_bp,
        'n_contigs': len(contigs),
        'classified_bp': classified_bp,
        'unclassified_pct': round(unclassified_bp / total_bp * 100, 3) if total_bp else '',
        # original
        'orig_phylum_dominant': od,
        'orig_phylum_dominant_pct': round(orig_dom_pct, 3),
        'orig_phylum_contam_pct': round(orig_contam_pct, 3),
        'orig_n_phyla': len(orig_by_phylum),
        # strict
        'strict_phylum_dominant': sd,
        'strict_phylum_dominant_pct': ('' if strict_dom_pct != strict_dom_pct
                                       else round(strict_dom_pct, 3)),
        'strict_phylum_contam_pct': ('' if strict_contam_pct != strict_contam_pct
                                     else round(strict_contam_pct, 3)),
        'strict_n_phyla': len(strict_by_phylum),
        'strict_qualifying_bp': strict_bp,
        'strict_qualifying_bp_frac': round(strict_bp / total_bp, 4) if total_bp else '',
        'strict_n_qualifying_contigs': len(qual),
        # novelty covariates (independent of which taxa were called)
        'total_kmers': tot_k,
        'informative_kmers': inf_k,
        'informative_kmer_density': round(inf_k / tot_k, 6) if tot_k else '',
        'mean_contig_confidence_bpw': round(wconf, 6),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--run-missing', action='store_true',
                    help='run Kraken2 for agree_clean genomes lacking output')
    ap.add_argument('--min-confidence', type=float, default=0.10,
                    help='minimum per-contig Kraken2 confidence (default 0.10)')
    ap.add_argument('--min-informative-kmers', type=int, default=50,
                    help='minimum informative k-mers per contig (default 50)')
    ap.add_argument('--min-contig-bp', type=int, default=1000,
                    help='minimum contig length (default 1000)')
    ap.add_argument('--verify-batching', type=int, default=0, metavar='N',
                    help='re-run N genomes that already have per-genome output '
                         'in batched mode and confirm identical classifications')
    ap.add_argument('--out-dir', default=str(EVID_DIR))
    # --- multi-database support (WS4.3 database-capacity test) ---------------
    ap.add_argument('--kraken2-db', default=None,
                    help='Kraken2 database directory (default: tools/kraken2_db, '
                         'the CAPPED k2_standard_08gb build used originally)')
    ap.add_argument('--kraken2-bin', default='kraken2',
                    help='kraken2 executable (use the same binary across '
                         'databases so the comparison is fair)')
    ap.add_argument('--results-dir', default=None,
                    help='directory holding/receiving per-genome .kraken2.txt '
                         'for THIS database (default: the historical dirs)')
    ap.add_argument('--db-tag', default=None,
                    help='suffix for output filenames, e.g. "k2std" -> '
                         'kraken2_metrics_k2std.tsv')
    ap.add_argument('--taxonomy-dir', default=None,
                    help='directory containing nodes.dmp/names.dmp (default: '
                         'tools/kraken2_db). Both databases use NCBI taxids.')
    ap.add_argument('--run-cohorts', default=None,
                    help='comma-separated cohorts to (re-)run Kraken2 on with '
                         'this database, e.g. disagreement_141,agree_clean')
    args = ap.parse_args()

    global KRAKEN2_DB, KRAKEN2_BIN, KRAKEN2_DIRS, NEW_KRAKEN2_DIR, TAXONOMY_DIR
    if args.kraken2_db:
        KRAKEN2_DB = Path(args.kraken2_db)
        if not (KRAKEN2_DB / 'hash.k2d').is_file():
            sys.exit(f"ERROR: no hash.k2d in {KRAKEN2_DB}")
    KRAKEN2_BIN = args.kraken2_bin
    if args.taxonomy_dir:
        TAXONOMY_DIR = Path(args.taxonomy_dir)
    for _f in ('nodes.dmp', 'names.dmp'):
        if not (TAXONOMY_DIR / _f).is_file():
            sys.exit(f"ERROR: {_f} not found in {TAXONOMY_DIR}; pass --taxonomy-dir")
    if args.results_dir:
        rd = Path(args.results_dir)
        rd.mkdir(parents=True, exist_ok=True)
        KRAKEN2_DIRS = [rd]        # score ONLY this database's outputs
        NEW_KRAKEN2_DIR = rd
    tag = f'_{args.db_tag}' if args.db_tag else ''

    hash_bytes = ((KRAKEN2_DB / 'hash.k2d').stat().st_size
                  if (KRAKEN2_DB / 'hash.k2d').is_file() else 0)
    print(f"Kraken2 DB : {KRAKEN2_DB}  (hash.k2d = {hash_bytes:,} bytes"
          f" = {hash_bytes/1e9:.1f} GB)")
    print(f"Kraken2 bin: {KRAKEN2_BIN}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_tsv = out_dir / 'cohort_genome_stats.tsv'
    if not stats_tsv.is_file():
        sys.exit(f"ERROR: {stats_tsv} missing; run scripts/77_build_cohorts_and_stats.py")
    with open(stats_tsv) as f:
        cohort_rows = list(csv.DictReader(f, delimiter='\t'))
    cohort_of = {r['accession']: r['cohort'] for r in cohort_rows}

    # ---- optional: run Kraken2 for the agree_clean controls ----
    if args.run_missing:
        todo_file = out_dir / 'kraken2_todo.txt'
        todo = [ln.strip() for ln in todo_file.read_text().splitlines() if ln.strip()]
        if todo:
            run_kraken2_batched(todo, args.threads, NEW_KRAKEN2_DIR, 'agree_clean')

    # ---- optional: (re-)run whole cohorts against THIS database ----
    if args.run_cohorts:
        want = set(args.run_cohorts.split(','))
        accs = [r['accession'] for r in cohort_rows if r['cohort'] in want]
        print(f"Running Kraken2 on {len(accs)} genome(s) from cohorts "
              f"{sorted(want)} against {KRAKEN2_DB.name}")
        run_kraken2_batched(accs, args.threads, NEW_KRAKEN2_DIR,
                            args.db_tag or 'db')

    print("Loading NCBI taxonomy (nodes.dmp / names.dmp) ...")
    tax = Taxonomy()
    print(f"  {len(tax.parent):,} nodes, {len(tax.name):,} names")

    # ---- optional batching validation ----
    validation = None
    if args.verify_batching:
        have = [r['accession'] for r in cohort_rows
                if r['cohort'] == 'disagreement_141'
                and (PROJECT_DIR / 'data' / 'ncbi' / 'kraken2_results'
                     / f"{r['accession']}.kraken2.txt").is_file()]
        sample = sorted(have)[:args.verify_batching]
        vdir = Path(tempfile.mkdtemp(prefix='k2verify_'))
        try:
            run_kraken2_batched(sample, args.threads, vdir, 'verify')
            agree = total = 0
            for acc in sample:
                a = {s: (st, t) for s, st, t, L, l in
                     read_kraken_output(PROJECT_DIR / 'data' / 'ncbi'
                                        / 'kraken2_results' / f'{acc}.kraken2.txt')}
                b = {s: (st, t) for s, st, t, L, l in
                     read_kraken_output(vdir / f'{acc}.kraken2.txt')}
                for s in a:
                    total += 1
                    agree += int(a.get(s) == b.get(s))
            validation = {'genomes': sample, 'contigs': total, 'identical': agree,
                          'pct': round(agree / total * 100, 4) if total else None}
            print(f"  batching validation: {agree}/{total} contig calls identical "
                  f"({validation['pct']}%)")
        finally:
            shutil.rmtree(vdir, ignore_errors=True)

    # ---- score every genome that has Kraken2 output ----
    print(f"\nScoring genomes (min_conf={args.min_confidence}, "
          f"min_informative_kmers={args.min_informative_kmers}, "
          f"min_contig_bp={args.min_contig_bp}) ...")
    metric_rows = []
    contig_rows = []
    n_missing = 0
    for r in cohort_rows:
        acc = r['accession']
        p = find_kraken_output(acc)
        if p is None:
            n_missing += 1
            continue
        contigs = [score_contig(tax, st, t, L, lca)
                   for s, st, t, L, lca in read_kraken_output(p)]
        if not contigs:
            continue
        m = genome_metrics(tax, contigs, args.min_confidence,
                           args.min_informative_kmers, args.min_contig_bp)
        m['accession'] = acc
        m['cohort'] = cohort_of.get(acc, '')
        m['v5_contamination'] = r['v5_contamination']
        m['checkm2_contamination'] = r['checkm2_contamination']
        m['v5_completeness'] = r['v5_completeness']
        m['checkm2_completeness'] = r['checkm2_completeness']
        m['asm_n_contigs'] = r['n_contigs']
        m['asm_n50'] = r['n50']
        m['asm_total_bp'] = r['total_bp']
        m['asm_gc_pct'] = r['gc_pct']
        metric_rows.append(m)
        for i, c in enumerate(contigs):
            contig_rows.append({'accession': acc, 'cohort': m['cohort'],
                                'contig_index': i, **c})

    print(f"  scored {len(metric_rows)} genomes ({n_missing} without Kraken2 output)")

    m_header = ['accession', 'cohort',
                'v5_completeness', 'v5_contamination',
                'checkm2_completeness', 'checkm2_contamination',
                'total_bp', 'n_contigs', 'classified_bp', 'unclassified_pct',
                'orig_phylum_dominant', 'orig_phylum_dominant_pct',
                'orig_phylum_contam_pct', 'orig_n_phyla',
                'strict_phylum_dominant', 'strict_phylum_dominant_pct',
                'strict_phylum_contam_pct', 'strict_n_phyla',
                'strict_qualifying_bp', 'strict_qualifying_bp_frac',
                'strict_n_qualifying_contigs',
                'total_kmers', 'informative_kmers', 'informative_kmer_density',
                'mean_contig_confidence_bpw',
                'asm_total_bp', 'asm_n_contigs', 'asm_n50', 'asm_gc_pct']
    out_tsv = out_dir / f'kraken2_metrics{tag}.tsv'
    with open(out_tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=m_header, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(metric_rows)

    c_header = ['accession', 'cohort', 'contig_index', 'length', 'status',
                'call_taxid', 'call_rank', 'kingdom', 'phylum',
                'total_kmers', 'ambiguous_kmers', 'informative_kmers',
                'clade_kmers', 'confidence']
    contig_tsv = out_dir / f'kraken2_contig_detail{tag}.tsv.gz'
    with gzip.open(contig_tsv, 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=c_header, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(contig_rows)

    # ---- console summary ----
    def med(vals):
        v = sorted(x for x in vals if x == x)
        return v[len(v) // 2] if v else float('nan')

    print(f"\n{'cohort':20}{'n':>5}{'orig_contam%':>14}{'strict_contam%':>16}"
          f"{'qual_bp_frac':>14}{'infK_density':>14}{'meanConf':>10}")
    per_cohort = {}
    for coh in ('disagreement_141', 'agree_clean', 'v5hq_only', 'neither_hq'):
        sel = [m for m in metric_rows if m['cohort'] == coh]
        if not sel:
            continue
        o = med([float(m['orig_phylum_contam_pct']) for m in sel])
        s = med([float(m['strict_phylum_contam_pct']) for m in sel
                 if m['strict_phylum_contam_pct'] != ''])
        q = med([float(m['strict_qualifying_bp_frac']) for m in sel
                 if m['strict_qualifying_bp_frac'] != ''])
        d = med([float(m['informative_kmer_density']) for m in sel
                 if m['informative_kmer_density'] != ''])
        c = med([float(m['mean_contig_confidence_bpw']) for m in sel])
        print(f"{coh:20}{len(sel):>5}{o:>14.2f}{s:>16.2f}{q:>14.3f}{d:>14.4f}{c:>10.4f}")
        per_cohort[coh] = {'n': len(sel), 'median_orig_phylum_contam_pct': o,
                           'median_strict_phylum_contam_pct': s,
                           'median_strict_qualifying_bp_frac': q,
                           'median_informative_kmer_density': d,
                           'median_mean_contig_confidence': c}

    # >5% flags under each metric, for the 141
    d141 = [m for m in metric_rows if m['cohort'] == 'disagreement_141']
    o_flag = sum(1 for m in d141 if float(m['orig_phylum_contam_pct']) > 5)
    s_ok = [m for m in d141 if m['strict_phylum_contam_pct'] != '']
    s_flag = sum(1 for m in s_ok if float(m['strict_phylum_contam_pct']) > 5)
    print(f"\ndisagreement_141: >5% cross-phylum contamination")
    print(f"  ORIGINAL metric: {o_flag}/{len(d141)} "
          f"({o_flag / len(d141) * 100:.1f}%)   [manuscript claims 109/141 = 77.3%]")
    if s_ok:
        print(f"  STRICT   metric: {s_flag}/{len(s_ok)} scoreable "
              f"({s_flag / len(s_ok) * 100:.1f}%)"
              f"   [{len(d141) - len(s_ok)} unscoreable: no qualifying contigs]")
    else:
        print(f"  STRICT   metric: NOT SCOREABLE for any of the {len(d141)} genomes "
              f"-- no contig meets the evidence thresholds")

    # ---- evidence-support distribution and threshold sweep -------------------
    # How much k-mer evidence actually underpins the ORIGINAL phylum calls?
    d141_contigs = [c for c in contig_rows
                    if c['cohort'] == 'disagreement_141' and c['status'] == 'C'
                    and c['phylum']]
    if d141_contigs:
        infs = sorted(c['informative_kmers'] for c in d141_contigs)
        confs = sorted(c['confidence'] for c in d141_contigs)
        bp_tot = sum(c['length'] for c in d141_contigs)
        print(f"\nEvidence behind the ORIGINAL phylum calls "
              f"({len(d141_contigs):,} classified contigs, disagreement_141):")
        for q, lbl in ((0.5, 'median'), (0.75, '75th pct'),
                       (0.9, '90th pct'), (0.99, '99th pct')):
            i = min(int(q * len(infs)), len(infs) - 1)
            j = min(int(q * len(confs)), len(confs) - 1)
            print(f"  {lbl:9s} informative k-mers/contig = {infs[i]:>6,}"
                  f"    confidence = {confs[j]:.5f}")
        for thr in (1, 2, 5, 10, 50):
            n = sum(1 for c in d141_contigs if c['informative_kmers'] <= thr)
            bp = sum(c['length'] for c in d141_contigs
                     if c['informative_kmers'] <= thr)
            print(f"  contigs with <= {thr:>3} informative k-mers: "
                  f"{n:>6,}/{len(d141_contigs):,} ({n / len(d141_contigs) * 100:5.1f}%) "
                  f"carrying {bp / bp_tot * 100:5.1f}% of classified bp")

    sweep = []
    print(f"\nThreshold sweep -- median strict contamination and scoreability "
          f"(disagreement_141, n={len(d141)}):")
    print(f"  {'min_conf':>9}{'min_infK':>10}{'scoreable':>11}"
          f"{'med_strict%':>13}{'med_qual_bp_frac':>18}")
    for mc, mi in ((0.0, 0), (0.001, 1), (0.005, 2), (0.01, 5),
                   (0.05, 10), (0.10, 50)):
        rows = []
        for r in cohort_rows:
            if cohort_of.get(r['accession']) != 'disagreement_141':
                continue
            cs = [c for c in contig_rows if c['accession'] == r['accession']]
            if not cs:
                continue
            rows.append(genome_metrics(tax, cs, mc, mi, args.min_contig_bp))
        ok = [x for x in rows if x['strict_phylum_contam_pct'] != '']
        msc = med([float(x['strict_phylum_contam_pct']) for x in ok]) if ok else float('nan')
        mqf = med([float(x['strict_qualifying_bp_frac']) for x in rows
                   if x['strict_qualifying_bp_frac'] != ''])
        print(f"  {mc:>9}{mi:>10}{len(ok):>11}{msc:>13.2f}{mqf:>18.4f}")
        sweep.append({'min_confidence': mc, 'min_informative_kmers': mi,
                      'n_scoreable': len(ok), 'n_total': len(rows),
                      'median_strict_contam_pct': None if msc != msc else round(msc, 3),
                      'median_qualifying_bp_frac': None if mqf != mqf else round(mqf, 4)})

    # ---- validate the ORIGINAL metric against the published analysis --------
    pub_path = (PROJECT_DIR / 'results' / 'ncbi_comparison'
                / 'kraken2_contamination_analysis.tsv')
    repro = None
    if pub_path.is_file():
        pub = {r['accession']: r for r in
               csv.DictReader(open(pub_path), delimiter='\t')}
        mine = {m['accession']: m for m in metric_rows}
        diffs, ndiff = [], []
        for a, p in pub.items():
            m = mine.get(a)
            if not m:
                continue
            diffs.append(abs(float(m['orig_phylum_contam_pct'])
                             - float(p['phylum_contam_pct'])))
            ndiff.append(abs(int(m['orig_n_phyla']) - int(p['phylum_n_taxa'])))
        if diffs:
            repro = {'n_compared': len(diffs),
                     'max_abs_diff_contam_pct': round(max(diffs), 6),
                     'mean_abs_diff_contam_pct': round(sum(diffs) / len(diffs), 6),
                     'max_abs_diff_n_phyla': max(ndiff),
                     'verdict': ('exact reproduction of scripts/61 (differences are '
                                 'rounding only)' if max(diffs) < 0.02
                                 else 'MISMATCH -- investigate')}
            print(f"\nORIGINAL-metric reproduction check vs {pub_path.name}: "
                  f"n={repro['n_compared']}, max|diff|="
                  f"{repro['max_abs_diff_contam_pct']} pp -> {repro['verdict']}")

    summary = {
        'original_metric_reproduction_check': repro,
        'thresholds': {'min_confidence': args.min_confidence,
                       'min_informative_kmers': args.min_informative_kmers,
                       'min_contig_bp': args.min_contig_bp},
        'original_metric': ('non-dominant-phylum resolved bp / total bp; every '
                            'classified contig contributes its full length '
                            'regardless of k-mer support (reproduces scripts/61)'),
        'strict_metric': ('non-dominant-phylum qualifying bp / qualifying bp; a '
                          'contig qualifies only with confidence >= min_confidence, '
                          'informative_kmers >= min_informative_kmers and '
                          'length >= min_contig_bp; unclassified and low-confidence '
                          'bp are excluded from the denominator'),
        'kraken2_db': str(KRAKEN2_DB),
        'kraken2_db_hash_bytes': hash_bytes,
        'kraken2_db_hash_gb': round(hash_bytes/1e9, 2),
        'kraken2_binary': KRAKEN2_BIN,
        'taxonomy_dir': str(TAXONOMY_DIR),
        'kraken2_version': _k2_version(KRAKEN2_BIN),
        'db_tag': args.db_tag or '',
        'n_genomes_scored': len(metric_rows),
        'per_cohort': per_cohort,
        'disagreement_141_flags_gt5pct': {
            'original': {'n_flagged': o_flag, 'n_total': len(d141)},
            'strict': {'n_flagged': s_flag, 'n_scoreable': len(s_ok),
                       'n_unscoreable': len(d141) - len(s_ok)},
            'manuscript_claim': '109/141 (77.3%)',
        },
        'batching_validation': validation,
        'threshold_sweep_disagreement_141': sweep,
        'kraken2_db_note': (
            'tools/kraken2_db hash.k2d is exactly 8,000,000,032 bytes, i.e. the '
            'capped k2_standard_08gb build. Minimizer down-sampling means only a '
            'small fraction of a query genome k-mers can match, which is why '
            'informative_kmer_density is ~5e-4 for these MAGs.'),
        'outputs': {'metrics': str(out_tsv), 'contig_detail': str(contig_tsv)},
    }
    (out_dir / f'kraken2_metrics_summary{tag}.json').write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_tsv}")
    print(f"Wrote {contig_tsv}")
    print(f"Wrote {out_dir / ('kraken2_metrics_summary' + tag + '.json')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
