#!/usr/bin/env python3
"""
WS4.5 -- database-independent contamination evidence: single-copy core-gene
duplication and copy divergence.

Why this is the decisive line of evidence
-----------------------------------------
Both Kraken2 (nucleotide LCA against a capped 8 GB standard database) and GUNC
(protein LCA against proGenomes/GTDB) depend on the query genome being
represented in a reference database.  For MAGs from uncultivated lineages that
condition fails, and both tools' signals become confounded with novelty --
WS4.3 showed the Kraken2 phylum metric is driven by assignments supported by a
median of 8 k-mers per contig.

Genes that are single-copy *by construction* provide contamination evidence that
needs no taxonomy database at all.  If a genome contains two copies of a gene
that occurs once per organism, the genome contains more than one organism --
regardless of whether either organism is in any database.  Copy **divergence**
separates the two competing explanations:

    high divergence (< 90% aa identity)  -> distinct organisms (contamination)
    near-identical (>= 95% aa identity)  -> recent duplication or assembly
                                            artifact, not contamination

This also answers Reviewer 2's request for "abnormal marker gene duplication
patterns" directly.

Method (follows the established pattern of scripts/09_identify_core_genes.py)
-----------------------------------------------------------------------------
1. Prodigal (``-p single``) gene calls per genome.
2. ``hmmsearch --cut_tc`` against BOTH profile sets shipped with this project:
   ``85_bcg.hmm`` (85 bacterial single-copy core genes) and ``uacg.hmm``
   (128 archaeal).  The domain is assigned from whichever set recovers the
   larger fraction of its profiles, so no external taxonomy is consulted.
3. For the assigned domain, group hits by profile (= gene family).  Families
   with >= 2 distinct proteins are duplicated.
4. For each duplicated family, extract the profile-aligned region of each copy
   (from ``--domtblout`` HMM/target coordinates) and compute pairwise global
   amino-acid identity (BLOSUM62) between copies, plus whether the copies sit
   on different contigs.

Usage
-----
    python scripts/79_scg_duplication.py --threads 16
    python scripts/79_scg_duplication.py --cohorts disagreement_141 --threads 8

Outputs (``results/revision/contamination_evidence/``)
------------------------------------------------------
    scg_duplication.tsv          per genome: duplication counts and divergence
    scg_duplication_families.tsv per genome x duplicated family detail
    scg_duplication_summary.json per-cohort summaries and method parameters
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
MAG_DIR = PROJECT_DIR / 'data' / 'ncbi' / 'mags'
BACT_HMM = PROJECT_DIR / '85_bcg.hmm'
ARCH_HMM = PROJECT_DIR / 'uacg.hmm'
EVID_DIR = PROJECT_DIR / 'results' / 'revision' / 'contamination_evidence'

N_BACT_PROFILES = 85
N_ARCH_PROFILES = 128

# ---------------------------------------------------------------------------
# Divergence thresholds, and why
# ---------------------------------------------------------------------------
# Core-gene amino-acid identity within a species is typically >97%, and remains
# ~90-95% within a genus; it falls well below 90% between families and phyla.
# Recently duplicated paralogs also start out near-identical and decay slowly.
# So two copies of a gene that is single-copy by construction:
#   >= NEAR_IDENTICAL_ID   near-identical  -> recent duplication or an assembly
#                                             artifact (e.g. a redundant contig)
#   <  DIVERGENT_ID        divergent       -> beyond intra-species/recent-duplication
#                                             expectation => distinct organisms
#   <  STRONGLY_DIVERGENT  strongly diverg.-> reported as a conservative tier
DIVERGENT_ID = 0.90
STRONGLY_DIVERGENT_ID = 0.80
NEAR_IDENTICAL_ID = 0.95

# ---------------------------------------------------------------------------
# Split-gene control (essential artifact control)
# ---------------------------------------------------------------------------
# A single gene broken across two contigs yields TWO partial hits to the same
# profile, which would be miscounted as a duplicated family -- and because the
# two fragments cover different parts of the profile, their apparent pairwise
# identity is low, mimicking divergence. A genuine duplicate must have both
# copies covering substantially the SAME profile region. We therefore require
# reciprocal overlap of the two hits' HMM coordinate spans.
MIN_HMM_SPAN_OVERLAP = 0.50   # reciprocal overlap fraction on profile coords
MIN_HMM_SPAN_COVERAGE = 0.30  # each copy must cover >=30% of the profile

# Efficiency: skip the 128-profile archaeal search when the bacterial set is
# already well recovered (a genome recovering >=50% of 85 bacterial single-copy
# core genes is bacterial). Purely a compute optimisation -- the criterion uses
# only recovered profile fractions, so the analysis stays database-free.
ARCHAEAL_SEARCH_SKIP_ABOVE = 0.50


# ---------------------------------------------------------------------------
# FASTA / alignment helpers
# ---------------------------------------------------------------------------
def read_faa(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    name = None
    parts: List[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if name:
                    seqs[name] = ''.join(parts)
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip().rstrip('*'))
    if name:
        seqs[name] = ''.join(parts)
    return seqs


_ALIGNER = None


def aligner():
    """Global BLOSUM62 pairwise aligner (created once per worker process)."""
    global _ALIGNER
    if _ALIGNER is None:
        from Bio.Align import PairwiseAligner, substitution_matrices
        a = PairwiseAligner()
        a.substitution_matrix = substitution_matrices.load('BLOSUM62')
        a.open_gap_score = -11
        a.extend_gap_score = -1
        a.mode = 'global'
        _ALIGNER = a
    return _ALIGNER


def pairwise_identity(s1: str, s2: str) -> Optional[float]:
    """Global amino-acid identity over aligned columns (gaps excluded)."""
    if not s1 or not s2:
        return None
    valid = set('ACDEFGHIKLMNPQRSTVWY')
    s1 = ''.join(c for c in s1.upper() if c in valid)
    s2 = ''.join(c for c in s2.upper() if c in valid)
    if len(s1) < 30 or len(s2) < 30:
        return None
    try:
        aln = aligner().align(s1, s2)[0]
    except Exception:
        return None
    a, b = str(aln[0]), str(aln[1])
    same = cols = 0
    for x, y in zip(a, b):
        if x == '-' or y == '-':
            continue
        cols += 1
        same += int(x == y)
    return (same / cols) if cols else None


def contig_of(protein_id: str) -> str:
    """Prodigal names proteins '<contig>_<geneIndex>'."""
    return protein_id.rsplit('_', 1)[0] if '_' in protein_id else protein_id


# ---------------------------------------------------------------------------
# Per-genome worker
# ---------------------------------------------------------------------------
def parse_domtblout(path: Path
                    ) -> Dict[str, Dict[str, Tuple[float, int, int, int, int, int]]]:
    """
    family -> {target: (bitscore, hmm_from, hmm_to, ali_from, ali_to, hmm_len)}
    keeping the best-scoring domain per (family, target).
    """
    best: Dict[str, Dict[str, Tuple[float, int, int, int, int, int]]] = defaultdict(dict)
    if not path.is_file():
        return best
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 23:
                continue
            target, family = p[0], p[3]
            try:
                hmm_len = int(p[5])
                score = float(p[13])
                hmm_from, hmm_to = int(p[15]), int(p[16])
                ali_from, ali_to = int(p[17]), int(p[18])
            except ValueError:
                continue
            cur = best[family].get(target)
            if cur is None or score > cur[0]:
                best[family][target] = (score, hmm_from, hmm_to,
                                        ali_from, ali_to, hmm_len)
    return best


def hmm_span_overlap(v1, v2) -> Tuple[float, float, float]:
    """
    Reciprocal overlap of two hits' profile (HMM) coordinate spans, and each
    hit's profile coverage.  Returns (reciprocal_overlap, cov1, cov2).
    """
    a1, a2 = v1[1], v1[2]
    b1, b2 = v2[1], v2[2]
    hmm_len = max(v1[5], v2[5], 1)
    len1 = max(0, a2 - a1 + 1)
    len2 = max(0, b2 - b1 + 1)
    inter = max(0, min(a2, b2) - max(a1, b1) + 1)
    recip = inter / min(len1, len2) if min(len1, len2) > 0 else 0.0
    return recip, len1 / hmm_len, len2 / hmm_len


def process_genome(args: Tuple[str, str]) -> Optional[Dict[str, object]]:
    acc, fasta_path = args
    tmpdir = tempfile.mkdtemp(prefix=f'scg_{acc}_')
    try:
        prot = os.path.join(tmpdir, 'proteins.faa')
        gff = os.path.join(tmpdir, 'genes.gff')
        r = subprocess.run(
            ['prodigal', '-i', fasta_path, '-a', prot, '-o', gff,
             '-f', 'gff', '-p', 'single', '-q'],
            capture_output=True, text=True, timeout=900)
        if r.returncode != 0 or not os.path.exists(prot) or os.path.getsize(prot) == 0:
            return {'accession': acc, 'error': f'prodigal failed: {r.stderr[:150]}'}

        seqs = read_faa(Path(prot))
        results = {}

        def search(dom: str, hmm: Path):
            dtbl = os.path.join(tmpdir, f'{dom}.domtbl')
            r = subprocess.run(
                ['hmmsearch', '--cut_tc', '--domtblout', dtbl, '--noali',
                 '--cpu', '1', str(hmm), prot],
                capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f'hmmsearch {dom} failed')
            return parse_domtblout(Path(dtbl))

        # Bacterial set first. The archaeal set (128 profiles) is only searched
        # when bacterial recovery is poor, i.e. when the genome plausibly is not
        # bacterial. This is still database-free -- it uses only the recovered
        # fraction of each single-copy profile set -- and avoids running the
        # larger archaeal search on the ~97% of this cohort that is bacterial.
        try:
            results['bacteria'] = search('bacteria', BACT_HMM)
        except RuntimeError as e:
            return {'accession': acc, 'error': str(e)}
        n_bact_fam = len(results['bacteria'])
        frac_b = n_bact_fam / N_BACT_PROFILES

        archaeal_skipped = frac_b >= ARCHAEAL_SEARCH_SKIP_ABOVE
        if archaeal_skipped:
            results['archaea'] = {}
            n_arch_fam = 0
            frac_a = 0.0
        else:
            try:
                results['archaea'] = search('archaea', ARCH_HMM)
            except RuntimeError as e:
                return {'accession': acc, 'error': str(e)}
            n_arch_fam = len(results['archaea'])
            frac_a = n_arch_fam / N_ARCH_PROFILES

        # Domain assignment: larger recovered fraction of its own profile set.
        domain = 'archaea' if frac_a > frac_b else 'bacteria'
        fams = results[domain]
        n_profiles = N_ARCH_PROFILES if domain == 'archaea' else N_BACT_PROFILES

        multi = {fam: t for fam, t in fams.items() if len(t) >= 2}
        copies = [len(t) for t in fams.values()]
        extra = sum(max(0, c - 1) for c in copies)

        pair_rows: List[Dict[str, object]] = []
        for fam, targets in sorted(multi.items()):
            items = sorted(targets.items(), key=lambda kv: -kv[1][0])
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    t1, v1 = items[i]
                    t2, v2 = items[j]
                    recip, cov1, cov2 = hmm_span_overlap(v1, v2)
                    sub1 = seqs.get(t1, '')[v1[3] - 1:v1[4]]
                    sub2 = seqs.get(t2, '')[v2[3] - 1:v2[4]]
                    pid = pairwise_identity(sub1, sub2)
                    # Split-gene control: a genuine duplicate needs both copies
                    # over the same profile region with adequate coverage.
                    valid = (recip >= MIN_HMM_SPAN_OVERLAP
                             and min(cov1, cov2) >= MIN_HMM_SPAN_COVERAGE)
                    pair_rows.append({
                        'accession': acc, 'domain': domain, 'family': fam,
                        'protein_1': t1, 'protein_2': t2,
                        'contig_1': contig_of(t1), 'contig_2': contig_of(t2),
                        'same_contig': str(contig_of(t1) == contig_of(t2)),
                        'hmm_span_1': f'{v1[1]}-{v1[2]}',
                        'hmm_span_2': f'{v2[1]}-{v2[2]}',
                        'hmm_reciprocal_overlap': round(recip, 4),
                        'hmm_coverage_1': round(cov1, 4),
                        'hmm_coverage_2': round(cov2, 4),
                        'passes_split_gene_control': str(valid),
                        'aa_identity': '' if pid is None else round(pid, 4),
                    })

        # Only pairs passing the split-gene control count as real duplicates.
        good = [p for p in pair_rows if p['passes_split_gene_control'] == 'True'
                and p['aa_identity'] != '']
        rejected_split = sum(1 for p in pair_rows
                             if p['passes_split_gene_control'] == 'False')
        pair_ids = [float(p['aa_identity']) for p in good]
        pair_diff_contig = sum(1 for p in good if p['same_contig'] == 'False')
        n_pairs = len(pair_rows)

        # Families surviving the split-gene control
        good_fams = {p['family'] for p in good}
        div_fams = {p['family'] for p in good
                    if float(p['aa_identity']) < DIVERGENT_ID}
        strong_div_fams = {p['family'] for p in good
                           if float(p['aa_identity']) < STRONGLY_DIVERGENT_ID}

        n_div = sum(1 for p in pair_ids if p < DIVERGENT_ID)
        n_strong = sum(1 for p in pair_ids if p < STRONGLY_DIVERGENT_ID)
        n_near = sum(1 for p in pair_ids if p >= NEAR_IDENTICAL_ID)
        pair_ids_sorted = sorted(pair_ids)

        fam_rows = []
        for fam in sorted(multi):
            fp = [p for p in good if p['family'] == fam]
            ids = [float(p['aa_identity']) for p in fp]
            fam_rows.append({
                'accession': acc, 'domain': domain, 'family': fam,
                'n_copies': len(multi[fam]),
                'n_pairs_total': sum(1 for p in pair_rows if p['family'] == fam),
                'n_pairs_valid': len(fp),
                'n_pairs_diff_contig': sum(1 for p in fp
                                           if p['same_contig'] == 'False'),
                'mean_pairwise_identity': (round(sum(ids) / len(ids), 4) if ids else ''),
                'min_pairwise_identity': (round(min(ids), 4) if ids else ''),
                'survives_split_gene_control': str(bool(fp)),
                'contigs': ';'.join(sorted({contig_of(t) for t in multi[fam]})),
            })

        return {
            'accession': acc,
            'domain': domain,
            'n_proteins': len(seqs),
            'n_profiles_in_set': n_profiles,
            'n_bact_families': n_bact_fam,
            'n_arch_families': n_arch_fam,
            'archaeal_search_skipped': str(archaeal_skipped),
            'n_families_detected': len(fams),
            'frac_profiles_detected': round(len(fams) / n_profiles, 4),
            'n_families_multicopy_raw': len(multi),
            'n_families_multicopy': len(good_fams),
            'n_pairs_rejected_split_gene': rejected_split,
            'frac_families_multicopy': (round(len(good_fams) / len(fams), 4)
                                        if fams else ''),
            'total_extra_copies': extra,
            'max_copies': max(copies) if copies else 0,
            'n_dup_pairs': n_pairs,
            'n_dup_pairs_valid': len(good),
            'n_dup_pairs_diff_contig': pair_diff_contig,
            'n_dup_pairs_scored': len(pair_ids),
            'mean_dup_pair_identity': (round(sum(pair_ids) / len(pair_ids), 4)
                                       if pair_ids else ''),
            'median_dup_pair_identity': (round(pair_ids_sorted[len(pair_ids) // 2], 4)
                                         if pair_ids else ''),
            'min_dup_pair_identity': (round(pair_ids_sorted[0], 4) if pair_ids else ''),
            'n_dup_pairs_divergent': n_div,
            'n_dup_pairs_strongly_divergent': n_strong,
            'n_dup_pairs_near_identical': n_near,
            'frac_dup_pairs_divergent': (round(n_div / len(pair_ids), 4)
                                         if pair_ids else ''),
            'n_divergent_families': len(div_fams),
            'n_strongly_divergent_families': len(strong_div_fams),
            # headline DB-independent contamination score: divergent duplicate
            # families (after the split-gene control) per 100 profiles searched
            'divergent_dup_families_per_100_profiles': round(
                len(div_fams) / n_profiles * 100, 3),
            'strongly_divergent_families_per_100_profiles': round(
                len(strong_div_fams) / n_profiles * 100, 3),
            '_families': fam_rows,
            '_pairs': pair_rows,
            'error': '',
        }
    except subprocess.TimeoutExpired:
        return {'accession': acc, 'error': 'timeout'}
    except Exception as e:  # pragma: no cover
        return {'accession': acc, 'error': str(e)[:200]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--cohorts', default='disagreement_141,agree_clean',
                    help='comma-separated cohorts from cohort_genome_stats.tsv')
    ap.add_argument('--limit', type=int, default=0, help='debug: cap genomes')
    ap.add_argument('--out-dir', default=str(EVID_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (BACT_HMM, ARCH_HMM):
        if not p.is_file():
            sys.exit(f"ERROR: HMM profile set not found: {p}")

    stats_tsv = out_dir / 'cohort_genome_stats.tsv'
    if not stats_tsv.is_file():
        sys.exit(f"ERROR: {stats_tsv} missing; run scripts/77_build_cohorts_and_stats.py")
    with open(stats_tsv) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    wanted = set(args.cohorts.split(','))
    sel = [r for r in rows if r['cohort'] in wanted]
    if args.limit:
        sel = sel[:args.limit]
    cohort_of = {r['accession']: r['cohort'] for r in rows}
    meta = {r['accession']: r for r in rows}

    jobs = []
    for r in sel:
        p = MAG_DIR / f"{r['accession']}.fna"
        if p.is_file():
            jobs.append((r['accession'], str(p)))
    print(f"Cohorts: {sorted(wanted)}")
    print(f"Genomes to process: {len(jobs)}  (threads={args.threads})")
    print(f"Profile sets: 85_bcg.hmm ({N_BACT_PROFILES}), uacg.hmm ({N_ARCH_PROFILES})")

    with Pool(processes=max(1, args.threads)) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(process_genome, jobs, chunksize=1)):
            results.append(res)
            if (i + 1) % 25 == 0 or (i + 1) == len(jobs):
                print(f"  {i + 1}/{len(jobs)} done")

    ok = [r for r in results if r and not r.get('error')]
    bad = [r for r in results if r and r.get('error')]
    print(f"  succeeded: {len(ok)}   failed: {len(bad)}")
    for b in bad[:10]:
        print(f"    {b['accession']}: {b['error']}")

    g_header = ['accession', 'cohort', 'domain', 'n_proteins',
                'n_profiles_in_set', 'n_bact_families', 'n_arch_families',
                'archaeal_search_skipped', 'n_families_detected', 'frac_profiles_detected',
                'n_families_multicopy_raw', 'n_families_multicopy',
                'n_pairs_rejected_split_gene', 'frac_families_multicopy',
                'total_extra_copies', 'max_copies',
                'n_dup_pairs', 'n_dup_pairs_valid', 'n_dup_pairs_diff_contig',
                'n_dup_pairs_scored',
                'mean_dup_pair_identity', 'median_dup_pair_identity',
                'min_dup_pair_identity', 'n_dup_pairs_divergent',
                'n_dup_pairs_strongly_divergent',
                'n_dup_pairs_near_identical', 'frac_dup_pairs_divergent',
                'n_divergent_families', 'n_strongly_divergent_families',
                'divergent_dup_families_per_100_profiles',
                'strongly_divergent_families_per_100_profiles',
                'v5_completeness', 'v5_contamination',
                'checkm2_completeness', 'checkm2_contamination',
                'total_bp', 'n_contigs', 'n50', 'gc_pct']
    g_rows = []
    f_rows = []
    p_rows = []
    for r in ok:
        m = meta.get(r['accession'], {})
        row = {k: r.get(k, '') for k in g_header}
        row['accession'] = r['accession']
        row['cohort'] = cohort_of.get(r['accession'], '')
        for k in ('v5_completeness', 'v5_contamination', 'checkm2_completeness',
                  'checkm2_contamination', 'total_bp', 'n_contigs', 'n50', 'gc_pct'):
            row[k] = m.get(k, '')
        g_rows.append(row)
        coh = row['cohort']
        for fr in r['_families']:
            fr['cohort'] = coh
        for pr in r['_pairs']:
            pr['cohort'] = coh
        f_rows.extend(r['_families'])
        p_rows.extend(r['_pairs'])
    g_rows.sort(key=lambda r: (r['cohort'], r['accession']))

    out_tsv = out_dir / 'scg_duplication.tsv'
    with open(out_tsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=g_header, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(g_rows)

    fam_tsv = out_dir / 'scg_duplication_families.tsv'
    if f_rows:
        with open(fam_tsv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(f_rows[0].keys()), delimiter='\t')
            w.writeheader()
            w.writerows(f_rows)

    # full per-pair divergence distribution (the raw evidence)
    pair_tsv = out_dir / 'scg_duplication_pairs.tsv'
    if p_rows:
        with open(pair_tsv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(p_rows[0].keys()), delimiter='\t')
            w.writeheader()
            w.writerows(p_rows)

    # ---- divergence distribution, per cohort ----
    print("\nInter-copy amino-acid identity distribution "
          "(pairs passing the split-gene control):")
    print(f"  {'cohort':20}{'n_pairs':>9}{'p10':>7}{'p25':>7}{'median':>8}"
          f"{'p75':>7}{'p90':>7}{'<0.80':>8}{'<0.90':>8}{'>=0.95':>8}")
    dist = {}
    for coh in sorted(wanted):
        ids = [float(p['aa_identity']) for p in p_rows
               if p['cohort'] == coh and p['passes_split_gene_control'] == 'True'
               and p['aa_identity'] != '']
        if not ids:
            print(f"  {coh:20}{0:>9}")
            dist[coh] = {'n_pairs': 0}
            continue
        arr = sorted(ids)
        import numpy as _np
        q = lambda x: float(_np.percentile(arr, x))
        lt80 = sum(1 for v in arr if v < STRONGLY_DIVERGENT_ID) / len(arr)
        lt90 = sum(1 for v in arr if v < DIVERGENT_ID) / len(arr)
        ge95 = sum(1 for v in arr if v >= NEAR_IDENTICAL_ID) / len(arr)
        print(f"  {coh:20}{len(arr):>9}{q(10):>7.3f}{q(25):>7.3f}"
              f"{q(50):>8.3f}{q(75):>7.3f}{q(90):>7.3f}"
              f"{lt80*100:>7.1f}%{lt90*100:>7.1f}%{ge95*100:>7.1f}%")
        dist[coh] = {'n_pairs': len(arr), 'p10': q(10), 'p25': q(25),
                     'median': q(50), 'p75': q(75), 'p90': q(90),
                     'frac_lt_0.80': lt80, 'frac_lt_0.90': lt90,
                     'frac_ge_0.95': ge95}
    n_rej = sum(1 for p in p_rows if p['passes_split_gene_control'] == 'False')
    print(f"  split-gene control rejected {n_rej}/{len(p_rows)} candidate pairs "
          f"({n_rej / max(1, len(p_rows)) * 100:.1f}%)")

    # ---- per-cohort summary ----
    def med(vals):
        v = sorted(x for x in vals if x == x)
        return v[len(v) // 2] if v else float('nan')

    def num(rows, field):
        return [float(r[field]) for r in rows if r[field] not in ('', None)]

    print(f"\n{'cohort':20}{'n':>5}{'domain':>10}{'profFrac':>10}{'multiFam':>10}"
          f"{'divFam/100':>12}{'medDupID':>10}{'diffCtg%':>10}")
    per_cohort = {}
    for coh in sorted(wanted):
        sel2 = [r for r in g_rows if r['cohort'] == coh]
        if not sel2:
            continue
        arch = sum(1 for r in sel2 if r['domain'] == 'archaea')
        pf = med(num(sel2, 'frac_profiles_detected'))
        mf = med(num(sel2, 'n_families_multicopy'))
        dv = med(num(sel2, 'divergent_dup_families_per_100_profiles'))
        di = med(num(sel2, 'median_dup_pair_identity'))
        tot_pairs = sum(num(sel2, 'n_dup_pairs'))
        tot_diff = sum(num(sel2, 'n_dup_pairs_diff_contig'))
        dc = (tot_diff / tot_pairs * 100) if tot_pairs else float('nan')
        print(f"{coh:20}{len(sel2):>5}{f'{arch}arc':>10}{pf:>10.3f}{mf:>10.1f}"
              f"{dv:>12.2f}{di:>10.3f}{dc:>10.1f}")
        per_cohort[coh] = {
            'n': len(sel2), 'n_archaea': arch,
            'median_frac_profiles_detected': pf,
            'median_n_families_multicopy': mf,
            'median_divergent_dup_families_per_100_profiles': dv,
            'median_dup_pair_identity': None if di != di else di,
            'pct_dup_pairs_on_different_contigs': None if dc != dc else round(dc, 2),
            'total_dup_pairs': tot_pairs,
        }

    summary = {
        'method': ('Prodigal -p single; hmmsearch --cut_tc against 85_bcg.hmm '
                   '(85 bacterial SCGs) and uacg.hmm (128 archaeal SCGs); domain '
                   'assigned by larger recovered profile fraction; duplicated '
                   'family = >=2 distinct proteins hitting one profile; copy '
                   'divergence = global BLOSUM62 pairwise aa identity over the '
                   'profile-aligned region'),
        'thresholds': {
            'divergent_below': DIVERGENT_ID,
            'strongly_divergent_below': STRONGLY_DIVERGENT_ID,
            'near_identical_at_or_above': NEAR_IDENTICAL_ID,
            'threshold_justification': (
                'Core-gene aa identity is typically >97% within a species and '
                '~90-95% within a genus, falling well below 90% between families '
                'and phyla; recent paralogs start near-identical. Two copies of a '
                'gene that is single-copy by construction with <90% identity are '
                'therefore beyond intra-species/recent-duplication expectation '
                'and indicate distinct organisms. <80% is reported as a '
                'conservative tier.'),
        },
        'split_gene_control': {
            'min_hmm_reciprocal_overlap': MIN_HMM_SPAN_OVERLAP,
            'min_hmm_coverage_each_copy': MIN_HMM_SPAN_COVERAGE,
            'rationale': ('A single gene split across contigs yields two partial '
                          'hits to one profile covering DIFFERENT profile regions, '
                          'which would be miscounted as a divergent duplicate. '
                          'Genuine duplicates must cover substantially the same '
                          'profile region in both copies.'),
            'n_pairs_rejected': n_rej,
            'n_pairs_total': len(p_rows),
        },
        'divergence_distribution_per_cohort': dist,
        'database_independent': True,
        'rationale': ('Genes single-copy by construction: >=2 divergent copies '
                      'implies >1 organism irrespective of database coverage, so '
                      'this evidence is immune to the novelty confound that '
                      'affects Kraken2 and GUNC.'),
        'limitations': [
            'Absence of duplicated core genes does NOT prove absence of '
            'contamination: a contaminant contributing few or no single-copy core '
            'genes (low-completeness contaminant, or one whose core genes were '
            'lost during assembly/binning) is invisible to this assay. Negative '
            'results are therefore uninformative about such contamination.',
            'Sensitivity scales with the number of profiles searched (85 bacterial '
            '/ 128 archaeal), so a contaminant contributing a single core gene may '
            'produce at most one duplicated family.',
            'Prodigal gene calling on fragmented assemblies can truncate genes; the '
            'HMM-span control removes split-gene pairs but a truncated copy that '
            'still covers the same profile region is retained.',
            'Recent genuine gene duplication and redundant contigs remain possible '
            'explanations at high identity, which is why the full divergence '
            'distribution is reported rather than a bare duplicate count.',
        ],
        'n_genomes': len(g_rows), 'n_failed': len(bad),
        'per_cohort': per_cohort,
        'outputs': {'per_genome': str(out_tsv), 'per_family': str(fam_tsv),
                    'per_pair': str(pair_tsv)},
    }
    (out_dir / 'scg_duplication_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_tsv}")
    print(f"Wrote {pair_tsv}")
    print(f"Wrote {fam_tsv}")
    print(f"Wrote {out_dir / 'scg_duplication_summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
