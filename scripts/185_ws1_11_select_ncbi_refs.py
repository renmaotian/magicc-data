#!/usr/bin/env python3
"""
WS1.11 (R1-m13) -- circularity safeguard: select reference genomes WITHOUT CheckM2.

THE OBJECTION
-------------
Every reference genome in this project passed a CheckM2-based curation filter
(GTDB metadata -> checkm2 completeness >= 98 %, contamination <= 2 %, contig count
< 100, N50 > 20 kbp, longest contig > 100 kbp -> 277,183 genomes -> 100,000 sampled
-> train/val/test).  Ground truth for BOTH training and benchmarking therefore comes
from genomes that CheckM2 judged near-complete and clean.  If CheckM2 systematically
mis-scores a class of genome, those genomes were removed from the pool, so MAGICC
learned a CheckM2-curated view of genome space AND the benchmark cannot reveal the
error, because the affected genomes are absent.

WHAT THIS SCRIPT DOES DIFFERENTLY
---------------------------------
References are selected by NCBI's ``assembly_level == "Complete Genome"`` taken from
``assembly_summary_{genbank,refseq}.txt`` -- a submitter/NCBI assembly annotation that
does not involve CheckM2 -- and **no CheckM2 filter is applied**.  The design
deliberately INCLUDES genomes the original curation would have rejected.

Two arms, matched:

  H_fail : NCBI-Complete references that WOULD HAVE FAILED the original curation
           filter.  *This subgroup is the experiment.*
  H_pass : NCBI-Complete references that would have PASSED it, matched 1:1 to the
           H_fail references at the deepest available taxonomic rank
           (species > genus > family > order > class > phylum > domain) and then by
           closest genome size.  Matching prevents the arms differing in lineage
           composition, which is the confound that invalidated the first WS1.6
           decision rule.

FILTERS THAT *ARE* APPLIED (none of them uses CheckM2)
-----------------------------------------------------
1. NCBI assembly_level == "Complete Genome" and version_status == "latest".
2. Taxonomic-consistency filter (protocol WS1.11): the genome must carry both a GTDB
   and an NCBI taxonomy and they must agree at genus level after normalising GTDB's
   alphabetic suffixes (g__Escherichia_A -> Escherichia).  Genomes with a missing or
   placeholder assignment at either source are dropped.  Drop counts are recorded.
3. Leakage exclusion: the genome must not be in the training split, the validation
   split, or the 2,000-genome 9-mer feature-selection set, under a GCA<->GCF
   cross-map keyed on the *version-stripped numeric assembly id* (GCA_000227705 ==
   GCF_000227705), which is strictly more conservative than matching full accessions
   because RefSeq and GenBank versions of one assembly can differ
   (1,379/226,328 such rows exist in data/gtdb/filtered_genomes.tsv).

CheckM2 scores are read ONLY to LABEL a reference as would-pass / would-fail after
selection, never to select or exclude one.

STATED BOUNDARY
---------------
Candidates must be present in GTDB, because the taxonomic-consistency filter needs a
GTDB taxonomy.  GTDB applies its own quality floor, so NCBI-Complete assemblies that
GTDB rejected outright are not reachable here; the count is reported.  That floor
(CheckM completeness >= 50 %, contamination <= 10 %) is far weaker than the project's
>= 98 % / <= 2 % filter, so the genomes that matter for R1-m13 -- those excluded by
the project's filter but not by GTDB's -- are exactly the ones this set contains.

OUTPUTS
-------
  data/benchmarks/set_H_ncbi/reference_selection.tsv     the 2 x N chosen references
  data/benchmarks/set_H_ncbi/candidate_pool.tsv.gz       every eligible candidate
  results/revision/circularity/ws1_11_reference_selection.json
  results/revision/circularity/ws1_11_filter_accounting.tsv

Usage:
    python scripts/185_ws1_11_select_ncbi_refs.py [--n-per-arm 200]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/path/to/magicc')
GTDB = ROOT / 'data' / 'gtdb'
NCBI = ROOT / 'data' / 'ncbi'
SPLITS = ROOT / 'data' / 'splits'
OUT_SET = ROOT / 'data' / 'benchmarks' / 'set_H_ncbi'
OUT_RES = ROOT / 'results' / 'revision' / 'circularity'

SEED_FAIL = 7011          # which would-have-failed references are drawn
SEED_MATCH = 7012         # tie-breaking inside the matched control search

# The project's ORIGINAL curation filter, reproduced verbatim from
# scripts/02_filter_genomes.py / Phase 1 of project_progress_and_results.md.
FILTER = dict(completeness_min=98.0, contamination_max=2.0,
              contig_count_max=100, n50_min=20_000, longest_contig_min=100_000)

GTDB_COLS = ['accession', 'checkm2_completeness', 'checkm2_contamination',
             'contig_count', 'n50_contigs', 'longest_contig', 'genome_size',
             'gtdb_taxonomy', 'ncbi_taxonomy', 'ncbi_assembly_level',
             'ncbi_genbank_assembly_accession', 'ncbi_genome_category',
             'gtdb_representative', 'ncbi_organism_name', 'ncbi_biosample',
             'ncbi_taxid', 'ncbi_species_taxid', 'gc_percentage',
             'ncbi_assembly_type', 'ncbi_genome_representation', 'ambiguous_bases',
             'ncbi_seq_rel_date', 'ncbi_submitter', 'mimag_high_quality']

RANKS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
_SUFFIX = re.compile(r'_[A-Z]+$')


# --------------------------------------------------------------------- helpers
def canonical_id(acc: str) -> str | None:
    """Version-stripped numeric assembly id shared by the GCA and GCF forms.

    'GB_GCA_000227705.3' -> '000227705'; 'GCF_000227705.2' -> '000227705'.
    """
    if not isinstance(acc, str) or not acc:
        return None
    a = acc.replace('GB_', '').replace('RS_', '')
    m = re.match(r'GC[AF]_(\d+)', a)
    return m.group(1) if m else None


def split_tax(series: pd.Series) -> pd.DataFrame:
    parts = series.fillna('').str.split(';', expand=True)
    parts = parts.reindex(columns=range(7))
    out = {}
    for i, r in enumerate(RANKS):
        col = parts[i].fillna('').astype(str).str.strip()
        out[r] = col.str.replace(r'^[a-z]__', '', regex=True)
    return pd.DataFrame(out, index=series.index)


def norm_taxon(x: str) -> str:
    """Strip GTDB alphabetic suffixes so g__Escherichia_A == Escherichia."""
    if not isinstance(x, str):
        return ''
    return _SUFFIX.sub('', x.strip())


def load_assembly_summaries() -> pd.DataFrame:
    """assembly_accession -> (assembly_level, version_status, ftp_path, ...)."""
    frames = []
    for name, src in (('assembly_summary_genbank.txt', 'genbank'),
                      ('assembly_summary_refseq.txt', 'refseq')):
        p = NCBI / name
        if not p.exists():
            raise SystemExit(f'FATAL: {p} missing')
        df = pd.read_csv(p, sep='\t', skiprows=1, dtype=str, low_memory=False,
                         usecols=lambda c: c in {
                             '#assembly_accession', 'assembly_level', 'version_status',
                             'ftp_path', 'asm_name', 'genome_rep', 'assembly_type',
                             'excluded_from_refseq', 'group', 'seq_rel_date',
                             'gbrs_paired_asm', 'paired_asm_comp', 'refseq_category',
                             'relation_to_type_material', 'biosample'})
        df = df.rename(columns={'#assembly_accession': 'assembly_accession'})
        df['summary_source'] = src
        frames.append(df)
    s = pd.concat(frames, ignore_index=True)
    return s


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-arm', type=int, default=200)
    args = ap.parse_args()
    OUT_SET.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    acct = []                       # filter accounting rows

    def step(stage, n, note=''):
        acct.append({'stage': stage, 'n_remaining': int(n), 'note': note})
        print(f'  {stage:<62s} n = {n:>8,}   {note}')

    print('=' * 92)
    print('WS1.11 -- NCBI-Complete reference selection (CheckM2 removed from the loop)')
    print('=' * 92)

    # ---------------------------------------------------------------- GTDB
    print('\n[1] GTDB metadata')
    frames = []
    for f, dom in (('bac120_metadata.tsv.gz', 'Bacteria'),
                   ('ar53_metadata.tsv.gz', 'Archaea')):
        d = pd.read_csv(GTDB / f, sep='\t', usecols=GTDB_COLS, dtype=str,
                        low_memory=False)
        d['domain_src'] = dom
        frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    for c in ('checkm2_completeness', 'checkm2_contamination', 'contig_count',
              'n50_contigs', 'longest_contig', 'genome_size', 'gc_percentage',
              'ambiguous_bases'):
        g[c] = pd.to_numeric(g[c], errors='coerce')
    step('GTDB genomes (bac120 + ar53)', len(g))

    g['canon_id'] = g['accession'].map(canonical_id)
    g['gtdb_is_refseq'] = g['accession'].str.startswith('RS_')
    g['primary_accession'] = g['accession'].str.replace('^(GB|RS)_', '', regex=True)

    # ------------------------------------------------- NCBI assembly summaries
    print('\n[2] NCBI assembly summaries (the CheckM2-independent selector)')
    s = load_assembly_summaries()
    print(f'    assembly_summary rows: {len(s):,} '
          f'({(s.summary_source == "genbank").sum():,} GenBank / '
          f'{(s.summary_source == "refseq").sum():,} RefSeq)')
    s_latest = s[s['version_status'] == 'latest']
    complete_acc = set(s_latest.loc[s_latest['assembly_level'] == 'Complete Genome',
                                    'assembly_accession'])
    chrom_acc = set(s_latest.loc[s_latest['assembly_level'] == 'Chromosome',
                                 'assembly_accession'])
    print(f'    NCBI "Complete Genome" + latest : {len(complete_acc):,}')
    print(f'    NCBI "Chromosome"      + latest : {len(chrom_acc):,}')

    smap = s.drop_duplicates('assembly_accession').set_index('assembly_accession')

    # A GTDB genome is a candidate iff *its own* accession (GCF for RS_, GCA for GB_)
    # is annotated Complete Genome in the corresponding NCBI summary.
    g['ncbi_level_from_summary'] = g['primary_accession'].map(
        smap['assembly_level']).fillna('')
    g['ncbi_version_status'] = g['primary_accession'].map(
        smap['version_status']).fillna('')
    g['ftp_path'] = g['primary_accession'].map(smap['ftp_path']).fillna('')
    g['ncbi_group'] = g['primary_accession'].map(smap['group']).fillna('')
    g['excluded_from_refseq'] = g['primary_accession'].map(
        smap['excluded_from_refseq']).fillna('')
    g['relation_to_type_material'] = g['primary_accession'].map(
        smap['relation_to_type_material']).fillna('')

    in_summary = g['ncbi_level_from_summary'] != ''
    print(f'    GTDB genomes found in an assembly summary: {int(in_summary.sum()):,} '
          f'/ {len(g):,}')
    agree = (g.loc[in_summary, 'ncbi_level_from_summary']
             == g.loc[in_summary, 'ncbi_assembly_level'])
    print(f'    GTDB ncbi_assembly_level agrees with the summary file: '
          f'{int(agree.sum()):,}/{int(in_summary.sum()):,} '
          f'({100 * agree.mean():.2f} %)  [cross-check only]')

    # NCBI-Complete assemblies that GTDB does not contain -> unreachable, reported
    gtdb_prok_acc = set(g['primary_accession'])
    prok_summary = s_latest[s_latest['group'].isin(['bacteria', 'archaea'])]
    prok_complete = set(prok_summary.loc[
        prok_summary['assembly_level'] == 'Complete Genome', 'assembly_accession'])
    unreachable = len(prok_complete - gtdb_prok_acc)
    print(f'    NCBI prokaryotic Complete Genomes NOT in GTDB (unreachable, stated '
          f'as a boundary): {unreachable:,} of {len(prok_complete):,}')

    cand = g[in_summary
             & (g['ncbi_level_from_summary'] == 'Complete Genome')
             & (g['ncbi_version_status'] == 'latest')].copy()
    step('NCBI assembly_level == "Complete Genome" (latest)', len(cand),
         'selector = NCBI/submitter annotation; NO CheckM2 filter applied')

    cand = cand[cand['ftp_path'].str.startswith('http')]
    step('has an NCBI ftp_path (downloadable)', len(cand))

    # ----------------------------------------- taxonomic-consistency filter
    print('\n[3] Taxonomic-consistency filter (GTDB vs NCBI)')
    gt = split_tax(cand['gtdb_taxonomy'])
    nt = split_tax(cand['ncbi_taxonomy'])
    for r in RANKS:
        cand[f'gtdb_{r}'] = gt[r].values
        cand[f'ncbi_{r}'] = nt[r].values

    n0 = len(cand)
    have_gtdb = (cand['gtdb_species'] != '') & (cand['gtdb_genus'] != '')
    drop_no_gtdb = int((~have_gtdb).sum())
    cand = cand[have_gtdb]
    have_ncbi = (cand['ncbi_genus'] != '') & (cand['ncbi_species'] != '')
    drop_no_ncbi = int((~have_ncbi).sum())
    cand = cand[have_ncbi]

    gg = cand['gtdb_genus'].map(norm_taxon)
    ng = cand['ncbi_genus'].map(norm_taxon)
    genus_ok = gg == ng
    drop_genus_mismatch = int((~genus_ok).sum())
    gd = cand['gtdb_domain'].map(norm_taxon)
    nd = cand['ncbi_domain'].map(norm_taxon)
    gp = cand['gtdb_phylum'].map(norm_taxon)
    np_ = cand['ncbi_phylum'].map(norm_taxon)
    print(f'    domain agreement : {int((gd == nd).sum()):,}/{len(cand):,}')
    print(f'    phylum agreement : {int((gp == np_).sum()):,}/{len(cand):,}')
    print(f'    genus  agreement : {int(genus_ok.sum()):,}/{len(cand):,}')
    cand = cand[genus_ok & (gd == nd)]
    step('taxonomic-consistency filter (GTDB<->NCBI genus + domain agree)', len(cand),
         f'dropped {drop_no_gtdb} no/placeholder GTDB assignment, '
         f'{drop_no_ncbi} no NCBI assignment, {drop_genus_mismatch} genus mismatch '
         f'(of {n0})')

    # --------------------------------------------------------- leakage exclusion
    print('\n[4] Leakage exclusion (GCA<->GCF cross-mapped, version-stripped)')
    fil = pd.read_csv(GTDB / 'filtered_genomes.tsv', sep='\t', dtype=str)
    xmap = defaultdict(set)     # canonical id -> {every accession string seen}
    for col in ('gtdb_accession', 'ncbi_accession', 'gcf_accession'):
        for a in fil[col].dropna():
            c = canonical_id(a)
            if c:
                xmap[c].add(a)
    n_strings = sum(len(v) for v in xmap.values())
    print(f'    cross-map: {len(fil):,} filtered rows -> {n_strings:,} accession '
          f'strings -> {len(xmap):,} canonical assemblies')

    def acc_set(path):
        return {ln.strip() for ln in open(path) if ln.strip()}

    train_acc = acc_set(SPLITS / 'train_accessions.txt')
    val_acc = acc_set(SPLITS / 'val_accessions.txt')
    test_acc = acc_set(SPLITS / 'test_accessions.txt')
    train_ids = {canonical_id(a) for a in train_acc} - {None}
    val_ids = {canonical_id(a) for a in val_acc} - {None}
    test_ids = {canonical_id(a) for a in test_acc} - {None}
    pool_ids = set(xmap)                     # the 277,183 CheckM2-filtered pool
    print(f'    train {len(train_ids):,} | val {len(val_ids):,} | test {len(test_ids):,}'
          f' | 277k pool {len(pool_ids):,} canonical ids')

    # 9-mer feature-selection genomes: same two files scripts/74_provenance_audit.py uses
    kmer_ids, n_kmer_rows = set(), 0
    for fn in ('selected_bacterial_1000.tsv', 'selected_archaeal_1000.tsv'):
        kp = ROOT / 'data' / 'kmer_selection' / fn
        if not kp.exists():
            raise SystemExit(f'FATAL: {kp} missing')
        kk = pd.read_csv(kp, sep='\t', dtype=str)
        n_kmer_rows += len(kk)
        for col in ('gtdb_accession', 'ncbi_accession', 'gcf_accession'):
            if col in kk.columns:
                kmer_ids |= {canonical_id(a) for a in kk[col].dropna()}
    kmer_ids -= {None}
    print(f'    9-mer feature-selection genomes: {n_kmer_rows} rows -> '
          f'{len(kmer_ids):,} canonical ids')

    cand['in_train'] = cand['canon_id'].isin(train_ids)
    cand['in_val'] = cand['canon_id'].isin(val_ids)
    cand['in_test'] = cand['canon_id'].isin(test_ids)
    cand['in_kmer_selection'] = cand['canon_id'].isin(kmer_ids)
    cand['in_277k_pool'] = cand['canon_id'].isin(pool_ids)
    print(f'    candidates in TRAIN {int(cand.in_train.sum()):,} | '
          f'VAL {int(cand.in_val.sum()):,} | TEST {int(cand.in_test.sum()):,} | '
          f'9-mer set {int(cand.in_kmer_selection.sum()):,}')

    cand = cand[~(cand['in_train'] | cand['in_val'] | cand['in_kmer_selection'])]
    step('excluded train / val / 9-mer-selection genomes', len(cand),
         'references are absent from training entirely')

    # ------------------------------------------- would-pass / would-fail labels
    print('\n[5] Labelling against the ORIGINAL curation filter (post hoc only)')
    c_comp = cand['checkm2_completeness']
    c_cont = cand['checkm2_contamination']
    pass_comp = c_comp >= FILTER['completeness_min']
    pass_cont = c_cont <= FILTER['contamination_max']
    pass_ctg = cand['contig_count'] < FILTER['contig_count_max']
    pass_n50 = cand['n50_contigs'] > FILTER['n50_min']
    pass_lc = cand['longest_contig'] > FILTER['longest_contig_min']
    cand['would_pass_original_filter'] = (pass_comp & pass_cont & pass_ctg
                                          & pass_n50 & pass_lc).fillna(False)
    cand['fails_checkm2_criteria'] = (~pass_comp | ~pass_cont).fillna(True)
    cand['fails_assembly_criteria'] = (~pass_ctg | ~pass_n50 | ~pass_lc).fillna(True)
    cand['checkm2_completeness_deficit'] = (
        FILTER['completeness_min'] - c_comp).clip(lower=0)
    cand['checkm2_contamination_excess'] = (
        c_cont - FILTER['contamination_max']).clip(lower=0)

    def severity(row):
        if row['would_pass_original_filter']:
            return 'pass'
        if row['checkm2_completeness'] < 90 or row['checkm2_contamination'] > 10:
            return 'severe'
        if row['checkm2_completeness'] < 96 or row['checkm2_contamination'] > 4:
            return 'moderate'
        return 'mild'
    cand['severity'] = cand.apply(severity, axis=1)

    n_pass = int(cand['would_pass_original_filter'].sum())
    n_fail = int((~cand['would_pass_original_filter']).sum())
    print(f'    would PASS the original filter : {n_pass:,}')
    print(f'    would FAIL the original filter : {n_fail:,}')
    print(f'      fails a CheckM2 criterion    : '
          f'{int(cand.loc[~cand.would_pass_original_filter, "fails_checkm2_criteria"].sum()):,}')
    print(f'      fails only assembly criteria : '
          f'{int((~cand.would_pass_original_filter & ~cand.fails_checkm2_criteria).sum()):,}')
    print('    severity strata among would-FAIL:')
    print(cand.loc[~cand.would_pass_original_filter, 'severity']
          .value_counts().to_string())

    cand.to_csv(OUT_SET / 'candidate_pool.tsv.gz', sep='\t', index=False,
                compression='gzip')
    print(f'    wrote {OUT_SET / "candidate_pool.tsv.gz"}')

    # --------------------------------------------------------------- sampling
    print(f'\n[6] Sampling {args.n_per_arm} would-FAIL references '
          f'(sqrt-proportional across severity strata, seed {SEED_FAIL})')
    failp = cand[~cand['would_pass_original_filter']].sort_values(
        'accession').reset_index(drop=True)
    strata = ['mild', 'moderate', 'severe']
    counts = {s: int((failp['severity'] == s).sum()) for s in strata}
    w = {s: np.sqrt(counts[s]) for s in strata}
    tot = sum(w.values())
    alloc = {s: int(round(args.n_per_arm * w[s] / tot)) for s in strata}
    # repair rounding and pool limits
    for s in strata:
        alloc[s] = min(alloc[s], counts[s])
    while sum(alloc.values()) != args.n_per_arm:
        d = args.n_per_arm - sum(alloc.values())
        order = sorted(strata, key=lambda s: counts[s] - alloc[s], reverse=(d > 0))
        for s in order:
            if d > 0 and alloc[s] < counts[s]:
                alloc[s] += 1
                d -= 1
            elif d < 0 and alloc[s] > 0:
                alloc[s] -= 1
                d += 1
            if d == 0:
                break
    print(f'    strata sizes {counts}  ->  allocation {alloc}  '
          '(sqrt-proportional, the same scheme Phase 1 used across phyla; it '
          'guarantees the severe stratum is represented)')

    rng = np.random.default_rng(SEED_FAIL)
    picks = []
    for s in strata:
        sub = failp[failp['severity'] == s].reset_index(drop=True)
        idx = np.sort(rng.choice(len(sub), size=alloc[s], replace=False))
        picks.append(sub.iloc[idx])
    fail_sel = pd.concat(picks, ignore_index=True).sort_values(
        'accession').reset_index(drop=True)
    fail_sel['arm'] = 'H_fail'

    # ------------------------------------------------------------- matching
    print(f'\n[7] Matched would-PASS controls (deepest common rank, then closest '
          f'genome size; seed {SEED_MATCH})')
    passp = cand[cand['would_pass_original_filter']].sort_values(
        'accession').reset_index(drop=True)
    match_rng = np.random.default_rng(SEED_MATCH)
    # jitter breaks exact ties deterministically without biasing the size criterion
    passp['_tie'] = match_rng.random(len(passp))
    idx_by = {r: defaultdict(list) for r in RANKS}
    for i, row in enumerate(passp.itertuples(index=False)):
        for r in RANKS:
            key = norm_taxon(getattr(row, f'gtdb_{r}'))
            if key:
                idx_by[r][key].append(i)
    used = set()
    ctrl_rows, match_levels = [], []
    order = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'domain']
    logsize_pass = np.log10(passp['genome_size'].astype(float).clip(lower=1))
    tie = passp['_tie'].to_numpy()
    for _, fr in fail_sel.iterrows():
        chosen, level = None, None
        target = np.log10(max(float(fr['genome_size']), 1.0))
        for r in order:
            key = norm_taxon(fr[f'gtdb_{r}'])
            if not key:
                continue
            pool = [i for i in idx_by[r].get(key, []) if i not in used]
            if not pool:
                continue
            pa = np.array(pool)
            d = np.abs(logsize_pass.values[pa] - target) + 1e-9 * tie[pa]
            chosen = int(pa[int(np.argmin(d))])
            level = r
            break
        if chosen is None:
            match_levels.append('unmatched')
            continue
        used.add(chosen)
        row = passp.iloc[chosen].copy()
        row['arm'] = 'H_pass'
        row['matched_to'] = fr['accession']
        row['match_level'] = level
        ctrl_rows.append(row)
        match_levels.append(level)
    ctrl_sel = pd.DataFrame(ctrl_rows).reset_index(drop=True)
    fail_sel['match_level'] = match_levels[:len(fail_sel)]
    fail_sel['matched_to'] = ''
    for i, r in enumerate(ctrl_rows):
        pass
    # link each fail row to its control
    m = {r['matched_to']: r['accession'] for _, r in ctrl_sel.iterrows()}
    fail_sel['matched_to'] = fail_sel['accession'].map(m).fillna('')
    print('    match level achieved:')
    print(pd.Series(match_levels).value_counts().to_string())
    print(f'    matched pairs: {len(ctrl_sel)} / {len(fail_sel)}')

    fail_sel = fail_sel[fail_sel['matched_to'] != ''].reset_index(drop=True)
    ctrl_sel = ctrl_sel[ctrl_sel['matched_to'].isin(set(fail_sel['accession']))
                        ].reset_index(drop=True)
    pair_id = {a: f'pair_{i:04d}' for i, a in enumerate(fail_sel['accession'])}
    fail_sel['pair_id'] = fail_sel['accession'].map(pair_id)
    ctrl_sel['pair_id'] = ctrl_sel['matched_to'].map(pair_id)
    ctrl_sel['match_level'] = ctrl_sel['match_level'].astype(str)

    sel = pd.concat([fail_sel, ctrl_sel], ignore_index=True)
    sel = sel.sort_values(['pair_id', 'arm']).reset_index(drop=True)
    sel.insert(0, 'ref_index', np.arange(len(sel)))
    sel['selection_seed_fail'] = SEED_FAIL
    sel['selection_seed_match'] = SEED_MATCH
    sel['fasta_path_resolved'] = [
        str(OUT_SET / 'references' / f'{a}.fna') for a in sel['primary_accession']]

    sel.to_csv(OUT_SET / 'reference_selection.tsv', sep='\t', index=False)
    step('FINAL selected references (both arms)', len(sel),
         f'{len(fail_sel)} would-FAIL + {len(ctrl_sel)} matched would-PASS')
    print(f'    wrote {OUT_SET / "reference_selection.tsv"}')

    # ------------------------------------------------------------------ report
    def arm_stats(d):
        gs = d['genome_size'].astype(float)
        return {
            'n': int(len(d)),
            'n_phyla': int(d['gtdb_phylum'].nunique()),
            'n_classes': int(d['gtdb_class'].nunique()),
            'n_orders': int(d['gtdb_order'].nunique()),
            'n_families': int(d['gtdb_family'].nunique()),
            'n_genera': int(d['gtdb_genus'].nunique()),
            'n_species': int(d['gtdb_species'].nunique()),
            'domain_counts': d['domain_src'].value_counts().to_dict(),
            'genome_size_Mbp': {'median': round(float(gs.median()) / 1e6, 4),
                                'min': round(float(gs.min()) / 1e6, 4),
                                'max': round(float(gs.max()) / 1e6, 4)},
            'contig_count_median': float(d['contig_count'].median()),
            'checkm2_completeness': {
                'median': float(d['checkm2_completeness'].median()),
                'min': float(d['checkm2_completeness'].min()),
                'max': float(d['checkm2_completeness'].max())},
            'checkm2_contamination': {
                'median': float(d['checkm2_contamination'].median()),
                'min': float(d['checkm2_contamination'].min()),
                'max': float(d['checkm2_contamination'].max())},
            'severity_counts': d['severity'].value_counts().to_dict(),
            'in_test_split': int(d['in_test'].sum()),
            'in_277k_pool': int(d['in_277k_pool'].sum()),
            'genome_category': d['ncbi_genome_category'].value_counts().to_dict(),
        }

    rep = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'generated_by': 'scripts/185_ws1_11_select_ncbi_refs.py',
        'objection': 'R1-m13 circularity: references were previously chosen with a '
                     'CheckM2-based filter, so the benchmark cannot reveal CheckM2 '
                     'failure modes on genomes that filter removed.',
        'selector': 'NCBI assembly_summary_{genbank,refseq}.txt '
                    'assembly_level == "Complete Genome" AND version_status == latest',
        'checkm2_used_for_selection': False,
        'original_curation_filter': FILTER,
        'ncbi_complete_prokaryotic_not_in_gtdb': int(unreachable),
        'gtdb_level_agrees_with_summary_file': float(agree.mean()),
        'seeds': {'fail_sampling': SEED_FAIL, 'control_matching': SEED_MATCH},
        'filter_accounting': acct,
        'candidate_pool': {
            'n_total': int(len(cand)),
            'n_would_pass': n_pass,
            'n_would_fail': n_fail,
            'would_fail_severity': cand.loc[~cand.would_pass_original_filter,
                                            'severity'].value_counts().to_dict(),
            'would_fail_reason': {
                'checkm2_completeness_lt_98': int(
                    (cand['checkm2_completeness'] < 98).sum()),
                'checkm2_contamination_gt_2': int(
                    (cand['checkm2_contamination'] > 2).sum()),
                'contig_count_ge_100': int((cand['contig_count'] >= 100).sum()),
                'n50_le_20kb': int((cand['n50_contigs'] <= 20000).sum()),
                'longest_contig_le_100kb': int(
                    (cand['longest_contig'] <= 100000).sum()),
            },
        },
        'severity_allocation': {'strata_sizes': counts, 'allocation': alloc,
                                'scheme': 'sqrt-proportional'},
        'match_level_counts': pd.Series(match_levels).value_counts().to_dict(),
        'arms': {'H_fail': arm_stats(fail_sel), 'H_pass': arm_stats(ctrl_sel)},
    }
    with open(OUT_RES / 'ws1_11_reference_selection.json', 'w') as f:
        json.dump(rep, f, indent=2, default=lambda o: (
            int(o) if isinstance(o, (np.integer,)) else
            float(o) if isinstance(o, (np.floating,)) else str(o)))
    pd.DataFrame(acct).to_csv(OUT_RES / 'ws1_11_filter_accounting.tsv',
                              sep='\t', index=False)
    print(f'\nwrote {OUT_RES / "ws1_11_reference_selection.json"}')
    print(f'wrote {OUT_RES / "ws1_11_filter_accounting.tsv"}')
    print('\nARM SUMMARY')
    for k, v in rep['arms'].items():
        print(f'  {k}: n={v["n"]} phyla={v["n_phyla"]} genera={v["n_genera"]} '
              f'size median {v["genome_size_Mbp"]["median"]} Mbp  '
              f'checkm2 comp median {v["checkm2_completeness"]["median"]} '
              f'cont median {v["checkm2_contamination"]["median"]}  '
              f'severity {v["severity_counts"]}')
    print('\nDONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
