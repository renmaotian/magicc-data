#!/usr/bin/env python3
"""
WS3.8 (R1-M3) — CAMI II ground truth in MAGICC's denominator convention, plus the two
bin sets the benchmark is built on.

=============================================================================
TRUTH DERIVATION  (this is the part reviewers will scrutinise; it is exact)
=============================================================================
CAMI II ships, for every sample:
  anonymous_gsa.fasta.gz  the gold-standard assembly (GSA) of that sample
  gsa_mapping.tsv.gz      one row per GSA contig:
                            anonymous_contig_id, genome_id, tax_id, contig_id,
                            number_reads, start_position, end_position
                          i.e. the EXACT source genome of every contig and the exact
                          interval of that source genome the contig reproduces
  binning_gs.tsv          the binning gold standard (same assignment, plus _LENGTH)

Every GSA contig therefore has (a) exactly one true source genome and (b) an exact
[start, end] interval in that genome. Truth needs no alignment and no inference.

For a bin B (a set of GSA contigs):
  dominant genome  d(B) = argmax_g  sum of contig lengths of contigs whose source is g
  R(d)             = FULL reference length of d = total bp of all sequences in
                     source_genomes/<d>.fasta  (NOT the assembled length, NOT the bp
                     present in the sample)

  completeness(B) = 100 * (retained bp of d in B)      / R(d)
  contamination(B)= 100 * (bp in B whose source is not d) / R(d)

That is MAGICC's own convention verbatim: `magicc/contamination.py` defines
    completeness  = dominant_actual_bp  / dominant_genome_full_length
    contamination = contaminant_total_bp / dominant_genome_full_length
(see compute_contamination_rate() and create_contaminated_genome()). Using the same
denominator for truth and for every tool is what R1-M5 asks for.

"Retained bp of d" is measured two ways and BOTH are written out:
  span_bp   sum over contigs of (end_position - start_position + 1)   [primary]
  union_bp  bp of the UNION of the [start,end] intervals per source contig, so that
            any overlapping GSA contigs are counted once
They are reported side by side; `union_bp` is the sensitivity analysis that proves the
primary number is not inflated by overlapping contigs. Completeness can never exceed
100 % under `union_bp` by construction.

=============================================================================
BIN SET (i) — GOLD-STANDARD BINS
=============================================================================
Contigs grouped by their true source genome, exactly as `binning_gs.tsv` specifies.
Pure by construction (contamination = 0). Their completeness is whatever CAMI's
independent read simulation + gold-standard assembly produced, so this is a completeness
gradient on third-party contigs that we did not choose, assemble or bin.

=============================================================================
BIN SET (ii) — CONSTRUCTED MIXED BINS
=============================================================================
A known completeness x contamination grid, stratified by the taxonomic distance between
dominant and contaminant (species -> genus -> family -> order -> class -> phylum), so it
is directly comparable to WS2 Set F.

Only the GROUPING is ours. Every contig is a CAMI contig, unmodified, and every truth
value comes from CAMI's own `gsa_mapping.tsv`. No sequence is edited, simulated or
re-assembled.

  dominant   : a source genome whose gold bin in that sample already reaches the target
               completeness; its contigs are subsampled (deterministic, seeded) until the
               target completeness is met
  contaminant: a different source genome at the required taxonomic distance; its contigs
               are subsampled until  contaminant_bp ~= target_cont/100 * R(dominant)
  distance   : rank of the lowest common ancestor of the two NCBI tax_ids that CAMI
               itself assigned, resolved through the NCBI taxonomy dump

In-domain constraint (protocol 4.4a): contamination % <= completeness % is enforced.

=============================================================================
OUTPUTS
=============================================================================
  data/real_data/cami2/derived/<ds>/source_genomes.tsv       ref lengths + taxonomy
  data/real_data/cami2/derived/<ds>/contigs_<sample>.parquet  per-contig truth
  data/real_data/cami2/bins/<ds>/gold/<sample>__<genome>.fna
  data/real_data/cami2/bins/<ds>/mixed/<bin_id>.fna
  results/revision/cami2/truth/<ds>_gold_truth.tsv
  results/revision/cami2/truth/<ds>_mixed_truth.tsv
  results/revision/cami2/truth/<ds>_truth_derivation.json

Usage:
    python scripts/196_ws38_cami2_truth_and_bins.py --dataset strain_madness
    python scripts/196_ws38_cami2_truth_and_bins.py --dataset marine --no-fasta
"""

import argparse
import gzip
import json
import os
import signal
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

import importlib.util as _ilu  # noqa: E402


def _load(path, name):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load(PROJECT_DIR / 'scripts' / '101_metrics_framework.py',
           'magicc_metrics_framework')
# NOTE: every derived seed below goes through fw.stable_hash (CRC-32). Python's built-in
# hash() is salted per process (PYTHONHASHSEED) and has already caused three
# irreproducibility defects on this project; it is never used here.

DATA_DIR = PROJECT_DIR / 'data' / 'real_data' / 'cami2'
DERIVED_DIR = DATA_DIR / 'derived'
BIN_DIR = DATA_DIR / 'bins'
RES_DIR = PROJECT_DIR / 'results' / 'revision' / 'cami2'
TRUTH_DIR = RES_DIR / 'truth'

TAXDUMP = PROJECT_DIR / 'tools' / 'kraken2_db_standard'

# ---- design constants (locked; changing them changes the benchmark) ----------
DESIGN_SEED = 38100                       # WS3.8
COMP_TARGETS = (60.0, 75.0, 90.0)
CONT_TARGETS = (2.0, 5.0, 10.0, 15.0, 20.0)
RANKS = ('species', 'genus', 'family', 'order', 'class', 'phylum')
MAX_PER_CELL = 25                         # dominants per (comp x cont x rank) cell
MIN_GOLD_COMP_FOR_DOMINANT = 55.0         # must clear MAGICC's 50 % floor with headroom
MAGICC_COMPLETENESS_FLOOR = 50.0
# Gold bins below the floor cannot be scored (V5 was trained on 50-100 % completeness
# only). FASTAs are written down to BELOW_FLOOR_PROBE_MIN so that a bounded, explicitly
# labelled below-floor probe cohort exists; bins below that are counted as censored in
# the truth table but never written or run.
BELOW_FLOOR_PROBE_MIN = 30.0
# CAMI's own metadata.tsv labels some marine "source genomes" as plasmid / virus
# (108 plasmids and 4 viruses). Those are not prokaryotic genomes: neither MAGICC nor
# CheckM2 defines completeness/contamination for them, and a plasmid's "full reference
# length" is not a genome size. They are kept in the truth table, flagged, and excluded
# from every bin set and from the analysis. Excluding them is stated, not silent.
EXCLUDED_NOVELTY = ('plasmid', 'virus')

DATASETS = {
    'marine': {'prefix': 'marmgCAMI2'},
    'strain_madness': {'prefix': 'strmgCAMI2'},
    'plant_associated': {'prefix': 'rhimgCAMI2'},
}


# ------------------------------------------------------------------ taxonomy
def load_taxonomy():
    """NCBI taxdump -> (parent, rank, name) dicts. Uses the full dump shipped with the
    kraken2 standard DB (2.68 M nodes), not a capped subset."""
    parent, rank = {}, {}
    with open(TAXDUMP / 'nodes.dmp') as fh:
        for line in fh:
            f = line.split('\t|\t')
            tid = int(f[0])
            parent[tid] = int(f[1])
            rank[tid] = f[2]
    names = {}
    with open(TAXDUMP / 'names.dmp') as fh:
        for line in fh:
            f = line.split('\t|\t')
            if f[3].startswith('scientific name'):
                names[int(f[0])] = f[1]
    return parent, rank, names


def lineage(tid, parent, rank):
    """tax_id -> {rank: tax_id} for the standard ranks, plus ordered ancestor list."""
    out, chain, seen = {}, [], set()
    cur = tid
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        r = rank.get(cur)
        if r and r not in out:
            out[r] = cur
        nxt = parent.get(cur)
        if nxt == cur:
            break
        cur = nxt
    return out, chain


def lca_rank(l1, c1, l2, c2, rankmap):
    """Lowest common ancestor of two tax_ids -> its rank name, restricted to RANKS.

    Returns the *finest* standard rank at which the two genomes agree. Two different
    strains carrying the same species tax_id give 'species'; a Klebsiella and an
    Escherichia give 'family'. When CAMI labelled a genome only to family level the
    result is bounded by that label, which is the honest, conservative answer.
    """
    s2 = set(c2)
    common = [t for t in c1 if t in s2]
    if not common:
        return None, None
    lca = common[0]                      # c1 is ordered leaf -> root
    r = rankmap.get(lca)
    if r in RANKS:
        return r, lca
    # LCA sits at a non-standard rank (e.g. 'clade'); walk up to the next standard rank
    idx = c1.index(lca)
    for t in c1[idx:]:
        rr = rankmap.get(t)
        if rr in RANKS:
            return rr, t
    return None, lca


# ------------------------------------------------------------------ genomes
def find_source_genome_dir(dataset):
    """The genome bundles use different internal layouts:
       strain_madness  .../short_read/source_genomes/
       marine          .../simulation_short_read/genomes/
    so locate the directory that actually holds the FASTAs."""
    root = DATA_DIR / 'genomes' / dataset
    for name in ('source_genomes', 'genomes'):
        cands = sorted(p for p in root.rglob(name) if p.is_dir()
                       and any(p.glob('*.fasta')))
        if cands:
            return cands[0]
    cands = sorted({p.parent for p in root.rglob('*.fasta')})
    if not cands:
        raise SystemExit(f'no genome FASTAs under {root}')
    return cands[0]


def fasta_total_length(path):
    """Total bp of a (possibly gzipped) FASTA."""
    op = gzip.open if str(path).endswith('.gz') else open
    total = 0
    with op(path, 'rt') as fh:
        for line in fh:
            if not line.startswith('>'):
                total += len(line.strip())
    return total


def load_genome_to_id(dataset):
    """CAMI genome_id -> source-genome FASTA basename, plus CAMI's own metadata.

    The two datasets label contigs differently:
        strain_madness  genome_id IS the FASTA basename (spades_ESBL8855_S12_L001)
        marine          genome_id is an OTU id (Otu255) and only
                        <setup>/genome_to_id.tsv resolves it to a FASTA
    Ignoring this silently loses every marine reference length, so the mapping is
    loaded whenever CAMI ships it and identity is used only as a fallback.

    metadata.tsv additionally carries CAMI's own `novelty_category`
    (known_strain / new_species / new_genus / ...), which is used as an independent
    novelty stratifier in the analysis.
    """
    root = DATA_DIR / dataset / 'setup'
    g2id, novelty = {}, {}
    for p in root.rglob('genome_to_id.tsv'):
        with open(p) as fh:
            for line in fh:
                f = line.rstrip('\n').split('\t')
                if len(f) >= 2:
                    base = Path(f[1]).name
                    for suf in ('.fasta.gz', '.fa.gz', '.fna.gz',
                                '.fasta', '.fa', '.fna'):
                        if base.endswith(suf):
                            base = base[: -len(suf)]
                            break
                    g2id[f[0]] = base
        break
    for p in root.rglob('metadata.tsv'):
        try:
            md = pd.read_csv(p, sep='\t', dtype=str)
            if 'genome_ID' in md.columns and 'novelty_category' in md.columns:
                novelty = dict(zip(md['genome_ID'], md['novelty_category']))
        except Exception:                                            # noqa: BLE001
            pass
        break
    return g2id, novelty


def reference_lengths(gdir):
    """genome_id -> full reference length. Prefers the shipped .fai (samtools index,
    exact) and falls back to reading the FASTA."""
    out, source = {}, {}
    for p in sorted(gdir.iterdir()):
        name = p.name
        if name.endswith('.fasta.fai') or name.endswith('.fa.fai'):
            gid = name.rsplit('.fasta.fai', 1)[0].rsplit('.fa.fai', 1)[0]
            total = 0
            with open(p) as fh:
                for line in fh:
                    parts = line.split('\t')
                    if len(parts) > 1:
                        total += int(parts[1])
            out[gid] = total
            source[gid] = 'fai'
    for p in sorted(gdir.iterdir()):
        n = p.name
        if n.endswith(('.fasta', '.fa', '.fna', '.fasta.gz', '.fa.gz', '.fna.gz')):
            gid = n
            for suf in ('.fasta.gz', '.fa.gz', '.fna.gz', '.fasta', '.fa', '.fna'):
                if gid.endswith(suf):
                    gid = gid[: -len(suf)]
                    break
            if gid not in out:
                out[gid] = fasta_total_length(p)
                source[gid] = 'fasta'
    return out, source


# ------------------------------------------------------------------ samples
def sample_dirs(dataset):
    """{sample_index: contigs_dir} for every sample extracted on disk."""
    root = DATA_DIR / dataset
    out = {}
    for d in sorted(root.glob('sample_*')):
        if not d.is_dir():
            continue
        idx = int(d.name.split('_')[1])
        hits = list(d.rglob('gsa_mapping.tsv.gz')) + list(d.rglob('gsa_mapping.tsv'))
        if hits:
            out[idx] = hits[0].parent
    return dict(sorted(out.items()))


def read_gsa_mapping(contigs_dir):
    """gsa_mapping -> DataFrame(contig, genome, taxid, src_contig, start, end, length)."""
    p = contigs_dir / 'gsa_mapping.tsv.gz'
    if not p.exists():
        p = contigs_dir / 'gsa_mapping.tsv'
    df = pd.read_csv(p, sep='\t', dtype={0: str, 1: str, 2: 'Int64', 3: str})
    df.columns = [c.lstrip('#') for c in df.columns]
    df = df.rename(columns={'anonymous_contig_id': 'contig', 'genome_id': 'genome',
                            'tax_id': 'taxid', 'contig_id': 'src_contig',
                            'start_position': 'start', 'end_position': 'end'})
    df['start'] = df['start'].astype('int64')
    df['end'] = df['end'].astype('int64')
    df['length'] = df['end'] - df['start'] + 1
    return df[['contig', 'genome', 'taxid', 'src_contig', 'start', 'end', 'length']]


def union_bp_per_genome(df):
    """Exact union of [start,end] intervals per (genome, src_contig): removes any bp
    double counting from overlapping GSA contigs."""
    out = defaultdict(int)
    df2 = df.sort_values(['genome', 'src_contig', 'start'], kind='mergesort')
    g_arr = df2['genome'].to_numpy()
    c_arr = df2['src_contig'].to_numpy()
    s_arr = df2['start'].to_numpy()
    e_arr = df2['end'].to_numpy()
    cur_key, cs, ce = None, None, None
    for i in range(len(df2)):
        key = (g_arr[i], c_arr[i])
        if key != cur_key:
            if cur_key is not None:
                out[cur_key[0]] += ce - cs + 1
            cur_key, cs, ce = key, s_arr[i], e_arr[i]
            continue
        if s_arr[i] <= ce + 1:
            ce = max(ce, e_arr[i])
        else:
            out[cur_key[0]] += ce - cs + 1
            cs, ce = s_arr[i], e_arr[i]
    if cur_key is not None:
        out[cur_key[0]] += ce - cs + 1
    return dict(out)


# ------------------------------------------------------------------ FASTA IO
def load_gsa_sequences(contigs_dir, wanted=None):
    """anonymous_contig_id -> sequence (str). `wanted` restricts to a set of ids."""
    p = contigs_dir / 'anonymous_gsa.fasta.gz'
    if not p.exists():
        p = contigs_dir / 'anonymous_gsa.fasta'
    op = gzip.open if str(p).endswith('.gz') else open
    seqs, cur, buf = {}, None, []
    with op(p, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if cur is not None and (wanted is None or cur in wanted):
                    seqs[cur] = ''.join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur is not None and (wanted is None or cur in wanted):
        seqs[cur] = ''.join(buf)
    return seqs


def write_fasta(path, records, width=80):
    tmp = path.with_suffix(path.suffix + '.part')
    with open(tmp, 'w') as fh:
        for name, seq in records:
            fh.write(f'>{name}\n')
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + '\n')
    tmp.replace(path)


# ------------------------------------------------------------------ bin design
def greedy_subsample(lengths, order, target_bp):
    """Pick indices in `order` until the cumulative length first reaches target_bp.
    Returns (indices, achieved_bp). Deterministic given `order`."""
    picked, tot = [], 0
    for i in order:
        if tot >= target_bp:
            break
        picked.append(i)
        tot += lengths[i]
    return picked, tot


def build_mixed_design(gold, dist_matrix, rng_seed):
    """Assemble the completeness x contamination x distance grid.

    gold : DataFrame of gold bins (sample, genome, span_bp, ref_len, completeness_pct)
    dist_matrix : {(genome_a, genome_b): lowest-common-ancestor rank}

    A contaminant is only eligible if the GSA of that same sample actually contains
    enough of its sequence to hit the target contamination bp
    (target_cont/100 * ref_len(dominant)). Without this filter the realised
    contamination silently falls far short of target whenever the partner genome is
    barely present in that sample, which would blur the very cells the analysis is
    about. Cells that still cannot be filled are simply left smaller and reported.
    """
    rng = np.random.default_rng(rng_seed)
    by_rank = defaultdict(list)
    for (a, b), r in dist_matrix.items():
        by_rank[r].append((a, b))

    # available GSA bp per (sample, genome): the ceiling on contaminant material
    avail = {(int(s), g): int(bp) for s, g, bp in
             gold[['sample', 'genome', 'span_bp']].itertuples(index=False)}

    cand = gold[gold['completeness_pct'] >= MIN_GOLD_COMP_FOR_DOMINANT].copy()
    rows, used = [], set()
    for comp_t in COMP_TARGETS:
        pool = cand[cand['completeness_pct'] >= comp_t]
        for rank in RANKS:
            partners = by_rank.get(rank, [])
            if not partners:
                continue
            pmap = defaultdict(list)
            for a, b in partners:
                pmap[a].append(b)
            eligible = pool[pool['genome'].isin(pmap.keys())]
            if len(eligible) == 0:
                continue
            for cont_t in CONT_TARGETS:
                sub = eligible.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
                taken = 0
                for _, r in sub.iterrows():
                    if taken >= MAX_PER_CELL:
                        break
                    key = (r['sample'], r['genome'], comp_t, cont_t, rank)
                    if key in used:
                        continue
                    need_bp = cont_t / 100.0 * float(r['ref_len'])
                    opts = [b for b in sorted(pmap[r['genome']])
                            if avail.get((int(r['sample']), b), 0) >= need_bp]
                    if not opts:
                        continue
                    contam = opts[int(rng.integers(len(opts)))]
                    used.add(key)
                    taken += 1
                    rows.append({
                        'sample': r['sample'], 'dominant': r['genome'],
                        'contaminant': contam, 'distance_rank': rank,
                        'target_completeness': comp_t, 'target_contamination': cont_t,
                        'ref_len': r['ref_len'],
                    })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--no-fasta', action='store_true', help='truth tables only')
    ap.add_argument('--max-samples', type=int, default=None)
    args = ap.parse_args()
    ds = args.dataset

    for d in (DERIVED_DIR / ds, TRUTH_DIR, BIN_DIR / ds / 'gold', BIN_DIR / ds / 'mixed'):
        d.mkdir(parents=True, exist_ok=True)

    print(f'== WS3.8 truth build: {ds} ==', flush=True)
    gdir = find_source_genome_dir(ds)
    ref_len_file, ref_src_file = reference_lengths(gdir)
    g2id, novelty = load_genome_to_id(ds)
    # re-key reference lengths on CAMI's genome_id (identity where CAMI uses the basename)
    ref_len, ref_src, fasta_base = {}, {}, {}
    if g2id:
        for gid, base in g2id.items():
            if base in ref_len_file:
                ref_len[gid] = ref_len_file[base]
                ref_src[gid] = ref_src_file.get(base)
                fasta_base[gid] = base
    for base, L in ref_len_file.items():
        ref_len.setdefault(base, L)
        ref_src.setdefault(base, ref_src_file.get(base))
        fasta_base.setdefault(base, base)
    print(f'  source genome FASTAs: {len(ref_len_file)} ({gdir})', flush=True)
    print(f'  genome_to_id entries: {len(g2id)}; novelty labels: {len(novelty)}; '
          f'resolvable ids: {len(ref_len)}', flush=True)

    parent, rankmap, names = load_taxonomy()
    print(f'  taxonomy nodes: {len(parent)}', flush=True)

    samples = sample_dirs(ds)
    if args.max_samples:
        samples = dict(list(samples.items())[: args.max_samples])
    print(f'  samples on disk: {len(samples)} -> {sorted(samples)}', flush=True)
    if not samples:
        raise SystemExit('no samples extracted yet')

    # ---------------- pass 1: per-sample contig truth + gold bins ------------
    gold_rows = []
    genome_taxid = {}
    contig_tables = {}
    for idx, cdir in samples.items():
        df = read_gsa_mapping(cdir)
        contig_tables[idx] = df
        for g, t in df[['genome', 'taxid']].drop_duplicates().itertuples(index=False):
            genome_taxid.setdefault(g, int(t))
        span = df.groupby('genome')['length'].sum()
        ncon = df.groupby('genome')['length'].size()
        uni = union_bp_per_genome(df)
        for g in span.index:
            R = ref_len.get(g)
            if not R:
                continue
            gold_rows.append({
                'dataset': ds, 'sample': idx, 'genome': g, 'taxid': genome_taxid.get(g),
                'ref_len': R, 'span_bp': int(span[g]), 'union_bp': int(uni.get(g, 0)),
                'n_contigs': int(ncon[g]),
                'cami_novelty_category': novelty.get(g, ''),
                'completeness_pct': 100.0 * span[g] / R,
                'completeness_pct_union': 100.0 * uni.get(g, 0) / R,
                'contamination_pct': 0.0,
            })
        print(f'  sample {idx}: {len(df):,} contigs, {df["genome"].nunique()} genomes',
              flush=True)

    gold = pd.DataFrame(gold_rows)
    gold['excluded_non_genome'] = gold['cami_novelty_category'].isin(EXCLUDED_NOVELTY)
    gold['scoreable_by_magicc'] = ((gold['completeness_pct'] >= MAGICC_COMPLETENESS_FLOOR)
                                   & ~gold['excluded_non_genome'])
    n_excl = int(gold['excluded_non_genome'].sum())
    if n_excl:
        print(f'  excluded as non-genome (CAMI label plasmid/virus): {n_excl} bins',
              flush=True)
    gold['bin_id'] = gold['sample'].astype(str).radd('s') + '__' + gold['genome']
    gold.to_csv(TRUTH_DIR / f'{ds}_gold_truth.tsv', sep='\t', index=False)
    print(f'  gold bins: {len(gold)}  scoreable(>=50% comp): '
          f'{int(gold["scoreable_by_magicc"].sum())}  censored: '
          f'{int((~gold["scoreable_by_magicc"]).sum())}', flush=True)

    # ---------------- taxonomy: pairwise distance ---------------------------
    lin, chains = {}, {}
    for g, t in genome_taxid.items():
        l, c = lineage(int(t), parent, rankmap)
        lin[g], chains[g] = l, c
    excluded = {g for g in genome_taxid
                if novelty.get(g, '') in EXCLUDED_NOVELTY}
    genomes = sorted(g for g in genome_taxid if g not in excluded)
    dist = {}
    for i, a in enumerate(genomes):
        for b in genomes[i + 1:]:
            r, lca = lca_rank(lin[a], chains[a], lin[b], chains[b], rankmap)
            if r in RANKS:
                dist[(a, b)] = r
                dist[(b, a)] = r
    rank_counts = pd.Series([v for k, v in dist.items() if k[0] < k[1]]).value_counts()
    print('  pairwise distance ranks (unordered pairs):')
    for r in RANKS:
        print(f'    {r:8s} {int(rank_counts.get(r, 0)):8d}', flush=True)

    sg = pd.DataFrame({
        'genome': genomes,
        'taxid': [genome_taxid[g] for g in genomes],
        'ref_len': [ref_len.get(g) for g in genomes],
        'ref_len_source': [ref_src.get(g) for g in genomes],
        'fasta_basename': [fasta_base.get(g, g) for g in genomes],
        'cami_novelty_category': [novelty.get(g, '') for g in genomes],
        'sci_name': [names.get(genome_taxid[g], '') for g in genomes],
    })
    for r in RANKS:
        sg[f'tax_{r}'] = [names.get(lin[g].get(r), '') for g in genomes]
    sg.to_csv(DERIVED_DIR / ds / 'source_genomes.tsv', sep='\t', index=False)

    # ---------------- mixed bin design --------------------------------------
    design = build_mixed_design(gold[~gold['excluded_non_genome']], dist, DESIGN_SEED)
    print(f'  mixed-bin design rows: {len(design)}', flush=True)

    mixed_rows, per_sample_plan = [], defaultdict(list)
    for _, row in design.iterrows():
        per_sample_plan[row['sample']].append(row)

    for idx, plan in sorted(per_sample_plan.items()):
        df = contig_tables[idx]
        by_genome = {g: sub for g, sub in df.groupby('genome')}
        for row in plan:
            d, x = row['dominant'], row['contaminant']
            R = float(row['ref_len'])
            dsub = by_genome[d]
            xsub = by_genome.get(x)
            if xsub is None or len(xsub) == 0:
                continue
            # Deterministic, process-independent seed (CRC-32 via fw.stable_hash).
            seed_key = (f'ws3.8|{ds}|s{idx}|{d}|{x}|'
                        f'c{row["target_completeness"]}|k{row["target_contamination"]}')
            rs = np.random.default_rng(DESIGN_SEED + fw.stable_hash(seed_key))
            dl = dsub['length'].to_numpy()
            do = rs.permutation(len(dl))
            dpick, dbp = greedy_subsample(dl, do, row['target_completeness'] / 100.0 * R)
            xl = xsub['length'].to_numpy()
            xo = rs.permutation(len(xl))
            xpick, xbp = greedy_subsample(xl, xo, row['target_contamination'] / 100.0 * R)
            comp = 100.0 * dbp / R
            cont = 100.0 * xbp / R
            if cont > comp:               # protocol 4.4a: stay in the training domain
                continue
            bin_id = (f's{idx}__{d}__X__{x}__c{int(row["target_completeness"])}'
                      f'__k{int(row["target_contamination"])}__{row["distance_rank"]}')
            mixed_rows.append({
                'dataset': ds, 'bin_id': bin_id, 'sample': idx,
                'dominant': d, 'contaminant': x,
                'dominant_taxid': genome_taxid.get(d), 'contaminant_taxid': genome_taxid.get(x),
                'distance_rank': row['distance_rank'],
                'target_completeness': row['target_completeness'],
                'target_contamination': row['target_contamination'],
                'ref_len': int(R), 'dominant_bp': int(dbp), 'contaminant_bp': int(xbp),
                'completeness_pct': comp, 'contamination_pct': cont,
                'n_contigs_dominant': len(dpick), 'n_contigs_contaminant': len(xpick),
                'cont_shortfall_rel': (row['target_contamination'] - cont)
                / row['target_contamination'],
                'dominant_contig_idx': ','.join(map(str, sorted(dpick))),
                'contaminant_contig_idx': ','.join(map(str, sorted(xpick))),
                'scoreable_by_magicc': comp >= MAGICC_COMPLETENESS_FLOOR,
            })

    mixed = pd.DataFrame(mixed_rows)
    if len(mixed):
        mixed['in_domain'] = mixed['contamination_pct'] <= mixed['completeness_pct']
    mixed.drop(columns=['dominant_contig_idx', 'contaminant_contig_idx'],
               errors='ignore').to_csv(TRUTH_DIR / f'{ds}_mixed_truth.tsv',
                                       sep='\t', index=False)
    print(f'  mixed bins realised: {len(mixed)}', flush=True)
    if len(mixed):
        print(mixed.groupby('distance_rank').size().to_string(), flush=True)

    # ---------------- write FASTAs ------------------------------------------
    if not args.no_fasta:
        for idx, cdir in samples.items():
            df = contig_tables[idx]
            print(f'  writing FASTAs for sample {idx} ...', flush=True)
            seqs = load_gsa_sequences(cdir)
            gold_dir = BIN_DIR / ds / 'gold'
            keep_gold = set(gold.loc[(gold['sample'] == idx)
                                     & (gold['completeness_pct'] >= BELOW_FLOOR_PROBE_MIN)
                                     & ~gold['excluded_non_genome'], 'genome'])
            for g, sub in df.groupby('genome'):
                if g not in keep_gold:
                    continue
                out = gold_dir / f's{idx}__{g}.fna'
                if out.exists():
                    continue
                recs = [(c, seqs[c]) for c in sub['contig'] if c in seqs]
                if recs:
                    write_fasta(out, recs)
            if len(mixed):
                msub = mixed[mixed['sample'] == idx]
                mix_dir = BIN_DIR / ds / 'mixed'
                by_genome = {g: s for g, s in df.groupby('genome')}
                for _, r in msub.iterrows():
                    out = mix_dir / f'{r["bin_id"]}.fna'
                    if out.exists():
                        continue
                    di = [int(v) for v in r['dominant_contig_idx'].split(',') if v != '']
                    xi = [int(v) for v in r['contaminant_contig_idx'].split(',') if v != '']
                    dc = by_genome[r['dominant']]['contig'].to_numpy()[di]
                    xc = by_genome[r['contaminant']]['contig'].to_numpy()[xi]
                    recs = [(c, seqs[c]) for c in list(dc) + list(xc) if c in seqs]
                    if recs:
                        write_fasta(out, recs)
            del seqs

    # ---------------- provenance --------------------------------------------
    meta = {
        'workstream': 'WS3.8 CAMI II external benchmark (R1-M3)',
        'dataset': ds,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'design_seed': DESIGN_SEED,
        'source_genome_dir': str(gdir.relative_to(PROJECT_DIR)),
        'n_source_genomes': len(ref_len),
        'n_samples': len(samples),
        'samples': sorted(samples),
        'truth_convention': {
            'completeness_pct': '100 * retained_dominant_bp / full_reference_length(dominant)',
            'contamination_pct': '100 * contaminant_bp / full_reference_length(dominant)',
            'denominator': 'FULL reference length of the dominant source genome '
                           '(total bp of all sequences in source_genomes/<genome>.fasta)',
            'bp_measure_primary': 'span_bp = sum(end_position - start_position + 1) from gsa_mapping.tsv',
            'bp_measure_sensitivity': 'union_bp = bp of the union of source intervals',
            'matches': 'magicc/contamination.py::compute_contamination_rate and '
                       'create_contaminated_genome',
        },
        'magicc_completeness_floor_pct': MAGICC_COMPLETENESS_FLOOR,
        'grid': {'completeness_targets': list(COMP_TARGETS),
                 'contamination_targets': list(CONT_TARGETS),
                 'distance_ranks': list(RANKS),
                 'max_dominants_per_cell': MAX_PER_CELL},
        'excluded_non_genome_labels': list(EXCLUDED_NOVELTY),
        'n_gold_bins': int(len(gold)),
        'n_gold_bins_excluded_non_genome': int(gold['excluded_non_genome'].sum()),
        'n_gold_bins_scoreable': int(gold['scoreable_by_magicc'].sum()),
        'n_gold_bins_censored_by_floor':
            int(((~gold['scoreable_by_magicc']) & ~gold['excluded_non_genome']).sum()),
        'n_mixed_bins': int(len(mixed)),
        'pairwise_distance_rank_counts': {r: int(rank_counts.get(r, 0)) for r in RANKS},
        'max_gold_completeness_pct': float(gold['completeness_pct'].max()),
        'n_gold_completeness_over_100': int((gold['completeness_pct'] > 100).sum()),
        'n_gold_union_completeness_over_100': int((gold['completeness_pct_union'] > 100).sum()),
    }
    (TRUTH_DIR / f'{ds}_truth_derivation.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in
                      ('n_gold_bins', 'n_gold_bins_scoreable',
                       'n_gold_bins_censored_by_floor', 'n_mixed_bins',
                       'n_gold_completeness_over_100')}, indent=2), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
