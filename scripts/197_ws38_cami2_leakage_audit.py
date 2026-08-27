#!/usr/bin/env python3
"""
WS3.8 (R1-M3) — leakage / provenance audit of the CAMI II source genomes.

WHY THIS IS NON-NEGOTIABLE
-------------------------
On this project a failure to cross-map GCA<->GCF accessions previously produced a
serious undercount: Sets C and D turned out to be 100 % / 90.3 % leaked into the
training split and had to be withdrawn (protocol 4.1). The same mistake must not be
repeated on an *external* benchmark, where leakage would be far more damaging.

THE COMPLICATION SPECIFIC TO CAMI II
------------------------------------
CAMI II does not name its source genomes by NCBI accession. It uses internal ids:
    strain_madness   spades_ESBL8855_S12_L001, MIKI-NS15, SP100_S82   (clinical isolates
                     sequenced for CAMI; no assembly accession in the distribution)
    marine           Teredinibacter_turnerae_T7901_genomic, Titti28_S8_contigs
                     (organism/strain names, plus newly sequenced isolates)
A naive accession join would therefore report "0 overlap" for a trivial reason and would
be worthless. The audit is run at THREE levels and all three are reported:

  L1  ASSEMBLY level. Every accession-shaped token found in the CAMI genome ids and in
      any CAMI metadata table is normalised and GCA<->GCF cross-mapped through
      data/gtdb/filtered_genomes.tsv (277,183 assemblies), exactly as
      scripts/074_provenance_audit.py does, then intersected with
      data/splits/{train,val,test}_genomes.tsv and the 2,000-genome 9-mer selection set.

  L2  ORGANISM/STRAIN level. CAMI genome ids are matched against NCBI's
      assembly_summary_{genbank,refseq}.txt organism_name + infraspecific_name to
      recover the GCA/GCF accession CAMI most likely used, and THOSE accessions are then
      cross-mapped as in L1. This catches leakage that L1 cannot see because CAMI
      stripped the accession from the file name.

  L3  SPECIES level. The NCBI tax_id CAMI itself assigns to each source genome is
      compared with the species of every genome in the training split. This is NOT
      leakage in the assembly sense, but it is the honest description of how novel the
      benchmark is to the model, and it must be stated: if a species is heavily
      represented in training, a failure on that species cannot be blamed on novelty.

The leakage-free subset (no L1 and no L2 hit in train or val or the 9-mer selection set)
is the PRIMARY analysis cohort. Everything is emitted as accession lists plus a SHA256
manifest, in the style of scripts/074_provenance_audit.py.

OUTPUTS
-------
  results/revision/cami2/provenance/<ds>_source_genome_audit.tsv
  results/revision/cami2/provenance/<ds>_leakage_summary.json
  results/revision/cami2/provenance/<ds>_accessions_{matched,leaked,clean}.txt
  results/revision/cami2/provenance/<ds>_species_vs_training.tsv
  results/revision/cami2/provenance/sha256_manifest.txt
  results/revision/cami2/provenance/README.md

Usage:
    python scripts/197_ws38_cami2_leakage_audit.py --dataset strain_madness
    python scripts/197_ws38_cami2_leakage_audit.py --dataset marine
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
KMER_DIR = DATA_DIR / 'kmer_selection'
CAMI_DIR = DATA_DIR / 'real_data' / 'cami2'
DERIVED_DIR = CAMI_DIR / 'derived'
RES_DIR = PROJECT_DIR / 'results' / 'revision' / 'cami2'
PROV_DIR = RES_DIR / 'provenance'

_VER = re.compile(r'\.\d+$')
_ACC = re.compile(r'(GC[AF])[_-]?(\d{9})(?:\.(\d+))?', re.IGNORECASE)
_NUM = re.compile(r'(\d{6,})')


# ------------------------------------------------------- accession handling
def strict_norm(acc):
    """GB_GCA_001822065.1 -> GCA_001822065 (keeps the GCA/GCF distinction)."""
    if not isinstance(acc, str) or not acc:
        return None
    a = acc.strip()
    for p in ('GB_', 'RS_'):
        if a.startswith(p):
            a = a[len(p):]
    return _VER.sub('', a)


def assembly_number(acc):
    """GB_GCA_001822065.1 -> '001822065' (shared by a GCA/GCF assembly pair)."""
    s = strict_norm(acc)
    if not s:
        return None
    m = _NUM.search(s)
    return m.group(1) if m else None


def build_crossmap():
    """canonical key per assembly, declared by GTDB's own accession triple."""
    f = pd.read_csv(DATA_DIR / 'gtdb' / 'filtered_genomes.tsv', sep='\t',
                    usecols=['gtdb_accession', 'ncbi_accession', 'gcf_accession'],
                    dtype=str)
    cmap = {}
    for gtdb, gca, gcf in f.itertuples(index=False):
        key = assembly_number(gca) or assembly_number(gcf) or assembly_number(gtdb)
        if not key:
            continue
        for a in (gtdb, gca, gcf):
            s = strict_norm(a)
            if s:
                cmap[s] = key
    return cmap


def canon(acc, cmap):
    s = strict_norm(acc)
    if not s:
        return None
    return cmap.get(s) or assembly_number(s)


def load_universe(cmap):
    """train / val / test / 9-mer-selection accession universes, canonicalised."""
    uni = {}
    for split in ('train', 'val', 'test'):
        df = pd.read_csv(SPLITS_DIR / f'{split}_genomes.tsv', sep='\t', dtype=str)
        accs = []
        for col in ('gtdb_accession', 'ncbi_accession', 'gcf_accession', 'accession'):
            if col in df.columns:
                accs += df[col].dropna().tolist()
        uni[split] = {canon(a, cmap) for a in accs} - {None}
    kacc = []
    for f in ('selected_bacterial_1000.tsv', 'selected_archaeal_1000.tsv'):
        p = KMER_DIR / f
        if p.exists():
            kacc += pd.read_csv(p, sep='\t', dtype=str)['gtdb_accession'].dropna().tolist()
    uni['kmer_selection'] = {canon(a, cmap) for a in kacc} - {None}
    return uni


def training_species(cmap):
    """species name -> number of training-split genomes, from the split tables."""
    out = {}
    for split in ('train', 'val'):
        df = pd.read_csv(SPLITS_DIR / f'{split}_genomes.tsv', sep='\t', dtype=str)
        col = 'gtdb_taxonomy' if 'gtdb_taxonomy' in df.columns else None
        if col is None:
            continue
        sp = df[col].dropna().str.extract(r's__([^;]*)')[0].str.strip()
        for name, n in sp.value_counts().items():
            if name:
                out.setdefault(name, {'train': 0, 'val': 0})[split] = int(n)
    return out


# ------------------------------------------------------- NCBI name matching
def _norm_name(s):
    s = re.sub(r'[^A-Za-z0-9]+', ' ', str(s)).lower().strip()
    return re.sub(r'\s+', ' ', s)


def load_ncbi_summaries():
    """organism/strain keys -> set of accessions, from NCBI assembly summaries."""
    idx = defaultdict(set)
    tax2acc = defaultdict(set)
    for fn in ('assembly_summary_genbank.txt', 'assembly_summary_refseq.txt'):
        p = DATA_DIR / 'ncbi' / fn
        if not p.exists():
            continue
        df = pd.read_csv(p, sep='\t', skiprows=1, dtype=str, low_memory=False,
                         usecols=lambda c: c in ('#assembly_accession', 'taxid',
                                                 'species_taxid', 'organism_name',
                                                 'infraspecific_name', 'asm_name'))
        df = df.rename(columns={'#assembly_accession': 'accession'})
        for r in df.itertuples(index=False):
            acc = getattr(r, 'accession', None)
            if not acc:
                continue
            org = getattr(r, 'organism_name', '') or ''
            strain = (getattr(r, 'infraspecific_name', '') or '').replace('strain=', '')
            asm = getattr(r, 'asm_name', '') or ''
            for key in {_norm_name(org), _norm_name(f'{org} {strain}'),
                        _norm_name(asm)}:
                if key and len(key) > 3:
                    idx[key].add(acc)
            t = getattr(r, 'taxid', None)
            if t and str(t).isdigit():
                tax2acc[int(t)].add(acc)
    return idx, tax2acc


def cami_name_keys(genome_id):
    """CAMI genome id -> candidate organism/strain keys for the NCBI join."""
    g = str(genome_id)
    for suf in ('_genomic', '_contigs', '_scaffolds', '_genome'):
        if g.endswith(suf):
            g = g[: -len(suf)]
    base = _norm_name(g)
    keys = {base}
    parts = base.split()
    if len(parts) >= 3:
        keys.add(' '.join(parts[:3]))
    if len(parts) >= 2:
        keys.add(' '.join(parts[:2]))
    return {k for k in keys if len(k) > 5}


# ------------------------------------------------------------------ hashing
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--no-manifest', action='store_true')
    args = ap.parse_args()
    ds = args.dataset
    PROV_DIR.mkdir(parents=True, exist_ok=True)

    sg_path = DERIVED_DIR / ds / 'source_genomes.tsv'
    if not sg_path.exists():
        raise SystemExit(f'run 196 first: {sg_path} missing')
    sg = pd.read_csv(sg_path, sep='\t', dtype={'genome': str})
    # marine labels contigs with OTU ids (Otu255); the informative string for both the
    # accession scan and the NCBI name join is the source-genome FASTA basename
    # (Teredinibacter_turnerae_T7901_genomic), so use it wherever CAMI supplies one.
    if 'fasta_basename' in sg.columns:
        sg['name_key_source'] = sg['fasta_basename'].fillna(sg['genome'])
    else:
        sg['name_key_source'] = sg['genome']
    print(f'== WS3.8 leakage audit: {ds} ({len(sg)} source genomes) ==', flush=True)

    cmap = build_crossmap()
    uni = load_universe(cmap)
    print(f'  crossmap entries {len(cmap):,}; universes: '
          + ', '.join(f'{k}={len(v):,}' for k, v in uni.items()), flush=True)

    nkey = dict(zip(sg['genome'], sg['name_key_source'].astype(str)))

    # ---------------- L1: accession tokens in the CAMI ids ------------------
    l1_acc = {}
    for g in sg['genome']:
        m = _ACC.search(nkey[g])
        if m:
            l1_acc[g] = f'{m.group(1).upper()}_{m.group(2)}'
    print(f'  L1 accession-shaped ids: {len(l1_acc)} / {len(sg)}', flush=True)

    # ---------------- L2: organism/strain -> NCBI accession -----------------
    # NOTE this join is deliberately OVER-inclusive: a CAMI strain name may match
    # several assemblies of the same organism, and every one of them is treated as a
    # potential leak. That can only OVER-state leakage, never hide it, which is the
    # correct direction of error for an audit.
    name_idx, tax2acc = load_ncbi_summaries()
    print(f'  NCBI summary keys: {len(name_idx):,}', flush=True)
    l2_acc, l2_key = {}, {}
    for g in sg['genome']:
        if g in l1_acc:
            continue
        hits = set()
        used = None
        for k in sorted(cami_name_keys(nkey[g]), key=len, reverse=True):
            if k in name_idx:
                hits = name_idx[k]
                used = k
                break
        if hits:
            l2_acc[g] = sorted(hits)
            l2_key[g] = used
    print(f'  L2 name-matched genomes: {len(l2_acc)} / {len(sg)}', flush=True)

    # ---------------- overlap computation -----------------------------------
    rows = []
    for r in sg.itertuples(index=False):
        g = r.genome
        accs = []
        level = 'none'
        if g in l1_acc:
            accs = [l1_acc[g]]
            level = 'L1_accession_in_id'
        elif g in l2_acc:
            accs = l2_acc[g]
            level = 'L2_ncbi_name_match'
        canons = {canon(a, cmap) for a in accs} - {None}
        hit = {k: bool(canons & v) for k, v in uni.items()}
        rows.append({
            'dataset': ds, 'genome': g, 'name_key_source': nkey[g], 'taxid': r.taxid,
            'cami_novelty_category': getattr(r, 'cami_novelty_category', ''),
            'sci_name': getattr(r, 'sci_name', ''),
            'tax_species': getattr(r, 'tax_species', ''),
            'tax_genus': getattr(r, 'tax_genus', ''),
            'tax_phylum': getattr(r, 'tax_phylum', ''),
            'match_level': level,
            'matched_accessions': ';'.join(sorted(accs)[:8]),
            'n_matched_accessions': len(accs),
            'ncbi_name_key': l2_key.get(g, ''),
            'in_train': hit['train'], 'in_val': hit['val'], 'in_test': hit['test'],
            'in_kmer_selection': hit['kmer_selection'],
        })
    audit = pd.DataFrame(rows)
    audit['leaked'] = audit['in_train'] | audit['in_val'] | audit['in_kmer_selection']
    audit.to_csv(PROV_DIR / f'{ds}_source_genome_audit.tsv', sep='\t', index=False)

    n_leak = int(audit['leaked'].sum())
    print(f'  LEAKED (train|val|9-mer-selection): {n_leak} / {len(audit)}', flush=True)
    print(f'  in_test only: {int((audit["in_test"] & ~audit["leaked"]).sum())}', flush=True)

    # ---------------- L3: species-level presence in training ----------------
    tsp = training_species(cmap)
    sp_rows = []
    for sp, sub in audit.groupby('tax_species'):
        if not sp:
            continue
        t = tsp.get(sp, {'train': 0, 'val': 0})
        sp_rows.append({'dataset': ds, 'species': sp, 'n_cami_genomes': len(sub),
                        'n_train_genomes_same_species': t.get('train', 0),
                        'n_val_genomes_same_species': t.get('val', 0),
                        'species_seen_in_training': t.get('train', 0) > 0})
    spdf = pd.DataFrame(sp_rows).sort_values('n_cami_genomes', ascending=False)
    spdf.to_csv(PROV_DIR / f'{ds}_species_vs_training.tsv', sep='\t', index=False)
    n_sp_seen = int(spdf['species_seen_in_training'].sum()) if len(spdf) else 0
    n_gen_sp_seen = int(spdf.loc[spdf['species_seen_in_training'],
                                 'n_cami_genomes'].sum()) if len(spdf) else 0
    print(f'  L3 species also present in TRAIN: {n_sp_seen}/{len(spdf)} species, '
          f'covering {n_gen_sp_seen}/{len(audit)} CAMI genomes', flush=True)

    # ---------------- accession lists ---------------------------------------
    (PROV_DIR / f'{ds}_accessions_matched.txt').write_text(
        '\n'.join(sorted({a for v in list(l1_acc.values()) for a in [v]}
                         | {a for vs in l2_acc.values() for a in vs})) + '\n')
    (PROV_DIR / f'{ds}_accessions_leaked.txt').write_text(
        '\n'.join(sorted({a for row in audit[audit['leaked']].itertuples(index=False)
                          for a in str(row.matched_accessions).split(';') if a})) + '\n')
    (PROV_DIR / f'{ds}_genomes_leakage_free.txt').write_text(
        '\n'.join(sorted(audit.loc[~audit['leaked'], 'genome'])) + '\n')
    (PROV_DIR / f'{ds}_genomes_leaked.txt').write_text(
        '\n'.join(sorted(audit.loc[audit['leaked'], 'genome'])) + '\n')

    summary = {
        'workstream': 'WS3.8 CAMI II leakage audit (R1-M3)',
        'dataset': ds,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'n_source_genomes': int(len(audit)),
        'L1_accession_in_cami_id': int(len(l1_acc)),
        'L2_ncbi_name_matched': int(len(l2_acc)),
        'n_unmatched_no_accession_recoverable':
            int((audit['match_level'] == 'none').sum()),
        'overlap_train': int(audit['in_train'].sum()),
        'overlap_val': int(audit['in_val'].sum()),
        'overlap_test': int(audit['in_test'].sum()),
        'overlap_kmer_selection': int(audit['in_kmer_selection'].sum()),
        'n_leaked_train_val_or_kmer': n_leak,
        'n_leakage_free': int((~audit['leaked']).sum()),
        'leakage_pct': round(100.0 * n_leak / max(1, len(audit)), 3),
        'L3_species_level': {
            'n_distinct_species': int(len(spdf)),
            'n_species_present_in_train': n_sp_seen,
            'n_cami_genomes_whose_species_is_in_train': n_gen_sp_seen,
            'pct_cami_genomes_species_in_train':
                round(100.0 * n_gen_sp_seen / max(1, len(audit)), 2),
        },
        'crossmap_source': 'data/gtdb/filtered_genomes.tsv (GCA<->GCF, 277,183 assemblies)',
        'universes': {k: len(v) for k, v in uni.items()},
    }
    (PROV_DIR / f'{ds}_leakage_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    # ---------------- SHA256 manifest ---------------------------------------
    if not args.no_manifest:
        lines = []
        for p in sorted((CAMI_DIR / 'raw').rglob('*')):
            if p.is_file():
                lines.append(f'{sha256_file(p)}  {p.relative_to(PROJECT_DIR)}')
        for p in sorted(PROV_DIR.glob(f'{ds}_*')):
            lines.append(f'{sha256_file(p)}  {p.relative_to(PROJECT_DIR)}')
        man = PROV_DIR / 'sha256_manifest.txt'
        prev = man.read_text().splitlines() if man.exists() else []
        merged = sorted(set(prev) | set(lines))
        man.write_text('\n'.join(merged) + '\n')
        print(f'  SHA256 manifest: {len(merged)} entries -> {man}', flush=True)

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
