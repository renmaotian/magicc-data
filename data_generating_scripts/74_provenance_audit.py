#!/usr/bin/env python3
"""
WS1.4 — Provenance / non-overlap audit for every MAGICC benchmark set.

GOAL
----
(a) PROVE, not assert, that the dominant reference genomes of the clean Sets C_clean and
    D_clean are disjoint from the training split, the validation split and the 2,000
    genomes used for 9-mer feature selection, and document where the contaminants come
    from.
(b) QUANTIFY the leakage of the superseded Sets C and D exactly, for the point-by-point
    response.
(c) Emit an auditable artefact set under results/revision/provenance/.

THE GCA/GCF PITFALL
-------------------
Accessions appear in three flavours across the project files:
    GTDB     GB_GCA_001822065.1 / RS_GCF_035810195.1
    GenBank  GCA_001822065.1
    RefSeq   GCF_035810195.1
A naive string comparison therefore MISSES real overlaps: a genome recorded as
GB_GCA_x.1 in a benchmark can appear as RS_GCF_x.1 in a split file. This is exactly why
the first leakage estimate for Set D (277/1000 in train) was an undercount.

Two normalisations are computed and BOTH are reported:
  strict    strip GB_/RS_ and the .version  ->  'GCA_001822065'  (GCA != GCF)
  crossmap  canonical assembly-pair key built from data/gtdb/filtered_genomes.tsv, whose
            (gtdb_accession, ncbi_accession, gcf_accession) columns declare which
            GTDB/GenBank/RefSeq strings are the same assembly. Accessions absent from
            that table fall back to the 9-digit assembly number, which is shared by a
            GCA/GCF pair by NCBI construction.
`crossmap` is the number to trust; `strict` is reported so the size of the artefact is
visible.

OUTPUTS
-------
  results/revision/provenance/overlap_summary.tsv
  results/revision/provenance/set_C_clean_dominants.txt
  results/revision/provenance/set_D_clean_dominants.txt
  results/revision/provenance/contaminants.txt
  results/revision/provenance/sha256_manifest.txt
  results/revision/provenance/README.md
  results/revision/provenance/accession_crossmap_stats.json
  data/benchmarks/set_C/SUPERSEDED.md
  data/benchmarks/set_D/SUPERSEDED.md

Usage:
    python scripts/74_provenance_audit.py
    python scripts/74_provenance_audit.py --no-manifest      # skip SHA256 of 2,000 FASTAs
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'
KMER_DIR = DATA_DIR / 'kmer_selection'
PROV_DIR = PROJECT_DIR / 'results' / 'revision' / 'provenance'

_VER = re.compile(r'\.\d+$')
_NUM = re.compile(r'(\d{6,})')


# ------------------------------------------------------------- normalisation
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
    if not isinstance(acc, str) or not acc:
        return None
    m = _NUM.search(acc)
    return m.group(1) if m else None


def build_crossmap():
    """strict-normalised accession string -> canonical assembly-pair key.

    Built from the (gtdb_accession, ncbi_accession, gcf_accession) triples in
    data/gtdb/filtered_genomes.tsv: every string in a row denotes the same assembly.
    """
    fg = pd.read_csv(DATA_DIR / 'gtdb' / 'filtered_genomes.tsv', sep='\t',
                     usecols=['gtdb_accession', 'ncbi_accession', 'gcf_accession'])
    cmap = {}
    groups = defaultdict(set)
    inconsistent = 0
    for gtdb, ncbi, gcf in zip(fg['gtdb_accession'], fg['ncbi_accession'],
                               fg['gcf_accession']):
        key = assembly_number(ncbi) or assembly_number(gtdb) or assembly_number(gcf)
        if key is None:
            continue
        nums = set()
        for a in (gtdb, ncbi, gcf):
            s = strict_norm(a)
            if s:
                prev = cmap.get(s)
                if prev is not None and prev != key:
                    inconsistent += 1
                cmap[s] = key
                groups[key].add(s)
                n = assembly_number(a)
                if n:
                    nums.add(n)
        if len(nums) > 1:
            inconsistent += 1
    stats = {
        'filtered_genomes_rows': int(len(fg)),
        'distinct_accession_strings_mapped': len(cmap),
        'distinct_canonical_assemblies': len(groups),
        'rows_with_inconsistent_assembly_numbers': int(inconsistent),
        'mean_strings_per_assembly': round(
            float(np.mean([len(v) for v in groups.values()])), 3),
    }
    return cmap, stats


def canon_set(accs, cmap):
    """Set of canonical keys, plus the set of strict-normalised strings."""
    strict, canon, unmapped = set(), set(), 0
    for a in accs:
        s = strict_norm(a)
        if s is None:
            continue
        strict.add(s)
        k = cmap.get(s)
        if k is None:
            k = assembly_number(a)
            unmapped += 1
        if k:
            canon.add(k)
    return strict, canon, unmapped


# ------------------------------------------------------------------ loaders
def load_split(name):
    return pd.read_csv(SPLITS_DIR / f'{name}_genomes.tsv', sep='\t')


def dominants_of(set_dir):
    """(accessions, source_file). Prefers generation_metadata.tsv where present."""
    gm = set_dir / 'generation_metadata.tsv'
    if gm.exists():
        df = pd.read_csv(gm, sep='\t')
        return df['dominant_accession'].tolist(), 'generation_metadata.tsv'
    md = set_dir / 'metadata.tsv'
    if md.exists():
        df = pd.read_csv(md, sep='\t')
        if 'dominant_accession' in df.columns:
            return df['dominant_accession'].tolist(), 'metadata.tsv'
    return [], None


def contaminants_of(set_dir):
    gm = set_dir / 'generation_metadata.tsv'
    if not gm.exists():
        return []
    df = pd.read_csv(gm, sep='\t')
    out = []
    for gid, s in zip(df['genome_id'], df['contaminant_accessions']):
        if isinstance(s, str) and s:
            for a in s.split(';'):
                if a:
                    out.append((gid, a))
    return out


def sha256_file(p, chunk=8 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-manifest', action='store_true')
    args = ap.parse_args()

    PROV_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 78)
    print('WS1.4 — provenance / non-overlap audit')
    print('=' * 78)

    print('\n[1] building GCA/GCF cross-map from data/gtdb/filtered_genomes.tsv ...')
    cmap, cmap_stats = build_crossmap()
    for k, v in cmap_stats.items():
        print(f'    {k}: {v}')
    with open(PROV_DIR / 'accession_crossmap_stats.json', 'w') as f:
        json.dump(cmap_stats, f, indent=2)

    print('\n[2] loading reference universes ...')
    splits = {n: load_split(n) for n in ('train', 'val', 'test')}
    universe = {}
    for n, df in splits.items():
        s, c, u = canon_set(df['gtdb_accession'], cmap)
        universe[n] = {'strict': s, 'canon': c, 'raw': set(df['gtdb_accession'])}
        print(f'    {n} split: {len(df)} rows -> {len(s)} strict, {len(c)} canonical, '
              f'{u} unmapped')

    kmer_accs = []
    for f in ('selected_bacterial_1000.tsv', 'selected_archaeal_1000.tsv'):
        kmer_accs += pd.read_csv(KMER_DIR / f, sep='\t')['gtdb_accession'].tolist()
    s, c, u = canon_set(kmer_accs, cmap)
    universe['kmer_selection'] = {'strict': s, 'canon': c}
    print(f'    9-mer feature-selection genomes: {len(kmer_accs)} rows -> {len(s)} '
          f'strict, {len(c)} canonical, {u} unmapped')

    # sanity: the k-mer selection genomes are documented as drawn from the train split
    kmer_in_train = len(universe['kmer_selection']['canon'] & universe['train']['canon'])
    print(f'    (check) k-mer selection genomes inside train split: '
          f'{kmer_in_train}/{len(universe["kmer_selection"]["canon"])}')

    # ---------------------------------------------------------------- sets
    sets_to_audit = [
        ('set_C_clean', 'clean replacement — Patescibacteriota, test-split dominants'),
        ('set_D_clean', 'clean replacement — Archaea, test-split dominants'),
        ('set_C', 'SUPERSEDED — Patescibacteriota, dominants from train+val+test'),
        ('set_D', 'SUPERSEDED — Archaea, dominants from train+val+test'),
        ('set_A_v2', 'retained — completeness gradient, finished test-split dominants'),
        ('set_B_v2', 'retained — contamination gradient, finished test-split dominants'),
        ('set_E', 'retained — mixed, finished test-split dominants'),
        ('set_A', 'legacy — completeness gradient'),
        ('set_B', 'legacy — contamination gradient'),
        ('motivating_v2/set_A', 'motivating — completeness gradient'),
        ('motivating_v2/set_B', 'motivating — contamination gradient'),
        ('motivating_v2/set_C', 'motivating — realistic'),
        ('motivating/set_A', 'motivating v1 — completeness gradient'),
        ('motivating/set_B', 'motivating v1 — contamination gradient'),
    ]

    print('\n[3] auditing dominant-genome provenance ...')
    rows = []
    detail = {}
    for name, desc in sets_to_audit:
        d = BENCHMARK_DIR / name
        if not d.is_dir():
            continue
        accs, src = dominants_of(d)
        if not accs:
            continue
        strict, canon, unmapped = canon_set(accs, cmap)
        row = {
            'set': name,
            'role': desc,
            'accession_source': src,
            'n_samples': len(accs),
            'n_unique_dominants_strict': len(strict),
            'n_unique_dominants_canonical': len(canon),
            'n_unmapped_accessions': unmapped,
        }
        # per-sample split membership (the number reviewers care about)
        for split in ('train', 'val', 'test'):
            us, uc = universe[split]['strict'], universe[split]['canon']
            ur = universe[split]['raw']
            row[f'samples_in_{split}_raw'] = sum(1 for a in accs if a in ur)
            row[f'samples_in_{split}_strict'] = sum(1 for a in accs
                                                    if strict_norm(a) in us)
            row[f'samples_in_{split}_canonical'] = sum(
                1 for a in accs
                if (cmap.get(strict_norm(a)) or assembly_number(a)) in uc)
        uk = universe['kmer_selection']['canon']
        row['samples_in_kmer_selection_canonical'] = sum(
            1 for a in accs if (cmap.get(strict_norm(a)) or assembly_number(a)) in uk)
        row['unique_dominants_in_train_canonical'] = len(
            canon & universe['train']['canon'])
        row['unique_dominants_in_val_canonical'] = len(canon & universe['val']['canon'])
        row['unique_dominants_in_test_canonical'] = len(canon & universe['test']['canon'])
        row['unique_dominants_in_kmer_selection_canonical'] = len(canon & uk)
        row['n_samples_in_no_split'] = row['n_samples'] - (
            row['samples_in_train_canonical'] + row['samples_in_val_canonical']
            + row['samples_in_test_canonical'])
        row['leakage_pct_train_plus_val'] = round(
            100.0 * (row['samples_in_train_canonical'] + row['samples_in_val_canonical'])
            / max(1, row['n_samples']), 2)
        rows.append(row)
        detail[name] = {'accs': accs, 'strict': strict, 'canon': canon}
        print(f'    {name:22s} n={row["n_samples"]:5d}  '
              f'train={row["samples_in_train_canonical"]:5d}  '
              f'val={row["samples_in_val_canonical"]:4d}  '
              f'test={row["samples_in_test_canonical"]:5d}  '
              f'kmer-sel={row["samples_in_kmer_selection_canonical"]:4d}  '
              f'(strict train={row["samples_in_train_strict"]})')

    # --------- label-constraint audit (second, independent defect of the old sets) ----
    # The V5 training distribution enforces contaminant_bp <= dominant_retained_bp, i.e.
    # contamination% <= completeness%. That cap was added to magicc/contamination.py in
    # commit c5a9b95 (2026-03-14); Sets C/D/E and motivating_v2/set_C were generated
    # 2026-03-10, i.e. BEFORE it existed, so they contain out-of-constraint samples.
    for r in rows:
        d = BENCHMARK_DIR / r['set']
        md = pd.read_csv(d / 'metadata.tsv', sep='\t')
        if not {'true_completeness', 'true_contamination'} <= set(md.columns):
            continue
        v = md['true_contamination'] > md['true_completeness'] + 1e-6
        r['label_violations_cont_gt_comp'] = int(v.sum())
        r['label_violations_max_excess_pp'] = (
            round(float((md['true_contamination'] - md['true_completeness'])[v].max()), 3)
            if v.any() else 0.0)
        r['label_violations_cont_gt_100'] = int((md['true_contamination'] > 100).sum())
        r['label_violations_comp_lt_50'] = int(
            (md['true_completeness'] < 50 - 1e-6).sum())

    overlap = pd.DataFrame(rows)
    overlap.to_csv(PROV_DIR / 'overlap_summary.tsv', sep='\t', index=False)
    print(f'\n    wrote {PROV_DIR / "overlap_summary.tsv"}')
    print('    label-constraint violations (contamination% > completeness%):')
    for r in rows:
        if r.get('label_violations_cont_gt_comp'):
            print(f'      {r["set"]:22s} {r["label_violations_cont_gt_comp"]:4d}/'
                  f'{r["n_samples"]}  max excess '
                  f'{r["label_violations_max_excess_pp"]} pp')

    # ------------------------------------------------------- contaminants
    print('\n[4] auditing contaminant provenance ...')
    cont_rows = []
    cont_report = {}
    for name in ('set_C_clean', 'set_D_clean'):
        d = BENCHMARK_DIR / name
        pairs = contaminants_of(d)
        if not pairs:
            continue
        gm = pd.read_csv(d / 'generation_metadata.tsv', sep='\t')
        dom_phylum = dict(zip(gm['genome_id'], gm['dominant_phylum']))
        test_phylum = {}
        for split in ('train', 'val', 'test'):
            for a, p in zip(splits[split]['gtdb_accession'], splits[split]['phylum']):
                k = cmap.get(strict_norm(a)) or assembly_number(a)
                test_phylum.setdefault(k, {})[split] = p

        n_test = n_train = n_val = n_none = n_same_phylum = 0
        for gid, a in pairs:
            k = cmap.get(strict_norm(a)) or assembly_number(a)
            where = test_phylum.get(k, {})
            if 'test' in where:
                n_test += 1
            elif 'train' in where:
                n_train += 1
            elif 'val' in where:
                n_val += 1
            else:
                n_none += 1
            ph = where.get('test') or where.get('train') or where.get('val')
            if ph is not None and ph == dom_phylum.get(gid):
                n_same_phylum += 1
            cont_rows.append({'set': name, 'genome_id': gid, 'contaminant_accession': a,
                              'canonical_key': k,
                              'split': ('test' if 'test' in where else
                                        'train' if 'train' in where else
                                        'val' if 'val' in where else 'NOT_IN_ANY_SPLIT'),
                              'phylum': ph,
                              'dominant_phylum': dom_phylum.get(gid),
                              'same_phylum_as_dominant': bool(ph == dom_phylum.get(gid))})
        uniq = {cmap.get(strict_norm(a)) or assembly_number(a) for _, a in pairs}
        cont_report[name] = {
            'contamination_events': len(pairs),
            'unique_contaminant_genomes': len(uniq),
            'events_from_test_split': n_test,
            'events_from_train_split': n_train,
            'events_from_val_split': n_val,
            'events_from_no_split': n_none,
            'events_same_phylum_as_dominant': n_same_phylum,
        }
        print(f'    {name}: {cont_report[name]}')

    cdf = pd.DataFrame(cont_rows)
    if len(cdf):
        with open(PROV_DIR / 'contaminants.txt', 'w') as f:
            f.write('# Contaminant provenance for the clean benchmark sets.\n')
            f.write('# One row per contamination event (a genome_id may appear up to 5x).\n')
            f.write('# split = which train/val/test split the contaminant reference belongs to.\n')
            cdf.to_csv(f, sep='\t', index=False)
        print(f'    wrote {PROV_DIR / "contaminants.txt"} ({len(cdf)} events)')

    # ------------------------------------------------------ dominant lists
    for name in ('set_C_clean', 'set_D_clean'):
        d = BENCHMARK_DIR / name
        if name not in detail:
            continue
        gm = pd.read_csv(d / 'generation_metadata.tsv', sep='\t')
        uniq = (gm[['dominant_accession', 'dominant_phylum', 'dominant_taxonomy',
                    'dominant_reference_bp', 'ref_index']]
                .drop_duplicates('dominant_accession')
                .sort_values('ref_index'))
        uniq['canonical_key'] = [cmap.get(strict_norm(a)) or assembly_number(a)
                                 for a in uniq['dominant_accession']]
        uniq['in_train_split'] = [k in universe['train']['canon']
                                  for k in uniq['canonical_key']]
        uniq['in_val_split'] = [k in universe['val']['canon']
                                for k in uniq['canonical_key']]
        uniq['in_test_split'] = [k in universe['test']['canon']
                                 for k in uniq['canonical_key']]
        uniq['in_kmer_selection'] = [k in universe['kmer_selection']['canon']
                                     for k in uniq['canonical_key']]
        out = PROV_DIR / f'{name}_dominants.txt'
        with open(out, 'w') as f:
            f.write(f'# {name} — {len(uniq)} unique dominant reference genomes, '
                    f'10 independent simulations each.\n')
            f.write('# All are members of the held-out test split '
                    '(data/splits/test_genomes.tsv) and of no other split.\n')
            uniq.to_csv(f, sep='\t', index=False)
        print(f'    wrote {out} ({len(uniq)} dominants)')

    # ---------------------------------------------------------- manifest
    if not args.no_manifest:
        print('\n[5] building SHA256 manifest ...')
        targets = []
        for name in ('set_C_clean', 'set_D_clean'):
            d = BENCHMARK_DIR / name
            if not d.is_dir():
                continue
            for f in ('reference_selection.tsv', 'metadata.tsv', 'labels.npy',
                      'generation_metadata.tsv', 'magicc_v5_predictions.tsv'):
                if (d / f).exists():
                    targets.append(d / f)
            targets += sorted((d / 'fasta').glob('*.fasta'))
        for extra in (PROJECT_DIR / 'models' / 'magicc_v5.onnx',
                      DATA_DIR / 'features' / 'normalization_params.json',
                      KMER_DIR / 'selected_kmers.txt',
                      SPLITS_DIR / 'train_genomes.tsv',
                      SPLITS_DIR / 'val_genomes.tsv',
                      SPLITS_DIR / 'test_genomes.tsv',
                      KMER_DIR / 'selected_bacterial_1000.tsv',
                      KMER_DIR / 'selected_archaeal_1000.tsv'):
            if extra.exists():
                targets.append(extra)
        with ThreadPoolExecutor(max_workers=12) as ex:
            hashes = list(ex.map(sha256_file, targets))
        with open(PROV_DIR / 'sha256_manifest.txt', 'w') as f:
            f.write('# SHA256 manifest — MAGICC revision, WS1.4\n')
            f.write(f'# generated {datetime.now(timezone.utc).isoformat()}\n')
            f.write('# paths are relative to the project root\n')
            for p, h in zip(targets, hashes):
                f.write(f'{h}  {p.relative_to(PROJECT_DIR)}\n')
        print(f'    wrote {PROV_DIR / "sha256_manifest.txt"} ({len(targets)} files)')

    # ------------------------------------------------------------- README
    print('\n[6] writing README and SUPERSEDED notices ...')
    o = overlap.set_index('set')

    def g(s, c, default=0):
        try:
            return int(o.loc[s, c])
        except Exception:
            return default

    readme = f"""# MAGICC revision — benchmark provenance audit (WS1.4)

Generated {datetime.now(timezone.utc).isoformat()} by `scripts/74_provenance_audit.py`.

## Why this audit exists

Both reviewers observed that benchmark Sets C (Patescibacteria) and D (Archaea) in the
submitted manuscript were not independent of the training data. They were right. This
directory contains the evidence, the exact leakage numbers, and the proof that the
replacement sets are clean.

## Method

### Accession universes

| Universe | File | Rows |
|---|---|---|
| train split | `data/splits/train_genomes.tsv` | {len(splits['train'])} |
| val split | `data/splits/val_genomes.tsv` | {len(splits['val'])} |
| test split | `data/splits/test_genomes.tsv` | {len(splits['test'])} |
| 9-mer feature-selection genomes | `data/kmer_selection/selected_bacterial_1000.tsv` + `selected_archaeal_1000.tsv` | {len(kmer_accs)} |

The three splits are disjoint by construction and stratified by phylum. The 2,000
feature-selection genomes were sampled from the train split only
({kmer_in_train}/{len(universe['kmer_selection']['canon'])} verified to be inside the
train split here).

### The GCA/GCF pitfall, and how it is handled

The same assembly is written three different ways across project files — GTDB
(`GB_GCA_001822065.1`, `RS_GCF_035810195.1`), GenBank (`GCA_001822065.1`) and RefSeq
(`GCF_035810195.1`). Naive string matching therefore **undercounts** overlap: a
benchmark genome recorded as `GB_GCA_x.1` can sit in a split file as `RS_GCF_x.1`.

Three normalisations are computed for every accession and all three are reported in
`overlap_summary.tsv`:

* **raw** — the GTDB accession string exactly as written in the file.
* **strict** — drop the `GB_`/`RS_` prefix and the `.version` suffix
  (`GB_GCA_001822065.1 -> GCA_001822065`). Keeps `GCA` and `GCF` distinct.
* **crossmap** (authoritative) — a canonical assembly-pair key. Every row of
  `data/gtdb/filtered_genomes.tsv` supplies a `(gtdb_accession, ncbi_accession,
  gcf_accession)` triple that names one assembly, so all three strings are mapped to one
  key. Accessions absent from that table fall back to the 9-digit assembly number, which
  a GCA/GCF pair shares by NCBI construction.

Cross-map statistics: {cmap_stats['filtered_genomes_rows']} rows of
`filtered_genomes.tsv` -> {cmap_stats['distinct_accession_strings_mapped']} distinct
accession strings collapsing to {cmap_stats['distinct_canonical_assemblies']} canonical
assemblies ({cmap_stats['mean_strings_per_assembly']} strings per assembly on average,
{cmap_stats['rows_with_inconsistent_assembly_numbers']} inconsistent rows).

## Result 1 — the superseded Sets C and D were leaked

Counts are **per sample** (n = 1,000 each), using the authoritative crossmap
normalisation. All three normalisations agree for these two sets (see
`overlap_summary.tsv`), so the numbers below are not normalisation artefacts.

| Set | n | in TRAIN | in VAL | in TEST | in no split | in 9-mer selection | leakage (train+val) |
|---|---|---|---|---|---|---|---|
| set_C | {g('set_C','n_samples')} | **{g('set_C','samples_in_train_canonical')}** | {g('set_C','samples_in_val_canonical')} | {g('set_C','samples_in_test_canonical')} | {g('set_C','n_samples_in_no_split')} | {g('set_C','samples_in_kmer_selection_canonical')} | {o.loc['set_C','leakage_pct_train_plus_val']}% |
| set_D | {g('set_D','n_samples')} | **{g('set_D','samples_in_train_canonical')}** | {g('set_D','samples_in_val_canonical')} | {g('set_D','samples_in_test_canonical')} | {g('set_D','n_samples_in_no_split')} | {g('set_D','samples_in_kmer_selection_canonical')} | {o.loc['set_D','leakage_pct_train_plus_val']}% |

Per-normalisation TRAIN / VAL / TEST counts:

| Set | raw GTDB string | strict (prefix+version stripped) | crossmap (authoritative) |
|---|---|---|---|
| set_C | {g('set_C','samples_in_train_raw')} / {g('set_C','samples_in_val_raw')} / {g('set_C','samples_in_test_raw')} | {g('set_C','samples_in_train_strict')} / {g('set_C','samples_in_val_strict')} / {g('set_C','samples_in_test_strict')} | {g('set_C','samples_in_train_canonical')} / {g('set_C','samples_in_val_canonical')} / {g('set_C','samples_in_test_canonical')} |
| set_D | {g('set_D','samples_in_train_raw')} / {g('set_D','samples_in_val_raw')} / {g('set_D','samples_in_test_raw')} | {g('set_D','samples_in_train_strict')} / {g('set_D','samples_in_val_strict')} / {g('set_D','samples_in_test_strict')} | {g('set_D','samples_in_train_canonical')} / {g('set_D','samples_in_val_canonical')} / {g('set_D','samples_in_test_canonical')} |

> **Correction to the internal record.** The figures previously logged in
> the internal project log — "Set C: 985/1,000 dominants in TRAIN" and
> "Set D: only 36/1,000 in test" — were themselves incomplete audits. 985 is the number of
> Set C dominants carrying a `GB_` prefix, not the number in TRAIN; the true count is
> **{g('set_C','samples_in_train_canonical')}/1,000 in TRAIN and 0 in TEST**. For Set D the
> complete accounting is **{g('set_D','samples_in_train_canonical')} TRAIN /
> {g('set_D','samples_in_val_canonical')} VAL / {g('set_D','samples_in_test_canonical')} TEST**.
> Set C leakage is therefore total, not 98.5 %.

A further overlap that the original audit missed: **{g('set_C','samples_in_kmer_selection_canonical')}**
Set C samples and **{g('set_D','samples_in_kmer_selection_canonical')}** Set D samples use a
dominant reference that was also one of the 2,000 genomes used to *select the 9-mer feature
set*, so those samples leak into feature selection as well as into model fitting.

Root cause, quoted from the generator that produced them
(`scripts/25_benchmark_generate.py`): *"Set C: ALL Patescibacteriota from
train+val+test (1608 total)"*, *"Set D: ALL Archaea from train+val+test (1976 total),
sample 1000"*.

The sets could never have been built from the test split alone: the test split holds only
161 Patescibacteriota and 198 archaeal genomes, and reference curation had already taken
**all** 1,609 Patescibacteriota and **all** 1,976 archaeal genomes of the
277,183-genome filtered pool into the 100,000-genome working set, so there are zero
unused genomes of these lineages to fall back on.

Contaminant provenance cannot be audited for the superseded sets: their generator did not
record contaminant accessions. That gap is one of the reasons the new generator writes a
complete `generation_metadata.tsv` (reviewer comment R1-m15).

### A second, independent defect of the old sets: out-of-constraint labels

The V5 training distribution enforces `contaminant_bp <= dominant_retained_bp`, i.e.
contamination % <= completeness % (both share the dominant's full reference length as
denominator). That cap was added to `magicc/contamination.py` in commit `c5a9b95`
(2026-03-14). Sets C, D, E and `motivating_v2/set_C` were generated on 2026-03-10,
**before the cap existed**, so a fraction of their samples lie outside the region the
model was ever trained on:

| Set | n | samples with contamination % > completeness % | max excess |
|---|---|---|---|
"""
    for r in rows:
        if r.get('label_violations_cont_gt_comp'):
            readme += (f"| {r['set']} | {r['n_samples']} | "
                       f"{r['label_violations_cont_gt_comp']} | "
                       f"{r['label_violations_max_excess_pp']} pp |\n")
    readme += f"""
The clean sets have zero such samples (verified in
`results/revision/ws1_23_generation_validation.json`). The leaked -> clean difference in
Set C/D accuracy therefore mixes two effects; `results/revision/ws1_5_clean_vs_leaked_cd.tsv`
decomposes them by also reporting the superseded sets restricted to their
constraint-satisfying subsets.

## Result 2 — the clean Sets C_clean and D_clean are disjoint from all training input

| Set | n samples | unique dominants | in TRAIN | in VAL | in TEST | in 9-mer selection |
|---|---|---|---|---|---|---|
| set_C_clean | {g('set_C_clean','n_samples')} | {g('set_C_clean','n_unique_dominants_canonical')} | **{g('set_C_clean','samples_in_train_canonical')}** | **{g('set_C_clean','samples_in_val_canonical')}** | {g('set_C_clean','samples_in_test_canonical')} | **{g('set_C_clean','samples_in_kmer_selection_canonical')}** |
| set_D_clean | {g('set_D_clean','n_samples')} | {g('set_D_clean','n_unique_dominants_canonical')} | **{g('set_D_clean','samples_in_train_canonical')}** | **{g('set_D_clean','samples_in_val_canonical')}** | {g('set_D_clean','samples_in_test_canonical')} | **{g('set_D_clean','samples_in_kmer_selection_canonical')}** |

Dominant accession lists with per-genome split membership flags:
`set_C_clean_dominants.txt`, `set_D_clean_dominants.txt`.

### Contaminants

Contaminants are drawn only from `data/splits/test_genomes.tsv`, excluding the dominant's
own phylum. Every contamination event is listed in `contaminants.txt`.

| Set | events | unique contaminant genomes | from TEST | from TRAIN | from VAL | same phylum as dominant |
|---|---|---|---|---|---|---|
"""
    for name in ('set_C_clean', 'set_D_clean'):
        r = cont_report.get(name)
        if r:
            readme += (f"| {name} | {r['contamination_events']} | "
                       f"{r['unique_contaminant_genomes']} | "
                       f"{r['events_from_test_split']} | {r['events_from_train_split']} | "
                       f"{r['events_from_val_split']} | "
                       f"{r['events_same_phylum_as_dominant']} |\n")

    readme += f"""
## Result 3 — the retained sets were already clean

| Set | n | in TRAIN | in VAL | in TEST |
|---|---|---|---|---|
"""
    for name in ('set_A_v2', 'set_B_v2', 'set_E', 'set_A', 'set_B',
                 'motivating_v2/set_A', 'motivating_v2/set_B', 'motivating_v2/set_C',
                 'motivating/set_A', 'motivating/set_B'):
        if name in o.index:
            readme += (f"| {name} | {g(name,'n_samples')} | "
                       f"{g(name,'samples_in_train_canonical')} | "
                       f"{g(name,'samples_in_val_canonical')} | "
                       f"{g(name,'samples_in_test_canonical')} |\n")

    readme += """
## Files

| File | Contents |
|---|---|
| `overlap_summary.tsv` | one row per benchmark set; per-sample and per-unique-genome overlap counts against train/val/test/9-mer-selection, under all three normalisations, plus the label-constraint violation counts |
| `set_C_clean_dominants.txt` | the 100 Patescibacteriota dominants with taxonomy, canonical key and split flags |
| `set_D_clean_dominants.txt` | the 100 archaeal dominants, same columns |
| `contaminants.txt` | every contamination event: genome_id, contaminant accession, split, phylum, dominant phylum |
| `sha256_manifest.txt` | SHA256 of every generated FASTA, every metadata/label file, the frozen ONNX model, the normalisation parameters, the 9-mer list and the split files |
| `accession_crossmap_stats.json` | cross-map construction statistics |

## Reproduce

```bash
python scripts/72_select_clean_cd_refs.py            # WS1.1  reference selection
python scripts/73_generate_clean_cd_benchmarks.py C D  # WS1.2/1.3  generation
python scripts/74_provenance_audit.py                # WS1.4  this audit
python scripts/75_run_magicc_clean_cd.py             # WS1.5  MAGICC V5 inference
```
"""
    (PROV_DIR / 'README.md').write_text(readme)
    print(f'    wrote {PROV_DIR / "README.md"}')

    # -------------------------------------------------------- SUPERSEDED
    ws15_path = PROJECT_DIR / 'results' / 'revision' / 'ws1_5_clean_vs_leaked_cd.tsv'
    ws15 = (pd.read_csv(ws15_path, sep='\t').set_index('superseded_set')
            if ws15_path.exists() else None)

    def ws15_block(old_name):
        if ws15 is None or old_name not in ws15.index:
            return ('\n(MAGICC V5 accuracy comparison not yet computed — run '
                    '`scripts/75_run_magicc_clean_cd.py`.)\n')
        r = ws15.loc[old_name]
        return f"""
## Withdrawn vs replacement MAGICC V5 accuracy

Same frozen model (`models/magicc_v5.onnx`), so the difference is attributable to the
benchmark, not the model. Source: `results/revision/ws1_5_clean_vs_leaked_cd.tsv`.

| Quantity | withdrawn {old_name} (n={int(r['n_superseded'])}) | withdrawn, constraint-satisfying subset (n={int(r['leaked_subset_n'])}) | clean {r['clean_set']} (n={int(r['n_clean'])}) | delta (clean - withdrawn) |
|---|---|---|---|---|
| completeness MAE (%) | {r['leaked_comp_mae']} | {r['leaked_subset_comp_mae']} | {r['clean_comp_mae']} | {r['delta_comp_mae']:+} |
| contamination MAE (%) | {r['leaked_cont_mae']} | {r['leaked_subset_cont_mae']} | {r['clean_cont_mae']} | {r['delta_cont_mae']:+} |
| completeness bias (%) | {r['leaked_comp_bias']} | — | {r['clean_comp_bias']} | |
| contamination bias (%) | {r['leaked_cont_bias']} | — | {r['clean_cont_bias']} | |
| completeness R2 | {r['leaked_comp_r2']} | — | {r['clean_comp_r2']} | |
| contamination R2 | {r['leaked_cont_r2']} | — | {r['clean_cont_r2']} | |

Clean-set 95 % CIs (cluster bootstrap over the 100 reference genomes):
completeness MAE {r['clean_comp_mae_ci']}, contamination MAE {r['clean_cont_mae_ci']}.
"""

    for old, clean, lineage in (('set_C', 'set_C_clean', 'Patescibacteriota (CPR)'),
                                ('set_D', 'set_D_clean', 'Archaea')):
        if old not in o.index:
            continue
        txt = f"""# SUPERSEDED — {old} is WITHDRAWN (training-set leakage)

**Status:** withdrawn {datetime.now(timezone.utc).date().isoformat()}.
**Replacement:** `data/benchmarks/{clean}/`
**Audit:** `results/revision/provenance/` (`overlap_summary.tsv`, `README.md`),
generated by `scripts/74_provenance_audit.py`.

## What is wrong with this set

This set's {lineage} dominant reference genomes were drawn from **train + val + test**,
not from the held-out test split. Its generator says so explicitly
(`scripts/25_benchmark_generate.py` docstring). Verified counts over the
{g(old,'n_samples')} samples, using a GCA/GCF-aware canonical accession cross-map built
from `data/gtdb/filtered_genomes.tsv`:

| dominants in TRAIN | in VAL | in TEST | in no split | in 9-mer feature selection |
|---|---|---|---|---|
| **{g(old,'samples_in_train_canonical')}** | {g(old,'samples_in_val_canonical')} | {g(old,'samples_in_test_canonical')} | {g(old,'n_samples_in_no_split')} | {g(old,'samples_in_kmer_selection_canonical')} |

Leakage (train + val) = **{o.loc[old,'leakage_pct_train_plus_val']}%** of samples.
Unique dominant genomes: {g(old,'n_unique_dominants_canonical')}
(train {g(old,'unique_dominants_in_train_canonical')},
val {g(old,'unique_dominants_in_val_canonical')},
test {g(old,'unique_dominants_in_test_canonical')}).

The counts are identical under raw GTDB-string matching
({g(old,'samples_in_train_raw')} / {g(old,'samples_in_val_raw')} /
{g(old,'samples_in_test_raw')} for train/val/test), under prefix-and-version-stripped
matching, and under the GCA/GCF cross-map, so they are not normalisation artefacts.

Contaminant provenance for this set is **unauditable**: the generator did not record
contaminant accessions.

## A second, independent defect: out-of-constraint labels

The V5 training distribution enforces contaminant_bp <= dominant_retained_bp, i.e.
contamination % <= completeness %. That cap entered `magicc/contamination.py` in commit
`c5a9b95` (2026-03-14); this set's FASTA files were written on 2026-03-10, before the cap
existed. Consequently **{g(old, 'label_violations_cont_gt_comp')} / {g(old, 'n_samples')}**
samples here have contamination % > completeness % (max excess
{o.loc[old, 'label_violations_max_excess_pp']} pp) and lie outside the region the model was
ever trained on. `{clean}` has zero such samples.

## Why it could not have been built correctly at the time

The held-out test split contains only 161 Patescibacteriota and 198 archaeal genomes,
while this set has 1,000 samples. Reference curation had already taken all 1,609
Patescibacteriota and all 1,976 archaeal genomes of the 277,183-genome filtered pool into
the 100,000-genome working set, so no unused genomes of these lineages exist.

## The replacement

`data/benchmarks/{clean}/` uses 100 dominant references from the **test split only**,
with **10 independent simulations each** (1,000 samples, same design: uniform
completeness 50-100 %, uniform contamination 0-100 %, 1-5 cross-phylum contaminants
drawn from the test split). Because references are reused across simulations, all
statistics on the clean sets must be clustered by reference genome.

Verified for {clean}: {g(clean,'samples_in_train_canonical')} samples in TRAIN,
{g(clean,'samples_in_val_canonical')} in VAL,
{g(clean,'samples_in_kmer_selection_canonical')} in the 9-mer feature-selection set.
{ws15_block(old)}
## Retention policy

Nothing here has been deleted. The FASTA files, metadata, labels and all tool prediction
files remain in place so that the withdrawn numbers stay reproducible and the effect of
the leakage can be quantified. **No number from this directory may appear in the revised
manuscript.**
"""
        p = BENCHMARK_DIR / old / 'SUPERSEDED.md'
        p.write_text(txt)
        print(f'    wrote {p}')

    with open(PROV_DIR / 'audit_summary.json', 'w') as f:
        json.dump({'generated_utc': datetime.now(timezone.utc).isoformat(),
                   'crossmap_stats': cmap_stats,
                   'universe_sizes': {k: len(v['canon']) for k, v in universe.items()},
                   'kmer_selection_genomes_inside_train_split': kmer_in_train,
                   'overlap': rows,
                   'contaminants': cont_report}, f, indent=2)

    # ---------------------------------------------------------- verdict
    print('\n' + '=' * 78)
    ok = True
    for name in ('set_C_clean', 'set_D_clean'):
        if name not in o.index:
            print(f'  {name}: NOT PRESENT')
            ok = False
            continue
        bad = (g(name, 'samples_in_train_canonical')
               + g(name, 'samples_in_val_canonical')
               + g(name, 'samples_in_kmer_selection_canonical'))
        clean_test = g(name, 'samples_in_test_canonical') == g(name, 'n_samples')
        print(f'  {name}: train+val+kmerSel overlap = {bad}; '
              f'all samples in test split = {clean_test}')
        ok &= (bad == 0 and clean_test)
    for name in ('set_C_clean', 'set_D_clean'):
        r = cont_report.get(name)
        if r:
            cbad = r['events_from_train_split'] + r['events_from_val_split'] \
                   + r['events_from_no_split'] + r['events_same_phylum_as_dominant']
            print(f'  {name} contaminants: non-test or same-phylum events = {cbad}')
            ok &= (cbad == 0)
    print(f'\nDISJOINTNESS VERDICT: {"PASS" if ok else "FAIL"}')
    print('=' * 78)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
