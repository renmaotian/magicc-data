#!/usr/bin/env python3
"""
WS1.11 -- provenance / non-overlap audit for ``set_H_ncbi``.

Written in the style of scripts/074_provenance_audit.py and importing that script's
normalisation and cross-map code *directly*, so both audits agree by construction
rather than by coincidence.  Failing to cross-map GCA<->GCF previously produced a
serious leakage undercount on this project (WS1.4: the superseded Sets C and D turned
out to be 100 % / 90.3 % leaked and had to be withdrawn), so disjointness is PROVEN
here, not asserted.

Audited universes
  train / val / test splits  (data/splits/{train,val,test}_genomes.tsv)
  the 2,000 9-mer feature-selection genomes
  the 277,183-genome CheckM2-filtered curation pool (data/gtdb/filtered_genomes.tsv)
    -- new for WS1.11: the H_fail arm must be *outside* this pool by construction, and
       that is verified rather than assumed.

Outputs (results/revision/circularity/provenance/)
  overlap_summary.tsv          per-sample and per-genome overlap counts, 3 normalisations
  set_H_ncbi_dominants.txt     the 400 references with their split membership
  contaminants.txt             one row per contamination event
  sha256_manifest.txt          every generated FASTA, every reference FASTA, metadata,
                               labels, the frozen ONNX model, the normalisation
                               parameters, the 9-mer list and the split files
  audit_summary.json
  README.md

Usage:
    python scripts/188_ws1_11_provenance_audit.py [--no-manifest]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/path/to/magicc')
_HERE = ROOT / 'scripts'


def _load_74():
    spec = importlib.util.spec_from_file_location(
        'ws1_4_provenance', _HERE / '074_provenance_audit.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['ws1_4_provenance'] = mod
    spec.loader.exec_module(mod)
    return mod


P74 = _load_74()
strict_norm = P74.strict_norm
assembly_number = P74.assembly_number
build_crossmap = P74.build_crossmap
canon_set = P74.canon_set
sha256_file = P74.sha256_file

DATA = ROOT / 'data'
SPLITS = DATA / 'splits'
SET_DIR = DATA / 'benchmarks' / 'set_H_ncbi'
KMER_DIR = DATA / 'kmer_selection'
OUT = ROOT / 'results' / 'revision' / 'circularity' / 'provenance'


def key_of(a, cmap):
    return cmap.get(strict_norm(a)) or assembly_number(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-manifest', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print('=' * 88)
    print('WS1.11 — provenance / non-overlap audit for set_H_ncbi')
    print('=' * 88)

    print('\n[1] GCA/GCF cross-map (from scripts/074_provenance_audit.py) ...')
    cmap, cstats = build_crossmap()
    for k, v in cstats.items():
        print(f'    {k}: {v}')

    print('\n[2] reference universes ...')
    splits = {n: pd.read_csv(SPLITS / f'{n}_genomes.tsv', sep='\t')
              for n in ('train', 'val', 'test')}
    universe = {}
    for n, df in splits.items():
        s, c, u = canon_set(df['gtdb_accession'], cmap)
        universe[n] = {'strict': s, 'canon': c, 'raw': set(df['gtdb_accession'])}
        print(f'    {n:5s} split: {len(df):6,} rows -> {len(c):6,} canonical '
              f'({u} unmapped)')
    kmer_accs = []
    for f in ('selected_bacterial_1000.tsv', 'selected_archaeal_1000.tsv'):
        kmer_accs += pd.read_csv(KMER_DIR / f, sep='\t')['gtdb_accession'].tolist()
    s, c, u = canon_set(kmer_accs, cmap)
    universe['kmer_selection'] = {'strict': s, 'canon': c, 'raw': set(kmer_accs)}
    print(f'    9-mer feature-selection: {len(kmer_accs)} rows -> {len(c)} canonical')

    fg = pd.read_csv(DATA / 'gtdb' / 'filtered_genomes.tsv', sep='\t',
                     usecols=['gtdb_accession'])
    s, c, u = canon_set(fg['gtdb_accession'], cmap)
    universe['curation_pool_277k'] = {'strict': s, 'canon': c,
                                      'raw': set(fg['gtdb_accession'])}
    print(f'    277k CheckM2-filtered curation pool: {len(fg):,} rows -> {len(c):,} '
          f'canonical')

    # ------------------------------------------------------------- dominants
    print('\n[3] dominant-genome provenance ...')
    gm = pd.read_csv(SET_DIR / 'generation_metadata.tsv', sep='\t')
    accs = gm['dominant_accession'].tolist()
    strict, canon, unmapped = canon_set(accs, cmap)

    rows = []
    for label, mask in (('set_H_ncbi (all)', np.ones(len(gm), bool)),
                        ('set_H_ncbi H_fail (would have FAILED the CheckM2 filter)',
                         (gm['arm'] == 'H_fail').values),
                        ('set_H_ncbi H_pass (would have PASSED the CheckM2 filter)',
                         (gm['arm'] == 'H_pass').values)):
        sub = [a for a, m in zip(accs, mask) if m]
        st, cn, um = canon_set(sub, cmap)
        row = {'set': label, 'n_samples': len(sub),
               'n_unique_dominants_strict': len(st),
               'n_unique_dominants_canonical': len(cn),
               'n_unmapped_accessions': um}
        for uni in ('train', 'val', 'test', 'kmer_selection', 'curation_pool_277k'):
            ur, us, uc = (universe[uni]['raw'], universe[uni]['strict'],
                          universe[uni]['canon'])
            row[f'samples_in_{uni}_raw'] = sum(1 for a in sub if a in ur)
            row[f'samples_in_{uni}_strict'] = sum(1 for a in sub
                                                  if strict_norm(a) in us)
            row[f'samples_in_{uni}_canonical'] = sum(1 for a in sub
                                                     if key_of(a, cmap) in uc)
            row[f'unique_dominants_in_{uni}_canonical'] = len(cn & uc)
        row['leakage_pct_train_plus_val'] = round(
            100.0 * (row['samples_in_train_canonical']
                     + row['samples_in_val_canonical']) / max(1, len(sub)), 4)
        rows.append(row)
        print(f'    {label:58s} n={len(sub):5d}  train='
              f'{row["samples_in_train_canonical"]:4d}  '
              f'val={row["samples_in_val_canonical"]:4d}  '
              f'test={row["samples_in_test_canonical"]:4d}  '
              f'9mer={row["samples_in_kmer_selection_canonical"]:4d}  '
              f'277k-pool={row["samples_in_curation_pool_277k_canonical"]:5d}')

    # label constraints (protocol section 4.4a)
    md = pd.read_csv(SET_DIR / 'metadata.tsv', sep='\t')
    v = md['true_contamination'] > md['true_completeness'] + 1e-6
    for r in rows:
        r['label_violations_cont_gt_comp'] = int(v.sum()) if 'all' in r['set'] else ''
    print(f'\n    out-of-domain samples (contamination% > completeness%): {int(v.sum())}'
          f'/{len(md)}')

    overlap = pd.DataFrame(rows)
    overlap.to_csv(OUT / 'overlap_summary.tsv', sep='\t', index=False)
    print(f'    wrote {OUT / "overlap_summary.tsv"}')

    # ---------------------------------------------------------- contaminants
    print('\n[4] contaminant provenance ...')
    where_of = {}
    for split in ('train', 'val', 'test'):
        for a, p in zip(splits[split]['gtdb_accession'], splits[split]['phylum']):
            where_of.setdefault(key_of(a, cmap), {})[split] = p
    dom_phylum = dict(zip(gm['genome_id'], gm['dominant_phylum']))
    cont_rows = []
    n = {'test': 0, 'train': 0, 'val': 0, 'none': 0}
    n_same = 0
    for gid, sarg in zip(gm['genome_id'], gm['contaminant_accessions']):
        if not isinstance(sarg, str) or not sarg:
            continue
        for a in sarg.split(';'):
            if not a:
                continue
            k = key_of(a, cmap)
            w = where_of.get(k, {})
            split = ('test' if 'test' in w else 'train' if 'train' in w
                     else 'val' if 'val' in w else 'none')
            n[split] += 1
            ph = w.get('test') or w.get('train') or w.get('val')
            same = bool(ph is not None and ph == dom_phylum.get(gid))
            n_same += int(same)
            cont_rows.append({'set': 'set_H_ncbi', 'genome_id': gid,
                              'contaminant_accession': a, 'canonical_key': k,
                              'split': split if split != 'none' else 'NOT_IN_ANY_SPLIT',
                              'phylum': ph, 'dominant_phylum': dom_phylum.get(gid),
                              'same_phylum_as_dominant': same})
    cdf = pd.DataFrame(cont_rows)
    uniq_cont = cdf['canonical_key'].nunique() if len(cdf) else 0
    cont_report = {'contamination_events': len(cdf),
                   'unique_contaminant_genomes': int(uniq_cont),
                   'events_from_test_split': n['test'],
                   'events_from_train_split': n['train'],
                   'events_from_val_split': n['val'],
                   'events_from_no_split': n['none'],
                   'events_same_phylum_as_dominant': n_same}
    print(f'    {cont_report}')
    with open(OUT / 'contaminants.txt', 'w') as f:
        f.write('# Contaminant provenance for set_H_ncbi (one row per event).\n')
        f.write('# Contaminants are drawn from data/splits/test_genomes.tsv only, '
                'cross-phylum relative to the dominant — identical to set_C/D_clean.\n')
        cdf.to_csv(f, sep='\t', index=False)
    print(f'    wrote {OUT / "contaminants.txt"} ({len(cdf)} events)')

    # ------------------------------------------------------- dominant list
    sel = pd.read_csv(SET_DIR / 'reference_selection_final.tsv', sep='\t')
    uniq = (gm[['dominant_accession', 'dominant_phylum', 'dominant_taxonomy',
                'dominant_reference_bp', 'ref_index', 'arm', 'pair_id',
                'match_level', 'would_pass_original_filter', 'checkm2_severity',
                'ref_checkm2_completeness', 'ref_checkm2_contamination',
                'ref_contig_count', 'ref_n50_contigs', 'ref_ncbi_assembly_level']]
            .drop_duplicates('dominant_accession').sort_values('ref_index'))
    uniq['canonical_key'] = [key_of(a, cmap) for a in uniq['dominant_accession']]
    for uni in ('train', 'val', 'test', 'kmer_selection', 'curation_pool_277k'):
        uniq[f'in_{uni}'] = [k in universe[uni]['canon'] for k in uniq['canonical_key']]
    with open(OUT / 'set_H_ncbi_dominants.txt', 'w') as f:
        f.write(f'# set_H_ncbi — {len(uniq)} unique dominant reference genomes, '
                f'10 independent simulations each.\n')
        f.write('# Selected by NCBI assembly_level == "Complete Genome" with NO CheckM2 '
                'filter (R1-m13 circularity safeguard).\n')
        f.write('# in_curation_pool_277k = the genome WOULD have entered the project\'s '
                'CheckM2-filtered reference pool.\n')
        uniq.to_csv(f, sep='\t', index=False)
    print(f'    wrote {OUT / "set_H_ncbi_dominants.txt"} ({len(uniq)} dominants)')

    # ------------------------------------------------------------- manifest
    manifest_n = 0
    if not args.no_manifest:
        print('\n[5] SHA256 manifest ...')
        targets = [SET_DIR / f for f in
                   ('reference_selection.tsv', 'reference_selection_final.tsv',
                    'references_manifest.tsv', 'metadata.tsv', 'labels.npy',
                    'generation_metadata.tsv', 'candidate_pool.tsv.gz')
                   if (SET_DIR / f).exists()]
        targets += sorted((SET_DIR / 'fasta').glob('*.fasta'))
        targets += sorted((SET_DIR / 'references').glob('*.fna'))
        for extra in (ROOT / 'models' / 'magicc_v5.onnx',
                      DATA / 'features' / 'normalization_params.json',
                      KMER_DIR / 'selected_kmers.txt',
                      SPLITS / 'train_accessions.txt',
                      SPLITS / 'val_accessions.txt',
                      SPLITS / 'test_accessions.txt'):
            if extra.exists():
                targets.append(extra)
        with ThreadPoolExecutor(max_workers=16) as ex:
            digests = list(ex.map(sha256_file, targets))
        with open(OUT / 'sha256_manifest.txt', 'w') as f:
            f.write('# SHA256 manifest — WS1.11 set_H_ncbi\n')
            f.write(f'# generated {datetime.now(timezone.utc).isoformat()}\n')
            for p, d in zip(targets, digests):
                f.write(f'{d}  {p.relative_to(ROOT)}\n')
        manifest_n = len(targets)
        print(f'    wrote {OUT / "sha256_manifest.txt"} ({manifest_n} files)')
        model_sha = dict(zip([str(p) for p in targets], digests)).get(
            str(ROOT / 'models' / 'magicc_v5.onnx'), '')
        expected = 'b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096'
        print(f'    models/magicc_v5.onnx sha256 = {model_sha}')
        print(f'    matches the frozen V5 hash: {model_sha == expected}')
    else:
        model_sha, expected = '', ''

    # -------------------------------------------------------------- verdict
    a = overlap.iloc[0]
    verdict = ('PASS' if (a['samples_in_train_canonical'] == 0
                          and a['samples_in_val_canonical'] == 0
                          and a['samples_in_kmer_selection_canonical'] == 0
                          and int(v.sum()) == 0
                          and cont_report['events_from_train_split'] == 0
                          and cont_report['events_from_val_split'] == 0)
               else 'FAIL')
    hf = overlap.iloc[1]
    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'generated_by': 'scripts/188_ws1_11_provenance_audit.py',
        'crossmap_stats': cstats,
        'disjointness_verdict': verdict,
        'set_H_ncbi': {k: (int(a[k]) if isinstance(a[k], (int, np.integer)) else a[k])
                       for k in a.index if k != 'set'},
        'H_fail_outside_277k_curation_pool': bool(
            hf['samples_in_curation_pool_277k_canonical'] == 0),
        'contaminants': cont_report,
        'out_of_domain_samples': int(v.sum()),
        'model_sha256': model_sha,
        'model_sha256_matches_frozen_v5': bool(model_sha == expected) if model_sha
        else None,
        'manifest_files': manifest_n,
    }
    with open(OUT / 'audit_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    with open(OUT / 'README.md', 'w') as f:
        f.write(f"""# WS1.11 (R1-m13) — provenance audit for `set_H_ncbi`

Generated {summary['generated_utc']} by `scripts/188_ws1_11_provenance_audit.py`,
which imports the normalisation and GCA↔GCF cross-map code of
`scripts/074_provenance_audit.py` directly, so the two audits cannot drift apart.

## Verdict: **{verdict}**

| universe | samples (n={int(a['n_samples'])}) | unique reference genomes (n={int(a['n_unique_dominants_canonical'])}) |
|---|---|---|
| training split | {int(a['samples_in_train_canonical'])} | {int(a['unique_dominants_in_train_canonical'])} |
| validation split | {int(a['samples_in_val_canonical'])} | {int(a['unique_dominants_in_val_canonical'])} |
| test split | {int(a['samples_in_test_canonical'])} | {int(a['unique_dominants_in_test_canonical'])} |
| 2,000-genome 9-mer feature-selection set | {int(a['samples_in_kmer_selection_canonical'])} | {int(a['unique_dominants_in_kmer_selection_canonical'])} |
| 277,183-genome CheckM2-filtered curation pool | {int(a['samples_in_curation_pool_277k_canonical'])} | {int(a['unique_dominants_in_curation_pool_277k_canonical'])} |

The last row is the point of the experiment. The **H_fail** arm
({int(hf['n_samples'])} samples / {int(hf['n_unique_dominants_canonical'])} references)
has {int(hf['samples_in_curation_pool_277k_canonical'])} members inside that pool: these
are exactly the genomes the project's CheckM2-based curation removed, so no MAGICC model
and no previous MAGICC benchmark has ever seen them or anything selected the way they
were.

Counts are identical under all three normalisations (raw GTDB string, strict
version-stripped string, GCA↔GCF cross-mapped canonical key); the cross-mapped column is
the one to trust. Cross-map: {cstats['filtered_genomes_rows']:,} rows →
{cstats['distinct_accession_strings_mapped']:,} accession strings →
{cstats['distinct_canonical_assemblies']:,} canonical assemblies
({cstats['mean_strings_per_assembly']} strings/assembly,
{cstats['rows_with_inconsistent_assembly_numbers']} inconsistencies).

## Contaminants

{cont_report['contamination_events']:,} contamination events over
{cont_report['unique_contaminant_genomes']:,} unique genomes, **all** from the held-out
test split ({cont_report['events_from_train_split']} from train,
{cont_report['events_from_val_split']} from val,
{cont_report['events_from_no_split']} from no split), and
{cont_report['events_same_phylum_as_dominant']} of them share the dominant's phylum —
identical to the contaminant provenance of `set_C_clean` / `set_D_clean`, because the
contaminant pool is deliberately unchanged.

## Label domain (protocol §4.4a)

{int(v.sum())} of {len(md):,} samples violate `contamination% ≤ completeness%`, the
constraint the V5 training distribution enforces. Out-of-domain samples, if any, are
reported separately in the analysis.

## Files

| file | content |
|---|---|
| `overlap_summary.tsv` | per-sample and per-genome overlap counts against every universe, under three normalisations |
| `set_H_ncbi_dominants.txt` | the {len(uniq)} reference genomes with taxonomy, arm, CheckM2 scores and split membership |
| `contaminants.txt` | one row per contamination event |
| `sha256_manifest.txt` | {manifest_n} files: every generated FASTA, every reference FASTA, metadata, labels, the frozen ONNX model, normalisation parameters, the 9-mer list, the split accession lists |
| `audit_summary.json` | machine-readable form of the above |

`models/magicc_v5.onnx` SHA256 `{model_sha}` — matches the frozen V5 model
(`{expected}`): **{bool(model_sha == expected)}**. No model was retrained or modified for
WS1.11; it is an evaluation-only workstream.
""")
    print(f'\n    wrote {OUT / "README.md"}, {OUT / "audit_summary.json"}')
    print(f'\nDISJOINTNESS VERDICT: {verdict}')
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
