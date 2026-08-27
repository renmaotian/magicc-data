#!/usr/bin/env python3
"""
WS1.11 -- download and verify the NCBI-Complete reference genomes chosen by
scripts/185_ws1_11_select_ncbi_refs.py.

Genomes that would have FAILED the project's CheckM2-based curation filter were never
downloaded during Phase 1 (that is the whole point of R1-m13), so both arms are fetched
fresh from the NCBI FTP path recorded in assembly_summary_{genbank,refseq}.txt.

Files land as data/benchmarks/set_H_ncbi/references/<accession>.fna (uncompressed, the
same form the Phase-1 references take in data/genomes/).

VERIFICATION (recorded, not assumed)
  * gzip trailer valid before the file is accepted;
  * total bp of the downloaded FASTA is compared with GTDB's recorded genome_size --
    a mismatch means the deposited assembly is not the one GTDB scored, and the
    reference is dropped;
  * contig count compared with GTDB's contig_count (reported, not enforced: GTDB counts
    contigs after splitting on runs of N, NCBI ships scaffolds);
  * SHA256 of every accepted .fna is written to references_manifest.tsv.

Fully resumable: an accession with an accepted .fna and a manifest row is skipped.

Usage:
    python scripts/186_ws1_11_fetch_refs.py [--workers 12]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path('/path/to/magicc')
SET_DIR = ROOT / 'data' / 'benchmarks' / 'set_H_ncbi'
REF_DIR = SET_DIR / 'references'
OUT_RES = ROOT / 'results' / 'revision' / 'circularity'
SIZE_TOL = 0.02          # |downloaded bp - GTDB genome_size| / GTDB genome_size


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fasta_stats(p: Path):
    n_contigs, bp, n_amb = 0, 0, 0
    with open(p) as f:
        for line in f:
            if line.startswith('>'):
                n_contigs += 1
            else:
                s = line.strip()
                bp += len(s)
                n_amb += sum(1 for c in s.upper() if c not in 'ACGT')
    return n_contigs, bp, n_amb


def fetch_one(task):
    acc, ftp, dest = task
    out = dest / f'{acc}.fna'
    if out.exists() and out.stat().st_size > 1000:
        return acc, 'cached', str(out)
    base = ftp.rstrip('/').split('/')[-1]
    url = ftp.rstrip('/') + '/' + base + '_genomic.fna.gz'
    tmp_gz = dest / f'{acc}.fna.gz.tmp'
    tmp = dest / f'{acc}.fna.tmp'
    for attempt in range(4):
        r = subprocess.run(
            ['curl', '-sS', '-L', '--fail', '--connect-timeout', '30',
             '--max-time', '900', '--retry', '2', '-o', str(tmp_gz), url],
            capture_output=True, text=True)
        if r.returncode == 0 and tmp_gz.exists() and tmp_gz.stat().st_size > 1000:
            try:
                with gzip.open(tmp_gz, 'rt') as fi, open(tmp, 'w') as fo:
                    for line in fi:
                        fo.write(line)
                os.replace(tmp, out)
                tmp_gz.unlink(missing_ok=True)
                return acc, 'ok', str(out)
            except Exception as e:                                   # noqa: BLE001
                tmp.unlink(missing_ok=True)
                tmp_gz.unlink(missing_ok=True)
                last = f'gunzip failed: {e!r}'
                time.sleep(2 * (attempt + 1))
                continue
        time.sleep(2 * (attempt + 1))
    tmp_gz.unlink(missing_ok=True)
    tmp.unlink(missing_ok=True)
    return acc, f'FAIL {url}', ''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    REF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(SET_DIR / 'reference_selection.tsv', sep='\t')
    print(f'references to fetch: {len(sel)}')
    tasks = [(a, f, REF_DIR) for a, f in zip(sel['primary_accession'], sel['ftp_path'])]

    t0 = time.time()
    status = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            acc, st, _ = fu.result()
            status[acc] = st
            if st.startswith('FAIL'):
                print(f'    {st}', flush=True)
            if i % 50 == 0:
                print(f'    {i}/{len(tasks)}  {time.time() - t0:.0f}s', flush=True)
    n_ok = sum(1 for v in status.values() if v in ('ok', 'cached'))
    print(f'download: {time.time() - t0:.0f}s, {n_ok}/{len(tasks)} available')

    print('\nverifying ...')
    rows = []
    for _, r in sel.iterrows():
        acc = r['primary_accession']
        p = REF_DIR / f'{acc}.fna'
        rec = {'ref_index': r['ref_index'], 'arm': r['arm'], 'pair_id': r['pair_id'],
               'gtdb_accession': r['accession'], 'accession': acc,
               'download_status': status.get(acc, 'missing'), 'path': str(p)}
        if not p.exists() or p.stat().st_size < 1000:
            rec.update({'accepted': False, 'reason': 'missing_or_empty'})
            rows.append(rec)
            continue
        nc, bp, namb = fasta_stats(p)
        gsize = float(r['genome_size'])
        dev = abs(bp - gsize) / max(gsize, 1.0)
        rec.update({'n_contigs_fasta': nc, 'bp_fasta': bp,
                    'gtdb_genome_size': int(gsize),
                    'bp_rel_deviation': round(dev, 6),
                    'gtdb_contig_count': int(r['contig_count']),
                    'n_ambiguous_bases': namb,
                    'sha256': sha256(p),
                    'bytes': p.stat().st_size})
        if dev > SIZE_TOL:
            rec.update({'accepted': False,
                        'reason': f'bp deviates {dev * 100:.2f}% from GTDB genome_size'})
        else:
            rec.update({'accepted': True, 'reason': ''})
        rows.append(rec)

    man = pd.DataFrame(rows)
    man.to_csv(SET_DIR / 'references_manifest.tsv', sep='\t', index=False)
    acc_ok = man['accepted'].fillna(False)
    print(f'accepted {int(acc_ok.sum())}/{len(man)}')
    if (~acc_ok).any():
        print(man.loc[~acc_ok, ['accession', 'arm', 'reason']].to_string(index=False))

    # a pair survives only if BOTH members do -- the matched design must stay balanced
    ok_pairs = set(man.loc[acc_ok, 'pair_id']) if acc_ok.any() else set()
    bad = {pid for pid in man['pair_id'].unique()
           if int(acc_ok[man['pair_id'] == pid].sum()) != 2}
    if bad:
        print(f'\ndropping {len(bad)} incomplete pairs to keep the matched design '
              f'balanced: {sorted(bad)[:10]}')
    keep = sel[~sel['pair_id'].isin(bad)].copy()
    keep = keep.sort_values(['pair_id', 'arm']).reset_index(drop=True)
    keep['ref_index'] = range(len(keep))
    keep.to_csv(SET_DIR / 'reference_selection_final.tsv', sep='\t', index=False)
    print(f'final references: {len(keep)} ({int((keep.arm == "H_fail").sum())} H_fail + '
          f'{int((keep.arm == "H_pass").sum())} H_pass), '
          f'{keep["pair_id"].nunique()} matched pairs')
    print(f'wrote {SET_DIR / "reference_selection_final.tsv"}')

    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'generated_by': 'scripts/186_ws1_11_fetch_refs.py',
        'n_requested': int(len(sel)),
        'n_downloaded': int(n_ok),
        'n_accepted': int(acc_ok.sum()),
        'n_pairs_dropped': len(bad),
        'size_tolerance_rel': SIZE_TOL,
        'n_final_references': int(len(keep)),
        'n_final_pairs': int(keep['pair_id'].nunique()),
        'bp_rel_deviation': {
            'median': float(man['bp_rel_deviation'].median()),
            'max': float(man['bp_rel_deviation'].max())} if 'bp_rel_deviation' in man
        else {},
    }
    with open(OUT_RES / 'ws1_11_reference_fetch.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'wrote {OUT_RES / "ws1_11_reference_fetch.json"}')
    return 0 if int(acc_ok.sum()) >= 2 else 1


if __name__ == '__main__':
    sys.exit(main())
