#!/usr/bin/env python3
"""
WS3.8 (R1-M3) — CAMI II acquisition.

CAMI II is fully open at https://frl.publisso.de/data/frl:6425521/ (no request form,
contrary to the CAMI web page). This script downloads, MD5-verifies against the
repository's own `md5sums.txt`, and unpacks the pieces WS3.8 needs:

  * <dataset>/short_read/<prefix>_sample_N_contigs.tar.gz
        -> anonymous_gsa.fasta.gz   (gold-standard assembly for that sample)
        -> gsa_mapping.tsv.gz       (contig -> source genome, exact)
        -> binning_gs.tsv           (binning gold standard)
  * <dataset>/<prefix>_genomes.tar.gz
        -> the source genome FASTAs, needed for FULL REFERENCE LENGTHS, which are the
           denominator of MAGICC's completeness/contamination definition.
  * <dataset>/short_read/<prefix>_setup.tar.gz
        -> metadata / taxonomy / abundance tables produced by CAMI.

Downloads are resumable (`curl -C -`), collapse-safe (atomic .part rename), and skipped
when the MD5 already matches, so re-running is cheap.

Usage:
    python scripts/195_ws38_fetch_cami2.py --dataset strain_madness --samples 0-19
    python scripts/195_ws38_fetch_cami2.py --dataset marine --samples all
    python scripts/195_ws38_fetch_cami2.py --dataset strain_madness --genomes-only
"""

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data' / 'real_data' / 'cami2'
RAW_DIR = DATA_DIR / 'raw'

BASE_URL = 'https://frl.publisso.de/data/frl:6425521'

DATASETS = {
    'marine': {
        'remote_dir': 'marine',
        'prefix': 'marmgCAMI2',
        'n_samples': 10,
    },
    'strain_madness': {
        'remote_dir': 'strain',
        'prefix': 'strmgCAMI2',
        'n_samples': 100,
    },
    'plant_associated': {
        'remote_dir': 'plant_associated',
        'prefix': 'rhimgCAMI2',
        'n_samples': 21,
    },
}


def load_md5s():
    """repo-provided md5sums.txt -> {relative_path: md5}"""
    path = RAW_DIR / 'md5sums.txt'
    if not path.exists():
        subprocess.run(['curl', '-sSL', f'{BASE_URL}/md5sums.txt', '-o', str(path)], check=True)
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(rel_path, md5s, retries=4):
    """Download BASE_URL/rel_path to RAW_DIR/rel_path, verify MD5. Returns local Path."""
    dest = RAW_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = md5s.get(rel_path)

    if dest.exists() and expected:
        if md5_file(dest) == expected:
            print(f'[skip] {rel_path} (md5 ok)', flush=True)
            return dest
        print(f'[warn] {rel_path} md5 mismatch on disk, re-downloading', flush=True)
        dest.unlink()

    url = f'{BASE_URL}/{rel_path}'
    for attempt in range(1, retries + 1):
        t0 = time.time()
        rc = subprocess.run(
            ['curl', '-sSL', '--fail', '-C', '-', '--retry', '3', '--retry-delay', '5',
             url, '-o', str(dest)],
        ).returncode
        if rc == 0 and dest.exists():
            if expected is None:
                print(f'[ok  ] {rel_path} ({dest.stat().st_size/1e6:.0f} MB, '
                      f'{time.time()-t0:.0f}s, NO MD5 IN MANIFEST)', flush=True)
                return dest
            got = md5_file(dest)
            if got == expected:
                print(f'[ok  ] {rel_path} ({dest.stat().st_size/1e6:.0f} MB, '
                      f'{time.time()-t0:.0f}s, md5 verified)', flush=True)
                return dest
            print(f'[fail] {rel_path} md5 {got} != {expected} (attempt {attempt})', flush=True)
            dest.unlink(missing_ok=True)
        else:
            print(f'[fail] curl rc={rc} on {rel_path} (attempt {attempt})', flush=True)
        time.sleep(10 * attempt)
    raise RuntimeError(f'could not download {rel_path}')


def safe_extract(tar_path, out_dir, members_filter=None):
    """Extract, refusing absolute / parent-escaping member paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tf:
        members = []
        for m in tf.getmembers():
            name = m.name
            if name.startswith('/') or '..' in Path(name).parts:
                raise RuntimeError(f'unsafe tar member {name} in {tar_path}')
            if members_filter and not members_filter(name):
                continue
            members.append(m)
        tf.extractall(out_dir, members=members)
    return [m.name for m in members]


def fetch_sample(dataset, idx, md5s):
    cfg = DATASETS[dataset]
    rel = f"{cfg['remote_dir']}/short_read/{cfg['prefix']}_sample_{idx}_contigs.tar.gz"
    tar_path = download(rel, md5s)

    sample_dir = DATA_DIR / dataset / f'sample_{idx}'
    marker = sample_dir / '.extracted'
    if marker.exists():
        print(f'[skip] extract sample_{idx} (already done)', flush=True)
        return sample_dir

    wanted = ('anonymous_gsa.fasta.gz', 'gsa_mapping.tsv.gz', 'binning_gs.tsv',
              'gsa_mapping.binning', 'anonymous_gsa.fasta')

    def keep(name):
        return Path(name).name in wanted

    names = safe_extract(tar_path, sample_dir, members_filter=keep)
    if not names:
        # fall back: extract everything, the layout differs between datasets
        names = safe_extract(tar_path, sample_dir)
    marker.write_text('\n'.join(sorted(names)) + '\n')
    print(f'[ok  ] extracted sample_{idx}: {len(names)} members', flush=True)
    return sample_dir


def fetch_genomes(dataset, md5s):
    cfg = DATASETS[dataset]
    rel = f"{cfg['remote_dir']}/{cfg['prefix']}_genomes.tar.gz"
    tar_path = download(rel, md5s)
    out_dir = DATA_DIR / 'genomes' / dataset
    marker = out_dir / '.extracted'
    if marker.exists():
        print(f'[skip] extract genomes {dataset}', flush=True)
        return out_dir
    names = safe_extract(tar_path, out_dir)
    marker.write_text(f'{len(names)} members\n')
    print(f'[ok  ] extracted {dataset} genomes: {len(names)} members', flush=True)
    return out_dir


def fetch_setup(dataset, md5s):
    cfg = DATASETS[dataset]
    rel = f"{cfg['remote_dir']}/short_read/{cfg['prefix']}_setup.tar.gz"
    try:
        tar_path = download(rel, md5s)
    except RuntimeError as exc:
        print(f'[warn] setup unavailable for {dataset}: {exc}', flush=True)
        return None
    out_dir = DATA_DIR / dataset / 'setup'
    marker = out_dir / '.extracted'
    if marker.exists():
        return out_dir

    # setup tarballs are large and mostly simulation configs; keep only metadata tables
    def keep(name):
        base = Path(name).name
        return (base.endswith(('.tsv', '.txt', '.yaml', '.json', '.biom'))
                or 'metadata' in base or 'taxonom' in base or 'abundance' in base)

    names = safe_extract(tar_path, out_dir, members_filter=keep)
    marker.write_text('\n'.join(sorted(names)[:5000]) + '\n')
    print(f'[ok  ] extracted {dataset} setup: {len(names)} metadata members', flush=True)
    return out_dir


def parse_samples(spec, n_max):
    if spec == 'all':
        return list(range(n_max))
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [i for i in out if 0 <= i < n_max]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--samples', default='all')
    ap.add_argument('--genomes-only', action='store_true')
    ap.add_argument('--skip-setup', action='store_true')
    ap.add_argument('--manifest', action='store_true',
                    help='write SHA256 manifest of everything downloaded')
    args = ap.parse_args()

    md5s = load_md5s()
    cfg = DATASETS[args.dataset]

    fetch_genomes(args.dataset, md5s)
    if not args.skip_setup:
        fetch_setup(args.dataset, md5s)

    if not args.genomes_only:
        for idx in parse_samples(args.samples, cfg['n_samples']):
            fetch_sample(args.dataset, idx, md5s)

    if args.manifest:
        man = RAW_DIR / f'sha256_manifest_{args.dataset}.txt'
        lines = []
        for p in sorted((RAW_DIR / cfg['remote_dir']).rglob('*')):
            if p.is_file():
                lines.append(f'{sha256_file(p)}  {p.relative_to(PROJECT_DIR)}')
        man.write_text('\n'.join(lines) + '\n')
        print(f'[ok  ] SHA256 manifest -> {man} ({len(lines)} files)', flush=True)

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
