#!/usr/bin/env python3
"""
WS3.3 / new-priority — fetch the genome FASTAs for the cohorts built by script 131.

Downloads <asm>_genomic.fna.gz from each assembly's NCBI FTP directory (served over
HTTPS).  Resumable: an existing, non-empty, gzip-valid file is skipped.  Deliberately
polite and low-impact: at most 6 concurrent connections, retries with backoff.

Output: data/real_data/reduced_genome/genomes/<ncbi_accession>.fna.gz
        results/revision/real_data/reduced_genome/fetch_manifest.tsv
"""

from __future__ import annotations

import csv
import gzip
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path("/path/to/magicc")
COHORTS = ROOT / "results/revision/real_data/reduced_genome/cohorts.tsv"
GENOMES = ROOT / "data/real_data/reduced_genome/genomes"
MANIFEST = ROOT / "results/revision/real_data/reduced_genome/fetch_manifest.tsv"

MAX_WORKERS = 6
RETRIES = 3
UA = "magicc-revision/1.0 (research use; contact tianrenmao@gmail.com)"


def valid_gz(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read(2) == b">C" or fh.read(0) == b"" and True
    except OSError:
        return False


def valid_fasta_gz(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with gzip.open(path, "rt") as fh:
            return fh.readline().startswith(">")
    except (OSError, UnicodeDecodeError):
        return False


def fetch(row: dict) -> dict:
    acc = row["ncbi_accession"]
    dest = GENOMES / f"{acc}.fna.gz"
    if valid_fasta_gz(dest):
        return {"accession": acc, "status": "cached", "bytes": dest.stat().st_size, "url": ""}
    ftp = row["ftp_path"].replace("ftp://", "https://").rstrip("/")
    url = f"{ftp}/{ftp.rsplit('/', 1)[-1]}_genomic.fna.gz"
    last = ""
    for attempt in range(RETRIES):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=180) as resp, open(dest, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            if valid_fasta_gz(dest):
                return {"accession": acc, "status": "ok", "bytes": dest.stat().st_size, "url": url}
            last = "invalid gzip/fasta"
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2 * (attempt + 1))
    if dest.exists():
        dest.unlink()
    return {"accession": acc, "status": f"FAILED ({last})", "bytes": 0, "url": url}


def main() -> int:
    GENOMES.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(COHORTS.open(), delimiter="\t"))
    print(f"[132] {len(rows)} genomes to ensure, {MAX_WORKERS} workers", flush=True)
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(fetch, r): r for r in rows}
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 100 == 0:
                nok = sum(1 for r in results if r["status"] in ("ok", "cached"))
                print(f"[132]   {done}/{len(rows)} ok={nok}", flush=True)
    by = {}
    for r in results:
        key = r["status"] if r["status"] in ("ok", "cached") else "FAILED"
        by[key] = by.get(key, 0) + 1
    total = sum(r["bytes"] for r in results)
    with MANIFEST.open("w") as fh:
        fh.write("accession\tstatus\tbytes\turl\n")
        for r in sorted(results, key=lambda x: x["accession"]):
            fh.write(f"{r['accession']}\t{r['status']}\t{r['bytes']}\t{r['url']}\n")
    print(f"[132] {by}  total {total/1e9:.2f} GB -> {GENOMES}")
    for r in results:
        if r["status"].startswith("FAILED"):
            print(f"[132]   FAIL {r['accession']}: {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
