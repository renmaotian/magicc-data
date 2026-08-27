#!/usr/bin/env python3
"""
WS3.6b / Track A3 — download the NCBI Tier-1 (same-BioSample) draft/complete pairs
discovered by scripts/130_ncbi_draft_complete_pairs.py.

Cohort: prokaryotes only (MAGICC is bacteria/archaea), draft/complete size ratio in
[0.80, 1.20] -> 1,097 pairs across 165 species. Same BioSample means the same physical
DNA isolate, so draft sequence absent from the complete assembly is assembly artefact or
contamination rather than strain divergence.

Downloads <acc>_<asmname>_genomic.fna.gz for both members of each pair into
    data/real_data/ncbi_pairs/{drafts,completes}/<accession>.fna.gz

Fully resumable: an accession already present with non-zero size and a valid gzip
trailer is skipped. A per-accession manifest with byte counts is written so a collapsed
run can be diagnosed. Bounded concurrency (default 8 connections, NCBI-friendly).

Usage:
    python scripts/137_fetch_ncbi_pairs.py --limit 100      # pilot
    python scripts/137_fetch_ncbi_pairs.py                  # full cohort
"""
from __future__ import annotations

import argparse
import gzip
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

ROOT = Path("/path/to/magicc")
PAIRS = ROOT / "results" / "revision" / "real_data" / "ncbi_pairs"
DEST = ROOT / "data" / "real_data" / "ncbi_pairs"


def gzip_ok(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 1000:
        return False
    try:
        with gzip.open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
        return True
    except Exception:
        return False


def fetch(args):
    acc, ftp, dest = args
    out = dest / f"{acc}.fna.gz"
    if gzip_ok(out):
        return acc, out.stat().st_size, "cached"
    url = ftp.rstrip("/") + "/" + ftp.rstrip("/").split("/")[-1] + "_genomic.fna.gz"
    tmp = dest / f"{acc}.fna.gz.tmp"
    for attempt in range(4):
        r = subprocess.run(
            ["curl", "-sS", "-L", "--fail", "--connect-timeout", "30",
             "--max-time", "900", "--retry", "2", "-o", str(tmp), url],
            capture_output=True, text=True)
        if r.returncode == 0 and gzip_ok(tmp):
            os.replace(tmp, out)
            return acc, out.stat().st_size, "ok"
        time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return acc, 0, f"FAIL {url}"


def select_cohort(limit=None, seed=1360):
    t1 = pd.read_csv(PAIRS / "tier1_same_biosample_pairs.tsv", sep="\t", dtype=str)
    t1["ratio"] = pd.to_numeric(t1["size_ratio_draft_over_complete"], errors="coerce")
    sel = t1[t1["group"].isin(["bacteria", "archaea"])
             & t1["ratio"].between(0.80, 1.20)].copy()
    sel = sel.sort_values("draft_accession").reset_index(drop=True)
    if limit:
        # stratified pilot: spread across species rather than the first N alphabetically
        per = max(1, limit // max(1, sel["species_taxid"].nunique()) + 1)
        idx = sel.groupby("species_taxid").head(per).index
        sel = (sel.loc[idx].sample(frac=1.0, random_state=seed).head(limit)
               .sort_values("draft_accession").reset_index(drop=True))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    (DEST / "drafts").mkdir(parents=True, exist_ok=True)
    (DEST / "completes").mkdir(parents=True, exist_ok=True)

    sel = select_cohort(args.limit or None)
    sel.to_csv(PAIRS / ("cohort_pilot.tsv" if args.limit else "cohort_full.tsv"),
               sep="\t", index=False)
    print(f"cohort: {len(sel)} pairs, {sel['species_taxid'].nunique()} species, "
          f"{sel['draft_accession'].nunique()} drafts, "
          f"{sel['complete_accession'].nunique()} completes")

    jobs = []
    seen = set()
    for _, r in sel.iterrows():
        for acc, ftp, sub in ((r["draft_accession"], r["draft_ftp"], "drafts"),
                              (r["complete_accession"], r["complete_ftp"], "completes")):
            key = (sub, acc)
            if key in seen:
                continue
            seen.add(key)
            jobs.append((acc, ftp, DEST / sub))
    print(f"{len(jobs)} unique files to ensure, {args.workers} connections")

    res, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            acc, size, status = f.result()
            res.append({"accession": acc, "bytes": size, "status": status})
            if i % 200 == 0 or i == len(futs):
                nf = sum(1 for x in res if x["status"].startswith("FAIL"))
                print(f"  {i}/{len(futs)}  fail={nf}  "
                      f"{sum(x['bytes'] for x in res)/1e9:.2f} GB  "
                      f"{time.time()-t0:.0f}s", flush=True)

    man = pd.DataFrame(res)
    man.to_csv(PAIRS / "download_manifest.tsv", sep="\t", index=False)
    fails = man[man["status"].str.startswith("FAIL")]
    print(f"\ndownloaded {man['bytes'].sum()/1e9:.2f} GB; "
          f"cached={int((man['status']=='cached').sum())} "
          f"ok={int((man['status']=='ok').sum())} fail={len(fails)}")
    if len(fails):
        print(fails.head(20).to_string())
    with open(PAIRS / "download_summary.json", "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "n_pairs": int(len(sel)), "n_files": len(jobs),
                   "bytes": int(man["bytes"].sum()),
                   "n_fail": int(len(fails))}, f, indent=2)


if __name__ == "__main__":
    main()
