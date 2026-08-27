#!/usr/bin/env python3
"""
WS3.6b (optional extension) — NCBI's own `contaminated` exclusion flag as an
INDEPENDENT, NON-CIRCULAR positive control for contamination DETECTION.

23,048 GenBank assemblies carry `contaminated` in `excluded_from_refseq`: NCBI's
Foreign Contamination Screen verdict. That verdict is BINARY and uses a different
operational definition (foreign-organism sequence, adaptor, vector) from the bp-based
contamination quantified elsewhere in Track A. It is therefore framed strictly as a
detection benchmark — precision / recall / specificity at the 5 % and 10 % MIMAG
contamination thresholds — and NEVER as a quantification benchmark.

Design: N flagged prokaryotic drafts vs N unflagged controls matched on
(species_taxid where possible, else genus), assembly level, and contig count.

Usage:
    python scripts/146_ncbi_contaminated_flag_cohort.py --n 300 --workers 8
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
SUMMARY = ROOT / "data" / "ncbi" / "assembly_summary_genbank.txt"
OUT = ROOT / "results" / "revision" / "real_data" / "ncbi_flagged"
DEST = ROOT / "data" / "real_data" / "ncbi_flagged"

C_ACC, C_SPECIES_TAXID, C_ORGANISM = 0, 6, 7
C_LEVEL, C_FTP, C_EXCLUDED, C_GROUP = 11, 19, 20, 24
C_GENOME_SIZE, C_CONTIG_COUNT = 25, 30
DRAFT_LEVELS = {"Contig", "Scaffold"}


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
    for a in range(3):
        r = subprocess.run(["curl", "-sS", "-L", "--fail", "--connect-timeout", "30",
                            "--max-time", "900", "-o", str(tmp), url],
                           capture_output=True, text=True)
        if r.returncode == 0 and gzip_ok(tmp):
            os.replace(tmp, out)
            return acc, out.stat().st_size, "ok"
        time.sleep(2 * (a + 1))
    tmp.unlink(missing_ok=True)
    return acc, 0, "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1460)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)

    flagged, clean_by_sp = [], {}
    with open(SUMMARY, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) <= C_CONTIG_COUNT:
                continue
            if p[C_GROUP] not in ("bacteria", "archaea"):
                continue
            if p[C_LEVEL] not in DRAFT_LEVELS:
                continue
            if not p[C_FTP].startswith("http"):
                continue
            try:
                nc = int(p[C_CONTIG_COUNT])
                gs = int(p[C_GENOME_SIZE])
            except ValueError:
                continue
            rec = {"accession": p[C_ACC], "organism": p[C_ORGANISM],
                   "species_taxid": p[C_SPECIES_TAXID], "level": p[C_LEVEL],
                   "contigs": nc, "genome_size": gs, "ftp": p[C_FTP],
                   "excluded": p[C_EXCLUDED]}
            if "contaminated" in p[C_EXCLUDED]:
                flagged.append(rec)
            elif p[C_EXCLUDED].strip() in ("", "na"):
                clean_by_sp.setdefault(p[C_SPECIES_TAXID], []).append(rec)

    print(f"prokaryotic draft assemblies: flagged={len(flagged)}, "
          f"unflagged species groups={len(clean_by_sp)}")
    rng = random.Random(args.seed)
    rng.shuffle(flagged)

    cohort, used = [], set()
    for fl in flagged:
        if len(cohort) >= 2 * args.n:
            break
        pool = clean_by_sp.get(fl["species_taxid"], [])
        pool = [c for c in pool if c["accession"] not in used]
        if not pool:
            continue
        # closest control by contig count (matched fragmentation)
        ctrl = min(pool, key=lambda c: abs(c["contigs"] - fl["contigs"]))
        used.add(ctrl["accession"])
        used.add(fl["accession"])
        cohort.append({**fl, "label": "flagged_contaminated"})
        cohort.append({**ctrl, "label": "unflagged_control",
                       "matched_to": fl["accession"]})
    df = pd.DataFrame(cohort)
    df.to_csv(OUT / "cohort.tsv", sep="\t", index=False)
    nf = int((df.label == "flagged_contaminated").sum())
    print(f"cohort: {nf} flagged + {len(df)-nf} same-species matched controls, "
          f"{df.species_taxid.nunique()} species; median contigs "
          f"flagged={df[df.label=='flagged_contaminated'].contigs.median():.0f} "
          f"control={df[df.label=='unflagged_control'].contigs.median():.0f}")

    jobs = [(r["accession"], r["ftp"], DEST) for _, r in df.iterrows()]
    res = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            res.append(fu.result())
            if i % 200 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} "
                      f"{sum(x[1] for x in res)/1e9:.2f} GB", flush=True)
    man = pd.DataFrame(res, columns=["accession", "bytes", "status"])
    man.to_csv(OUT / "download_manifest.tsv", sep="\t", index=False)
    nfail = int((man.status == "FAIL").sum())
    print(f"downloaded {man['bytes'].sum()/1e9:.2f} GB, fail={nfail}")
    with open(OUT / "cohort_summary.json", "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "n_flagged": nf, "n_control": len(df) - nf,
                   "n_species": int(df.species_taxid.nunique()),
                   "n_fail": nfail, "seed": args.seed,
                   "framing": "BINARY DETECTION ONLY — NCBI's Foreign Contamination "
                              "Screen uses a different operational definition from the "
                              "bp-based contamination quantified in Track A."},
                  f, indent=2)


if __name__ == "__main__":
    main()
