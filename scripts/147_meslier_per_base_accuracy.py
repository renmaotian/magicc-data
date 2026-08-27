#!/usr/bin/env python3
"""
WS3.6 A1 supporting evidence — separate the two axes along which the seven MOCK1
assemblies differ: contig N50 (fragmentation) and PER-BASE accuracy (platform error
mode, especially indels).

Motivation. CheckM2's own output shows that one of the seven assemblies has median
coding density 0.75 and median gene length 132 bp against ~0.88 and ~265-305 bp for the
others, with MORE predicted CDS - the signature of indel-induced frameshifts breaking
ORFs. Since protein-based estimators depend on intact ORFs and a nucleotide k-mer method
does not, this must be measured rather than assumed.

Method. Base-level alignment (minimap2 -c) of the largest contigs of each assembly
against the pooled 91-reference index, then the CIGAR is parsed into
substitutions / insertions / deletions per aligned base. Only the largest contigs are
used (bounded by --max-bp) so the exact alignment stays cheap; those contigs carry the
bulk of each assembly anyway.

Outputs: results/revision/real_data/meslier/per_base_error_profile.tsv

Usage: python scripts/147_meslier_per_base_accuracy.py [--threads 8] [--max-bp 30000000]
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
MES = ROOT / "data" / "real_data" / "meslier2022"
WORK = MES / "work"
OUT = ROOT / "results" / "revision" / "real_data" / "meslier"
MINIMAP2 = "/path/to/conda/envs/magicc2/bin/minimap2"
ASSEMBLIES = ["pacbio", "minion", "illumina", "s5", "proton", "mgiseq_2000",
              "mgiseq_t7"]
CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def top_contigs(path, max_bp):
    recs, name, seq = [], None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(seq)))
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line.strip())
    if name is not None:
        recs.append((name, "".join(seq)))
    recs.sort(key=lambda r: -len(r[1]))
    out, tot = [], 0
    for n, s in recs:
        if tot >= max_bp:
            break
        out.append((n, s))
        tot += len(s)
    return out, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-bp", type=int, default=30_000_000)
    args = ap.parse_args()
    ref = WORK / "refs_all.fna"
    rows = []
    for tech in ASSEMBLIES:
        contigs, tot = top_contigs(MES / "assemblies" / f"MOCK_001.{tech}.fasta.gz",
                                   args.max_bp)
        with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as tf:
            for n, s in contigs:
                tf.write(f">{n}\n{s}\n")
            qpath = tf.name
        paf = WORK / f"acc_{tech}.paf"
        if not paf.exists():
            tmp = str(paf) + ".tmp"
            r = subprocess.run([MINIMAP2, "-x", "asm10", "-c", "-t", str(args.threads),
                                "--secondary=no", "-o", tmp, str(ref), qpath],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-2000:])
            os.replace(tmp, paf)
        os.unlink(qpath)

        M = I = D = NM = aln = 0
        with open(paf) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                cg = nm = None
                for tag in p[12:]:
                    if tag.startswith("cg:Z:"):
                        cg = tag[5:]
                    elif tag.startswith("NM:i:"):
                        nm = int(tag[5:])
                if cg is None:
                    continue
                for num, op in CIGAR_RE.findall(cg):
                    n = int(num)
                    if op in "M=X":
                        M += n
                    elif op == "I":
                        I += n
                    elif op == "D":
                        D += n
                if nm is not None:
                    NM += nm
                aln += int(p[3]) - int(p[2])
        base = max(1, M)
        rows.append({
            "assembly": tech, "n_contigs_used": len(contigs), "bp_used": tot,
            "aligned_M_bp": M, "insertion_bp": I, "deletion_bp": D, "NM_total": NM,
            "indel_bp_per_kb": round(1000.0 * (I + D) / base, 4),
            "substitution_per_kb": round(1000.0 * max(0, NM - I - D) / base, 4),
            "total_error_per_kb": round(1000.0 * NM / base, 4),
            "mean_bp_between_indels": round(base / max(1, I + D), 1),
        })
        print(f"  {tech:<12} indels/kb={rows[-1]['indel_bp_per_kb']:8.4f} "
              f"subs/kb={rows[-1]['substitution_per_kb']:8.4f} "
              f"bp between indels={rows[-1]['mean_bp_between_indels']:9.1f}",
              flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_base_error_profile.tsv", sep="\t", index=False)
    print(f"\nwrote {OUT/'per_base_error_profile.tsv'}")


if __name__ == "__main__":
    main()
