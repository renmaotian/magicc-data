#!/usr/bin/env python3
"""
WS3.6 A1 robustness — is the Meslier ground truth sensitive to the minimap2 preset?

The truth in Track A1 comes from minimap2 `-x asm10`. A reviewer can reasonably ask
whether a different divergence assumption would move the numbers, so the same
reference-anchored truth is recomputed under `asm5` (<=5 % divergence) and `asm20`
(<=20 % divergence) for one representative assembly and compared bin by bin.

Outputs: results/revision/real_data/meslier/truth_preset_sensitivity.tsv

Usage: python scripts/148_meslier_truth_sensitivity.py [--assembly illumina]
                                                       [--threads 6]
"""
from __future__ import annotations

import argparse
import gzip
import os
import signal
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
MES = ROOT / "data" / "real_data" / "meslier2022"
WORK = MES / "work"
OUT = ROOT / "results" / "revision" / "real_data" / "meslier"
MINIMAP2 = "/path/to/anaconda3/envs/magicc2/bin/minimap2"
MANUAL_ACC = {"Desulfovibrio_desulfuricans_ND132": "GCA_000189295.2",
              "Desulfovibrio_vulgaris_Hildenborough": "GCA_000195755.1"}


def merge_len(iv):
    if not iv:
        return 0
    iv = sorted(iv)
    tot, cs, ce = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    return tot + ce - cs


def ref_index():
    seq2acc, acc_len = {}, defaultdict(int)
    for p in sorted((MES / "references").glob("*.fna.gz")):
        base = p.name.replace(".fna.gz", "")
        if base.startswith("MOCK_"):
            continue
        acc, cur, name = None, 0, None
        with gzip.open(p, "rt") as f:
            for line in f:
                if line.startswith(">"):
                    if name is not None:
                        seq2acc[name] = acc
                        acc_len[acc] += cur
                    name = line[1:].split()[0]
                    if acc is None:
                        acc = (name.split("|")[0] if name.startswith("GC")
                               else MANUAL_ACC[base])
                    cur = 0
                else:
                    cur += len(line.strip())
        if name is not None:
            seq2acc[name] = acc
            acc_len[acc] += cur
    return seq2acc, dict(acc_len)


def truth_from_paf(paf, seq2acc, acc_len):
    q_iv = defaultdict(lambda: defaultdict(list))
    t_iv = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(paf) as f:
        for line in f:
            p = line.split("\t", 12)
            acc = seq2acc.get(p[5])
            if acc is None:
                continue
            q_iv[p[0]][acc].append((int(p[2]), int(p[3])))
            t_iv[p[0]][acc][p[5]].append((int(p[7]), int(p[8])))
    assign = {c: max(a, key=lambda k: merge_len(a[k])) for c, a in q_iv.items()}
    bins = defaultdict(list)
    for c, a in assign.items():
        bins[a].append(c)
    rows = []
    for acc, contigs in bins.items():
        by_seq = defaultdict(list)
        for c in contigs:
            for tn, iv in t_iv[c][acc].items():
                by_seq[tn].extend(iv)
        cov = sum(merge_len(v) for v in by_seq.values())
        rows.append({"accession": acc, "n_contigs": len(contigs),
                     "completeness": 100.0 * cov / acc_len[acc]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly", default="illumina")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    seq2acc, acc_len = ref_index()
    asm = MES / "assemblies" / f"MOCK_001.{args.assembly}.fasta.gz"
    frames = {}
    for preset in ("asm5", "asm10", "asm20"):
        paf = (WORK / f"MOCK_001.{args.assembly}.paf" if preset == "asm10"
               else WORK / f"sens_{args.assembly}_{preset}.paf")
        if not paf.exists():
            tmp = str(paf) + ".tmp"
            r = subprocess.run([MINIMAP2, "-x", preset, "-t", str(args.threads),
                                "--secondary=no", "-o", tmp,
                                str(WORK / "refs_all.fna"), str(asm)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-2000:])
            os.replace(tmp, paf)
        frames[preset] = truth_from_paf(paf, seq2acc, acc_len).set_index("accession")
        print(f"  {preset}: {len(frames[preset])} bins", flush=True)

    m = pd.DataFrame({p: f["completeness"] for p, f in frames.items()})
    m = m.dropna()
    m["asm5_minus_asm10"] = m["asm5"] - m["asm10"]
    m["asm20_minus_asm10"] = m["asm20"] - m["asm10"]
    m = m.reset_index()
    m.to_csv(OUT / "truth_preset_sensitivity.tsv", sep="\t", index=False)
    print(f"\nassembly={args.assembly}, bins compared in all three presets: {len(m)}")
    for c in ("asm5_minus_asm10", "asm20_minus_asm10"):
        v = m[c].values
        print(f"  {c}: mean {v.mean():+.4f} pp, median {np.median(v):+.4f}, "
              f"max |Δ| {np.abs(v).max():.4f} pp, "
              f"n with |Δ|>1 pp: {(np.abs(v) > 1).sum()}")
    print(f"wrote {OUT/'truth_preset_sensitivity.tsv'}")


if __name__ == "__main__":
    main()
