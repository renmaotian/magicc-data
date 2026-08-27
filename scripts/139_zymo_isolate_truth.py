#!/usr/bin/env python3
"""
WS3.6 / Track A2 — Zymo isolate draft/complete pairs.

Nicholls et al. 2019 (GigaScience 8:giz043) published SPAdes Illumina draft assemblies
for the 10 ZymoBIOMICS isolates as one multi-FASTA with a two-letter organism prefix on
every contig. ZymoBIOMICS ships the matching COMPLETE reference genomes. The two yeasts
are eukaryotes and are excluded (MAGICC is bacteria/archaea only), leaving
8 bacterial draft/complete pairs with zero assembly compute.

Ground truth uses MAGICC's denominator convention, identical to Track A1:
    completeness  = |union of own-reference bp covered by the draft| / own reference
                    FULL length (all replicons) x 100
    contamination = draft bp aligning to ANOTHER Zymo reference (and not to its own)
                    / own reference full length x 100
    contamination_upper = (as above + draft bp aligning to no reference)
                    / own reference full length x 100

Aligning every draft against the pooled 10-organism reference (not only its own) is what
makes the contamination term meaningful: cross-isolate carry-over is detectable.

Outputs (results/revision/real_data/zymo/):
    zymo_truth.tsv, drafts/<code>.fasta

Usage: python scripts/139_zymo_isolate_truth.py [--threads 8]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
ZYMO = ROOT / "data" / "real_data" / "zymo"
GENOMES = ZYMO / "ZymoBIOMICS.STD.refseq.v2" / "Genomes"
ISOLATES = ZYMO / "Zymo-Isolates-SPAdes-Illumina.fasta"
WORK = ZYMO / "work"
DRAFTS = ZYMO / "drafts"
OUT = ROOT / "results" / "revision" / "real_data" / "zymo"

# two-letter contig prefix -> Zymo reference FASTA basename
CODE2REF = {
    "bs": "Bacillus_subtilis_complete_genome",
    "ec": "Escherichia_coli_complete_genome",
    "ef": "Enterococcus_faecalis_complete_genome",
    "lf": "Lactobacillus_fermentum_complete_genome",
    "lm": "Listeria_monocytogenes_complete_genome",
    "pa": "Pseudomonas_aeruginosa_complete_genome",
    "sa": "Staphylococcus_aureus_complete_genome",
    "se": "Salmonella_enterica_complete_genome",
    "cn": "Cryptococcus_neoformans_draft_genome",     # eukaryote — excluded
    "sc": "Saccharomyces_cerevisiae_draft_genome",    # eukaryote — excluded
}
EUKARYOTES = {"cn", "sc"}
ORGANISM = {
    "bs": "Bacillus subtilis", "ec": "Escherichia coli",
    "ef": "Enterococcus faecalis", "lf": "Limosilactobacillus fermentum",
    "lm": "Listeria monocytogenes", "pa": "Pseudomonas aeruginosa",
    "sa": "Staphylococcus aureus", "se": "Salmonella enterica",
}


def merge_intervals(iv):
    if not iv:
        return 0, []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return sum(e - s for s, e in out), out


def subtract_length(a_iv, b_iv):
    _, a = merge_intervals(a_iv)
    _, b = merge_intervals(b_iv)
    if not a:
        return 0
    if not b:
        return sum(e - s for s, e in a)
    total, j = 0, 0
    for s, e in a:
        cur, k = s, j
        while k < len(b) and b[k][0] < e:
            bs, be = b[k]
            if be <= cur:
                k += 1
                continue
            if bs > cur:
                total += min(bs, e) - cur
            cur = max(cur, be)
            if cur >= e:
                break
            k += 1
        if cur < e:
            total += e - cur
        j = max(0, k - 1)
    return total


def read_fasta(path):
    name, seq = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].strip(), []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for d in (WORK, DRAFTS, OUT):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("WS3.6 A2 — Zymo isolate draft/complete pairs")
    print("=" * 78)

    # ---- pooled reference with code-tagged sequence ids
    pooled = WORK / "zymo_refs_all.fna"
    seq2code, ref_len, ref_nseq = {}, defaultdict(int), defaultdict(int)
    if not pooled.exists() or args.force:
        with open(pooled, "w") as o:
            for code, base in CODE2REF.items():
                p = GENOMES / f"{base}.fasta"
                for i, (hdr, seq) in enumerate(read_fasta(p)):
                    sid = f"{code}|{i}"
                    o.write(f">{sid}\n{seq}\n")
    for code, base in CODE2REF.items():
        for i, (hdr, seq) in enumerate(read_fasta(GENOMES / f"{base}.fasta")):
            seq2code[f"{code}|{i}"] = code
            ref_len[code] += len(seq)
            ref_nseq[code] += 1
    print(f"pooled reference: {len(seq2code)} sequences, "
          f"{sum(ref_len.values())/1e6:.2f} Mbp, {len(CODE2REF)} organisms")

    # ---- split the isolate multi-FASTA by two-letter prefix
    by_code = defaultdict(list)
    for hdr, seq in read_fasta(ISOLATES):
        cid = hdr.split()[0]
        code = cid.split("_")[0]
        by_code[code].append((cid, seq))
    print("draft contigs per code:",
          {c: len(v) for c, v in sorted(by_code.items())})
    for code, recs in by_code.items():
        if code in EUKARYOTES:
            continue
        p = DRAFTS / f"{code}.fasta"
        if not p.exists() or args.force:
            with open(p, "w") as o:
                for cid, seq in recs:
                    o.write(f">{cid}\n")
                    for i in range(0, len(seq), 80):
                        o.write(seq[i:i + 80] + "\n")

    # ---- align every bacterial draft against the pooled 10-organism reference
    rows = []
    for code in sorted(by_code):
        if code in EUKARYOTES:
            continue
        paf = WORK / f"{code}.paf"
        if not paf.exists() or args.force:
            tmp = str(paf) + ".tmp"
            r = subprocess.run(
                ["conda", "run", "-n", "magicc2", "minimap2", "-x", "asm10",
                 "-t", str(args.threads), "--secondary=no", "-o", tmp,
                 str(pooled), str(DRAFTS / f"{code}.fasta")],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-2000:])
            os.replace(tmp, paf)

        q_iv = defaultdict(lambda: defaultdict(list))
        t_iv = defaultdict(list)
        clen = {}
        with open(paf) as f:
            for line in f:
                p = line.split("\t", 12)
                q, ql, qs, qe = p[0], int(p[1]), int(p[2]), int(p[3])
                tc = seq2code.get(p[5])
                if tc is None:
                    continue
                clen[q] = ql
                q_iv[q][tc].append((qs, qe))
                if tc == code:
                    t_iv[p[5]].append((int(p[7]), int(p[8])))

        covered = sum(merge_intervals(v)[0] for v in t_iv.values())
        contam_bp, unaligned_bp, draft_bp = 0, 0, 0
        lens = []
        for cid, seq in by_code[code]:
            L = len(seq)
            draft_bp += L
            lens.append(L)
            dom = q_iv[cid].get(code, [])
            oth = [x for c2, iv in q_iv[cid].items() if c2 != code for x in iv]
            contam_bp += subtract_length(oth, dom)
            allq = [x for iv in q_iv[cid].values() for x in iv]
            unaligned_bp += L - merge_intervals(allq)[0]
        lens.sort(reverse=True)
        cum, n50 = 0, 0
        for L in lens:
            cum += L
            if cum >= draft_bp / 2:
                n50 = L
                break
        R = ref_len[code]
        rows.append({
            "code": code, "organism": ORGANISM[code],
            "draft_fasta": str(DRAFTS / f"{code}.fasta"),
            "n_contigs": len(by_code[code]), "draft_bp": draft_bp, "draft_n50": n50,
            "ref_len": R, "ref_nseq": ref_nseq[code],
            "draft_over_ref": round(draft_bp / R, 4),
            "covered_ref_bp": covered, "contam_bp": contam_bp,
            "unaligned_bp": unaligned_bp,
            "true_completeness": round(100.0 * covered / R, 4),
            "true_contamination": round(100.0 * contam_bp / R, 4),
            "true_contamination_upper": round(100.0 * (contam_bp + unaligned_bp) / R, 4),
        })
        print(f"  {code} {ORGANISM[code]:<28} comp={rows[-1]['true_completeness']:6.2f}  "
              f"cont={rows[-1]['true_contamination']:6.3f}  "
              f"cont_upper={rows[-1]['true_contamination_upper']:6.3f}  "
              f"contigs={len(by_code[code])}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "zymo_truth.tsv", sep="\t", index=False)
    print(f"\nwrote {OUT/'zymo_truth.tsv'} ({len(df)} pairs)")
    with open(OUT / "zymo_truth_summary.json", "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "n_pairs": len(df), "excluded_eukaryotes": sorted(EUKARYOTES),
                   "aligner": "minimap2 -x asm10 --secondary=no",
                   "completeness_median": float(df["true_completeness"].median()),
                   "contamination_median": float(df["true_contamination"].median())},
                  f, indent=2)


if __name__ == "__main__":
    main()
