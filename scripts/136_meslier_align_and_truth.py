#!/usr/bin/env python3
"""
WS3.6 / Track A1 — Meslier et al. 2022 mock communities.
Step 1: align the seven pre-computed MOCK1 assemblies to the 91 reference genomes,
build reference-anchored bins, and derive ground truth in MAGICC's own denominator
convention.

Data (already on disk, no download):
    data/real_data/meslier2022/references/*.fna.gz     91 reference genomes
    data/real_data/meslier2022/assemblies/MOCK_001.<tech>.fasta.gz   7 assemblies
    data/real_data/meslier2022/composition_S1.tsv      91 rows, exact MOCK1/2/3 abundances

Ground-truth definition (MAGICC convention, protocol R1-M4/R1-M5):
    completeness(bin)  = |union of dominant-reference bp covered by the bin's contigs|
                         / (dominant reference FULL length, all replicons) x 100
    contamination(bin) = (bin query bp that align to a reference OTHER than the dominant
                          one, excluding query bp that also align to the dominant one)
                         / (dominant reference FULL length) x 100
    contamination_upper(bin) = (as above + bin query bp aligning to NO reference)
                         / (dominant reference FULL length) x 100

Binning ("reference-anchored"): every contig with >=1 alignment is assigned to the
reference accumulating the most aligned query bp for that contig; one bin per reference
per assembly. Contigs with no alignment to any of the 91 references are not binned and
are reported as a per-assembly QC quantity.

Outputs (results/revision/real_data/meslier/):
    reference_index.tsv        per-reference length, accession, MOCK membership, split
    alignment_qc.tsv           per-assembly alignment QC
    bin_truth.tsv              one row per (assembly, bin) with ground truth
Bins written to data/real_data/meslier2022/bins/<assembly>/<accession>.fasta

Resumable: minimap2 PAFs and bin FASTAs are skipped when already complete.

Usage:  python scripts/136_meslier_align_and_truth.py [--threads 12] [--force]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
MES = ROOT / "data" / "real_data" / "meslier2022"
REFDIR = MES / "references"
ASMDIR = MES / "assemblies"
WORK = MES / "work"
BINDIR = MES / "bins"
OUT = ROOT / "results" / "revision" / "real_data" / "meslier"
SPLITDIR = ROOT / "data" / "splits"

ASSEMBLIES = ["pacbio", "minion", "illumina", "s5", "proton", "mgiseq_2000", "mgiseq_t7"]

# Two reference FASTAs carry RefSeq headers instead of the GCA|seqid convention.
MANUAL_ACC = {
    "Desulfovibrio_desulfuricans_ND132": "GCA_000189295.2",
    "Desulfovibrio_vulgaris_Hildenborough": "GCA_000195755.1",
}

MIN_BIN_BP = 50_000  # bins smaller than this cannot be meaningfully scored by any tool


# --------------------------------------------------------------------- helpers
def merge_intervals(iv):
    """Total length of the union of a list of half-open [s,e) intervals."""
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
    """Length of (union(a) \\ union(b)) for half-open interval lists."""
    _, a = merge_intervals(a_iv)
    _, b = merge_intervals(b_iv)
    if not a:
        return 0
    if not b:
        return sum(e - s for s, e in a)
    total = 0
    j = 0
    for s, e in a:
        cur = s
        while j > 0 and b[j - 1][1] > cur:
            j -= 1
        k = j
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
        j = k
    return total


def read_fasta_stream(path):
    op = gzip.open if str(path).endswith(".gz") else open
    name, seq = None, []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].strip(), []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


# ----------------------------------------------------------------- ref indexing
def build_reference_index():
    """seqid -> accession ; accession -> (label, total_bp, n_seq)."""
    seq2acc, acc_len, acc_nseq, acc_label = {}, defaultdict(int), defaultdict(int), {}
    for p in sorted(REFDIR.glob("*.fna.gz")):
        base = p.name.replace(".fna.gz", "")
        if base.startswith("MOCK_"):
            continue
        acc = None
        for hdr, seq in read_fasta_stream(p):
            sid = hdr.split()[0]
            if acc is None:
                acc = sid.split("|")[0] if sid.startswith("GC") else MANUAL_ACC[base]
            seq2acc[sid] = acc
            acc_len[acc] += len(seq)
            acc_nseq[acc] += 1
        acc_label[acc] = base
    return seq2acc, dict(acc_len), dict(acc_nseq), acc_label


def load_splits():
    """accession (GCA and GCF forms) -> split label."""
    mapping = {}
    for split in ("train", "val", "test"):
        f = SPLITDIR / f"{split}_genomes.tsv"
        df = pd.read_csv(f, sep="\t", usecols=["ncbi_accession", "gcf_accession"],
                         dtype=str)
        for col in ("ncbi_accession", "gcf_accession"):
            for a in df[col].dropna().unique():
                a = a.strip()
                if a and a.lower() != "nan":
                    mapping[a] = split
                    mapping[a.split(".")[0]] = split  # version-insensitive
    return mapping


# -------------------------------------------------------------------- minimap2
def run_minimap2(tech, threads, force=False):
    paf = WORK / f"MOCK_001.{tech}.paf"
    done = WORK / f"MOCK_001.{tech}.paf.done"
    if paf.exists() and done.exists() and not force:
        print(f"  [{tech}] PAF present — reuse ({paf.stat().st_size/1e6:.1f} MB)")
        return paf
    asm = ASMDIR / f"MOCK_001.{tech}.fasta.gz"
    tmp = WORK / f"MOCK_001.{tech}.paf.tmp"
    cmd = ["minimap2", "-x", "asm10", "-t", str(threads), "--secondary=no",
           "-o", str(tmp), str(WORK / "refs_all.fna"), str(asm)]
    print(f"  [{tech}] minimap2 ...", flush=True)
    env = dict(os.environ)
    r = subprocess.run(["conda", "run", "-n", "magicc2"] + cmd, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"minimap2 failed for {tech}: {r.stderr[-2000:]}")
    os.replace(tmp, paf)
    done.write_text(datetime.now(timezone.utc).isoformat())
    return paf


# --------------------------------------------------------------- PAF -> truth
def process_assembly(tech, paf, seq2acc, acc_len, mock1_accs, threads, force=False):
    """Return (bin rows, qc row). Writes bin FASTAs."""
    # contig -> acc -> query intervals ; contig -> acc -> target SEQUENCE -> intervals
    # (target intervals MUST be kept per reference sequence: 26 of the 91 references
    #  are multi-replicon, and pooling their coordinates would silently merge distinct
    #  replicons into one coordinate space and under-count covered bp)
    q_iv = defaultdict(lambda: defaultdict(list))
    t_iv = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    contig_len = {}
    n_lines = 0
    # minimap2's dv:f: tag is the approximate per-base sequence divergence INCLUDING
    # indels. Weighted by aligned length it gives the assembly's per-base accuracy
    # against the known reference — the axis that is distinct from fragmentation.
    dv_num, dv_den = 0.0, 0
    with open(paf) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            q, ql, qs, qe = p[0], int(p[1]), int(p[2]), int(p[3])
            tname, ts, te = p[5], int(p[7]), int(p[8])
            acc = seq2acc.get(tname)
            if acc is None:
                continue
            contig_len[q] = ql
            q_iv[q][acc].append((qs, qe))
            t_iv[q][acc][tname].append((ts, te))
            n_lines += 1
            for tag in p[12:]:
                if tag.startswith("dv:f:"):
                    dv_num += float(tag[5:]) * (qe - qs)
                    dv_den += qe - qs
                    break

    # contig -> dominant accession (most aligned query bp)
    assign = {}
    contig_aligned_bp = {}
    for c, accs in q_iv.items():
        best, best_bp, tot_iv = None, -1, []
        for a, iv in accs.items():
            bp, _ = merge_intervals(iv)
            tot_iv.extend(iv)
            if bp > best_bp:
                best, best_bp = a, bp
        assign[c] = best
        contig_aligned_bp[c] = merge_intervals(tot_iv)[0]

    bins = defaultdict(list)
    for c, a in assign.items():
        bins[a].append(c)

    # ---- per-bin truth
    rows = []
    for acc, contigs in bins.items():
        ref_len = acc_len[acc]
        tgt_by_seq = defaultdict(list)
        contam_bp, unaligned_bp, bin_bp = 0, 0, 0
        lens = []
        for c in contigs:
            bin_bp += contig_len[c]
            lens.append(contig_len[c])
            for tname, iv in t_iv[c][acc].items():
                tgt_by_seq[tname].extend(iv)
            dom_q = q_iv[c][acc]
            oth_q = [x for a2, iv in q_iv[c].items() if a2 != acc for x in iv]
            contam_bp += subtract_length(oth_q, dom_q)
            unaligned_bp += contig_len[c] - contig_aligned_bp[c]
        covered = sum(merge_intervals(v)[0] for v in tgt_by_seq.values())
        lens.sort(reverse=True)
        cum, n50 = 0, 0
        for L in lens:
            cum += L
            if cum >= bin_bp / 2:
                n50 = L
                break
        rows.append({
            "bin_id": f"{tech}__{acc}",
            "assembly": tech, "accession": acc, "n_contigs": len(contigs),
            "bin_bp": bin_bp, "bin_n50": n50, "ref_len": ref_len,
            "covered_ref_bp": covered, "contam_bp": contam_bp,
            "unaligned_bp": unaligned_bp,
            # bin bp per bp of reference actually recovered: >1 means the assembly
            # carries redundant copies of the same reference locus (a real property of
            # metagenome assemblies, and a source of marker duplication for
            # duplication-based estimators)
            "bin_bp_per_covered_ref_bp": round(bin_bp / max(1, covered), 4),
            "true_completeness": round(100.0 * covered / ref_len, 6),
            "true_contamination": round(100.0 * contam_bp / ref_len, 6),
            "true_contamination_upper": round(
                100.0 * (contam_bp + unaligned_bp) / ref_len, 6),
            "in_mock1": acc in mock1_accs,
        })

    # ---- QC
    total_contigs = 0
    total_bp = 0
    unassigned_contigs = 0
    unassigned_bp = 0
    for name, seq in read_fasta_stream(ASMDIR / f"MOCK_001.{tech}.fasta.gz"):
        cid = name.split()[0]
        total_contigs += 1
        total_bp += len(seq)
        if cid not in assign:
            unassigned_contigs += 1
            unassigned_bp += len(seq)
    binned_mock1_bp = sum(r["bin_bp"] for r in rows if r["in_mock1"])
    qc = {
        "assembly": tech, "paf_alignments": n_lines,
        "weighted_mean_divergence_vs_reference": (round(dv_num / dv_den, 6)
                                                  if dv_den else float("nan")),
        "total_contigs": total_contigs, "total_bp": total_bp,
        "contigs_with_alignment": len(assign),
        "unassigned_contigs": unassigned_contigs, "unassigned_bp": unassigned_bp,
        "unassigned_bp_pct": round(100.0 * unassigned_bp / total_bp, 3),
        "n_bins": len(rows),
        "n_bins_mock1": sum(1 for r in rows if r["in_mock1"]),
        "bp_in_mock1_bins": binned_mock1_bp,
        "bp_in_nonmock1_bins": sum(r["bin_bp"] for r in rows if not r["in_mock1"]),
    }

    # ---- write bin FASTAs (only bins on MOCK1 references above the size floor)
    keep = {r["accession"] for r in rows
            if r["in_mock1"] and r["bin_bp"] >= MIN_BIN_BP}
    bdir = BINDIR / tech
    bdir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in bdir.glob("*.fasta")}
    if not keep.issubset(existing) or force:
        want = {c: assign[c] for c in assign if assign[c] in keep}
        handles = {}
        try:
            for a in keep:
                handles[a] = open(bdir / f"{a}.fasta.tmp", "w")
            for name, seq in read_fasta_stream(ASMDIR / f"MOCK_001.{tech}.fasta.gz"):
                cid = name.split()[0]
                a = want.get(cid)
                if a is not None:
                    handles[a].write(f">{cid}\n")
                    for i in range(0, len(seq), 80):
                        handles[a].write(seq[i:i + 80] + "\n")
        finally:
            for h in handles.values():
                h.close()
        for a in keep:
            os.replace(bdir / f"{a}.fasta.tmp", bdir / f"{a}.fasta")
        for p in bdir.glob("*.fasta"):
            if p.stem not in keep:
                p.unlink()
    for r in rows:
        r["bin_fasta"] = (str(bdir / f"{r['accession']}.fasta")
                          if r["accession"] in keep else "")
    return rows, qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "refs_all.fna").exists():
        raise SystemExit("missing work/refs_all.fna — build with zcat references/*.fna.gz")

    print("=" * 78)
    print("WS3.6 A1 — Meslier MOCK1: reference-anchored bins + ground truth")
    print("=" * 78)

    seq2acc, acc_len, acc_nseq, acc_label = build_reference_index()
    print(f"references: {len(acc_len)} genomes, {len(seq2acc)} sequences, "
          f"{sum(acc_len.values())/1e6:.1f} Mbp")

    comp = pd.read_csv(MES / "composition_S1.tsv", sep="\t")
    comp["accession"] = comp["Genbank Assembly Accession ID"].str.strip()
    splits = load_splits()
    for m in ("1", "2", "3"):
        comp[f"mock{m}_pct"] = pd.to_numeric(comp[f"MOCK{m} %  genomes"],
                                             errors="coerce")
    mock1_accs = set(comp.loc[comp["mock1_pct"].notna(), "accession"])
    print(f"MOCK1 organisms: {len(mock1_accs)}")

    ridx = comp[["accession", "Organism Name GTDB release 07-RS207", "Strain Name",
                 "Kingdom", "Phylum GTBD", "Genome Status", "GC %",
                 "mock1_pct", "mock2_pct", "mock3_pct"]].copy()
    ridx.columns = ["accession", "organism", "strain", "kingdom", "gtdb_phylum",
                    "genome_status", "gc_pct", "mock1_pct", "mock2_pct", "mock3_pct"]
    ridx["ref_len"] = ridx["accession"].map(acc_len)
    ridx["ref_nseq"] = ridx["accession"].map(acc_nseq)
    ridx["ref_label"] = ridx["accession"].map(acc_label)
    ridx["split"] = ridx["accession"].map(
        lambda a: splits.get(a, splits.get(a.split(".")[0], "none")))
    ridx["in_mock1"] = ridx["accession"].isin(mock1_accs)
    ridx["leakage_free"] = ~ridx["split"].isin(["train", "val"])
    ridx.to_csv(OUT / "reference_index.tsv", sep="\t", index=False)
    print("split counts (all 91):", ridx["split"].value_counts().to_dict())
    print("split counts (MOCK1 71):",
          ridx[ridx.in_mock1]["split"].value_counts().to_dict())
    print(f"leakage-free (not train/val): all91={int(ridx.leakage_free.sum())}, "
          f"mock1={int((ridx.leakage_free & ridx.in_mock1).sum())}")

    all_rows, qc_rows = [], []
    for tech in ASSEMBLIES:
        paf = run_minimap2(tech, args.threads, force=args.force)
        rows, qc = process_assembly(tech, paf, seq2acc, acc_len, mock1_accs,
                                    args.threads, force=args.force)
        all_rows.extend(rows)
        qc_rows.append(qc)
        print(f"  [{tech}] bins={qc['n_bins']} (MOCK1 {qc['n_bins_mock1']}), "
              f"unassigned {qc['unassigned_bp_pct']}% of bp, "
              f"non-MOCK1 bin bp={qc['bp_in_nonmock1_bins']/1e6:.2f} Mbp")

    truth = pd.DataFrame(all_rows)
    meta_cols = ["accession", "organism", "kingdom", "gtdb_phylum", "genome_status",
                 "gc_pct", "mock1_pct", "split", "leakage_free", "ref_nseq"]
    truth = truth.merge(ridx[meta_cols], on="accession", how="left")
    truth = truth.sort_values(["assembly", "true_completeness"], ascending=[True, False])
    truth.to_csv(OUT / "bin_truth.tsv", sep="\t", index=False)
    pd.DataFrame(qc_rows).to_csv(OUT / "alignment_qc.tsv", sep="\t", index=False)

    m1 = truth[truth.in_mock1]
    print(f"\nwrote {OUT/'bin_truth.tsv'}  ({len(truth)} bins, {len(m1)} on MOCK1 refs)")
    print("\nMOCK1 bins with a FASTA written (>= %d bp):" % MIN_BIN_BP)
    print(m1[m1.bin_fasta != ""].groupby("assembly").agg(
        n=("accession", "size"),
        comp_med=("true_completeness", "median"),
        comp_ge50=("true_completeness", lambda s: int((s >= 50).sum())),
        cont_med=("true_contamination", "median"),
        cont_max=("true_contamination", "max"),
        contup_med=("true_contamination_upper", "median")).to_string())

    with open(OUT / "step1_summary.json", "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "assemblies": ASSEMBLIES, "min_bin_bp": MIN_BIN_BP,
                   "n_references": len(acc_len), "n_mock1": len(mock1_accs),
                   "qc": qc_rows}, f, indent=2, default=str)
    print("DONE")


if __name__ == "__main__":
    main()
