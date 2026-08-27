#!/usr/bin/env python3
"""
WS3.6b / Track A3 — ground truth for NCBI Tier-1 (same-BioSample) draft/complete pairs.

For every pair the draft assembly is aligned to its own complete reference with
minimap2. Same BioSample means the same physical DNA isolate, so draft sequence absent
from the complete assembly is assembly artefact or genuine contamination rather than
strain divergence.

Ground truth in MAGICC's denominator convention (identical to Tracks A1/A2):
    completeness  = |union of complete-reference bp covered by the draft|
                    / complete reference FULL length x 100
    contamination = draft bp that align nowhere on the complete reference
                    / complete reference FULL length x 100

STATED LIMITATION (protocol §8): the unaligned fraction conflates true contamination
with (i) accessory/plasmid content absent from the chosen complete assembly and
(ii) assembly artefact. It is therefore reported as an UPPER BOUND on contamination.
Tier 1 (same BioSample) minimises (i). This also serves WS4.4 (reference-anchored
unexplained-sequence analysis).

Per-pair checkpoints in <out>/checkpoints/<draft_acc>.json make the run collapse-safe
and resumable.

Usage:
    python scripts/140_ncbi_pair_truth.py --cohort pilot --workers 6
    python scripts/140_ncbi_pair_truth.py --cohort full  --workers 12
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
PAIRS = ROOT / "results" / "revision" / "real_data" / "ncbi_pairs"
DEST = ROOT / "data" / "real_data" / "ncbi_pairs"
MINIMAP2 = "/path/to/conda/envs/magicc2/bin/minimap2"


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


def fasta_stats(path):
    """total bp, n contigs, N50, and per-contig length."""
    lens, cur = [], 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    lens.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
    if cur:
        lens.append(cur)
    total = sum(lens)
    lens.sort(reverse=True)
    cum, n50 = 0, 0
    for L in lens:
        cum += L
        if cum >= total / 2:
            n50 = L
            break
    return total, len(lens), n50


def one_pair(rec):
    draft_acc, complete_acc, ckpt_dir = rec
    ck = Path(ckpt_dir) / f"{draft_acc}.json"
    if ck.exists():
        try:
            return json.loads(ck.read_text())
        except Exception:  # noqa: BLE001
            ck.unlink(missing_ok=True)
    d = DEST / "drafts" / f"{draft_acc}.fna.gz"
    c = DEST / "completes" / f"{complete_acc}.fna.gz"
    if not d.exists() or not c.exists():
        out = {"draft_accession": draft_acc, "complete_accession": complete_acc,
               "status": "missing_file"}
        ck.write_text(json.dumps(out))
        return out
    try:
        draft_bp, draft_contigs, draft_n50 = fasta_stats(d)
        ref_bp, ref_seqs, _ = fasta_stats(c)
        with tempfile.NamedTemporaryFile(suffix=".paf", delete=False) as tf:
            paf = tf.name
        r = subprocess.run([MINIMAP2, "-x", "asm10", "-t", "1", "--secondary=no",
                            "-o", paf, str(c), str(d)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            os.unlink(paf)
            out = {"draft_accession": draft_acc, "complete_accession": complete_acc,
                   "status": f"minimap2_rc{r.returncode}"}
            ck.write_text(json.dumps(out))
            return out
        t_iv = defaultdict(list)
        q_iv = defaultdict(list)
        with open(paf) as f:
            for line in f:
                p = line.split("\t", 10)
                q_iv[p[0]].append((int(p[2]), int(p[3])))
                t_iv[p[5]].append((int(p[7]), int(p[8])))
        os.unlink(paf)
        covered = sum(merge_len(v) for v in t_iv.values())
        aligned_q = sum(merge_len(v) for v in q_iv.values())
        unaligned = draft_bp - aligned_q
        out = {
            "draft_accession": draft_acc, "complete_accession": complete_acc,
            "status": "ok", "draft_bp": draft_bp, "draft_contigs": draft_contigs,
            "draft_n50": draft_n50, "ref_bp": ref_bp, "ref_seqs": ref_seqs,
            "covered_ref_bp": covered, "aligned_draft_bp": aligned_q,
            "unaligned_draft_bp": unaligned,
            "true_completeness": round(100.0 * covered / ref_bp, 6),
            "true_contamination_upper": round(100.0 * unaligned / ref_bp, 6),
        }
    except Exception as e:  # noqa: BLE001
        out = {"draft_accession": draft_acc, "complete_accession": complete_acc,
               "status": f"error:{e}"}
    ck.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["pilot", "full"], default="full")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    coh = PAIRS / ("cohort_pilot.tsv" if args.cohort == "pilot" else "cohort_full.tsv")
    sel = pd.read_csv(coh, sep="\t", dtype=str)
    ckpt = PAIRS / f"checkpoints_{args.cohort}"
    ckpt.mkdir(parents=True, exist_ok=True)
    print(f"cohort={args.cohort}: {len(sel)} pairs, "
          f"{sel['species_taxid'].nunique()} species, workers={args.workers}")

    jobs = [(r["draft_accession"], r["complete_accession"], str(ckpt))
            for _, r in sel.iterrows()]
    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one_pair, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if i % 100 == 0 or i == len(futs):
                bad = sum(1 for x in res if x["status"] != "ok")
                print(f"  {i}/{len(futs)}  bad={bad}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(res)
    meta = sel[["draft_accession", "complete_accession", "organism", "group",
                "species_taxid", "strain", "draft_level", "complete_level",
                "draft_biosample", "draft_excluded_from_refseq",
                "size_ratio_draft_over_complete"]]
    df = df.merge(meta, on=["draft_accession", "complete_accession"], how="left")
    out = PAIRS / f"pair_truth_{args.cohort}.tsv"
    df.to_csv(out, sep="\t", index=False)
    ok = df[df.status == "ok"]
    print(f"\nwrote {out}  ({len(df)} rows, {len(ok)} ok)")
    if len(ok):
        print(ok[["true_completeness", "true_contamination_upper",
                  "draft_contigs", "draft_n50"]].describe().to_string())
    with open(PAIRS / f"pair_truth_{args.cohort}_summary.json", "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "cohort": args.cohort, "n": int(len(df)), "n_ok": int(len(ok)),
                   "aligner": "minimap2 -x asm10 --secondary=no (draft -> complete)"},
                  f, indent=2)


if __name__ == "__main__":
    main()
