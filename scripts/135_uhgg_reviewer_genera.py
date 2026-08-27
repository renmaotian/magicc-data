#!/usr/bin/env python3
"""
WS3.3 — secondary cohort: run MAGICC V5 on the UHGG genomes of the reviewer-named
genera that UHGG actually contains (CAG-557, UMGS1491, HGM10766).

Why this is only a secondary cohort: UHGG v2.0.2 ships **CheckM v1** completeness and
contamination, not CheckM2, so a MAGICC-vs-UHGG comparison is not the CheckM2
comparison Reviewer 2 describes.  It is run anyway because (i) UHGG is the most likely
source of the reviewer's exploratory analysis and (ii) its MAGs are markedly more
fragmented than the GTDB genomes of the same genera, which lets us test whether the
magnitude of the disagreement grows with fragmentation.

UHGG genomes are distributed as Prokka GFF with an embedded ``##FASTA`` block; the
nucleotide FASTA is extracted from it.

Outputs (results/revision/real_data/reduced_genome/)
  uhgg_reviewer_genera_predictions.tsv
  uhgg_reviewer_genera_summary.json
"""

from __future__ import annotations

import csv
import gzip
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import onnxruntime as ort                                            # noqa: E402
from magicc.assembly_stats import compute_assembly_stats             # noqa: E402
from magicc.kmer_counter import KmerCounter                          # noqa: E402
from magicc.normalization import FeatureNormalizer                   # noqa: E402

META = PROJECT_DIR / "data/real_data/uhgg/genomes-all_metadata.tsv"
FASTA_DIR = PROJECT_DIR / "data/real_data/uhgg/reviewer_genera_fasta"
OUTDIR = PROJECT_DIR / "results/revision/real_data/reduced_genome"

SELECTED_KMERS = str(PROJECT_DIR / "data/kmer_selection/selected_kmers.txt")
NORMALIZATION = str(PROJECT_DIR / "data/features/normalization_params.json")
ONNX_MODEL = str(PROJECT_DIR / "models/magicc_v5.onnx")

GENERA = ["CAG-557", "UMGS1491", "HGM10766"]
MAX_WORKERS = 6
UA = "magicc-revision/1.0 (research use; contact tianrenmao@gmail.com)"


def select() -> list[dict]:
    out = []
    with META.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            for g in GENERA:
                if f";g__{g};" in row["Lineage"]:
                    row["genus"] = g
                    out.append(row)
                    break
    return out


def fetch_one(row: dict) -> dict:
    acc = row["Genome"]
    dest = FASTA_DIR / f"{acc}.fna.gz"
    if dest.exists() and dest.stat().st_size > 1000:
        return {"genome": acc, "status": "cached"}
    url = row["FTP_download"].replace("ftp://", "https://")
    try:
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=180) as resp:
            raw = gzip.decompress(resp.read()).decode("utf-8", "replace")
    except Exception as exc:                                          # noqa: BLE001
        return {"genome": acc, "status": f"FAILED {type(exc).__name__}: {exc}"}
    if "##FASTA" not in raw:
        return {"genome": acc, "status": "FAILED no ##FASTA block"}
    fasta = raw.split("##FASTA", 1)[1].lstrip("\n")
    with gzip.open(dest, "wt") as out:
        out.write(fasta)
    return {"genome": acc, "status": "ok"}


_kc = None


def _init(kmers_path: str) -> None:
    global _kc
    _kc = KmerCounter(kmers_path)
    _kc.count_contigs(["ACGTACGTACGTACGTACGT" * 50])


def _feat(args):
    idx, path = args
    try:
        contigs, cur = [], []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith(">"):
                    if cur:
                        contigs.append("".join(cur).upper())
                        cur = []
                else:
                    cur.append(line.strip())
        if cur:
            contigs.append("".join(cur).upper())
        contigs = [c for c in contigs if c]
        if not contigs:
            return idx, None, None, "no_contigs"
        kc = _kc.count_contigs(contigs)
        return idx, kc.astype(np.float32), \
            compute_assembly_stats(_kc.total_kmer_count(kc), kc).astype(np.float32), None
    except Exception as exc:                                          # noqa: BLE001
        return idx, None, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = select()
    print(f"[135] UHGG genomes in {GENERA}: {len(rows)}", flush=True)

    ok = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for fut in as_completed({pool.submit(fetch_one, r): r for r in rows}):
            res = fut.result()
            if res["status"].startswith("FAILED"):
                print(f"[135]   {res['genome']}: {res['status']}", flush=True)
            else:
                ok += 1
    print(f"[135] fetched/cached {ok}/{len(rows)}", flush=True)

    work = [(i, str(FASTA_DIR / f"{r['Genome']}.fna.gz"))
            for i, r in enumerate(rows)
            if (FASTA_DIR / f"{r['Genome']}.fna.gz").exists()]
    got = {}
    t0 = time.time()
    with Pool(processes=MAX_WORKERS, initializer=_init, initargs=(SELECTED_KMERS,)) as pool:
        for idx, k, a, err in pool.imap_unordered(_feat, work, chunksize=4):
            if err:
                print(f"[135]   WARNING idx={idx}: {err}", flush=True)
            else:
                got[idx] = (k, a)
    idxs = sorted(got)
    n = len(idxs)
    print(f"[135] features for {n} genomes in {time.time() - t0:.1f}s", flush=True)

    norm = FeatureNormalizer.load(NORMALIZATION)
    kn = norm.normalize_kmer(np.stack([got[i][0] for i in idxs])).astype(np.float32)
    an = norm.normalize_assembly(np.stack([got[i][1] for i in idxs])).astype(np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(ONNX_MODEL, so, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    preds = sess.run([sess.get_outputs()[0].name], {ins[0]: kn, ins[1]: an})[0]

    cols = ["Genome", "genus", "Genome_type", "Length", "N_contigs", "N50", "GC_content",
            "Completeness", "Contamination", "Lineage",
            "magicc_completeness", "magicc_contamination"]
    out = OUTDIR / "uhgg_reviewer_genera_predictions.tsv"
    recs = []
    with out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for j, i in enumerate(idxs):
            r = dict(rows[i])
            r["magicc_completeness"] = float(preds[j, 0])
            r["magicc_contamination"] = float(preds[j, 1])
            recs.append(r)
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    summary = {
        "script": "scripts/135_uhgg_reviewer_genera.py",
        "catalogue": "MGnify human-gut v2.0.2 (UHGG)",
        "comparator_caveat": "UHGG ships CheckM v1 values, NOT CheckM2",
        "n": len(recs),
        "per_genus": [],
    }
    for g in GENERA:
        sub = [r for r in recs if r["genus"] == g]
        if not sub:
            continue
        summary["per_genus"].append({
            "genus": g, "n": len(sub),
            "length_median": int(np.median([int(r["Length"]) for r in sub])),
            "contigs_median": int(np.median([int(r["N_contigs"]) for r in sub])),
            "uhgg_checkm1_completeness_median": round(
                float(np.median([float(r["Completeness"]) for r in sub])), 2),
            "uhgg_checkm1_contamination_median": round(
                float(np.median([float(r["Contamination"]) for r in sub])), 2),
            "magicc_completeness_median": round(
                float(np.median([r["magicc_completeness"] for r in sub])), 2),
            "magicc_contamination_median": round(
                float(np.median([r["magicc_contamination"] for r in sub])), 2),
            "delta_completeness_MAGICC_minus_CheckM1_median": round(
                float(np.median([r["magicc_completeness"] - float(r["Completeness"])
                                 for r in sub])), 2),
            "delta_contamination_MAGICC_minus_CheckM1_median": round(
                float(np.median([r["magicc_contamination"] - float(r["Contamination"])
                                 for r in sub])), 2),
            "pct_magicc_cont_gt_5": round(
                100 * float(np.mean([r["magicc_contamination"] > 5 for r in sub])), 1),
        })
    (OUTDIR / "uhgg_reviewer_genera_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
