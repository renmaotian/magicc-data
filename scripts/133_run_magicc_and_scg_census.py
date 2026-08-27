#!/usr/bin/env python3
"""
WS3.3 / new-priority — run MAGICC V5 on the reduced-genome cohorts and, for the
mechanism test, count how many single-copy core-gene families are *detectable* in each
genome.

Two quantities per genome:
  (a) MAGICC V5 predicted completeness / contamination.
      Direct ONNX Runtime, following scripts/56 and scripts/75 exactly:
        FASTA -> selected-9-mer counts -> 7 k-mer-summary features -> normalisation
              -> models/magicc_v5.onnx
      The `magicc` CLI is deliberately not used (it is being edited concurrently).
  (b) n_scg_detected: distinct core-gene families hit by ``hmmsearch --cut_tc`` on
      Prodigal proteins, against 85_bcg.hmm (bacteria) or uacg.hmm (archaea).

Why (b): genome size and the number of detectable core genes are confounded.  If the
MAGICC-vs-CheckM2 delta tracks n_scg_detected after conditioning on genome size, the
mechanism sits in the feature space (fewer core-gene 9-mers to observe).  If it tracks
raw size instead, the mechanism is the k-mer-count scale itself.

Threads are capped at 6 while the GPU holdout retrain occupies the machine.

Outputs (results/revision/real_data/reduced_genome/)
  magicc_v5_predictions.tsv
  scg_census.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import onnxruntime as ort                                            # noqa: E402
from magicc.assembly_stats import compute_assembly_stats             # noqa: E402
from magicc.kmer_counter import KmerCounter                          # noqa: E402
from magicc.normalization import FeatureNormalizer                   # noqa: E402

OUTDIR = PROJECT_DIR / "results/revision/real_data/reduced_genome"
COHORTS = OUTDIR / "cohorts.tsv"
GENOMES = PROJECT_DIR / "data/real_data/reduced_genome/genomes"

SELECTED_KMERS = str(PROJECT_DIR / "data/kmer_selection/selected_kmers.txt")
NORMALIZATION = str(PROJECT_DIR / "data/features/normalization_params.json")
ONNX_MODEL = str(PROJECT_DIR / "models/magicc_v5.onnx")
BACT_HMM = PROJECT_DIR / "85_bcg.hmm"
ARCH_HMM = PROJECT_DIR / "uacg.hmm"

N_WORKERS = 6           # machine is loaded; protocol cap
BATCH_SIZE = 64
ONNX_THREADS = 1


# ----------------------------------------------------------------- FASTA I/O
def read_contigs(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    contigs, cur = [], []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    contigs.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        contigs.append("".join(cur).upper())
    return [c for c in contigs if c]


# ------------------------------------------------------------ MAGICC features
_kc = None


def _init_kmer(kmers_path: str) -> None:
    global _kc
    _kc = KmerCounter(kmers_path)
    _kc.count_contigs(["ACGTACGTACGTACGTACGT" * 50])       # warm the JIT


def _feat_worker(args):
    idx, path = args
    try:
        contigs = read_contigs(Path(path))
        if not contigs:
            return idx, None, None, "no_contigs"
        kc = _kc.count_contigs(contigs)
        asm = compute_assembly_stats(_kc.total_kmer_count(kc), kc)
        return idx, kc.astype(np.float32), asm.astype(np.float32), None
    except Exception as exc:                                          # noqa: BLE001
        return idx, None, None, f"{type(exc).__name__}: {exc}"


def run_magicc(rows: list[dict]) -> None:
    out = OUTDIR / "magicc_v5_predictions.tsv"
    work = []
    for i, r in enumerate(rows):
        p = GENOMES / f"{r['ncbi_accession']}.fna.gz"
        if p.exists():
            work.append((i, str(p)))
    print(f"[133] MAGICC on {len(work)}/{len(rows)} genomes with a FASTA present", flush=True)

    t0 = time.time()
    got: dict[int, tuple] = {}
    with Pool(processes=N_WORKERS, initializer=_init_kmer,
              initargs=(SELECTED_KMERS,)) as pool:
        for n, (idx, kmer, asm, err) in enumerate(
                pool.imap_unordered(_feat_worker, work, chunksize=4), 1):
            if err:
                print(f"[133]   WARNING idx={idx}: {err}", flush=True)
            else:
                got[idx] = (kmer, asm)
            if n % 200 == 0:
                print(f"[133]   features {n}/{len(work)}", flush=True)
    feat_t = time.time() - t0
    idxs = sorted(got)
    n = len(idxs)
    print(f"[133] features: {feat_t:.1f}s ({feat_t / max(1, n) * 1000:.1f} ms/genome, "
          f"{N_WORKERS} workers)", flush=True)

    normalizer = FeatureNormalizer.load(NORMALIZATION)
    kn = normalizer.normalize_kmer(np.stack([got[i][0] for i in idxs])).astype(np.float32)
    an = normalizer.normalize_assembly(
        np.stack([got[i][1] for i in idxs])).astype(np.float32)

    so = ort.SessionOptions()
    so.intra_op_num_threads = ONNX_THREADS
    so.inter_op_num_threads = ONNX_THREADS
    sess = ort.InferenceSession(ONNX_MODEL, so, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name

    preds = np.zeros((n, 2), dtype=np.float32)
    t1 = time.time()
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        preds[s:e] = sess.run([out_name], {in_names[0]: kn[s:e], in_names[1]: an[s:e]})[0]
    print(f"[133] inference: {time.time() - t1:.2f}s", flush=True)

    cols = list(rows[0].keys()) + ["magicc_completeness", "magicc_contamination", "model"]
    with out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for j, i in enumerate(idxs):
            r = dict(rows[i])
            r["magicc_completeness"] = f"{preds[j, 0]:.4f}"
            r["magicc_contamination"] = f"{preds[j, 1]:.4f}"
            r["model"] = "magicc_v5.onnx"
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[133] wrote {out}", flush=True)


# ------------------------------------------------------------------ SCG census
def _scg_worker(args):
    acc, path, domain = args
    tmp = tempfile.mkdtemp(prefix="scg_")
    try:
        fa = os.path.join(tmp, "g.fna")
        with gzip.open(path, "rt") as src, open(fa, "w") as dst:
            shutil.copyfileobj(src, dst)
        prot = os.path.join(tmp, "p.faa")
        r = subprocess.run(
            ["prodigal", "-i", fa, "-a", prot, "-o", os.path.join(tmp, "g.gff"),
             "-p", "meta", "-q"],
            capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(prot):
            return {"accession": acc, "error": f"prodigal: {r.stderr[:120]}"}
        n_prot = sum(1 for line in open(prot) if line.startswith(">"))

        hmm = ARCH_HMM if domain == "Archaea" else BACT_HMM
        tbl = os.path.join(tmp, "h.tbl")
        r = subprocess.run(
            ["hmmsearch", "--cut_tc", "--tblout", tbl, "--noali", "--cpu", "1",
             str(hmm), prot],
            capture_output=True, text=True)
        if r.returncode != 0:
            return {"accession": acc, "error": f"hmmsearch: {r.stderr[:120]}"}
        fams, hits = set(), 0
        with open(tbl) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                p = line.split()
                if len(p) < 3:
                    continue
                fams.add(p[2])          # query (profile) name
                hits += 1
        return {"accession": acc, "domain": domain, "n_proteins": n_prot,
                "n_scg_detected": len(fams), "n_scg_hits": hits,
                "n_scg_profiles": 128 if domain == "Archaea" else 85, "error": ""}
    except Exception as exc:                                          # noqa: BLE001
        return {"accession": acc, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_scg(rows: list[dict], limit: int | None) -> None:
    out = OUTDIR / "scg_census.tsv"
    done: set[str] = set()
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if not r.get("error"):
                    done.add(r["accession"])
        print(f"[133] SCG census: {len(done)} already done, resuming", flush=True)

    work = []
    for r in rows:
        acc = r["ncbi_accession"]
        p = GENOMES / f"{acc}.fna.gz"
        if p.exists() and acc not in done:
            work.append((acc, str(p), r["domain"]))
    if limit:
        work = work[:limit]
    print(f"[133] SCG census on {len(work)} genomes, {N_WORKERS} workers", flush=True)
    if not work:
        return

    cols = ["accession", "domain", "n_proteins", "n_scg_detected", "n_scg_hits",
            "n_scg_profiles", "error"]
    new = not out.exists()
    t0 = time.time()
    with out.open("a") as fh:
        if new:
            fh.write("\t".join(cols) + "\n")
        with Pool(processes=N_WORKERS) as pool:
            for n, res in enumerate(pool.imap_unordered(_scg_worker, work, chunksize=1), 1):
                fh.write("\t".join(str(res.get(c, "")) for c in cols) + "\n")
                if n % 50 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"[133]   scg {n}/{len(work)} ({el / n:.1f}s/genome, "
                          f"eta {(len(work) - n) * el / n / 60:.0f} min)", flush=True)
    print(f"[133] wrote {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-magicc", action="store_true")
    ap.add_argument("--skip-scg", action="store_true")
    ap.add_argument("--scg-limit", type=int, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(COHORTS.open(), delimiter="\t"))
    print(f"[133] cohorts: {len(rows)} genomes", flush=True)
    if not args.skip_magicc:
        run_magicc(rows)
    if not args.skip_scg:
        run_scg(rows, args.scg_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
