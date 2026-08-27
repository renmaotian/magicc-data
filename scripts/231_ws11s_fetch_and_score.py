#!/usr/bin/env python3
"""
WS11.S -- fetch the sampled SPIRE MAGs and score them with the frozen magicc_v5.onnx.

Streaming, resumable and collapse-safe:
  * work is done in batches; after each batch the predictions, checksums and fetch status
    are flushed to disk and the FASTAs are DELETED, so peak disk is one batch
    (~1.5 GB) and an interrupted run resumes exactly where it stopped;
  * genomes are processed in an order that keeps the design valid under truncation --
    S1_small (census), S2_rare_phylum (census), then S3_main in SRS order, so any prefix
    of S3 that has been reached is still a simple random sample of S3;
  * the model SHA256 is verified against the frozen value BEFORE any inference.

SPIRE serves one gzipped FASTA per genome at https://spire.embl.de/download_file/<id>.
A stable ~3.1 % of catalogue ids return HTTP 404 server-side (measured 123/4,000 on a
seeded random probe, and 21/612 on the earlier five-genus cohort); those are recorded as
non-response, never silently dropped.

Usage: 231_ws11s_fetch_and_score.py [--workers N] [--feat N] [--limit N]

Outputs (results/revision/ws11/spire_catalogue/)
  magicc_predictions.tsv      appended per batch (gzipped by script 233)
  fetch_status.tsv            per-genome HTTP outcome (ok / HTTP404 / FAILED ...)
  checksums.tsv               sha256 + byte size of every FASTA actually scored
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/revision/ws11/spire_catalogue"
STAGE = ROOT / "data/real_data/spire/catalogue_srs_fasta"
COHORT = OUT / "cohort_definition.tsv.gz"
PRED = OUT / "magicc_predictions.tsv"
STATUS = OUT / "fetch_status.tsv"
SUMS = OUT / "checksums.tsv"

ONNX_MODEL = ROOT / "models/magicc_v5.onnx"
ONNX_SHA256 = "b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096"
SELECTED_KMERS = str(ROOT / "data/kmer_selection/selected_kmers.txt")
NORMALIZATION = str(ROOT / "data/features/normalization_params.json")

URL = "https://spire.embl.de/download_file/{gid}"
UA = "magicc-revision/1.0 (academic revision analysis; tianrenmao@gmail.com)"
BATCH = 1500
RETRIES = 3


# --------------------------------------------------------------------- fetch --
def fetch_one(gid: str) -> tuple[str, str, int, str]:
    dest = STAGE / f"{gid}.fa.gz"
    if dest.exists() and dest.stat().st_size > 1000:
        return gid, "cached", dest.stat().st_size, ""
    last = ""
    for attempt in range(RETRIES):
        try:
            req = Request(URL.format(gid=gid), headers={"User-Agent": UA})
            with urlopen(req, timeout=180) as resp:
                data = resp.read()
            if len(data) < 200:
                raise ValueError(f"short body {len(data)}")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
            return gid, "ok", len(data), hashlib.sha256(data).hexdigest()
        except HTTPError as exc:
            if exc.code in (404, 410):
                return gid, f"HTTP{exc.code}", 0, ""
            last = f"HTTP{exc.code}"
        except (URLError, OSError, ValueError, TimeoutError) as exc:  # noqa: BLE001
            last = type(exc).__name__
        time.sleep(2 * (attempt + 1))
    return gid, f"FAILED {last}", 0, ""


# ------------------------------------------------------------------ features --
_kc = None


def _init(kmers_path: str) -> None:
    global _kc
    from magicc.kmer_counter import KmerCounter
    _kc = KmerCounter(kmers_path)
    _kc.count_contigs(["ACGTACGTACGTACGTACGT" * 50])


def _feat(gid: str):
    from magicc.assembly_stats import compute_assembly_stats
    path = STAGE / f"{gid}.fa.gz"
    try:
        contigs, cur = [], []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith(">"):
                    if cur:
                        contigs.append("".join(cur).upper()); cur = []
                else:
                    cur.append(line.strip())
        if cur:
            contigs.append("".join(cur).upper())
        contigs = [c for c in contigs if c]
        if not contigs:
            return gid, None, None, "no_contigs"
        kc = _kc.count_contigs(contigs)
        stats = compute_assembly_stats(_kc.total_kmer_count(kc), kc)
        return gid, kc.astype(np.float32), stats.astype(np.float32), ""
    except Exception as exc:                                          # noqa: BLE001
        return gid, None, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16, help="download threads")
    ap.add_argument("--feat", type=int, default=9, help="feature-extraction processes")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new genomes (0 = all)")
    args = ap.parse_args()

    digest = hashlib.sha256(ONNX_MODEL.read_bytes()).hexdigest()
    if digest != ONNX_SHA256:
        print(f"[231] FATAL model SHA256 mismatch: {digest}", file=sys.stderr)
        return 2
    print(f"[231] model SHA256 verified {digest}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)

    order = []
    rank = {"S1_small": 0, "S2_rare_phylum": 1, "S3_main": 2}
    with gzip.open(COHORT, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            order.append((rank[r["stratum"]], int(r["srs_order"] or 0), r["genome_id"]))
    order.sort()
    ids = [g for _, _, g in order]
    print(f"[231] cohort {len(ids):,} genomes", flush=True)

    done = set()
    if PRED.exists():
        with PRED.open() as fh:
            next(fh, None)
            for line in fh:
                done.add(line.split("\t", 1)[0])
    if STATUS.exists():
        with STATUS.open() as fh:
            next(fh, None)
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if p[1].startswith("HTTP4") or p[1].startswith("no_contigs") or p[1].startswith("featfail"):
                    done.add(p[0])
    todo = [g for g in ids if g not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[231] already resolved {len(done):,}; to do {len(todo):,}", flush=True)
    if not todo:
        return 0

    import onnxruntime as ort
    from magicc.normalization import FeatureNormalizer
    norm = FeatureNormalizer.load(NORMALIZATION)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX_MODEL), so, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    out0 = sess.get_outputs()[0].name

    new_pred = not PRED.exists()
    pf = PRED.open("a"); sf = STATUS.open("a"); cf = SUMS.open("a")
    if new_pred:
        pf.write("genome_id\tmagicc_completeness\tmagicc_contamination\n")
        sf.write("genome_id\tstatus\tbytes\n")
        cf.write("genome_id\tsha256\tbytes\n")

    pool = Pool(processes=args.feat, initializer=_init, initargs=(SELECTED_KMERS,))
    t_start = time.time()
    n_done = n_ok = n_404 = n_fail = 0
    try:
        for s in range(0, len(todo), BATCH):
            batch = todo[s:s + BATCH]
            t0 = time.time()
            sha = {}
            with ThreadPoolExecutor(max_workers=args.workers) as tp:
                futs = [tp.submit(fetch_one, g) for g in batch]
                for fut in as_completed(futs):
                    gid, st, nb, h = fut.result()
                    sf.write(f"{gid}\t{st}\t{nb}\n")
                    if st in ("ok", "cached"):
                        n_ok += 1
                        if h:
                            sha[gid] = (h, nb)
                    elif st.startswith("HTTP4"):
                        n_404 += 1
                    else:
                        n_fail += 1
            t_dl = time.time() - t0
            present = [g for g in batch if (STAGE / f"{g}.fa.gz").exists()]
            for g in present:
                if g not in sha:
                    p = STAGE / f"{g}.fa.gz"
                    b = p.read_bytes()
                    sha[g] = (hashlib.sha256(b).hexdigest(), len(b))
            t1 = time.time()
            gids, ks, as_ = [], [], []
            for gid, k, a, err in pool.imap_unordered(_feat, present, chunksize=8):
                if err:
                    sf.write(f"{gid}\tfeatfail:{err}\t0\n")
                else:
                    gids.append(gid); ks.append(k); as_.append(a)
            if gids:
                kn = norm.normalize_kmer(np.stack(ks)).astype(np.float32)
                an = norm.normalize_assembly(np.stack(as_)).astype(np.float32)
                preds = np.zeros((len(gids), 2), np.float32)
                for b0 in range(0, len(gids), 256):
                    b1 = min(b0 + 256, len(gids))
                    preds[b0:b1] = sess.run([out0], {ins[0]: kn[b0:b1], ins[1]: an[b0:b1]})[0]
                for j, gid in enumerate(gids):
                    pf.write(f"{gid}\t{preds[j, 0]:.4f}\t{preds[j, 1]:.4f}\n")
                    h, nb = sha.get(gid, ("", 0))
                    cf.write(f"{gid}\t{h}\t{nb}\n")
            t_sc = time.time() - t1
            for g in present:
                try:
                    (STAGE / f"{g}.fa.gz").unlink()
                except OSError:
                    pass
            pf.flush(); sf.flush(); cf.flush()
            os.fsync(pf.fileno()); os.fsync(sf.fileno()); os.fsync(cf.fileno())
            n_done += len(batch)
            el = time.time() - t_start
            rate = n_done / el
            eta = (len(todo) - n_done) / rate / 3600 if rate > 0 else float("nan")
            print(f"[231] {n_done:,}/{len(todo):,} scored={len(gids)} ok={n_ok} 404={n_404} "
                  f"fail={n_fail} dl={t_dl:.0f}s score={t_sc:.0f}s rate={rate:.1f}/s "
                  f"eta={eta:.2f}h", flush=True)
    finally:
        pool.close(); pool.join()
        pf.close(); sf.close(); cf.close()
    print(f"[231] done in {(time.time() - t_start) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
