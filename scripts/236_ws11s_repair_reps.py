#!/usr/bin/env python3
"""
WS11.S -- close the coverage gap in the representatives census.

The tar stream can lose members to an interrupted batch.  This script recomputes the
authoritative list of SPIRE 95 %-ANI MAG representatives that are in the analysis frame,
finds any that carry no MAGICC prediction, and fetches and scores exactly those over the
per-genome endpoint, so `reps_census` really is a census and its coverage is verifiable.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

ROOT = Path("/path/to/magicc")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/revision/ws11/spire_catalogue"
STAGE = ROOT / "data/real_data/spire/reps_stage"
REPS = ROOT / "data/real_data/spire/spire_v1_representatives.tsv.gz"
META = ROOT / "data/real_data/spire/spire_v1_genome_metadata.tsv.gz"
ONNX_MODEL = ROOT / "models/magicc_v5.onnx"
ONNX_SHA256 = "b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096"
SELECTED_KMERS = str(ROOT / "data/kmer_selection/selected_kmers.txt")
NORMALIZATION = str(ROOT / "data/features/normalization_params.json")
URL = "https://spire.embl.de/download_file/{gid}"
UA = "magicc-revision/1.0 (academic revision analysis; tianrenmao@gmail.com)"


def rep_ids_in_frame() -> set[str]:
    reps = set()
    with gzip.open(REPS, "rt") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].startswith("spire_mag_"):
                reps.add(p[1])
    frame = set()
    with gzip.open(META, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["genome_id"] in reps:
                try:
                    float(r["genome_size"]); float(r["completeness"]); float(r["contamination"])
                except ValueError:
                    continue
                frame.add(r["genome_id"])
    return frame


_kc = None


def _init(p):
    global _kc
    from magicc.kmer_counter import KmerCounter
    _kc = KmerCounter(p); _kc.count_contigs(["ACGTACGTACGTACGTACGT" * 50])


def _feat(gid):
    from magicc.assembly_stats import compute_assembly_stats
    try:
        contigs, cur = [], []
        with gzip.open(STAGE / f"{gid}.fa.gz", "rt") as fh:
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
        return gid, kc.astype(np.float32), \
            compute_assembly_stats(_kc.total_kmer_count(kc), kc).astype(np.float32), ""
    except Exception as exc:                                          # noqa: BLE001
        return gid, None, None, f"{type(exc).__name__}: {exc}"


def fetch(gid):
    dest = STAGE / f"{gid}.fa.gz"
    for a in range(3):
        try:
            req = Request(URL.format(gid=gid), headers={"User-Agent": UA})
            with urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) < 200:
                raise ValueError("short")
            dest.write_bytes(data)
            return gid, "ok", hashlib.sha256(data).hexdigest(), len(data)
        except HTTPError as e:
            if e.code in (404, 410):
                return gid, f"HTTP{e.code}", "", 0
        except (URLError, OSError, ValueError, TimeoutError):
            pass
        time.sleep(2 * (a + 1))
    return gid, "FAILED", "", 0


def main() -> int:
    if hashlib.sha256(ONNX_MODEL.read_bytes()).hexdigest() != ONNX_SHA256:
        print("[236] FATAL model SHA256 mismatch", file=sys.stderr); return 2
    STAGE.mkdir(parents=True, exist_ok=True)
    want = rep_ids_in_frame()
    have = set()
    p = OUT / "reps_predictions.tsv"
    if p.exists():
        with p.open() as fh:
            next(fh)
            for line in fh:
                have.add(line.split("\t", 1)[0])
    missing = sorted(want - have)
    print(f"[236] representatives in frame {len(want):,}; scored {len(want & have):,}; "
          f"missing {len(missing):,}", flush=True)
    if not missing:
        (OUT / "reps_coverage.tsv").write_text(
            f"rep_mags_in_frame\tscored\tmissing\n{len(want)}\t{len(want & have)}\t0\n")
        return 0
    pf = p.open("a"); cf = (OUT / "reps_checksums.tsv").open("a")
    import onnxruntime as ort
    from magicc.normalization import FeatureNormalizer
    norm = FeatureNormalizer.load(NORMALIZATION)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX_MODEL), so, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]; out0 = sess.get_outputs()[0].name
    pool = Pool(processes=3, initializer=_init, initargs=(SELECTED_KMERS,))
    n404 = 0
    try:
        for s in range(0, len(missing), 1000):
            b = missing[s:s + 1000]
            sha = {}
            with ThreadPoolExecutor(max_workers=12) as tp:
                for f in as_completed([tp.submit(fetch, g) for g in b]):
                    gid, stt, h, nb = f.result()
                    if stt == "ok":
                        sha[gid] = (h, nb)
                    elif stt.startswith("HTTP4"):
                        n404 += 1
            present = [g for g in b if (STAGE / f"{g}.fa.gz").exists()]
            gids, ks, as_ = [], [], []
            for gid, k, a, err in pool.imap_unordered(_feat, present, chunksize=8):
                if not err:
                    gids.append(gid); ks.append(k); as_.append(a)
            if gids:
                kn = norm.normalize_kmer(np.stack(ks)).astype(np.float32)
                an = norm.normalize_assembly(np.stack(as_)).astype(np.float32)
                pr = np.zeros((len(gids), 2), np.float32)
                for b0 in range(0, len(gids), 256):
                    b1 = min(b0 + 256, len(gids))
                    pr[b0:b1] = sess.run([out0], {ins[0]: kn[b0:b1], ins[1]: an[b0:b1]})[0]
                for j, gid in enumerate(gids):
                    pf.write(f"{gid}\t{pr[j,0]:.4f}\t{pr[j,1]:.4f}\n")
                    h, nb = sha.get(gid, ("", 0)); cf.write(f"{gid}\t{h}\t{nb}\n")
            for g in present:
                try:
                    (STAGE / f"{g}.fa.gz").unlink()
                except OSError:
                    pass
            pf.flush(); cf.flush()
            print(f"[236] {min(s+1000, len(missing))}/{len(missing)} repaired", flush=True)
    finally:
        pool.close(); pool.join(); pf.close(); cf.close()
    have = set()
    with p.open() as fh:
        next(fh)
        for line in fh:
            have.add(line.split("\t", 1)[0])
    (OUT / "reps_coverage.tsv").write_text(
        "rep_mags_in_frame\tscored\tmissing\tmissing_http404\n"
        f"{len(want)}\t{len(want & have)}\t{len(want - have)}\t{n404}\n")
    print(f"[236] final coverage {len(want & have):,}/{len(want):,} "
          f"({100*len(want & have)/len(want):.3f} %); {n404} HTTP404", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
