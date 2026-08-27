#!/usr/bin/env python3
"""
WS11.S -- second cohort: a COMPLETE CENSUS of SPIRE's dereplicated catalogue.

`spire_representative_genomes.tar` (86,849,024,000 B) holds one representative genome per
95 %-ANI cluster: 14,943 members are proGenomes isolates (`specI_v4_*`, skipped -- they
are not SPIRE MAGs and carry no SPIRE CheckM2 row) and the rest are `spire_mag_*` cluster
representatives.  The archive is name-sorted, so every `spire_mag_*` member lies in one
contiguous run beginning at byte 14,507,918,848 (verified header); this script streams
only that run (72.34 GB) rather than the whole archive.

The tar is parsed from the byte stream directly -- never written to disk in full.  A
checkpoint (byte offset, counters) is flushed at every batch boundary, so a dropped
connection or a killed process resumes at the last completed batch.  Scored FASTAs are
deleted immediately; sha256 and the prediction are kept.

This cohort is a CENSUS of a different population from the per-MAG catalogue sample:
representatives are the best genome of their cluster and are quality-enriched.  Its rate
must never be quoted as the per-MAG catalogue rate.

Outputs (results/revision/ws11/spire_catalogue/)
  reps_predictions.tsv  reps_checksums.tsv  reps_checkpoint.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from urllib.request import Request, urlopen

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/revision/ws11/spire_catalogue"
STAGE = ROOT / "data/real_data/spire/reps_stage"
PRED = OUT / "reps_predictions.tsv"
SUMS = OUT / "reps_checksums.tsv"
CKPT = OUT / "reps_checkpoint.json"

TAR_URL = "https://black.embl.de/~fullam/spire/representatives/spire_representative_genomes.tar"
TAR_BYTES = 86_849_024_000
FIRST_MAG_OFFSET = 14_507_918_848
PREFIX = "spire_representative_genomes/"
UA = "magicc-revision/1.0 (academic revision analysis; tianrenmao@gmail.com)"

ONNX_MODEL = ROOT / "models/magicc_v5.onnx"
ONNX_SHA256 = "b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096"
SELECTED_KMERS = str(ROOT / "data/kmer_selection/selected_kmers.txt")
NORMALIZATION = str(ROOT / "data/features/normalization_params.json")
BATCH = 1500


def header_ok(hdr: bytes) -> bool:
    if len(hdr) != 512 or hdr[257:262] != b"ustar":
        return False
    try:
        want = int(hdr[148:156].split(b"\0")[0].strip() or b"0", 8)
    except ValueError:
        return False
    raw = bytearray(hdr)
    raw[148:156] = b" " * 8
    return sum(raw) == want


class Stream:
    """Resumable forward-only byte reader over an HTTP range request."""

    def __init__(self, url: str, offset: int, end: int):
        self.url, self.pos, self.end, self.fh = url, offset, end, None

    def _open(self) -> None:
        req = Request(self.url, headers={"User-Agent": UA,
                                         "Range": f"bytes={self.pos}-{self.end - 1}"})
        self.fh = urlopen(req, timeout=300)

    def read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            if self.pos >= self.end:
                break
            if self.fh is None:
                for attempt in range(6):
                    try:
                        self._open(); break
                    except Exception as exc:                          # noqa: BLE001
                        print(f"[232]   reconnect {attempt}: {type(exc).__name__}: {exc}", flush=True)
                        time.sleep(5 * (attempt + 1))
                else:
                    raise RuntimeError("cannot reopen stream")
            try:
                chunk = self.fh.read(min(1 << 20, n - len(buf)))
            except Exception as exc:                                  # noqa: BLE001
                print(f"[232]   stream error {type(exc).__name__}: {exc}; reconnecting", flush=True)
                try:
                    self.fh.close()
                except Exception:                                     # noqa: BLE001
                    pass
                self.fh = None
                continue
            if not chunk:
                try:
                    self.fh.close()
                except Exception:                                     # noqa: BLE001
                    pass
                self.fh = None
                if self.pos >= self.end:
                    break
                continue
            buf += chunk
            self.pos += len(chunk)
        return bytes(buf)


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
        return gid, kc.astype(np.float32), \
            compute_assembly_stats(_kc.total_kmer_count(kc), kc).astype(np.float32), ""
    except Exception as exc:                                          # noqa: BLE001
        return gid, None, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", type=int, default=9)
    args = ap.parse_args()

    digest = hashlib.sha256(ONNX_MODEL.read_bytes()).hexdigest()
    if digest != ONNX_SHA256:
        print(f"[232] FATAL model SHA256 mismatch: {digest}", file=sys.stderr)
        return 2
    print(f"[232] model SHA256 verified {digest}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    for p in STAGE.glob("*.fa.gz"):
        p.unlink()

    ck = json.loads(CKPT.read_text()) if CKPT.exists() else {
        "offset": FIRST_MAG_OFFSET, "n_members": 0, "n_scored": 0, "n_skipped": 0}
    if ck["offset"] >= TAR_BYTES:
        print("[232] already complete", flush=True)
        return 0
    print(f"[232] resuming at byte {ck['offset']:,} ({100 * ck['offset'] / TAR_BYTES:.2f} %), "
          f"{ck['n_scored']:,} already scored", flush=True)

    import onnxruntime as ort
    from magicc.normalization import FeatureNormalizer
    norm = FeatureNormalizer.load(NORMALIZATION)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX_MODEL), so, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]; out0 = sess.get_outputs()[0].name

    new = not PRED.exists()
    pf = PRED.open("a"); cf = SUMS.open("a")
    if new:
        pf.write("genome_id\tmagicc_completeness\tmagicc_contamination\n")
        cf.write("genome_id\tsha256\tbytes\n")

    st = Stream(TAR_URL, ck["offset"], TAR_BYTES)
    pool = Pool(processes=args.feat, initializer=_init, initargs=(SELECTED_KMERS,))
    t0 = time.time()
    batch: list[tuple[str, str, int]] = []
    zero_run = 0
    try:
        while True:
            hdr = st.read(512)
            if len(hdr) < 512:
                break
            if hdr == b"\0" * 512:
                zero_run += 1
                if zero_run >= 2:
                    break
                continue
            zero_run = 0
            if not header_ok(hdr):
                print(f"[232] FATAL bad tar header at ~{st.pos:,}", file=sys.stderr)
                return 3
            name = hdr[0:100].split(b"\0")[0].decode("utf-8", "replace")
            size = int(hdr[124:136].split(b"\0")[0].strip() or b"0", 8)
            pad = (512 - size % 512) % 512
            body = st.read(size)
            st.read(pad)
            ck["n_members"] += 1
            base = name[len(PREFIX):] if name.startswith(PREFIX) else name
            if not base.startswith("spire_mag_") or not base.endswith(".fa.gz"):
                ck["n_skipped"] += 1
                continue
            gid = base[:-6]
            (STAGE / f"{gid}.fa.gz").write_bytes(body)
            batch.append((gid, hashlib.sha256(body).hexdigest(), len(body)))
            if len(batch) >= BATCH:
                _flush(batch, pool, norm, sess, ins, out0, pf, cf, ck, st, t0)
                batch = []
        if batch:
            _flush(batch, pool, norm, sess, ins, out0, pf, cf, ck, st, t0)
        ck["offset"] = TAR_BYTES
        CKPT.write_text(json.dumps(ck) + "\n")
    finally:
        pool.close(); pool.join(); pf.close(); cf.close()
    print(f"[232] finished: {ck['n_scored']:,} representatives scored, "
          f"{ck['n_skipped']:,} non-MAG members skipped, {(time.time() - t0) / 3600:.2f} h", flush=True)
    return 0


def _flush(batch, pool, norm, sess, ins, out0, pf, cf, ck, st, t0) -> None:
    gids, ks, as_ = [], [], []
    for gid, k, a, err in pool.imap_unordered(_feat, [b[0] for b in batch], chunksize=8):
        if err:
            print(f"[232]   featfail {gid}: {err}", flush=True)
        else:
            gids.append(gid); ks.append(k); as_.append(a)
    if gids:
        kn = norm.normalize_kmer(np.stack(ks)).astype(np.float32)
        an = norm.normalize_assembly(np.stack(as_)).astype(np.float32)
        preds = np.zeros((len(gids), 2), np.float32)
        for b0 in range(0, len(gids), 256):
            b1 = min(b0 + 256, len(gids))
            preds[b0:b1] = sess.run([out0], {ins[0]: kn[b0:b1], ins[1]: an[b0:b1]})[0]
        h = {b[0]: (b[1], b[2]) for b in batch}
        for j, gid in enumerate(gids):
            pf.write(f"{gid}\t{preds[j, 0]:.4f}\t{preds[j, 1]:.4f}\n")
            cf.write(f"{gid}\t{h[gid][0]}\t{h[gid][1]}\n")
        ck["n_scored"] += len(gids)
    for b in batch:
        try:
            (STAGE / f"{b[0]}.fa.gz").unlink()
        except OSError:
            pass
    pf.flush(); cf.flush(); os.fsync(pf.fileno()); os.fsync(cf.fileno())
    ck["offset"] = st.pos
    CKPT.write_text(json.dumps(ck) + "\n")
    el = time.time() - t0
    frac = (st.pos - FIRST_MAG_OFFSET) / (TAR_BYTES - FIRST_MAG_OFFSET)
    print(f"[232] {ck['n_scored']:,} scored, byte {st.pos:,} ({100 * frac:.2f} %), "
          f"eta {el / max(frac, 1e-9) * (1 - frac) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    sys.exit(main())
