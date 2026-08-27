#!/usr/bin/env python3
"""
WS7.12 — inference determinism check.

Addresses R1-M8 / R1-m17 / E2: fixed inputs must give identical outputs across
thread counts, input encodings, invocation modes and execution environments.

What this checks (all on CPU; MAGICC inference is CPU-only by design):

  1. model artefact       models/magicc_v5.onnx SHA256 == results/revision/model_card.json
  2. fixed-input probe    the model_card determinism probe reproduces exactly
  3. thread invariance    N genomes at 1 / 2 / 4 threads -> byte-identical TSV
  4. gz invariance        X.fasta vs X.fasta.gz          -> byte-identical TSV
  5. input-mode invariance  --input DIR vs --input-list  -> byte-identical values
  6. repeat invariance    the same command run twice     -> byte-identical TSV
  7. batch-size invariance  --batch-size 1 / 64 / 512    -> byte-identical TSV

Everything is run through the installed `magicc` console script, i.e. exactly
what a reviewer would invoke -- not through an internal API.

This script deliberately does NOT claim that training is reproducible. V5 was
trained without a random seed (protocol section 4.4b); that is reported, not
hidden.

Usage:
    python scripts/115_determinism_check.py [--n-genomes 24] [--max-threads 4]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MODEL_SHA_EXPECTED = "b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def values_only(tsv: Path) -> bytes:
    """Prediction values keyed by genome name, order-independent."""
    rows = []
    with open(tsv, encoding="utf-8") as f:
        f.readline()
        for ln in f:
            if ln.strip():
                rows.append(ln.rstrip("\n").split("\t"))
    rows.sort(key=lambda r: r[0])
    return "\n".join("\t".join(r) for r in rows).encode()


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, passed: bool, detail: str, evidence: dict | None = None) -> None:
        self.checks.append(dict(check=name, result="PASS" if passed else "FAIL",
                                detail=detail, evidence=evidence or {}))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    @property
    def ok(self) -> bool:
        return all(c["result"] == "PASS" for c in self.checks)


def run_magicc(magicc: str, args: list[str], env: dict) -> None:
    r = subprocess.run([magicc, "predict", *args], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"magicc failed ({r.returncode}):\n{r.stderr[-4000:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--n-genomes", type=int, default=24)
    ap.add_argument("--max-threads", type=int, default=4,
                    help="upper bound on threads; other agents share this machine")
    ap.add_argument("--source-set", default="set_E",
                    help="benchmark set to draw real genomes from")
    args = ap.parse_args()

    project = Path(args.project_dir).resolve()
    out_dir = project / "results" / "revision" / "reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()

    magicc = shutil.which("magicc")
    if not magicc:
        print("FATAL: the `magicc` console script is not on PATH; "
              "activate the environment where the package is installed.", file=sys.stderr)
        return 2

    model = project / "models" / "magicc_v5.onnx"
    norm = project / "data" / "features" / "normalization_params.json"
    kmers = project / "data" / "kmer_selection" / "selected_kmers.txt"
    resources = ["--model", str(model), "--normalization", str(norm), "--kmers", str(kmers)]

    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    print("WS7.12 determinism check")
    print(f"  console script : {magicc}")

    # ---- 1. model artefact ------------------------------------------------
    card = json.loads((project / "results" / "revision" / "model_card.json").read_text())
    actual = sha256_file(model)
    rep.add("model_artefact_sha256",
            actual == MODEL_SHA_EXPECTED == card["onnx"]["sha256"],
            f"models/magicc_v5.onnx = {actual}",
            dict(expected=MODEL_SHA_EXPECTED, model_card=card["onnx"]["sha256"], actual=actual))

    # ---- 2. fixed-input probe from the model card -------------------------
    try:
        import numpy as np
        import onnxruntime as ort
        probe = card["onnx"]["determinism_probe"]
        rng = np.random.default_rng(20260726)
        km = rng.standard_normal((4, 9249)).astype(np.float32)
        asm = rng.standard_normal((4, 7)).astype(np.float32)
        probe_sha = sha256_bytes(km.tobytes() + asm.tobytes())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"kmer_features": km, "assembly_features": asm})[0]
        expected = np.asarray(probe["expected_output"], dtype=np.float32)
        exact = bool(np.array_equal(got, expected))
        maxdev = float(np.max(np.abs(got - expected)))
        rep.add("model_card_determinism_probe", exact,
                ("reproduced bit-exactly" if exact else f"max deviation {maxdev:.3e}")
                + f"; probe input sha256 {'matches' if probe_sha == probe['probe_input_sha256'] else 'DIFFERS'}",
                dict(probe_input_sha256=probe_sha,
                     probe_input_sha256_expected=probe["probe_input_sha256"],
                     max_abs_deviation=maxdev,
                     onnxruntime_version=ort.__version__))
    except Exception as exc:                                    # pragma: no cover
        rep.add("model_card_determinism_probe", False, f"could not run: {exc!r}")

    # ---- fixtures ---------------------------------------------------------
    src_dir = project / "data" / "benchmarks" / args.source_set / "fasta"
    fastas = sorted(src_dir.glob("*.fasta"))[: args.n_genomes]
    if len(fastas) < 2:
        print(f"FATAL: fewer than 2 FASTA files in {src_dir}", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="magicc_determinism_"))
    try:
        plain, gz = tmp / "plain", tmp / "gz"
        plain.mkdir(); gz.mkdir()
        for f in fastas:
            shutil.copy2(f, plain / f.name)
            with open(f, "rb") as fin, gzip.open(gz / (f.name + ".gz"), "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)
        listing = tmp / "paths.txt"
        listing.write_text("\n".join(
            str(plain / f.name) if i % 2 == 0 else str(gz / (f.name + ".gz"))
            for i, f in enumerate(fastas)) + "\n")
        print(f"  fixtures       : {len(fastas)} genomes from {args.source_set}")

        threads = sorted({1, 2, args.max_threads})
        digests: dict[str, str] = {}

        # ---- 3. thread invariance -----------------------------------------
        for t in threads:
            o = tmp / f"t{t}.tsv"
            run_magicc(magicc, ["--input", str(plain), "--output", str(o),
                                "--threads", str(t), "--quiet", *resources], env)
            digests[f"threads={t}"] = sha256_file(o)
        tset = {digests[f"threads={t}"] for t in threads}
        rep.add("thread_count_invariance", len(tset) == 1,
                f"threads {threads} -> {len(tset)} distinct output digest(s)",
                {f"threads={t}": digests[f"threads={t}"] for t in threads})
        ref_digest = digests[f"threads={threads[0]}"]

        # ---- 4. gz invariance ---------------------------------------------
        ogz = tmp / "gz.tsv"
        run_magicc(magicc, ["--input", str(gz), "--output", str(ogz),
                            "--threads", str(threads[-1]), "--quiet", *resources], env)
        rep.add("gzip_input_invariance", sha256_file(ogz) == ref_digest,
                "X.fasta and X.fasta.gz give byte-identical predictions",
                dict(plain=ref_digest, gzipped=sha256_file(ogz)))

        # ---- 5. input-mode invariance -------------------------------------
        olist = tmp / "list.tsv"
        run_magicc(magicc, ["--input-list", str(listing), "--output", str(olist),
                            "--threads", str(threads[-1]), "--quiet", *resources], env)
        same = sha256_bytes(values_only(olist)) == sha256_bytes(values_only(tmp / f"t{threads[0]}.tsv"))
        rep.add("input_mode_invariance", same,
                "--input DIR and --input-list (mixed plain/gz) agree on every genome",
                dict(directory_values=sha256_bytes(values_only(tmp / f"t{threads[0]}.tsv")),
                     list_values=sha256_bytes(values_only(olist))))

        # ---- 6. repeat invariance -----------------------------------------
        orep = tmp / "repeat.tsv"
        run_magicc(magicc, ["--input", str(plain), "--output", str(orep),
                            "--threads", str(threads[-1]), "--quiet", *resources], env)
        rep.add("repeat_run_invariance", sha256_file(orep) == ref_digest,
                "identical command run twice gives byte-identical output",
                dict(first=ref_digest, second=sha256_file(orep)))

        # ---- 7. batch-size invariance -------------------------------------
        bdig = {}
        for bs in (1, 64, 512):
            o = tmp / f"b{bs}.tsv"
            run_magicc(magicc, ["--input", str(plain), "--output", str(o), "--batch-size", str(bs),
                                "--threads", str(threads[-1]), "--quiet", *resources], env)
            bdig[f"batch={bs}"] = sha256_file(o)
        rep.add("batch_size_invariance", len(set(bdig.values())) == 1 and ref_digest in set(bdig.values()),
                f"batch sizes 1/64/512 -> {len(set(bdig.values()))} distinct digest(s)", bdig)

        payload_digest = ref_digest
        n_genomes = len(fastas)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- write the report -------------------------------------------------
    result = dict(
        schema="magicc-determinism-report/1.0",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        generated_by="scripts/115_determinism_check.py",
        workstream_item="WS7.12",
        reviewer_comments=["R1-M8", "R1-m17", "E2"],
        n_genomes=n_genomes,
        source_set=args.source_set,
        thread_counts_tested=sorted({1, 2, args.max_threads}),
        reference_output_sha256=payload_digest,
        console_script=magicc,
        environment=dict(python=sys.version.split()[0], platform=sys.platform,
                         executable=sys.executable),
        checks=rep.checks,
        overall="PASS" if rep.ok else "FAIL",
        training_reproducibility=dict(
            seeded=False,
            statement=("V5 training was never seeded (verified by code inspection of "
                       "magicc/trainer.py and scripts/53_train_v5_run3.py), so the released "
                       "weights cannot be re-derived bit-exactly from the training data. "
                       "The released artefact is pinned by SHA256 instead. This report "
                       "establishes determinism of INFERENCE only."),
            protocol_reference="the internal design protocol section 4.4b"),
    )
    (out_dir / "determinism_report.json").write_text(json.dumps(result, indent=2) + "\n",
                                                     encoding="utf-8")
    print(f"\n  overall: {result['overall']}")
    print(f"  report : {out_dir / 'determinism_report.json'}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
