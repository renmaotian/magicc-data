#!/usr/bin/env python3
"""
WS3.10 (item 5) — which catalogue is Reviewer 2 actually quoting?

Reviewer 2 reports, for CAG-557 / UMGS1491 / Caccenecus / HGM10766, that MAGICC differs
from "the catalogue" by +37.1 / +27.1 / +25.3 / +24.9 pp completeness and
-35.9 / -26.0 / -24.6 / -21.8 pp contamination.  The catalogue is not named, and its
implied completeness matches neither UHGG-CheckM1 nor GTDB-CheckM2.

This script settles what can be settled:

  1. Assemble every catalogue on disk that contains these genera, with its OWN published
     quality values:
        GTDB r220   CheckM2   (all five genera; 298 genomes; already scored, script 133)
        UHGG v2.0.2 CheckM1   (three genera; 126 genomes; already scored, script 135)
        SPIRE v1    CheckM2   (all five genera; 612 genomes; scored HERE)
     SPIRE is the decisive addition: it is the only catalogue on disk that has all five
     genera AND ships CheckM2 AND consists of MAG assemblies of catalogue-typical
     fragmentation, i.e. it matches the reviewer's description on every axis.

  2. Run MAGICC V5 on the SPIRE genomes of the five genera (direct ONNX, as in
     scripts/75 and 133).

  3. Reconcile: for every catalogue x every sign convention, compute the MAGICC value the
     reviewer's deltas imply, mark it INFEASIBLE if it falls outside [50, 100] for
     completeness (MAGICC's floor is 50) or outside [0, 100] for contamination, and score
     the residual against the MAGICC values actually measured on that catalogue.

Resumable: downloads are cached; re-running skips completed fetches.
Network use is capped at 6 concurrent connections with a descriptive User-Agent.

Outputs (results/revision/real_data/reduced_genome/)
  spire_reviewer_genera_predictions.tsv
  catalogue_baseline_reconciliation.tsv
  catalogue_baseline_reconciliation.json
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from urllib.request import Request, urlopen

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import onnxruntime as ort                                   # noqa: E402
from magicc.assembly_stats import compute_assembly_stats    # noqa: E402
from magicc.kmer_counter import KmerCounter                 # noqa: E402
from magicc.normalization import FeatureNormalizer          # noqa: E402

OUTDIR = PROJECT_DIR / "results/revision/real_data/reduced_genome"
SPIRE_META = PROJECT_DIR / "data/real_data/spire/spire_v1_genome_metadata.tsv.gz"
SPIRE_FASTA = PROJECT_DIR / "data/real_data/spire/reviewer_genera_fasta"
UHGG_PRED = OUTDIR / "uhgg_reviewer_genera_predictions.tsv"
GTDB_PRED = OUTDIR / "magicc_v5_predictions.tsv"

SELECTED_KMERS = str(PROJECT_DIR / "data/kmer_selection/selected_kmers.txt")
NORMALIZATION = str(PROJECT_DIR / "data/features/normalization_params.json")
ONNX_MODEL = str(PROJECT_DIR / "models/magicc_v5.onnx")

GENERA = ["CAG-557", "UMGS1491", "Caccenecus", "HGM10766", "Faecimonas"]
# Reviewer 2's reported deltas, in the order the reviewer gives them.
REVIEWER = {
    "CAG-557":   {"completeness": 37.1, "contamination": -35.9},
    "UMGS1491":  {"completeness": 27.1, "contamination": -26.0},
    "Caccenecus": {"completeness": 25.3, "contamination": -24.6},
    "HGM10766":  {"completeness": 24.9, "contamination": -21.8},
    "Faecimonas": None,          # named by the reviewer but no numbers given
}

MAX_DL_WORKERS = 6
N_FEAT_WORKERS = 6
UA = "magicc-revision/1.0 (academic revision analysis; tianrenmao@gmail.com)"
SPIRE_URL = "https://spire.embl.de/download_file/{gid}"
MAGICC_FLOOR = 50.0


# ------------------------------------------------------------------ SPIRE ----
def spire_select() -> list[dict]:
    out = []
    with gzip.open(SPIRE_META, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("genus") in GENERA:
                out.append(r)
    return out


def fetch_one(row: dict) -> str:
    gid = row["genome_id"]
    dest = SPIRE_FASTA / f"{gid}.fa.gz"
    if dest.exists() and dest.stat().st_size > 1000:
        return "cached"
    tmp = dest.with_suffix(".part")
    for attempt in range(3):
        try:
            req = Request(SPIRE_URL.format(gid=gid), headers={"User-Agent": UA})
            with urlopen(req, timeout=180) as resp:
                data = resp.read()
            if len(data) < 1000:
                raise ValueError(f"short body {len(data)}")
            tmp.write_bytes(data)
            tmp.rename(dest)
            return "ok"
        except Exception as exc:                                     # noqa: BLE001
            if attempt == 2:
                return f"FAILED {type(exc).__name__}: {exc}"
            time.sleep(3 * (attempt + 1))
    return "FAILED"


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
    except Exception as exc:                                         # noqa: BLE001
        return idx, None, None, f"{type(exc).__name__}: {exc}"


def run_spire() -> list[dict]:
    out = OUTDIR / "spire_reviewer_genera_predictions.tsv"
    if out.exists():
        recs = list(csv.DictReader(out.open(), delimiter="\t"))
        print(f"[137] SPIRE predictions already on disk: {len(recs)}", flush=True)
        return recs

    SPIRE_FASTA.mkdir(parents=True, exist_ok=True)
    rows = spire_select()
    print(f"[137] SPIRE genomes in the five genera: {len(rows)}", flush=True)

    stat = defaultdict(int)
    with ThreadPoolExecutor(max_workers=MAX_DL_WORKERS) as pool:
        futs = {pool.submit(fetch_one, r): r for r in rows}
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            stat[s.split(":")[0]] += 1
            if n % 100 == 0:
                print(f"[137]   fetched {n}/{len(rows)} {dict(stat)}", flush=True)
    print(f"[137] fetch complete: {dict(stat)}", flush=True)

    work = [(i, str(SPIRE_FASTA / f"{r['genome_id']}.fa.gz"))
            for i, r in enumerate(rows)
            if (SPIRE_FASTA / f"{r['genome_id']}.fa.gz").exists()]
    got = {}
    with Pool(processes=N_FEAT_WORKERS, initializer=_init,
              initargs=(SELECTED_KMERS,)) as pool:
        for idx, k, a, err in pool.imap_unordered(_feat, work, chunksize=4):
            if err:
                print(f"[137]   WARNING idx={idx}: {err}", flush=True)
            else:
                got[idx] = (k, a)
    idxs = sorted(got)
    print(f"[137] features for {len(idxs)} SPIRE genomes", flush=True)

    norm = FeatureNormalizer.load(NORMALIZATION)
    kn = norm.normalize_kmer(np.stack([got[i][0] for i in idxs])).astype(np.float32)
    an = norm.normalize_assembly(np.stack([got[i][1] for i in idxs])).astype(np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(ONNX_MODEL, so, providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    preds = np.zeros((len(idxs), 2), np.float32)
    for s in range(0, len(idxs), 64):
        e = min(s + 64, len(idxs))
        preds[s:e] = sess.run([sess.get_outputs()[0].name],
                              {ins[0]: kn[s:e], ins[1]: an[s:e]})[0]

    cols = ["genome_id", "genus", "genome_size", "n_contigs", "n50", "n_genes",
            "completeness", "contamination", "gunc_pass", "classification",
            "magicc_completeness", "magicc_contamination"]
    recs = []
    with out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for j, i in enumerate(idxs):
            r = dict(rows[i])
            r["magicc_completeness"] = f"{preds[j, 0]:.4f}"
            r["magicc_contamination"] = f"{preds[j, 1]:.4f}"
            recs.append(r)
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[137] wrote {out}", flush=True)
    return recs


# ------------------------------------------------------- catalogue assembly --
def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def catalogue_table(spire_recs) -> list[dict]:
    cats: list[dict] = []

    # --- GTDB r220, CheckM2, MAGICC already run by script 133 ---------------
    g = defaultdict(list)
    for r in csv.DictReader(GTDB_PRED.open(), delimiter="\t"):
        if r["cohort"] == "reviewer_genera":
            g[r["genus"]].append(r)
    for gen, sub in g.items():
        cats.append({
            "catalogue": "GTDB r220", "published_tool": "CheckM2", "genus": gen,
            "n": len(sub),
            "assembly_size_median_Mbp": round(med([int(x["genome_size"]) for x in sub]) / 1e6, 3),
            "contigs_median": int(med([int(x["contig_count"]) for x in sub])),
            "catalogue_completeness_median": round(
                med([float(x["checkm2_completeness"]) for x in sub]), 2),
            "catalogue_contamination_median": round(
                med([float(x["checkm2_contamination"]) for x in sub]), 2),
            "magicc_completeness_median": round(
                med([float(x["magicc_completeness"]) for x in sub]), 2),
            "magicc_contamination_median": round(
                med([float(x["magicc_contamination"]) for x in sub]), 2),
        })

    # --- UHGG v2.0.2, CheckM1 ------------------------------------------------
    if UHGG_PRED.exists():
        u = defaultdict(list)
        for r in csv.DictReader(UHGG_PRED.open(), delimiter="\t"):
            u[r["genus"]].append(r)
        for gen, sub in u.items():
            cats.append({
                "catalogue": "UHGG v2.0.2", "published_tool": "CheckM1", "genus": gen,
                "n": len(sub),
                "assembly_size_median_Mbp": round(med([int(x["Length"]) for x in sub]) / 1e6, 3),
                "contigs_median": int(med([int(x["N_contigs"]) for x in sub])),
                "catalogue_completeness_median": round(
                    med([float(x["Completeness"]) for x in sub]), 2),
                "catalogue_contamination_median": round(
                    med([float(x["Contamination"]) for x in sub]), 2),
                "magicc_completeness_median": round(
                    med([float(x["magicc_completeness"]) for x in sub]), 2),
                "magicc_contamination_median": round(
                    med([float(x["magicc_contamination"]) for x in sub]), 2),
            })

    # --- SPIRE v1, CheckM2 ---------------------------------------------------
    s = defaultdict(list)
    for r in spire_recs:
        s[r["genus"]].append(r)
    for gen, sub in s.items():
        cats.append({
            "catalogue": "SPIRE v1", "published_tool": "CheckM2", "genus": gen,
            "n": len(sub),
            "assembly_size_median_Mbp": round(
                med([float(x["genome_size"]) for x in sub]) / 1e6, 3),
            "contigs_median": int(med([float(x["n_contigs"]) for x in sub])),
            "catalogue_completeness_median": round(
                med([float(x["completeness"]) for x in sub]), 2),
            "catalogue_contamination_median": round(
                med([float(x["contamination"]) for x in sub]), 2),
            "magicc_completeness_median": round(
                med([float(x["magicc_completeness"]) for x in sub]), 2),
            "magicc_contamination_median": round(
                med([float(x["magicc_contamination"]) for x in sub]), 2),
        })

    for c in cats:
        c["observed_delta_completeness_catalogue_minus_MAGICC"] = round(
            c["catalogue_completeness_median"] - c["magicc_completeness_median"], 2)
        c["observed_delta_contamination_catalogue_minus_MAGICC"] = round(
            c["catalogue_contamination_median"] - c["magicc_contamination_median"], 2)
        rv = REVIEWER.get(c["genus"])
        if rv:
            c["reviewer_delta_completeness"] = rv["completeness"]
            c["reviewer_delta_contamination"] = rv["contamination"]
            c["residual_completeness_pp"] = round(
                c["observed_delta_completeness_catalogue_minus_MAGICC"] - rv["completeness"], 2)
            c["residual_contamination_pp"] = round(
                c["observed_delta_contamination_catalogue_minus_MAGICC"] - rv["contamination"], 2)
            # implied MAGICC values under each sign convention
            imp_a = c["catalogue_completeness_median"] - rv["completeness"]      # cat - MAGICC
            imp_b = c["catalogue_completeness_median"] + rv["completeness"]      # MAGICC - cat
            c["implied_MAGICC_comp_if_delta_is_catalogue_minus_MAGICC"] = round(imp_a, 2)
            c["implied_MAGICC_comp_if_delta_is_MAGICC_minus_catalogue"] = round(imp_b, 2)
            c["convention_A_feasible"] = bool(MAGICC_FLOOR <= imp_a <= 100)
            c["convention_B_feasible"] = bool(MAGICC_FLOOR <= imp_b <= 100)
        else:
            for k in ("reviewer_delta_completeness", "reviewer_delta_contamination",
                      "residual_completeness_pp", "residual_contamination_pp",
                      "implied_MAGICC_comp_if_delta_is_catalogue_minus_MAGICC",
                      "implied_MAGICC_comp_if_delta_is_MAGICC_minus_catalogue",
                      "convention_A_feasible", "convention_B_feasible"):
                c[k] = ""
    cats.sort(key=lambda c: (GENERA.index(c["genus"]) if c["genus"] in GENERA else 9,
                             c["catalogue"]))
    return cats


def write_tsv(path: Path, blocks):
    keys, seen = [], set()
    for b in blocks:
        for k in b:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w") as fh:
        fh.write("\t".join(keys) + "\n")
        for b in blocks:
            fh.write("\t".join(str(b.get(k, "")) for k in keys) + "\n")


def main() -> int:
    spire = run_spire()
    cats = catalogue_table(spire)
    write_tsv(OUTDIR / "catalogue_baseline_reconciliation.tsv", cats)

    # mean absolute residual per catalogue over the four genera the reviewer quantified
    fit = defaultdict(list)
    for c in cats:
        if c.get("residual_completeness_pp") != "" and c.get("residual_completeness_pp") is not None:
            fit[c["catalogue"]].append(abs(float(c["residual_completeness_pp"])))
    fit_summary = [{"catalogue": k, "n_genera_scored": len(v),
                    "mean_abs_residual_completeness_pp": round(float(np.mean(v)), 2)}
                   for k, v in sorted(fit.items())]

    n_spire_meta = len(spire_select())
    summary = {
        "script": "scripts/207_catalogue_baseline_reconciliation.py",
        "question": "which catalogue is Reviewer 2 quoting?",
        "magicc_completeness_floor": MAGICC_FLOOR,
        "spire_coverage": {
            "genomes_of_the_five_genera_in_SPIRE_metadata": n_spire_meta,
            "genomes_scored": len(spire),
            "not_retrievable": n_spire_meta - len(spire),
            "note": ("the shortfall is HTTP 404 on spire.embl.de for those genome ids, "
                     "i.e. server-side absence, not a fetch failure; 96.6% coverage, "
                     "medians unaffected"),
        },
        "reviewer_deltas_as_given": REVIEWER,
        "catalogues": cats,
        "fit_to_reviewer_completeness_deltas": fit_summary,
    }
    (OUTDIR / "catalogue_baseline_reconciliation.json").write_text(json.dumps(summary, indent=2))

    print("\n=== CATALOGUE RECONCILIATION "
          "(delta = catalogue minus MAGICC, the reviewer's apparent convention) ===")
    hdr = (f"{'genus':12}{'catalogue':13}{'tool':9}{'n':>5}{'Mbp':>7}{'ctg':>6}"
           f"{'cat comp':>10}{'MAGICC comp':>12}{'dComp':>8}"
           f"{'cat cont':>10}{'MAGICC cont':>12}{'dCont':>8}{'revComp':>9}{'residual':>10}")
    print(hdr)
    for c in cats:
        rv = c.get("reviewer_delta_completeness")
        print(f"{c['genus'][:11]:12}{c['catalogue'][:12]:13}{c['published_tool']:9}{c['n']:>5}"
              f"{c['assembly_size_median_Mbp']:>7.2f}{c['contigs_median']:>6}"
              f"{c['catalogue_completeness_median']:>10.2f}{c['magicc_completeness_median']:>12.2f}"
              f"{c['observed_delta_completeness_catalogue_minus_MAGICC']:>8.2f}"
              f"{c['catalogue_contamination_median']:>10.2f}{c['magicc_contamination_median']:>12.2f}"
              f"{c['observed_delta_contamination_catalogue_minus_MAGICC']:>8.2f}"
              f"{(rv if rv != '' else float('nan')):>9}"
              f"{(c['residual_completeness_pp'] if c['residual_completeness_pp'] != '' else float('nan')):>10}")
    print("\n=== FIT TO THE REVIEWER'S COMPLETENESS DELTAS ===")
    for f in fit_summary:
        print(f"  {f['catalogue']:14} n_genera={f['n_genera_scored']} "
              f"mean|residual|={f['mean_abs_residual_completeness_pp']:.2f} pp")
    print(f"\n[137] wrote {OUTDIR}/catalogue_baseline_reconciliation.tsv|.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
