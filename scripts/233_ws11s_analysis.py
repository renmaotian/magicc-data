#!/usr/bin/env python3
"""
WS11.S -- catalogue-scale disagreement analysis: SPIRE v1 published CheckM2 vs MAGICC V5.

BINDING FRAMING.  Catalogue MAGs carry NO ground truth.  Every quantity below is a
DISAGREEMENT between two estimators, never an error.  The ground-truthed anchor is
`set_C_clean`, where MAGICC is the tool in error (-8.68 pp completeness, +5.09 pp
contamination against truth; CheckM2 -0.70 / -1.16).  Nothing here licenses "MAGICC is
right and CheckM2 is wrong".

Cohorts (each reported separately, never pooled):
  catalogue_weighted   S1 u S2 u S3 with design weights -- the catalogue-wide estimate
  S3_srs               the 200,000-genome SRS alone, unweighted -- design check
  S1_small_census      every frame MAG < 1 Mbp
  S2_rare_phylum_census
  reps_census          every SPIRE 95 % ANI MAG representative (a DIFFERENT population)

Uncertainty: 95 % percentile cluster bootstrap, primary clustering unit = **family**
(as in WS3.10), secondary = **spire_cluster** (95 % ANI). Seeds via fw.stable_hash under
PYTHONHASHSEED=0. Weighted estimates are Hajek ratios.

Outputs (results/revision/ws11/spire_catalogue/): classification_change_matrix.tsv,
classification_change_by_stratum.tsv, delta_distributions.tsv, size_dose_response.tsv,
floor_censoring.tsv, small_vs_large_cohort_comparison.tsv, nonresponse.tsv,
ws11s_headline.json
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

ROOT = Path("/path/to/magicc")
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("fw", ROOT / "scripts/101_metrics_framework.py")
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)

OUT = ROOT / "results/revision/ws11/spire_catalogue"
META = ROOT / "data/real_data/spire/spire_v1_genome_metadata.tsv.gz"
N_BOOT = 2000
FLOOR = 50.0
SIZE_LABELS = ["<1Mb", "1-2Mb", "2-3Mb", "3-5Mb", ">5Mb"]
SIZE_EDGES = [1e6, 2e6, 3e6, 5e6]
LR_EDGES = [-1.0, -0.5, -0.25, 0.25]
LR_LABELS = ["<-1 (>=2x reduced)", "-1 to -0.5", "-0.5 to -0.25",
             "-0.25 to +0.25 (typical)", ">+0.25 (larger than lineage)"]
RANK = {"LQ": 0, "MQ": 1, "HQ": 2}


def mimag(comp: np.ndarray, cont: np.ndarray) -> np.ndarray:
    out = np.full(comp.shape, "LQ", dtype="<U2")
    out[(comp >= 50.0) & (cont < 10.0)] = "MQ"
    out[(comp >= 90.0) & (cont < 5.0)] = "HQ"
    return out


# --------------------------------------------------------------------- data --
def load() -> dict:
    pred = {}
    for f in ("magicc_predictions.tsv", "reps_predictions.tsv"):
        p = OUT / f
        if not p.exists():
            continue
        with p.open() as fh:
            next(fh)
            for line in fh:
                a = line.rstrip("\n").split("\t")
                if len(a) == 3:
                    pred[a[0]] = (float(a[1]), float(a[2]))
    print(f"[233] {len(pred):,} MAGICC predictions on disk", flush=True)

    rep_ids = set()
    rp = OUT / "reps_predictions.tsv"
    if rp.exists():
        with rp.open() as fh:
            next(fh)
            for line in fh:
                rep_ids.add(line.split("\t", 1)[0])

    coh = {}
    with gzip.open(OUT / "cohort_definition.tsv.gz", "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            coh[r["genome_id"]] = r

    need = set(coh) | rep_ids
    meta = {}
    with gzip.open(META, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["genome_id"] in need:
                meta[r["genome_id"]] = r
    print(f"[233] cohort {len(coh):,}; reps scored {len(rep_ids):,}; metadata joined {len(meta):,}",
          flush=True)
    return {"pred": pred, "coh": coh, "meta": meta, "rep_ids": rep_ids}


def build(ids, meta, coh, pred, weights):
    """Assemble the analysis arrays for a list of genome ids."""
    keep = [g for g in ids if g in pred and g in meta]
    n = len(keep)
    o = {k: np.empty(n) for k in ("size", "c_comp", "c_cont", "m_comp", "m_cont", "w", "lr")}
    fam = np.empty(n, dtype=object); ph = np.empty(n, dtype=object)
    clus = np.empty(n, dtype=object); dom = np.empty(n, dtype=object)
    stratum = np.empty(n, dtype=object)
    for i, g in enumerate(keep):
        m = meta[g]
        o["size"][i] = float(m["genome_size"])
        o["c_comp"][i] = float(m["completeness"]); o["c_cont"][i] = float(m["contamination"])
        o["m_comp"][i], o["m_cont"][i] = pred[g]
        o["w"][i] = weights(g)
        fam[i] = m["family"] or "unclassified"; ph[i] = m["phylum"] or "unclassified"
        clus[i] = m["spire_cluster"]; dom[i] = m["domain"] or "unclassified"
        stratum[i] = coh[g]["stratum"] if g in coh else "reps"
    o.update({"gid": np.array(keep), "family": fam, "phylum": ph, "cluster": clus,
              "domain": dom, "stratum": stratum})
    o["c_class"] = mimag(o["c_comp"], o["c_cont"])
    o["m_class"] = mimag(o["m_comp"], o["m_cont"])
    o["d_comp"] = o["m_comp"] - o["c_comp"]
    o["d_cont"] = o["m_cont"] - o["c_cont"]
    o["size_bin"] = np.digitize(o["size"], SIZE_EDGES)
    return o


def add_lineage_relative(o, ph_median):
    o["lr"] = np.log2(o["size"] / np.array([ph_median.get(p, np.nan) for p in o["phylum"]]))
    o["lr_bin"] = np.digitize(o["lr"], LR_EDGES)
    return o


# ---------------------------------------------------------------- estimators --
def wmean(y, w, m=None):
    if m is not None:
        y, w = y[m], w[m]
    return float(np.sum(w * y) / np.sum(w)) if w.size and np.sum(w) > 0 else float("nan")


def wmedian(y, w, m=None):
    if m is not None:
        y, w = y[m], w[m]
    if y.size == 0:
        return float("nan")
    o = np.argsort(y, kind="stable")
    ys, ws = y[o], w[o]
    c = np.cumsum(ws)
    return float(ys[np.searchsorted(c, c[-1] / 2.0)])


def wprop(mask_event, w, m=None):
    if m is not None:
        mask_event, w = mask_event[m], w[m]
    tw = np.sum(w)
    return float(np.sum(w * mask_event) / tw) if tw > 0 else float("nan")


def wslope(x, y, w, m=None):
    if m is not None:
        x, y, w = x[m], y[m], w[m]
    if x.size < 3:
        return float("nan"), float("nan")
    sw = np.sum(w); sx = np.sum(w * x); sy = np.sum(w * y)
    sxx = np.sum(w * x * x); sxy = np.sum(w * x * y)
    den = sw * sxx - sx * sx
    if den == 0:
        return float("nan"), float("nan")
    b = (sw * sxy - sx * sy) / den
    a = (sy - b * sx) / sw
    return float(b), float(a)


def cluster_boot(cluster_labels, seed, n_boot=N_BOOT):
    """Yield resampled row-index arrays, resampling clusters with replacement."""
    uniq, inv = np.unique(cluster_labels, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(uniq.size))
    ends = np.searchsorted(inv[order], np.arange(uniq.size), side="right")
    idx_by_cluster = [order[s:e] for s, e in zip(starts, ends)]
    rng = np.random.default_rng(seed)
    k = uniq.size
    for _ in range(n_boot):
        pick = rng.integers(0, k, size=k)
        yield np.concatenate([idx_by_cluster[j] for j in pick]), k


def ci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if v.size < 20:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
