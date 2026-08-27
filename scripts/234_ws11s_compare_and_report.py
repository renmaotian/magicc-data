#!/usr/bin/env python3
"""
WS11.S -- catalogue-scale vs 750-genome cohort comparison, and the WS11.S report.

The WS3.10 cohort that the manuscript currently cites is **750 GTDB r220 MAGs sampled
150 per genome-size bin** (script 131, seed 13100) plus 320 control-genus MAGs; the SPIRE
part of WS3.10 was a separate 612/591-genome five-genus cohort.  Neither is a
catalogue-representative sample, so neither rate is a catalogue rate.  Two distinct
things could make the small cohort's numbers differ from the catalogue's:

  (a) the DESIGN -- 150 per size bin over-represents <1 Mbp by ~6x and >5 Mbp by ~13x
      relative to the catalogue, which moves any size-dependent statistic;
  (b) the CATALOGUE -- GTDB r220 is not SPIRE v1.

(a) is isolated here by drawing replicate 150-per-size-bin cohorts FROM the SPIRE
catalogue data measured in this workstream and re-running the same estimators; the spread
across replicates is the sampling distribution of the 750-genome design.  (b) cannot be
isolated without rescoring GTDB and is stated, not estimated.

Outputs: small_vs_large_cohort_comparison.tsv, WS11_S_REPORT.md
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
import numpy as np  # noqa: E402

ROOT = Path("/path/to/magicc")
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("fw", ROOT / "scripts/101_metrics_framework.py")
fw = importlib.util.module_from_spec(_spec); sys.modules["fw"] = fw; _spec.loader.exec_module(fw)

OUT = ROOT / "results/revision/ws11/spire_catalogue"
META = ROOT / "data/real_data/spire/spire_v1_genome_metadata.tsv.gz"
SIZE_LABELS = ["<1Mb", "1-2Mb", "2-3Mb", "3-5Mb", ">5Mb"]
SIZE_EDGES = [1e6, 2e6, 3e6, 5e6]
N_REPLICATES = 500
N_PER_BIN = 150

# WS3.10, results/revision/real_data/reduced_genome/ -- the cohort the manuscript cites.
WS310 = {
    "cohort": "WS3.10 size_stratified (GTDB r220, 150 MAGs per size bin, seed 13100)",
    "n": 750, "n_family_clusters": 361,
    "slope_d_comp_per_log10Mbp": 15.31, "slope_ci95": [12.31, 18.31],
    "median_d_comp": {"<1Mb": -12.60, "1-2Mb": -2.42, "2-3Mb": 0.28, "3-5Mb": 0.65, ">5Mb": 0.64},
    "median_d_cont": {"<1Mb": 0.61, "1-2Mb": -0.07, "2-3Mb": -0.20, "3-5Mb": -0.12, ">5Mb": 0.31},
    "hq_downgrade": {"<1Mb": 0.2733, "1-2Mb": 0.1933, "2-3Mb": 0.1067, "3-5Mb": 0.2067, ">5Mb": 0.2400},
    "pct_HQ_checkm2": {"<1Mb": 40.7, "1-2Mb": 42.0, "2-3Mb": 52.7, "3-5Mb": 64.0, ">5Mb": 78.7},
    "pct_HQ_magicc": {"<1Mb": 14.7, "1-2Mb": 28.0, "2-3Mb": 53.3, "3-5Mb": 56.7, ">5Mb": 63.3},
    "floor_n": {"<1Mb": 3, "1-2Mb": 0, "2-3Mb": 0, "3-5Mb": 0, ">5Mb": 0},
    "source": "results/revision/real_data/reduced_genome/{size_bin_deltas,mimag_threshold_impact}.tsv",
}
SPIRE_5GENERA = {"cohort": "WS3.10 SPIRE v1 five reviewer genera", "n": 591,
                 "floor_n": 19, "floor_rate": 19 / 591}


def mimag(comp, cont):
    out = np.full(comp.shape, "LQ", dtype="<U2")
    out[(comp >= 50.0) & (cont < 10.0)] = "MQ"
    out[(comp >= 90.0) & (cont < 5.0)] = "HQ"
    return out


def load_analysed():
    pred = {}
    for f in ("magicc_predictions.tsv",):
        with (OUT / f).open() as fh:
            next(fh)
            for line in fh:
                a = line.rstrip("\n").split("\t")
                if len(a) == 3:
                    pred[a[0]] = (float(a[1]), float(a[2]))
    gid, size, cc, ck, mc, mk, fam = [], [], [], [], [], [], []
    with gzip.open(OUT / "cohort_definition.tsv.gz", "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            g = r["genome_id"]
            if g not in pred:
                continue
            gid.append(g); size.append(float(r["genome_size"]))
            cc.append(float(r["checkm2_completeness"])); ck.append(float(r["checkm2_contamination"]))
            mc.append(pred[g][0]); mk.append(pred[g][1]); fam.append(r["family"])
    o = {"gid": np.array(gid), "size": np.array(size), "c_comp": np.array(cc),
         "c_cont": np.array(ck), "m_comp": np.array(mc), "m_cont": np.array(mk),
         "family": np.array(fam, dtype=object)}
    o["d_comp"] = o["m_comp"] - o["c_comp"]; o["d_cont"] = o["m_cont"] - o["c_cont"]
    o["c_class"] = mimag(o["c_comp"], o["c_cont"]); o["m_class"] = mimag(o["m_comp"], o["m_cont"])
    o["size_bin"] = np.digitize(o["size"], SIZE_EDGES)
    return o


def replicate_designs(o):
    """Redraw the 150-per-size-bin design from the SPIRE catalogue data, N_REPLICATES times."""
    rng = np.random.default_rng(fw.stable_hash("WS11.S|mimic750|seed"))
    bins = [np.flatnonzero(o["size_bin"] == b) for b in range(5)]
    x = np.log10(o["size"] / 1e6)
    slopes, permed, perhq = [], {l: [] for l in SIZE_LABELS}, {l: [] for l in SIZE_LABELS}
    for _ in range(N_REPLICATES):
        pick = np.concatenate([rng.choice(b, size=min(N_PER_BIN, b.size), replace=False)
                               for b in bins])
        xs, ys = x[pick], o["d_comp"][pick]
        slopes.append(float(np.polyfit(xs, ys, 1)[0]))
        for bi, lab in enumerate(SIZE_LABELS):
            m = pick[o["size_bin"][pick] == bi]
            if m.size == 0:
                continue
            permed[lab].append(float(np.median(o["d_comp"][m])))
            hq = o["c_class"][m] == "HQ"
            perhq[lab].append(float(np.mean(o["m_class"][m][hq] != "HQ")) if hq.sum() else np.nan)
    return slopes, permed, perhq


def q(v):
    a = np.asarray([x for x in v if np.isfinite(x)])
    return (float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) \
        if a.size else (np.nan, np.nan, np.nan)


def read_tsv(path):
    with (OUT / path).open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    o = load_analysed()
    print(f"[234] analysed catalogue-sample genomes: {o['size'].size:,}", flush=True)
    slopes, permed, perhq = replicate_designs(o)
    head = json.loads((OUT / "ws11s_headline.json").read_text())
    strat = read_tsv("classification_change_by_stratum.tsv")

    def get(cohort, grouping, stratum, stat, unit="family"):
        for r in strat:
            if (r["cohort"] == cohort and r["grouping"] == grouping and r["stratum"] == stratum
                    and r["statistic"] == stat and r["clustering_unit"] == unit):
                return float(r["estimate"]), float(r["ci95_lo"]), float(r["ci95_hi"]), int(r["n_genomes"])
        return (np.nan,) * 3 + (0,)

    rows = []
    s_m, s_lo, s_hi = q(slopes)
    cw = head.get("catalogue_weighted", {})
    rows.append(["size_slope_completeness_pp_per_log10Mbp", "WS3.10 750-MAG GTDB size-stratified",
                 750, WS310["slope_d_comp_per_log10Mbp"], *WS310["slope_ci95"],
                 "family-clustered bootstrap; published in WS3.10"])
    rows.append(["size_slope_completeness_pp_per_log10Mbp",
                 "SPIRE catalogue, 150-per-size-bin design re-drawn (n=750 x 500 replicates)",
                 750, s_m, s_lo, s_hi,
                 "spread ACROSS replicate designs, i.e. the sampling distribution of the "
                 "750-genome design applied to SPIRE"])
    e = cw.get("slope_d_comp_per_log10Mbp", {})
    rows.append(["size_slope_completeness_pp_per_log10Mbp", "SPIRE catalogue_weighted",
                 cw.get("n_analysed", 0), e.get("estimate", np.nan),
                 *(e.get("ci95", [np.nan, np.nan])),
                 "design-weighted OLS, family-clustered bootstrap"])
    for lab in SIZE_LABELS:
        m, lo, hi = q(permed[lab])
        rows.append([f"median_delta_completeness[{lab}]",
                     "WS3.10 750-MAG GTDB size-stratified", 150,
                     WS310["median_d_comp"][lab], np.nan, np.nan, "published in WS3.10"])
        rows.append([f"median_delta_completeness[{lab}]",
                     "SPIRE catalogue, 150-per-bin design re-drawn", 150, m, lo, hi,
                     "across-replicate spread"])
        est, l, h, n = get("catalogue_weighted", "size_bin", lab, "mean_d_comp")
        rows.append([f"mean_delta_completeness[{lab}]", "SPIRE catalogue_weighted", n, est, l, h,
                     "design+non-response weighted mean, family-clustered bootstrap"])
        mh, lh, hh = q(perhq[lab])
        rows.append([f"HQ_downgrade_rate[{lab}]", "WS3.10 750-MAG GTDB size-stratified", 150,
                     WS310["hq_downgrade"][lab], np.nan, np.nan, "published in WS3.10"])
        rows.append([f"HQ_downgrade_rate[{lab}]", "SPIRE catalogue, 150-per-bin design re-drawn",
                     150, mh, lh, hh, "across-replicate spread"])
        est, l, h, n = get("catalogue_weighted", "size_bin", lab, "HQ_downgrade_rate")
        rows.append([f"HQ_downgrade_rate[{lab}]", "SPIRE catalogue_weighted", n, est, l, h,
                     "denominator = genomes SPIRE publishes as HQ in this size bin"])
    for stat in ("p_class_change", "HQ_downgrade_rate", "MQ_to_HQ_rate", "MQ_to_LQ_rate",
                 "floor_rate", "mean_d_comp", "mean_d_cont"):
        for cname in ("catalogue_weighted", "catalogue_weighted_no_nr_adjustment",
                      "S3_srs_unweighted", "S1_small_census", "reps_census"):
            est, l, h, n = get(cname, "overall", "ALL", stat)
            if n:
                rows.append([stat, f"SPIRE {cname}", n, est, l, h, ""])
    rows.append(["floor_rate", "WS3.10 SPIRE five reviewer genera", SPIRE_5GENERA["n"],
                 SPIRE_5GENERA["floor_rate"], np.nan, np.nan,
                 "19/591; the previously published floor-censoring figure"])

    with (OUT / "small_vs_large_cohort_comparison.tsv").open("w") as fh:
        fh.write("quantity\tcohort\tn\testimate\tci95_lo\tci95_hi\tnote\n")
        for r in rows:
            fh.write("\t".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"[234] wrote small_vs_large_cohort_comparison.tsv ({len(rows)} rows)", flush=True)

    dist = {"design_replicate_slope_median": s_m, "design_replicate_slope_ci": [s_lo, s_hi],
            "ws310_slope": WS310["slope_d_comp_per_log10Mbp"], "ws310_ci": WS310["slope_ci95"]}
    (OUT / "design_effect.json").write_text(json.dumps(dist, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
