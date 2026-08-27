#!/usr/bin/env python3
"""
WS3.10 — mechanism model + ground-truthed cross-check for the reduced-genome
over-calling hypothesis.

Script 134 established the *phenomenology* (per-genus deltas, size-bin dose response,
marginal/partial Spearman correlations).  This script adds the four things that turn a
phenomenology into a falsification test:

  A. Dose-response quantified as a MODEL, not a bin table.
     OLS of the MAGICC-minus-CheckM2 delta on log10(genome size) with cluster-robust
     (GTDB family) standard errors, on the size-stratified MAG cohort and separately
     on the non-reduced control genera.  R2 = coefficient of determination
     (1 - SS_res/SS_tot), per protocol 4.4d.  Never squared Pearson.

  B. Mechanism separation: genome SIZE vs number of DETECTABLE single-copy core genes.
     Multiple regression on standardised predictors, with
       - cluster-robust CIs on the standardised betas,
       - partial (incremental) R2 for each predictor,
       - nested-model comparison size-only / SCG-only / both,
       - a within-size-bin stratified test, which is the assumption-free version.

  C. The "implied complete genome size" probe -- a direct mechanistic read-out.
     If a completeness estimator is shrinking every genome toward a typical
     training-set genome length L, then size / (completeness/100) should cluster near L
     regardless of the true lineage size.  Comparing that quantity between MAGICC and
     CheckM2 shows *what the model thinks a complete genome is*.

  D. Cross-check against set_C_clean, where GROUND TRUTH EXISTS.
     Same estimator (MAGICC minus CheckM2) computed on 1,000 synthetic Patescibacteriota
     with known truth, so the real-data disagreement can be anchored to a signed error
     against truth.  Clustered by reference genome (100 refs x 10 simulations).

  E. Reduced-relative-to-lineage stratification, and MIMAG-threshold consequences.

Read-only on data/ and on script 133's outputs.  Single process, no network.

Outputs (results/revision/real_data/reduced_genome/)
  mechanism_models.json           everything, machine readable
  mechanism_models.tsv            regression coefficient table
  lineage_relative_size_deltas.tsv
  mimag_threshold_impact.tsv
  set_C_clean_crosscheck.tsv
  fig_mechanism_and_crosscheck.png/.pdf
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTDIR = PROJECT_DIR / "results/revision/real_data/reduced_genome"
PRED = OUTDIR / "magicc_v5_predictions.tsv"
SCG = OUTDIR / "scg_census.tsv"
GTDB = [PROJECT_DIR / "data/gtdb/bac120_metadata.tsv.gz",
        PROJECT_DIR / "data/gtdb/ar53_metadata.tsv.gz"]
SET_C = PROJECT_DIR / "data/benchmarks/set_C_clean"

REVIEWER_GENERA = ["CAG-557", "UMGS1491", "Caccenecus", "HGM10766", "Faecimonas"]
SIZE_BIN_LABELS = ["<1Mb", "1-2Mb", "2-3Mb", "3-5Mb", ">5Mb"]
BOOT = 5000
SEED = 13600

BLUE, SKY, GREY, VERM, YELLOW, BLACK = (
    "#0072B2", "#56B4E9", "#999999", "#D55E00", "#E69F00", "#000000")


# ============================================================== statistics ====
def r2_cod(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination, 1 - SS_res/SS_tot (protocol 4.4d)."""
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def ols_cluster(X: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    """OLS with cluster-robust (CR1) covariance.  X must already include intercept."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    uniq = np.unique(clusters)
    g = len(uniq)
    meat = np.zeros((k, k))
    for c in uniq:
        m = clusters == c
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = XtX_inv @ (scale * meat) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    # t with g-1 df
    df = max(g - 1, 1)
    tcrit = _t_ppf975(df)
    return {
        "beta": beta, "se": se, "n": n, "n_clusters": g,
        "ci_lo": beta - tcrit * se, "ci_hi": beta + tcrit * se,
        "t": np.divide(beta, se, out=np.zeros_like(beta), where=se > 0),
        "p": np.array([2 * _t_sf(abs(t), df) if s > 0 else float("nan")
                       for t, s in zip(np.divide(beta, np.where(se > 0, se, 1)), se)]),
        "r2": r2_cod(y, X @ beta),
    }


def _t_ppf975(df: int) -> float:
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_sf(mid, df) > 0.025:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _t_sf(t: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    val = front * (f - 1.0)
    return val if x < (a + 1) / (a + b + 2) else 1.0 - _betainc(b, a, 1 - x)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    n = len(a)
    if n < 4:
        return float("nan"), float("nan")
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if den == 0:
        return float("nan"), float("nan")
    rho = float((ra * rb).sum() / den)
    if abs(rho) >= 1:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    return rho, 2 * _t_sf(abs(t), n - 2)


def cluster_boot_median(x: np.ndarray, clusters: np.ndarray, n_boot=BOOT, seed=SEED):
    """Median with a cluster-bootstrap 95% CI (resample clusters with replacement)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[uniq[p]] for p in pick])
        meds[b] = np.median(x[sel])
    return (float(np.median(x)), float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


def cluster_boot_mean(x: np.ndarray, clusters: np.ndarray, n_boot=BOOT, seed=SEED):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[uniq[p]] for p in pick])
        out[b] = np.mean(x[sel])
    return (float(np.mean(x)), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)))


def cluster_boot_prop(x: np.ndarray, clusters: np.ndarray, n_boot=BOOT, seed=SEED):
    return cluster_boot_mean(np.asarray(x, float), clusters, n_boot, seed)


# ================================================================== loading ===
def lineage_median_sizes():
    acc = defaultdict(list)
    for path in GTDB:
        with gzip.open(path, "rt") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                try:
                    comp = float(row["checkm2_completeness"])
                    cont = float(row["checkm2_contamination"])
                    size = int(row["genome_size"])
                except (ValueError, KeyError):
                    continue
                if comp < 90 or cont > 5 or size <= 0:
                    continue
                est = size / (comp / 100.0)
                t = row["gtdb_taxonomy"].split(";")
                if len(t) > 1:
                    acc["p:" + t[1]].append(est)
                if len(t) > 4:
                    acc["f:" + t[4]].append(est)
    return {k: float(np.median(v)) for k, v in acc.items() if len(v) >= 20}


def load_real():
    rows = list(csv.DictReader(PRED.open(), delimiter="\t"))
    scg = {}
    for r in csv.DictReader(SCG.open(), delimiter="\t"):
        if not r.get("error") and r.get("n_scg_detected"):
            scg[r["accession"]] = r
    med = lineage_median_sizes()
    keep = []
    for r in rows:
        r["mc"] = float(r["magicc_completeness"])
        r["mx"] = float(r["magicc_contamination"])
        r["cc"] = float(r["checkm2_completeness"])
        r["cx"] = float(r["checkm2_contamination"])
        r["d_comp"] = r["mc"] - r["cc"]
        r["d_cont"] = r["mx"] - r["cx"]
        r["size"] = int(r["genome_size"])
        r["log10_size"] = math.log10(r["size"])
        s = scg.get(r["ncbi_accession"])
        r["n_scg"] = float(s["n_scg_detected"]) if s else None
        r["n_prot"] = float(s["n_proteins"]) if s else None
        t = r["gtdb_taxonomy"].split(";")
        r["phylum"] = t[1] if len(t) > 1 else ""
        r["family"] = t[4] if len(t) > 4 else ""
        r["est_full_size_ckm2"] = r["size"] / max(r["cc"] / 100.0, 1e-6)
        r["est_full_size_magicc"] = r["size"] / max(r["mc"] / 100.0, 1e-6)
        pm = med.get("p:" + r["phylum"])
        r["phylum_median_size"] = pm
        r["log2_size_vs_phylum"] = (math.log2(r["est_full_size_ckm2"] / pm) if pm else None)
        try:
            r["log10_contigs"] = math.log10(max(int(r["contig_count"]), 1))
        except (ValueError, KeyError):
            r["log10_contigs"] = None
        keep.append(r)
    return keep, med


def load_set_c():
    """set_C_clean: truth + MAGICC V5 + CheckM2, merged on genome_id."""
    mag = {r["genome_id"]: r for r in csv.DictReader(
        (SET_C / "magicc_v5_predictions.tsv").open(), delimiter="\t")}
    ck = {r["genome_id"]: r for r in csv.DictReader(
        (SET_C / "checkm2_predictions.tsv").open(), delimiter="\t")}
    out = []
    for gid, m in mag.items():
        c = ck.get(gid)
        if not c:
            continue
        out.append({
            "genome_id": gid,
            "ref": m["dominant_accession"],
            "true_comp": float(m["true_completeness"]),
            "true_cont": float(m["true_contamination"]),
            "m_comp": float(m["pred_completeness"]),
            "m_cont": float(m["pred_contamination"]),
            "c_comp": float(c["pred_completeness"]),
            "c_cont": float(c["pred_contamination"]),
            "total_length": int(m["total_length"]),
            "n_contigs": int(m["n_contigs"]),
        })
    return out


# ================================================================ analyses ====
def dose_response(rows, label, cluster_key="family"):
    """delta ~ 1 + log10(size), cluster-robust by GTDB family."""
    sub = [r for r in rows if r[cluster_key]]
    if len(sub) < 20:
        return []
    X = np.column_stack([np.ones(len(sub)),
                         np.array([r["log10_size"] for r in sub])])
    cl = np.array([r[cluster_key] for r in sub])
    out = []
    for tgt, key in (("delta_completeness", "d_comp"), ("delta_contamination", "d_cont")):
        y = np.array([r[key] for r in sub], float)
        fit = ols_cluster(X, y, cl)
        out.append({
            "analysis": "dose_response_log10size", "pool": label, "target": tgt,
            "n": fit["n"], "n_clusters": fit["n_clusters"],
            "intercept": round(float(fit["beta"][0]), 3),
            "slope_per_log10Mbp": round(float(fit["beta"][1]), 3),
            "slope_ci95": [round(float(fit["ci_lo"][1]), 3), round(float(fit["ci_hi"][1]), 3)],
            "slope_p": f"{float(fit['p'][1]):.3g}",
            "R2_coefficient_of_determination": round(fit["r2"], 4),
        })
    return out


def zscore(v):
    v = np.asarray(v, float)
    s = v.std(ddof=0)
    return (v - v.mean()) / (s if s > 0 else 1.0)


def mechanism_model(rows, label, cluster_key="family"):
    """Standardised multiple regression + incremental R2 + nested comparison."""
    sub = [r for r in rows
           if r["n_scg"] is not None and r["log10_contigs"] is not None
           and r["log2_size_vs_phylum"] is not None and r[cluster_key]]
    if len(sub) < 30:
        return [], []
    names = ["log10_genome_size", "n_scg_detected", "log2_size_vs_phylum", "log10_contig_count"]
    cols = {
        "log10_genome_size": zscore([r["log10_size"] for r in sub]),
        "n_scg_detected": zscore([r["n_scg"] for r in sub]),
        "log2_size_vs_phylum": zscore([r["log2_size_vs_phylum"] for r in sub]),
        "log10_contig_count": zscore([r["log10_contigs"] for r in sub]),
    }
    cl = np.array([r[cluster_key] for r in sub])
    coef_rows, nested_rows = [], []
    for tgt, key in (("delta_completeness", "d_comp"), ("delta_contamination", "d_cont")):
        y = np.array([r[key] for r in sub], float)
        X = np.column_stack([np.ones(len(sub))] + [cols[n] for n in names])
        full = ols_cluster(X, y, cl)
        for i, nm in enumerate(names, start=1):
            drop = [j for j in range(X.shape[1]) if j != i]
            red = ols_cluster(X[:, drop], y, cl)
            coef_rows.append({
                "analysis": "standardised_multiple_regression", "pool": label, "target": tgt,
                "predictor": nm, "n": full["n"], "n_clusters": full["n_clusters"],
                "std_beta_pp_per_SD": round(float(full["beta"][i]), 3),
                "ci95": [round(float(full["ci_lo"][i]), 3), round(float(full["ci_hi"][i]), 3)],
                "p": f"{float(full['p'][i]):.3g}",
                "partial_R2_increment": round(full["r2"] - red["r2"], 4),
                "full_model_R2": round(full["r2"], 4),
            })
        # nested models
        for mlabel, use in (("size_only", ["log10_genome_size"]),
                            ("scg_only", ["n_scg_detected"]),
                            ("size_plus_scg", ["log10_genome_size", "n_scg_detected"]),
                            ("full_4_predictor", names)):
            Xm = np.column_stack([np.ones(len(sub))] + [cols[n] for n in use])
            fm = ols_cluster(Xm, y, cl)
            nested_rows.append({
                "analysis": "nested_model_R2", "pool": label, "target": tgt,
                "model": mlabel, "predictors": "+".join(use),
                "n": fm["n"], "R2_coefficient_of_determination": round(fm["r2"], 4)})
    return coef_rows, nested_rows


def tool_attribution(rows, label, cluster_key="family"):
    """Which INPUT does each tool's own estimate track: assembly size, or SCG count?

    The MAGICC-minus-CheckM2 delta is a difference of two estimators, so a predictor of
    the delta may act through either tool.  Regressing each tool's own prediction on the
    same standardised predictors attributes the dependence to a specific tool, which is
    what makes the mechanism claim about MAGICC rather than about the contrast.
    """
    sub = [r for r in rows if r["n_scg"] is not None and r[cluster_key]]
    if len(sub) < 30:
        return []
    z_size = zscore([r["log10_size"] for r in sub])
    z_scg = zscore([r["n_scg"] for r in sub])
    cl = np.array([r[cluster_key] for r in sub])
    X = np.column_stack([np.ones(len(sub)), z_size, z_scg])
    out = []
    for tool, ck, xk in (("MAGICC_V5", "mc", "mx"), ("CheckM2", "cc", "cx")):
        for metric, key in (("completeness", ck), ("contamination", xk)):
            y = np.array([r[key] for r in sub], float)
            fit = ols_cluster(X, y, cl)
            for i, nm in enumerate(("log10_genome_size", "n_scg_detected"), start=1):
                drop = [j for j in range(X.shape[1]) if j != i]
                red = ols_cluster(X[:, drop], y, cl)
                out.append({
                    "analysis": "tool_attribution", "pool": label, "tool": tool,
                    "target": f"{tool}_{metric}", "predictor": nm,
                    "n": fit["n"], "n_clusters": fit["n_clusters"],
                    "std_beta_pp_per_SD": round(float(fit["beta"][i]), 3),
                    "ci95": [round(float(fit["ci_lo"][i]), 3), round(float(fit["ci_hi"][i]), 3)],
                    "p": f"{float(fit['p'][i]):.3g}",
                    "partial_R2_increment": round(fit["r2"] - red["r2"], 4),
                    "full_model_R2": round(fit["r2"], 4)})
    return out


def set_c_size_response(sc):
    """Ground-truthed size response: does MAGICC's SIGNED ERROR grow as assemblies shrink?

    set_C_clean has known truth, so this is not a tool-vs-tool contrast.  Clustered by
    dominant reference genome.
    """
    out = []
    cl = np.array([r["ref"] for r in sc])
    size = np.array([r["total_length"] for r in sc], float)
    ncont = np.array([r["n_contigs"] for r in sc], float)
    for tool, cp, ct in (("MAGICC_V5", "m_comp", "m_cont"), ("CheckM2", "c_comp", "c_cont")):
        for metric, key in (("completeness", cp), ("contamination", ct)):
            err = np.array([r[key] - r["true_comp" if metric == "completeness" else "true_cont"]
                            for r in sc], float)
            rho_s, p_s = spearman(size, err)
            rho_c, p_c = spearman(ncont, err)
            X = np.column_stack([np.ones(len(sc)), zscore(np.log10(size))])
            fit = ols_cluster(X, err, cl)
            out.append({
                "analysis": "set_C_clean_size_response", "tool": tool, "target": metric,
                "n": len(sc), "n_reference_clusters": int(len(np.unique(cl))),
                "rho_assembly_size": round(rho_s, 4), "p_size": f"{p_s:.3g}",
                "rho_n_contigs": round(rho_c, 4), "p_contigs": f"{p_c:.3g}",
                "std_beta_pp_per_SD_log10size": round(float(fit["beta"][1]), 3),
                "ci95": [round(float(fit["ci_lo"][1]), 3), round(float(fit["ci_hi"][1]), 3)],
                "p": f"{float(fit['p'][1]):.3g}"})
    return out


def within_bin_scg(rows):
    """Assumption-free version: inside each size bin, does n_scg still predict delta?"""
    out = []
    edges = [(0, 1e6, "<1Mb"), (1e6, 2e6, "1-2Mb"), (2e6, 3e6, "2-3Mb"),
             (3e6, 5e6, "3-5Mb"), (5e6, 1e12, ">5Mb")]
    for lo, hi, lab in edges:
        sub = [r for r in rows if lo <= r["size"] < hi and r["n_scg"] is not None]
        if len(sub) < 25:
            continue
        nscg = np.array([r["n_scg"] for r in sub])
        size = np.array([r["size"] for r in sub], float)
        for tgt, key in (("delta_completeness", "d_comp"), ("delta_contamination", "d_cont")):
            y = np.array([r[key] for r in sub])
            rho_s, p_s = spearman(nscg, y)
            rho_z, p_z = spearman(size, y)
            out.append({"analysis": "within_size_bin", "bin": lab, "n": len(sub), "target": tgt,
                        "rho_n_scg_detected": round(rho_s, 4), "p_n_scg": f"{p_s:.3g}",
                        "rho_genome_size": round(rho_z, 4), "p_size": f"{p_z:.3g}"})
    return out


def implied_size_probe(rows, med):
    """What does each tool think a COMPLETE genome of this organism is?"""
    out = []
    groups = {"reviewer_genera": [r for r in rows if r["cohort"] == "reviewer_genera"]}
    for lab in SIZE_BIN_LABELS:
        groups[f"size_stratified:{lab}"] = [
            r for r in rows if r["cohort"] == "size_stratified" and r["stratum"] == lab]
    groups["control_genera"] = [r for r in rows if r["cohort"] == "control_genera"]
    for lab, sub in groups.items():
        if not sub:
            continue
        im = np.array([r["est_full_size_magicc"] for r in sub]) / 1e6
        ic = np.array([r["est_full_size_ckm2"] for r in sub]) / 1e6
        obs = np.array([r["size"] for r in sub]) / 1e6
        ph = np.array([r["phylum_median_size"] for r in sub
                       if r["phylum_median_size"]]) / 1e6
        out.append({
            "analysis": "implied_complete_genome_size_Mbp", "group": lab, "n": len(sub),
            "observed_size_median": round(float(np.median(obs)), 3),
            "implied_by_MAGICC_median": round(float(np.median(im)), 3),
            "implied_by_CheckM2_median": round(float(np.median(ic)), 3),
            "lineage_phylum_median": round(float(np.median(ph)), 3) if len(ph) else None,
            "implied_by_MAGICC_IQR": [round(float(np.percentile(im, 25)), 3),
                                      round(float(np.percentile(im, 75)), 3)],
            "implied_by_CheckM2_IQR": [round(float(np.percentile(ic, 25)), 3),
                                       round(float(np.percentile(ic, 75)), 3)],
        })
    # dispersion over the whole panel: does MAGICC shrink every genome to one length?
    im = np.array([r["est_full_size_magicc"] for r in rows]) / 1e6
    ic = np.array([r["est_full_size_ckm2"] for r in rows]) / 1e6
    obs = np.array([r["size"] for r in rows]) / 1e6
    out.append({
        "analysis": "implied_complete_genome_size_Mbp", "group": "ALL", "n": len(rows),
        "observed_size_median": round(float(np.median(obs)), 3),
        "implied_by_MAGICC_median": round(float(np.median(im)), 3),
        "implied_by_CheckM2_median": round(float(np.median(ic)), 3),
        "lineage_phylum_median": None,
        "implied_by_MAGICC_IQR": [round(float(np.percentile(im, 25)), 3),
                                  round(float(np.percentile(im, 75)), 3)],
        "implied_by_CheckM2_IQR": [round(float(np.percentile(ic, 25)), 3),
                                   round(float(np.percentile(ic, 75)), 3)],
    })
    return out


def lineage_relative(rows):
    bins = [(-99, -1.0, "<-1 (>=2x reduced)"), (-1.0, -0.5, "-1 to -0.5"),
            (-0.5, -0.25, "-0.5 to -0.25"), (-0.25, 0.25, "-0.25 to +0.25 (typical)"),
            (0.25, 99, ">+0.25 (larger than lineage)")]
    out = []
    for lo, hi, lab in bins:
        sub = [r for r in rows if r["log2_size_vs_phylum"] is not None
               and lo <= r["log2_size_vs_phylum"] < hi]
        if len(sub) < 10:
            continue
        cl = np.array([r["family"] or r["phylum"] for r in sub])
        dc = np.array([r["d_comp"] for r in sub])
        dx = np.array([r["d_cont"] for r in sub])
        mc, mcl, mch = cluster_boot_median(dc, cl)
        mx, mxl, mxh = cluster_boot_median(dx, cl)
        out.append({
            "log2_size_vs_phylum_bin": lab, "n": len(sub),
            "n_family_clusters": int(len(np.unique(cl))),
            "genome_size_median_Mbp": round(float(np.median([r["size"] for r in sub])) / 1e6, 3),
            "delta_completeness_median": round(mc, 2),
            "delta_completeness_ci95": [round(mcl, 2), round(mch, 2)],
            "delta_contamination_median": round(mx, 2),
            "delta_contamination_ci95": [round(mxl, 2), round(mxh, 2)],
            "pct_magicc_cont_ge5_while_checkm2_lt5": round(100 * float(np.mean(
                [(r["mx"] >= 5) and (r["cx"] < 5) for r in sub])), 1),
        })
    return out


def mimag_impact(rows):
    groups = {"reviewer_genera(all 5)": [r for r in rows if r["cohort"] == "reviewer_genera"]}
    for g in REVIEWER_GENERA:
        groups[g] = [r for r in rows if r["genus"] == g]
    for lab in SIZE_BIN_LABELS:
        groups[f"size_stratified:{lab}"] = [
            r for r in rows if r["cohort"] == "size_stratified" and r["stratum"] == lab]
    groups["control_genera(all 8)"] = [r for r in rows if r["cohort"] == "control_genera"]
    out = []
    for lab, sub in groups.items():
        if not sub:
            continue
        cl = np.array([r["family"] or r["phylum"] for r in sub])
        ff = np.array([(r["cx"] < 5) and (r["mx"] >= 5) for r in sub], float)
        hq_c = np.array([(r["cc"] >= 90) and (r["cx"] < 5) for r in sub], float)
        hq_m = np.array([(r["mc"] >= 90) and (r["mx"] >= 0) and (r["mx"] < 5) for r in sub], float)
        downg = np.array([((r["cc"] >= 90) and (r["cx"] < 5)) and
                          not ((r["mc"] >= 90) and (r["mx"] < 5)) for r in sub], float)
        p, lo, hi = cluster_boot_prop(ff, cl)
        d, dlo, dhi = cluster_boot_prop(downg, cl)
        out.append({
            "group": lab, "n": len(sub),
            "pct_HQ_by_CheckM2": round(100 * float(hq_c.mean()), 1),
            "pct_HQ_by_MAGICC": round(100 * float(hq_m.mean()), 1),
            "false_fail_rate_cont5_vs_CheckM2": round(float(p), 4),
            "false_fail_ci95": [round(lo, 4), round(hi, 4)],
            "HQ_downgrade_rate_vs_CheckM2": round(float(d), 4),
            "HQ_downgrade_ci95": [round(dlo, 4), round(dhi, 4)],
            "n_at_magicc_completeness_floor_50": int(sum(1 for r in sub if r["mc"] <= 50.5)),
        })
    return out


def set_c_crosscheck(sc):
    """Ground-truthed anchor.  Clustered by reference genome (100 refs x 10 sims)."""
    strata = {
        "ALL set_C_clean": lambda r: True,
        "clean only (true cont < 5%)": lambda r: r["true_cont"] < 5,
        "clean AND HQ-truth (cont<5, comp>=90)": lambda r: r["true_cont"] < 5 and r["true_comp"] >= 90,
        "in-domain (cont <= comp)": lambda r: r["true_cont"] <= r["true_comp"],
    }
    out = []
    for lab, fn in strata.items():
        sub = [r for r in sc if fn(r)]
        if len(sub) < 10:
            continue
        cl = np.array([r["ref"] for r in sub])
        rec = {"stratum": lab, "n": len(sub), "n_reference_clusters": int(len(np.unique(cl))),
               "assembly_size_median_Mbp": round(
                   float(np.median([r["total_length"] for r in sub])) / 1e6, 3)}
        for tag, a, b in (("MAGICC_minus_truth", "m", "true"),
                          ("CheckM2_minus_truth", "c", "true"),
                          ("MAGICC_minus_CheckM2", "m", "c")):
            for metric in ("comp", "cont"):
                key_a = f"{a}_{metric}" if a != "true" else f"true_{metric}"
                key_b = f"{b}_{metric}" if b != "true" else f"true_{metric}"
                d = np.array([r[key_a] - r[key_b] for r in sub], float)
                mn, lo, hi = cluster_boot_mean(d, cl)
                md, mlo, mhi = cluster_boot_median(d, cl)
                rec[f"{tag}_{metric}_mean"] = round(mn, 3)
                rec[f"{tag}_{metric}_mean_ci95"] = [round(lo, 3), round(hi, 3)]
                rec[f"{tag}_{metric}_median"] = round(md, 3)
                rec[f"{tag}_{metric}_median_ci95"] = [round(mlo, 3), round(mhi, 3)]
        ff = np.array([(r["true_cont"] < 5) and (r["m_cont"] >= 5) for r in sub], float)
        ffc = np.array([(r["true_cont"] < 5) and (r["c_cont"] >= 5) for r in sub], float)
        p, plo, phi = cluster_boot_prop(ff, cl)
        q, qlo, qhi = cluster_boot_prop(ffc, cl)
        rec["MAGICC_false_fail_rate_cont5"] = round(float(p), 4)
        rec["MAGICC_false_fail_ci95"] = [round(plo, 4), round(phi, 4)]
        rec["CheckM2_false_fail_rate_cont5"] = round(float(q), 4)
        rec["CheckM2_false_fail_ci95"] = [round(qlo, 4), round(qhi, 4)]
        out.append(rec)
    return out


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


# ==================================================================== figure ===
def figure(rows, lin_rel, cross, implied):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[136] matplotlib unavailable; skipping figure")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    # (a) implied complete genome size: MAGICC vs CheckM2 vs observed
    ax = axes[0]
    obs = np.array([r["size"] for r in rows]) / 1e6
    im = np.array([r["est_full_size_magicc"] for r in rows]) / 1e6
    ic = np.array([r["est_full_size_ckm2"] for r in rows]) / 1e6
    ax.scatter(obs, ic, s=10, c=SKY, marker="o", alpha=.5, linewidths=0, label="CheckM2")
    ax.scatter(obs, im, s=10, c=VERM, marker="^", alpha=.5, linewidths=0, label="MAGICC V5")
    lim = [0.3, 12]
    ax.plot(lim, lim, ls="--", c=BLACK, lw=.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("observed assembly size (Mbp)")
    ax.set_ylabel("implied COMPLETE genome size (Mbp)\nsize / (completeness/100)")
    ax.set_title("(a) what each tool thinks a complete genome is")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # (b) delta vs size RELATIVE TO LINEAGE
    ax = axes[1]
    labs = [b["log2_size_vs_phylum_bin"] for b in lin_rel]
    xs = np.arange(len(labs))
    for key, ci, col, mk, lab in (
            ("delta_contamination_median", "delta_contamination_ci95", VERM, "^", "contamination"),
            ("delta_completeness_median", "delta_completeness_ci95", BLUE, "o", "completeness")):
        med = [b[key] for b in lin_rel]
        lo = [b[key] - b[ci][0] for b in lin_rel]
        hi = [b[ci][1] - b[key] for b in lin_rel]
        ax.errorbar(xs, med, yerr=[lo, hi], color=col, marker=mk, ms=7, lw=1.6,
                    capsize=3, label=lab)
    ax.axhline(0, color=BLACK, lw=.9, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([l.split(" ")[0] for l in labs], fontsize=8)
    ax.set_xlabel("log2(estimated complete size / phylum median size)")
    ax.set_ylabel("median MAGICC − CheckM2 (pp), 95% CI")
    ax.set_title("(b) reduction RELATIVE TO LINEAGE drives the effect")
    ax.legend(fontsize=8, frameon=False)

    # (c) ground-truthed anchor on set_C_clean
    ax = axes[2]
    rec = next((r for r in cross if r["stratum"].startswith("clean only")), cross[0])
    names = ["completeness", "contamination"]
    series = [("MAGICC − truth", "MAGICC_minus_truth", VERM),
              ("CheckM2 − truth", "CheckM2_minus_truth", SKY),
              ("MAGICC − CheckM2", "MAGICC_minus_CheckM2", GREY)]
    w = 0.26
    for i, (lab, key, col) in enumerate(series):
        vals = [rec[f"{key}_comp_mean"], rec[f"{key}_cont_mean"]]
        errs = [[vals[0] - rec[f"{key}_comp_mean_ci95"][0],
                 vals[1] - rec[f"{key}_cont_mean_ci95"][0]],
                [rec[f"{key}_comp_mean_ci95"][1] - vals[0],
                 rec[f"{key}_cont_mean_ci95"][1] - vals[1]]]
        ax.bar(np.arange(2) + (i - 1) * w, vals, width=w, color=col, label=lab)
        ax.errorbar(np.arange(2) + (i - 1) * w, vals, yerr=errs, fmt="none",
                    ecolor=BLACK, capsize=3, lw=1)
    ax.axhline(0, color=BLACK, lw=.9)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(names)
    ax.set_ylabel("signed error (pp), cluster-bootstrap 95% CI")
    ax.set_title(f"(c) ground truth: set_C_clean, {rec['stratum']}\n(n={rec['n']}, "
                 f"{rec['n_reference_clusters']} reference clusters)")
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig_mechanism_and_crosscheck.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[136] wrote {OUTDIR}/fig_mechanism_and_crosscheck.png|.pdf")


# ====================================================================== main ===
def main() -> int:
    rows, med = load_real()
    print(f"[136] real cohorts: {len(rows)} genomes; "
          f"{sum(1 for r in rows if r['n_scg'] is not None)} with SCG census", flush=True)

    size_strat = [r for r in rows if r["cohort"] == "size_stratified"]
    ctrl = [r for r in rows if r["cohort"] == "control_genera"]
    non_rev = [r for r in rows if r["cohort"] != "reviewer_genera"]

    dose = (dose_response(size_strat, "size_stratified_MAGs")
            + dose_response(ctrl, "non_reduced_control_genera")
            + dose_response(non_rev, "size_stratified+controls")
            + dose_response(rows, "all_cohorts"))

    coef_a, nest_a = mechanism_model(size_strat, "size_stratified_MAGs")
    coef_b, nest_b = mechanism_model(non_rev, "size_stratified+controls")
    coef_c, nest_c = mechanism_model(rows, "all_cohorts")
    coefs, nested = coef_a + coef_b + coef_c, nest_a + nest_b + nest_c

    wb = within_bin_scg(rows)
    attrib = (tool_attribution(size_strat, "size_stratified_MAGs")
              + tool_attribution(non_rev, "size_stratified+controls")
              + tool_attribution(rows, "all_cohorts"))
    implied = implied_size_probe(rows, med)
    lin_rel = lineage_relative(rows)
    mim = mimag_impact(rows)

    sc = load_set_c()
    print(f"[136] set_C_clean merged rows: {len(sc)}", flush=True)
    cross = set_c_crosscheck(sc)
    sc_size = set_c_size_response(sc)

    write_tsv(OUTDIR / "mechanism_models.tsv",
              dose + coefs + nested + wb + attrib + implied + sc_size)
    write_tsv(OUTDIR / "lineage_relative_size_deltas.tsv", lin_rel)
    write_tsv(OUTDIR / "mimag_threshold_impact.tsv", mim)
    write_tsv(OUTDIR / "set_C_clean_crosscheck.tsv", cross)

    summary = {
        "script": "scripts/206_reduced_genome_mechanism_and_crosscheck.py",
        "seed": SEED, "n_bootstrap": BOOT,
        "R2_convention": "coefficient of determination, 1 - SS_res/SS_tot (protocol 4.4d)",
        "sign_convention": "all real-data deltas are MAGICC minus CheckM2",
        "clustering": "GTDB family for real cohorts; dominant reference genome for set_C_clean",
        "magicc_completeness_floor": 50.0,
        "n_real_genomes": len(rows),
        "dose_response": dose,
        "mechanism_coefficients": coefs,
        "nested_model_R2": nested,
        "within_size_bin": wb,
        "tool_attribution": attrib,
        "set_C_clean_size_response": sc_size,
        "implied_complete_genome_size": implied,
        "lineage_relative_size": lin_rel,
        "mimag_threshold_impact": mim,
        "set_C_clean_crosscheck": cross,
    }
    (OUTDIR / "mechanism_models.json").write_text(json.dumps(summary, indent=2))

    figure(rows, lin_rel, cross, implied)

    # ------------------------------------------------------------ console ----
    print("\n=== A. DOSE RESPONSE (delta ~ log10 genome size, cluster-robust by family) ===")
    for d in dose:
        print(f"  {d['pool']:30} {d['target']:20} slope={d['slope_per_log10Mbp']:+7.2f} pp/decade "
              f"CI{d['slope_ci95']} p={d['slope_p']:>9} R2={d['R2_coefficient_of_determination']:+.3f} "
              f"(n={d['n']}, {d['n_clusters']} clusters)")

    print("\n=== B. MECHANISM: standardised betas (pp per SD) ===")
    for c in coefs:
        print(f"  {c['pool']:28} {c['target']:20} {c['predictor']:22} "
              f"beta={c['std_beta_pp_per_SD']:+7.3f} CI{c['ci95']} p={c['p']:>9} "
              f"partialR2={c['partial_R2_increment']:+.4f}")
    print("\n  nested model R2 (coefficient of determination):")
    for n in nested:
        print(f"    {n['pool']:28} {n['target']:20} {n['model']:18} "
              f"R2={n['R2_coefficient_of_determination']:+.4f}")

    print("\n=== B2. WITHIN SIZE BIN ===")
    for w in wb:
        print(f"  {w['bin']:8} {w['target']:20} n={w['n']:4} "
              f"rho(nSCG)={w['rho_n_scg_detected']:+.3f} p={w['p_n_scg']:>9}   "
              f"rho(size)={w['rho_genome_size']:+.3f} p={w['p_size']:>9}")

    print("\n=== B3. TOOL ATTRIBUTION (each tool's OWN estimate ~ z(log10 size) + z(n_SCG)) ===")
    for a in attrib:
        print(f"  {a['pool']:28} {a['target']:28} {a['predictor']:20} "
              f"beta={a['std_beta_pp_per_SD']:+7.3f} CI{a['ci95']} p={a['p']:>9} "
              f"partialR2={a['partial_R2_increment']:+.4f} fullR2={a['full_model_R2']:+.4f}")

    print("\n=== B4. GROUND-TRUTHED SIZE RESPONSE ON set_C_clean (error vs TRUTH) ===")
    for s in sc_size:
        print(f"  {s['tool']:10} {s['target']:14} rho(size)={s['rho_assembly_size']:+.3f} "
              f"p={s['p_size']:>10}  rho(contigs)={s['rho_n_contigs']:+.3f} p={s['p_contigs']:>10}  "
              f"beta/SD={s['std_beta_pp_per_SD_log10size']:+7.3f} CI{s['ci95']} p={s['p']:>10}")

    print("\n=== C. IMPLIED COMPLETE GENOME SIZE (Mbp) ===")
    for i in implied:
        print(f"  {i['group']:28} n={i['n']:4} observed={i['observed_size_median']:6.2f}  "
              f"MAGICC={i['implied_by_MAGICC_median']:6.2f} IQR{i['implied_by_MAGICC_IQR']}  "
              f"CheckM2={i['implied_by_CheckM2_median']:6.2f} IQR{i['implied_by_CheckM2_IQR']}  "
              f"lineage={i['lineage_phylum_median']}")

    print("\n=== E. REDUCTION RELATIVE TO LINEAGE ===")
    for b in lin_rel:
        print(f"  {b['log2_size_vs_phylum_bin']:32} n={b['n']:4} size={b['genome_size_median_Mbp']:5.2f}Mb "
              f"dComp={b['delta_completeness_median']:+7.2f}{b['delta_completeness_ci95']} "
              f"dCont={b['delta_contamination_median']:+7.2f}{b['delta_contamination_ci95']} "
              f"falseFail={b['pct_magicc_cont_ge5_while_checkm2_lt5']:5.1f}%")

    print("\n=== E2. MIMAG THRESHOLD IMPACT ===")
    for m in mim:
        print(f"  {m['group']:28} n={m['n']:4} HQ_CkM2={m['pct_HQ_by_CheckM2']:5.1f}% "
              f"HQ_MAGICC={m['pct_HQ_by_MAGICC']:5.1f}% "
              f"falseFail={m['false_fail_rate_cont5_vs_CheckM2']:.3f}{m['false_fail_ci95']} "
              f"HQdowngrade={m['HQ_downgrade_rate_vs_CheckM2']:.3f} "
              f"at_floor={m['n_at_magicc_completeness_floor_50']}")

    print("\n=== D. set_C_clean CROSS-CHECK (ground truth exists) ===")
    for r in cross:
        print(f"\n  --- {r['stratum']} (n={r['n']}, {r['n_reference_clusters']} refs, "
              f"median assembly {r['assembly_size_median_Mbp']} Mbp) ---")
        for tag in ("MAGICC_minus_truth", "CheckM2_minus_truth", "MAGICC_minus_CheckM2"):
            print(f"      {tag:22} comp {r[tag+'_comp_mean']:+7.2f} {r[tag+'_comp_mean_ci95']}"
                  f"   cont {r[tag+'_cont_mean']:+7.2f} {r[tag+'_cont_mean_ci95']}")
        print(f"      false-fail @5% cont: MAGICC {r['MAGICC_false_fail_rate_cont5']:.3f} "
              f"{r['MAGICC_false_fail_ci95']}  CheckM2 {r['CheckM2_false_fail_rate_cont5']:.3f} "
              f"{r['CheckM2_false_fail_ci95']}")

    print(f"\n[136] wrote {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
