#!/usr/bin/env python3
"""
WS3.10 mitigation diagnostic — PART 1: mechanism demonstration.

Establishes, on ground-truthed data, that MAGICC V5's completeness head reads
ABSOLUTE assembly size rather than lineage-relative core content.

Three demonstrations, in increasing directness:

  M1  Matched-true-completeness test.  For a correctly specified estimator the
      partial effect of assembly size on the *predicted* completeness, holding
      TRUE completeness fixed, is exactly zero.  We regress
          pred_comp ~ 1 + true_comp + true_comp^2 + z(log10 assembly bp)
      with reference-clustered (CR1) SEs and report the standardised beta and
      the partial R^2 of the size term.  Same model for CheckM2.

  M2  Implied-reference-length recovery.  Completeness = retained_bp / L.
      The estimator therefore implies  L_hat = retained_bp / (pred_comp/100).
      retained_bp is ground truth here, so L_hat is the estimator's own belief
      about the reference length.  Regressing log10 L_hat on log10 L_true gives
      a recovery slope lambda; lambda = 1 means reference length is recovered,
      lambda < 1 means the belief is shrunk toward a central prior.  Reported
      per size stratum as the median L_hat / L_true ratio as well.

  M3  Which observable carries absolute size.  Correlation of the 7 k-mer
      summary features (the model's non-compositional input branch) with
      log10 assembly bp, computed on the frozen V5 test split.

Outputs -> results/revision/real_data/reduced_genome/mitigation/

R^2 is the coefficient of determination throughout (1 - SS_res/SS_tot).
Clusters = dominant reference genome accession.  Seeds via fw.stable_hash.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util

def _load_framework():
    spec = importlib.util.spec_from_file_location(
        "magicc_metrics_framework", ROOT / "scripts" / "101_metrics_framework.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["magicc_metrics_framework"] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()

OUT = ROOT / "results" / "revision" / "real_data" / "reduced_genome" / "mitigation"
OUT.mkdir(parents=True, exist_ok=True)

SETS = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]
N_BOOT = 5000
CI = 0.95


# --------------------------------------------------------------------------
# cluster-robust OLS (CR1) — same implementation as script 136
# --------------------------------------------------------------------------
def r2_cod(y, yhat):
    y = np.asarray(y, float)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _t_sf(t, df):
    from math import atan, sqrt, pi, gamma
    # regularised incomplete beta via continued fraction
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    from math import lgamma, exp, log
    if x <= 0:
        ib = 0.0
    elif x >= 1:
        ib = 1.0
    else:
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = exp(a * log(x) + b * log(1 - x) - lbeta)
        if x < (a + 1) / (a + b + 2):
            ib = front * betacf(a, b, x) / a
        else:
            ib = 1.0 - exp(b * log(1 - x) + a * log(x) - lbeta) * betacf(b, a, 1 - x) / b
    return 0.5 * ib


def _t_ppf975(df):
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_sf(mid, df) > 0.025:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ols_cluster(X, y, clusters):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
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
    df = max(g - 1, 1)
    tc = _t_ppf975(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, beta / np.where(se > 0, se, 1), np.nan)
    return {
        "beta": beta, "se": se, "n": n, "n_clusters": g,
        "ci_lo": beta - tc * se, "ci_hi": beta + tc * se, "t": tvals,
        "p": np.array([2 * _t_sf(abs(t), df) if np.isfinite(t) else np.nan for t in tvals]),
        "r2": r2_cod(y, X @ beta),
    }


def zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0)


def cluster_boot(values, clusters, stat, seed, n_boot=N_BOOT):
    """Percentile CI of stat(values[idx]) resampling clusters with replacement."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([idx_by[j] for j in picks[b]])
        out[b] = stat(values[idx])
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(stat(values)), float(lo), float(hi)


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------
def load_pooled():
    frames = []
    for s in SETS:
        base = ROOT / "data" / "benchmarks" / s
        md = pd.read_csv(base / "metadata.tsv", sep="\t")
        keep = ["genome_id", "true_completeness", "true_contamination",
                "dominant_accession", "dominant_phylum", "n_contigs", "total_length"]
        md = md[keep].copy()
        mg = pd.read_csv(base / "magicc_v5_predictions.tsv", sep="\t")[
            ["genome_id", "pred_completeness", "pred_contamination"]]
        mg.columns = ["genome_id", "magicc_comp", "magicc_cont"]
        ck = pd.read_csv(base / "checkm2_predictions.tsv", sep="\t")[
            ["genome_id", "pred_completeness", "pred_contamination"]]
        ck.columns = ["genome_id", "checkm2_comp", "checkm2_cont"]
        d = md.merge(mg, on="genome_id").merge(ck, on="genome_id")
        d["set"] = s
        d["genome_key"] = s + ":" + d["genome_id"]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    # exact identity, verified to 1e-8 against generation_metadata dominant_reference_bp:
    #   total_length = retained_bp + contaminant_bp,  comp = 100*retained/L, cont = 100*cont_bp/L
    tot = d["true_completeness"] + d["true_contamination"]
    d["ref_len_true"] = 100.0 * d["total_length"] / tot
    d["retained_bp"] = d["total_length"] * d["true_completeness"] / tot
    d["log10_asm_mbp"] = np.log10(d["total_length"] / 1e6)
    d["log10_ref_mbp"] = np.log10(d["ref_len_true"] / 1e6)
    return d


def size_stratum(mbp):
    b = pd.cut(mbp, [0, 1, 1.5, 2, 3, 5, np.inf],
               labels=["<1 Mbp", "1-1.5 Mbp", "1.5-2 Mbp", "2-3 Mbp", "3-5 Mbp", ">5 Mbp"])
    return b


# --------------------------------------------------------------------------
# M1 — matched true completeness
# --------------------------------------------------------------------------
def m1_matched_completeness(d):
    """Effect of size on the PREDICTED completeness holding TRUE completeness
    (and true contamination) fixed.  For an unbiased estimator this is 0.

    Controlling for true contamination matters: assembly bp = L*(comp+cont)/100,
    so an uncontrolled 'assembly size' term partly measures contaminant load,
    which inflates marker-based completeness (CheckM2) for reasons unrelated to
    genome size.  The reference-length specification with contamination
    controlled is therefore the primary one.
    """
    rows = []
    subsets = {
        "all": np.ones(len(d), bool),
        "low_contamination(true<5)": (d["true_contamination"] < 5).values,
        "near_complete(true_comp>=90)": (d["true_completeness"] >= 90).values,
        "clean_AND_near_complete": ((d["true_contamination"] < 5) &
                                    (d["true_completeness"] >= 90)).values,
    }
    for sname, mask in subsets.items():
        s = d[mask]
        if len(s) < 60:
            continue
        cl = s["dominant_accession"].values
        tc = s["true_completeness"].values
        tk = s["true_contamination"].values
        near_const = tc.std() < 1e-6
        for tool, col in [("MAGICC_V5", "magicc_comp"), ("CheckM2", "checkm2_comp")]:
            y = s[col].values
            for size_var, lab in [("log10_ref_mbp", "z_log10_reference_Mbp"),
                                  ("log10_asm_mbp", "z_log10_assembly_Mbp")]:
                zs = zscore(s[size_var].values)
                ctrl = [np.ones(len(s))]
                if not near_const:
                    ctrl += [tc, tc ** 2]
                if tk.std() > 1e-6:
                    ctrl += [tk]
                Xred = np.column_stack(ctrl)
                Xfull = np.column_stack(ctrl + [zs])
                ffull = ols_cluster(Xfull, y, cl)
                fred = ols_cluster(Xred, y, cl)
                j = Xfull.shape[1] - 1
                rows.append({
                    "analysis": "M1_matched_true_completeness",
                    "subset": sname, "tool": tool, "size_predictor": lab,
                    "controls": "true_comp, true_comp^2, true_cont" if not near_const
                                else "true_cont",
                    "n": len(s), "n_reference_clusters": s["dominant_accession"].nunique(),
                    "beta_pp_per_SD": round(float(ffull["beta"][j]), 4),
                    "ci95": f"[{ffull['ci_lo'][j]:.3f}, {ffull['ci_hi'][j]:.3f}]",
                    "p": f"{ffull['p'][j]:.3e}",
                    "partial_R2_increment": round(ffull["r2"] - fred["r2"], 4),
                    "full_model_R2_CoD": round(ffull["r2"], 4),
                    "expected_beta_if_unbiased": 0.0,
                })
    # binned version: signed error inside narrow true-completeness bands
    # x REFERENCE-length stratum, restricted to low contamination so that the
    # size axis is genome size and not contaminant load.
    binned = []
    d = d.copy()
    d["comp_band"] = pd.cut(d["true_completeness"], [0, 60, 70, 80, 90, 95, 100.01],
                            labels=["<60", "60-70", "70-80", "80-90", "90-95", "95-100"])
    d["ref_stratum"] = size_stratum(d["ref_len_true"] / 1e6)
    for cont_lab, cont_mask in [("all", np.ones(len(d), bool)),
                                ("true_cont<5", (d["true_contamination"] < 5).values)]:
        sub = d[cont_mask]
        for band, g in sub.groupby("comp_band", observed=True):
            for st, gg in g.groupby("ref_stratum", observed=True):
                if len(gg) < 15:
                    continue
                seed = fw.stable_hash(f"m1binned|{cont_lab}|{band}|{st}") % (2 ** 31)
                em, lo_m, hi_m = cluster_boot(gg["magicc_comp"] - gg["true_completeness"],
                                              gg["dominant_accession"], np.mean, seed)
                ec, lo_c, hi_c = cluster_boot(gg["checkm2_comp"] - gg["true_completeness"],
                                              gg["dominant_accession"], np.mean, seed + 1)
                binned.append({
                    "contamination_subset": cont_lab,
                    "true_completeness_band": str(band),
                    "reference_size_stratum": str(st),
                    "n": len(gg), "n_reference_clusters": gg["dominant_accession"].nunique(),
                    "true_comp_mean": round(float(gg["true_completeness"].mean()), 2),
                    "MAGICC_signed_comp_error_pp": round(em, 3),
                    "MAGICC_ci95": f"[{lo_m:.2f}, {hi_m:.2f}]",
                    "CheckM2_signed_comp_error_pp": round(ec, 3),
                    "CheckM2_ci95": f"[{lo_c:.2f}, {hi_c:.2f}]",
                })
    return pd.DataFrame(rows), pd.DataFrame(binned)


# --------------------------------------------------------------------------
# M2 — implied reference length
# --------------------------------------------------------------------------
def m2_implied_reference_length(d):
    d = d.copy()
    # implied reference length from each estimator's own completeness call
    for tool, col in [("MAGICC_V5", "magicc_comp"), ("CheckM2", "checkm2_comp")]:
        c = np.clip(d[col].values, 1.0, None) / 100.0
        d[f"implied_L_{tool}"] = d["retained_bp"].values / c
    rows = []
    subsets = {
        "all": np.ones(len(d), bool),
        "low_contamination(true<5)": (d["true_contamination"] < 5).values,
    }
    for sname, mask in subsets.items():
        s = d[mask]
        if len(s) < 60:
            continue
        cl = s["dominant_accession"].values
        xs = np.log10(s["ref_len_true"].values)
        for tool in ["MAGICC_V5", "CheckM2"]:
            y = np.log10(s[f"implied_L_{tool}"].values)
            X = np.column_stack([np.ones(len(s)), xs])
            f = ols_cluster(X, y, cl)
            rows.append({
                "analysis": "M2_implied_reference_length_recovery",
                "subset": sname, "tool": tool, "n": len(s),
                "n_reference_clusters": s["dominant_accession"].nunique(),
                "recovery_slope_lambda": round(float(f["beta"][1]), 4),
                "ci95": f"[{f['ci_lo'][1]:.3f}, {f['ci_hi'][1]:.3f}]",
                "p": f"{f['p'][1]:.3e}",
                "R2_CoD": round(f["r2"], 4),
                "ideal_lambda": 1.0,
                "pct_shrinkage_per_decade_of_L": round(100 * (1 - float(f["beta"][1])), 3),
            })
    # per-stratum ratios (median and mean), low-contamination subset separated
    strat = []
    d["ref_stratum"] = size_stratum(d["ref_len_true"] / 1e6)
    for cont_lab, cont_mask in [("all", np.ones(len(d), bool)),
                                ("true_cont<5", (d["true_contamination"] < 5).values)]:
        sub = d[cont_mask]
        for st, g in sub.groupby("ref_stratum", observed=True):
            if len(g) < 15:
                continue
            seed = fw.stable_hash(f"m2ratio|{cont_lab}|{st}") % (2 ** 31)
            rm, lom, him = cluster_boot(g["implied_L_MAGICC_V5"] / g["ref_len_true"],
                                        g["dominant_accession"], np.mean, seed)
            rc, loc, hic = cluster_boot(g["implied_L_CheckM2"] / g["ref_len_true"],
                                        g["dominant_accession"], np.mean, seed + 1)
            strat.append({
                "contamination_subset": cont_lab,
                "reference_size_stratum": str(st), "n": len(g),
                "n_reference_clusters": g["dominant_accession"].nunique(),
                "ref_len_median_Mbp": round(float(g["ref_len_true"].median() / 1e6), 3),
                "MAGICC_implied_over_true_mean": round(rm, 4),
                "MAGICC_ci95": f"[{lom:.3f}, {him:.3f}]",
                "CheckM2_implied_over_true_mean": round(rc, 4),
                "CheckM2_ci95": f"[{loc:.3f}, {hic:.3f}]",
                "ideal": 1.0,
            })
    return pd.DataFrame(rows), pd.DataFrame(strat), d


# --------------------------------------------------------------------------
# M3 — which observable carries absolute size (frozen V5 test split)
# --------------------------------------------------------------------------
def m3_feature_size_channel():
    import h5py
    from magicc.assembly_stats import FEATURE_NAMES
    p = ROOT / "data" / "features" / "magicc_v5_features.h5"
    rows = []
    with h5py.File(p, "r") as f:
        n = f["test/assembly_features"].shape[0]
        rng = np.random.default_rng(fw.stable_hash("m3_feature_probe") % (2 ** 31))
        idx = np.sort(rng.choice(n, size=min(40000, n), replace=False))
        A = f["test/assembly_features"][idx]           # normalised, as fed to the model
        meta = f["test/metadata"][idx]
    comp = meta["completeness"].astype(float)
    cont = meta["contamination"].astype(float)
    L = meta["genome_full_length"].astype(float)
    asm_bp = L * (comp + cont) / 100.0
    ok = np.isfinite(asm_bp) & (asm_bp > 0)
    x = np.log10(asm_bp[ok])
    for j, name in enumerate(FEATURE_NAMES):
        v = A[ok, j].astype(float)
        r = np.corrcoef(v, x)[0, 1] if v.std() > 0 else np.nan
        # also: correlation with true completeness and with reference length
        rc = np.corrcoef(v, comp[ok])[0, 1] if v.std() > 0 else np.nan
        rl = np.corrcoef(v, np.log10(L[ok]))[0, 1] if v.std() > 0 else np.nan
        rows.append({
            "analysis": "M3_summary_feature_size_channel",
            "feature": name, "n": int(ok.sum()),
            "pearson_r_vs_log10_assembly_bp": round(float(r), 4),
            "pearson_r_vs_true_completeness": round(float(rc), 4),
            "pearson_r_vs_log10_reference_bp": round(float(rl), 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# M4 — is it SIZE, or is it Patescibacteriota?  (disambiguation)
# --------------------------------------------------------------------------
def m4_size_vs_lineage(d):
    """Within the clean + near-complete regime (where the reduced-genome
    false-fail lives), separate 'small genome' from 'that one small phylum'."""
    s = d[(d["true_contamination"] < 5) & (d["true_completeness"] >= 95)].copy()
    s["is_patesci"] = s["dominant_phylum"] == "Patescibacteriota"
    s["ref_stratum"] = size_stratum(s["ref_len_true"] / 1e6)
    rows = []
    for st, g in s.groupby("ref_stratum", observed=True):
        for lab, gg in [("Patescibacteriota", g[g.is_patesci]),
                        ("other phyla", g[~g.is_patesci])]:
            if len(gg) < 8:
                continue
            seed = fw.stable_hash(f"m4|{st}|{lab}") % (2 ** 31)
            ec, lc, hc = cluster_boot(gg["magicc_comp"] - gg["true_completeness"],
                                      gg["dominant_accession"], np.mean, seed)
            ek, lk, hk = cluster_boot(gg["magicc_cont"] - gg["true_contamination"],
                                      gg["dominant_accession"], np.mean, seed + 1)
            rows.append({
                "analysis": "M4_size_vs_lineage (clean & near-complete)",
                "reference_size_stratum": str(st), "lineage": lab, "n": len(gg),
                "n_reference_clusters": gg["dominant_accession"].nunique(),
                "phyla": ", ".join(f"{k}:{v}" for k, v in
                                   gg["dominant_phylum"].value_counts().head(3).items()),
                "MAGICC_comp_bias_pp": round(ec, 3), "comp_ci95": f"[{lc:.2f}, {hc:.2f}]",
                "MAGICC_cont_bias_pp": round(ek, 3), "cont_ci95": f"[{lk:.2f}, {hk:.2f}]",
            })
    # regression: does size still matter once the phylum indicator is in?
    reg = []
    if len(s) >= 100:
        cl = s["dominant_accession"].values
        zs = zscore(np.log10(s["ref_len_true"].values / 1e6))
        pat = s["is_patesci"].values.astype(float)
        for tool, col, tcol in [("MAGICC_V5", "magicc_comp", "true_completeness"),
                                ("CheckM2", "checkm2_comp", "true_completeness")]:
            y = (s[col] - s[tcol]).values
            for name, X in [("size only", np.column_stack([np.ones(len(s)), zs])),
                            ("phylum only", np.column_stack([np.ones(len(s)), pat])),
                            ("size + phylum", np.column_stack([np.ones(len(s)), zs, pat]))]:
                f = ols_cluster(X, y, cl)
                reg.append({
                    "analysis": "M4_regression_signed_comp_error",
                    "tool": tool, "model": name, "n": len(s),
                    "n_reference_clusters": s["dominant_accession"].nunique(),
                    "beta_size_pp_per_SD": round(float(f["beta"][1]), 4)
                    if name != "phylum only" else "",
                    "size_ci95": f"[{f['ci_lo'][1]:.3f}, {f['ci_hi'][1]:.3f}]"
                    if name != "phylum only" else "",
                    "beta_Patescibacteriota_pp": round(float(f["beta"][-1]), 4)
                    if name != "size only" else "",
                    "patesci_ci95": f"[{f['ci_lo'][-1]:.3f}, {f['ci_hi'][-1]:.3f}]"
                    if name != "size only" else "",
                    "R2_CoD": round(f["r2"], 4),
                })
    return pd.DataFrame(rows), pd.DataFrame(reg)


# --------------------------------------------------------------------------
# M5 — are the completeness under-call and the contamination over-call the
#      SAME event, or two?  (the recalibration result depends on this)
# --------------------------------------------------------------------------
def m5_comp_cont_coupling(d):
    rows = []
    d = d.copy()
    d["ref_stratum"] = size_stratum(d["ref_len_true"] / 1e6)
    d["err_c"] = d["magicc_comp"] - d["true_completeness"]
    d["err_k"] = d["magicc_cont"] - d["true_contamination"]
    for lab, s in [("clean & near-complete, ref <2 Mbp",
                    d[(d.true_contamination < 5) & (d.true_completeness >= 95) &
                      (d.ref_len_true < 2e6)]),
                   ("clean & near-complete, ref >=2 Mbp",
                    d[(d.true_contamination < 5) & (d.true_completeness >= 95) &
                      (d.ref_len_true >= 2e6)]),
                   ("clean & near-complete, ALL",
                    d[(d.true_contamination < 5) & (d.true_completeness >= 95)])]:
        if len(s) < 30:
            continue
        x, y = s["err_c"].values, s["err_k"].values
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        rho = float(np.corrcoef(rx, ry)[0, 1])
        seed = fw.stable_hash(f"m5|{lab}") % (2 ** 31)
        # cluster bootstrap of Spearman rho
        cl = s["dominant_accession"].values
        uniq, inv = np.unique(cl, return_inverse=True)
        idx_by = [np.where(inv == i)[0] for i in range(len(uniq))]
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(uniq), size=(2000, len(uniq)))
        bs = np.empty(2000)
        for b in range(2000):
            ii = np.concatenate([idx_by[j] for j in picks[b]])
            bs[b] = np.corrcoef(pd.Series(x[ii]).rank().values,
                                pd.Series(y[ii]).rank().values)[0, 1]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        # does the contamination over-call survive conditioning on the
        # completeness error?
        X = np.column_stack([np.ones(len(s)), x])
        f = ols_cluster(X, y, cl)
        rows.append({
            "analysis": "M5_completeness_contamination_coupling",
            "subset": lab, "n": len(s),
            "n_reference_clusters": s["dominant_accession"].nunique(),
            "spearman_rho_err_comp_vs_err_cont": round(rho, 4),
            "rho_ci95": f"[{lo:.3f}, {hi:.3f}]",
            "mean_comp_error_pp": round(float(x.mean()), 3),
            "mean_cont_error_pp": round(float(y.mean()), 3),
            "cont_error_intercept_at_zero_comp_error_pp": round(float(f["beta"][0]), 3),
            "intercept_ci95": f"[{f['ci_lo'][0]:.3f}, {f['ci_hi'][0]:.3f}]",
            "slope_cont_per_comp_error": round(float(f["beta"][1]), 4),
            "R2_CoD": round(f["r2"], 4),
        })
    return pd.DataFrame(rows)


def main():
    d = load_pooled()
    d.to_csv(OUT / "pooled_groundtruth_predictions.tsv", sep="\t", index=False)
    print(f"[140] pooled n={len(d)} refs={d.dominant_accession.nunique()}")

    m1, m1b = m1_matched_completeness(d)
    m1.to_csv(OUT / "mechanism_m1_matched_completeness.tsv", sep="\t", index=False)
    m1b.to_csv(OUT / "mechanism_m1_binned_signed_error.tsv", sep="\t", index=False)
    print(m1.to_string(index=False))

    m2, m2s, d2 = m2_implied_reference_length(d)
    m2.to_csv(OUT / "mechanism_m2_implied_reference_length.tsv", sep="\t", index=False)
    m2s.to_csv(OUT / "mechanism_m2_implied_length_by_stratum.tsv", sep="\t", index=False)
    print(m2.to_string(index=False))
    print(m2s.to_string(index=False))

    m5 = m5_comp_cont_coupling(d)
    m5.to_csv(OUT / "mechanism_m5_comp_cont_coupling.tsv", sep="\t", index=False)
    print(m5.to_string(index=False))

    m4, m4r = m4_size_vs_lineage(d)
    m4.to_csv(OUT / "mechanism_m4_size_vs_lineage.tsv", sep="\t", index=False)
    m4r.to_csv(OUT / "mechanism_m4_regression.tsv", sep="\t", index=False)
    print(m4.to_string(index=False))
    print(m4r.to_string(index=False))

    try:
        m3 = m3_feature_size_channel()
        m3.to_csv(OUT / "mechanism_m3_feature_size_channel.tsv", sep="\t", index=False)
        print(m3.to_string(index=False))
    except Exception as e:  # pragma: no cover
        print(f"[140] M3 skipped: {e}")
        m3 = pd.DataFrame()

    summary = {
        "script": "scripts/212_mitigation_mechanism.py",
        "n_samples": int(len(d)),
        "n_reference_clusters": int(d.dominant_accession.nunique()),
        "sets": SETS,
        "n_boot": N_BOOT,
        "m1": m1.to_dict("records"),
        "m2": m2.to_dict("records"),
        "m2_by_stratum": m2s.to_dict("records"),
        "m3": m3.to_dict("records") if len(m3) else [],
        "m4_size_vs_lineage": m4.to_dict("records"),
        "m4_regression": m4r.to_dict("records"),
        "m5_comp_cont_coupling": m5.to_dict("records"),
    }
    (OUT / "mechanism_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[140] wrote -> {OUT}")


if __name__ == "__main__":
    main()
