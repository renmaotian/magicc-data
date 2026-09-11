#!/usr/bin/env python
"""
106_domain_size_and_gunc.py  --  WS5.8 / WS5.9 / WS5.10
=======================================================
Three analyses that scripts 102-105 do not cover, built on the same framework
(``scripts/101_metrics_framework.py``) and the same config so that every CI is a
cluster bootstrap over the dominant reference genome and every R^2 is the
coefficient of determination.

WS5.8  TRAINING-DOMAIN RESTRICTION (protocol section 4.4a).
       The ``contaminant_bp <= dominant_actual_bp`` cap entered the generator on
       2026-03-14, after Sets C/D/E were built, so those sets contain samples
       with contamination% > completeness% that lie OUTSIDE the region V5 was
       trained on. Every set is therefore reported three ways -- ``all``,
       ``in_domain`` (contamination <= completeness, PRIMARY) and
       ``out_of_domain`` (robustness only) -- for accuracy, MIMAG-inspired
       classification and the QC decision thresholds. The two subsets are never
       silently pooled.

WS5.9  GENOME-SIZE / CORE-GENE DEPENDENCE OF MAGICC'S ERROR (protocol 3.10).
       MAGICC over-calls contamination and under-calls completeness on reduced
       genomes. This quantifies the effect as a function of the DOMINANT
       reference genome's size across every set, for MAGICC and for the
       competitors (so it can be shown to be tool-specific or not), and -- when
       the core-gene census is available -- separates genome size from the
       number of detectable single-copy core genes, which would implicate the
       feature space rather than the k-mer count scale.

WS5.10 GUNC AS A CONTAMINATION *DETECTION* COMPARATOR (protocol 4.4c item 5).
       GUNC emits a clade-separation score and a pass/fail call, not
       percentages, so it must never appear in the MAE table. Reported instead:
       pass/fail rates stratified by true contamination, agreement with the
       truth at the 5% and 10% MIMAG contamination thresholds, and the Spearman
       correlation of CSS against true contamination alongside the same
       correlation for MAGICC's and CheckM2's predicted contamination.

    python scripts/106_domain_size_and_gunc.py [--config ...] [--n-boot 2000]
        [--analyses domain size gunc] [--scg-census] [--threads 6]

``--scg-census`` runs Prodigal + hmmsearch --cut_tc over the UNIQUE dominant
reference genomes and caches the per-reference count of detectable single-copy
core-gene families; it is slow (~1 h at 6 threads for ~1,650 genomes) and is
skipped unless requested. The size analysis consumes the cache when present.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent


def _load_framework():
    spec = importlib.util.spec_from_file_location(
        "magicc_metrics_framework", _HERE / "101_metrics_framework.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["magicc_metrics_framework"] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
from scipy import stats as sps   # noqa: E402

METRICS = ("completeness", "contamination")
LEAKAGE_FREE_5 = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]
MANUSCRIPT_5 = ["set_A_v2", "set_B_v2", "set_C", "set_D", "set_E"]

# Protocol 3.10 bins, in bp.
SIZE_EDGES = [0.0, 1e6, 2e6, 3e6, 5e6, np.inf]
SIZE_LABELS = ["<1 Mbp", "1-2 Mbp", "2-3 Mbp", "3-5 Mbp", ">=5 Mbp"]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def bootstrapper(df: pd.DataFrame, cfg, n_boot: int, salt: int) -> "fw.Bootstrapper":
    bs = fw.Bootstrapper(clusters=df["cluster_id"].to_numpy(), n_iter=n_boot,
                         ci_level=cfg.ci_level, seed=cfg.seed + salt)
    bs.resample_indices()
    return bs


def accuracy_block(df: pd.DataFrame, tool: str, metric: str,
                   bs: "fw.Bootstrapper") -> Optional[dict]:
    """MAE / bias / RMSE / R^2 (CoD) / r^2 (Pearson, separately labelled)."""
    tcol, pcol = f"true_{metric}", f"pred_{metric}__{tool}"
    if pcol not in df.columns:
        return None
    y = df[tcol].to_numpy(float)
    yh = df[pcol].to_numpy(float)
    ok = ~(np.isnan(y) | np.isnan(yh))
    if ok.sum() == 0:
        return None
    var = float(np.var(y[ok]))
    b = bs.ci_multi(lambda idx, _y=y, _h=yh: {
        "mae": fw.mae(_y[idx], _h[idx]),
        "bias": fw.bias(_y[idx], _h[idx]),
        "rmse": fw.rmse(_y[idx], _h[idx]),
        "r2": fw.r2_coefficient_of_determination(_y[idx], _h[idx]),
        "r2_pearson_sq": fw.r2_pearson_squared(_y[idx], _h[idx])})
    rec = {"n": int(ok.sum()), "n_clusters": bs.n_clusters,
           "clustered_bootstrap": bool(bs.clustered),
           "true_value_variance": var, "true_value_constant": var == 0.0}
    for k in ("mae", "bias", "rmse"):
        rec[k] = b[k]["estimate"]
        rec[f"{k}_ci_lo"] = b[k]["ci_lo"]
        rec[f"{k}_ci_hi"] = b[k]["ci_hi"]
    if var == 0.0:
        rec.update({"r2": np.nan, "r2_ci_lo": np.nan, "r2_ci_hi": np.nan,
                    "r2_pearson_sq": np.nan,
                    "r2_omitted_reason": (
                        f"R^2 omitted: the true {metric} is constant at "
                        f"{y[ok][0]:.4g}% for all {int(ok.sum())} genomes, so the "
                        f"total sum of squares is zero and no fraction of variance "
                        f"can be explained. MAE / RMSE / mean signed error remain "
                        f"well defined and are reported instead.")})
    else:
        rec.update({"r2": b["r2"]["estimate"], "r2_ci_lo": b["r2"]["ci_lo"],
                    "r2_ci_hi": b["r2"]["ci_hi"],
                    "r2_pearson_sq": b["r2_pearson_sq"]["estimate"],
                    "r2_omitted_reason": ""})
    return rec


def load_all_sets(cfg, names: Sequence[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for b in fw.discover_sets(cfg, include_missing=False):
        if b.name not in names:
            continue
        df, tools, _ = fw.load_set(cfg, b)
        df = df.copy()
        df["set"] = b.name
        df["set_label"] = b.label
        df["status"] = b.status
        df["tier"] = b.tier
        # ---- integrity: every prediction must have joined -------------------
        for t in tools:
            n_ok = int(df[f"pred_completeness__{t}"].notna().sum())
            if n_ok != len(df):
                raise RuntimeError(
                    f"JOIN FAILURE {b.name}/{t}: only {n_ok}/{len(df)} genomes "
                    f"carry a prediction after the merge")
        df.attrs["tools"] = tools
        out[b.name] = df
    return out


def domain_mask(df: pd.DataFrame) -> np.ndarray:
    """Training domain of V4/V5: contamination% <= completeness%."""
    return (df["true_contamination"].to_numpy(float)
            <= df["true_completeness"].to_numpy(float) + 1e-9)


# ---------------------------------------------------------------------------
# WS5.8  training-domain restriction
# ---------------------------------------------------------------------------


def analysis_domain(cfg, frames: Dict[str, pd.DataFrame], n_boot: int):
    acc_rows, cls_rows, thr_rows = [], [], []
    labels = cfg.mimag_classes
    tcfg = cfg.raw["thresholds"]

    def one(df, setname, setlabel, status, tier, subset, salt):
        if len(df) == 0:
            return
        tools = [t for t in cfg.tools if f"pred_completeness__{t}" in df.columns
                 and df[f"pred_completeness__{t}"].notna().any()]
        bs = bootstrapper(df, cfg, n_boot, salt)
        base = {"set": setname, "set_label": setlabel, "status": status,
                "tier": tier, "subset": subset}
        for tool in tools:
            for metric in METRICS:
                r = accuracy_block(df, tool, metric, bs)
                if r is None:
                    continue
                acc_rows.append({**base, "tool": tool,
                                 "tool_label": cfg.tool_label(tool),
                                 "tool_role": cfg.tool_role(tool),
                                 "metric": metric, **r})
            # ---- MIMAG-inspired classification --------------------------
            tc = fw.encode_labels(df["true_mimag"].to_numpy(), labels)
            pc = fw.encode_labels(df[f"mimag__{tool}"].to_numpy(), labels)
            k = len(labels)
            m = fw.metrics_from_cm(fw.cm_from_codes(tc, pc, k), labels)
            b = bs.ci_multi(lambda idx, _t=tc, _p=pc: {
                "macro_f1": fw.metrics_from_cm(
                    fw.cm_from_codes(_t[idx], _p[idx], k), labels)["macro_f1"],
                "accuracy": fw.metrics_from_cm(
                    fw.cm_from_codes(_t[idx], _p[idx], k), labels)["accuracy"]})
            hi = labels.index("high")
            cls_rows.append({
                **base, "tool": tool, "tool_label": cfg.tool_label(tool),
                "n": len(df), "n_clusters": bs.n_clusters,
                "n_true_high": int((tc == hi).sum()),
                "n_true_medium": int((tc == labels.index("medium")).sum()),
                "n_true_low": int((tc == labels.index("low")).sum()),
                "macro_f1": m["macro_f1"],
                "macro_f1_ci_lo": b["macro_f1"]["ci_lo"],
                "macro_f1_ci_hi": b["macro_f1"]["ci_hi"],
                "accuracy": m["accuracy"],
                "accuracy_ci_lo": b["accuracy"]["ci_lo"],
                "accuracy_ci_hi": b["accuracy"]["ci_hi"],
                "high_precision": float(m["precision"][hi]),
                "high_recall": float(m["recall"][hi]),
                "high_f1": float(m["f1"][hi])})
            # ---- decision thresholds -------------------------------------
            for crit, taus in (("contamination", tcfg["contamination"]),
                               ("completeness", tcfg["completeness"])):
                y = df[f"true_{crit}"].to_numpy(float)
                yh = df[f"pred_{crit}__{tool}"].to_numpy(float)
                for tau in taus:
                    tm = fw.threshold_metrics(y, yh, float(tau), crit)
                    bb = bs.ci_multi(lambda idx, _y=y, _h=yh, _t=float(tau), _c=crit: {
                        "fpr": fw.threshold_metrics(_y[idx], _h[idx], _t, _c)["false_pass_rate"],
                        "ffr": fw.threshold_metrics(_y[idx], _h[idx], _t, _c)["false_fail_rate"]})
                    thr_rows.append({
                        **base, "tool": tool, "criterion": crit, "tau": float(tau),
                        "n": tm["n"], "n_clusters": bs.n_clusters,
                        "n_true_pass": tm["n_true_pass"], "n_true_fail": tm["n_true_fail"],
                        "n_false_pass": tm["n_false_pass"], "n_false_fail": tm["n_false_fail"],
                        "false_pass_rate": tm["false_pass_rate"],
                        "false_pass_rate_ci_lo": bb["fpr"]["ci_lo"],
                        "false_pass_rate_ci_hi": bb["fpr"]["ci_hi"],
                        "false_fail_rate": tm["false_fail_rate"],
                        "false_fail_rate_ci_lo": bb["ffr"]["ci_lo"],
                        "false_fail_rate_ci_hi": bb["ffr"]["ci_hi"]})

    for i, (name, df) in enumerate(frames.items()):
        ind = domain_mask(df)
        one(df, name, df["set_label"].iloc[0], df["status"].iloc[0],
            df["tier"].iloc[0], "all", 100 + i)
        one(df[ind], name, df["set_label"].iloc[0], df["status"].iloc[0],
            df["tier"].iloc[0], "in_domain", 200 + i)
        if (~ind).sum() >= 10:
            one(df[~ind], name, df["set_label"].iloc[0], df["status"].iloc[0],
                df["tier"].iloc[0], "out_of_domain", 300 + i)

    # ---- pooled leakage-free five sets, three ways -------------------------
    if set(LEAKAGE_FREE_5) <= set(frames):
        common = None
        for s in LEAKAGE_FREE_5:
            ts = {t for t in cfg.tools
                  if f"pred_completeness__{t}" in frames[s].columns}
            common = ts if common is None else (common & ts)
        common = [t for t in cfg.tools if t in common]
        keep_base = ["cluster_id", "true_completeness", "true_contamination",
                     "true_mimag", "set"]
        for variant, restrict in (("all", "none"),
                                  ("E_in_domain", "set_E"),
                                  ("all_in_domain", "every_set")):
            parts = []
            for s in LEAKAGE_FREE_5:
                d = frames[s]
                if restrict == "every_set" or (restrict == "set_E" and s == "set_E"):
                    d = d[domain_mask(d)]
                cols = keep_base + [c for t in common for c in
                                    (f"pred_completeness__{t}", f"pred_contamination__{t}",
                                     f"mimag__{t}")]
                sub = d[cols].copy()
                sub["cluster_id"] = s + "|" + sub["cluster_id"].astype(str)
                parts.append(sub)
            big = pd.concat(parts, ignore_index=True)
            one(big, "POOLED_leakage_free_5_sets", "POOLED: " + ", ".join(LEAKAGE_FREE_5),
                "pooled", "pooled", variant, 400 + len(variant))

    return (pd.DataFrame(acc_rows), pd.DataFrame(cls_rows), pd.DataFrame(thr_rows))


# ---------------------------------------------------------------------------
# WS5.9  genome-size dependence
# ---------------------------------------------------------------------------


def gtdb_size_map(cfg) -> pd.DataFrame:
    p = cfg.project_root / "data" / "gtdb" / "filtered_genomes.tsv"
    g = pd.read_csv(p, sep="\t",
                    usecols=["gtdb_accession", "ncbi_accession", "gcf_accession",
                             "genome_size", "contig_count", "n50_contigs",
                             "domain", "phylum"])
    rows = []
    for col in ("gtdb_accession", "ncbi_accession", "gcf_accession"):
        s = g[[col, "genome_size", "domain", "phylum"]].rename(columns={col: "acc"})
        rows.append(s[s.acc.notna() & (s.acc.astype(str) != "")])
    m = pd.concat(rows, ignore_index=True).drop_duplicates(subset="acc")
    return m.set_index("acc")


def attach_size(frames: Dict[str, pd.DataFrame], smap: pd.DataFrame) -> pd.DataFrame:
    """Long frame: one row per (genome, tool) with the dominant reference size."""
    out = []
    for name, df in frames.items():
        acc = df["dominant_accession"].astype(str)
        sz = acc.map(smap["genome_size"])
        n_missing = int(sz.isna().sum())
        if n_missing:
            raise RuntimeError(f"{name}: {n_missing} dominant accessions have no "
                               f"GTDB genome_size -- refusing to proceed")
        d = df.copy()
        d["dominant_genome_size"] = sz.to_numpy(float)
        d["log10_size_mbp"] = np.log10(d["dominant_genome_size"] / 1e6)
        d["size_bin"] = pd.cut(d["dominant_genome_size"], bins=SIZE_EDGES,
                               labels=SIZE_LABELS, right=False)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def analysis_size(cfg, frames: Dict[str, pd.DataFrame], n_boot: int,
                  scg: Optional[pd.DataFrame]):
    smap = gtdb_size_map(cfg)
    big = attach_size(frames, smap)
    tcfg = cfg.raw["thresholds"]

    bin_rows, corr_rows = [], []

    def per_frame(d: pd.DataFrame, scope: str, salt: int):
        tools = [t for t in cfg.tools if f"pred_completeness__{t}" in d.columns
                 and d[f"pred_completeness__{t}"].notna().any()]
        # ---- binned ------------------------------------------------------
        for j, lab in enumerate(SIZE_LABELS):
            sub = d[d["size_bin"] == lab]
            if len(sub) < 20:
                continue
            bs = bootstrapper(sub, cfg, n_boot, salt + 7 * j)
            for tool in tools:
                rec = {"scope": scope, "size_bin": lab, "tool": tool,
                       "tool_label": cfg.tool_label(tool), "n": len(sub),
                       "n_clusters": bs.n_clusters,
                       "median_size_mbp": float(sub.dominant_genome_size.median() / 1e6)}
                for metric in METRICS:
                    a = accuracy_block(sub, tool, metric, bs)
                    if a is None:
                        continue
                    rec[f"mae_{metric}"] = a["mae"]
                    rec[f"mae_{metric}_ci_lo"] = a["mae_ci_lo"]
                    rec[f"mae_{metric}_ci_hi"] = a["mae_ci_hi"]
                    rec[f"bias_{metric}"] = a["bias"]
                    rec[f"bias_{metric}_ci_lo"] = a["bias_ci_lo"]
                    rec[f"bias_{metric}_ci_hi"] = a["bias_ci_hi"]
                # the reduced-genome failure mode: false-fail at 5% contamination
                y = sub["true_contamination"].to_numpy(float)
                yh = sub[f"pred_contamination__{tool}"].to_numpy(float)
                tm = fw.threshold_metrics(y, yh, 5.0, "contamination")
                bb = bs.ci_multi(lambda idx, _y=y, _h=yh: {
                    "ffr": fw.threshold_metrics(_y[idx], _h[idx], 5.0,
                                                "contamination")["false_fail_rate"]})
                rec.update({"n_true_pass_cont5": tm["n_true_pass"],
                            "n_false_fail_cont5": tm["n_false_fail"],
                            "false_fail_rate_cont5": tm["false_fail_rate"],
                            "false_fail_rate_cont5_ci_lo": bb["ffr"]["ci_lo"],
                            "false_fail_rate_cont5_ci_hi": bb["ffr"]["ci_hi"]})
                bin_rows.append(rec)
        # ---- continuous: Spearman vs log10 size ---------------------------
        bs = bootstrapper(d, cfg, n_boot, salt + 991)
        x = d["log10_size_mbp"].to_numpy(float)
        for tool in tools:
            for metric in METRICS:
                for kind in ("abs", "signed"):
                    e = d[f"{'abs_' if kind == 'abs' else ''}err_{metric}__{tool}"].to_numpy(float)
                    ok = ~(np.isnan(x) | np.isnan(e))
                    if ok.sum() < 30:
                        continue
                    rho = float(sps.spearmanr(x[ok], e[ok]).statistic)
                    ci = bs.ci(lambda idx, _x=x, _e=e: float(
                        sps.spearmanr(_x[idx], _e[idx]).statistic))
                    p, _ = bs.p_two_sided(lambda idx, _x=x, _e=e: float(
                        sps.spearmanr(_x[idx], _e[idx]).statistic), null=0.0)
                    corr_rows.append({
                        "scope": scope, "tool": tool, "tool_label": cfg.tool_label(tool),
                        "metric": metric, "error_kind": kind,
                        "covariate": "log10_dominant_genome_size_mbp",
                        "n": int(ok.sum()), "n_clusters": bs.n_clusters,
                        "spearman_rho": rho, "rho_ci_lo": ci["ci_lo"],
                        "rho_ci_hi": ci["ci_hi"], "p_cluster_bootstrap": p})

    per_frame(big[big.set.isin(LEAKAGE_FREE_5)].copy(), "POOLED_leakage_free_5_sets", 1000)
    for i, s in enumerate(LEAKAGE_FREE_5):
        if s in frames:
            per_frame(big[big.set == s].copy(), s, 2000 + 100 * i)

    # ---- genome size vs detectable core genes ------------------------------
    scg_rows = []
    if scg is not None and len(scg):
        ref = big.groupby("dominant_accession").agg(
            genome_size=("dominant_genome_size", "first"),
            domain=("dominant_domain", "first"),
            phylum=("dominant_phylum", "first"),
            n=("genome_id", "size")).reset_index()
        ref = ref.merge(scg, left_on="dominant_accession", right_on="accession",
                        how="inner")
        if len(ref) >= 30:
            ref["log10_size_mbp"] = np.log10(ref.genome_size / 1e6)
            # per-reference MAGICC error, the unit at which both covariates live
            pr = (big[big.set.isin(LEAKAGE_FREE_5)]
                  .groupby("dominant_accession")
                  .agg(mae_comp=("abs_err_completeness__magicc_v5", "mean"),
                       mae_cont=("abs_err_contamination__magicc_v5", "mean"),
                       bias_comp=("err_completeness__magicc_v5", "mean"),
                       bias_cont=("err_contamination__magicc_v5", "mean"))
                  .reset_index())
            ref = ref.merge(pr, on="dominant_accession", how="inner")
            for target in ("mae_comp", "mae_cont", "bias_comp", "bias_cont"):
                y = ref[target].to_numpy(float)
                a = ref["log10_size_mbp"].to_numpy(float)
                b = ref["n_scg_detected"].to_numpy(float)
                ok = ~(np.isnan(y) | np.isnan(a) | np.isnan(b))
                if ok.sum() < 30:
                    continue

                def partial(u, v, w):
                    ru = sps.rankdata(u); rv = sps.rankdata(v); rw = sps.rankdata(w)
                    r_uv = np.corrcoef(ru, rv)[0, 1]
                    r_uw = np.corrcoef(ru, rw)[0, 1]
                    r_vw = np.corrcoef(rv, rw)[0, 1]
                    den = np.sqrt((1 - r_uw ** 2) * (1 - r_vw ** 2))
                    return float((r_uv - r_uw * r_vw) / den) if den > 0 else np.nan

                scg_rows.append({
                    "target": target, "n_references": int(ok.sum()),
                    "rho_size": float(sps.spearmanr(a[ok], y[ok]).statistic),
                    "p_size": float(sps.spearmanr(a[ok], y[ok]).pvalue),
                    "rho_n_scg": float(sps.spearmanr(b[ok], y[ok]).statistic),
                    "p_n_scg": float(sps.spearmanr(b[ok], y[ok]).pvalue),
                    "rho_size_given_n_scg": partial(y[ok], a[ok], b[ok]),
                    "rho_n_scg_given_size": partial(y[ok], b[ok], a[ok]),
                    "rho_size_vs_n_scg": float(sps.spearmanr(a[ok], b[ok]).statistic)})
    return pd.DataFrame(bin_rows), pd.DataFrame(corr_rows), pd.DataFrame(scg_rows), big


# ---------------------------------------------------------------------------
# WS5.10  GUNC detection comparator
# ---------------------------------------------------------------------------


GUNC_CANDIDATES = ["gunc_predictions.tsv", "gunc_normalized.tsv"]


def find_gunc(cfg, setname: str) -> Optional[Path]:
    roots = [cfg.benchmark_root / setname,
             cfg.project_root / "results" / "revision" / "benchmark" / "gunc" / setname,
             cfg.project_root / "results" / "revision" / "gunc" / setname]
    for r in roots:
        for f in GUNC_CANDIDATES:
            p = r / f
            if p.exists():
                return p
    return None


def analysis_gunc(cfg, frames: Dict[str, pd.DataFrame], n_boot: int):
    rows, corr, notes = [], [], []
    for name, df in frames.items():
        p = find_gunc(cfg, name)
        if p is None:
            continue
        g = pd.read_csv(p, sep="\t")
        idc = next((c for c in ("genome_id", "genome", "genome_name") if c in g.columns), None)
        css = next((c for c in ("clade_separation_score_adjusted", "css",
                                "clade_separation_score") if c in g.columns), None)
        passc = next((c for c in ("pass.GUNC", "pass_GUNC", "gunc_pass", "pass") if c in g.columns), None)
        if idc is None or css is None:
            notes.append(f"{name}: {p} lacks a usable id/CSS column ({list(g.columns)[:10]})")
            continue
        sub = g[[c for c in (idc, css, passc) if c]].rename(
            columns={idc: "genome_id", css: "gunc_css"})
        if passc:
            sub = sub.rename(columns={passc: "gunc_pass_raw"})
        n_before = len(df)
        d = df.merge(sub, on="genome_id", how="left")
        assert len(d) == n_before, f"{name}: GUNC merge duplicated rows"
        n_match = int(d["gunc_css"].notna().sum())
        notes.append(f"{name}: GUNC from {p} -- {n_match}/{len(d)} genomes matched")
        if n_match < 0.5 * len(d):
            continue
        d = d[d["gunc_css"].notna()].copy()
        if "gunc_pass_raw" in d.columns:
            d["gunc_fail"] = ~d["gunc_pass_raw"].astype(str).str.lower().isin(
                ["true", "1", "yes", "pass"])
        else:
            d["gunc_fail"] = d["gunc_css"] > 0.45     # GUNC default CSS cutoff
        bs = bootstrapper(d, cfg, n_boot, 5000)
        # (a) pass/fail stratified by true contamination
        for lab in cfg.raw["strata"]["contamination_bins"]["labels"]:
            s = d[d["true_cont_bin"] == lab]
            if len(s) < 5:
                continue
            rows.append({"set": name, "stratum": lab, "n": len(s),
                         "n_clusters": int(s.cluster_id.nunique()),
                         "gunc_fail_rate": float(s.gunc_fail.mean()),
                         "median_true_contamination": float(s.true_contamination.median()),
                         "median_css": float(s.gunc_css.median())})
        # (b) agreement at the MIMAG thresholds
        for tau in (5.0, 10.0):
            t_fail = d.true_contamination.to_numpy(float) >= tau
            gf = d.gunc_fail.to_numpy(bool)
            rows.append({
                "set": name, "stratum": f"AGREEMENT at contamination tau={tau:g}%",
                "n": len(d), "n_clusters": int(d.cluster_id.nunique()),
                "n_true_fail": int(t_fail.sum()), "n_gunc_fail": int(gf.sum()),
                "agreement": float((t_fail == gf).mean()),
                "sensitivity": float(gf[t_fail].mean()) if t_fail.any() else np.nan,
                "specificity": float((~gf[~t_fail]).mean()) if (~t_fail).any() else np.nan})
        # (c) Spearman CSS vs truth, alongside MAGICC / CheckM2
        y = d.true_contamination.to_numpy(float)
        cands = {"GUNC CSS": d.gunc_css.to_numpy(float)}
        for t in ("magicc_v5", "checkm2"):
            c = f"pred_contamination__{t}"
            if c in d.columns:
                cands[cfg.tool_short(t) + " predicted contamination"] = d[c].to_numpy(float)
        for lab, v in cands.items():
            ok = ~(np.isnan(y) | np.isnan(v))
            if ok.sum() < 30:
                continue
            ci = bs.ci(lambda idx, _y=y, _v=v: float(sps.spearmanr(_y[idx], _v[idx]).statistic))
            corr.append({"set": name, "predictor": lab, "n": int(ok.sum()),
                         "n_clusters": bs.n_clusters,
                         "spearman_rho_vs_true_contamination":
                             float(sps.spearmanr(y[ok], v[ok]).statistic),
                         "rho_ci_lo": ci["ci_lo"], "rho_ci_hi": ci["ci_hi"]})
    return pd.DataFrame(rows), pd.DataFrame(corr), notes


# ---------------------------------------------------------------------------
# core-gene census (optional, slow)
# ---------------------------------------------------------------------------


def _census_one(args):
    acc, fasta, hmm, tmproot = args
    try:
        with tempfile.TemporaryDirectory(dir=tmproot) as td:
            faa = Path(td) / "p.faa"
            r = subprocess.run(["prodigal", "-i", str(fasta), "-a", str(faa),
                                "-p", "single", "-q", "-o", os.devnull],
                               capture_output=True)
            if r.returncode != 0 or not faa.exists() or faa.stat().st_size == 0:
                r = subprocess.run(["prodigal", "-i", str(fasta), "-a", str(faa),
                                    "-p", "meta", "-q", "-o", os.devnull],
                                   capture_output=True)
            if not faa.exists() or faa.stat().st_size == 0:
                return (acc, -1, -1)
            tbl = Path(td) / "t.tbl"
            subprocess.run(["hmmsearch", "--cut_tc", "--cpu", "1",
                            "--tblout", str(tbl), str(hmm), str(faa)],
                           capture_output=True)
            fams, hits = set(), 0
            with open(tbl) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) > 3:
                        fams.add(parts[3])
                        hits += 1
            return (acc, len(fams), hits)
    except Exception:
        return (acc, -1, -1)


def run_scg_census(cfg, frames: Dict[str, pd.DataFrame], threads: int) -> pd.DataFrame:
    """Prodigal + hmmsearch --cut_tc over unique dominant reference genomes."""
    from multiprocessing import Pool
    out_p = cfg.out_dir / "ws5.9_scg_census_reference_genomes.tsv"
    done = {}
    if out_p.exists():
        prev = pd.read_csv(out_p, sep="\t")
        done = dict(zip(prev.accession, prev.n_scg_detected))

    refs = {}
    for name, df in frames.items():
        for acc, dom in zip(df.dominant_accession.astype(str),
                            df.dominant_domain.astype(str)):
            refs.setdefault(acc, dom)
    # locate FASTAs from the split tables
    paths = {}
    for sp in ("train", "val", "test"):
        f = cfg.splits_dir / f"{sp}_genomes.tsv"
        if not f.exists():
            continue
        t = pd.read_csv(f, sep="\t")
        acol = next((c for c in ("gtdb_accession", "accession") if c in t.columns), None)
        pcol = next((c for c in ("fasta_path", "path") if c in t.columns), None)
        if acol is None or pcol is None:
            continue
        for a, p in zip(t[acol].astype(str), t[pcol].astype(str)):
            paths[a] = p

    root = cfg.project_root
    jobs = []
    bac = root / "85_bcg.hmm"
    arc = root / "uacg.hmm"
    tmproot = Path(os.environ.get("TMPDIR", "/tmp"))
    n_nopath = 0
    for acc, dom in refs.items():
        if acc in done:
            continue
        p = paths.get(acc)
        if p is None:
            n_nopath += 1
            continue
        fp = Path(p)
        if not fp.exists():
            # re-root historical absolute paths onto this checkout
            fp = root / Path(*fp.parts[fp.parts.index("data"):]) \
                if "data" in fp.parts else fp
        if not fp.exists():
            n_nopath += 1
            continue
        jobs.append((acc, fp, arc if str(dom).lower().startswith("arch") else bac, tmproot))

    print(f"[106/scg] {len(refs)} unique references, {len(done)} cached, "
          f"{len(jobs)} to run, {n_nopath} without a resolvable FASTA")
    rows = []
    if jobs:
        with Pool(threads) as pool:
            for i, (acc, nfam, nhit) in enumerate(pool.imap_unordered(_census_one, jobs, 4), 1):
                rows.append({"accession": acc, "n_scg_detected": nfam, "n_scg_hits": nhit,
                             "domain": refs[acc]})
                if i % 100 == 0:
                    print(f"[106/scg] {i}/{len(jobs)}", flush=True)
    new = pd.DataFrame(rows)
    if out_p.exists():
        new = pd.concat([pd.read_csv(out_p, sep="\t"), new], ignore_index=True)
    if len(new):
        new = new.drop_duplicates(subset="accession")
        new = new[new.n_scg_detected >= 0]
        new.to_csv(out_p, sep="\t", index=False)
        print(f"[106/scg] wrote {out_p} ({len(new)} references)")
    return new


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def ci(row, k, fmt="{:.2f}"):
    v, lo, hi = row.get(k), row.get(f"{k}_ci_lo"), row.get(f"{k}_ci_hi")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return fmt.format(v)
    return f"{fmt.format(v)} ({fmt.format(lo)}-{fmt.format(hi)})"


def write_reports(cfg, D, C, T, B, R, S, G, Gc, gnotes, n_boot):
    o = cfg.out_dir
    L = [
        "# WS5.8-5.10 - training-domain restriction, genome-size dependence, GUNC detection",
        "",
        f"Cluster bootstrap over `{cfg.cluster_col}`, {n_boot} resamples, "
        f"{int(cfg.ci_level * 100)}% percentile intervals. "
        "**R^2 is the coefficient of determination (1 - SS_res/SS_tot)** everywhere; "
        "the squared Pearson correlation is kept in the TSVs as `r2_pearson_sq` and "
        "is never called R^2.",
        "",
        fw.caption_denominator(cfg),
        "",
        "## WS5.8  Training-domain restriction (protocol section 4.4a)",
        "",
        "MAGICC V4/V5 were trained only where contamination% <= completeness%. Sets "
        "C, D and E predate that constraint and contain out-of-domain samples. The "
        "**in-domain subset is the primary analysis**; the out-of-domain subset is a "
        "separate robustness analysis. They are never pooled silently.",
        "",
    ]
    if len(D):
        cnt = (D[D.metric == "completeness"]
               .groupby(["set", "subset"], observed=True)["n"].max().unstack())
        L += ["### Samples per domain subset", "",
              fw.md_table(cnt.reset_index().fillna(0), "{:.0f}"), ""]
        for setname in D["set"].drop_duplicates():
            sub = D[D.set == setname]
            if set(sub.subset) == {"all"}:
                continue
            L += [f"### {setname}", ""]
            for metric in METRICS:
                m = sub[sub.metric == metric]
                if m.empty:
                    continue
                tab = []
                for _, r in m.iterrows():
                    tab.append({"subset": r["subset"], "tool": cfg.tool_short(r.tool),
                                "n": r.n, "clusters": r.n_clusters,
                                "MAE (95% CI)": ci(r, "mae"),
                                "bias (95% CI)": ci(r, "bias"),
                                "R2 (CoD)": "-" if not np.isfinite(r.get("r2", np.nan))
                                            else f"{r['r2']:.3f}",
                                "r2 Pearson (NOT R2)":
                                    "-" if not np.isfinite(r.get("r2_pearson_sq", np.nan))
                                    else f"{r['r2_pearson_sq']:.3f}"})
                L += [f"**{metric}**", "", fw.md_table(pd.DataFrame(tab)), ""]
    if len(C):
        L += ["### MIMAG-inspired classification by domain subset", "",
              fw.md_table(C.assign(tool=C.tool.map(cfg.tool_short))[
                  ["set", "subset", "tool", "n", "n_clusters", "macro_f1",
                   "accuracy", "high_precision", "high_recall"]], "{:.3f}"), ""]
    if len(T):
        t5 = T[(T.criterion == "contamination") & (T.tau == 5.0)]
        L += ["### False-fail / false-pass at the 5% contamination threshold, by domain subset",
              "",
              "`false_fail_rate = P(predicted FAIL | truly PASS)` -- a genuinely clean "
              "genome wrongly rejected. `false_pass_rate = P(predicted PASS | truly FAIL)`.",
              "",
              fw.md_table(pd.DataFrame([{
                  "set": r["set"], "subset": r["subset"], "tool": cfg.tool_short(r.tool),
                  "n true pass": r.n_true_pass, "n true fail": r.n_true_fail,
                  "false-fail rate (95% CI)": ci(r, "false_fail_rate", "{:.3f}"),
                  "false-pass rate (95% CI)": ci(r, "false_pass_rate", "{:.3f}")}
                  for _, r in t5.iterrows()])), ""]

    L += ["", "## WS5.9  Genome-size dependence of the error (protocol 3.10)", "",
          "Binned by the **dominant reference genome's** full length (GTDB "
          "`genome_size`), which is also the denominator of both percentages.", ""]
    if len(B):
        for scope in B.scope.drop_duplicates():
            s = B[B.scope == scope]
            L += [f"### {scope}", "",
                  fw.md_table(pd.DataFrame([{
                      "size bin": r.size_bin, "tool": cfg.tool_short(r.tool),
                      "n": r.n, "clusters": r.n_clusters,
                      "median Mbp": f"{r.median_size_mbp:.2f}",
                      "comp MAE": ci(r, "mae_completeness"),
                      "comp bias": ci(r, "bias_completeness"),
                      "cont MAE": ci(r, "mae_contamination"),
                      "cont bias": ci(r, "bias_contamination"),
                      "false-fail @5% cont": ci(r, "false_fail_rate_cont5", "{:.3f}")}
                      for _, r in s.iterrows()])), ""]
    if len(R):
        L += ["### Spearman correlation of the error with log10(dominant genome size, Mbp)",
              "",
              "Negative rho for a signed error means the tool becomes more negative "
              "(under-calls) as the genome grows; positive rho on an absolute error "
              "means larger genomes are harder.", "",
              fw.md_table(pd.DataFrame([{
                  "scope": r.scope, "tool": cfg.tool_short(r.tool), "metric": r.metric,
                  "error": r.error_kind, "n": r.n, "clusters": r.n_clusters,
                  "rho (95% CI)": f"{r.spearman_rho:.3f} ({r.rho_ci_lo:.3f}-{r.rho_ci_hi:.3f})",
                  "p (cluster bootstrap)": f"{r.p_cluster_bootstrap:.4g}"}
                  for _, r in R.iterrows()])), ""]
    if len(S):
        L += ["### Genome size vs number of detectable single-copy core genes", "",
              "Per dominant reference genome. `n_scg_detected` = distinct core-gene "
              "families hit by `hmmsearch --cut_tc` (85_bcg.hmm for bacteria, "
              "uacg.hmm for archaea) on Prodigal proteins. Partial Spearman "
              "correlations separate the two covariates.", "",
              fw.md_table(S, "{:.3f}"), ""]
    else:
        L += ["_Core-gene census not available; re-run with `--scg-census` to "
              "separate genome size from the number of detectable core genes._", ""]

    L += ["", "## WS5.10  GUNC as a contamination DETECTION comparator", "",
          "GUNC reports a clade-separation score (CSS) and a pass/fail call, not a "
          "contamination percentage, so it is **deliberately absent from every MAE "
          "table**. It is evaluated here only as a detector.", ""]
    if len(G):
        L += [fw.md_table(G, "{:.3f}"), ""]
    if len(Gc):
        L += ["### Spearman correlation with true contamination", "",
              fw.md_table(Gc, "{:.3f}"), ""]
    if gnotes:
        L += ["### Provenance / join notes", ""] + [f"* {n}" for n in gnotes] + [""]
    if not len(G) and not len(Gc):
        L += ["**No GUNC output was found for any benchmark set at the time of this "
              "run.** Searched `data/benchmarks/<set>/gunc_predictions.tsv`, "
              "`results/revision/benchmark/gunc/<set>/`, `results/revision/gunc/<set>/`. "
              "GUNC on `set_D_clean` was observed running concurrently; re-run "
              "`python scripts/106_domain_size_and_gunc.py --analyses gunc` once it "
              "finishes and the section will populate automatically.", ""]

    (o / "ws5.8_10_domain_size_gunc.md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--analyses", nargs="*",
                    default=["domain", "size", "gunc"])
    ap.add_argument("--scg-census", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tiers", nargs="*", default=["primary", "reported"])
    args = ap.parse_args()

    cfg = fw.load_config(args.config)
    n_boot = args.n_boot or cfg.n_boot
    names = [b.name for b in fw.discover_sets(cfg, include_missing=False,
                                              tiers=args.tiers)]
    print(f"[106] sets: {names}")
    frames = load_all_sets(cfg, names)
    print(f"[106] loaded {len(frames)} sets, join integrity verified")

    scg = None
    p_scg = cfg.out_dir / "ws5.9_scg_census_reference_genomes.tsv"
    if args.scg_census:
        scg = run_scg_census(cfg, frames, args.threads)
    elif p_scg.exists():
        scg = pd.read_csv(p_scg, sep="\t")

    D = C = T = B = R = S = G = Gc = pd.DataFrame()
    gnotes: List[str] = []
    if "domain" in args.analyses:
        print("[106] WS5.8 training-domain restriction ...")
        D, C, T = analysis_domain(cfg, frames, n_boot)
        D.to_csv(cfg.out_dir / "ws5.8_domain_restriction_accuracy.tsv", sep="\t", index=False)
        C.to_csv(cfg.out_dir / "ws5.8_domain_restriction_mimag.tsv", sep="\t", index=False)
        T.to_csv(cfg.out_dir / "ws5.8_domain_restriction_thresholds.tsv", sep="\t", index=False)
    if "size" in args.analyses:
        print("[106] WS5.9 genome-size dependence ...")
        B, R, S, big = analysis_size(cfg, frames, n_boot, scg)
        B.to_csv(cfg.out_dir / "ws5.9_size_bins.tsv", sep="\t", index=False)
        R.to_csv(cfg.out_dir / "ws5.9_size_correlations.tsv", sep="\t", index=False)
        if len(S):
            S.to_csv(cfg.out_dir / "ws5.9_size_vs_core_genes.tsv", sep="\t", index=False)
        cols = ["set", "genome_id", "dominant_accession", "dominant_phylum",
                "dominant_domain", "dominant_genome_size", "size_bin",
                "true_completeness", "true_contamination"]
        cols += [c for c in big.columns if c.startswith(("err_", "abs_err_"))]
        big[cols].to_csv(cfg.out_dir / "ws5.9_per_genome_with_size.tsv.gz",
                         sep="\t", index=False, compression="gzip")
    if "gunc" in args.analyses:
        print("[106] WS5.10 GUNC detection comparator ...")
        G, Gc, gnotes = analysis_gunc(cfg, frames, n_boot)
        if len(G):
            G.to_csv(cfg.out_dir / "ws5.10_gunc_detection.tsv", sep="\t", index=False)
        if len(Gc):
            Gc.to_csv(cfg.out_dir / "ws5.10_gunc_correlations.tsv", sep="\t", index=False)

    write_reports(cfg, D, C, T, B, R, S, G, Gc, gnotes, n_boot)
    print(f"[106] wrote ws5.8-5.10 outputs to {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
