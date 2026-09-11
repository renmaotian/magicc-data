#!/usr/bin/env python
"""
102_mimag_and_thresholds.py  --  WS5.1 + WS5.2
==============================================
WS5.1  MIMAG-inspired 3-class quality classification, PER BENCHMARK SET (never as
       a range): confusion matrices, class-specific precision/recall/F1, support
       and class balance, macro / micro / weighted F1, accuracy, balanced
       accuracy, Cohen's kappa -- all with cluster-bootstrap 95% CIs.

WS5.2  Threshold-focused analysis at the decision boundaries that matter in
       practice (Reviewer 2, major comment 2): contamination 5% and 10%,
       completeness 50% and 90%. False-pass and false-fail rates, sensitivity
       and specificity with cluster-bootstrap 95% CIs.

Everything is driven by scripts/config_revision_metrics.yaml; adding a benchmark
set or a tool requires no change to this script.

    python scripts/102_mimag_and_thresholds.py [--config ...] [--n-boot 2000]
                                               [--tiers primary reported secondary]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

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


# ---------------------------------------------------------------------------
# WS5.1  MIMAG metrics for one (set, tool)
# ---------------------------------------------------------------------------


def mimag_for_tool(df: pd.DataFrame, tool: str, labels: List[str],
                   bs: "fw.Bootstrapper") -> dict:
    t_code = fw.encode_labels(df["true_mimag"].to_numpy(), labels)
    p_code = fw.encode_labels(df[f"mimag__{tool}"].to_numpy(), labels)
    k = len(labels)
    # Classes observed in the FULL ground truth. Fixed here so that the
    # `*_observed` macro averages keep one definition across every bootstrap
    # replicate (see metrics_from_cm docstring).
    obs_mask = fw.cm_from_codes(t_code, p_code, k).sum(axis=1) > 0

    def stats(idx: np.ndarray) -> Dict[str, float]:
        cm = fw.cm_from_codes(t_code[idx], p_code[idx], k)
        m = fw.metrics_from_cm(cm, labels, observed_mask=obs_mask)
        out: Dict[str, float] = {
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "micro_f1": m["micro_f1"], "weighted_f1": m["weighted_f1"],
            "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
            "balanced_accuracy": m["balanced_accuracy"], "cohen_kappa": m["cohen_kappa"],
            "macro_f1_observed": m["macro_f1_observed"],
            "macro_precision_observed": m["macro_precision_observed"],
            "macro_recall_observed": m["macro_recall_observed"],
        }
        for i, lab in enumerate(labels):
            out[f"precision__{lab}"] = m["precision"][i]
            out[f"recall__{lab}"] = m["recall"][i]
            out[f"f1__{lab}"] = m["f1"][i]
        return out

    boot = bs.ci_multi(stats)
    cm = fw.cm_from_codes(t_code, p_code, k)
    m = fw.metrics_from_cm(cm, labels, observed_mask=obs_mask)
    return {"cm": cm, "point": m, "boot": boot,
            "n": int(len(df)), "n_clusters": bs.n_clusters,
            "clustered": bs.clustered}


# ---------------------------------------------------------------------------
# WS5.2  threshold metrics for one (set, tool)
# ---------------------------------------------------------------------------

THRESH_KEYS = ["false_pass_rate", "false_fail_rate", "sensitivity", "specificity",
               "ppv_fail", "npv_pass", "accuracy", "balanced_accuracy"]


def thresholds_for_tool(df: pd.DataFrame, tool: str, cfg,
                        bs: "fw.Bootstrapper") -> List[dict]:
    rows = []
    specs = ([("contamination", t) for t in cfg.raw["thresholds"]["contamination"]] +
             [("completeness", t) for t in cfg.raw["thresholds"]["completeness"]])
    for criterion, tau in specs:
        true = df[f"true_{criterion}"].to_numpy(float)
        pred = df[f"pred_{criterion}__{tool}"].to_numpy(float)
        point = fw.threshold_metrics(true, pred, tau, criterion)

        def _stat(idx, _t=true, _p=pred, _tau=tau, _c=criterion):
            m = fw.threshold_metrics(_t[idx], _p[idx], _tau, _c)
            return {k: m[k] for k in THRESH_KEYS}

        boot = bs.ci_multi(_stat)
        rec = {"criterion": criterion, "threshold": tau,
               "pass_rule": (f"contamination < {tau:g}%" if criterion == "contamination"
                             else f"completeness >= {tau:g}%")}
        rec.update({k: v for k, v in point.items()
                    if k.startswith("n") or k in ("accuracy", "balanced_accuracy")})
        for k in THRESH_KEYS:
            rec[k] = point[k]
            rec[f"{k}_ci_lo"] = boot[k]["ci_lo"]
            rec[f"{k}_ci_hi"] = boot[k]["ci_hi"]
        rec["boot_clustered"] = bs.clustered
        rec["boot_n_clusters"] = bs.n_clusters
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_set(cfg, bs_obj, n_boot: int):
    """Everything for one benchmark set (all tools).

    A single Bootstrapper is built per set and reused for every tool and every
    threshold, so all tools within a set share the same bootstrap resamples
    (common random numbers) and the expensive cluster-resample index cache is
    built only once.
    """
    df, tools, _ = fw.load_set(cfg, bs_obj)
    labels = cfg.mimag_classes
    bs = fw.Bootstrapper(clusters=df["cluster_id"].to_numpy(), n_iter=n_boot,
                         ci_level=cfg.ci_level,
                         seed=cfg.seed + (fw.stable_hash(bs_obj.name) % 100000))
    bs.resample_indices()   # build the cache once

    class_rows, overall_rows, cm_rows, thr_rows = [], [], [], []
    balance = df["true_mimag"].value_counts()
    for tool in tools:
        res = mimag_for_tool(df, tool, labels, bs)
        cm, pt, bt = res["cm"], res["point"], res["boot"]
        base = {"set": bs_obj.name, "set_label": bs_obj.label, "status": bs_obj.status,
                "tier": bs_obj.tier, "tool": tool, "tool_label": cfg.tool_label(tool),
                "tool_role": cfg.tool_role(tool), "n": res["n"],
                "n_clusters": res["n_clusters"], "clustered_bootstrap": res["clustered"]}
        for i, lab in enumerate(labels):
            r = dict(base)
            r.update({
                "mimag_class": lab,
                "support_true": int(pt["support"][i]),
                "support_true_frac": float(pt["support"][i] / max(1, res["n"])),
                "n_predicted": int(pt["predicted_count"][i]),
                "precision": pt["precision"][i],
                "precision_ci_lo": bt[f"precision__{lab}"]["ci_lo"],
                "precision_ci_hi": bt[f"precision__{lab}"]["ci_hi"],
                "recall": pt["recall"][i],
                "recall_ci_lo": bt[f"recall__{lab}"]["ci_lo"],
                "recall_ci_hi": bt[f"recall__{lab}"]["ci_hi"],
                "f1": pt["f1"][i],
                "f1_ci_lo": bt[f"f1__{lab}"]["ci_lo"],
                "f1_ci_hi": bt[f"f1__{lab}"]["ci_hi"]})
            class_rows.append(r)
        r = dict(base)
        for key in ("accuracy", "macro_f1", "macro_f1_observed", "micro_f1",
                    "weighted_f1", "macro_precision", "macro_recall",
                    "macro_precision_observed", "macro_recall_observed",
                    "balanced_accuracy", "cohen_kappa"):
            r[key] = pt[key]
            r[f"{key}_ci_lo"] = bt[key]["ci_lo"]
            r[f"{key}_ci_hi"] = bt[key]["ci_hi"]
        r["class_balance_true"] = json.dumps({l: int(balance.get(l, 0)) for l in labels})
        r["n_classes_observed"] = pt["n_classes_observed"]
        r["classes_absent_in_truth"] = pt["classes_absent_in_truth"]
        overall_rows.append(r)
        for i, lt in enumerate(labels):
            for j, lp in enumerate(labels):
                cm_rows.append({**base, "true_class": lt, "pred_class": lp,
                                "n": int(cm[i, j]),
                                "row_frac": float(cm[i, j] / cm[i].sum()) if cm[i].sum() else np.nan})
        for rec in thresholds_for_tool(df, tool, cfg, bs):
            thr_rows.append({**base, **rec})
    return class_rows, overall_rows, cm_rows, thr_rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--tiers", nargs="*", default=["primary", "reported", "secondary"])
    args = ap.parse_args(argv)

    cfg = fw.load_config(args.config)
    n_boot = args.n_boot or cfg.n_boot
    sets = [b for b in fw.discover_sets(cfg, include_missing=False, tiers=args.tiers)]
    usable = []
    for b in sets:
        _, tools, _ = fw.load_set(cfg, b)
        if tools:
            usable.append(b)
        else:
            print(f"[skip] {b.name}: no tool predictions present yet")
    print(f"[102] {len(usable)} sets x up to {len(cfg.tools)} tools, n_boot={n_boot}")

    res = []
    for b in usable:
        print(f"  [102] {b.name} ...", flush=True)
        res.append(process_set(cfg, b, n_boot))

    cls = pd.DataFrame([r for x in res for r in x[0]])
    ovr = pd.DataFrame([r for x in res for r in x[1]])
    cms = pd.DataFrame([r for x in res for r in x[2]])
    thr = pd.DataFrame([r for x in res for r in x[3]])

    order = {b.name: i for i, b in enumerate(usable)}
    torder = {t: i for i, t in enumerate(cfg.tools)}
    for d in (cls, ovr, cms, thr):
        d["_s"] = d["set"].map(order)
        d["_t"] = d["tool"].map(torder)
    cls = cls.sort_values(["_s", "_t", "mimag_class"]).drop(columns=["_s", "_t"])
    ovr = ovr.sort_values(["_s", "_t"]).drop(columns=["_s", "_t"])
    cms = cms.sort_values(["_s", "_t"]).drop(columns=["_s", "_t"])
    thr = thr.sort_values(["_s", "criterion", "threshold", "_t"]).drop(columns=["_s", "_t"])

    o = cfg.out_dir
    cls.to_csv(o / "ws5.1_mimag_class_metrics.tsv", sep="\t", index=False)
    ovr.to_csv(o / "ws5.1_mimag_overall_metrics.tsv", sep="\t", index=False)
    cms.to_csv(o / "ws5.1_mimag_confusion_matrices.tsv", sep="\t", index=False)
    thr.to_csv(o / "ws5.2_threshold_analysis.tsv", sep="\t", index=False)

    write_reports(cfg, cls, ovr, cms, thr, usable, n_boot)
    print(f"[102] wrote ws5.1_* and ws5.2_* to {o}")
    return 0


# ---------------------------------------------------------------------------
# Markdown reports
# ---------------------------------------------------------------------------

MIMAG_DEF = (
    "MIMAG-inspired quality classes, applied to completeness and contamination "
    "ONLY: **high** = completeness >= 90% AND contamination < 5%; **medium** = "
    "completeness >= 50% AND contamination < 10% (and not high); **low** = all "
    "others. NOTE: the full MIMAG standard (Bowers et al. 2017) additionally "
    "requires 23S/16S/5S rRNA genes and >= 18 tRNAs for the high-quality tier; "
    "those criteria cannot be evaluated from completeness/contamination "
    "estimates and are NOT applied here, so the tier names are "
    "MIMAG-inspired rather than strict MIMAG. The submitted manuscript is "
    "internally inconsistent about the completeness boundary (line 53 uses "
    "'>= 90% complete', the Results text uses '> 90% completeness'); the "
    "inclusive form '>= 90%' is used throughout, matching Bowers et al."
)


def _ci(v, lo, hi, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def write_reports(cfg, cls, ovr, cms, thr, sets, n_boot):
    den = fw.caption_denominator(cfg)
    boot_note = (f"All 95% confidence intervals are percentile bootstrap intervals "
                 f"({n_boot} replicates) with **clusters resampled by dominant "
                 f"reference genome** (`dominant_accession`), so genomes simulated "
                 f"from the same reference are resampled together.")

    # ---- WS5.1 ----
    L = ["# WS5.1 - MIMAG-inspired quality classification, per benchmark set", "",
         MIMAG_DEF, "", den, "", boot_note, "",
         "Sets flagged **SUPERSEDED** were built with dominant genomes drawn from "
         "the training split (Set C: 1000/1000 dominants in train; Set D: 796 train "
         "/ 107 val / 97 test) and are reported only for transparency; the clean "
         "replacements are `set_C_clean` and `set_D_clean`.", ""]
    for b in sets:
        sub_o = ovr[ovr["set"] == b.name]
        sub_c = cls[cls["set"] == b.name]
        if sub_o.empty:
            continue
        flag = "  **[SUPERSEDED - training-set leakage]**" if b.status == "superseded_leaky" else ""
        L += [f"## {b.name} - {b.label}{flag}", "",
              f"*n* = {int(sub_o['n'].iloc[0])}, clusters (distinct dominant reference "
              f"genomes) = {int(sub_o['n_clusters'].iloc[0])}. Design: {b.design}", ""]
        bal = json.loads(sub_o["class_balance_true"].iloc[0])
        tot = sum(bal.values()) or 1
        absent = str(sub_o["classes_absent_in_truth"].iloc[0] or "")
        L += ["**True class balance:** " + ", ".join(
            f"{k} = {v} ({100 * v / tot:.1f}%)" for k, v in bal.items()), ""]
        small = {k: v for k, v in bal.items() if 0 < v < 10}
        if small:
            L += [f"> **Small-support class(es): "
                  + ", ".join(f"{k} (n={v})" for k, v in small.items())
                  + ".** Per-class F1 for these classes is estimated from fewer "
                  "than 10 genomes; the CI is correspondingly wide and the macro "
                  "average is dominated by sampling noise in that class. Individual "
                  "data points are shown for these classes in the figures.", ""]
        if absent:
            L += [f"> **Class(es) absent from the ground truth by design: "
                  f"{absent}.** The 3-class macro F1 therefore averages a "
                  f"structurally undefined class scored as 0 and is bounded above "
                  f"by {sub_o['n_classes_observed'].iloc[0]}/3; the "
                  f"`macro F1 (observed classes)` column is the interpretable "
                  f"summary for this set. The submitted manuscript's Table S3 "
                  f"used the 3-class form.", ""]
        t = pd.DataFrame({
            "tool": sub_o["tool_label"],
            "accuracy": [_ci(*r) for r in zip(sub_o.accuracy, sub_o.accuracy_ci_lo, sub_o.accuracy_ci_hi)],
            "macro F1 (3 classes)": [_ci(*r) for r in zip(sub_o.macro_f1, sub_o.macro_f1_ci_lo, sub_o.macro_f1_ci_hi)],
            "macro F1 (observed classes)": [_ci(*r) for r in zip(sub_o.macro_f1_observed, sub_o.macro_f1_observed_ci_lo, sub_o.macro_f1_observed_ci_hi)],
            "micro F1 (=accuracy)": [_ci(*r) for r in zip(sub_o.micro_f1, sub_o.micro_f1_ci_lo, sub_o.micro_f1_ci_hi)],
            "weighted F1": [_ci(*r) for r in zip(sub_o.weighted_f1, sub_o.weighted_f1_ci_lo, sub_o.weighted_f1_ci_hi)],
            "balanced acc.": [_ci(*r) for r in zip(sub_o.balanced_accuracy, sub_o.balanced_accuracy_ci_lo, sub_o.balanced_accuracy_ci_hi)],
            "Cohen kappa": [_ci(*r) for r in zip(sub_o.cohen_kappa, sub_o.cohen_kappa_ci_lo, sub_o.cohen_kappa_ci_hi)],
        })
        L += ["### Overall classification performance (estimate [95% CI])", "",
              fw.md_table(t), ""]
        t2 = pd.DataFrame({
            "tool": sub_c["tool_label"], "class": sub_c["mimag_class"],
            "support (true n)": sub_c["support_true"].astype(int),
            "n predicted": sub_c["n_predicted"].astype(int),
            "precision": [_ci(*r) for r in zip(sub_c.precision, sub_c.precision_ci_lo, sub_c.precision_ci_hi)],
            "recall": [_ci(*r) for r in zip(sub_c.recall, sub_c.recall_ci_lo, sub_c.recall_ci_hi)],
            "F1": [_ci(*r) for r in zip(sub_c.f1, sub_c.f1_ci_lo, sub_c.f1_ci_hi)],
        })
        L += ["### Per-class precision / recall / F1 (estimate [95% CI])", "",
              fw.md_table(t2), ""]
        L += ["### Confusion matrices (rows = true class, columns = predicted class)", ""]
        sub_m = cms[cms["set"] == b.name]
        for tool in sub_m["tool"].drop_duplicates():
            g = sub_m[sub_m["tool"] == tool]
            piv = g.pivot(index="true_class", columns="pred_class", values="n").reindex(
                index=cfg.mimag_classes, columns=cfg.mimag_classes).fillna(0).astype(int)
            piv.insert(0, "true \\ pred", piv.index)
            L += [f"**{cfg.tool_label(tool)}**", "", fw.md_table(piv), ""]
    (cfg.out_dir / "ws5.1_mimag.md").write_text("\n".join(L) + "\n")

    # ---- WS5.2 ----
    L = ["# WS5.2 - Threshold-focused QC decision analysis", "",
         "Reviewer 2 (major comment 2) asks for the analysis to be reorganised "
         "around the decision thresholds that matter in practice. For each "
         "criterion a genome either **passes** or **fails**:", "",
         "* contamination criterion at tau: pass <=> contamination < tau "
         "(tau = 5% and 10%)",
         "* completeness criterion at tau: pass <=> completeness >= tau "
         "(tau = 50% and 90%)", "",
         "Error definitions, with the denominator of each rate stated explicitly:", "",
         "* **false-pass rate** = n(truly fails AND predicted passes) / "
         "n(truly fails). The dangerous error: a genome that should be "
         "discarded is retained. Equals 1 - sensitivity.",
         "* **false-fail rate** = n(truly passes AND predicted fails) / "
         "n(truly passes). A usable genome is discarded. Equals 1 - specificity.",
         "* **sensitivity** = n(truly fails AND predicted fails) / n(truly fails).",
         "* **specificity** = n(truly passes AND predicted passes) / n(truly passes).",
         "", den, "", boot_note, ""]
    for b in sets:
        sub = thr[thr["set"] == b.name]
        if sub.empty:
            continue
        flag = "  **[SUPERSEDED - training-set leakage]**" if b.status == "superseded_leaky" else ""
        L += [f"## {b.name} - {b.label}{flag}", ""]
        for (crit, tau), g in sub.groupby(["criterion", "threshold"], sort=False):
            n_fail = int(g["n_true_fail"].iloc[0])
            n_pass = int(g["n_true_pass"].iloc[0])
            if n_fail == 0 or n_pass == 0:
                L += [f"### {g['pass_rule'].iloc[0]}", "",
                      f"_Not informative for this set: n(truly fails) = {n_fail}, "
                      f"n(truly passes) = {n_pass}; one class is empty by design._", ""]
                continue
            t = pd.DataFrame({
                "tool": g["tool_label"],
                "n false-pass": g["n_false_pass"].astype(int),
                "false-pass rate": [_ci(*r) for r in zip(g.false_pass_rate, g.false_pass_rate_ci_lo, g.false_pass_rate_ci_hi)],
                "n false-fail": g["n_false_fail"].astype(int),
                "false-fail rate": [_ci(*r) for r in zip(g.false_fail_rate, g.false_fail_rate_ci_lo, g.false_fail_rate_ci_hi)],
                "sensitivity": [_ci(*r) for r in zip(g.sensitivity, g.sensitivity_ci_lo, g.sensitivity_ci_hi)],
                "specificity": [_ci(*r) for r in zip(g.specificity, g.specificity_ci_lo, g.specificity_ci_hi)],
                "balanced acc.": [_ci(*r) for r in zip(g.balanced_accuracy, g.balanced_accuracy_ci_lo, g.balanced_accuracy_ci_hi)],
            })
            L += [f"### {g['pass_rule'].iloc[0]}  "
                  f"(n truly passes = {n_pass}, n truly fails = {n_fail})", "",
                  fw.md_table(t), ""]
    (cfg.out_dir / "ws5.2_threshold_analysis.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
