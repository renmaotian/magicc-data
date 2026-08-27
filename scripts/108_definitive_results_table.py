#!/usr/bin/env python
"""
108_definitive_results_table.py  --  WS5 consolidated deliverable
=================================================================
Builds the single definitive, leakage-free results table for the revised
manuscript from the outputs of scripts 102-106, after competitor predictions
became available for ``set_C_clean`` and ``set_D_clean``.

Conventions enforced here (protocol sections 4.4d, 4.4a, 8):
  * R^2 is ALWAYS the coefficient of determination (1 - SS_res/SS_tot).
    Squared Pearson is carried in a separate, explicitly named column and is
    never labelled R^2. R^2 is omitted where the true value has zero variance.
  * All CIs are cluster bootstraps over the dominant reference genome.
  * MIMAG thresholds are labelled "MIMAG-inspired" throughout.
  * Results are reported wherever they fall, including where MAGICC loses.

Outputs (results/revision/metrics/):
  definitive_table_5set.tsv        per set x tool, long form, every statistic
  definitive_table_5set_wide.tsv   manuscript-shaped wide table
  definitive_table_5set.md         rendered markdown
  definitive_headline.json         pooled headline numbers + counter-findings
  definitive_thresholds.tsv        5%/10% contamination, 50%/90% completeness
  definitive_mimag.tsv             MIMAG-inspired P/R/F1 + macro F1 per set/tool
  definitive_domain_restriction.tsv  in-domain (primary) vs out-of-domain

    python scripts/108_definitive_results_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/path/to/magicc")
M = ROOT / "results/revision/metrics"

# The definitive leakage-free five sets, in manuscript order.
CLEAN5 = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]
MANU5 = ["set_A_v2", "set_B_v2", "set_C", "set_D", "set_E"]
TOOLS = ["magicc_v5", "checkm2", "cocopye", "deepcheck"]
SHORT = {"magicc_v5": "MAGICC v5", "magicc_v4": "MAGICC v4", "magicc_v3": "MAGICC v3",
         "checkm2": "CheckM2", "cocopye": "CoCoPyE", "deepcheck": "DeepCheck"}
SETLAB = {"set_A_v2": "Set A (completeness gradient, 0% contamination)",
          "set_B_v2": "Set B (contamination gradient, 100% completeness)",
          "set_C_clean": "Set C-clean (Patescibacteriota, held-out test split)",
          "set_D_clean": "Set D-clean (Archaea, held-out test split)",
          "set_E": "Set E (realistic mixture)",
          "POOLED_leakage_free_5_sets": "POOLED, leakage-free 5 sets (n=5,000)",
          "POOLED_manuscript_5_sets": "POOLED, submitted-manuscript 5 sets (leaky C/D)"}

DENOM = ("completeness (%) = retained dominant-genome bp / full reference length of "
         "the dominant genome x 100; contamination (%) = total contaminant bp / full "
         "reference length of the dominant genome x 100. Both percentages share the "
         "same denominator and are independent measures.")


def fmt(v, lo=None, hi=None, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    s = f"{v:.{nd}f}"
    if lo is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{lo:.{nd}f}, {hi:.{nd}f}]"
    return s


def main() -> int:
    A = pd.read_csv(M / "ws5.5_table_S2_rebuilt.tsv", sep="\t")
    rows_wanted = CLEAN5 + ["POOLED_leakage_free_5_sets",
                            "POOLED_manuscript_5_sets"] + MANU5
    A = A[A["set"].isin(rows_wanted) & A["tool"].isin(TOOLS)].copy()

    # ---------------- long form -------------------------------------------
    long = A[["set", "set_label", "status", "tier", "tool", "tool_label", "metric",
              "n", "n_clusters", "true_value_constant", "mae", "mae_ci_lo",
              "mae_ci_hi", "rmse", "rmse_ci_lo", "rmse_ci_hi", "bias", "bias_ci_lo",
              "bias_ci_hi", "r2", "r2_ci_lo", "r2_ci_hi", "r2_pearson_sq",
              "r2_omitted_reason"]].copy()
    long = long.rename(columns={
        "r2": "r2_coefficient_of_determination",
        "r2_ci_lo": "r2_coefficient_of_determination_ci_lo",
        "r2_ci_hi": "r2_coefficient_of_determination_ci_hi",
        "r2_pearson_sq": "pearson_r_squared_NOT_R2",
        "bias": "signed_bias_pred_minus_true"})
    long["set_order"] = long["set"].map({s: i for i, s in enumerate(rows_wanted)})
    long["tool_order"] = long["tool"].map({t: i for i, t in enumerate(TOOLS)})
    long = long.sort_values(["set_order", "metric", "tool_order"]).drop(
        columns=["set_order", "tool_order"])
    long.to_csv(M / "definitive_table_5set.tsv", sep="\t", index=False)

    # ---------------- wide, manuscript-shaped ------------------------------
    def grab(s, t, metric):
        r = A[(A["set"] == s) & (A.tool == t) & (A.metric == metric)]
        return r.iloc[0] if len(r) else None

    wide, md = [], []
    md.append("# Definitive leakage-free results table (MAGICC revision, WS5)\n")
    md.append(f"Generated from `results/revision/metrics/` after CheckM2 / CoCoPyE / "
              f"DeepCheck predictions became available for `set_C_clean` and "
              f"`set_D_clean`.\n")
    md.append("**Conventions.** R^2 is the coefficient of determination "
              "(1 - SS_res/SS_tot), never squared Pearson; it is omitted where the "
              "true value has zero variance. All 95% CIs are cluster bootstraps over "
              "the dominant reference genome. Signed bias = mean(predicted - true), "
              "so negative = under-estimate.\n")
    md.append(f"**Denominators.** {DENOM}\n")

    for s in CLEAN5 + ["POOLED_leakage_free_5_sets", "POOLED_manuscript_5_sets"]:
        present = A[A["set"] == s]
        if not len(present):
            continue
        n = int(present["n"].iloc[0])
        nc = int(present["n_clusters"].iloc[0])
        md.append(f"\n## {SETLAB.get(s, s)}\n")
        md.append(f"n = {n} genomes; {nc} reference-genome clusters "
                  f"(CIs are cluster bootstraps over these).\n")
        md.append("| Tool | Completeness MAE (95% CI) | Completeness R^2 | "
                  "Completeness bias | Contamination MAE (95% CI) | "
                  "Contamination R^2 | Contamination bias |")
        md.append("|---|---|---|---|---|---|---|")
        for t in TOOLS:
            c, x = grab(s, t, "completeness"), grab(s, t, "contamination")
            if c is None and x is None:
                continue
            rec = {"set": s, "set_label": SETLAB.get(s, s), "tool": t,
                   "tool_label": SHORT[t], "n": n, "n_clusters": nc}
            for tag, r in (("completeness", c), ("contamination", x)):
                if r is None:
                    continue
                rec[f"{tag}_mae"] = r["mae"]
                rec[f"{tag}_mae_ci_lo"] = r["mae_ci_lo"]
                rec[f"{tag}_mae_ci_hi"] = r["mae_ci_hi"]
                rec[f"{tag}_r2_cod"] = r["r2"]
                rec[f"{tag}_r2_cod_ci_lo"] = r["r2_ci_lo"]
                rec[f"{tag}_r2_cod_ci_hi"] = r["r2_ci_hi"]
                rec[f"{tag}_bias"] = r["bias"]
                rec[f"{tag}_bias_ci_lo"] = r["bias_ci_lo"]
                rec[f"{tag}_bias_ci_hi"] = r["bias_ci_hi"]
                rec[f"{tag}_r2_omitted_reason"] = r.get("r2_omitted_reason", "")
            wide.append(rec)
            md.append(
                f"| {SHORT[t]} | "
                f"{fmt(rec.get('completeness_mae'), rec.get('completeness_mae_ci_lo'), rec.get('completeness_mae_ci_hi'))} | "
                f"{fmt(rec.get('completeness_r2_cod'), rec.get('completeness_r2_cod_ci_lo'), rec.get('completeness_r2_cod_ci_hi'), 3)} | "
                f"{fmt(rec.get('completeness_bias'), rec.get('completeness_bias_ci_lo'), rec.get('completeness_bias_ci_hi'))} | "
                f"{fmt(rec.get('contamination_mae'), rec.get('contamination_mae_ci_lo'), rec.get('contamination_mae_ci_hi'))} | "
                f"{fmt(rec.get('contamination_r2_cod'), rec.get('contamination_r2_cod_ci_lo'), rec.get('contamination_r2_cod_ci_hi'), 3)} | "
                f"{fmt(rec.get('contamination_bias'), rec.get('contamination_bias_ci_lo'), rec.get('contamination_bias_ci_hi'))} |")
        # R^2 omission notes
        for t in TOOLS:
            for metric in ("completeness", "contamination"):
                r = grab(s, t, metric)
                if r is not None and isinstance(r.get("r2_omitted_reason"), str) \
                        and r["r2_omitted_reason"].strip():
                    md.append(f"\n> R^2 omitted for {metric}: {r['r2_omitted_reason']}")
                    break
            else:
                continue
            break

    W = pd.DataFrame(wide)
    W.to_csv(M / "definitive_table_5set_wide.tsv", sep="\t", index=False)

    # ---------------- thresholds ------------------------------------------
    T = pd.read_csv(M / "ws5.2_threshold_analysis.tsv", sep="\t")
    T = T[T["set"].isin(CLEAN5) & T["tool"].isin(TOOLS)].copy()
    keep = ["set", "set_label", "tool", "tool_label", "criterion", "threshold",
            "n", "n_clusters", "n_true_pass", "n_true_fail", "n_false_pass",
            "n_false_fail", "false_pass_rate", "false_pass_rate_ci_lo",
            "false_pass_rate_ci_hi", "false_fail_rate", "false_fail_rate_ci_lo",
            "false_fail_rate_ci_hi", "sensitivity", "sensitivity_ci_lo",
            "sensitivity_ci_hi", "specificity", "specificity_ci_lo",
            "specificity_ci_hi", "balanced_accuracy", "balanced_accuracy_ci_lo",
            "balanced_accuracy_ci_hi"]
    T = T[[c for c in keep if c in T.columns]]
    T["so"] = T["set"].map({s: i for i, s in enumerate(CLEAN5)})
    T["to"] = T["tool"].map({t: i for i, t in enumerate(TOOLS)})
    T = T.sort_values(["so", "criterion", "threshold", "to"]).drop(columns=["so", "to"])
    T.to_csv(M / "definitive_thresholds.tsv", sep="\t", index=False)

    # set_C_clean false-fail at 5% contamination, all four tools (counter-finding)
    ff = T[(T["set"] == "set_C_clean") & (T.criterion == "contamination") &
           (T.threshold == 5.0)]
    ff_rows = {r["tool"]: {"false_fail_rate": float(r["false_fail_rate"]),
                           "ci": [float(r["false_fail_rate_ci_lo"]),
                                  float(r["false_fail_rate_ci_hi"])],
                           "n_false_fail": int(r["n_false_fail"]),
                           "n_true_pass": int(r["n_true_pass"])}
               for _, r in ff.iterrows()}

    # ---------------- MIMAG-inspired --------------------------------------
    MC = pd.read_csv(M / "ws5.1_mimag_class_metrics.tsv", sep="\t")
    MO = pd.read_csv(M / "ws5.1_mimag_overall_metrics.tsv", sep="\t")
    MC = MC[MC["set"].isin(CLEAN5) & MC["tool"].isin(TOOLS)]
    MO = MO[MO["set"].isin(CLEAN5) & MO["tool"].isin(TOOLS)]
    mim = MC.merge(MO[["set", "tool", "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
                       "accuracy", "balanced_accuracy", "cohen_kappa",
                       "class_balance_true"]], on=["set", "tool"], how="left")
    mim["so"] = mim["set"].map({s: i for i, s in enumerate(CLEAN5)})
    mim["to"] = mim["tool"].map({t: i for i, t in enumerate(TOOLS)})
    mim = mim.sort_values(["so", "to", "mimag_class"]).drop(columns=["so", "to"])
    mim.to_csv(M / "definitive_mimag.tsv", sep="\t", index=False)

    # ---------------- training-domain restriction (protocol 4.4a) ---------
    dom_path = M / "ws5.8_domain_restriction_accuracy.tsv"
    dom_summary = {}
    if dom_path.exists():
        D = pd.read_csv(dom_path, sep="\t")
        D = D[D["set"].isin(CLEAN5) & D["tool"].isin(TOOLS)].copy()
        D["so"] = D["set"].map({s: i for i, s in enumerate(CLEAN5)})
        D["to"] = D["tool"].map({t: i for i, t in enumerate(TOOLS)})
        D = D.sort_values(["so", "subset", "metric", "to"]).drop(columns=["so", "to"])
        D.to_csv(M / "definitive_domain_restriction.tsv", sep="\t", index=False)
        for s in CLEAN5:
            sub = D[D["set"] == s]
            counts = {k: int(sub[(sub.subset == k) & (sub.tool == "magicc_v5") &
                                 (sub.metric == "completeness")]["n"].iloc[0])
                      if len(sub[(sub.subset == k) & (sub.tool == "magicc_v5") &
                                 (sub.metric == "completeness")]) else 0
                      for k in ("all", "in_domain", "out_of_domain")}
            dom_summary[s] = counts

    # ---------------- headline JSON ---------------------------------------
    def pooled(setname):
        out = {}
        for t in TOOLS:
            e = {}
            for metric in ("completeness", "contamination"):
                r = grab(setname, t, metric)
                if r is None:
                    continue
                e[metric] = {
                    "mae": round(float(r["mae"]), 4),
                    "mae_ci95": [round(float(r["mae_ci_lo"]), 4),
                                 round(float(r["mae_ci_hi"]), 4)],
                    "r2_coefficient_of_determination":
                        None if not np.isfinite(r["r2"]) else round(float(r["r2"]), 4),
                    "r2_ci95": (None if not np.isfinite(r["r2"])
                                else [round(float(r["r2_ci_lo"]), 4),
                                      round(float(r["r2_ci_hi"]), 4)]),
                    "pearson_r_squared_NOT_R2":
                        None if not np.isfinite(r.get("r2_pearson_sq", np.nan))
                        else round(float(r["r2_pearson_sq"]), 4),
                    "signed_bias": round(float(r["bias"]), 4),
                    "signed_bias_ci95": [round(float(r["bias_ci_lo"]), 4),
                                         round(float(r["bias_ci_hi"]), 4)],
                    "n": int(r["n"]), "n_clusters": int(r["n_clusters"])}
            if e:
                out[t] = e
        return out

    headline = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "definition_notes": {
            "r2": "coefficient of determination, 1 - SS_res/SS_tot (protocol 4.4d); "
                  "squared Pearson reported separately and never labelled R^2; "
                  "omitted where the true value has zero variance",
            "ci": "95% percentile cluster bootstrap over the dominant reference genome",
            "bias": "mean(predicted - true) in percentage points; negative = under-estimate",
            "mimag": "MIMAG-inspired (completeness/contamination only): high >=90% "
                     "completeness AND <5% contamination; medium >=50% AND <10%",
            "denominators": DENOM},
        "definitive_leakage_free_5_sets": CLEAN5,
        "pooled_leakage_free_5_sets": pooled("POOLED_leakage_free_5_sets"),
        "pooled_submitted_manuscript_5_sets_leaky_CD": pooled("POOLED_manuscript_5_sets"),
        "per_set": {s: pooled(s) for s in CLEAN5},
        "counter_finding_set_C_clean_false_fail_at_5pct_contamination": {
            "description": "Fraction of genuinely clean (true contamination < 5%) "
                           "Patescibacteriota wrongly flagged as contaminated. "
                           "Reported for all four tools so MAGICC's value is "
                           "contextualised rather than presented in isolation.",
            "by_tool": ff_rows},
        "training_domain_restriction_sample_counts": dom_summary,
    }
    (M / "definitive_headline.json").write_text(json.dumps(headline, indent=2) + "\n")

    # append the counter-finding + threshold section to the markdown
    md.append("\n## Honest counter-finding: `set_C_clean` false-fail rate at the "
              "5% contamination threshold (MIMAG-inspired)\n")
    md.append("Denominator = genomes whose TRUE contamination is < 5% "
              f"(n = {ff['n_true_pass'].iloc[0] if len(ff) else 'n/a'}); a false fail "
              "is such a genome predicted at >= 5% contamination.\n")
    md.append("| Tool | False-fail rate (95% CI) | n false fails / n truly clean |")
    md.append("|---|---|---|")
    for t in TOOLS:
        if t in ff_rows:
            r = ff_rows[t]
            md.append(f"| {SHORT[t]} | {r['false_fail_rate']:.3f} "
                      f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | "
                      f"{r['n_false_fail']} / {r['n_true_pass']} |")
    (M / "definitive_table_5set.md").write_text("\n".join(md) + "\n")

    print("wrote:")
    for f in ["definitive_table_5set.tsv", "definitive_table_5set_wide.tsv",
              "definitive_table_5set.md", "definitive_headline.json",
              "definitive_thresholds.tsv", "definitive_mimag.tsv",
              "definitive_domain_restriction.tsv"]:
        p = M / f
        print(f"  {p}  ({p.stat().st_size if p.exists() else 0} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
