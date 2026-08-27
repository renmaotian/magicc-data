#!/usr/bin/env python3
"""
WS3.10 mitigation diagnostic — PART 3: post-hoc, size-conditioned recalibration
of MAGICC V5's outputs, and the test of whether it generalises.

NOTHING here modifies models/magicc_v5.onnx.  The recalibrator is a small
post-processing function applied to V5's outputs; V5 stays frozen.

Recalibrator
------------
Inputs are only things available at inference time with no reference database:
    pred_completeness, pred_contamination, log10(assembly Mbp), log10(n_contigs)
Targets are the ground-truth completeness / contamination.
Two families are fitted and both are reported:
    linear : residual (truth - pred) ~ pred + pred^2 + z(log10 size)
             + z(log10 size) x pred + hinge(log10 size below 2 Mbp) + log10 contigs
    gbm    : HistGradientBoostingRegressor(max_depth=3, 200 iter) on the residual

Evaluation protocols (fit and test are ALWAYS disjoint)
------------------------------------------------------
  CV-A  reference-disjoint  : 5-fold GroupKFold on dominant reference accession.
                              Deployment case "lineage represented in training".
  CV-B  phylum-disjoint     : leave-one-phylum-out.  The recalibrator has never
                              seen the test lineage, but V5 has.
  CV-C  set-disjoint        : leave-one-benchmark-set-out (robustness).
  CV-D  GENUINELY NOVEL     : the leave-phylum-out holdout model (WS1.6) scores
                              lineages it never saw in training.  The
                              recalibrator is fitted on that SAME model's
                              predictions for in-distribution lineages and
                              applied to the six held-out phyla.  This is the
                              ceiling test: if a learned size prior cannot
                              transfer to a genuinely novel lineage, it fails
                              here.

Everything is evaluated on ground truth, clustered by reference genome, with
5,000-iteration cluster bootstrap CIs on the PAIRED change (recalibrated minus
raw), so that any degradation on normal-size genomes is visible with a CI.

Outputs -> results/revision/real_data/reduced_genome/mitigation/
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_framework():
    spec = importlib.util.spec_from_file_location(
        "magicc_metrics_framework", ROOT / "scripts" / "101_metrics_framework.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["magicc_metrics_framework"] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()

from sklearn.ensemble import HistGradientBoostingRegressor   # noqa: E402
from sklearn.model_selection import GroupKFold               # noqa: E402

OUT = ROOT / "results" / "revision" / "real_data" / "reduced_genome" / "mitigation"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 5000
SETS = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]


# --------------------------------------------------------------------------
def paired_cluster_boot(delta, clusters, seed, n_boot=N_BOOT, stat=np.mean):
    delta = np.asarray(delta, float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    b = np.empty(n_boot)
    for k in range(n_boot):
        b[k] = stat(delta[np.concatenate([idx_by[j] for j in picks[k]])])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    p_lo = (np.sum(b <= 0) + 1) / (n_boot + 1)
    p_hi = (np.sum(b >= 0) + 1) / (n_boot + 1)
    return float(stat(delta)), float(lo), float(hi), float(min(1.0, 2 * min(p_lo, p_hi)))


def design(pred_c, pred_k, asm_bp, n_contigs):
    lg = np.log10(np.clip(asm_bp, 1, None) / 1e6)
    lc = np.log10(np.clip(n_contigs, 1, None))
    return np.column_stack([pred_c, pred_k, lg, lc])


def fit_linear(X, y):
    pc, pk, lg, lc = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    z = (lg - lg.mean()) / (lg.std() + 1e-9)
    hinge = np.minimum(lg - np.log10(2.0), 0.0)      # active only below 2 Mbp
    D = np.column_stack([np.ones(len(X)), pc / 100, (pc / 100) ** 2, pk / 100,
                         z, z * (pc / 100), hinge, hinge * (pc / 100), lc])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return {"kind": "linear", "beta": beta, "lg_mean": lg.mean(), "lg_std": lg.std() + 1e-9}


def apply_linear(mdl, X):
    pc, pk, lg, lc = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    z = (lg - mdl["lg_mean"]) / mdl["lg_std"]
    hinge = np.minimum(lg - np.log10(2.0), 0.0)
    D = np.column_stack([np.ones(len(X)), pc / 100, (pc / 100) ** 2, pk / 100,
                         z, z * (pc / 100), hinge, hinge * (pc / 100), lc])
    return D @ mdl["beta"]


def fit_gbm(X, y, seed):
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.06,
                                      l2_regularization=1.0, min_samples_leaf=40,
                                      random_state=seed)
    m.fit(X, y)
    return {"kind": "gbm", "model": m}


def apply_model(mdl, X):
    return apply_linear(mdl, X) if mdl["kind"] == "linear" else mdl["model"].predict(X)


def recalibrate(fit_df, test_df, kind, seed):
    """Fit on fit_df, return recalibrated completeness/contamination for test_df."""
    Xf = design(fit_df.pred_c.values, fit_df.pred_k.values,
                fit_df.total_length.values, fit_df.n_contigs.values)
    Xt = design(test_df.pred_c.values, test_df.pred_k.values,
                test_df.total_length.values, test_df.n_contigs.values)
    out = {}
    for tgt, pcol, tcol in [("comp", "pred_c", "true_c"), ("cont", "pred_k", "true_k")]:
        resid = fit_df[tcol].values - fit_df[pcol].values
        mdl = fit_linear(Xf, resid) if kind == "linear" else fit_gbm(Xf, resid, seed)
        corr = apply_model(mdl, Xt)
        v = test_df[pcol].values + corr
        out[tgt] = np.clip(v, 0.0, 100.0) if tgt == "comp" else np.clip(v, 0.0, None)
    return out["comp"], out["cont"]


# --------------------------------------------------------------------------
def load_pooled():
    frames = []
    for s in SETS:
        base = ROOT / "data" / "benchmarks" / s
        md = pd.read_csv(base / "metadata.tsv", sep="\t")[
            ["genome_id", "true_completeness", "true_contamination",
             "dominant_accession", "dominant_phylum", "n_contigs", "total_length"]]
        mg = pd.read_csv(base / "magicc_v5_predictions.tsv", sep="\t")[
            ["genome_id", "pred_completeness", "pred_contamination"]]
        d = md.merge(mg, on="genome_id")
        d["set"] = s
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"true_completeness": "true_c", "true_contamination": "true_k",
                          "pred_completeness": "pred_c", "pred_contamination": "pred_k"})
    tot = d.true_c + d.true_k
    d["ref_len_true"] = 100.0 * d.total_length / tot
    d["asm_Mbp"] = d.total_length / 1e6
    d["ref_Mbp"] = d.ref_len_true / 1e6
    d["size_stratum"] = pd.cut(d.ref_Mbp, [0, 1, 1.5, 2, 3, 5, np.inf],
                               labels=["<1 Mbp", "1-1.5 Mbp", "1.5-2 Mbp",
                                       "2-3 Mbp", "3-5 Mbp", ">5 Mbp"]).astype(str)
    d["reduced"] = np.where(d.ref_Mbp < 2.0, "reduced (<2 Mbp)", "normal (>=2 Mbp)")
    return d


def evaluate(d, cal_c, cal_k, protocol, extra_strata=True):
    d = d.copy()
    d["cal_c"], d["cal_k"] = cal_c, cal_k
    rows = []

    def block(g, label):
        if len(g) < 15:
            return
        seed = fw.stable_hash(f"{protocol}|{label}") % (2 ** 31)
        rec = {"protocol": protocol, "stratum": label, "n": len(g),
               "n_reference_clusters": g.dominant_accession.nunique()}
        for m, pc, cc in [("comp", "pred_c", "cal_c"), ("cont", "pred_k", "cal_k")]:
            t = g[f"true_{'c' if m == 'comp' else 'k'}"].values
            raw_ae = np.abs(g[pc].values - t)
            cal_ae = np.abs(g[cc].values - t)
            e, lo, hi, p = paired_cluster_boot(cal_ae - raw_ae, g.dominant_accession,
                                               seed + (0 if m == "comp" else 1))
            be, blo, bhi, bp = paired_cluster_boot(
                (g[cc].values - t) - (g[pc].values - t), g.dominant_accession,
                seed + (2 if m == "comp" else 3))
            rec.update({
                f"{m}_MAE_raw": round(float(raw_ae.mean()), 3),
                f"{m}_MAE_recal": round(float(cal_ae.mean()), 3),
                f"{m}_dMAE": round(e, 3),
                f"{m}_dMAE_ci95": f"[{lo:.3f}, {hi:.3f}]",
                f"{m}_dMAE_p": f"{p:.4f}",
                f"{m}_bias_raw": round(float((g[pc].values - t).mean()), 3),
                f"{m}_bias_recal": round(float((g[cc].values - t).mean()), 3),
                f"{m}_dbias": round(be, 3),
                f"{m}_dbias_ci95": f"[{blo:.3f}, {bhi:.3f}]",
            })
        rows.append(rec)

    block(d, "ALL")
    for lab, g in d.groupby("reduced"):
        block(g, lab)
    if extra_strata:
        for lab in ["<1 Mbp", "1-1.5 Mbp", "1.5-2 Mbp", "2-3 Mbp", "3-5 Mbp", ">5 Mbp"]:
            g = d[d.size_stratum == lab]
            block(g, f"ref {lab}")
        if "set" in d.columns:
            for lab, g in d.groupby("set"):
                block(g, f"set:{lab}")
        if "dominant_phylum" in d.columns:
            for lab, g in d.groupby("dominant_phylum"):
                if len(g) >= 60:
                    block(g, f"phylum:{lab}")
        # the ground-truthed reduced-genome anchor from WS3.10
        anchor = d[(d.get("set", pd.Series(index=d.index, dtype=object)) == "set_C_clean")
                   & (d.true_k < 5) & (d.true_c >= 90)] if "set" in d.columns else d.iloc[:0]
        if len(anchor) >= 15:
            block(anchor, "ANCHOR set_C_clean clean+HQ (WS3.10 -8.68/+5.09)")
    return pd.DataFrame(rows), d


def false_fail(d, ccol, kcol, tau=5.0):
    clean = d[d.true_k < tau]
    if len(clean) == 0:
        return float("nan")
    return float((clean[kcol] >= tau).mean())


# --------------------------------------------------------------------------
def main():
    d = load_pooled()
    print(f"[142] pooled n={len(d)} refs={d.dominant_accession.nunique()}")
    all_eval, all_ff = [], []

    for kind in ["linear", "gbm"]:
        # ---------- CV-A: reference-disjoint 5-fold ------------------------
        oof_c = np.full(len(d), np.nan)
        oof_k = np.full(len(d), np.nan)
        gkf = GroupKFold(n_splits=5)
        for f, (tr, te) in enumerate(gkf.split(d, groups=d.dominant_accession)):
            c, k = recalibrate(d.iloc[tr], d.iloc[te], kind,
                               fw.stable_hash(f"cvA|{kind}|{f}") % (2 ** 31))
            oof_c[te], oof_k[te] = c, k
        ev, dA = evaluate(d, oof_c, oof_k, f"CV-A reference-disjoint [{kind}]")
        all_eval.append(ev)
        all_ff.append({"protocol": f"CV-A reference-disjoint [{kind}]",
                       "set": "set_C_clean",
                       "false_fail_raw": round(false_fail(dA[dA.set == "set_C_clean"],
                                                          "pred_c", "pred_k"), 4),
                       "false_fail_recal": round(false_fail(dA[dA.set == "set_C_clean"],
                                                            "cal_c", "cal_k"), 4),
                       "n_truly_clean": int((dA[dA.set == "set_C_clean"].true_k < 5).sum())})

        # ---------- CV-B: phylum-disjoint ----------------------------------
        oof_c = np.full(len(d), np.nan)
        oof_k = np.full(len(d), np.nan)
        for ph in d.dominant_phylum.unique():
            te = np.where(d.dominant_phylum.values == ph)[0]
            tr = np.where(d.dominant_phylum.values != ph)[0]
            if len(te) < 20 or len(tr) < 200:
                continue
            c, k = recalibrate(d.iloc[tr], d.iloc[te], kind,
                               fw.stable_hash(f"cvB|{kind}|{ph}") % (2 ** 31))
            oof_c[te], oof_k[te] = c, k
        m = ~np.isnan(oof_c)
        ev, _ = evaluate(d[m], oof_c[m], oof_k[m], f"CV-B phylum-disjoint [{kind}]")
        all_eval.append(ev)

        # ---------- CV-C: set-disjoint -------------------------------------
        oof_c = np.full(len(d), np.nan)
        oof_k = np.full(len(d), np.nan)
        for st in SETS:
            te = np.where(d.set.values == st)[0]
            tr = np.where(d.set.values != st)[0]
            c, k = recalibrate(d.iloc[tr], d.iloc[te], kind,
                               fw.stable_hash(f"cvC|{kind}|{st}") % (2 ** 31))
            oof_c[te], oof_k[te] = c, k
        ev, dC = evaluate(d, oof_c, oof_k, f"CV-C set-disjoint [{kind}]")
        all_eval.append(ev)
        all_ff.append({"protocol": f"CV-C set-disjoint [{kind}]", "set": "set_C_clean",
                       "false_fail_raw": round(false_fail(dC[dC.set == "set_C_clean"],
                                                          "pred_c", "pred_k"), 4),
                       "false_fail_recal": round(false_fail(dC[dC.set == "set_C_clean"],
                                                            "cal_c", "cal_k"), 4),
                       "n_truly_clean": int((dC[dC.set == "set_C_clean"].true_k < 5).sum())})

    ev_all = pd.concat(all_eval, ignore_index=True)
    ev_all.to_csv(OUT / "recalibration_cv_results.tsv", sep="\t", index=False)
    pd.DataFrame(all_ff).to_csv(OUT / "recalibration_false_fail.tsv", sep="\t", index=False)
    print(ev_all[ev_all.stratum.isin(["ALL", "reduced (<2 Mbp)", "normal (>=2 Mbp)"])]
          [["protocol", "stratum", "n", "comp_MAE_raw", "comp_MAE_recal", "comp_dMAE",
            "comp_dMAE_ci95", "comp_bias_raw", "comp_bias_recal",
            "cont_MAE_raw", "cont_MAE_recal", "cont_dMAE"]].to_string(index=False))

    # ---------------- CV-D: genuinely novel lineages ------------------------
    h = pd.read_csv(ROOT / "results" / "revision" / "holdout" /
                    "per_sample_predictions.tsv.gz", sep="\t")
    h = h.rename(columns={"true_completeness": "true_c", "true_contamination": "true_k"})
    h["ref_Mbp"] = h.dominant_genome_size / 1e6
    h["size_stratum"] = pd.cut(h.ref_Mbp, [0, 1, 1.5, 2, 3, 5, np.inf],
                               labels=["<1 Mbp", "1-1.5 Mbp", "1.5-2 Mbp",
                                       "2-3 Mbp", "3-5 Mbp", ">5 Mbp"]).astype(str)
    h["reduced"] = np.where(h.ref_Mbp < 2.0, "reduced (<2 Mbp)", "normal (>=2 Mbp)")

    novel_rows, novel_eval = [], []
    for base_model, cc, kk in [("holdout_model", "holdout_completeness", "holdout_contamination"),
                               ("V5", "V5_completeness", "V5_contamination")]:
        hh = h.rename(columns={cc: "pred_c", kk: "pred_k"}).copy()
        fitd = hh[hh.group == "in_distribution"]
        for kind in ["linear", "gbm"]:
            for grp in sorted(hh.group.unique()):
                if grp == "in_distribution":
                    continue
                te = hh[hh.group == grp]
                c, k = recalibrate(fitd, te, kind,
                                   fw.stable_hash(f"cvD|{base_model}|{kind}|{grp}") % (2 ** 31))
                ev, _ = evaluate(te, c, k,
                                 f"CV-D novel lineage [{base_model} + {kind}]",
                                 extra_strata=False)
                ev["held_out_group"] = grp
                ev["base_model"] = base_model
                ev["recalibrator"] = kind
                novel_eval.append(ev)
            # in-distribution control, reference-disjoint CV inside in_distribution
            oof_c = np.full(len(fitd), np.nan)
            oof_k = np.full(len(fitd), np.nan)
            gkf = GroupKFold(n_splits=5)
            for f, (tr, te) in enumerate(gkf.split(fitd, groups=fitd.dominant_accession)):
                c, k = recalibrate(fitd.iloc[tr], fitd.iloc[te], kind,
                                   fw.stable_hash(f"cvDctl|{base_model}|{kind}|{f}") % (2 ** 31))
                oof_c[te], oof_k[te] = c, k
            ev, _ = evaluate(fitd, oof_c, oof_k,
                             f"CV-D novel lineage [{base_model} + {kind}]",
                             extra_strata=False)
            ev["held_out_group"] = "in_distribution (CONTROL, ref-disjoint CV)"
            ev["base_model"] = base_model
            ev["recalibrator"] = kind
            novel_eval.append(ev)
    nv = pd.concat(novel_eval, ignore_index=True)
    nv.to_csv(OUT / "recalibration_novel_lineage.tsv", sep="\t", index=False)
    print("\n[142] CV-D novel lineage (ALL stratum):")
    print(nv[nv.stratum == "ALL"][["base_model", "recalibrator", "held_out_group", "n",
                                   "comp_MAE_raw", "comp_MAE_recal", "comp_dMAE",
                                   "comp_bias_raw", "comp_bias_recal",
                                   "cont_MAE_raw", "cont_MAE_recal"]].to_string(index=False))

    summary = {
        "script": "scripts/142_size_conditioned_recalibration.py",
        "model_frozen": "models/magicc_v5.onnx unchanged; recalibration is post-hoc",
        "n_pooled": int(len(d)),
        "n_reference_clusters": int(d.dominant_accession.nunique()),
        "n_boot": N_BOOT,
        "recalibrator_inputs": ["pred_completeness", "pred_contamination",
                                "log10(assembly Mbp)", "log10(n_contigs)"],
        "cv_results": ev_all.to_dict("records"),
        "false_fail": all_ff,
        "novel_lineage": nv.to_dict("records"),
    }
    (OUT / "recalibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[142] wrote -> {OUT}")


if __name__ == "__main__":
    main()
