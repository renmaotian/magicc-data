#!/usr/bin/env python3
"""
WS3.10 mitigation diagnostic — PART 4: the CEILING test.

Part 3 (script 142) fitted the size-conditioned recalibrator on at most 5,000
samples.  A weak result there could just be a small fitting corpus.  This script
removes that excuse and asks the question that actually matters:

    Is a post-hoc size-conditioned correction a MECHANISM that transfers, or a
    LINEAGE-SPECIFIC PRIOR that cannot?

Three experiments, all with fit/test disjoint and clustered by reference genome.

  A. BEST-CASE known-lineage recalibration.
     Fit on ~60,000 ground-truthed samples from V5's own held-out TEST split
     (9,999 reference genomes; ZERO reference overlap with the five benchmark
     sets - verified), apply to the five ground-truthed benchmark sets.

  B. NOVEL-LINEAGE TRANSFER.
     The WS1.6 leave-phylum-out model never saw six phyla.  Fit the recalibrator
     on ~60,000 samples of that SAME model's predictions over its own
     in-distribution test split (8,034 references, contains none of the six
     panel phyla - verified), then apply it to that model's predictions on the
     six genuinely novel phyla.  Same model, same recalibrator family, large
     fitting corpus; the only thing that changes is whether the test lineage was
     in training.

  C. ORACLE within-lineage upper bound.
     Fit the recalibrator ON the novel lineage itself, with reference-disjoint
     5-fold CV so it is still honest, and test on held-out references of that
     same lineage.  This is what a correction could achieve if labelled data
     from the novel lineage were available.  The gap between C and B is exactly
     the part of the correction that is a learned lineage prior rather than a
     transferable mechanism.

Recalibrator inputs are only inference-observable: predicted completeness,
predicted contamination, log10(assembly Mbp).  Neither ONNX model is modified.

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

import h5py                                                   # noqa: E402
import onnxruntime as ort                                     # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor    # noqa: E402
from sklearn.model_selection import GroupKFold                # noqa: E402

OUT = ROOT / "results" / "revision" / "real_data" / "reduced_genome" / "mitigation"
OUT.mkdir(parents=True, exist_ok=True)

N_FIT = 60000
N_BOOT = 5000
N_THREADS = 4
SETS = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]


# --------------------------------------------------------------------------
def paired_cluster_boot(delta, clusters, seed, n_boot=N_BOOT):
    delta = np.asarray(delta, float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    b = np.empty(n_boot)
    for k in range(n_boot):
        b[k] = np.nanmean(delta[np.concatenate([idx_by[j] for j in picks[k]])])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(np.nanmean(delta)), float(lo), float(hi)


def design(pred_c, pred_k, asm_bp):
    return np.column_stack([pred_c, pred_k, np.log10(np.clip(asm_bp, 1, None) / 1e6)])


def fit_corrections(Xf, yc, yk, Xt, seed, kind="gbm"):
    """Fit residual (truth - pred) models on the fit set and return the
    CORRECTIONS for the test set.  Callers must add them to the raw prediction."""
    outs = []
    for y in (yc, yk):
        if kind == "gbm":
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                              learning_rate=0.06, l2_regularization=1.0,
                                              min_samples_leaf=40, random_state=seed)
            m.fit(Xf, y)
            outs.append(m.predict(Xt))
        else:
            lg = Xf[:, 2]
            mu, sd = lg.mean(), lg.std() + 1e-9

            def D(X):
                pc, pk, l = X[:, 0], X[:, 1], X[:, 2]
                z = (l - mu) / sd
                h = np.minimum(l - np.log10(2.0), 0.0)
                return np.column_stack([np.ones(len(X)), pc / 100, (pc / 100) ** 2,
                                        pk / 100, z, z * (pc / 100), h, h * (pc / 100)])
            beta, *_ = np.linalg.lstsq(D(Xf), y, rcond=None)
            outs.append(D(Xt) @ beta)
    return outs


def score(df, cal_c, cal_k, label, protocol, extra=None):
    rows = []
    df = df.copy()
    df["cal_c"], df["cal_k"] = np.clip(cal_c, 0, 100), np.clip(cal_k, 0, None)
    seed = fw.stable_hash(f"{protocol}|{label}") % (2 ** 31)
    rec = {"protocol": protocol, "stratum": label, "n": len(df),
           "n_reference_clusters": df.cluster.nunique()}
    for m, pc, cc, tc in [("comp", "pred_c", "cal_c", "true_c"),
                          ("cont", "pred_k", "cal_k", "true_k")]:
        t = df[tc].values
        raw_ae, cal_ae = np.abs(df[pc].values - t), np.abs(df[cc].values - t)
        e, lo, hi = paired_cluster_boot(cal_ae - raw_ae, df.cluster, seed)
        be, blo, bhi = paired_cluster_boot((df[cc].values - t) - (df[pc].values - t),
                                           df.cluster, seed + 1)
        rec.update({
            f"{m}_MAE_raw": round(float(raw_ae.mean()), 3),
            f"{m}_MAE_recal": round(float(cal_ae.mean()), 3),
            f"{m}_dMAE": round(e, 3), f"{m}_dMAE_ci95": f"[{lo:.3f}, {hi:.3f}]",
            f"{m}_bias_raw": round(float((df[pc].values - t).mean()), 3),
            f"{m}_bias_recal": round(float((df[cc].values - t).mean()), 3),
            f"{m}_dbias": round(be, 3), f"{m}_dbias_ci95": f"[{blo:.3f}, {bhi:.3f}]",
        })
    if extra:
        rec.update(extra)
    rows.append(rec)
    return rows


def load_fit_corpus(h5_path, onnx_path, tag, n_fit=N_FIT):
    """Run a frozen ONNX model over a stratified subsample of its own test split."""
    cache = OUT / f"fit_corpus_{tag}.tsv.gz"
    if cache.exists():
        d = pd.read_csv(cache, sep="\t")
        print(f"[143] fit corpus {tag} (cached): n={len(d)} refs={d.cluster.nunique()}")
        return d
    rng = np.random.default_rng(fw.stable_hash(f"143fit|{tag}") % (2 ** 31))
    with h5py.File(h5_path, "r") as f:
        meta = f["test/metadata"][:]
        comp = meta["completeness"].astype(float)
        cont = meta["contamination"].astype(float)
        L = meta["genome_full_length"].astype(float)
        acc = np.array([a.decode() for a in meta["dominant_accession"]])
        phy = np.array([a.decode() for a in meta["dominant_phylum"]])
        bins = np.digitize(L / 1e6, [1.0, 1.5, 2.0, 3.0, 5.0])
        take = []
        per = n_fit // (bins.max() + 1)
        for b in range(bins.max() + 1):
            cand = np.where(bins == b)[0]
            take.append(rng.choice(cand, size=min(per, len(cand)), replace=False))
        idx = np.sort(np.concatenate(take))
        K = f["test/kmer_features"][idx]
        A = f["test/assembly_features"][idx]
    comp, cont, L, acc, phy = comp[idx], cont[idx], L[idx], acc[idx], phy[idx]
    so = ort.SessionOptions()
    so.intra_op_num_threads = N_THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    P = np.empty((len(idx), 2), float)
    B = 1024
    for i in range(0, len(idx), B):
        P[i:i + B] = sess.run(None, {"kmer_features": K[i:i + B].astype(np.float32),
                                     "assembly_features": A[i:i + B].astype(np.float32)})[0]
    d = pd.DataFrame({"true_c": comp, "true_k": cont, "pred_c": P[:, 0], "pred_k": P[:, 1],
                      "cluster": acc, "phylum": phy, "ref_len": L})
    d["total_length"] = d.ref_len * (d.true_c + d.true_k) / 100.0
    print(f"[143] fit corpus {tag}: n={len(d)} refs={d.cluster.nunique()} "
          f"comp MAE={np.abs(d.pred_c - d.true_c).mean():.3f} "
          f"| ref<1Mbp {(d.ref_len < 1e6).mean()*100:.1f}% "
          f"| ref<2Mbp {(d.ref_len < 2e6).mean()*100:.1f}%")
    d.to_csv(cache, sep="\t", index=False)
    return d


def main():
    results, meta_notes = [], {}

    # ================= A. best-case known-lineage recalibration =============
    fitA = load_fit_corpus(ROOT / "data/features/magicc_v5_features.h5",
                           ROOT / "models/magicc_v5.onnx", "V5")
    meta_notes["fitA"] = {"n": int(len(fitA)), "refs": int(fitA.cluster.nunique()),
                          "pct_ref_lt_1Mbp": float((fitA.ref_len < 1e6).mean()),
                          "pct_ref_lt_2Mbp": float((fitA.ref_len < 2e6).mean())}
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
    bench = pd.concat(frames, ignore_index=True).rename(columns={
        "true_completeness": "true_c", "true_contamination": "true_k",
        "pred_completeness": "pred_c", "pred_contamination": "pred_k",
        "dominant_accession": "cluster"})
    bench["ref_len"] = 100.0 * bench.total_length / (bench.true_c + bench.true_k)
    bench["reduced"] = np.where(bench.ref_len < 2e6, "reduced (<2 Mbp)", "normal (>=2 Mbp)")
    bench["ref_stratum"] = pd.cut(bench.ref_len / 1e6, [0, 1, 1.5, 2, 3, 5, np.inf],
                                  labels=["<1 Mbp", "1-1.5 Mbp", "1.5-2 Mbp", "2-3 Mbp",
                                          "3-5 Mbp", ">5 Mbp"]).astype(str)

    Xf = design(fitA.pred_c.values, fitA.pred_k.values, fitA.total_length.values)
    Xt = design(bench.pred_c.values, bench.pred_k.values, bench.total_length.values)
    for kind in ["gbm", "linear"]:
        dc, dk = fit_corrections(Xf, (fitA.true_c - fitA.pred_c).values,
                                 (fitA.true_k - fitA.pred_k).values, Xt,
                                 fw.stable_hash(f"143A|{kind}") % (2 ** 31), kind)
        cc = bench.pred_c.values + dc          # correction added back to prediction
        kk = bench.pred_k.values + dk
        prot = f"A. best-case known-lineage [fit n={len(fitA)} V5 test split, {kind}]"
        results += score(bench, cc, kk, "ALL", prot)
        for lab, g in bench.groupby("reduced"):
            results += score(g, cc[g.index], kk[g.index], lab, prot)
        for lab in ["<1 Mbp", "1-1.5 Mbp", "1.5-2 Mbp", "2-3 Mbp", "3-5 Mbp", ">5 Mbp"]:
            g = bench[bench.ref_stratum == lab]
            if len(g) >= 15:
                results += score(g, cc[g.index], kk[g.index], f"ref {lab}", prot)
        for lab, g in bench.groupby("set"):
            results += score(g, cc[g.index], kk[g.index], f"set:{lab}", prot)
        anc = bench[(bench.set == "set_C_clean") & (bench.true_k < 5) & (bench.true_c >= 90)]
        if len(anc) >= 10:
            results += score(anc, cc[anc.index], kk[anc.index],
                             "ANCHOR set_C_clean clean+HQ", prot)

    # ================= B / C. novel-lineage transfer and oracle =============
    fitB = load_fit_corpus(ROOT / "data/features/holdout_phylum_features.h5",
                           ROOT / "models/magicc_holdout_phylum.onnx", "HOLDOUT")
    meta_notes["fitB"] = {"n": int(len(fitB)), "refs": int(fitB.cluster.nunique()),
                          "pct_ref_lt_1Mbp": float((fitB.ref_len < 1e6).mean()),
                          "pct_ref_lt_2Mbp": float((fitB.ref_len < 2e6).mean()),
                          "panel_phyla_present": int(fitB.phylum.isin(
                              ["Patescibacteriota", "DPANN", "Bacteroidota", "Bacteroidota_A",
                               "Halobacteriota", "Campylobacterota"]).sum())}

    h = pd.read_csv(ROOT / "results/revision/holdout/per_sample_predictions.tsv.gz", sep="\t")
    h = h.rename(columns={"true_completeness": "true_c", "true_contamination": "true_k",
                          "holdout_completeness": "pred_c",
                          "holdout_contamination": "pred_k",
                          "dominant_accession": "cluster"})
    h["ref_len"] = h.dominant_genome_size

    XfB = design(fitB.pred_c.values, fitB.pred_k.values, fitB.total_length.values)
    for kind in ["gbm", "linear"]:
        mdl_seed = fw.stable_hash(f"143B|{kind}") % (2 ** 31)
        for grp in sorted(h.group.unique()):
            g = h[h.group == grp].reset_index(drop=True)
            Xt = design(g.pred_c.values, g.pred_k.values, g.total_length.values)
            dc, dk = fit_corrections(XfB, (fitB.true_c - fitB.pred_c).values,
                                     (fitB.true_k - fitB.pred_k).values, Xt,
                                     mdl_seed, kind)
            cc = g.pred_c.values + dc
            kk = g.pred_k.values + dk
            prot = (f"B. NOVEL-lineage transfer [holdout model, fit n={len(fitB)} "
                    f"in-distribution, {kind}]")
            results += score(g, cc, kk, f"group:{grp}", prot,
                             extra={"lineage_novel_to_base_model":
                                    "no" if grp == "in_distribution" else "YES"})

            # C. oracle: fit ON this lineage, reference-disjoint 5-fold CV
            if g.cluster.nunique() >= 10:
                oc = np.full(len(g), np.nan)
                ok = np.full(len(g), np.nan)
                gkf = GroupKFold(n_splits=5)
                Xg = design(g.pred_c.values, g.pred_k.values, g.total_length.values)
                for fi, (tr, te) in enumerate(gkf.split(g, groups=g.cluster)):
                    a, b = fit_corrections(
                        Xg[tr], (g.true_c - g.pred_c).values[tr],
                        (g.true_k - g.pred_k).values[tr], Xg[te],
                        fw.stable_hash(f"143C|{kind}|{grp}|{fi}") % (2 ** 31), kind)
                    oc[te] = g.pred_c.values[te] + a
                    ok[te] = g.pred_k.values[te] + b
                prot_c = (f"C. ORACLE within-lineage [holdout model, fit ON the lineage, "
                          f"ref-disjoint 5-fold, {kind}]")
                results += score(g, oc, ok, f"group:{grp}", prot_c,
                                 extra={"lineage_novel_to_base_model":
                                        "no" if grp == "in_distribution" else "YES"})

    df = pd.DataFrame(results)
    df.to_csv(OUT / "recalibration_ceiling.tsv", sep="\t", index=False)

    show = ["protocol", "stratum", "n", "comp_MAE_raw", "comp_MAE_recal", "comp_dMAE",
            "comp_dMAE_ci95", "comp_bias_raw", "comp_bias_recal",
            "cont_MAE_raw", "cont_MAE_recal", "cont_dMAE"]
    print("\n=== A. best-case known-lineage ===")
    print(df[df.protocol.str.startswith("A.") & df.protocol.str.contains("gbm")][show]
          .to_string(index=False))
    print("\n=== B. novel-lineage transfer vs C. oracle (gbm) ===")
    sel = df[(df.protocol.str.startswith(("B.", "C."))) & df.protocol.str.contains("gbm")]
    print(sel[show].to_string(index=False))

    (OUT / "recalibration_ceiling_summary.json").write_text(json.dumps({
        "script": "scripts/215_recalibration_generalization_ceiling.py",
        "models_unmodified": ["models/magicc_v5.onnx", "models/magicc_holdout_phylum.onnx"],
        "fit_corpora": meta_notes,
        "recalibrator_inputs": ["pred_completeness", "pred_contamination",
                                "log10(assembly Mbp)"],
        "n_boot": N_BOOT,
        "results": df.to_dict("records"),
    }, indent=2))
    print(f"\n[143] wrote -> {OUT}")


if __name__ == "__main__":
    main()
