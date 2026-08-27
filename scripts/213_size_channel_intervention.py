#!/usr/bin/env python3
"""
WS3.10 mitigation diagnostic — PART 2: CAUSAL size-channel intervention on the
FROZEN V5 model.  No retraining, no weight change, no new model version.

Question
--------
MAGICC estimates completeness = retained_bp / L, where the reference length L is
unobservable and must be inferred.  Two mechanisms could produce the observed
reduced-genome under-call:

  (a) a biased *composition -> L* map (the learned lineage prior is wrong for
      small lineages), or
  (b) the estimator additionally reading the *absolute* size of the assembly as
      evidence about L (so a small assembly is read as "a fragment of something
      bigger").

These are separable by intervention.  Take a real assembly, hold its k-mer
COMPOSITION exactly fixed, and rescale the whole thing by a factor s (a
counterfactual assembly of the same organism that is s x longer/shorter).
Define

      phi = d log(predicted completeness) / d log(s)

  phi = 1  ->  the size change is booked entirely as a completeness change, i.e.
               the model's belief about L depends on composition ALONE.  This is
               the design intent; any residual error is then mechanism (a).
  phi < 1  ->  part of the size change is booked as a change in L: the model is
               reading absolute size as evidence about reference length.
               1 - phi is exactly the strength of mechanism (b).

Three interventions isolate the channel through which absolute size enters:
  coherent      - raw k-mer counts rescaled AND the 7 summary features recomputed
  assembly_only - only the 7 summary features moved (k-mer branch untouched)
  kmer_only     - only the 9,249 k-mer counts rescaled (assembly branch untouched)

Substrate: the frozen V5 held-out TEST split (data/features/magicc_v5_features.h5),
which carries ground-truth completeness, contamination and reference length.
Raw counts are recovered exactly by inverting the stored normalisation
(z -> expm1(z*sd + mean)); the round trip is verified and reported.

Outputs -> results/revision/real_data/reduced_genome/mitigation/
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_framework():
    spec = importlib.util.spec_from_file_location(
        "magicc_metrics_framework", ROOT / "scripts" / "101_metrics_framework.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["magicc_metrics_framework"] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_framework()

import h5py                                        # noqa: E402
import onnxruntime as ort                          # noqa: E402
import pandas as pd                                # noqa: E402

from magicc.assembly_stats import compute_assembly_stats, FEATURE_NAMES  # noqa: E402
from magicc.normalization import FeatureNormalizer                       # noqa: E402

OUT = ROOT / "results" / "revision" / "real_data" / "reduced_genome" / "mitigation"
OUT.mkdir(parents=True, exist_ok=True)

H5 = ROOT / "data" / "features" / "magicc_v5_features.h5"
NORM = ROOT / "data" / "features" / "normalization_params.json"
ONNX = ROOT / "models" / "magicc_v5.onnx"

N_SAMPLE = 4000
SCALES = [0.50, 0.71, 0.85, 1.00, 1.18, 1.41, 2.00]
SEED = fw.stable_hash("213_size_channel_intervention") % (2 ** 31)
N_THREADS = 4
N_BOOT = 2000


def recompute_assembly(counts: np.ndarray) -> np.ndarray:
    """7 summary features from raw counts, exactly as magicc/pipeline.py does."""
    out = np.zeros((counts.shape[0], 7), dtype=np.float64)
    for i in range(counts.shape[0]):
        tot = counts[i].sum()
        log10_total = float(np.log10(tot)) if tot > 0 else 0.0
        out[i] = compute_assembly_stats(log10_total, counts[i])
    return out


def cluster_boot_mean(vals, clusters, seed, n_boot=N_BOOT):
    vals = np.asarray(vals, float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    b = np.empty(n_boot)
    for k in range(n_boot):
        b[k] = np.nanmean(vals[np.concatenate([idx_by[j] for j in picks[k]])])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(np.nanmean(vals)), float(lo), float(hi)


def main():
    rng = np.random.default_rng(SEED)

    # ---------------- load a stratified subsample of the frozen test split ---
    with h5py.File(H5, "r") as f:
        meta = f["test/metadata"][:]
        n = meta.shape[0]
        comp = meta["completeness"].astype(float)
        cont = meta["contamination"].astype(float)
        L = meta["genome_full_length"].astype(float)
        acc = np.array([a.decode() for a in meta["dominant_accession"]])
        phy = np.array([a.decode() for a in meta["dominant_phylum"]])
        # stratify by reference length so small genomes are well represented
        bins = np.digitize(L / 1e6, [1.0, 1.5, 2.0, 3.0, 5.0])
        take = []
        per = N_SAMPLE // (bins.max() + 1)
        for b in range(bins.max() + 1):
            cand = np.where(bins == b)[0]
            take.append(rng.choice(cand, size=min(per, len(cand)), replace=False))
        idx = np.sort(np.concatenate(take))
        Kz = f["test/kmer_features"][idx].astype(np.float64)
        Az = f["test/assembly_features"][idx].astype(np.float64)
    comp, cont, L, acc, phy = comp[idx], cont[idx], L[idx], acc[idx], phy[idx]
    m = len(idx)
    print(f"[141] n={m} samples, {len(np.unique(acc))} reference clusters")

    # ---------------- invert normalisation to raw k-mer counts --------------
    norm = FeatureNormalizer.load(str(NORM))
    kmean = np.asarray(norm.kmer_mean, float)
    kstd = np.asarray(norm.kmer_std, float)
    raw = np.expm1(Kz * kstd + kmean)
    raw = np.rint(np.clip(raw, 0, None))

    # round-trip verification
    Kz_rt = norm.normalize_kmer(raw)
    rt_kmer = float(np.abs(Kz_rt - Kz).max())
    Az_rt = norm.normalize_assembly(recompute_assembly(raw))
    rt_asm = float(np.abs(Az_rt - Az).max())
    print(f"[141] round-trip max|dz| kmer={rt_kmer:.3e} assembly={rt_asm:.3e}")

    # ---------------- ONNX session (CPU, 4 threads) -------------------------
    so = ort.SessionOptions()
    so.intra_op_num_threads = N_THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])

    def predict(kz, az):
        out = np.empty((kz.shape[0], 2), dtype=np.float64)
        B = 512
        for i in range(0, kz.shape[0], B):
            out[i:i + B] = sess.run(
                None, {"kmer_features": kz[i:i + B].astype(np.float32),
                       "assembly_features": az[i:i + B].astype(np.float32)})[0]
        return out

    base = predict(Kz_rt, Az_rt)          # baseline on round-tripped inputs
    base_stored = predict(Kz, Az)         # baseline on stored inputs
    print(f"[141] baseline comp MAE vs truth = {np.abs(base_stored[:,0]-comp).mean():.3f} pp; "
          f"round-trip pred max diff = {np.abs(base-base_stored).max():.4f} pp")

    # ---------------- interventions -----------------------------------------
    records = []
    per_sample = {}
    for s in SCALES:
        if s < 1.0:
            # binomial thinning: the generative model for a shorter assembly of
            # the same organism (each k-mer occurrence retained with prob s)
            r = np.random.default_rng(SEED + int(round(s * 1000)))
            scaled = r.binomial(raw.astype(np.int64), s).astype(np.float64)
        else:
            scaled = np.rint(raw * s)
        A_scaled = recompute_assembly(scaled)
        Kz_s = norm.normalize_kmer(scaled)
        Az_s = norm.normalize_assembly(A_scaled)

        variants = {
            "coherent": (Kz_s, Az_s),
            "assembly_only": (Kz_rt, Az_s),
            "kmer_only": (Kz_s, Az_rt),
        }
        for vname, (kz, az) in variants.items():
            p = predict(kz, az)
            per_sample[(vname, s)] = p
            dlog_c = np.log(np.clip(p[:, 0], 1e-6, None)) - np.log(np.clip(base[:, 0], 1e-6, None))
            phi = dlog_c / np.log(s) if s != 1.0 else np.full(m, np.nan)
            dcomp = p[:, 0] - base[:, 0]
            seed = fw.stable_hash(f"141|{vname}|{s}") % (2 ** 31)
            e_d, lo_d, hi_d = cluster_boot_mean(dcomp, acc, seed)
            if s != 1.0:
                e_p, lo_p, hi_p = cluster_boot_mean(phi, acc, seed + 1)
            else:
                e_p = lo_p = hi_p = float("nan")
            records.append({
                "variant": vname, "scale_s": s, "log10_s": round(float(np.log10(s)), 4),
                "n": m, "n_reference_clusters": int(len(np.unique(acc))),
                "mean_delta_pred_completeness_pp": round(e_d, 4),
                "delta_ci95": f"[{lo_d:.3f}, {hi_d:.3f}]",
                "phi_attribution_to_completeness": round(e_p, 4),
                "phi_ci95": f"[{lo_p:.3f}, {hi_p:.3f}]" if np.isfinite(e_p) else "",
                "mean_pred_completeness": round(float(p[:, 0].mean()), 3),
                "mean_pred_contamination": round(float(p[:, 1].mean()), 3),
            })
            print(f"  s={s:<5} {vname:<14} dcomp={e_d:+7.3f} pp  phi={e_p:+.3f}")

    df = pd.DataFrame(records)
    df.to_csv(OUT / "intervention_size_channel.tsv", sep="\t", index=False)

    # ---------------- pooled phi from a per-sample regression ---------------
    # slope of log(pred comp) on log(s) across all non-unit scales, per variant,
    # with reference-clustered bootstrap
    pooled = []
    logs = np.array([np.log(s) for s in SCALES])
    for vname in ["coherent", "assembly_only", "kmer_only"]:
        Y = np.column_stack([np.log(np.clip(per_sample[(vname, s)][:, 0], 1e-6, None))
                             for s in SCALES])
        xc = logs - logs.mean()
        slope = (Y - Y.mean(axis=1, keepdims=True)) @ xc / (xc ** 2).sum()
        seed = fw.stable_hash(f"141pooled|{vname}") % (2 ** 31)
        e, lo, hi = cluster_boot_mean(slope, acc, seed)
        pooled.append({"variant": vname, "phi_pooled_regression": round(e, 4),
                       "ci95": f"[{lo:.3f}, {hi:.3f}]", "n": m,
                       "n_reference_clusters": int(len(np.unique(acc))),
                       "phi_if_composition_determines_L": 1.0,
                       "phi_if_size_determines_L": 0.0})
        print(f"[141] pooled phi {vname}: {e:.4f} [{lo:.3f}, {hi:.3f}]")

    # ---------------- LOCAL phi, censoring-aware ----------------------------
    # MAGICC's completeness head is bounded (floor ~50, ceiling 100); large
    # rescales drive many samples into the bounds and compress phi downwards.
    # The primary estimate therefore uses the two smallest rescales (+-0.07 in
    # log10, i.e. x0.85 / x1.18) and, additionally, is repeated on the subset
    # whose predictions stay strictly inside the bounds at every scale used.
    LOCAL = [0.85, 1.00, 1.18]
    logs_l = np.log(np.array(LOCAL))
    xl = logs_l - logs_l.mean()
    for vname in ["coherent", "assembly_only", "kmer_only"]:
        Yl = np.column_stack([np.log(np.clip(per_sample[(vname, s)][:, 0], 1e-6, None))
                              for s in LOCAL])
        sl = (Yl - Yl.mean(axis=1, keepdims=True)) @ xl / (xl ** 2).sum()
        raw_p = np.column_stack([per_sample[(vname, s)][:, 0] for s in LOCAL])
        unc = (raw_p > 52.0).all(axis=1) & (raw_p < 99.0).all(axis=1)
        seed = fw.stable_hash(f"141local|{vname}") % (2 ** 31)
        e, lo, hi = cluster_boot_mean(sl, acc, seed)
        eu, lou, hiu = cluster_boot_mean(sl[unc], acc[unc], seed + 7) if unc.sum() > 30 \
            else (float("nan"),) * 3
        pooled.append({"variant": vname + " [LOCAL x0.85-x1.18]",
                       "phi_pooled_regression": round(e, 4),
                       "ci95": f"[{lo:.3f}, {hi:.3f}]", "n": m,
                       "n_reference_clusters": int(len(np.unique(acc))),
                       "phi_if_composition_determines_L": 1.0,
                       "phi_if_size_determines_L": 0.0,
                       "phi_uncensored_subset": round(eu, 4),
                       "uncensored_ci95": f"[{lou:.3f}, {hiu:.3f}]",
                       "n_uncensored": int(unc.sum()),
                       # 1 - phi is the elasticity of the implied reference length to
                       # observed size ONLY for the coherent rescale; the branch-only
                       # variants are a mediation decomposition with deliberately
                       # inconsistent inputs, so the ratio has no such reading.
                       "implied_L_elasticity_to_observed_size":
                           round(1 - e, 4) if vname == "coherent" else "",
                       "note": ("coherent rescale: composition fixed, absolute size moved"
                                if vname == "coherent"
                                else "mediation decomposition only (inconsistent input "
                                     "pair); do not read as an elasticity")})
        print(f"[141] LOCAL phi {vname}: {e:.4f} [{lo:.3f}, {hi:.3f}]  "
              f"uncensored {eu:.4f} (n={unc.sum()})")

    # phi stratified by true reference length (is the leak size-dependent?)
    strat = []
    refbin = pd.cut(L / 1e6, [0, 1, 1.5, 2, 3, 5, np.inf],
                    labels=["<1 Mbp", "1-1.5 Mbp", "1.5-2 Mbp", "2-3 Mbp", "3-5 Mbp", ">5 Mbp"])
    refbin = np.asarray(refbin.astype(str))
    Yc = np.column_stack([np.log(np.clip(per_sample[("coherent", s)][:, 0], 1e-6, None))
                          for s in LOCAL])
    slope_c = (Yc - Yc.mean(axis=1, keepdims=True)) @ xl / (xl ** 2).sum()
    for lab in ["<1 Mbp", "1-1.5 Mbp", "1.5-2 Mbp", "2-3 Mbp", "3-5 Mbp", ">5 Mbp"]:
        sel = refbin == lab
        if sel.sum() < 30:
            continue
        seed = fw.stable_hash(f"141strat|{lab}") % (2 ** 31)
        e, lo, hi = cluster_boot_mean(slope_c[sel], acc[sel], seed)
        strat.append({"reference_size_stratum": str(lab), "n": int(sel.sum()),
                      "n_reference_clusters": int(len(np.unique(acc[sel]))),
                      "phi_coherent_LOCAL": round(e, 4), "ci95": f"[{lo:.3f}, {hi:.3f}]",
                      "implied_L_elasticity": round(1 - e, 4),
                      "mean_true_completeness": round(float(comp[sel].mean()), 2)})
    pd.DataFrame(pooled).to_csv(OUT / "intervention_phi_pooled.tsv", sep="\t", index=False)
    pd.DataFrame(strat).to_csv(OUT / "intervention_phi_by_reference_size.tsv",
                               sep="\t", index=False)
    print(pd.DataFrame(strat).to_string(index=False))

    summary = {
        "script": "scripts/213_size_channel_intervention.py",
        "model": "models/magicc_v5.onnx (FROZEN, unmodified)",
        "substrate": "data/features/magicc_v5_features.h5 :: test split",
        "n": m, "n_reference_clusters": int(len(np.unique(acc))),
        "scales": SCALES, "seed": int(SEED),
        "roundtrip_max_abs_dz_kmer": rt_kmer,
        "roundtrip_max_abs_dz_assembly": rt_asm,
        "roundtrip_max_abs_pred_diff_pp": float(np.abs(base - base_stored).max()),
        "baseline_comp_MAE_pp": float(np.abs(base_stored[:, 0] - comp).mean()),
        "baseline_cont_MAE_pp": float(np.abs(base_stored[:, 1] - cont).mean()),
        "phi_pooled": pooled,
        "phi_by_reference_size": strat,
        "per_scale": records,
        "interpretation": {
            "phi_eq_1": "size change booked entirely as completeness; model's belief "
                        "about reference length depends on composition alone",
            "phi_lt_1": "model reads absolute assembly size as evidence about reference "
                        "length; 1-phi is the strength of that channel",
        },
    }
    (OUT / "intervention_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[141] wrote -> {OUT}")


if __name__ == "__main__":
    main()
