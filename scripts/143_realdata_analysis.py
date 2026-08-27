#!/usr/bin/env python3
"""
WS3 Track A — accuracy analysis for the three real-data cohorts with genuine ground
truth (protocol §4.4e: these are now the only real-data validation routes).

  A1  meslier  Meslier et al. 2022 MOCK1, seven pre-computed assemblies
               (N50 4.0 kb -> 2.01 Mb) -> fragmentation gradient on real data
  A2  zymo     8 ZymoBIOMICS isolate draft/complete pairs
  A3  ncbi     NCBI Tier-1 same-BioSample draft/complete pairs

Binding conventions (protocol §4.4d, §5, task spec):
  * R^2 is ALWAYS the coefficient of determination 1 - SS_res/SS_tot, never squared
    Pearson. Squared Pearson is reported separately and labelled r2_pearson.
  * MIMAG-inspired: HQ >= 90 % completeness AND < 5 % contamination;
    MQ >= 50 % AND < 10 %.
  * Two-sided tests; effect sizes with 95 % CIs; observations clustered by reference
    genome / organism / species, since the same reference recurs across assemblies.
  * MAGICC has a stated 50 % completeness floor: the primary cohort is restricted to
    bins whose TRUE completeness >= 50 %, and the below-floor regime is reported
    separately rather than silently mixed in.
  * Denominators are stated explicitly in every emitted table (R1-M5).

Outputs land in results/revision/real_data/<track>/.

Usage:  python scripts/143_realdata_analysis.py --track meslier|zymo|ncbi|all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/path/to/magicc")
RD = ROOT / "results" / "revision" / "real_data"
SPLITDIR = ROOT / "data" / "splits"

_spec = importlib.util.spec_from_file_location(
    "mf", ROOT / "scripts" / "101_metrics_framework.py")
mf = importlib.util.module_from_spec(_spec)
sys.modules["mf"] = mf
_spec.loader.exec_module(mf)

TOOLS = ["magicc", "checkm2", "cocopye", "deepcheck"]
TOOL_LABEL = {"magicc": "MAGICC V5", "checkm2": "CheckM2 1.0.1",
              "cocopye": "CoCoPyE 0.5.0", "deepcheck": "DeepCheck"}
N_BOOT = 2000
SEED = 1430
# minimum SD of the ground truth (percentage points) for R^2 to be
# reported at all — see the note in metrics_row() and R1-m19
R2_MIN_TRUTH_SD = 5.0

DENOM = ("completeness = retained dominant bp / dominant reference FULL length x 100; "
         "contamination = contaminant bp / dominant reference FULL length x 100 "
         "(MAGICC convention; identical denominator for truth and for every tool's "
         "target quantity)")


# ------------------------------------------------------------------ statistics
def _mae(t, p):
    return float(np.mean(np.abs(p - t)))


def metrics_row(df, tool, tcomp="true_completeness", tcont="true_contamination",
                cluster=None, cohort="", note=""):
    cc, xc = f"{tool}_completeness", f"{tool}_contamination"
    if cc not in df.columns:
        return None
    d = df[df[cc].notna() & df[xc].notna()]
    if len(d) == 0:
        return None
    tc, pc = d[tcomp].values.astype(float), d[cc].values.astype(float)
    tx, px = d[tcont].values.astype(float), d[xc].values.astype(float)
    cl = d[cluster].values if cluster and cluster in d.columns else None
    bs = mf.Bootstrapper(clusters=cl, n_rows=len(d), n_iter=N_BOOT, seed=SEED)

    def ci(fn):
        r = bs.ci(fn)
        return round(r["ci_lo"], 4), round(r["ci_hi"], 4)

    cmae_lo, cmae_hi = ci(lambda i: _mae(tc[i], pc[i]))
    cbias_lo, cbias_hi = ci(lambda i: float(np.mean(pc[i] - tc[i])))
    xmae_lo, xmae_hi = ci(lambda i: _mae(tx[i], px[i]))
    xbias_lo, xbias_hi = ci(lambda i: float(np.mean(px[i] - tx[i])))
    return {
        "cohort": cohort, "tool": TOOL_LABEL[tool], "tool_key": tool,
        "n": len(d), "n_clusters": int(len(np.unique(cl))) if cl is not None else len(d),
        "cluster_unit": cluster or "sample",
        "comp_mae": round(_mae(tc, pc), 4),
        "comp_mae_ci_lo": cmae_lo, "comp_mae_ci_hi": cmae_hi,
        "comp_bias": round(float(np.mean(pc - tc)), 4),
        "comp_bias_ci_lo": cbias_lo, "comp_bias_ci_hi": cbias_hi,
        "comp_rmse": round(mf.rmse(tc, pc), 4),
        "comp_r2": round(mf.r2_coefficient_of_determination(tc, pc), 4),
        "comp_r2_pearson": round(mf.r2_pearson_squared(tc, pc), 4),
        "cont_mae": round(_mae(tx, px), 4),
        "cont_mae_ci_lo": xmae_lo, "cont_mae_ci_hi": xmae_hi,
        "cont_bias": round(float(np.mean(px - tx)), 4),
        "cont_bias_ci_lo": xbias_lo, "cont_bias_ci_hi": xbias_hi,
        "cont_rmse": round(mf.rmse(tx, px), 4),
        "cont_r2": round(mf.r2_coefficient_of_determination(tx, px), 4),
        "cont_r2_pearson": round(mf.r2_pearson_squared(tx, px), 4),
        "true_comp_mean": round(float(tc.mean()), 3),
        "true_cont_mean": round(float(tx.mean()), 4),
        "true_comp_sd": round(float(tc.std(ddof=1)), 3) if len(tc) > 1 else np.nan,
        "true_cont_sd": round(float(tx.std(ddof=1)), 4) if len(tx) > 1 else np.nan,
        # R1-m19: R^2 must not be reported where the truth has (near-)zero variance.
        # R^2 = 1 - MSE/Var(truth), so when the truth barely varies the statistic
        # measures the smallness of the denominator rather than predictive skill and
        # goes sharply negative for any tool with even a small bias. Two real-data
        # axes are in exactly that regime: contamination in the mock cohorts (bins are
        # near-pure by construction) and completeness in the isolate draft cohorts
        # (every draft is 87-100 % complete). Threshold: SD(truth) >= 5 pp, stated
        # explicitly so the rule is auditable rather than ad hoc.
        "r2_min_truth_sd_pp": R2_MIN_TRUTH_SD,
        "comp_r2_interpretable": bool(len(tc) > 2
                                      and tc.std(ddof=1) >= R2_MIN_TRUTH_SD),
        "cont_r2_interpretable": bool(len(tx) > 2
                                      and tx.std(ddof=1) >= R2_MIN_TRUTH_SD),
        "denominator": DENOM, "note": note,
    }


def paired_test(df, a, b, metric, tcol, cluster=None, cohort=""):
    """Two-sided paired comparison of |error| between two tools on the same genomes."""
    ca, cb = f"{a}_{metric}", f"{b}_{metric}"
    if ca not in df.columns or cb not in df.columns:
        return None
    d = df[df[ca].notna() & df[cb].notna()]
    if len(d) < 5:
        return None
    t = d[tcol].values.astype(float)
    ea = np.abs(d[ca].values.astype(float) - t)
    eb = np.abs(d[cb].values.astype(float) - t)
    diff = ea - eb                      # negative -> tool a closer to truth
    try:
        w = stats.wilcoxon(ea, eb, alternative="two-sided",
                           zero_method="wilcox")
        p = float(w.pvalue)
    except ValueError:
        p = float("nan")
    cl = d[cluster].values if cluster and cluster in d.columns else None
    bs = mf.Bootstrapper(clusters=cl, n_rows=len(d), n_iter=N_BOOT, seed=SEED)
    r = bs.ci(lambda i: float(np.mean(diff[i])))
    hl = mf.hodges_lehmann_paired(diff)
    return {"cohort": cohort, "metric": metric,
            "tool_a": TOOL_LABEL[a], "tool_b": TOOL_LABEL[b], "n": len(d),
            "mae_a": round(float(ea.mean()), 4), "mae_b": round(float(eb.mean()), 4),
            "mean_diff_a_minus_b": round(float(diff.mean()), 4),
            "diff_ci_lo": round(r["ci_lo"], 4), "diff_ci_hi": round(r["ci_hi"], 4),
            "hodges_lehmann_shift": round(float(hl), 4),
            "rank_biserial": round(float(mf.rank_biserial_paired(diff)), 4),
            "wilcoxon_p_two_sided": p,
            "winner": (TOOL_LABEL[a] if diff.mean() < 0 else TOOL_LABEL[b]),
            "denominator": DENOM}


def mimag_row(df, tool, tcomp, tcont, cohort=""):
    cc, xc = f"{tool}_completeness", f"{tool}_contamination"
    if cc not in df.columns:
        return None
    d = df[df[cc].notna()]
    if len(d) == 0:
        return None

    def cls(c, x):
        c, x = np.asarray(c, float), np.asarray(x, float)
        out = np.full(len(c), "LQ", dtype=object)
        out[(c >= 50) & (x < 10)] = "MQ"
        out[(c >= 90) & (x < 5)] = "HQ"
        return out

    ta = cls(d[tcomp], d[tcont])
    pa = cls(d[cc], d[xc])
    tc = d[tcont].values.astype(float)
    px = d[xc].values.astype(float)
    pc = d[cc].values.astype(float)
    truly_clean5 = tc < 5
    truly_clean10 = tc < 10
    return {
        "cohort": cohort, "tool": TOOL_LABEL[tool], "tool_key": tool, "n": len(d),
        "true_HQ": int((ta == "HQ").sum()), "pred_HQ": int((pa == "HQ").sum()),
        "true_MQ": int((ta == "MQ").sum()), "pred_MQ": int((pa == "MQ").sum()),
        "true_LQ": int((ta == "LQ").sum()), "pred_LQ": int((pa == "LQ").sum()),
        "class_agreement_pct": round(100.0 * float((ta == pa).mean()), 2),
        # contamination-threshold behaviour on genomes that are truly clean
        "n_truly_below_5pct_cont": int(truly_clean5.sum()),
        "false_fail_5pct_n": int((truly_clean5 & (px >= 5)).sum()),
        "false_fail_5pct_rate": (round(100.0 * float((px[truly_clean5] >= 5).mean()), 2)
                                 if truly_clean5.sum() else np.nan),
        "n_truly_below_10pct_cont": int(truly_clean10.sum()),
        "false_fail_10pct_rate": (round(100.0 * float((px[truly_clean10] >= 10).mean()), 2)
                                  if truly_clean10.sum() else np.nan),
        # completeness-threshold behaviour
        "true_comp_ge90_n": int((d[tcomp] >= 90).sum()),
        "pred_comp_ge90_n": int((pc >= 90).sum()),
        "comp90_recall_pct": (round(100.0 * float((pc[(d[tcomp] >= 90).values] >= 90).mean()), 2)
                              if (d[tcomp] >= 90).sum() else np.nan),
        "denominator": DENOM,
    }


def leakage_map():
    m = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(SPLITDIR / f"{split}_genomes.tsv", sep="\t",
                         usecols=["ncbi_accession", "gcf_accession"], dtype=str)
        for col in ("ncbi_accession", "gcf_accession"):
            for a in df[col].dropna().unique():
                a = str(a).strip()
                if a and a.lower() != "nan":
                    m[a] = split
                    m[a.split(".")[0]] = split
    return m


def train_species():
    sp = set()
    for split in ("train", "val"):
        df = pd.read_csv(SPLITDIR / f"{split}_genomes.tsv", sep="\t",
                         usecols=["gtdb_taxonomy"], dtype=str)
        for t in df["gtdb_taxonomy"].dropna():
            for f in t.split(";"):
                if f.startswith("s__"):
                    s = f[3:].strip()
                    if s:
                        sp.add(s)
    return sp


def emit(rows, path, sort=None):
    df = pd.DataFrame([r for r in rows if r])
    if sort:
        df = df.sort_values(sort)
    df.to_csv(path, sep="\t", index=False)
    print(f"  wrote {path} ({len(df)} rows)")
    return df


def orf_compromised_assemblies(d, p):
    """Assemblies whose ORFs are broken by indels, diagnosed from CheckM2's own output.

    The seven MOCK1 assemblies differ along TWO axes, not one: contig N50
    (fragmentation) and per-base accuracy (platform error mode). Indel-dense assemblies
    frameshift ORFs, which is invisible to a nucleotide k-mer method but fatal to a
    protein-based one. Coding density and mean gene length come from CheckM2 itself, so
    the diagnosis does not depend on this pipeline. Short mean gene length alone is NOT
    sufficient — highly fragmented short-read assemblies also truncate genes at contig
    ends — hence the joint coding-density criterion.
    """
    qc_path = d / "checkm2_output" / "quality_report.tsv"
    if not qc_path.exists():
        return set()
    q = pd.read_csv(qc_path, sep="\t")
    q["assembly"] = q["Name"].astype(str).str.split("__").str[0]
    aqc = q.groupby("assembly").agg(
        n_bins=("Name", "size"),
        median_coding_density=("Coding_Density", "median"),
        median_gene_length=("Average_Gene_Length", "median"),
        median_n_cds=("Total_Coding_Sequences", "median")).reset_index()
    aqc["median_bin_n50"] = aqc["assembly"].map(
        p.groupby("assembly")["bin_n50"].median())
    aqc["orf_integrity_compromised"] = ((aqc["median_coding_density"] < 0.80)
                                        & (aqc["median_gene_length"] < 160))
    pbp = d / "per_base_error_profile.tsv"
    if pbp.exists():
        aqc = aqc.merge(pd.read_csv(pbp, sep="\t")[
            ["assembly", "indel_bp_per_kb", "substitution_per_kb",
             "mean_bp_between_indels"]], on="assembly", how="left")
    aqc.to_csv(d / "assembly_sequence_qc.tsv", sep="\t", index=False)
    fs = set(aqc.loc[aqc.orf_integrity_compromised, "assembly"])
    print(f"  assemblies with compromised ORF integrity "
          f"(indel/frameshift signature): {sorted(fs) or 'none'}")
    return fs


# ---------------------------------------------------------------------- A1
def analyse_meslier():
    d = RD / "meslier"
    p = pd.read_csv(d / "predictions.tsv", sep="\t")
    p = p[p["bin_fasta"].notna() & (p["bin_fasta"] != "")].copy()
    p["assembly_n50"] = p.groupby("assembly")["bin_n50"].transform("median")
    print(f"A1 meslier: {len(p)} bins, {p['accession'].nunique()} organisms, "
          f"{p['assembly'].nunique()} assemblies")

    # ORF-integrity flag is needed before the cohorts are defined, so that the pooled
    # numbers can be reported both with and without the assembly whose ORFs are broken
    # by indels. Pooling them silently would let one platform's error mode masquerade
    # as a general result.
    frameshift_asm = orf_compromised_assemblies(d, p)
    intact = ~p["assembly"].isin(frameshift_asm)
    fs = sorted(frameshift_asm)

    cohorts = {
        "primary_leakage_free_comp>=50":
            p[(p.leakage_free) & (p.true_completeness >= 50)],
        "primary_leakage_free_comp>=50_ORF_intact":
            p[(p.leakage_free) & (p.true_completeness >= 50) & intact],
        "all_MOCK1_comp>=50": p[p.true_completeness >= 50],
        "all_MOCK1_comp>=50_ORF_intact": p[(p.true_completeness >= 50) & intact],
        "ORF_compromised_assembly_only": p[(p.true_completeness >= 50) & (~intact)],
        "leakage_free_all_bins": p[p.leakage_free],
        "all_MOCK1_all_bins": p,
        "below_50pct_floor": p[p.true_completeness < 50],
    }
    notes = {
        "primary_leakage_free_comp>=50":
            "PRIMARY. Reference organism not in MAGICC TRAIN or VAL, and true "
            "completeness >= 50 % (MAGICC's stated operating floor). All 7 assemblies.",
        "primary_leakage_free_comp>=50_ORF_intact":
            "PRIMARY, restricted to assemblies with intact ORFs. This isolates the "
            "fragmentation effect from the per-base-accuracy effect: without it, one "
            f"indel-dense assembly ({fs}) dominates the pooled competitor error.",
        "all_MOCK1_comp>=50": "All MOCK1 organisms (includes TRAIN/VAL references), "
                              "true completeness >= 50 %. All 7 assemblies.",
        "all_MOCK1_comp>=50_ORF_intact":
            "All MOCK1 organisms, ORF-intact assemblies only.",
        "ORF_compromised_assembly_only":
            f"Only the assembly/assemblies whose ORFs are broken by indels ({fs}). "
            "Reported separately because this is a per-base-accuracy result, not a "
            "fragmentation result.",
        "leakage_free_all_bins": "Leakage-free organisms, no completeness floor.",
        "all_MOCK1_all_bins": "Everything, no completeness floor.",
        "below_50pct_floor": "Below MAGICC's stated 50 % completeness floor — reported "
                             "for transparency; MAGICC does not claim this regime.",
    }
    rows, mrows, prows = [], [], []
    for name, sub in cohorts.items():
        for t in TOOLS:
            rows.append(metrics_row(sub, t, cluster="accession", cohort=name,
                                    note=notes[name]))
            mrows.append(mimag_row(sub, t, "true_completeness", "true_contamination",
                                   cohort=name))
        for t in [x for x in TOOLS if x != "magicc"]:
            for metric, tcol in (("completeness", "true_completeness"),
                                 ("contamination", "true_contamination")):
                prows.append(paired_test(sub, "magicc", t, metric, tcol,
                                         cluster="accession", cohort=name))
    emit(rows, d / "metrics_by_cohort.tsv")
    emit(mrows, d / "mimag_by_cohort.tsv")
    emit(prows, d / "paired_tests.tsv")

    # ---- fragmentation gradient on the balanced panel
    cnt = p.groupby("accession")["assembly"].nunique()
    panel = set(cnt[cnt == p["assembly"].nunique()].index)
    ok50 = p[p.true_completeness >= 50].groupby("accession")["assembly"].nunique()
    panel50 = set(ok50[ok50 == p["assembly"].nunique()].index)
    print(f"  balanced panel: {len(panel)} organisms in all assemblies; "
          f"{len(panel50)} of them >= 50 % complete in every assembly")
    asm_n50 = p.groupby("assembly")["bin_n50"].median().sort_values(ascending=False)
    grows = []
    for pname, pset in (("balanced_panel_all", panel),
                        ("balanced_panel_comp>=50", panel50),
                        ("balanced_panel_comp>=50_leakage_free",
                         panel50 & set(p[p.leakage_free]["accession"]))):
        for asm in asm_n50.index:
            sub = p[(p.assembly == asm) & (p.accession.isin(pset))]
            if len(sub) < 3:
                continue
            for t in TOOLS:
                r = metrics_row(sub, t, cluster="accession",
                                cohort=f"{pname}|{asm}",
                                note="fragmentation gradient")
                if r:
                    r.update({"panel": pname, "assembly": asm,
                              "n_panel_organisms": len(pset),
                              "assembly_median_bin_n50": int(asm_n50[asm]),
                              "assembly_contigs": int(sub["n_contigs"].sum()),
                              "orf_integrity_compromised": asm in frameshift_asm})
                    grows.append(r)
    g = emit(grows, d / "fragmentation_gradient.tsv")


    # ---- error-vs-fragmentation slope per tool (log10 N50).
    # Computed on the ORF-intact assemblies only, because an assembly whose ORFs are
    # broken by indels is not a point on a fragmentation axis: it is a point on a
    # per-base-accuracy axis, and mixing the two would misattribute the failure.
    srows = []
    for pname in g["panel"].unique():
        gg = g[(g.panel == pname) & (~g.orf_integrity_compromised)]
        for t in TOOLS:
            s = gg[gg.tool_key == t].sort_values("assembly_median_bin_n50",
                                                 ascending=False)
            if len(s) < 4:
                continue
            x = np.log10(s["assembly_median_bin_n50"].values.astype(float))
            for m in ("comp_mae", "cont_mae"):
                y = s[m].values.astype(float)
                sl, ic, r, pv, se = stats.linregress(x, y)
                srows.append({"panel": pname, "tool": TOOL_LABEL[t], "metric": m,
                              "slope_per_log10_N50": round(float(sl), 4),
                              "slope_se": round(float(se), 4),
                              "p_two_sided": float(pv),
                              "pearson_r": round(float(r), 4),
                              "value_at_best_N50": round(float(y[0]), 4),
                              "value_at_worst_N50": round(float(y[-1]), 4),
                              "degradation": round(float(y[-1] - y[0]), 4),
                              "assemblies_used": ",".join(s["assembly"]),
                              "note": "ORF-intact assemblies only; assemblies whose "
                                      "coding density and mean gene length show the "
                                      "indel/frameshift signature are excluded and "
                                      "reported separately"})
    emit(srows, d / "fragmentation_slopes.tsv")

    # ---- per-base-accuracy exhibit: ORF-compromised vs ORF-intact assemblies
    if frameshift_asm:
        erows = []
        base = p[(p.true_completeness >= 50) & (p.accession.isin(panel50))]
        for label, sub in (("ORF_compromised_assemblies",
                            base[base.assembly.isin(frameshift_asm)]),
                           ("ORF_intact_assemblies",
                            base[~base.assembly.isin(frameshift_asm)])):
            for t in TOOLS:
                r = metrics_row(sub, t, cluster="accession", cohort=label,
                                note="Same balanced panel of organisms; the only "
                                     "difference is per-base assembly accuracy. "
                                     f"ORF-compromised assemblies: "
                                     f"{sorted(frameshift_asm)}.")
                if r:
                    r["assemblies"] = ",".join(sorted(sub["assembly"].unique()))
                    erows.append(r)
        emit(erows, d / "per_base_accuracy_effect.tsv")

    # ---- per-ORGANISM contamination false positives.
    # Whether a tool's false positives are spread over bins or concentrated in
    # particular organisms is the difference between a noise floor and a systematic
    # blind spot, and only the latter is a usable limitation statement.
    clean = p[(p.true_completeness >= 50) & (p.true_contamination < 1.0)]
    orows = []
    for acc, sub in clean.groupby("accession"):
        row = {"accession": acc, "organism": sub["organism"].iloc[0],
               "gtdb_phylum": sub["gtdb_phylum"].iloc[0],
               "kingdom": sub["kingdom"].iloc[0],
               "leakage_free": bool(sub["leakage_free"].iloc[0]),
               "n_bins": len(sub),
               "true_cont_max": round(float(sub["true_contamination"].max()), 4)}
        for t in TOOLS:
            c = f"{t}_contamination"
            if c in sub.columns and sub[c].notna().any():
                row[f"{t}_cont_median"] = round(float(sub[c].median()), 3)
                row[f"{t}_n_bins_ge5pct"] = int((sub[c] >= 5).sum())
        orows.append(row)
    od = pd.DataFrame(orows).sort_values("magicc_cont_median", ascending=False)
    od.to_csv(d / "per_organism_contamination_fp.tsv", sep="\t", index=False)
    print(f"  wrote {d/'per_organism_contamination_fp.tsv'} "
          f"({len(od)} organisms, truly-clean bins only)")

    # ---- per-bin long table for Source Data
    keep = ["assembly", "accession", "organism", "kingdom", "gtdb_phylum", "split",
            "leakage_free", "mock1_pct", "n_contigs", "bin_bp", "bin_n50", "ref_len",
            "true_completeness", "true_contamination", "true_contamination_upper"]
    keep += [c for c in p.columns if any(c.startswith(t + "_") for t in TOOLS)]
    p[keep].to_csv(d / "per_bin_results.tsv", sep="\t", index=False)
    print(f"  wrote {d/'per_bin_results.tsv'}")
    return p


# ---------------------------------------------------------------------- A2
def analyse_zymo():
    d = RD / "zymo"
    p = pd.read_csv(d / "predictions.tsv", sep="\t")
    print(f"A2 zymo: {len(p)} isolate pairs")
    rows, mrows, prows = [], [], []
    for t in TOOLS:
        rows.append(metrics_row(p, t, cluster="code", cohort="zymo_8_isolates",
                                note="8 ZymoBIOMICS bacterial isolate SPAdes drafts vs "
                                     "their complete references (2 yeasts excluded as "
                                     "eukaryotes). Near-complete, near-clean: a "
                                     "precision / false-positive test."))
        mrows.append(mimag_row(p, t, "true_completeness", "true_contamination",
                               cohort="zymo_8_isolates"))
    for t in [x for x in TOOLS if x != "magicc"]:
        for metric, tcol in (("completeness", "true_completeness"),
                             ("contamination", "true_contamination")):
            prows.append(paired_test(p, "magicc", t, metric, tcol, cluster="code",
                                     cohort="zymo_8_isolates"))
    emit(rows, d / "metrics_by_cohort.tsv")
    emit(mrows, d / "mimag_by_cohort.tsv")
    emit(prows, d / "paired_tests.tsv")
    p.to_csv(d / "per_pair_results.tsv", sep="\t", index=False)
    print(f"  wrote {d/'per_pair_results.tsv'}")
    return p


# ---------------------------------------------------------------------- A3
def analyse_ncbi():
    d = RD / "ncbi_pairs"
    p = pd.read_csv(d / "predictions.tsv", sep="\t")
    p = p[p.status == "ok"].copy()
    lm = leakage_map()
    ts = train_species()
    p["draft_split"] = p["draft_accession"].map(
        lambda a: lm.get(a, lm.get(str(a).split(".")[0], "none")))
    p["complete_split"] = p["complete_accession"].map(
        lambda a: lm.get(a, lm.get(str(a).split(".")[0], "none")))
    p["binomial"] = p["organism"].astype(str).str.split().str[:2].str.join(" ")
    p["species_in_train"] = p["binomial"].isin(ts)
    p["leakage_free"] = (~p["draft_split"].isin(["train", "val"])
                         & ~p["complete_split"].isin(["train", "val"]))
    p["true_contamination"] = p["true_contamination_upper"]
    print(f"A3 ncbi: {len(p)} pairs, {p['species_taxid'].nunique()} species; "
          f"draft in TRAIN/VAL: {int((p.draft_split.isin(['train','val'])).sum())}, "
          f"complete in TRAIN/VAL: "
          f"{int((p.complete_split.isin(['train','val'])).sum())}; "
          f"leakage-free pairs: {int(p.leakage_free.sum())}; "
          f"species absent from training: {int((~p.species_in_train).sum())}")

    cohorts = {
        "primary_leakage_free": p[p.leakage_free],
        "all_tier1_pairs": p,
        "species_absent_from_training": p[~p.species_in_train],
        "flagged_contaminated_by_NCBI":
            p[p["draft_excluded_from_refseq"].fillna("").str.contains("contaminated")],
    }
    notes = {
        "primary_leakage_free": "PRIMARY. Neither draft nor complete accession appears "
                                "in MAGICC's TRAIN or VAL split (GCA<->GCF cross-mapped).",
        "all_tier1_pairs": "All Tier-1 same-BioSample prokaryotic pairs, size ratio "
                           "0.80-1.20.",
        "species_absent_from_training": "Binomial not present as a GTDB species name in "
                                        "TRAIN or VAL.",
        "flagged_contaminated_by_NCBI": "Drafts carrying NCBI's own `contaminated` "
                                        "exclusion flag — independent, non-circular "
                                        "positive control (binary label).",
    }
    rows, mrows, prows = [], [], []
    for name, sub in cohorts.items():
        if len(sub) < 5:
            continue
        for t in TOOLS:
            rows.append(metrics_row(sub, t, cluster="species_taxid", cohort=name,
                                    note=notes[name]))
            mrows.append(mimag_row(sub, t, "true_completeness", "true_contamination",
                                   cohort=name))
        for t in [x for x in TOOLS if x != "magicc"]:
            for metric, tcol in (("completeness", "true_completeness"),
                                 ("contamination", "true_contamination")):
                prows.append(paired_test(sub, "magicc", t, metric, tcol,
                                         cluster="species_taxid", cohort=name))
    emit(rows, d / "metrics_by_cohort.tsv")
    emit(mrows, d / "mimag_by_cohort.tsv")
    emit(prows, d / "paired_tests.tsv")

    # ---- stratify by draft fragmentation (contig count quartiles)
    q = p["draft_contigs"].quantile([0, .25, .5, .75, 1.0]).values
    p["frag_bin"] = pd.cut(p["draft_contigs"], bins=np.unique(q),
                           include_lowest=True, duplicates="drop")
    frows = []
    for b, sub in p.groupby("frag_bin", observed=True):
        for t in TOOLS:
            r = metrics_row(sub, t, cluster="species_taxid",
                            cohort=f"draft_contigs {b}",
                            note="stratified by draft fragmentation")
            if r:
                r["median_draft_contigs"] = int(sub["draft_contigs"].median())
                r["median_draft_n50"] = int(sub["draft_n50"].median())
                frows.append(r)
    emit(frows, d / "fragmentation_strata.tsv")

    # ---- NCBI contaminated-flag detection (binary, NOT quantification)
    det = []
    flagged = p["draft_excluded_from_refseq"].fillna("").str.contains("contaminated")
    if flagged.sum() >= 3:
        for t in TOOLS:
            xc = f"{t}_contamination"
            if xc not in p.columns or p[xc].isna().all():
                continue
            for tau in (5.0, 10.0):
                tp = int(((p[xc] >= tau) & flagged).sum())
                fp = int(((p[xc] >= tau) & ~flagged).sum())
                fn = int(((p[xc] < tau) & flagged).sum())
                tn = int(((p[xc] < tau) & ~flagged).sum())
                det.append({
                    "tool": TOOL_LABEL[t], "threshold_pct": tau,
                    "n_flagged": int(flagged.sum()), "n_unflagged": int((~flagged).sum()),
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "recall_pct": round(100 * tp / max(1, tp + fn), 2),
                    "precision_pct": round(100 * tp / max(1, tp + fp), 2),
                    "specificity_pct": round(100 * tn / max(1, tn + fp), 2),
                    "note": "BINARY DETECTION ONLY. NCBI's Foreign Contamination Screen "
                            "uses a different operational definition from the bp-based "
                            "contamination quantified elsewhere here, so this table must "
                            "not be read as a quantification benchmark."})
        emit(det, d / "ncbi_contaminated_flag_detection.tsv")

    p.to_csv(d / "per_pair_results.tsv", sep="\t", index=False)
    print(f"  wrote {d/'per_pair_results.tsv'}")
    return p


# ------------------------------------------------- A3 optional: NCBI flag cohort
def analyse_flagged():
    """NCBI `contaminated` exclusion flag as a BINARY DETECTION positive control.

    NCBI's Foreign Contamination Screen uses a different operational definition from the
    bp-based contamination quantified in Tracks A1-A3, so this can only ever be a
    detection benchmark: precision / recall / specificity at the MIMAG thresholds.
    It is explicitly NOT a quantification benchmark, and no MAE is computed.
    """
    d = RD / "ncbi_flagged"
    coh = d / "cohort.tsv"
    if not coh.exists():
        print("A3-optional: cohort absent, skipping")
        return None
    c = pd.read_csv(coh, sep="\t", dtype=str)
    mg = d / "magicc_v5_predictions.tsv"
    if not mg.exists():
        print("A3-optional: MAGICC predictions absent, skipping")
        return None
    p = c.merge(pd.read_csv(mg, sep="\t").rename(columns={"genome_id": "accession"}),
                on="accession", how="left")
    for name, fn in (("checkm2", parse_checkm2_flag), ("cocopye", parse_cocopye_flag)):
        t = fn(d)
        if t is not None:
            p = p.merge(t, on="accession", how="left")
    p["contigs"] = pd.to_numeric(p["contigs"], errors="coerce")
    p.to_csv(d / "per_genome_results.tsv", sep="\t", index=False)
    flagged = p["label"] == "flagged_contaminated"
    print(f"A3-optional flagged cohort: {int(flagged.sum())} flagged + "
          f"{int((~flagged).sum())} same-species matched controls, "
          f"{p['species_taxid'].nunique()} species")

    rows = []
    for t in TOOLS:
        xc = f"{t}_contamination"
        if xc not in p.columns or p[xc].isna().all():
            continue
        sub = p[p[xc].notna()]
        f = sub["label"] == "flagged_contaminated"
        try:
            u = stats.mannwhitneyu(sub.loc[f, xc], sub.loc[~f, xc],
                                   alternative="two-sided")
            pv, delta = float(u.pvalue), mf.cliffs_delta(sub.loc[f, xc].values,
                                                         sub.loc[~f, xc].values)
        except ValueError:
            pv, delta = float("nan"), float("nan")
        for tau in (5.0, 10.0):
            tp = int(((sub[xc] >= tau) & f).sum())
            fp = int(((sub[xc] >= tau) & ~f).sum())
            fn_ = int(((sub[xc] < tau) & f).sum())
            tn = int(((sub[xc] < tau) & ~f).sum())
            rows.append({
                "tool": TOOL_LABEL[t], "threshold_pct": tau, "n": len(sub),
                "n_flagged": int(f.sum()), "n_control": int((~f).sum()),
                "median_cont_flagged": round(float(sub.loc[f, xc].median()), 3),
                "median_cont_control": round(float(sub.loc[~f, xc].median()), 3),
                "mannwhitney_p_two_sided": pv, "cliffs_delta": round(delta, 4),
                "TP": tp, "FP": fp, "FN": fn_, "TN": tn,
                "recall_pct": round(100 * tp / max(1, tp + fn_), 2),
                "precision_pct": round(100 * tp / max(1, tp + fp), 2),
                "specificity_pct": round(100 * tn / max(1, tn + fp), 2),
                "balanced_accuracy_pct": round(
                    50 * (tp / max(1, tp + fn_) + tn / max(1, tn + fp)), 2),
                "note": "BINARY DETECTION ONLY — NCBI's Foreign Contamination Screen "
                        "uses a different operational definition from the bp-based "
                        "contamination quantified in Tracks A1-A3; no MAE is defined."})
    emit(rows, d / "detection_metrics.tsv")
    return p


def parse_checkm2_flag(d):
    qr = d / "checkm2_output" / "quality_report.tsv"
    if not qr.is_file():
        return None
    q = pd.read_csv(qr, sep="\t")
    return pd.DataFrame({"accession": q["Name"].astype(str),
                         "checkm2_completeness": q["Completeness"].astype(float),
                         "checkm2_contamination": q["Contamination"].astype(float)})


def parse_cocopye_flag(d):
    raw = d / "cocopye_raw_output.csv"
    if not raw.is_file():
        return None
    c = pd.read_csv(raw)
    if "3_completeness" not in c.columns:
        return None
    return pd.DataFrame({
        "accession": c["bin"].astype(str),
        "cocopye_completeness": (c["3_completeness"].fillna(c.get("2_completeness"))
                                 * 100.0).astype(float),
        "cocopye_contamination": (c["3_contamination"].fillna(c.get("2_contamination"))
                                  * 100.0).astype(float)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="all",
                    choices=["meslier", "zymo", "ncbi", "flagged", "all"])
    args = ap.parse_args()
    print("=" * 78)
    print("WS3 Track A — real-data accuracy analysis")
    print("R2 = coefficient of determination (1 - SS_res/SS_tot) throughout")
    print("=" * 78)
    done = {}
    if args.track in ("meslier", "all"):
        done["meslier"] = analyse_meslier()
    if args.track in ("zymo", "all"):
        done["zymo"] = analyse_zymo()
    if args.track in ("ncbi", "all"):
        done["ncbi"] = analyse_ncbi()
    if args.track in ("flagged", "all"):
        r = analyse_flagged()
        if r is not None:
            done["ncbi_flagged"] = r
    with open(RD / "analysis_run.json", "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                   "tracks": {k: int(len(v)) for k, v in done.items()},
                   "n_bootstrap": N_BOOT, "seed": SEED,
                   "r2_convention": "coefficient of determination (1 - SS_res/SS_tot)",
                   "mimag": "HQ >=90% comp AND <5% cont; MQ >=50% AND <10% "
                            "(MIMAG-inspired; rRNA/tRNA criteria not evaluable)",
                   "denominator": DENOM}, f, indent=2)
    print("\nDONE")


if __name__ == "__main__":
    main()
