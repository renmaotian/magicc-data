#!/usr/bin/env python3
"""
WS3.3 / new-priority — test the hypothesis that MAGICC systematically over-calls
contamination and under-calls completeness on reduced genomes, and that this explains
Reviewer 2's observed disagreements for CAG-557, UMGS1491, Caccenecus, HGM10766 and
Faecimonas.

Inputs
  results/revision/real_data/reduced_genome/magicc_v5_predictions.tsv   (script 133)
  results/revision/real_data/reduced_genome/scg_census.tsv              (script 133, optional)

Analyses
  1. Per-genus deltas for the five reviewer genera, reported in BOTH sign conventions,
     because Reviewer 2's text is ambiguous about the direction of the contamination
     column.  Comparator = GTDB's published CheckM2 values.
  2. Delta as a function of genome size across the size-stratified MAG cohort, with the
     non-reduced control genera overlaid, to establish whether the effect is specific to
     the named genera or a smooth function of genome size.
  3. Mechanism: is the effect carried by genome size itself, or by the number of
     *detectable* single-copy core genes?  Spearman rho plus partial rho controlling for
     the other variable (rank-residual method).
  4. Colour-vision-deficiency-safe figure (no red/green discrimination).

Outputs
  reviewer_genera_deltas.tsv · size_bin_deltas.tsv · mechanism_correlations.tsv
  reduced_genome_effect.json · fig_reduced_genome_effect.png/.pdf
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTDIR = PROJECT_DIR / "results/revision/real_data/reduced_genome"
GTDB = [PROJECT_DIR / "data/gtdb/bac120_metadata.tsv.gz",
        PROJECT_DIR / "data/gtdb/ar53_metadata.tsv.gz"]
PRED = OUTDIR / "magicc_v5_predictions.tsv"
SCG = OUTDIR / "scg_census.tsv"

REVIEWER_GENERA = ["CAG-557", "UMGS1491", "Caccenecus", "HGM10766", "Faecimonas"]
SIZE_BIN_LABELS = ["<1Mb", "1-2Mb", "2-3Mb", "3-5Mb", ">5Mb"]
BOOT = 5000
SEED = 13400

# CVD-safe palette used elsewhere in this revision (no green; shape as redundant cue)
BLUE, SKY, GREY, VERM, YELLOW, BLACK = (
    "#0072B2", "#56B4E9", "#999999", "#D55E00", "#E69F00", "#000000")


# ------------------------------------------------------------------ statistics
def median_ci(x: np.ndarray, n_boot: int = BOOT, seed: int = SEED):
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def spearman(a: np.ndarray, b: np.ndarray):
    """Spearman rho and a two-sided p-value from the t approximation."""
    n = len(a)
    if n < 4:
        return float("nan"), float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if denom == 0:
        return float("nan"), float("nan")
    rho = float((ra * rb).sum() / denom)
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    return rho, _t_sf(abs(t), n - 2) * 2


def partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Spearman rho of a,b controlling for c (residuals of rank regression)."""
    n = len(a)
    if n < 5:
        return float("nan"), float("nan")
    ra, rb, rc = _rank(a), _rank(b), _rank(c)

    def resid(y, x):
        x = x - x.mean()
        y = y - y.mean()
        beta = float((x * y).sum() / max((x * x).sum(), 1e-12))
        return y - beta * x

    er, eb = resid(ra, rc), resid(rb, rc)
    denom = math.sqrt(float((er ** 2).sum()) * float((eb ** 2).sum()))
    if denom == 0:
        return float("nan"), float("nan")
    rho = float((er * eb).sum() / denom)
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 3) / (1 - rho ** 2))
    return rho, _t_sf(abs(t), n - 3) * 2


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    # average ties
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


def _t_sf(t: float, df: int) -> float:
    """Upper-tail P(T > t) for Student t, via the regularized incomplete beta."""
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
    for i in range(0, 300):
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


# ------------------------------------------------- lineage-typical genome size
def lineage_median_sizes() -> dict[str, float]:
    """Median *estimated complete* genome size per GTDB phylum and per family.

    Estimated complete size = genome_size / (checkm2_completeness/100), so that a
    partially recovered MAG does not drag the lineage median down.  Restricted to
    genomes with CheckM2 completeness >= 90 and contamination <= 5.
    """
    import gzip as _gz
    acc: dict[str, list[float]] = defaultdict(list)
    for path in GTDB:
        with _gz.open(path, "rt") as fh:
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
    med = {k: float(np.median(v)) for k, v in acc.items() if len(v) >= 20}
    print(f"[134] lineage medians: {sum(1 for k in med if k[0]=='p')} phyla, "
          f"{sum(1 for k in med if k[0]=='f')} families")
    return med


# ----------------------------------------------------------------------- data
def load() -> list[dict]:
    rows = list(csv.DictReader(PRED.open(), delimiter="\t"))
    scg = {}
    if SCG.exists():
        for r in csv.DictReader(SCG.open(), delimiter="\t"):
            if not r.get("error") and r.get("n_scg_detected"):
                scg[r["accession"]] = r
    for r in rows:
        r["mc"] = float(r["magicc_completeness"])
        r["mx"] = float(r["magicc_contamination"])
        r["cc"] = float(r["checkm2_completeness"])
        r["cx"] = float(r["checkm2_contamination"])
        r["d_comp"] = r["mc"] - r["cc"]          # MAGICC - CheckM2
        r["d_cont"] = r["mx"] - r["cx"]          # MAGICC - CheckM2
        r["size"] = int(r["genome_size"])
        s = scg.get(r["ncbi_accession"])
        r["n_scg"] = int(s["n_scg_detected"]) if s else None
        r["n_scg_profiles"] = int(s["n_scg_profiles"]) if s else None
        r["scg_frac"] = (r["n_scg"] / r["n_scg_profiles"]) if s else None
    med = lineage_median_sizes()
    for r in rows:
        t = r["gtdb_taxonomy"].split(";")
        r["est_full_size"] = r["size"] / max(r["cc"] / 100.0, 1e-6)
        pm = med.get("p:" + t[1]) if len(t) > 1 else None
        fm = med.get("f:" + t[4]) if len(t) > 4 else None
        r["phylum_median_size"] = pm
        r["family_median_size"] = fm
        r["log2_size_vs_phylum"] = (math.log2(r["est_full_size"] / pm) if pm else None)
        r["log2_size_vs_family"] = (math.log2(r["est_full_size"] / fm) if fm else None)
    n_p = sum(1 for r in rows if r["log2_size_vs_phylum"] is not None)
    print(f"[134] {len(rows)} predictions; {len(scg)} with SCG census; "
          f"{n_p} with a phylum-median reference size")
    return rows


def block(rows: list[dict], label: str) -> dict:
    dc = np.array([r["d_comp"] for r in rows])
    dx = np.array([r["d_cont"] for r in rows])
    mc, mcl, mch = median_ci(dc)
    mx, mxl, mxh = median_ci(dx)
    return {
        "label": label,
        "n": len(rows),
        "genome_size_median_bp": int(np.median([r["size"] for r in rows])) if rows else None,
        "magicc_completeness_median": round(float(np.median([r["mc"] for r in rows])), 2),
        "checkm2_completeness_median": round(float(np.median([r["cc"] for r in rows])), 2),
        "magicc_contamination_median": round(float(np.median([r["mx"] for r in rows])), 2),
        "checkm2_contamination_median": round(float(np.median([r["cx"] for r in rows])), 2),
        "delta_completeness_MAGICC_minus_CheckM2_median": round(mc, 2),
        "delta_completeness_ci95": [round(mcl, 2), round(mch, 2)],
        "delta_contamination_MAGICC_minus_CheckM2_median": round(mx, 2),
        "delta_contamination_ci95": [round(mxl, 2), round(mxh, 2)],
        "pct_magicc_cont_gt_5": round(100 * float(np.mean([r["mx"] > 5 for r in rows])), 1),
        "pct_checkm2_cont_gt_5": round(100 * float(np.mean([r["cx"] > 5 for r in rows])), 1),
        "pct_magicc_comp_ge_90": round(100 * float(np.mean([r["mc"] >= 90 for r in rows])), 1),
        "pct_checkm2_comp_ge_90": round(100 * float(np.mean([r["cc"] >= 90 for r in rows])), 1),
        "n_scg_detected_median": (
            int(np.median([r["n_scg"] for r in rows if r["n_scg"] is not None]))
            if any(r["n_scg"] is not None for r in rows) else None),
        "contig_count_median": (
            int(np.median([int(r["contig_count"]) for r in rows if r["contig_count"]]))
            if rows else None),
        "log2_size_vs_phylum_median": (
            round(float(np.median([r["log2_size_vs_phylum"] for r in rows
                                   if r["log2_size_vs_phylum"] is not None])), 3)
            if any(r["log2_size_vs_phylum"] is not None for r in rows) else None),
    }


def write_tsv(path: Path, blocks: list[dict]) -> None:
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


def figure(rows: list[dict], per_bin: list[dict], per_genus: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[134] matplotlib unavailable, skipping figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # (a) contamination delta vs genome size, all cohorts
    ax = axes[0]
    style = {"size_stratified": (GREY, "o", "size-stratified MAGs"),
             "control_genera": (BLUE, "s", "non-reduced control genera"),
             "reviewer_genera": (VERM, "^", "reviewer-named genera")}
    for coh, (col, mk, lab) in style.items():
        sub = [r for r in rows if r["cohort"] == coh]
        if not sub:
            continue
        ax.scatter([r["size"] / 1e6 for r in sub], [r["d_cont"] for r in sub],
                   s=13, c=col, marker=mk, alpha=0.55, linewidths=0, label=lab)
    ax.axhline(0, color=BLACK, lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("genome size (Mbp, log scale)")
    ax.set_ylabel("contamination:  MAGICC − CheckM2 (pp)")
    ax.set_title("(a) contamination disagreement vs genome size")
    ax.legend(fontsize=8, frameon=False)

    # (b) completeness delta vs genome size
    ax = axes[1]
    for coh, (col, mk, lab) in style.items():
        sub = [r for r in rows if r["cohort"] == coh]
        if not sub:
            continue
        ax.scatter([r["size"] / 1e6 for r in sub], [r["d_comp"] for r in sub],
                   s=13, c=col, marker=mk, alpha=0.55, linewidths=0, label=lab)
    ax.axhline(0, color=BLACK, lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("genome size (Mbp, log scale)")
    ax.set_ylabel("completeness:  MAGICC − CheckM2 (pp)")
    ax.set_title("(b) completeness disagreement vs genome size")

    # (c) size-bin medians with CI
    ax = axes[2]
    labs = [b["label"].split(":")[-1] for b in per_bin]
    xs = np.arange(len(labs))
    for key, col, mk, lab in (
            ("delta_contamination_MAGICC_minus_CheckM2_median", VERM, "^", "contamination"),
            ("delta_completeness_MAGICC_minus_CheckM2_median", BLUE, "o", "completeness")):
        ci = ("delta_contamination_ci95" if "contam" in key else "delta_completeness_ci95")
        med = [b[key] for b in per_bin]
        lo = [b[key] - b[ci][0] for b in per_bin]
        hi = [b[ci][1] - b[key] for b in per_bin]
        ax.errorbar(xs, med, yerr=[lo, hi], color=col, marker=mk, ms=7, lw=1.6,
                    capsize=3, label=lab)
    ax.axhline(0, color=BLACK, lw=0.9, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_xlabel("genome size bin (size-stratified MAG cohort)")
    ax.set_ylabel("median MAGICC − CheckM2 (pp), 95% CI")
    ax.set_title("(c) effect is a function of genome size")
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig_reduced_genome_effect.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[134] wrote {OUTDIR}/fig_reduced_genome_effect.png|.pdf")


def main() -> int:
    rows = load()

    # ---- 1. reviewer genera -------------------------------------------------
    per_genus = []
    for g in REVIEWER_GENERA:
        sub = [r for r in rows if r["genus"] == g]
        if sub:
            per_genus.append(block(sub, g))
    rg = [r for r in rows if r["cohort"] == "reviewer_genera"]
    if rg:
        per_genus.append(block(rg, "ALL five reviewer genera"))
    write_tsv(OUTDIR / "reviewer_genera_deltas.tsv", per_genus)

    # ---- 2. size bins + controls -------------------------------------------
    per_bin = []
    for lab in SIZE_BIN_LABELS:
        sub = [r for r in rows if r["cohort"] == "size_stratified" and r["stratum"] == lab]
        if sub:
            per_bin.append(block(sub, f"size_stratified:{lab}"))
    per_control = []
    for g in sorted({r["stratum"] for r in rows if r["cohort"] == "control_genera"}):
        sub = [r for r in rows if r["cohort"] == "control_genera" and r["stratum"] == g]
        if sub:
            per_control.append(block(sub, g))
    cg = [r for r in rows if r["cohort"] == "control_genera"]
    if cg:
        per_control.append(block(cg, "ALL non-reduced control genera"))
    write_tsv(OUTDIR / "size_bin_deltas.tsv", per_bin + per_control)

    # ---- 3. mechanism -------------------------------------------------------
    mech = []
    pools = {
        "all_cohorts": rows,
        "size_stratified_only": [r for r in rows if r["cohort"] == "size_stratified"],
        "excluding_reviewer_genera": [r for r in rows if r["cohort"] != "reviewer_genera"],
    }
    for name, sub in pools.items():
        if len(sub) < 10:
            continue
        size = np.array([r["size"] for r in sub], float)
        for tgt, key in (("delta_contamination", "d_cont"), ("delta_completeness", "d_comp")):
            y = np.array([r[key] for r in sub], float)
            rho, p = spearman(size, y)
            mech.append({"pool": name, "n": len(sub), "target": tgt,
                         "predictor": "genome_size", "rho": round(rho, 4),
                         "p": f"{p:.3g}", "controlling_for": ""})
        # fragmentation as an alternative explanation
        frag = [(r, int(r["contig_count"])) for r in sub if r["contig_count"]]
        if len(frag) >= 10:
            nc = np.array([c for _, c in frag], float)
            sz = np.array([r["size"] for r, _ in frag], float)
            for tgt, key in (("delta_contamination", "d_cont"),
                             ("delta_completeness", "d_comp")):
                y = np.array([r[key] for r, _ in frag], float)
                rho, p = spearman(nc, y)
                mech.append({"pool": name, "n": len(frag), "target": tgt,
                             "predictor": "contig_count", "rho": round(rho, 4),
                             "p": f"{p:.3g}", "controlling_for": ""})
                rho2, p2 = partial_spearman(nc, y, sz)
                mech.append({"pool": name, "n": len(frag), "target": tgt,
                             "predictor": "contig_count", "rho": round(rho2, 4),
                             "p": f"{p2:.3g}", "controlling_for": "genome_size"})
        # genome size RELATIVE TO THE LINEAGE (the sharper hypothesis)
        rel = [r for r in sub if r["log2_size_vs_phylum"] is not None]
        if len(rel) >= 10:
            lr = np.array([r["log2_size_vs_phylum"] for r in rel], float)
            sz = np.array([r["size"] for r in rel], float)
            for tgt, key in (("delta_contamination", "d_cont"),
                             ("delta_completeness", "d_comp")):
                y = np.array([r[key] for r in rel], float)
                rho, p = spearman(lr, y)
                mech.append({"pool": name, "n": len(rel), "target": tgt,
                             "predictor": "log2_size_vs_phylum", "rho": round(rho, 4),
                             "p": f"{p:.3g}", "controlling_for": ""})
                rho2, p2 = partial_spearman(lr, y, sz)
                mech.append({"pool": name, "n": len(rel), "target": tgt,
                             "predictor": "log2_size_vs_phylum", "rho": round(rho2, 4),
                             "p": f"{p2:.3g}", "controlling_for": "genome_size"})
                rho3, p3 = partial_spearman(sz, y, lr)
                mech.append({"pool": name, "n": len(rel), "target": tgt,
                             "predictor": "genome_size", "rho": round(rho3, 4),
                             "p": f"{p3:.3g}", "controlling_for": "log2_size_vs_phylum"})
        withscg = [r for r in sub if r["n_scg"] is not None]
        if len(withscg) >= 10:
            size2 = np.array([r["size"] for r in withscg], float)
            nscg = np.array([r["n_scg"] for r in withscg], float)
            rho_sz_scg, p_sz_scg = spearman(size2, nscg)
            mech.append({"pool": name, "n": len(withscg), "target": "n_scg_detected",
                         "predictor": "genome_size", "rho": round(rho_sz_scg, 4),
                         "p": f"{p_sz_scg:.3g}", "controlling_for": ""})
            for tgt, key in (("delta_contamination", "d_cont"),
                             ("delta_completeness", "d_comp")):
                y = np.array([r[key] for r in withscg], float)
                r1, p1 = spearman(nscg, y)
                mech.append({"pool": name, "n": len(withscg), "target": tgt,
                             "predictor": "n_scg_detected", "rho": round(r1, 4),
                             "p": f"{p1:.3g}", "controlling_for": ""})
                r2, p2 = partial_spearman(nscg, y, size2)
                mech.append({"pool": name, "n": len(withscg), "target": tgt,
                             "predictor": "n_scg_detected", "rho": round(r2, 4),
                             "p": f"{p2:.3g}", "controlling_for": "genome_size"})
                r3, p3 = partial_spearman(size2, y, nscg)
                mech.append({"pool": name, "n": len(withscg), "target": tgt,
                             "predictor": "genome_size", "rho": round(r3, 4),
                             "p": f"{p3:.3g}", "controlling_for": "n_scg_detected"})
    write_tsv(OUTDIR / "mechanism_correlations.tsv", mech)

    # ---- 4. leakage-free sensitivity ---------------------------------------
    clean = [r for r in rows if r["split"] not in ("train", "val")]
    sens = [block(rows, "ALL genomes"), block(clean, "excluding TRAIN/VAL genomes")]

    summary = {
        "script": "scripts/134_analyze_reduced_genome_effect.py",
        "hypothesis": ("MAGICC over-calls contamination and under-calls completeness on "
                       "reduced genomes; this explains the reviewer's disagreements"),
        "comparator": "GTDB published CheckM2",
        "n_genomes_scored": len(rows),
        "sign_convention": "all deltas are MAGICC minus CheckM2 (positive = MAGICC higher)",
        "reviewer_genera": per_genus,
        "size_bins": per_bin,
        "control_genera": per_control,
        "mechanism": mech,
        "leakage_sensitivity": sens,
    }
    (OUTDIR / "reduced_genome_effect.json").write_text(json.dumps(summary, indent=2))

    figure(rows, per_bin, per_genus)

    # ---- console report -----------------------------------------------------
    print("\n=== REVIEWER GENERA (deltas are MAGICC − CheckM2) ===")
    hdr = (f"{'genus':26}{'n':>5}{'size Mb':>9}{'MAGICC comp':>12}{'CkM2 comp':>10}"
           f"{'dComp':>8}{'MAGICC cont':>12}{'CkM2 cont':>10}{'dCont':>8}{'nSCG':>6}")
    print(hdr)
    for b in per_genus:
        print(f"{b['label'][:25]:26}{b['n']:>5}{b['genome_size_median_bp']/1e6:>9.2f}"
              f"{b['magicc_completeness_median']:>12.2f}{b['checkm2_completeness_median']:>10.2f}"
              f"{b['delta_completeness_MAGICC_minus_CheckM2_median']:>8.2f}"
              f"{b['magicc_contamination_median']:>12.2f}{b['checkm2_contamination_median']:>10.2f}"
              f"{b['delta_contamination_MAGICC_minus_CheckM2_median']:>8.2f}"
              f"{(b['n_scg_detected_median'] if b['n_scg_detected_median'] is not None else -1):>6}")
    print("\n=== SIZE BINS + CONTROLS ===")
    print(hdr)
    for b in per_bin + per_control:
        print(f"{b['label'][:25]:26}{b['n']:>5}{b['genome_size_median_bp']/1e6:>9.2f}"
              f"{b['magicc_completeness_median']:>12.2f}{b['checkm2_completeness_median']:>10.2f}"
              f"{b['delta_completeness_MAGICC_minus_CheckM2_median']:>8.2f}"
              f"{b['magicc_contamination_median']:>12.2f}{b['checkm2_contamination_median']:>10.2f}"
              f"{b['delta_contamination_MAGICC_minus_CheckM2_median']:>8.2f}"
              f"{(b['n_scg_detected_median'] if b['n_scg_detected_median'] is not None else -1):>6}")
    print("\n=== MECHANISM (Spearman) ===")
    for m in mech:
        ctrl = f" | ctrl {m['controlling_for']}" if m["controlling_for"] else ""
        print(f"  {m['pool']:28} {m['target']:20} ~ {m['predictor']:16}"
              f" rho={m['rho']:+.3f} p={m['p']:>10}{ctrl}  (n={m['n']})")
    print("\n=== LEAKAGE SENSITIVITY ===")
    for b in sens:
        print(f"  {b['label']:34} n={b['n']:>5} dComp="
              f"{b['delta_completeness_MAGICC_minus_CheckM2_median']:+6.2f} "
              f"dCont={b['delta_contamination_MAGICC_minus_CheckM2_median']:+6.2f}")
    print(f"\n[134] wrote {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
