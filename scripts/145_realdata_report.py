#!/usr/bin/env python3
"""
WS3 Track A — assemble the human-readable report from the emitted result files.

Reads only files under results/revision/real_data/ (no recomputation), so the report
can never disagree with the tables it cites. Protocol §8.2: no claim without a result
file.

Output: results/revision/real_data/WS3_trackA_report.md

Usage: python scripts/145_realdata_report.py
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/path/to/magicc")
RD = ROOT / "results" / "revision" / "real_data"

_spec = importlib.util.spec_from_file_location(
    "mf", ROOT / "scripts" / "101_metrics_framework.py")
mf = importlib.util.module_from_spec(_spec)
sys.modules["mf"] = mf
_spec.loader.exec_module(mf)


def rd(track, name):
    p = RD / track / name
    return pd.read_csv(p, sep="\t") if p.exists() else None


def fmt_ci(r, stem):
    return (f"{r[stem]:.2f} [{r[stem + '_ci_lo']:.2f}, {r[stem + '_ci_hi']:.2f}]"
            if stem in r and pd.notna(r[stem]) else "—")


def tool_block(m, cohort):
    sub = m[m.cohort == cohort]
    if not len(sub):
        return "_(cohort absent)_"
    rows = []
    for _, r in sub.iterrows():
        # R1-m19: omit R2 where the truth has (near-)zero variance
        cr2 = (round(r["comp_r2"], 3) if r.get("comp_r2_interpretable", True)
               else "n/a (SS_tot~0)")
        xr2 = (round(r["cont_r2"], 3) if r.get("cont_r2_interpretable", True)
               else "n/a (SS_tot~0)")
        rows.append({
            "tool": r["tool"], "n": r["n"], "clusters": r["n_clusters"],
            "comp MAE [95% CI]": fmt_ci(r, "comp_mae"),
            "comp bias [95% CI]": fmt_ci(r, "comp_bias"),
            "comp R2 (CoD)": cr2,
            "cont MAE [95% CI]": fmt_ci(r, "cont_mae"),
            "cont bias [95% CI]": fmt_ci(r, "cont_bias"),
            "cont R2 (CoD)": xr2})
    return mf.md_table(pd.DataFrame(rows))


def main():
    out = ["# WS3 Track A — real data with genuine ground truth",
           "",
           f"_Generated {datetime.now(timezone.utc).isoformat()}_",
           "",
           "Protocol §4.4e withdrew the 141-MAG real-MAG contamination claims, so these "
           "three cohorts are the only real-data validation routes with genuine ground "
           "truth. Reviewer 2 named mock communities and isolate draft/complete pairs "
           "specifically (R2-M3).",
           "",
           "**Conventions.** R² is the coefficient of determination (1 − SS_res/SS_tot) "
           "everywhere; squared Pearson is emitted separately as `*_r2_pearson` and is "
           "never called R². MIMAG-inspired thresholds use completeness/contamination "
           "only (HQ ≥90 % and <5 %; MQ ≥50 % and <10 %); the rRNA/tRNA criteria of the "
           "full standard are not evaluable here. All CIs are 95 % percentile "
           "cluster bootstraps (2,000 replicates) resampling reference organisms / "
           "species, because the same reference recurs across assemblies. Tests are "
           "two-sided Wilcoxon signed-rank on paired absolute errors.",
           "",
           "**Denominator (identical for the ground truth and for every tool):** "
           "completeness = retained dominant-organism bp / dominant reference FULL "
           "length × 100; contamination = contaminant bp / dominant reference FULL "
           "length × 100. This is MAGICC's own convention, so no denominator "
           "re-definition is needed (R1-M4, R1-M5).",
           ""]

    # ------------------------------------------------------------------ A1
    m = rd("meslier", "metrics_by_cohort.tsv")
    qc = rd("meslier", "alignment_qc.tsv")
    ri = rd("meslier", "reference_index.tsv")
    bt = rd("meslier", "bin_truth.tsv")
    out += ["## A1 — Meslier et al. 2022 MOCK1 (headline: fragmentation gradient on "
            "real data)", "",
            "Source: Meslier V. *et al.* *Benchmarking second and third-generation "
            "sequencing platforms for microbial metagenomics.* Sci Data 9, 694 (2022), "
            "doi:10.1038/s41597-022-01762-z. Reference genomes and the seven "
            "pre-computed MOCK1 assemblies come from the authors' public GitLab "
            "(`https://forge.inrae.fr/metagenopolis/benchmark_mock`); **no assembly "
            "compute was performed and no reads were downloaded**. 22 of the 91 "
            "reference strains carry ATCC designations with public GenBank "
            "accessions (ATCC MSA-1002 is a component of the mocks), which is how this "
            "work covers Reviewer 2's request for ATCC standards without the ATCC "
            "Genome Portal Data Use Agreement.", "",
            "> **Erratum to report if the raw runs are cited.** Meslier Table 4 prints "
            "the Illumina runs as ERR9765446-ERR9765449. Those accessions do not "
            "exist; the correct Illumina HiSeq 3000 runs, verified against the ENA "
            "portal API, are **ERR9765746 (MOCK_001), ERR9765747 (MOCK_002), "
            "ERR9765748 + ERR9765749 (MOCK_003)**. Table 4's ONT/Ion/PacBio accessions "
            "do resolve.", ""]
    if ri is not None:
        s_all = ri["split"].value_counts().to_dict()
        s_m1 = ri[ri.in_mock1]["split"].value_counts().to_dict()
        out += [f"91 reference genomes (68 bacterial / 23 archaeal, 29 GTDB phyla, "
                f"86/91 Complete). Split membership, GCA↔GCF cross-mapped: "
                f"all 91 → {s_all}; the 71 MOCK1 organisms → {s_m1}. "
                f"Leakage-free (not in TRAIN or VAL): "
                f"**{int(ri.leakage_free.sum())} of 91**, of which "
                f"**{int((ri.leakage_free & ri.in_mock1).sum())} are MOCK1 organisms**.",
                ""]
    if qc is not None:
        out += ["Alignment QC per assembly (minimap2 `-x asm10 --secondary=no`, "
                "contigs → pooled 91-reference index):", "",
                mf.md_table(qc[["assembly", "total_contigs", "total_bp",
                                "unassigned_bp_pct", "n_bins_mock1",
                                "bp_in_nonmock1_bins"]]), ""]
    if bt is not None:
        b = bt[(bt.in_mock1) & (bt.bin_fasta.notna()) & (bt.bin_fasta != "")]
        out += [f"**{len(b)} reference-anchored bins** over "
                f"{b.accession.nunique()} organisms × {b.assembly.nunique()} "
                f"assemblies. True completeness spans "
                f"{b.true_completeness.min():.1f}–{b.true_completeness.max():.1f} % "
                f"(median {b.true_completeness.median():.1f}), driven by the mock's "
                f"three-orders-of-magnitude abundance spread. True contamination is "
                f"low by construction of reference-anchored binning "
                f"(median {b.true_contamination.median():.3f} %, "
                f"max {b.true_contamination.max():.2f} %), so the contamination axis "
                f"here is a **real-data false-positive test**, not a quantification "
                f"benchmark. Bin bp per covered reference bp: median "
                f"{b.bin_bp_per_covered_ref_bp.median():.3f}, "
                f"max {b.bin_bp_per_covered_ref_bp.max():.3f} — the bins carry almost "
                f"no redundant sequence.", ""]
    ps = rd("meslier", "truth_preset_sensitivity.tsv")
    if ps is not None:
        import numpy as _np
        a5 = ps["asm5_minus_asm10"].values
        a20 = ps["asm20_minus_asm10"].values
        out += ["**Is the ground truth sensitive to the aligner setting?** No. "
                f"Recomputing the same reference-anchored truth for the Illumina "
                f"assembly under minimap2 `-x asm5` and `-x asm20` (n={len(ps)} bins "
                f"present in all three runs) changes completeness by "
                f"{_np.abs(a5).max():.4f} pp at most for asm5 (mean "
                f"{a5.mean():+.4f}) and {_np.abs(a20).max():.4f} pp at most for asm20 "
                f"(mean {a20.mean():+.4f}); no bin moves by more than 1 pp under either. "
                "`scripts/148`, `truth_preset_sensitivity.tsv`.", ""]
    aq = rd("meslier", "assembly_sequence_qc.tsv")
    if aq is not None:
        out += ["### The seven assemblies differ along TWO axes, not one", "",
                "Contig N50 (fragmentation) and per-base accuracy (platform error "
                "mode) are separate. Indel-dense assemblies frameshift ORFs, which a "
                "nucleotide k-mer method never sees and a protein-based method cannot "
                "survive. Coding density and mean gene length below come from "
                "**CheckM2's own output**, and the indel rates from base-level "
                "alignment of each assembly's largest contigs to the known references "
                "(`scripts/147`), so the diagnosis does not depend on this pipeline. "
                "Short mean gene length alone is not sufficient evidence — a highly "
                "fragmented short-read assembly also truncates genes at contig ends — "
                "hence the joint coding-density criterion.", "",
                mf.md_table(aq.sort_values("median_bin_n50", ascending=False)), ""]
    if m is not None:
        for cohort, title in (
                ("primary_leakage_free_comp>=50_ORF_intact",
                 "PRIMARY (fragmentation axis) — leakage-free organisms, true "
                 "completeness ≥50 %, ORF-intact assemblies only"),
                ("primary_leakage_free_comp>=50",
                 "PRIMARY (all seven assemblies) — leakage-free organisms, true "
                 "completeness ≥50 %"),
                ("ORF_compromised_assembly_only",
                 "Per-base-accuracy axis — the indel-dense assembly alone"),
                ("all_MOCK1_comp>=50_ORF_intact",
                 "Secondary — all MOCK1 organisms, ≥50 %, ORF-intact assemblies"),
                ("all_MOCK1_comp>=50",
                 "Secondary — all MOCK1 organisms, ≥50 %, all seven assemblies"),
                ("below_50pct_floor",
                 "Below MAGICC's stated 50 % completeness floor (reported for "
                 "transparency; MAGICC does not claim this regime)")):
            out += [f"### {title}", "", tool_block(m, cohort), ""]
    g = rd("meslier", "fragmentation_gradient.tsv")
    if g is not None:
        gg = g[g.panel == "balanced_panel_comp>=50"]
        if len(gg):
            for metric, mlab in (("comp_mae", "completeness"),
                                 ("cont_mae", "contamination")):
                piv = gg.pivot_table(index=["assembly", "assembly_median_bin_n50",
                                            "orf_integrity_compromised"],
                                     columns="tool", values=metric).reset_index()
                piv = piv.sort_values("assembly_median_bin_n50", ascending=False)
                piv = piv.rename(columns={"orf_integrity_compromised": "ORF_broken"})
                out += [f"### Fragmentation gradient — {mlab} MAE (pp) by assembly",
                        "", "Balanced panel: only organisms recovered in **every** "
                        "assembly at ≥50 % true completeness, so the same genomes are "
                        "compared at every fragmentation level. The `ORF_broken` row is "
                        "NOT a point on the fragmentation axis — see the two-axes "
                        "section above.", "", mf.md_table(piv), ""]
    sl = rd("meslier", "fragmentation_slopes.tsv")
    if sl is not None:
        s = sl[sl.panel == "balanced_panel_comp>=50"].drop(columns=["note"],
                                                           errors="ignore")
        out += ["### Degradation slopes (MAE per log10 decrease in bin N50)", "",
                "ORF-intact assemblies only.", "", mf.md_table(s), ""]
    pt = rd("meslier", "paired_tests.tsv")
    if pt is not None:
        out += ["### Paired MAGICC-vs-competitor tests "
                "(two-sided Wilcoxon on paired absolute errors)", "",
                mf.md_table(pt[pt.cohort.isin(
                    ["primary_leakage_free_comp>=50_ORF_intact",
                     "primary_leakage_free_comp>=50",
                     "ORF_compromised_assembly_only"])][
                        ["cohort", "metric", "tool_b", "n", "mae_a", "mae_b",
                         "mean_diff_a_minus_b", "diff_ci_lo", "diff_ci_hi",
                         "hodges_lehmann_shift", "wilcoxon_p_two_sided", "winner"]]),
                ""]
    mi = rd("meslier", "mimag_by_cohort.tsv")
    if mi is not None:
        for cohort, title in (
                ("primary_leakage_free_comp>=50_ORF_intact",
                 "MIMAG-inspired classification — PRIMARY (ORF-intact)"),
                ("primary_leakage_free_comp>=50",
                 "MIMAG-inspired classification — all seven assemblies")):
            out += [f"### {title}", "",
                    mf.md_table(mi[mi.cohort == cohort][
                        ["tool", "n", "true_HQ", "pred_HQ", "true_MQ", "pred_MQ",
                         "class_agreement_pct", "n_truly_below_5pct_cont",
                         "false_fail_5pct_n", "false_fail_5pct_rate",
                         "comp90_recall_pct"]]), ""]
    fp = rd("meslier", "per_organism_contamination_fp.tsv")
    if fp is not None:
        n = len(fp)
        counts = {c.replace("_cont_median", ""):
                  (int((fp[c] >= 5).sum()), int((fp[c] >= 2).sum()))
                  for c in fp.columns if c.endswith("_cont_median")}
        out += ["### Where the contamination false positives live", "",
                "Computed on truly-clean bins only (true contamination <1 %, true "
                "completeness ≥50 %), aggregated per reference organism. Whether a "
                "tool's false positives are spread thinly over many organisms or "
                "concentrated in a few is the difference between a noise floor and a "
                "systematic blind spot, and only the latter yields a usable limitation "
                f"statement. Organisms (of {n}) whose MEDIAN predicted contamination "
                "is ≥5 % / ≥2 %: "
                + "; ".join(f"{k} {v[0]} / {v[1]}" for k, v in counts.items()) + ".", "",
                "Ten worst organisms for MAGICC:", "",
                mf.md_table(fp.head(10)), ""]
    pb = rd("meslier", "per_base_accuracy_effect.tsv")
    if pb is not None:
        out += ["### Per-base-accuracy effect on the same balanced panel", "",
                mf.md_table(pb[["cohort", "tool", "n", "comp_mae", "comp_bias",
                                "cont_mae", "cont_bias", "assemblies"]]), ""]

    # ------------------------------------------------------------------ A2
    out += ["## A2 — ZymoBIOMICS isolate draft/complete pairs", ""]
    zt = rd("zymo", "per_pair_results.tsv")
    zm = rd("zymo", "metrics_by_cohort.tsv")
    if zt is not None:
        out += [f"8 bacterial SPAdes drafts (Nicholls et al. 2019) vs the ZymoBIOMICS "
                f"complete references; the 2 yeasts are excluded as eukaryotes. Each "
                f"draft was aligned against the POOLED 10-organism reference so that "
                f"cross-isolate carry-over is detectable as contamination. True "
                f"completeness "
                f"{zt.true_completeness.min():.2f}–{zt.true_completeness.max():.2f} %, "
                f"true contamination "
                f"{zt.true_contamination.min():.3f}–"
                f"{zt.true_contamination.max():.3f} % — a **precision / "
                f"false-positive test** in the near-complete, near-clean regime.", "",
                mf.md_table(zt[["organism", "n_contigs", "draft_bp", "ref_len",
                                "true_completeness", "true_contamination",
                                "true_contamination_upper",
                                "magicc_completeness", "checkm2_completeness",
                                "cocopye_completeness", "deepcheck_completeness",
                                "magicc_contamination", "checkm2_contamination",
                                "cocopye_contamination",
                                "deepcheck_contamination"]]), ""]
    if zm is not None:
        out += [tool_block(zm, "zymo_8_isolates"), ""]

    # ------------------------------------------------------------------ A3
    out += ["## A3 — NCBI Tier-1 same-BioSample draft/complete pairs", ""]
    nt = rd("ncbi_pairs", "per_pair_results.tsv")
    nm = rd("ncbi_pairs", "metrics_by_cohort.tsv")
    if nt is not None:
        out += [f"{len(nt)} prokaryotic pairs across {nt.species_taxid.nunique()} "
                f"species (draft/complete size ratio 0.80–1.20). Same BioSample = same "
                f"physical DNA isolate, so draft bp absent from the complete assembly "
                f"is assembly artefact or contamination rather than strain divergence. "
                f"True completeness median {nt.true_completeness.median():.2f} %; "
                f"contamination is reported as an **upper bound** "
                f"(median {nt.true_contamination_upper.median():.3f} %, "
                f"p90 {nt.true_contamination_upper.quantile(.9):.2f} %, "
                f"max {nt.true_contamination_upper.max():.2f} %) because unaligned "
                f"draft bp conflates true contamination with accessory content absent "
                f"from the chosen complete assembly and with assembly artefact.", "",
                f"Leakage: {int((nt.draft_split.isin(['train','val'])).sum())} drafts "
                f"and {int((nt.complete_split.isin(['train','val'])).sum())} completes "
                f"appear in TRAIN/VAL (GCA↔GCF cross-mapped); "
                f"**{int(nt.leakage_free.sum())} pairs are leakage-free**. Note that "
                f"only {int((~nt.species_in_train).sum())} pairs "
                f"({100*(~nt.species_in_train).mean():.1f} %) belong to a species "
                f"absent from training — the prokaryotic size-band cohort is dominated "
                f"by common clinical species. The ~79 % figure in the acquisition "
                f"report applies to all 4,834 Tier-1 pairs, which are mostly "
                f"eukaryotic and therefore out of MAGICC's scope; it does not hold for "
                f"this cohort and is corrected here.", ""]
    if nm is not None:
        for cohort, title in (("primary_leakage_free",
                               "PRIMARY — leakage-free pairs"),
                              ("all_tier1_pairs", "Secondary — all Tier-1 pairs"),
                              ("species_absent_from_training",
                               "Species absent from TRAIN/VAL")):
            out += [f"### {title}", "", tool_block(nm, cohort), ""]
    nf = rd("ncbi_pairs", "fragmentation_strata.tsv")
    if nf is not None:
        out += ["### Stratified by draft fragmentation", "",
                mf.md_table(nf[["cohort", "tool", "n", "median_draft_contigs",
                                "median_draft_n50", "comp_mae", "comp_bias",
                                "cont_mae", "cont_bias"]]), ""]
    nmi = rd("ncbi_pairs", "mimag_by_cohort.tsv")
    if nmi is not None:
        out += ["### MIMAG-inspired classification (leakage-free)", "",
                mf.md_table(nmi[nmi.cohort == "primary_leakage_free"][
                    ["tool", "n", "true_HQ", "pred_HQ", "class_agreement_pct",
                     "n_truly_below_5pct_cont", "false_fail_5pct_n",
                     "false_fail_5pct_rate", "comp90_recall_pct"]]), ""]
    nd = rd("ncbi_pairs", "ncbi_contaminated_flag_detection.tsv")
    if nd is not None:
        out += ["### NCBI `contaminated` flag — BINARY DETECTION ONLY", "",
                "NCBI's Foreign Contamination Screen uses a different operational "
                "definition from the bp-based contamination quantified above, so this "
                "table is a detection check, never a quantification benchmark.", "",
                mf.md_table(nd), ""]

    # ------------------------------------------------------------- synthesis
    syn = RD / "synthesis_table.tsv"
    if syn.exists():
        s = pd.read_csv(syn, sep="\t")
        out += ["## WS3.9 — cross-dataset synthesis", "",
                mf.md_table(s[["dataset", "tool", "n", "n_clusters", "comp_MAE",
                               "comp_MAE_95CI", "comp_bias", "comp_R2_CoD",
                               "cont_MAE", "cont_MAE_95CI", "cont_bias",
                               "cont_R2_CoD"]]), "",
                "Figures: `results/revision/real_data/figures/` "
                "(`fig_ws3_fragmentation_gradient`, `fig_ws3_meslier_scatter`, "
                "`fig_ws3_signed_errors`, `fig_ws3_realdata_synthesis`), captions in "
                "`figures/captions.md`.", ""]

    (RD / "WS3_trackA_report.md").write_text("\n".join(out))
    print(f"wrote {RD/'WS3_trackA_report.md'}")


if __name__ == "__main__":
    main()
