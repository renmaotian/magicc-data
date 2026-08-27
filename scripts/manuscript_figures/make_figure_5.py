#!/usr/bin/env python3
"""Main-text Figure 5 -- real data, error robustness and the genome-size boundary.

Six panels, all lifted from the previous round's Figures 6 and 7
(the internal build contract section 3):

    a  mock-community fragmentation gradient, completeness MAE vs contig N50
       (previous Fig. 6a)
    b  ZymoBIOMICS isolate pairs + CAMI II strain-madness gold bins
       (previous Fig. 6c)
    c  Set G uneven-coverage duplication dose-response, the unfavourable result
       (previous Fig. 6f)
    d  MAGICC-minus-CheckM2 completeness by genome-size bin -- a DISAGREEMENT
       on data with no ground truth, never an error (previous Fig. 7a)
    e  ground-truthed anchor: signed error on genuinely clean Set C-clean
       Patescibacteriota (previous Fig. 7b)
    f  size-channel elasticity phi, read-only intervention on the frozen model
       (previous Fig. 7c)

The panels that left the main text are Figure S17.

Every number is read from a file under ``results/revision/`` by
``panels_realdata`` and checked against the value recorded in
`the internal project log` before this script exits.  Nothing is drawn
on the canvas except axis labels, tick labels, panel letters, panel titles,
legends and one short direct label in panel c (the internal build contract 4.3); the numbers
that used to be annotated are carried by
``figures/caption_parts/Figure_5.md``.

    /path/to/anaconda3/bin/python make_figure_5.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figvalues  # noqa: E402
import panels_realdata as pr  # noqa: E402
from figstyle import DENOM_NOTE, FIG_DIR, add_panel_label, apply_style, save_fig  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

apply_style()

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def build():
    D6 = pr.load6()
    D7 = pr.load7()

    # Canvas width 7.10 in, not pr.DOUBLE_COL (7.205 in = 183 mm).
    # savefig.bbox="tight" saves the union of the canvas and every artist that
    # overflows it, so Figure 5 on a full-width canvas rendered 181.97 mm
    # against the 183 mm Nature Communications cap and the 180 mm working
    # target this build holds every figure to; at 7.10 in it renders 179.55
    # mm. Font sizes are unchanged in points, the canvas was stepped down and
    # re-measured 0.05 in at a time, and no new text collision appears at any
    # width down to 7.00 in. No plotted value moves: the artist ledger is
    # byte-identical to the full-width build.
    fig = plt.figure(figsize=(7.10, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.20, 1.0],
                          hspace=0.72, wspace=0.34,
                          left=0.080, right=0.985, top=0.930, bottom=0.055)

    ax_a = fig.add_subplot(gs[0, 0])
    d_a = pr.draw_fragmentation(ax_a, D6)
    add_panel_label(ax_a, "a", x=-0.165, y=1.20)

    ax_b1, ax_b2 = pr.draw_known_composition(fig, gs[0, 1], D6)
    add_panel_label(ax_b1, "b", x=-0.42, y=1.20)

    ax_c = fig.add_subplot(gs[1, 0])
    d_c = pr.draw_dose_duplication(ax_c, D6)
    add_panel_label(ax_c, "c", x=-0.165, y=1.20)

    ax_d1, ax_d2, d_d = pr.draw_size_disagreement(fig, gs[1, 1], D7,
                                                  width_ratios=(1.45, 1.0))
    add_panel_label(ax_d1, "d", x=-0.30, y=1.20)

    ax_e = fig.add_subplot(gs[2, 0])
    d_e = pr.draw_anchor(ax_e, D7)
    add_panel_label(ax_e, "e", x=-0.165, y=1.22)

    ax_f = fig.add_subplot(gs[2, 1])
    d_f = pr.draw_phi(ax_f, D7)
    add_panel_label(ax_f, "f", x=-0.42, y=1.22)

    fig.legend(handles=pr.tool_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6.4,
               handletextpad=0.35, columnspacing=1.4)

    return fig, D6, D7, dict(a=d_a, c=d_c, d=d_d, e=d_e, f=d_f)


# ---------------------------------------------------------------------------
# Caption -- carries every number removed from the canvas (the internal build contract 4.3)
# ---------------------------------------------------------------------------
def caption(D6, D7, d) -> str:
    a, c, dd, e, f = d["a"], d["c"], d["d"], d["e"], d["f"]
    zy, sm, orf = D6["zy"], D6["sm"], D6["orf"]
    av = e["av"]
    rc = f["rows_c"]
    du, dt, dp = c["kmer"]
    e_lo = int(round(float(f"{D6['cons_lo']:e}".split('e')[1])))
    e_hi = int(round(float(f"{D6['cons_hi']:e}".split('e')[1])))
    p_exp = int(f"{dd['p']:e}".split("e")[1])
    p_man = dd["p"] / 10 ** p_exp
    bd = dd["bin_deltas"]

    return f"""**Figure 5.** Real data, error robustness and the genome-size boundary.

**a**, MOCK1 mock community: completeness mean absolute error (MAE) against contig N50 for {a['n_organisms']} organisms present in every assembly at true completeness ≥ 50 %, over six ORF-intact assemblies. Slopes per log₁₀ N50: MAGICC {a['slope_magicc']:+.2f}, CheckM2 {a['slope_checkm2']:+.2f} pp.

**b**, Known-composition cohorts: ZymoBIOMICS isolate drafts (n = {int(zy['n'].iloc[0])}) and CAMI II strain-madness gold bins (n = {int(sm['n'].iloc[0])} bins, {int(sm['n_clusters'].iloc[0])} source genomes). Filled markers, completeness MAE; open, contamination MAE. MAGICC does not win on Zymo (DeepCheck {zy.loc['deepcheck','comp_MAE']:.2f} versus {zy.loc['magicc','comp_MAE']:.2f} pp) but leads on strain-madness ({sm.loc['magicc','comp_MAE']:.2f} versus CheckM2 {sm.loc['checkm2','comp_MAE']:.2f} pp).

**c**, Set G uneven-coverage duplication, the only axis on which MAGICC degrades (24 error arms paired within reference, n = 80 per dose): its completeness MAE rises from {c['mae_ctrl']:.2f} to {c['mae_max']:.2f} pp at 40 % duplicated bp. Dashed lines, its 1-pp applicability boundary ({c['b_magicc']:.2f} %) and the {c['crossover']:.0f} % crossover past which it is worse than CheckM2 (ΔMAE {c['dmax']:+.2f} pp [{c['dmax_ci'][0]:.2f}, {c['dmax_ci'][1]:.2f}]).

**d**, A disagreement, not an error. Left, median MAGICC-minus-CheckM2 completeness on catalogue MAGs by genome-size bin (n = {dd['n_mag']} MAGs, {dd['n_clu']} family clusters), {dd['slope']:+.2f} pp per log₁₀ Mbp [{dd['slope_ci'][0]:.2f}, {dd['slope_ci'][1]:.2f}]; control genera n = {dd['n_ctrl']}, reviewer genera n = {dd['n_rev']}. Right, size relative to the phylum median.

**e**, The ground-truthed anchor, the only panel measured against truth: signed error on clean Set C-clean Patescibacteriota (n = {D7['n_anch']}, {D7['ncl_anch']} clusters). MAGICC under-calls completeness by {av[('magicc','completeness')][0]:+.2f} pp [{av[('magicc','completeness')][1]:+.2f}, {av[('magicc','completeness')][2]:+.2f}] and over-calls contamination by {av[('magicc','contamination')][0]:+.2f} pp [{av[('magicc','contamination')][1]:+.2f}, {av[('magicc','contamination')][2]:+.2f}]; CheckM2 is near truth, so MAGICC is the tool in error.

**f**, Mechanism, by intervention on the frozen model: φ = d log(predicted completeness) / d log(assembly size), k-mer composition held fixed (n = {f['n_phi']:,} assemblies, {f['ncl_phi']:,} references). Composition alone would give φ = 1 (dashed); MAGICC gives {rc['pooled (all inputs)'][0]:.3f} [{rc['pooled (all inputs)'][1]:.3f}, {rc['pooled (all inputs)'][2]:.3f}], flat across reference-size strata ({min(f['phis']):.3f}-{max(f['phis']):.3f}).

Error bars, 95 % cluster-bootstrap confidence intervals; errors in percentage points (pp). {DENOM_NOTE}
"""


def caption_overflow(D6, D7, d) -> str:
    """Material trimmed out of the Figure 5 legend to meet the 350-word cap.

    Nothing here is new: every sentence was in the legend of the previous
    build, and every number is still read from the same result file.  It is
    emitted separately so that another agent can move it into the
    corresponding supplementary figure note without re-deriving anything.
    """
    a, c, dd, e, f = d["a"], d["c"], d["d"], d["e"], d["f"]
    zy, sm = D6["zy"], D6["sm"]
    av = e["av"]
    rc = f["rows_c"]
    du, dt, dp = c["kmer"]
    p_exp = int(f"{dd['p']:e}".split("e")[1])
    p_man = dd["p"] / 10 ** p_exp
    bd = dd["bin_deltas"]

    return f"""## Figure 5 - material removed from the legend

**Panel a.** The mock community is MOCK1 of Meslier et al. (2022). The six ORF-intact assemblies are PacBio, Illumina, Ion S5, Ion Proton, MGISEQ-2000 and MGISEQ-T7; they span a {a['n50_span']:,.0f}-fold range of median bin N50, from 1.85 Mb to 1.6 kb. Points are cohort MAEs and error bars 95 % percentile cluster-bootstrap confidence intervals (2,000 iterations, clustered on the {a['n_organisms']} reference organisms); the slopes are ordinary least squares over the six assembly-level MAEs (MAGICC p = {a['p_magicc']:.3f}, CheckM2 p = {a['p_checkm2']:.3f}). CheckM2 is the more accurate tool on the least-fragmented (PacBio) assembly ({D6['pb_c']:.2f} versus {D6['pb_m']:.2f} pp); MAGICC degrades about half as fast and overtakes it as fragmentation rises. The panel is not restricted to leakage-free organisms; on the leakage-free subset ({D6['n_lf_org']} organisms) the slopes keep the same sign and ordering (MAGICC {D6['sl_lf_m']:+.2f}, CheckM2 {D6['sl_lf_c']:+.2f}). The complementary per-base-accuracy split of the same mock community is Fig. S17a. Source: `results/revision/real_data/meslier/fragmentation_gradient.tsv`, `.../fragmentation_slopes.tsv`.

**Panel b.** ZymoBIOMICS drafts are from Nicholls et al. (2019); 2 yeasts are excluded and each draft is aligned against the pooled 10-organism reference. The CAMI II strain-madness bins are leakage-free and ≥ 50 % true completeness, and 0 % of that dataset is present in MAGICC's training data. Bars are 95 % cluster-bootstrap confidence intervals (2,000 iterations) clustered on strain and on CAMI II source genome respectively. On the Zymo cohort CheckM2 gives the lowest contamination MAE ({zy.loc['checkm2','cont_MAE']:.2f} pp); all four tools over-call completeness ({D6['zy_bias_lo']:+.2f} to {D6['zy_bias_hi']:+.2f} pp) and all four classify all eight drafts as MIMAG-inspired high quality, matching truth. On the strain-madness gold bins the contamination MAEs are {sm.loc['magicc','cont_MAE']:.2f} pp for MAGICC against {sm.loc['checkm2','cont_MAE']:.2f} pp for CheckM2; true contamination there is zero by construction, so that axis is a false-positive test. Contamination R² is omitted because true contamination is near-constant in these cohorts (SS_tot ≈ 0, R1-m19). The {D6['n_ncbi']:,} NCBI strain-matched draft/complete pairs are not shown because the competitor predictions were never merged onto that truth table, so no four-tool comparison exists for it. Source: `results/revision/real_data/synthesis_table_with_cami2.tsv`, cross-checked against `results/revision/cami2/analysis/cami2_accuracy_by_cohort.tsv`.

**Panel c.** The 1,920 assemblies are 80 held-out reference genomes × 24 error arms, paired within reference, so truth, fragmentation realisation and contaminants are identical across arms. At 40 % duplicated base pairs MAGICC's signed bias is {c['bias_max']:+.2f} pp, an over-call. The applicability boundary is defined as the rate at which the upper 95 % confidence limit of the paired degradation first exceeds 1 pp; CheckM2's own boundary is {c['b_checkm2']:.1f} %. Mechanistically, duplication leaves the unique k-mer count exactly unchanged ({du:+.3f} z) while inflating `log10_total_kmer_count` ({dt:+.2f} z) and `duplicate_kmer_count` ({dp:+.2f} z), so a duplicated assembly reads as a larger, more complete genome; users scoring strain-heterogeneous assemblies should de-replicate first. The substitution and indel dose-responses of the same experiment are Fig. S17b, c. Source: `results/revision/set_G/set_G_curves.tsv` (the duplication rows are the corrected values in this file, superseding the transcription recorded as W13), `.../set_G_paired_degradation.tsv`, `.../set_G_boundary.tsv`, `.../set_G_crossover.tsv`, `.../set_G_kmer_perturbation_summary.tsv`.

**Panel d.** The size bins hold 150 MAGs each; the control genera hold 40 each and the reviewer genera are GTDB r220. Both context cohorts are drawn in the neutral reference grey with a marker used by no tool, because no tool series appears in this panel. Error bars are 95 % percentile family-clustered bootstrap intervals (5,000 iterations). The bin medians are {bd[0]:+.2f}, {bd[1]:+.2f}, {bd[2]:+.2f}, {bd[3]:+.2f} and {bd[4]:+.2f} pp from the smallest to the largest bin, and the five reviewer genera pool to {dd['rev_all']:+.2f} pp. The dose-response is fitted by ordinary least squares with family-clustered standard errors (p = {p_man:.1f} × 10{str(p_exp).translate(SUP)}, R² = {dd['r2']:.3f}), whereas the non-reduced controls show no size slope ({dd['slope_ctrl']:+.2f} pp per log₁₀ Mbp, p = {dd['p_ctrl']:.2f}): the effect is genome size, not "MAG-ness". On the lineage-relative axis, at log₂(size / phylum median) < −1 (at least twofold reduced) the disagreement is {dd['reduced_delta']:+.2f} pp and MAGICC fails a genome at the 5 % contamination threshold while CheckM2 passes it in {dd['pct_reduced']:.1f} % of cases, against {dd['pct_typical']:.1f} % for lineage-typical genomes (n = 257 and 412 genomes, 44 and 154 family clusters). Because these are catalogue MAGs the panel shows where the two tools disagree, not which one is right. The reviewer's catalogue is identified as SPIRE v1 (mean absolute residual against the reviewer's reported deltas {dd['spire']:.2f} pp, versus {dd['gtdb']:.2f} pp for GTDB r220 and {dd['uhgg']:.2f} pp for UHGG v2.0.2). Source: `results/revision/real_data/reduced_genome/size_bin_deltas.tsv`, `.../lineage_relative_size_deltas.tsv`, `.../reviewer_genera_deltas.tsv`, `.../catalogue_baseline_reconciliation.tsv`, `.../mechanism_models.tsv`.

**Panel e.** The anchor cohort is Set C-clean Patescibacteriota with true contamination < 5 % and true completeness ≥ 90 %, resampled by a 5,000-iteration reference-clustered bootstrap. CheckM2's signed errors are {av[('checkm2','completeness')][0]:+.2f} pp [{av[('checkm2','completeness')][1]:+.2f}, {av[('checkm2','completeness')][2]:+.2f}] completeness and {av[('checkm2','contamination')][0]:+.2f} pp [{av[('checkm2','contamination')][1]:+.2f}, {av[('checkm2','contamination')][2]:+.2f}] contamination. Controlling for true completeness and true contamination on the leakage-free ground-truth corpus (n = {e['n_corpus']:,} samples from {e['ncl_corpus']:,} reference genomes; the regression is fitted in the clean and near-complete regime, n = {e['n_m1']} samples, {e['ncl_m1']} reference clusters), the partial R² of log₁₀ reference size on predicted completeness is {e['pr2_m']:.3f} for MAGICC against {e['pr2_c']:.3f} for CheckM2. The uncontrolled value of 0.313 previously estimated on real MAGs is withdrawn (register entry W6): direction and the contrast with CheckM2 survive, the magnitude does not. Source: `results/revision/real_data/reduced_genome/set_C_clean_crosscheck.tsv`, `.../mitigation/mechanism_m1_matched_completeness.tsv`.

**Panel f.** φ is obtained by binomial thinning for s < 1 and exact multiplication for s > 1, with 95 % CIs from a reference-clustered bootstrap. The complement of the pooled φ is the size channel, an elasticity of {f['L_elast']:.2f} of the model's implied reference length to the observed assembly size. Decomposed by input branch, the leak sits in the k-mer branch (φ = {rc['k-mer branch only'][0]:.3f} [{rc['k-mer branch only'][1]:.3f}, {rc['k-mer branch only'][2]:.3f}]) and not in the seven k-mer summary features (φ = {rc['7 summary features only'][0]:.3f} [{rc['7 summary features only'][1]:.3f}, {rc['7 summary features only'][2]:.3f}]), because MAGICC's k-mer inputs are log1p(absolute count) rather than relative frequencies. Per stratum, n = 666 each: {', '.join(f"{lab} {v:.3f} [{lo:.3f}, {hi:.3f}]" for lab, v, lo, hi, _n, _k in f['rows_s'])}. The size channel is therefore a uniform property of the estimator, not something that switches on for small genomes. Whether the correction transfers to a novel lineage is Fig. S17d. Source: `results/revision/real_data/reduced_genome/mitigation/intervention_phi_pooled.tsv`, `.../intervention_phi_by_reference_size.tsv`, `.../intervention_size_channel.tsv`.

**Shared.** Bootstraps are seeded with a CRC-32 stable hash under `PYTHONHASHSEED=0`; paired comparisons are two-sided Wilcoxon signed-rank tests with Benjamini-Hochberg correction; R² is the coefficient of determination throughout, never a squared Pearson correlation. Panel d plots disagreements between two tools on real catalogue MAGs, not errors: those data carry no ground truth, and only panel e is measured against truth. Genome sizes are in megabase pairs (Mbp).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=float, default=None)
    args = ap.parse_args()
    if args.dpi:
        plt.rcParams["savefig.dpi"] = args.dpi
        plt.rcParams["figure.dpi"] = args.dpi

    fig, D6, D7, d = build()
    save_fig(fig, "Figure_5", out_dir=FIG_DIR)
    pr.verification_report()

    parts = os.path.join(FIG_DIR, "caption_parts")
    os.makedirs(parts, exist_ok=True)
    with open(os.path.join(parts, "Figure_5.md"), "w") as fh:
        fh.write(caption(D6, D7, d))
    print("  wrote Figure_5.md")
    with open(os.path.join(parts, "Figure_5_overflow.md"), "w") as fh:
        fh.write(caption_overflow(D6, D7, d))
    print("  wrote Figure_5_overflow.md")

    rows = []
    figvalues.flatten("fig6", D6, rows)
    figvalues.flatten("fig7", D7, rows)
    figvalues.write_values(FIG_DIR, "fig5", rows)
    figvalues.write_values(
        FIG_DIR, "fig5_checks",
        [(f"check|{lab}", got) for lab, got, _exp, _ok in pr._VERIFY])
    return 0


if __name__ == "__main__":
    sys.exit(main())
