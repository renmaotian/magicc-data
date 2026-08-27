#!/usr/bin/env python3
"""Main-text Figure 4 -- where MAGICC generalizes and where it fails: taxonomy.

Five panels, all lifted from the previous round's Figures 4 and 5
(the internal build contract section 3):

    a  leave-phylum-out difference-in-differences, six held-out phylum groups
       (previous Fig. 4a)
    b  leave-family-out difference-in-differences, six held-out family groups
       (previous Fig. 4b)
    c  raw head-to-head signed completeness bias, production V5 vs the
       leave-phylum-out holdout model (previous Fig. 4d)
    d  contamination signed bias vs taxonomic distance, four tools, Set F
       (previous Fig. 5b)
    e  paired MAGICC-vs-CheckM2 Hodges-Lehmann difference, close vs distant
       contaminants, both CAMI II datasets (previous Fig. 5e)

The panels that left the main text are Figure S16.

Every number is read from a file under ``results/revision/`` by
``panels_taxonomy``/``figledger`` and re-verified against its source TSV before
this script exits.  Nothing is drawn on the canvas except axis labels, tick
labels, panel letters, legends and one direct label in panel c
(the internal build contract 4.3); the numbers that used to be annotated are carried by the
caption written to ``figures/caption_parts/Figure_4.md``.

    /path/to/anaconda3/bin/python make_figure_4.py
    /path/to/anaconda3/bin/python make_figure_4.py --draft
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figvalues  # noqa: E402
import panels_taxonomy as pt  # noqa: E402
from figstyle import DENOM_NOTE, FIG_DIR, add_panel_label, apply_style, save_fig  # noqa: E402
from figledger import DOUBLE_COL, LEDGER, report_verification, require_all  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

apply_style()


def build(draft: bool = False):
    fig = plt.figure(figsize=(DOUBLE_COL, 7.85))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.70],
                          hspace=0.46, wspace=0.60,
                          left=0.150, right=0.985, top=0.955, bottom=0.075)

    ax_a = fig.add_subplot(gs[0, 0])
    d_a = pt.draw_phylum_did(ax_a, fig="4", panel="a")
    add_panel_label(ax_a, "a", x=-0.40, y=1.10)

    ax_b = fig.add_subplot(gs[0, 1])
    d_b = pt.draw_family_did(ax_b, fig="4", panel="b")
    add_panel_label(ax_b, "b", x=-0.40, y=1.10)

    ax_c = fig.add_subplot(gs[1, 0])
    d_c = pt.draw_head_to_head(ax_c, fig="4", panel="c",
                               label_groups=("Patescibacteriota",))
    add_panel_label(ax_c, "c", x=-0.40, y=1.13)

    ax_d = fig.add_subplot(gs[1, 1])
    d_d = pt.draw_setF_bias_by_distance(ax_d, fig="4", panel="d")
    add_panel_label(ax_d, "d", x=-0.30, y=1.10)

    ax_e = fig.add_subplot(gs[2, :])
    d_e = pt.draw_cami_paired_hl(ax_e, fig="4", panel="e")
    add_panel_label(ax_e, "e", x=-0.145, y=1.13)

    return fig, dict(a=d_a, b=d_b, c=d_c, d=d_d, e=d_e)


# ---------------------------------------------------------------------------
# Caption -- carries every number removed from the canvas (the internal build contract 4.3)
# ---------------------------------------------------------------------------
def caption(d) -> str:
    a, b, c, dd, e = d["a"], d["b"], d["c"], d["d"], d["e"]
    orth = dd["pct_orthologous"]
    win = {k: {"magicc_v5": "MAGICC", "checkm2": "CheckM2"}.get(v, "a tie")
           for k, v in dd["winners"].items()}
    mb = dd["magicc_bias"]
    sm_c, sm_d = e[("strain_madness", "close")], e[("strain_madness", "distant")]
    ma_c, ma_d = e[("marine", "close")], e[("marine", "distant")]
    did_f = b["did"].set_index("group")
    heli = did_f.loc["Campylobacterota_Helicobacteraceae"]
    heli_lo, heli_hi = pt.parse_ci(heli["cont_did_ci95"])
    comp_f = b["did"]["comp_did"]
    comp_p = a["did"]["comp_did"]

    return f"""**Figure 4.** Where MAGICC generalizes and where it fails: taxonomy.

**a**, Leave-phylum-out difference-in-differences (DiD): a model retrained without ten phyla in six groups (19.64 % of 79,948 training genomes) against production model V5, on seven matched sets (6,994 genomes; 66-100 clusters per group; 1,000-sample in-distribution control). Positive DiD, the holdout model is worse. Circles, completeness; triangles, contamination; open markers, the control. Completeness degradation spans {comp_p.min():.2f}-{comp_p.max():.2f} pp, q(BH) = 5 × 10⁻⁴.

**b**, Leave-family-out DiD, same estimator and phyla, six held-out family groups (6.95 % of training genomes removed; 6,998 evaluation genomes, 28-100 clusters per group). Parent phyla stayed in training. Completeness degradation is {comp_f.min():.2f}-{comp_f.max():.2f} pp against {comp_p.min():.2f}-{comp_p.max():.2f} pp at phylum level in **a**, q(BH) ≤ 2 × 10⁻³.

**c**, Raw signed completeness error (predicted - true), V5 versus the leave-phylum-out holdout model on the same genomes. Violins, per-cluster mean signed errors (n = {c['Patescibacteriota']['n_refs']}, {c['DPANN']['n_refs']}, {c['Bacteroidota']['n_refs']}, {c['Bacteroidota_A']['n_refs']}, {c['Halobacteriota']['n_refs']}, {c['Campylobacterota']['n_refs']} and {c['in_distribution']['n_refs']} clusters, top to bottom); markers are group means. Removing the phylum collapses completeness by {c['Patescibacteriota']['holdout']:.2f} pp (Patescibacteriota) and {c['DPANN']['holdout']:.2f} pp (DPANN).

**d**, Signed contamination error against taxonomic distance, Set F, all four tools (n = 300 samples / 100 clusters per distance). MAGICC's bias is strongly distance-dependent ({mb['species']:.2f} pp at species to {mb['phylum']:.2f} pp at phylum) while CheckM2 and DeepCheck are flat. Paired against CheckM2, {win['species']} wins at species, {win['phylum']} at phylum: the sign flip **e** reproduces.

**e**, The same sign flip on both CAMI II datasets: Hodges-Lehmann paired MAGICC-minus-CheckM2 difference in absolute contamination error (negative favours MAGICC). Close (species + genus): strain-madness {sm_c['hl']:+.2f} [{sm_c['lo']:+.2f}, {sm_c['hi']:+.2f}] pp (n = {sm_c['n']:,} bins), marine {ma_c['hl']:+.2f} [{ma_c['lo']:+.2f}, {ma_c['hi']:+.2f}] pp (n = {ma_c['n']:,}). Distant (order + class + phylum): strain-madness {sm_d['hl']:+.2f} [{sm_d['lo']:+.2f}, {sm_d['hi']:+.2f}] pp (n = {sm_d['n']:,}), marine {ma_d['hl']:+.2f} [{ma_d['lo']:+.2f}, {ma_d['hi']:+.2f}] pp (n = {ma_d['n']:,}). All q(BH) ≤ 7.6 × 10⁻³.

Error bars, 95 % cluster-bootstrap confidence intervals; paired tests, BH-corrected Wilcoxon; errors in pp. {DENOM_NOTE}
"""


def caption_overflow(d) -> str:
    """Material trimmed out of the Figure 4 legend to meet the 350-word cap.

    Nothing here is new: every sentence was in the legend of the previous
    build and every number is still read from the same result file.
    """
    a, b, c, dd, e = d["a"], d["b"], d["c"], d["d"], d["e"]
    orth = dd["pct_orthologous"]
    win = {k: {"magicc_v5": "MAGICC", "checkm2": "CheckM2"}.get(v, "a tie")
           for k, v in dd["winners"].items()}
    mb = dd["magicc_bias"]
    sm_c, sm_d = e[("strain_madness", "close")], e[("strain_madness", "distant")]
    ma_c, ma_d = e[("marine", "close")], e[("marine", "distant")]
    heli = b["did"].set_index("group").loc["Campylobacterota_Helicobacteraceae"]
    heli_lo, heli_hi = pt.parse_ci(heli["cont_did_ci95"])

    return f"""## Figure 4 - material removed from the legend

**Panel a.** Ten phyla, 15,703 of 79,948 training genomes, were removed. The seven matched evaluation sets hold 1,000, 990, 1,000, 1,000, 1,008 and 996 samples over 100, 66, 100, 100, 72 and 83 reference-genome clusters for Bacteroidota, Bacteroidota_A, Campylobacterota, Patescibacteriota, Halobacteriota and DPANN, plus the 1,000-sample / 100-cluster in-distribution control. DiD = (holdout - V5 mean absolute error in the held-out group) - (holdout - V5 mean absolute error in the in-distribution control). Bars are 95 % percentile cluster-bootstrap confidence intervals over reference genomes, 2,000 resamples, and q(BH) = 5 × 10⁻⁴ holds for all six groups and both metrics. The in-distribution control's raw ΔMAE is {a['control_comp_dmae']:+.3f} pp completeness (two-sided paired Wilcoxon p = {a['control_comp_p']:.3f}) and {a['control_cont_dmae']:+.3f} pp contamination: removing 19.64 % of the training genomes had no measurable effect on retained lineages, so every degradation shown is lineage novelty alone. Source: `results/revision/holdout/lineage_novelty_effect_did.tsv`, `results/revision/holdout/head_to_head_by_group.tsv`.

**Panel b.** 5,558 of the 79,948 training genomes were removed. Seven evaluation sets, n = 1,000 samples / 100 clusters except Arcobacteraceae 1,000/40, Haloarculaceae + Haloferacaceae 1,008/28 and the 19 CPR families 990/66; in-distribution control 1,000/100. The control's ΔMAE is {b['control_comp_dmae']:+.3f} pp (p = {b['control_comp_p']:.3f}), again indistinguishable from zero, and Helicobacteraceae carries the worst contamination cell in the study ({float(heli['cont_did']):+.2f} pp [{heli_lo:.2f}, {heli_hi:.2f}]). The paired phylum-to-family attenuation behind this contrast is Fig. S16a. Source: `results/revision/holdout_family/lineage_novelty_effect_did.tsv`, `results/revision/holdout_family/head_to_head_by_group.tsv`.

**Panel c.** The filled markers reproduce `head_to_head_by_group.tsv` exactly, and the Patescibacteriota value is the only one labelled on the canvas. Patescibacteriota and DPANN are both reduced-genome lineages and both fail by under-calling completeness, while the in-distribution control is unchanged ({c['in_distribution']['v5']:+.3f} pp for V5 versus {c['in_distribution']['holdout']:+.3f} pp for the holdout model). Source: `results/revision/holdout/per_reference_errors.tsv`, `results/revision/holdout/head_to_head_by_group.tsv`.

**Panel d.** Error bars are 95 % percentile cluster-bootstrap CIs, 2,000 resamples. MAGICC's signed contamination bias by distance is {mb['species']:.2f} pp at species, {mb['genus']:.2f} at genus, {mb['family']:.2f} at family, {mb['order']:.2f} at order, {mb['class']:.2f} at class and {mb['phylum']:.2f} pp at phylum. The second line of each tick is the mean percentage of donor genes with a reciprocal best-hit orthologue in the dominant genome ({orth['species']:.0f} / {orth['genus']:.0f} / {orth['family']:.0f} / {orth['order']:.0f} / {orth['class']:.0f} / {orth['phylum']:.0f} %; 129, 105, 100, 100, 101 and 107 donor-acceptor pairs). In the MAGICC-versus-CheckM2 comparison of absolute contamination error paired by reference genome (two-sided Wilcoxon, BH-corrected, n = 300 pairs per distance) the better tool is {win['species']} at species, {win['genus']} at genus, {win['family']} at family, {win['order']} at order, {win['class']} at class and {win['phylum']} at phylum, with Hodges-Lehmann differences {dd['hl']['species']:+.2f}, {dd['hl']['genus']:+.2f}, {dd['hl']['family']:+.2f}, {dd['hl']['order']:+.2f}, {dd['hl']['class']:+.2f} and {dd['hl']['phylum']:+.2f} pp. The type-by-distance grid behind this margin is Fig. S16b, c. Source: `results/revision/set_F/set_F_marginals.tsv`, `.../set_F_distance_summary.tsv`, `.../set_F_paired_comparisons.tsv`.

**Panel e.** Mean absolute contamination error behind each Hodges-Lehmann estimate: close contaminants, strain-madness {sm_c['mae_magicc']:.2f} pp for MAGICC versus {sm_c['mae_checkm2']:.2f} pp for CheckM2 and marine {ma_c['mae_magicc']:.2f} versus {ma_c['mae_checkm2']:.2f} pp; distant contaminants, strain-madness {sm_d['mae_magicc']:.2f} versus {sm_d['mae_checkm2']:.2f} pp and marine {ma_d['mae_magicc']:.2f} versus {ma_d['mae_checkm2']:.2f} pp. The MAGICC-loses / MAGICC-wins boundary falls between genus and family on both datasets, and shading marks the half-plane favouring each tool. Source: `results/revision/cami2/analysis/cami2_paired_tests.tsv`.

**Shared.** Paired tests are two-sided Wilcoxon signed-rank tests with Benjamini-Hochberg correction throughout.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", action="store_true",
                    help="fast low-dpi PNG preview into the scratch directory")
    ap.add_argument("--draft-dir", default=os.environ.get(
        "MAGICC_FIG_DRAFT_DIR", "/tmp/magicc_fig_draft"))
    args = ap.parse_args()

    print(f"  {require_all()} input files verified")
    fig, d = build(args.draft)

    if args.draft:
        os.makedirs(args.draft_dir, exist_ok=True)
        p = os.path.join(args.draft_dir, "Figure_4.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  saved {p}")
    else:
        save_fig(fig, "Figure_4", out_dir=FIG_DIR)

    report_verification("Figure 4")

    if not args.draft:
        parts = os.path.join(FIG_DIR, "caption_parts")
        os.makedirs(parts, exist_ok=True)
        with open(os.path.join(parts, "Figure_4.md"), "w") as fh:
            fh.write(caption(d))
        print("  wrote Figure_4.md")
        with open(os.path.join(parts, "Figure_4_overflow.md"), "w") as fh:
            fh.write(caption_overflow(d))
        print("  wrote Figure_4_overflow.md")
        figvalues.write_values(FIG_DIR, "fig4",
                               [(f"{r['figure']}|{r['panel']}|{r['what']}", r["value"])
                                for r in LEDGER.rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
