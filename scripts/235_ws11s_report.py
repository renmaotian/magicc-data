#!/usr/bin/env python3
"""WS11.S -- assemble WS11_S_REPORT.md from the analysis tables (no new computation)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/path/to/magicc")
OUT = ROOT / "results/revision/ws11/spire_catalogue"
SIZE_LABELS = ["<1Mb", "1-2Mb", "2-3Mb", "3-5Mb", ">5Mb"]
LR_LABELS = ["<-1 (>=2x reduced)", "-1 to -0.5", "-0.5 to -0.25",
             "-0.25 to +0.25 (typical)", ">+0.25 (larger than lineage)"]


def tsv(name):
    with (OUT / name).open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    frame = json.loads((OUT / "sampling_frame.json").read_text())
    head = json.loads((OUT / "ws11s_headline.json").read_text())
    strat = tsv("classification_change_by_stratum.tsv")
    mat = tsv("classification_change_matrix.tsv")
    nr = tsv("nonresponse.tsv")
    cmp_ = tsv("small_vs_large_cohort_comparison.tsv")
    floor = tsv("floor_censoring.tsv")

    idx = {}
    for r in strat:
        idx[(r["cohort"], r["clustering_unit"], r["grouping"], r["stratum"], r["statistic"])] = r

    def g(c, gr, st, stat, unit="family"):
        return idx.get((c, unit, gr, st, stat))

    def pct(r, dec=2):
        if r is None:
            return "n/a"
        return (f"{100 * float(r['estimate']):.{dec}f} % "
                f"[{100 * float(r['ci95_lo']):.{dec}f}, {100 * float(r['ci95_hi']):.{dec}f}]")

    def dpct(r, dec=2):
        if r is None or not r.get("design_ci95_lo"):
            return "n/a"
        return (f"{100 * float(r['estimate']):.{dec}f} % "
                f"[{100 * float(r['design_ci95_lo']):.{dec}f}, "
                f"{100 * float(r['design_ci95_hi']):.{dec}f}]")

    def pp(r, dec=2):
        if r is None:
            return "n/a"
        return (f"{float(r['estimate']):+.{dec}f} pp "
                f"[{float(r['ci95_lo']):+.{dec}f}, {float(r['ci95_hi']):+.{dec}f}]")

    L = []
    A = L.append
    cw = head["catalogue_weighted"]
    fr = frame["frame"]
    n_an = cw["n_analysed"]
    A("# WS11.S — catalogue-scale SPIRE run: how often does the MIMAG-inspired class change?\n")
    A("**This is a disagreement analysis, not an error measurement.** Catalogue MAGs carry no "
      "ground truth. Everything below compares MAGICC V5 against SPIRE v1's *published* CheckM2 "
      "values; where the two disagree, nothing here says which is right. The ground-truthed "
      "anchor remains `set_C_clean`, where **MAGICC is the tool in error** (−8.68 pp completeness "
      "and +5.09 pp contamination against truth, versus CheckM2's −0.70 and −1.16), and that is "
      "the interpretation that governs the direction of every disagreement reported here.\n")
    A("Frozen `models/magicc_v5.onnx`, SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070"
      "ec182b2b3096`, verified before every inference run (scripts 231 and 232).\n")

    A("## 1. Cohorts and denominators\n")
    A(f"Frame: the **{fr['N']:,}** SPIRE v1 MAGs carrying a published CheckM2 completeness, "
      f"contamination and genome size ({frame['out_of_frame_no_published_checkm2']} of the "
      f"{frame['metadata_rows']:,} metadata rows have none and are out of frame). "
      f"{fr['n_phyla']} phyla, {fr['n_families']:,} families, {fr['n_spire_clusters']:,} 95 %-ANI "
      "clusters.\n")
    A("| Cohort | What it is | n analysed | Denominator meaning |")
    A("|---|---|---|---|")
    names = {
        "catalogue_weighted": ("stratified probability sample of the catalogue, design-weighted "
                               "and post-stratified for non-response", "the whole catalogue"),
        "catalogue_weighted_no_nr_adjustment": ("same, design weight only", "sensitivity check"),
        "S3_srs_unweighted": ("the 200,000-genome simple random sample alone, unweighted",
                              "stratum S3 (93.6 % of the catalogue)"),
        "S1_small_census": ("every frame MAG < 1 Mbp", "the sub-megabase catalogue"),
        "S2_rare_phylum_census": ("every frame MAG in a phylum with < 1,000 catalogue MAGs",
                                  "the rare-phylum catalogue"),
        "reps_census": ("every SPIRE 95 %-ANI MAG cluster representative",
                        "the dereplicated catalogue — a DIFFERENT population"),
    }
    for c, (what, den) in names.items():
        if c in head:
            A(f"| `{c}` | {what} | {head[c]['n_analysed']:,} | {den} |")
    A("")
    st = frame["strata"]
    A("Design (script 230, seed "
      f"{frame['seed']} = CRC-32(\"{frame['seed_string']}\"), numpy PCG64): "
      + "; ".join(f"**{h}** N={st[h]['N']:,} n={st[h]['n']:,} f={st[h]['sampling_fraction']:.4f} "
                  f"w={st[h]['design_weight']:.5f}" for h in st) + ".\n")
    A(f"The sampled cohort is **{frame['sample']['n']:,} genomes = "
      f"{100 * frame['sample']['fraction_of_frame']:.2f} % of the catalogue**; "
      f"{n_an:,} of them were retrieved and scored "
      f"(**{100 * n_an / fr['N']:.2f} % of the catalogue**). "
      f"Cluster bootstrap: 2,000 iterations, primary unit **family** "
      f"({cw['n_family_clusters']:,} families in the analysed cohort), secondary unit "
      f"**spire_cluster** ({cw.get('n_spire_clusters', 0):,} 95 %-ANI clusters).\n")

    A("## 2. Headline — proportion of catalogue MAGs whose MIMAG-inspired class changes\n")
    r = g("catalogue_weighted", "overall", "ALL", "p_class_change")
    A(f"**{pct(r)}** of SPIRE v1 catalogue MAGs change MIMAG-inspired class when MAGICC V5 is "
      f"substituted for the published CheckM2 values.\n")
    A(f"- denominator: all {fr['N']:,} catalogue MAGs, estimated from {n_an:,} analysed genomes;\n"
      f"- cohort: `catalogue_weighted`; clustering unit: **family**; 2,000-iteration cluster "
      "bootstrap. This is the conservative, model-based interval and uses the same convention as "
      "WS3.10.\n")
    A(f"**Design-based interval for the same quantity: {dpct(r)}.** The catalogue proportion is a "
      "finite-population descriptive parameter: its only sampling uncertainty comes from which S3 "
      "genomes were drawn (the S1 and S2 strata are censuses and contribute exactly zero "
      "variance), and because S3 is a simple random sample of *genomes*, family clustering does "
      "not enter the design variance. Quote the design interval for statements about **this** "
      "catalogue and the family-clustered interval for statements about MAGICC's behaviour in "
      "general.\n")
    rc = head["catalogue_weighted"].get("p_class_change_cluster_bootstrap")
    if rc:
        A(f"Clustering on 95 %-ANI species clusters instead of families gives "
          f"{100 * rc['estimate']:.2f} % [{100 * rc['ci95'][0]:.2f}, {100 * rc['ci95'][1]:.2f}] — "
          "the point estimate is identical by construction; only the interval changes.\n")
    A("| Direction | Rate [95 % CI] | Denominator |")
    A("|---|---|---|")
    A(f"| any class change | {pct(g('catalogue_weighted','overall','ALL','p_class_change'))} "
      f"(design {dpct(g('catalogue_weighted','overall','ALL','p_class_change'))}) | all MAGs |")
    A(f"| downgrade (class falls) | {pct(g('catalogue_weighted','overall','ALL','p_downgrade'))} | all MAGs |")
    A(f"| upgrade (class rises) | {pct(g('catalogue_weighted','overall','ALL','p_upgrade'))} | all MAGs |")
    hqr = g('catalogue_weighted', 'overall', 'ALL', 'HQ_downgrade_rate')
    nhq = None
    for row in mat:
        if row["cohort"] == "catalogue_weighted" and row["clustering_unit"] == "family" and row["checkm2_class"] == "HQ":
            nhq = int(row["row_denominator_n"])
    A(f"| HQ → not HQ | {pct(hqr)} | MAGs SPIRE publishes as HQ (n = {nhq:,} analysed) |")
    A(f"| HQ → MQ | {pct(g('catalogue_weighted','overall','ALL','HQ_to_MQ_rate'))} | published HQ |")
    A(f"| HQ → LQ | {pct(g('catalogue_weighted','overall','ALL','HQ_to_LQ_rate'))} | published HQ |")
    A(f"| MQ → HQ | {pct(g('catalogue_weighted','overall','ALL','MQ_to_HQ_rate'))} | published MQ |")
    A(f"| MQ → LQ | {pct(g('catalogue_weighted','overall','ALL','MQ_to_LQ_rate'))} | published MQ |")
    A("")
    A(f"Overall high-quality fraction: SPIRE's published CheckM2 calls "
      f"{pct(g('catalogue_weighted','overall','ALL','p_HQ_checkm2'))} of the catalogue HQ; MAGICC "
      f"calls {pct(g('catalogue_weighted','overall','ALL','p_HQ_magicc'))}.\n")
    A("Design check — the unweighted 200,000-genome SRS alone (stratum S3, 93.6 % of the "
      f"catalogue) gives {pct(g('S3_srs_unweighted','overall','ALL','p_class_change'))} for any "
      "class change; the weighted catalogue estimate above is the one to quote.\n")

    A("### 2.1 Confusion matrix (published CheckM2 class × MAGICC class)\n")
    A("Weighted proportion of the catalogue in each cell, `catalogue_weighted`, family-clustered "
      "bootstrap. Row-conditional rates are in the table above.\n")
    A("| published \\ MAGICC | HQ | MQ | LQ | row n analysed |")
    A("|---|---|---|---|---|")
    cells = {(r["checkm2_class"], r["magicc_class"]): r for r in mat
             if r["cohort"] == "catalogue_weighted" and r["clustering_unit"] == "family"}
    for a in ("HQ", "MQ", "LQ"):
        row = [f"| **{a}** "]
        for b in ("HQ", "MQ", "LQ"):
            c = cells.get((a, b))
            row.append(f"| {100 * float(c['weighted_proportion_of_cohort']):.3f} % "
                       f"[{100 * float(c['ci95_lo']):.3f}, {100 * float(c['ci95_hi']):.3f}] "
                       if c else "| n/a ")
        n = cells.get((a, "HQ"))
        row.append(f"| {int(n['row_denominator_n']):,} |" if n else "| 0 |")
        A("".join(row))
    A("")
    A("MAGICC cannot express completeness below 50, so it can only reach LQ through "
      "contamination ≥ 10 — read the LQ column with that in mind.\n")

    A("## 3. Stratified rates\n")
    A("### 3.1 By genome size (`catalogue_weighted`)\n")
    A("| size bin | n analysed | any class change | HQ → not HQ | published HQ % | MAGICC HQ % |")
    A("|---|---|---|---|---|---|")
    for lab in SIZE_LABELS:
        r1 = g("catalogue_weighted", "size_bin", lab, "p_class_change")
        A(f"| {lab} | {int(r1['n_genomes']):,} | {pct(r1)} | "
          f"{pct(g('catalogue_weighted','size_bin',lab,'HQ_downgrade_rate'))} | "
          f"{pct(g('catalogue_weighted','size_bin',lab,'p_HQ_checkm2'))} | "
          f"{pct(g('catalogue_weighted','size_bin',lab,'p_HQ_magicc'))} |")
    A("")
    A("The `<1Mb` row is a **census** of every sub-megabase MAG in the catalogue "
      f"({head['S1_small_census']['n_analysed']:,} analysed), not a sample.\n")

    A("### 3.2 By reduction relative to the lineage — log2(size / phylum median)\n")
    A("| stratum | n analysed | any class change | HQ → not HQ | mean Δcompleteness |")
    A("|---|---|---|---|---|")
    for lab in LR_LABELS:
        r1 = g("catalogue_weighted", "lineage_reduction", lab, "p_class_change")
        if r1 is None:
            continue
        A(f"| {lab} | {int(r1['n_genomes']):,} | {pct(r1)} | "
          f"{pct(g('catalogue_weighted','lineage_reduction',lab,'HQ_downgrade_rate'))} | "
          f"{pp(g('catalogue_weighted','lineage_reduction',lab,'mean_d_comp'))} |")
    A("")

    A("### 3.3 By phylum\n")
    A("Phyla with ≥ 200 analysed genomes, ordered by class-change rate; "
      "`catalogue_weighted`, family-clustered.\n")
    ph = [r for r in strat if r["cohort"] == "catalogue_weighted" and r["grouping"] == "phylum"
          and r["statistic"] == "p_class_change" and r["clustering_unit"] == "family"]
    ph.sort(key=lambda r: -float(r["estimate"]))
    A("| phylum | n analysed | any class change | HQ → not HQ | mean Δcompleteness |")
    A("|---|---|---|---|---|")
    for r1 in ph:
        lab = r1["stratum"]
        A(f"| {lab} | {int(r1['n_genomes']):,} | {pct(r1)} | "
          f"{pct(g('catalogue_weighted','phylum',lab,'HQ_downgrade_rate'))} | "
          f"{pp(g('catalogue_weighted','phylum',lab,'mean_d_comp'))} |")
    A("")

    A("## 4. Delta distributions and the size dose–response\n")
    dd = tsv("delta_distributions.tsv")
    A("MAGICC − CheckM2, `catalogue_weighted`, weighted medians and means with "
      "family-clustered bootstrap CIs.\n")
    A("| stratum | metric | weighted median [95 % CI] | weighted mean [95 % CI] |")
    A("|---|---|---|---|")
    for r1 in dd:
        if r1["cohort"] != "catalogue_weighted" or r1["grouping"] not in ("overall", "size_bin"):
            continue
        A(f"| {r1['stratum']} | {r1['metric'].replace('_MAGICC_minus_CheckM2','')} | "
          f"{float(r1['weighted_median']):+.2f} [{float(r1['ci95_lo']):+.2f}, "
          f"{float(r1['ci95_hi']):+.2f}] | {float(r1['weighted_mean']):+.2f} "
          f"[{float(r1['mean_ci95_lo']):+.2f}, {float(r1['mean_ci95_hi']):+.2f}] |")
    A("")
    dose = tsv("size_dose_response.tsv")
    A("**Size dose–response** (weighted OLS of Δ on log10 assembly Mbp, family-clustered):\n")
    A("| cohort | subset | n | metric | slope pp per log10 Mbp [95 % CI] |")
    A("|---|---|---|---|---|")
    for r1 in dose:
        if r1["subset"] != "overall:ALL":
            continue
        A(f"| {r1['cohort']} | {r1['subset']} | {int(r1['n_genomes']):,} | "
          f"{r1['metric'].replace('_MAGICC_minus_CheckM2','')} | "
          f"{float(r1['slope_pp_per_log10Mbp']):+.2f} [{float(r1['ci95_lo']):+.2f}, "
          f"{float(r1['ci95_hi']):+.2f}] |")
    A("")

    A("## 5. Floor censoring\n")
    A("MAGICC cannot express completeness below 50, so a prediction sitting at the floor makes "
      "its completeness delta a **lower bound** on the disagreement.\n")
    A("| cohort | stratum | n | n ≤ 50.5 | n ≤ 50.0 | rate [95 % CI] | min MAGICC completeness |")
    A("|---|---|---|---|---|---|---|")
    for r1 in floor:
        if r1["grouping"] not in ("overall", "size_bin"):
            continue
        if r1["cohort"] not in ("catalogue_weighted", "S1_small_census", "reps_census"):
            continue
        A(f"| {r1['cohort']} | {r1['stratum']} | {int(r1['n_genomes']):,} | "
          f"{r1['n_at_floor_le_50.5']} | {r1['n_at_floor_le_50.0']} | "
          f"{100 * float(r1['rate_le_50.5']):.3f} % [{100 * float(r1['ci95_lo']):.3f}, "
          f"{100 * float(r1['ci95_hi']):.3f}] | {float(r1['min_magicc_completeness']):.2f} |")
    A("")

    A("## 6. Non-response\n")
    ov = [r for r in nr if r["grouping"] == "overall"][0]
    A(f"{int(ov['n_retrieved']):,} of {int(ov['n_sampled']):,} sampled genomes were retrieved "
      f"(**{100 * float(ov['response_rate']):.2f} %**). Non-response is a persistent server-side "
      "HTTP 404 on `spire.embl.de/download_file/<id>`, verified by slow single retries, and it is "
      "clustered by originating study. Every frame variable is known for the non-respondents, so "
      "the analysis post-stratifies on (stratum × size bin × published class); the "
      "`catalogue_weighted_no_nr_adjustment` cohort is the unadjusted sensitivity check.\n")
    A("| grouping | level | n sampled | n retrieved | response rate |")
    A("|---|---|---|---|---|")
    for r1 in nr:
        if r1["grouping"] in ("overall", "stratum", "size_bin", "checkm2_class", "recovery"):
            A(f"| {r1['grouping']} | {r1['level']} | {int(r1['n_sampled']):,} | "
              f"{int(r1['n_retrieved']):,} | {100 * float(r1['response_rate']):.2f} % |")
    A("")
    a1 = g("catalogue_weighted", "overall", "ALL", "p_class_change")
    a2 = g("catalogue_weighted_no_nr_adjustment", "overall", "ALL", "p_class_change")
    A(f"Non-response adjustment moves the headline from {pct(a2)} to {pct(a1)}.\n")

    A("## 7. Catalogue scale versus the 750-genome cohort\n")
    A("The cohort the manuscript currently cites is **750 GTDB r220 MAGs drawn 150 per "
      "genome-size bin** (WS3.10, script 131, seed 13100). That design over-samples `<1Mb` by "
      "about 6× and `>5Mb` by about 13× relative to the catalogue, so its pooled numbers are not "
      "catalogue rates. Two things separate it from the catalogue estimate — the **design** and "
      "the **catalogue** (GTDB r220 is not SPIRE v1). The design effect is isolated below by "
      "redrawing the 150-per-size-bin design 500 times from the SPIRE data measured here; the "
      "catalogue difference cannot be isolated without rescoring GTDB and is stated, not "
      "estimated.\n")
    A("| quantity | cohort | n | estimate [95 % CI] |")
    A("|---|---|---|---|")
    for r1 in cmp_:
        lo, hi = r1["ci95_lo"], r1["ci95_hi"]
        ci = "" if lo in ("nan", "") else f" [{float(lo):+.3g}, {float(hi):+.3g}]"
        A(f"| {r1['quantity']} | {r1['cohort']} | {r1['n']} | {float(r1['estimate']):+.4g}{ci} |")
    A("")

    A("## 8. Files\n")
    for f in sorted(OUT.iterdir()):
        if f.is_file():
            A(f"- `{f.name}` ({f.stat().st_size:,} B)")
    A("\n## 9. Reproduction\n")
    A("```\npython3 scripts/230_ws11s_sampling_frame.py\n"
      "python3 scripts/231_ws11s_fetch_and_score.py --workers 14 --feat 9\n"
      "python3 scripts/232_ws11s_representatives.py --feat 6\n"
      "PYTHONHASHSEED=0 python3 scripts/233_ws11s_analysis.py\n"
      "PYTHONHASHSEED=0 python3 scripts/234_ws11s_compare_and_report.py\n"
      "PYTHONHASHSEED=0 python3 scripts/235_ws11s_report.py\n```\n")
    (OUT / "WS11_S_REPORT.md").write_text("\n".join(L) + "\n")
    print(f"[235] wrote {OUT / 'WS11_S_REPORT.md'} ({len(L)} lines)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
