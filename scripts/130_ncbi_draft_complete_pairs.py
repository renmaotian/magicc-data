#!/usr/bin/env python3
"""
WS3 Phase 1 (Priority 3) — discover strain-matched draft/complete assembly pairs in
NCBI GenBank at scale, as a candidate large-N ground-truth cohort for MAGICC.

Rationale
---------
Reviewer 2 (Major comment 3) asks for validation on "real draft genomes, MAGs, and
SAGs derived from isolates ... for which high-quality complete reference genomes are
available", because "the expected genome content is known and assembly quality can be
assessed relative to a reference".

The cheapest large-N source of such pairs is NCBI itself: many isolates have BOTH a
fragmented draft assembly and a later complete (or chromosome-level) assembly.  Two
levels of matching strength are computed:

  TIER 1 (strongest) — same BioSample, one assembly Complete Genome/Chromosome and
      another Contig/Scaffold.  Same BioSample means the same physical DNA isolate, so
      any sequence in the draft that is absent from the complete assembly is either
      assembly artefact or genuine contamination, not strain divergence.

  TIER 2 (strong)    — same species_taxid AND identical normalised strain designation
      (infraspecific_name / isolate), different BioSamples.  Same named strain from
      possibly different labs/passages: a small amount of real divergence is possible.

Outputs (results/revision/real_data/ncbi_pairs/)
  tier1_same_biosample_pairs.tsv
  tier2_same_strain_pairs.tsv
  pairs_summary.json
  contaminated_flagged_assemblies_summary.json   (NCBI's own "contaminated" exclusion flag)

Read-only on data/ncbi/. No network access. Single-threaded, ~2 min, <8 GB RAM.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/path/to/magicc")
SUMMARY = ROOT / "data" / "ncbi" / "assembly_summary_genbank.txt"
OUTDIR = ROOT / "results" / "revision" / "real_data" / "ncbi_pairs"

COMPLETE_LEVELS = {"Complete Genome", "Chromosome"}
DRAFT_LEVELS = {"Contig", "Scaffold"}

# Columns (0-based) in assembly_summary_*.txt
C_ACC, C_BIOPROJECT, C_BIOSAMPLE = 0, 1, 2
C_REFSEQ_CAT, C_TAXID, C_SPECIES_TAXID, C_ORGANISM = 4, 5, 6, 7
C_INFRA, C_ISOLATE, C_VERSION_STATUS, C_LEVEL = 8, 9, 10, 11
C_RELDATE, C_ASMNAME, C_SUBMITTER = 14, 15, 16
C_FTP, C_EXCLUDED, C_TYPEMAT = 19, 20, 21
C_GROUP = 24
C_GENOME_SIZE, C_GC, C_SCAFFOLD_COUNT, C_CONTIG_COUNT = 25, 27, 29, 30

# MAGICC is bacteria/archaea only -> everything else must be excluded
PROKARYOTE_GROUPS = {"bacteria", "archaea"}

_STRAIN_STRIP = re.compile(r"^(strain=|type strain:\s*)", re.I)
_NONALNUM = re.compile(r"[^a-z0-9]+")
# Strain designations that carry no discriminating information
UNINFORMATIVE = {
    "", "na", "none", "unknown", "missing", "notapplicable", "notcollected",
    "notprovided", "null", "nostrain", "unnamed", "wildtype", "wt", "typestrain",
}


def norm_strain(infra: str, isolate: str) -> str:
    """Normalise a strain designation to a comparison key."""
    raw = infra if infra and infra != "na" else isolate
    if not raw or raw == "na":
        return ""
    raw = _STRAIN_STRIP.sub("", raw.strip())
    key = _NONALNUM.sub("", raw.lower())
    if key in UNINFORMATIVE or len(key) < 2:
        return ""
    return key


def load() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    excl_counter: dict[str, int] = defaultdict(int)
    group_counter: dict[str, int] = defaultdict(int)
    contaminated_by_level: dict[str, int] = defaultdict(int)
    n_lines = 0
    with SUMMARY.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_lines += 1
            p = line.rstrip("\n").split("\t")
            if len(p) < 31:
                continue
            level = p[C_LEVEL]
            excluded = p[C_EXCLUDED]
            if "contaminated" in excluded:
                contaminated_by_level[level] += 1
                for tok in excluded.split(";"):
                    excl_counter[tok.strip()] += 1
            if level not in COMPLETE_LEVELS and level not in DRAFT_LEVELS:
                continue
            if p[C_VERSION_STATUS] != "latest":
                continue
            group_counter[p[C_GROUP]] += 1
            rows.append(
                {
                    "accession": p[C_ACC],
                    "biosample": p[C_BIOSAMPLE],
                    "species_taxid": p[C_SPECIES_TAXID],
                    "organism": p[C_ORGANISM],
                    "strain_key": norm_strain(p[C_INFRA], p[C_ISOLATE]),
                    "strain_raw": p[C_INFRA] if p[C_INFRA] != "na" else p[C_ISOLATE],
                    "group": p[C_GROUP],
                    "level": level,
                    "release_date": p[C_RELDATE],
                    "asm_name": p[C_ASMNAME],
                    "submitter": p[C_SUBMITTER],
                    "ftp_path": p[C_FTP],
                    "excluded_from_refseq": excluded,
                    "type_material": p[C_TYPEMAT],
                    "genome_size": p[C_GENOME_SIZE],
                    "gc_percent": p[C_GC],
                    "scaffold_count": p[C_SCAFFOLD_COUNT],
                    "contig_count": p[C_CONTIG_COUNT],
                    "refseq_category": p[C_REFSEQ_CAT],
                }
            )
    stats = {
        "summary_file": str(SUMMARY),
        "groups_seen": dict(sorted(group_counter.items(), key=lambda kv: -kv[1])[:20]),
        "data_lines_read": n_lines,
        "assemblies_kept_latest_and_leveled": len(rows),
        "contaminated_flag_by_level": dict(contaminated_by_level),
        "contaminated_flag_total": sum(contaminated_by_level.values()),
        "exclusion_tokens_co_occurring_with_contaminated": dict(
            sorted(excl_counter.items(), key=lambda kv: -kv[1])[:25]
        ),
    }
    return rows, stats


def size_ratio(draft: dict, comp: dict) -> float | None:
    try:
        d, c = int(draft["genome_size"]), int(comp["genome_size"])
        if c <= 0:
            return None
        return d / c
    except (ValueError, KeyError):
        return None


def build_pairs(groups: dict[str, list[dict]], tier: str) -> list[dict]:
    """For each group emit one row per (draft, best complete) pair."""
    out: list[dict] = []
    for key, members in groups.items():
        comps = [m for m in members if m["level"] in COMPLETE_LEVELS]
        drafts = [m for m in members if m["level"] in DRAFT_LEVELS]
        if not comps or not drafts:
            continue
        # Prefer 'Complete Genome' over 'Chromosome'; then fewest scaffolds; then newest
        comps.sort(
            key=lambda m: (
                m["level"] != "Complete Genome",
                int(m["scaffold_count"]) if m["scaffold_count"].isdigit() else 10**9,
                m["release_date"],
            )
        )
        ref = comps[0]
        for d in drafts:
            out.append(
                {
                    "tier": tier,
                    "group_key": key,
                    "draft_accession": d["accession"],
                    "complete_accession": ref["accession"],
                    "organism": d["organism"],
                    "group": d["group"],
                    "species_taxid": d["species_taxid"],
                    "strain": d["strain_raw"],
                    "draft_level": d["level"],
                    "complete_level": ref["level"],
                    "draft_biosample": d["biosample"],
                    "complete_biosample": ref["biosample"],
                    "draft_contigs": d["contig_count"],
                    "complete_contigs": ref["contig_count"],
                    "draft_bp": d["genome_size"],
                    "complete_bp": ref["genome_size"],
                    "size_ratio_draft_over_complete": (
                        f"{size_ratio(d, ref):.4f}" if size_ratio(d, ref) else "NA"
                    ),
                    "draft_gc": d["gc_percent"],
                    "complete_gc": ref["gc_percent"],
                    "draft_excluded_from_refseq": d["excluded_from_refseq"],
                    "complete_excluded_from_refseq": ref["excluded_from_refseq"],
                    "draft_submitter": d["submitter"],
                    "complete_submitter": ref["submitter"],
                    "draft_release_date": d["release_date"],
                    "complete_release_date": ref["release_date"],
                    "n_complete_in_group": len(comps),
                    "n_draft_in_group": len(drafts),
                    "draft_ftp": d["ftp_path"],
                    "complete_ftp": ref["ftp_path"],
                }
            )
    return out


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def describe(rows: list[dict], label: str) -> dict:
    ratios = [
        float(r["size_ratio_draft_over_complete"])
        for r in rows
        if r["size_ratio_draft_over_complete"] != "NA"
    ]
    ratios.sort()

    def q(p: float) -> float | None:
        if not ratios:
            return None
        return round(ratios[min(len(ratios) - 1, int(p * len(ratios)))], 4)

    species = {r["species_taxid"] for r in rows}
    return {
        "label": label,
        "n_pairs": len(rows),
        "n_unique_drafts": len({r["draft_accession"] for r in rows}),
        "n_unique_complete_refs": len({r["complete_accession"] for r in rows}),
        "n_unique_species_taxid": len(species),
        "size_ratio_quantiles": {"p05": q(0.05), "p25": q(0.25), "p50": q(0.50),
                                  "p75": q(0.75), "p95": q(0.95)},
        "n_pairs_ratio_0.80_1.20": sum(1 for x in ratios if 0.80 <= x <= 1.20),
        "n_pairs_ratio_gt_1.20": sum(1 for x in ratios if x > 1.20),
        "n_pairs_ratio_lt_0.80": sum(1 for x in ratios if x < 0.80),
        "n_pairs_prokaryote_only": sum(
            1 for r in rows if r["group"] in PROKARYOTE_GROUPS
        ),
        "n_pairs_prokaryote_and_ratio_0.80_1.20": sum(
            1 for r in rows
            if r["group"] in PROKARYOTE_GROUPS
            and r["size_ratio_draft_over_complete"] != "NA"
            and 0.80 <= float(r["size_ratio_draft_over_complete"]) <= 1.20
        ),
        "n_drafts_flagged_contaminated": sum(
            1 for r in rows if "contaminated" in r["draft_excluded_from_refseq"]
        ),
        "n_drafts_derived_from_metagenome": sum(
            1 for r in rows if "derived from metagenome" in r["draft_excluded_from_refseq"]
        ),
        "n_drafts_derived_from_single_cell": sum(
            1 for r in rows if "derived from single cell" in r["draft_excluded_from_refseq"]
        ),
        "top_species": sorted(
            {
                s: sum(1 for r in rows if r["species_taxid"] == s) for s in species
            }.items(),
            key=lambda kv: -kv[1],
        )[:15],
    }


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"[130] reading {SUMMARY} ...", flush=True)
    rows, stats = load()
    print(f"[130] {stats['data_lines_read']:,} data lines; "
          f"{len(rows):,} latest complete/draft assemblies", flush=True)

    # --- TIER 1: same BioSample -------------------------------------------------
    by_bs: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["biosample"] and r["biosample"] != "na":
            by_bs[r["biosample"]].append(r)
    tier1 = build_pairs(by_bs, "same_biosample")
    print(f"[130] TIER 1 same-BioSample pairs: {len(tier1):,}", flush=True)

    # --- TIER 2: same species_taxid + same normalised strain, different biosample
    t1_bs = {r["draft_biosample"] for r in tier1} | {r["complete_biosample"] for r in tier1}
    by_strain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["strain_key"]:
            by_strain[f"{r['species_taxid']}|{r['strain_key']}"].append(r)
    tier2_all = build_pairs(by_strain, "same_strain")
    tier2 = [
        r for r in tier2_all
        if r["draft_biosample"] != r["complete_biosample"]
        and not (r["draft_biosample"] in t1_bs and r["complete_biosample"] in t1_bs)
    ]
    print(f"[130] TIER 2 same-strain pairs (excl. tier-1 biosamples): {len(tier2):,}",
          flush=True)

    write_tsv(OUTDIR / "tier1_same_biosample_pairs.tsv", tier1)
    write_tsv(OUTDIR / "tier2_same_strain_pairs.tsv", tier2)

    summary = {
        "script": "scripts/130_ncbi_draft_complete_pairs.py",
        "input_stats": stats,
        "tier1": describe(tier1, "same BioSample (strongest match)"),
        "tier2": describe(tier2, "same species_taxid + same strain designation"),
        "definitions": {
            "complete_levels": sorted(COMPLETE_LEVELS),
            "draft_levels": sorted(DRAFT_LEVELS),
            "tier1": "identical BioSample accession => same physical DNA isolate",
            "tier2": "identical species_taxid and normalised strain string, different BioSample",
            "ground_truth_derivation": (
                "completeness = reference bases covered by draft alignment / reference "
                "total bases; contamination = draft bases not alignable to the reference "
                "(minus accessory/plasmid content) / reference total bases. Requires "
                "whole-genome alignment (minimap2/nucmer) of draft vs complete."
            ),
        },
    }
    (OUTDIR / "pairs_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("tier1", "tier2")}, indent=2))
    print(f"[130] wrote {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
