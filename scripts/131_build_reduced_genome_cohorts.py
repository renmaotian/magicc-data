#!/usr/bin/env python3
"""
WS3.3 / new-priority — build the cohorts needed to test the hypothesis:

    "MAGICC systematically over-calls contamination and under-calls completeness on
     reduced genomes, and this explains the reviewer's observed MAGICC-vs-CheckM2
     disagreements for CAG-557, UMGS1491, Caccenecus, HGM10766 and Faecimonas."

Comparator is GTDB's **published CheckM2** values (`checkm2_completeness`,
`checkm2_contamination` in data/gtdb/{bac120,ar53}_metadata.tsv.gz), which exist for all
732,475 GTDB genomes.  UHGG cannot serve here: it ships CheckM **v1**, so a
MAGICC-vs-UHGG comparison would not be a CheckM2 comparison at all.

Three cohorts
  R  reviewer_genera   all GTDB genomes of the five named genera (n = 298)
  S  size_stratified   MAGs sampled across genome-size bins (<1, 1-2, 2-3, 3-5, >5 Mbp)
  C  control_genera    well-populated NON-reduced genera, so the contrast is not
                       confined to the regime where failure is expected

Every row records TRAIN/VAL/TEST membership so leakage can be reported, and the NCBI
FTP path so acquisition is a scripted fetch.

Read-only on data/gtdb/, data/ncbi/, data/splits/.  No network.  Single-threaded.
Output: results/revision/real_data/reduced_genome/cohorts.tsv (+ cohorts_summary.json)
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/path/to/magicc")
GTDB = [ROOT / "data/gtdb/bac120_metadata.tsv.gz", ROOT / "data/gtdb/ar53_metadata.tsv.gz"]
ASM_SUMMARY = ROOT / "data/ncbi/assembly_summary_genbank.txt"
SPLITS = ROOT / "data/splits"
OUTDIR = ROOT / "results/revision/real_data/reduced_genome"

REVIEWER_GENERA = ["CAG-557", "UMGS1491", "Caccenecus", "HGM10766", "Faecimonas"]

# Non-reduced, well-populated gut/environmental MAG genera used as controls.
# Chosen a priori for genome size >> 2 Mbp and high MAG counts, before looking at results.
CONTROL_GENERA = [
    "Bacteroides", "Prevotella", "Faecalibacterium", "Blautia",
    "Escherichia", "Alistipes", "Ruminococcus", "Bifidobacterium",
]

SIZE_BINS = [(0, 1_000_000), (1_000_000, 2_000_000), (2_000_000, 3_000_000),
             (3_000_000, 5_000_000), (5_000_000, 10**12)]
SIZE_BIN_LABELS = ["<1Mb", "1-2Mb", "2-3Mb", "3-5Mb", ">5Mb"]
N_PER_SIZE_BIN = 150
N_PER_CONTROL_GENUS = 40
SEED = 13100


def load_splits() -> dict[str, str]:
    """base accession (no version, no GB_/RS_ prefix) -> split name."""
    out: dict[str, str] = {}
    for name in ("train", "val", "test"):
        path = SPLITS / f"{name}_genomes.tsv"
        with path.open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                for key in ("ncbi_accession", "gcf_accession", "gtdb_accession"):
                    val = (row.get(key) or "").strip()
                    if val:
                        out[re.sub(r"^(GB_|RS_)", "", val).split(".")[0]] = name
    return out


def load_ftp() -> dict[str, str]:
    """accession (with version) -> ftp_path, plus versionless fallback."""
    exact: dict[str, str] = {}
    loose: dict[str, str] = {}
    with ASM_SUMMARY.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 21 or not p[19] or p[19] == "na":
                continue
            exact[p[0]] = p[19]
            loose.setdefault(p[0].split(".")[0], p[19])
    return {"exact": exact, "loose": loose}


def genus_of(tax: str) -> str:
    m = re.search(r";g__([^;]*)", tax)
    return m.group(1) if m else ""


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    ftp = load_ftp()
    print(f"[131] splits: {len(splits):,} accessions; ftp paths: {len(ftp['exact']):,}", flush=True)

    genomes: list[dict] = []
    for path in GTDB:
        with gzip.open(path, "rt") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                try:
                    comp = float(row["checkm2_completeness"])
                    cont = float(row["checkm2_contamination"])
                    size = int(row["genome_size"])
                except (ValueError, KeyError):
                    continue
                acc = row.get("ncbi_genbank_assembly_accession") or ""
                if not acc or acc == "none":
                    acc = re.sub(r"^(GB_|RS_)", "", row["accession"])
                genomes.append({
                    "gtdb_accession": row["accession"],
                    "ncbi_accession": acc,
                    "genus": genus_of(row["gtdb_taxonomy"]),
                    "gtdb_taxonomy": row["gtdb_taxonomy"],
                    "domain": row["gtdb_taxonomy"].split(";")[0].replace("d__", ""),
                    "genome_category": row.get("ncbi_genome_category", ""),
                    "assembly_level": row.get("ncbi_assembly_level", ""),
                    "checkm2_completeness": comp,
                    "checkm2_contamination": cont,
                    "checkm2_model": row.get("checkm2_model", ""),
                    "genome_size": size,
                    "contig_count": row.get("contig_count", ""),
                    "gc_percentage": row.get("gc_percentage", ""),
                    "n50_contigs": row.get("n50_contigs", ""),
                    "split": splits.get(acc.split(".")[0], "none"),
                    "ftp_path": ftp["exact"].get(acc) or ftp["loose"].get(acc.split(".")[0], ""),
                })
    print(f"[131] GTDB genomes with CheckM2 values: {len(genomes):,}", flush=True)

    rng = random.Random(SEED)
    selected: list[dict] = []
    seen: set[str] = set()

    def add(g: dict, cohort: str, stratum: str) -> bool:
        if g["gtdb_accession"] in seen or not g["ftp_path"]:
            return False
        seen.add(g["gtdb_accession"])
        r = dict(g)
        r["cohort"] = cohort
        r["stratum"] = stratum
        selected.append(r)
        return True

    # --- Cohort R: reviewer genera (everything, no sampling) -------------------
    missing_ftp = Counter()
    for g in genomes:
        if g["genus"] in REVIEWER_GENERA:
            if not add(g, "reviewer_genera", g["genus"]):
                missing_ftp[g["genus"]] += 1
    print(f"[131] cohort R reviewer_genera: {sum(1 for s in selected if s['cohort']=='reviewer_genera')} "
          f"(dropped for missing ftp: {dict(missing_ftp)})", flush=True)

    # --- Cohort S: size-stratified MAGs ---------------------------------------
    # MAGs only, in-domain for MAGICC (completeness >= 50), not already selected.
    pool = [g for g in genomes
            if g["genome_category"] == "derived from metagenome"
            and g["checkm2_completeness"] >= 50
            and g["checkm2_contamination"] <= 10
            and g["ftp_path"]
            and g["gtdb_accession"] not in seen]
    by_bin: dict[str, list[dict]] = defaultdict(list)
    for g in pool:
        for (lo, hi), lab in zip(SIZE_BINS, SIZE_BIN_LABELS):
            if lo <= g["genome_size"] < hi:
                by_bin[lab].append(g)
                break
    for lab in SIZE_BIN_LABELS:
        cand = by_bin[lab]
        cand.sort(key=lambda g: g["gtdb_accession"])          # deterministic order
        rng.shuffle(cand)
        n = 0
        for g in cand:
            if n >= N_PER_SIZE_BIN:
                break
            if add(g, "size_stratified", lab):
                n += 1
        print(f"[131]   size bin {lab:6} pool {len(cand):>7,} -> selected {n}", flush=True)

    # --- Cohort C: non-reduced control genera (MAGs) --------------------------
    for gen in CONTROL_GENERA:
        cand = [g for g in genomes
                if g["genus"] == gen
                and g["genome_category"] == "derived from metagenome"
                and g["checkm2_completeness"] >= 50
                and g["checkm2_contamination"] <= 10
                and g["ftp_path"]
                and g["gtdb_accession"] not in seen]
        cand.sort(key=lambda g: g["gtdb_accession"])
        rng.shuffle(cand)
        n = 0
        for g in cand:
            if n >= N_PER_CONTROL_GENUS:
                break
            if add(g, "control_genera", gen):
                n += 1
        print(f"[131]   control genus {gen:18} pool {len(cand):>6,} -> selected {n}", flush=True)

    cols = ["cohort", "stratum", "gtdb_accession", "ncbi_accession", "genus", "domain",
            "genome_category", "assembly_level", "checkm2_completeness",
            "checkm2_contamination", "checkm2_model", "genome_size", "contig_count",
            "n50_contigs", "gc_percentage", "split", "gtdb_taxonomy", "ftp_path"]
    out = OUTDIR / "cohorts.tsv"
    with out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in selected:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    summary = {
        "script": "scripts/131_build_reduced_genome_cohorts.py",
        "seed": SEED,
        "comparator": "GTDB published checkm2_completeness / checkm2_contamination",
        "why_not_uhgg": "UHGG v2.0.2 ships CheckM v1 values, not CheckM2",
        "n_total": len(selected),
        "by_cohort": dict(Counter(r["cohort"] for r in selected)),
        "by_stratum": dict(Counter(f"{r['cohort']}:{r['stratum']}" for r in selected)),
        "by_split": dict(Counter(r["split"] for r in selected)),
        "by_split_and_cohort": {
            c: dict(Counter(r["split"] for r in selected if r["cohort"] == c))
            for c in {r["cohort"] for r in selected}
        },
        "reviewer_genera": REVIEWER_GENERA,
        "control_genera": CONTROL_GENERA,
        "size_bins": SIZE_BIN_LABELS,
        "estimated_download_gb": round(sum(r["genome_size"] for r in selected) * 0.29 / 1e9, 2),
    }
    (OUTDIR / "cohorts_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[131] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
