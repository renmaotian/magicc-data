#!/usr/bin/env python3
"""
WS7.5 / WS7.6 / WS7.7 — assemble the public reproducibility artifact bundle.

Addresses reviewer comments R1-M8 (full reproducibility artifacts in the repo),
R1-m14 (release the selected 9-mer list with domain assignment, prevalence and
selection criteria) and R1-m15 (release the synthetic generator with all random
seeds and per-sample metadata), plus editorial requirements E5/E6.

Outputs (all under results/revision/reproducibility/release/):

  kmers/selected_9mers.tsv          9,249 canonical 9-mers, domain assignment,
                                    bacterial + archaeal prevalence, rank,
                                    selection reason                    [WS7.5]
  kmers/kmer_selection_criteria.md  how the list was produced, verbatim
                                    thresholds and genome provenance    [WS7.5]
  splits/                           train/val/test accession lists + a
                                    split-composition summary           [WS7.6]
  benchmarks/benchmark_inventory.tsv   one row per benchmark set: n genomes,
                                    generator script, seed scheme, retained or
                                    superseded, available predictions   [WS7.6]
  generator/seed_provenance.tsv     one row per retained set giving the exact
                                    RNG construction used for every sample,
                                    so any sample can be regenerated    [WS7.7]
  generator/per_sample_metadata_index.tsv  which per-sample columns exist for
                                    which set, and where                [WS7.7]
  normalization/                    the exact normalization parameters used at
                                    inference, with checksums           [WS7.6]
  MANIFEST.sha256                   checksums of every released artifact
  RELEASE_MANIFEST.json             machine-readable index
  README.md                         what is here and how to use it

This script EXTENDS results/revision/provenance/sha256_manifest.txt (WS1.4)
rather than duplicating it: large inputs already hashed there are referenced,
not re-hashed or re-copied.

Idempotent and resumable: re-running overwrites the generated files in place
and re-verifies checksums.  No heavy compute, no network.

Usage:
    python scripts/112_release_artifacts.py [--project-dir PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Static provenance facts, all verified against the generator sources.
# Each entry records how a benchmark set's per-sample RNG was constructed, so
# any individual genome can be regenerated bit-exactly.  (WS7.7 / R1-m15)
# --------------------------------------------------------------------------
SEED_PROVENANCE = [
    # set, status, generator script, base seed, per-sample RNG construction, notes
    dict(set_name="set_C_clean", status="retained (primary)",
         generator="scripts/73_generate_clean_cd_benchmarks.py",
         base_seed="7300000",
         design_rng="np.random.default_rng(7300000)  -> 1000 target completeness ~U[0.50,1.00), "
                    "1000 target contamination ~U[0,100)",
         per_sample_rng="np.random.default_rng(7300000 + 1000*ref_index + replicate)",
         per_sample_seed_recorded="YES — column `seed` of generation_metadata.tsv",
         n_samples=1000,
         notes="100 test-split Patescibacteriota references x 10 simulations. "
               "Reference draw: np.random.default_rng(7001).choice(n, 100, replace=False) "
               "over candidates sorted by gtdb_accession."),
    dict(set_name="set_D_clean", status="retained (primary)",
         generator="scripts/73_generate_clean_cd_benchmarks.py",
         base_seed="7400000",
         design_rng="np.random.default_rng(7400000)  -> 1000 target completeness ~U[0.50,1.00), "
                    "1000 target contamination ~U[0,100)",
         per_sample_rng="np.random.default_rng(7400000 + 1000*ref_index + replicate)",
         per_sample_seed_recorded="YES — column `seed` of generation_metadata.tsv",
         n_samples=1000,
         notes="100 test-split archaeal references x 10 simulations. "
               "Reference draw: np.random.default_rng(7002).choice(n, 100, replace=False) "
               "over candidates sorted by gtdb_accession."),
    dict(set_name="set_A_v2", status="retained (primary)",
         generator="scripts/34_generate_finished_benchmarks.py::generate_completeness_set",
         base_seed="300",
         design_rng="np.random.default_rng(300 + int(completeness*1000)) per completeness level",
         per_sample_rng="np.random.default_rng(300 + idx)",
         per_sample_seed_recorded="DERIVABLE — 300 + row index of metadata.tsv",
         n_samples=1000,
         notes="Completeness gradient, 0 % contamination; dominants are finished "
               "(Complete Genome / Chromosome) test-split genomes."),
    dict(set_name="set_B_v2", status="retained (primary)",
         generator="scripts/34_generate_finished_benchmarks.py::generate_contamination_set",
         base_seed="400",
         design_rng="np.random.default_rng(400 + contamination + 5000) per contamination level",
         per_sample_rng="np.random.default_rng(400 + idx)",
         per_sample_seed_recorded="DERIVABLE — 400 + row index of metadata.tsv",
         n_samples=1000,
         notes="Contamination gradient at 100 % completeness."),
    dict(set_name="set_E", status="retained (primary)",
         generator="scripts/34_generate_finished_benchmarks.py::generate_set_e",
         base_seed="500",
         design_rng="np.random.default_rng(500)",
         per_sample_rng="np.random.default_rng(500 + idx)",
         per_sample_seed_recorded="DERIVABLE — 500 + row index of metadata.tsv",
         n_samples=1000,
         notes="Uniform completeness x contamination. Predates the "
               "contaminant_bp <= dominant_retained_bp cap (added 2026-03-14, commit c5a9b95), "
               "so 132/1000 samples have contamination% > completeness% and lie outside the "
               "V5 training domain; see protocol section 4.4a."),
    dict(set_name="motivating_v2/set_A", status="retained (secondary)",
         generator="scripts/34_generate_finished_benchmarks.py::generate_completeness_set",
         base_seed="100",
         design_rng="np.random.default_rng(100 + int(completeness*1000))",
         per_sample_rng="np.random.default_rng(100 + idx)",
         per_sample_seed_recorded="DERIVABLE — 100 + row index of metadata.tsv",
         n_samples=1000, notes="Motivating completeness gradient."),
    dict(set_name="motivating_v2/set_B", status="retained (secondary)",
         generator="scripts/34_generate_finished_benchmarks.py::generate_contamination_set",
         base_seed="200",
         design_rng="np.random.default_rng(200 + contamination + 5000)",
         per_sample_rng="np.random.default_rng(200 + idx)",
         per_sample_seed_recorded="DERIVABLE — 200 + row index of metadata.tsv",
         n_samples=1000, notes="Motivating contamination gradient."),
    dict(set_name="motivating_v2/set_C", status="retained (secondary)",
         generator="scripts/41_generate_motivating_set_c.py",
         base_seed="300",
         design_rng="np.random.default_rng(300)",
         per_sample_rng="np.random.default_rng(300 + idx)",
         per_sample_seed_recorded="DERIVABLE — 300 + row index of metadata.tsv",
         n_samples=1000,
         notes="Non-redundant contamination emphasis. Also predates the "
               "contamination cap (122/1000 out-of-domain samples)."),
    dict(set_name="motivating/set_A", status="retained (legacy, superseded by motivating_v2)",
         generator="scripts/31_motivating_benchmark_generate.py",
         base_seed="12345",
         design_rng="np.random.default_rng(12345 + int(completeness*1000) + 7777)",
         per_sample_rng="np.random.default_rng(12345 + idx + 500000)",
         per_sample_seed_recorded="DERIVABLE — 12345 + row index + 500000",
         n_samples=600, notes="Superseded by motivating_v2/set_A (finished-genome dominants)."),
    dict(set_name="motivating/set_B", status="retained (legacy, superseded by motivating_v2)",
         generator="scripts/31_motivating_benchmark_generate.py",
         base_seed="12345",
         design_rng="np.random.default_rng(12345 + contamination + 9999)",
         per_sample_rng="np.random.default_rng(12345 + idx + 600000)",
         per_sample_seed_recorded="DERIVABLE — 12345 + row index + 600000",
         n_samples=1100, notes="Superseded by motivating_v2/set_B."),
    dict(set_name="set_A", status="retained (legacy, superseded by set_A_v2)",
         generator="scripts/25_benchmark_generate.py::generate_set_a",
         base_seed="42",
         design_rng="np.random.default_rng(42 + int(completeness*1000))",
         per_sample_rng="np.random.default_rng(42 + idx + 100000)",
         per_sample_seed_recorded="DERIVABLE — 42 + row index + 100000",
         n_samples=600, notes="Superseded by set_A_v2."),
    dict(set_name="set_B", status="retained (legacy, superseded by set_B_v2)",
         generator="scripts/25_benchmark_generate.py::generate_set_b",
         base_seed="42",
         design_rng="np.random.default_rng(42 + contamination + 5000)",
         per_sample_rng="np.random.default_rng(42 + idx + 200000)",
         per_sample_seed_recorded="DERIVABLE — 42 + row index + 200000",
         n_samples=600, notes="Superseded by set_B_v2."),
    dict(set_name="set_C", status="WITHDRAWN — training-data leakage",
         generator="scripts/25_benchmark_generate.py::generate_set_c",
         base_seed="42",
         design_rng="np.random.default_rng(42 + 300000)",
         per_sample_rng="np.random.default_rng(42 + idx + 300000)",
         per_sample_seed_recorded="DERIVABLE — 42 + row index + 300000",
         n_samples=1000,
         notes="WITHDRAWN. 1000/1000 dominants are TRAIN-split genomes. "
               "Replaced by set_C_clean. Released only so the leakage is auditable."),
    dict(set_name="set_D", status="WITHDRAWN — training-data leakage",
         generator="scripts/25_benchmark_generate.py::generate_set_d",
         base_seed="42",
         design_rng="np.random.default_rng(42 + 400000)",
         per_sample_rng="np.random.default_rng(42 + idx + 400000)",
         per_sample_seed_recorded="DERIVABLE — 42 + row index + 400000",
         n_samples=1000,
         notes="WITHDRAWN. 796 TRAIN / 107 VAL / 97 TEST dominants. "
               "Replaced by set_D_clean."),
]

RELEASE_SUBDIRS = ("kmers", "splits", "benchmarks", "generator", "normalization")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path):
    """Minimal dependency-free TSV reader -> (header, rows)."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    return header, rows


# ==========================================================================
# WS7.5 — the selected 9-mer list
# ==========================================================================
def build_kmer_release(project: Path, out: Path) -> dict:
    ks_dir = project / "data" / "kmer_selection"
    stats = json.loads((ks_dir / "kmer_selection_stats.json").read_text())

    ann_header, ann_rows = read_tsv(ks_dir / "selected_kmers_annotated.tsv")
    assert ann_header == ["kmer", "source", "bacterial_prevalence", "archaeal_prevalence"], ann_header

    plain = [ln.strip() for ln in (ks_dir / "selected_kmers.txt").read_text().splitlines() if ln.strip()]
    assert len(plain) == len(ann_rows) == stats["n_final_merged"], \
        f"k-mer count mismatch: plain={len(plain)} annotated={len(ann_rows)} stats={stats['n_final_merged']}"
    assert [r[0] for r in ann_rows] == plain, "annotated k-mer order differs from selected_kmers.txt"

    # The selection rule is a RANK cutoff (`head(9000)` / `head(1000)` over the
    # prevalence tables sorted descending), not a prevalence-value threshold.
    # Prevalence ties therefore straddle the boundary: 9,008 bacterial 9-mers
    # reach the cutoff prevalence but only 9,000 were admitted.  The authoritative
    # record of what was actually selected is the `source` column, so that is
    # what is released; the value thresholds are reported as informational.
    bact_cut = stats["bacterial_prevalence_cutoff"]
    arch_cut = stats["archaeal_prevalence_cutoff"]
    SOURCE_TO_DOMAIN = {"bacterial": "bacterial", "archaeal": "archaeal",
                        "bacterial+archaeal": "shared"}
    REASON = {
        "bacterial": f"selected in the top-9,000 bacterial prevalence ranking (cutoff prevalence {bact_cut}/1000)",
        "archaeal": f"selected in the top-1,000 archaeal prevalence ranking (cutoff prevalence {arch_cut}/1000)",
        "shared": "selected in BOTH the top-9,000 bacterial and the top-1,000 archaeal ranking",
    }

    n_bact = n_arch = n_shared = 0
    n_tie_at_bact_cut = n_tie_at_arch_cut = 0
    lines = ["\t".join([
        "feature_index", "kmer", "domain_assignment",
        "bacterial_prevalence_per_1000", "archaeal_prevalence_per_1000",
        "at_bacterial_rank_cutoff_prevalence", "at_archaeal_rank_cutoff_prevalence",
        "selection_reason",
    ])]
    for i, (kmer, source, bp, ap) in enumerate(ann_rows):
        domain = SOURCE_TO_DOMAIN.get(source)
        if domain is None:
            raise AssertionError(f"unexpected source value {source!r} for k-mer {kmer}")
        n_bact += domain in ("bacterial", "shared")
        n_arch += domain in ("archaeal", "shared")
        n_shared += domain == "shared"
        tie_b = int(bp) == bact_cut
        tie_a = int(ap) == arch_cut
        n_tie_at_bact_cut += tie_b
        n_tie_at_arch_cut += tie_a
        lines.append("\t".join([str(i), kmer, domain, bp, ap,
                                "1" if tie_b else "0", "1" if tie_a else "0", REASON[domain]]))

    (out / "kmers" / "selected_9mers.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Cross-check against the recorded selection statistics.
    assert n_bact == stats["n_bacterial_selected"], (n_bact, stats["n_bacterial_selected"])
    assert n_arch == stats["n_archaeal_selected"], (n_arch, stats["n_archaeal_selected"])
    assert n_shared == stats["n_overlap"], (n_shared, stats["n_overlap"])

    # How many k-mers in the FULL prevalence tables reach the cutoff value?  The
    # excess over the rank cutoff is the number excluded purely by tie-breaking.
    def n_at_or_above(path: Path, cutoff: int) -> int:
        hdr, rows = read_tsv(path)
        assert hdr == ["kmer", "prevalence"], hdr
        return sum(1 for r in rows if int(r[1]) >= cutoff)

    n_bact_reaching = n_at_or_above(ks_dir / "bacterial_kmer_prevalence.tsv", bact_cut)
    n_arch_reaching = n_at_or_above(ks_dir / "archaeal_kmer_prevalence.tsv", arch_cut)
    n_bact_pass = n_bact
    n_arch_pass = n_arch
    n_both = n_shared

    criteria = f"""# Selected 9-mer feature set — selection criteria and provenance  (WS7.5, R1-m14)

MAGICC's k-mer branch consumes a fixed vector of **{stats['n_final_merged']:,} canonical 9-mers**.
`selected_9mers.tsv` in this directory is the authoritative release of that list.

## Feature order is load-bearing

Column `feature_index` is the position of the k-mer in the ONNX model input
`kmer_features [batch, {stats['n_final_merged']}]`. It is the line order of
`data/kmer_selection/selected_kmers.txt` (and of the copy shipped inside the
Python package at `magicc/data/selected_kmers.txt`). Re-ordering the list
invalidates the model.

## How the list was built

1. **Reference genomes.** {stats['n_bacterial_selected'] and '1,000'} bacterial and 1,000 archaeal
   representative genomes were sampled **from the TRAIN split only**, seed 42
   (`data/kmer_selection/selected_bacterial_1000.tsv`,
   `selected_archaeal_1000.tsv`). No validation or test genome contributed to
   feature selection; this is audited in
   `results/revision/provenance/` (WS1.4).
2. **Single-copy core genes** were identified per domain with Prodigal + HMMER
   against `85_bcg.hmm` (bacteria) and `uacg.hmm` (archaea)
   (`scripts/09_identify_core_genes.py`). Feature selection is therefore
   annotation-dependent; **inference is not** (see R1-m5).
3. **Canonical 9-mer counting** over the core-gene nucleotide sequences
   (`scripts/10_count_9mers.py`), giving 4^9/2 = 131,072 canonical 9-mers per
   domain.
4. **Prevalence** = the number of the 1,000 domain reference genomes in which
   the k-mer occurs at least once (range 0–1000).
5. **Selection** (`scripts/11_select_kmers.py`): the prevalence tables were sorted
   descending and the top **{stats['n_bacterial_selected']:,} bacterial** and top
   **{stats['n_archaeal_selected']:,} archaeal** k-mers were taken (`head(N)`) and merged.
   **{stats['n_overlap']:,}** k-mers are in both lists, so the union is
   **{stats['n_final_merged']:,}** unique canonical 9-mers.

## The cutoff is a RANK cutoff, not a prevalence-value threshold

This matters for exact reproduction and is stated explicitly rather than left
implicit. The selection takes a fixed *number* of k-mers, so k-mers tied at the
cutoff prevalence straddle the boundary and are separated only by their position
in the sorted table:

| Domain | k-mers taken (rank cutoff) | prevalence range | k-mers in the full table reaching the cutoff prevalence | excluded purely by tie-breaking |
|---|---|---|---|---|
| Bacteria | {stats['n_bacterial_selected']:,} | {stats['bacterial_prevalence_cutoff']} – {stats['bacterial_max_prevalence']} | {n_bact_reaching:,} | {n_bact_reaching - stats['n_bacterial_selected']:,} |
| Archaea  | {stats['n_archaeal_selected']:,} | {stats['archaeal_prevalence_cutoff']} – {stats['archaeal_max_prevalence']} | {n_arch_reaching:,} | {n_arch_reaching - stats['n_archaeal_selected']:,} |
| **Union (model input)** | **{stats['n_final_merged']:,}** | — | — | — |

{n_tie_at_bact_cut:,} released k-mers sit exactly at the bacterial cutoff prevalence and
{n_tie_at_arch_cut:,} exactly at the archaeal cutoff (columns
`at_bacterial_rank_cutoff_prevalence` / `at_archaeal_rank_cutoff_prevalence`).
Reproducing the selection therefore requires the released prevalence tables, not
just the two threshold numbers. Domain assignment in `selected_9mers.tsv` comes
from the recorded selection provenance (`source` column of the original
`selected_kmers_annotated.tsv`), never from re-applying a value threshold.

Composition of the union: {stats['n_bact_only']:,} bacterial-only,
{stats['n_arch_only']:,} archaeal-only, {stats['n_overlap']:,} shared.

Note that **no 9-mer is present in all 1,000 genomes of either domain**
(`kmers_in_all_genomes = 0` in both `bacterial_kmer_stats.json` and
`archaeal_kmer_stats.json`); the highest bacterial prevalence is
{stats['bacterial_max_prevalence']}/1000 and the highest archaeal
{stats['archaeal_max_prevalence']}/1000. The features are therefore
*high-prevalence*, not universal.

## Transform applied before the model sees them

`log10(count + 1)`, then z-scoring with the stored per-feature mean/std in
`data/features/normalization_params.json` (released here under
`normalization/`). Both steps are in `magicc/normalization.py`.

## Files

| File | Content |
|---|---|
| `selected_9mers.tsv` | the released list: index, sequence, domain assignment, both prevalences, both threshold flags, selection reason |
| `bacterial_kmer_prevalence.tsv` | prevalence of **all** 131,072 canonical 9-mers in the 1,000 bacterial genomes |
| `archaeal_kmer_prevalence.tsv` | the same for archaea |
| `kmer_selection_stats.json` | the machine-readable selection record |
| `selected_bacterial_1000.tsv`, `selected_archaeal_1000.tsv` | the 2,000 feature-selection genomes, with accessions and taxonomy |

The two full prevalence tables are included so that the selection can be
**independently recomputed**, not merely inspected.
"""
    (out / "kmers" / "kmer_selection_criteria.md").write_text(criteria, encoding="utf-8")

    for name in ("bacterial_kmer_prevalence.tsv", "archaeal_kmer_prevalence.tsv",
                 "kmer_selection_stats.json",
                 "selected_bacterial_1000.tsv", "selected_archaeal_1000.tsv"):
        shutil.copy2(ks_dir / name, out / "kmers" / name)

    return dict(n_kmers=stats["n_final_merged"], n_bacterial=n_bact_pass,
                n_archaeal=n_arch_pass, n_shared=n_both,
                bacterial_prevalence_range=[stats["bacterial_prevalence_cutoff"],
                                            stats["bacterial_max_prevalence"]],
                archaeal_prevalence_range=[stats["archaeal_prevalence_cutoff"],
                                           stats["archaeal_max_prevalence"]])


# ==========================================================================
# WS7.6 — splits, benchmark metadata, normalization parameters
# ==========================================================================
def build_splits_release(project: Path, out: Path) -> dict:
    splits_dir = project / "data" / "splits"
    missing = set(ln.strip() for ln in (splits_dir / "missing_accessions.txt").read_text().splitlines()
                  if ln.strip())

    summary = {}
    total_gap = 0
    acc_sets, tsv_sets = {}, {}
    for split in ("train", "val", "test"):
        acc_p = splits_dir / f"{split}_accessions.txt"
        tsv_p = splits_dir / f"{split}_genomes.tsv"
        shutil.copy2(acc_p, out / "splits" / acc_p.name)

        # *_accessions.txt holds NCBI GenBank accessions (GCA_x.y); the genome
        # tables are keyed by GTDB accession (GB_GCA_x.y / RS_GCF_x.y) and carry
        # the NCBI accession in column `ncbi_accession`.  Compare on that.
        acc = set(ln.strip() for ln in acc_p.read_text().splitlines() if ln.strip())
        hdr, rows = read_tsv(tsv_p)
        ncbi_i = hdr.index("ncbi_accession")
        tsv_ncbi = set(r[ncbi_i] for r in rows)
        acc_sets[split], tsv_sets[split] = acc, tsv_ncbi

        gap = acc - tsv_ncbi
        total_gap += len(gap)
        # Every selected accession that has no genome row must be a recorded
        # download failure -- nothing may vanish silently.
        unexplained = gap - missing
        assert not unexplained, \
            f"{split}: {len(unexplained)} accessions in the split list have no genome row " \
            f"and are not in missing_accessions.txt, e.g. {sorted(unexplained)[:5]}"
        assert not (tsv_ncbi - acc), \
            f"{split}: genome table contains accessions absent from the split list"

        summary[split] = dict(
            n_accessions_selected=len(acc),
            n_genomes_available=len(tsv_ncbi),
            n_download_failures=len(gap),
            accession_list=f"splits/{acc_p.name}",
            full_table_in_repo=str(tsv_p.relative_to(project)),
            full_table_sha256=sha256_file(tsv_p),
        )
        # Keep the historical field name used elsewhere in the revision.
        summary[split]["n_genomes"] = len(tsv_ncbi)

    assert total_gap == len(missing), \
        f"download-failure accounting mismatch: {total_gap} gaps vs {len(missing)} recorded"

    # Disjointness, asserted here as well as in the WS1.4 audit so the released
    # bundle is self-validating.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for label, S in (("selected accessions", acc_sets), ("available genomes", tsv_sets)):
            inter = S[a] & S[b]
            assert not inter, f"{a}/{b} splits overlap on {len(inter)} {label}"
    summary["disjointness_verified"] = True
    summary["download_failures"] = dict(
        n=len(missing), file="splits/missing_accessions.txt",
        note="Accessions selected into a split whose genome download failed. They are "
             "absent from *_genomes.tsv and were never used for training, feature "
             "selection or benchmarking.")

    shutil.copy2(splits_dir / "split_statistics.json", out / "splits" / "split_statistics.json")
    shutil.copy2(splits_dir / "missing_accessions.txt", out / "splits" / "missing_accessions.txt")
    (out / "splits" / "README.md").write_text(
        "# Train / validation / test splits (WS7.6)\n\n"
        "| Split | accessions selected | genomes available | download failures |\n"
        "|---|---|---|---|\n"
        + "".join(f"| {s} | {summary[s]['n_accessions_selected']:,} | "
                  f"{summary[s]['n_genomes_available']:,} | {summary[s]['n_download_failures']} |\n"
                  for s in ("train", "val", "test"))
        + f"\nThe splits are stratified by phylum and **mutually disjoint** (verified here on\n"
          f"both the selected-accession sets and the available-genome sets).\n\n"
          f"`*_accessions.txt` lists NCBI GenBank accessions (`GCA_x.y`). The full tables\n"
          f"`data/splits/*_genomes.tsv` are keyed by GTDB accession (`GB_GCA_x.y` /\n"
          f"`RS_GCF_x.y`) and carry the NCBI accession in column `ncbi_accession`; that is\n"
          f"the join key. Naive string matching between the two conventions undercounts\n"
          f"overlap and is the exact mistake that hid the Set C/D leakage originally --\n"
          f"see `results/revision/provenance/README.md`.\n\n"
          f"{len(missing)} selected accessions ({total_gap} across the three splits) failed to\n"
          f"download and are listed in `missing_accessions.txt`. They have no row in\n"
          f"`*_genomes.tsv` and were never used for training, feature selection or\n"
          f"benchmarking. The counts used throughout the manuscript are the\n"
          f"**available-genome** counts: train {summary['train']['n_genomes']:,}, "
          f"val {summary['val']['n_genomes']:,}, test {summary['test']['n_genomes']:,}.\n",
        encoding="utf-8")
    return summary


def build_normalization_release(project: Path, out: Path) -> dict:
    src = project / "data" / "features" / "normalization_params.json"
    dst = out / "normalization" / "normalization_params.json"
    shutil.copy2(src, dst)
    params = json.loads(src.read_text())
    keys = sorted(params.keys())
    info = dict(path_in_repo=str(src.relative_to(project)),
                sha256=sha256_file(src),
                size_bytes=src.stat().st_size,
                top_level_keys=keys)
    # Also ship the copy that is inside the installed Python package, and prove
    # the two are identical -- a reviewer installing from PyPI gets these.
    pkg = project / "magicc" / "data" / "normalization_params.json"
    if pkg.is_file():
        info["packaged_copy"] = str(pkg.relative_to(project))
        info["packaged_copy_sha256"] = sha256_file(pkg)
        info["packaged_copy_identical"] = (info["packaged_copy_sha256"] == info["sha256"])
    (out / "normalization" / "normalization_README.md").write_text(
        "# Normalization parameters (WS7.6)\n\n"
        "`normalization_params.json` holds the per-feature mean/std applied at\n"
        "inference: k-mer counts are transformed with `log10(count + 1)` and then\n"
        "z-scored with these values; the 7 k-mer summary features are z-scored with\n"
        "their own stored scalers. Implementation: `magicc/normalization.py`.\n\n"
        "These parameters were fitted by streaming Welford statistics over the\n"
        "**training** synthesis stream only (reservoir sampling for the summary-feature\n"
        "scalers), so no validation or test genome influenced them.\n\n"
        f"SHA256 `{info['sha256']}`\n\n"
        "Top-level keys: " + ", ".join(f"`{k}`" for k in keys) + "\n",
        encoding="utf-8")
    return info


def build_benchmark_inventory(project: Path, out: Path) -> dict:
    bench = project / "data" / "benchmarks"
    prov_by_set = {p["set_name"]: p for p in SEED_PROVENANCE}

    header = ["set", "status", "n_genomes", "metadata_columns", "has_generation_metadata",
              "generation_metadata_columns", "generator_script", "base_seed",
              "per_sample_seed_recorded", "predictions_available", "metadata_sha256"]
    rows = []
    counted = {}
    for name, prov in prov_by_set.items():
        d = bench / name
        meta = d / "metadata.tsv"
        if not meta.is_file():
            print(f"  ! missing {meta} — skipped", file=sys.stderr)
            continue
        hdr, body = read_tsv(meta)
        gen = d / "generation_metadata.tsv"
        gen_cols = 0
        if gen.is_file():
            gen_hdr, _ = read_tsv(gen)
            gen_cols = len(gen_hdr)
        preds = sorted(p.name for p in d.glob("*_predictions.tsv"))
        rows.append([
            name, prov["status"], str(len(body)), str(len(hdr)),
            "yes" if gen.is_file() else "no", str(gen_cols),
            prov["generator"], prov["base_seed"], prov["per_sample_seed_recorded"],
            ";".join(preds) if preds else "-", sha256_file(meta),
        ])
        counted[name] = len(body)
        if prov["n_samples"] != len(body):
            print(f"  ! {name}: recorded n_samples={prov['n_samples']} but metadata has {len(body)}",
                  file=sys.stderr)

    (out / "benchmarks" / "benchmark_inventory.tsv").write_text(
        "\n".join(["\t".join(header)] + ["\t".join(r) for r in rows]) + "\n", encoding="utf-8")
    return dict(n_sets=len(rows), genomes_per_set=counted,
                total_benchmark_genomes=sum(counted.values()))


# ==========================================================================
# WS7.7 — the synthetic generator: seeds and per-sample metadata
# ==========================================================================
def build_generator_release(project: Path, out: Path) -> dict:
    header = ["set", "status", "n_samples", "generator_script", "base_seed",
              "design_rng", "per_sample_rng", "per_sample_seed_recorded", "notes"]
    rows = [[p["set_name"], p["status"], str(p["n_samples"]), p["generator"], p["base_seed"],
             p["design_rng"], p["per_sample_rng"], p["per_sample_seed_recorded"], p["notes"]]
            for p in SEED_PROVENANCE]
    (out / "generator" / "seed_provenance.tsv").write_text(
        "\n".join(["\t".join(header)] + ["\t".join(r) for r in rows]) + "\n", encoding="utf-8")

    # Column-level index of what per-sample metadata is available where.
    bench = project / "data" / "benchmarks"
    idx_lines = ["\t".join(["set", "file", "n_columns", "columns"])]
    n_full = 0
    for p in SEED_PROVENANCE:
        d = bench / p["set_name"]
        for fname in ("metadata.tsv", "generation_metadata.tsv"):
            f = d / fname
            if not f.is_file():
                continue
            hdr, _ = read_tsv(f)
            idx_lines.append("\t".join([p["set_name"], fname, str(len(hdr)), ",".join(hdr)]))
            if fname == "generation_metadata.tsv":
                n_full += 1
    (out / "generator" / "per_sample_metadata_index.tsv").write_text(
        "\n".join(idx_lines) + "\n", encoding="utf-8")

    readme = f"""# Synthetic benchmark generator — seeds and per-sample metadata  (WS7.7, R1-m15)

Reviewer 1 asked for the generator to be released "with all random seeds and
per-sample metadata (dominant ID, contaminant IDs, target/observed completeness
and contamination, fragmentation and dropout parameters)". This directory states
exactly what exists, for which set, and how complete it is. It does not
overstate coverage.

## Generator source

| Component | Module |
|---|---|
| Fragmentation / dropout simulation | `magicc/fragmentation.py` |
| Contamination event construction | `magicc/contamination.py` |
| Clean Sets C/D driver (fully instrumented) | `scripts/73_generate_clean_cd_benchmarks.py` |
| Finished-genome sets A_v2 / B_v2 / E driver | `scripts/34_generate_finished_benchmarks.py` |
| Original Sets A–D driver | `scripts/25_benchmark_generate.py` |
| Motivating sets driver | `scripts/31_motivating_benchmark_generate.py`, `scripts/41_generate_motivating_set_c.py` |

## Two tiers of seed provenance — stated plainly

**Tier 1 — fully recorded per-sample seeds ({n_full} sets: `set_C_clean`, `set_D_clean`).**
`generation_metadata.tsv` has 43 columns and stores the literal integer seed of
every sample together with the complete parameter set that produced it:
dominant accession/phylum/domain/taxonomy/source-split/reference bp/retained bp,
contaminant accessions + phyla + reference bp + bp + source split, requested and
selected contaminant counts, target **and** observed completeness and
contamination, quality tier with its contig-count / N50 / minimum-contig /
log-normal-sigma ranges, all six dropout parameters (coverage mean, coverage
sigma, coverage threshold, GC-loss strength, repeat-loss probability,
low-complexity threshold), the constraint-guard flag, and per-source contig
counts. Regeneration was **tested**: three FASTA files were deleted, their
checkpoint lines removed, and the rerun reproduced them byte-identically.

**Tier 2 — derivable per-sample seeds (all other sets).**
The earlier drivers did not persist a `seed` column, but they are deterministic:
each sample's RNG is `np.random.default_rng(BASE + row_index + OFFSET)` with the
constants given in `seed_provenance.tsv`. Any sample can therefore be
regenerated from the released `metadata.tsv` row index without guessing. What is
**not** recoverable from the released files alone for these sets is the drawn
fragmentation/dropout parameters, because they were not written out — they are
reproduced by rerunning the generator with the stated seed, not read from a
table.

`per_sample_metadata_index.tsv` lists, for every set and file, the exact columns
available, so no reader has to assume.

## Files

| File | Content |
|---|---|
| `seed_provenance.tsv` | one row per set: generator, base seed, design RNG, per-sample RNG construction, whether seeds are recorded or derivable |
| `per_sample_metadata_index.tsv` | which per-sample columns exist in which file for which set |
"""
    (out / "generator" / "README.md").write_text(readme, encoding="utf-8")
    return dict(n_sets_documented=len(SEED_PROVENANCE), n_sets_with_recorded_seeds=n_full)


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    project = Path(args.project_dir).resolve()
    out = project / "results" / "revision" / "reproducibility" / "release"
    for sub in RELEASE_SUBDIRS:
        (out / sub).mkdir(parents=True, exist_ok=True)

    print("WS7.5  selected 9-mer list ...")
    kmer_info = build_kmer_release(project, out)
    print(f"   {kmer_info['n_kmers']:,} k-mers "
          f"({kmer_info['n_bacterial']:,} bacterial / {kmer_info['n_archaeal']:,} archaeal / "
          f"{kmer_info['n_shared']:,} shared)")

    print("WS7.6  splits ...")
    split_info = build_splits_release(project, out)
    print("   " + ", ".join(f"{s}={split_info[s]['n_genomes']:,}" for s in ("train", "val", "test"))
          + "  disjointness verified")

    print("WS7.6  normalization parameters ...")
    norm_info = build_normalization_release(project, out)

    print("WS7.6  benchmark inventory ...")
    bench_info = build_benchmark_inventory(project, out)
    print(f"   {bench_info['n_sets']} sets, {bench_info['total_benchmark_genomes']:,} genomes")

    print("WS7.7  generator seed provenance ...")
    gen_info = build_generator_release(project, out)
    print(f"   {gen_info['n_sets_documented']} sets documented, "
          f"{gen_info['n_sets_with_recorded_seeds']} with literal per-sample seeds")

    # ---- model + container artefacts referenced by checksum ---------------
    model = project / "models" / "magicc_v5.onnx"
    card = json.loads((project / "results" / "revision" / "model_card.json").read_text())
    model_sha = sha256_file(model)
    if model_sha != card["onnx"]["sha256"]:
        print(f"FATAL: models/magicc_v5.onnx sha256 {model_sha} != model_card {card['onnx']['sha256']}",
              file=sys.stderr)
        return 1
    print(f"WS7.6  frozen model verified against model_card.json ({model_sha[:16]}...)")

    # ---- bundle README ---------------------------------------------------
    (out / "README.md").write_text(f"""# MAGICC reproducibility release bundle

Generated by `scripts/112_release_artifacts.py`
({datetime.now(timezone.utc).date().isoformat()}).
Covers protocol items **WS7.5, WS7.6, WS7.7** and reviewer comments
**R1-M8, R1-m14, R1-m15** plus editorial requirements **E5/E6**.

## Contents

| Directory | Item | What it contains |
|---|---|---|
| `kmers/` | WS7.5 (R1-m14) | the {kmer_info['n_kmers']:,} selected canonical 9-mers with domain assignment, bacterial and archaeal prevalence, rank-cutoff flags and selection reason; the **full** 131,072-k-mer prevalence tables for both domains so the selection can be recomputed; the 2,000 feature-selection genomes |
| `splits/` | WS7.6 | train/val/test accession lists, split statistics, download-failure list, and SHA256 of the full genome tables |
| `benchmarks/` | WS7.6 | one row per benchmark set: size, generator, seed scheme, retained/withdrawn status, available predictions, metadata checksum |
| `generator/` | WS7.7 (R1-m15) | exact RNG construction for every sample of every set, and a column-level index of the per-sample metadata that exists |
| `normalization/` | WS7.6 | the inference-time normalization parameters, with checksums against the copy shipped inside the Python package |

## Key numbers

* Model: **V5**, `models/magicc_v5.onnx`, SHA256 `{model_sha}` — verified against `results/revision/model_card.json` every time this script runs.
* Features: **{kmer_info['n_kmers']:,}** canonical 9-mers ({kmer_info['n_bacterial']:,} bacterial ranking, {kmer_info['n_archaeal']:,} archaeal ranking, {kmer_info['n_shared']:,} shared) + 7 k-mer summary features.
* Splits: train **{split_info['train']['n_genomes']:,}**, val **{split_info['val']['n_genomes']:,}**, test **{split_info['test']['n_genomes']:,}**, mutually disjoint (verified here, not asserted).
* Benchmarks: **{bench_info['n_sets']}** sets, **{bench_info['total_benchmark_genomes']:,}** genomes, of which **{gen_info['n_sets_with_recorded_seeds']}** sets carry literal per-sample seeds and the rest carry derivable ones.

## Relationship to the other manifests

This bundle **extends** rather than duplicates
`results/revision/provenance/sha256_manifest.txt` (the WS1.4 provenance audit,
2,021 entries covering benchmark FASTA and metadata). Large inputs already
hashed there are referenced by path and checksum, not copied.

`MANIFEST.sha256` covers every file in this bundle;
`RELEASE_MANIFEST.json` is the machine-readable index.

## Honest scope statements

* Two benchmark sets (`set_C`, `set_D`) are included in the inventory with status
  **WITHDRAWN**. They are released so the training-data leakage that invalidated
  them is auditable, not because their results stand.
* Per-sample seeds are *recorded* for `set_C_clean` / `set_D_clean` and
  *derivable* for the rest; `generator/README.md` says precisely what is and is
  not recoverable for each. The earlier drivers did not persist the drawn
  fragmentation/dropout parameters, so those are reproduced by rerunning the
  generator with the stated seed rather than read from a table.
* The released **weights** are not bit-reproducible from scratch because V5
  training was never seeded (protocol section 4.4b). The artefact is pinned by
  SHA256 and *inference* is deterministic; see
  `results/revision/reproducibility/DETERMINISM.md`.
""", encoding="utf-8")

    # ---- manifest --------------------------------------------------------
    entries = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST.sha256", "RELEASE_MANIFEST.json"):
            entries.append((sha256_file(p), str(p.relative_to(out))))
    manifest_lines = [
        "# SHA256 manifest — MAGICC reproducibility release bundle (WS7.5/7.6/7.7)",
        f"# generated {datetime.now(timezone.utc).isoformat()}",
        "# paths are relative to results/revision/reproducibility/release/",
        "#",
        "# Large inputs already hashed by the WS1.4 provenance audit are NOT",
        "# duplicated here; see results/revision/provenance/sha256_manifest.txt",
    ] + [f"{h}  {rel}" for h, rel in entries]
    (out / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    release = dict(
        schema="magicc-release-bundle/1.0",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        generated_by="scripts/112_release_artifacts.py",
        workstream_items=["WS7.5", "WS7.6", "WS7.7"],
        reviewer_comments=["R1-M8", "R1-m14", "R1-m15", "E5", "E6"],
        model=dict(path="models/magicc_v5.onnx", sha256=model_sha,
                   version=card["frozen_model"]["model_version"],
                   card="results/revision/model_card.json"),
        kmers=kmer_info,
        splits=split_info,
        normalization=norm_info,
        benchmarks=bench_info,
        generator=gen_info,
        upstream_manifest="results/revision/provenance/sha256_manifest.txt",
        n_files_in_bundle=len(entries),
    )
    (out / "RELEASE_MANIFEST.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")

    print(f"\nBundle: {out}")
    print(f"  {len(entries)} files, MANIFEST.sha256 + RELEASE_MANIFEST.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
