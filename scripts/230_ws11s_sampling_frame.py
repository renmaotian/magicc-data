#!/usr/bin/env python3
"""
WS11.S (resubmission3, the internal build contract §5.3) -- catalogue-scale SPIRE sampling frame.

Reviewer 2 (R2-M3) asked for MAGICC to be applied to one or more large public MAG
collections and for the proportion whose MIMAG-inspired classification changes.  The
previously delivered cohorts were 612/591 SPIRE MAGs of five named genera plus a
750-genome size-stratified GTDB r220 cohort -- neither is a catalogue-representative
sample, and neither rate may be read as a catalogue rate.

This script defines, ONCE and reproducibly, the probability sampling design for the
catalogue-scale run.

FRAME
  data/real_data/spire/spire_v1_genome_metadata.tsv.gz, 1,158,553 rows.
  Rows without published CheckM2 completeness/contamination/genome_size are OUT OF
  FRAME (a disagreement analysis needs the published value); 85 such rows exist.
  Analysis frame N = 1,158,468.

DESIGN -- stratified random sample, strata defined on frame variables only:
  S1 small        genome_size < 1 Mbp                       N =  59,297   f = 1.000
  S2 rare_phylum  not S1 and phylum has < 1,000 frame MAGs  N =  14,586   f = 1.000
  S3 main         everything else                           N =1,084,581  f = 200000/N
  Design weight w_i = N_h / n_h.  S1 and S2 are complete censuses (w = 1); the headline
  catalogue-wide rate is the weighted (Hajek) estimate over S1 u S2 u S3, and the
  SRS-only unweighted estimate on S3 is reported alongside as a design check.

  S1 is a census because it is the stratum the scientific question lives in (MAGICC's
  size channel) and it is only 5.1 % of the catalogue.  A separate "extreme reduction"
  stratum (log2(size / phylum median) < -2) was evaluated and NOT created: 1,188 of the
  1,192 such genomes already fall in S1 u S2, so S1 u S2 is already a 99.7 % census of
  it.

  The S3 sample is the first n elements of a single seeded uniform random permutation of
  the S3 members, so any prefix is a valid simple random sample without replacement and
  the sample can be grown or truncated without invalidating the design.

SECOND, INDEPENDENT COHORT -- `reps` (see script 232): all 92,063 SPIRE species-level
  (95 % ANI) cluster representatives that are MAGs and are in the frame.  This is a
  COMPLETE CENSUS of the dereplicated catalogue, not a sample; it is a different
  population from the per-MAG catalogue (representatives are the best genome of their
  cluster) and its rate must never be quoted as the per-MAG catalogue rate.

Read-only on data/.  No network.  Single-threaded.

Outputs (results/revision/ws11/spire_catalogue/)
  cohort_definition.tsv.gz   one row per sampled genome, with stratum, weight, published
                             CheckM2 values, size, taxonomy, cluster and the derived
                             lineage-relative size.
  sampling_frame.json        the design, its seed, every stratum N/n/f, and the frame
                             totals needed for weighting and for post-stratification.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/path/to/magicc")
META = ROOT / "data/real_data/spire/spire_v1_genome_metadata.tsv.gz"
OUT = ROOT / "results/revision/ws11/spire_catalogue"

SEED_STRING = "WS11.S/spire_catalogue/SRS"
N_MAIN = 200_000
SMALL_BP = 1_000_000
RARE_PHYLUM_MAX = 1_000
SIZE_EDGES = [1e6, 2e6, 3e6, 5e6]
SIZE_LABELS = ["<1Mb", "1-2Mb", "2-3Mb", "3-5Mb", ">5Mb"]


def mimag(comp: float, cont: float) -> str:
    """MIMAG-inspired class (always labelled '-inspired' in prose)."""
    if comp >= 90.0 and cont < 5.0:
        return "HQ"
    if comp >= 50.0 and cont < 10.0:
        return "MQ"
    return "LQ"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gid, size, comp, cont, ph, fam, clus, dom, ncontig, n50, spec = ([] for _ in range(11))
    n_rows = 0
    dropped = 0
    with gzip.open(META, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            n_rows += 1
            try:
                s = float(r["genome_size"]); c = float(r["completeness"]); k = float(r["contamination"])
            except (ValueError, KeyError):
                dropped += 1
                continue
            gid.append(r["genome_id"]); size.append(s); comp.append(c); cont.append(k)
            ph.append(r["phylum"] or "unclassified")
            fam.append(r["family"] or "unclassified")
            spec.append(r["species"] or "unclassified")
            clus.append(r["spire_cluster"]); dom.append(r["domain"] or "unclassified")
            ncontig.append(r["n_contigs"] or ""); n50.append(r["n50"] or "")
    gid = np.array(gid); size = np.array(size); comp = np.array(comp); cont = np.array(cont)
    ph = np.array(ph); fam = np.array(fam); clus = np.array(clus); dom = np.array(dom)
    N = len(gid)
    print(f"[230] metadata rows {n_rows:,}; out of frame (no published CheckM2) {dropped}; frame N = {N:,}",
          flush=True)

    ph_count = Counter(ph.tolist())
    ph_median = {p: float(np.median(size[ph == p])) for p in ph_count}
    lin_rel = np.log2(size / np.array([ph_median[p] for p in ph]))

    s1 = size < SMALL_BP
    rare = np.array([ph_count[p] < RARE_PHYLUM_MAX for p in ph])
    s2 = (~s1) & rare
    s3 = ~(s1 | s2)
    strata = np.where(s1, "S1_small", np.where(s2, "S2_rare_phylum", "S3_main"))

    seed = zlib.crc32(SEED_STRING.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    idx3 = np.flatnonzero(s3)
    perm3 = rng.permutation(idx3.size)
    n_main = min(N_MAIN, idx3.size)
    take3 = idx3[perm3[:n_main]]
    order3 = {int(g): r for r, g in enumerate(take3)}

    sel = np.concatenate([np.flatnonzero(s1), np.flatnonzero(s2), take3])
    sel.sort()

    Nh = {"S1_small": int(s1.sum()), "S2_rare_phylum": int(s2.sum()), "S3_main": int(s3.sum())}
    nh = {"S1_small": int(s1.sum()), "S2_rare_phylum": int(s2.sum()), "S3_main": int(n_main)}
    wh = {h: Nh[h] / nh[h] for h in Nh}
    print(f"[230] seed {seed}; strata " + ", ".join(f"{h}: N={Nh[h]:,} n={nh[h]:,} w={wh[h]:.5f}" for h in Nh),
          flush=True)

    cls = np.array([mimag(c, k) for c, k in zip(comp, cont)])
    sbin = np.digitize(size, SIZE_EDGES)

    cd = OUT / "cohort_definition.tsv.gz"
    cols = ["genome_id", "stratum", "design_weight", "srs_order", "checkm2_completeness",
            "checkm2_contamination", "checkm2_class", "genome_size", "n_contigs", "n50",
            "size_bin", "domain", "phylum", "family", "species", "spire_cluster",
            "phylum_median_size", "log2_size_over_phylum_median"]
    with gzip.open(cd, "wt", compresslevel=6) as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for i in sel:
            h = strata[i]
            w.writerow([gid[i], h, f"{wh[h]:.6f}", order3.get(int(i), ""),
                        f"{comp[i]:.2f}", f"{cont[i]:.2f}", cls[i], int(size[i]),
                        ncontig[i], n50[i], SIZE_LABELS[sbin[i]], dom[i], ph[i], fam[i],
                        spec[i], clus[i], f"{ph_median[ph[i]]:.1f}", f"{lin_rel[i]:.4f}"])
    print(f"[230] wrote {cd} with {len(sel):,} sampled genomes "
          f"({100 * len(sel) / N:.2f} % of the frame)", flush=True)

    frame_tot = {
        "N": int(N),
        "by_stratum": Nh,
        "by_size_bin": {SIZE_LABELS[b]: int((sbin == b).sum()) for b in range(5)},
        "by_class": {c: int((cls == c).sum()) for c in ("HQ", "MQ", "LQ")},
        "by_domain": {d: int((dom == d).sum()) for d in sorted(set(dom.tolist()))},
        "n_phyla": int(len(set(ph.tolist()))),
        "n_families": int(len(set(fam.tolist()))),
        "n_spire_clusters": int(len(set(clus.tolist()))),
        "total_bp": float(size.sum()),
    }
    sample_tot = {
        "n": int(len(sel)),
        "fraction_of_frame": float(len(sel) / N),
        "by_stratum": nh,
        "by_size_bin": {SIZE_LABELS[b]: int((sbin[sel] == b).sum()) for b in range(5)},
        "n_phyla": int(len(set(ph[sel].tolist()))),
        "n_families": int(len(set(fam[sel].tolist()))),
        "n_spire_clusters": int(len(set(clus[sel].tolist()))),
    }
    design = {
        "workstream": "WS11.S",
        "catalogue": "SPIRE v1",
        "catalogue_metadata_file": str(META.relative_to(ROOT)),
        "published_quality_tool": "CheckM2 (as published by SPIRE v1)",
        "ground_truth": None,
        "interpretation": ("Catalogue MAGs carry NO ground truth. Every quantity here is a "
                           "DISAGREEMENT between MAGICC V5 and SPIRE's published CheckM2, not an "
                           "error measurement. The ground-truthed anchor is set_C_clean, where "
                           "MAGICC is the tool in error (-8.68 pp completeness, +5.09 pp "
                           "contamination vs truth; CheckM2 -0.70 / -1.16)."),
        "metadata_rows": n_rows,
        "out_of_frame_no_published_checkm2": dropped,
        "frame": frame_tot,
        "design_type": "stratified random sample; S1 and S2 complete censuses, S3 SRS without replacement",
        "seed_string": SEED_STRING,
        "seed": int(seed),
        "rng": "numpy.random.default_rng (PCG64)",
        "strata": {h: {"N": Nh[h], "n": nh[h], "sampling_fraction": nh[h] / Nh[h],
                       "design_weight": wh[h]} for h in Nh},
        "stratum_definitions": {
            "S1_small": f"genome_size < {SMALL_BP} bp",
            "S2_rare_phylum": f"not S1 and phylum has < {RARE_PHYLUM_MAX} frame MAGs",
            "S3_main": "all remaining frame MAGs",
        },
        "sample": sample_tot,
        "mimag_inspired_thresholds": {"HQ": "completeness >= 90 and contamination < 5",
                                      "MQ": "completeness >= 50 and contamination < 10",
                                      "LQ": "otherwise"},
        "bootstrap_clustering_unit_primary": "family (as in WS3.10)",
        "bootstrap_clustering_unit_secondary": "spire_cluster (95 % ANI, species level)",
        "second_cohort": {
            "name": "reps",
            "definition": "all SPIRE 95 % ANI cluster representatives that are MAGs and in the frame",
            "n": 92063,
            "is_census": True,
            "population": "dereplicated SPIRE catalogue (species-level representatives)",
            "warning": "a different population from the per-MAG catalogue; never quote as the per-MAG rate",
        },
    }
    (OUT / "sampling_frame.json").write_text(json.dumps(design, indent=2) + "\n")
    print(f"[230] wrote {OUT / 'sampling_frame.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
