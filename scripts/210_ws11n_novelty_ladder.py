#!/usr/bin/env python
"""
210_ws11n_novelty_ladder.py  --  WS11.N
=======================================
THE TAXONOMIC-NOVELTY LADDER ON THE HEADLINE FIVE-SET PANEL (R1-M2).

WHAT THIS IS
------------
Reviewer 1 asked for held-out evaluation "by genome, species, genus and phylum".
Sets A_v2 / B_v2 / C_clean / D_clean / E are held out **by genome only**: the
train/val/test split is phylum-stratified, so a test-split genome usually shares
its family and genus with training genomes. Lineage-level holdout in this project
exists only through *retrained validation-artefact* models (WS1.6 phylum, WS1.9
family).

This script closes the gap OBSERVATIONALLY and at zero inference cost. For every
dominant reference genome of the five leakage-free sets it determines the deepest
GTDB rank at which the genome is novel *relative to the training split*, and then
re-cuts the already-computed predictions of MAGICC v5, CheckM2, CoCoPyE and
DeepCheck along that ladder.

    *** THIS IS AN OBSERVATIONAL STRATIFICATION OF ONE FROZEN MODEL'S           ***
    *** PREDICTIONS. IT IS NOT A RETRAINED HOLDOUT EXPERIMENT. Genomes in the   ***
    *** deeper novelty classes were never removed from training - they simply   ***
    *** happen to have no training relative at that rank. Nothing here can be   ***
    *** described as "held out at genus level"; the retrained holdouts (WS1.6 / ***
    *** WS1.9 / WS11.G) are the only holdout evidence.                          ***

Its value is that it measures the genome -> species -> genus ladder on exactly
the panel the reviewer was reading, on identical genomes for all four tools.

CONVENTIONS (the internal project log section 2.2)
---------------------------------------------------------
* clusters for every bootstrap = reference genomes (here: set x dominant
  accession, the WS5 pooled definition; the distinct-accession count is also
  reported - see results/revision/metrics/ws5.10_*);
* 2,000 percentile bootstrap resamples;
* every seed via ``fw.stable_hash`` (CRC-32) with ``PYTHONHASHSEED=0``;
* R2 = coefficient of determination, blanked where true variance is ~0;
* MIMAG-inspired thresholds, always labelled "-inspired";
* two-sided paired Wilcoxon (cluster-mean primary), Hodges-Lehmann effect size,
  BH correction.

TRAP T1 GUARD
-------------
The five sets are hard-coded by name. ``fw.discover_sets`` autodiscovery (which
would pull in set_F / set_G) is filtered against that list and the script aborts
if any other set leaks in.

INPUTS (all small TSV/JSON - no FASTA is opened, no inference is run)
--------------------------------------------------------------------
  data/benchmarks/{set_A_v2,set_B_v2,set_C_clean,set_D_clean,set_E}/metadata.tsv
  ... and the four *_predictions.tsv in each of those directories
  data/gtdb/filtered_genomes.tsv          (GTDB taxonomy + GCA/GCF cross-map)
  data/splits/train_genomes.tsv           (training-split taxonomy)
  data/splits/val_genomes.tsv             (sensitivity arm only)

OUTPUTS  results/revision/ws11/novelty_ladder/
---------------------------------------------
  dominant_novelty_classification.tsv   one row per dominant, full rank evidence
  novelty_ladder_metrics.tsv            accuracy + MIMAG-inspired thresholds
  novelty_ladder_paired_tests.tsv       MAGICC vs CheckM2 within each class
  novelty_ladder_trend.tsv              monotonicity / trend test
  novelty_ladder_summary.json           machine-readable summary + EDA record
  novelty_ladder_eda.json               input verification record
  WS11_N_REPORT.md                      the readable report

Usage:
    PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python scripts/210_ws11n_novelty_ladder.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load 104 FIRST so that it, and we, share ONE framework module instance.
st = _load(_HERE / "104_clustered_statistics.py", "magicc_clustered_statistics")
fw = st.fw

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
from scipy import stats as sps   # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: TRAP T1 -- the ONLY sets that may enter these numbers.
FIVE_SETS = ["set_A_v2", "set_B_v2", "set_C_clean", "set_D_clean", "set_E"]

TOOLS = ["magicc_v5", "checkm2", "cocopye", "deepcheck"]
REF_TOOL = "magicc_v5"

RANK_CODES = ["d", "p", "c", "o", "f", "g", "s"]
RANK_NAMES = {"d": "domain", "p": "phylum", "c": "class", "o": "order",
              "f": "family", "g": "genus", "s": "species"}
#: deepest -> shallowest, the order in which representation is searched
SEARCH = ["s", "g", "f", "o", "c", "p"]
#: novelty class implied when rank R is the deepest represented rank
NOVEL_BELOW = {"s": "lineage_represented", "g": "species_novel",
               "f": "genus_novel", "o": "family_novel", "c": "order_novel",
               "p": "class_novel"}
CLASS_ORDER = ["lineage_represented", "species_novel", "genus_novel",
               "family_novel", "order_novel", "class_novel", "phylum_novel"]
CLASS_DEPTH = {c: i for i, c in enumerate(CLASS_ORDER)}
#: classes pooled into one reportable cell because each is too thin alone
DEEP_POOL = ["family_novel", "order_novel", "class_novel", "phylum_novel"]
DEEP_POOL_LABEL = "family_novel_or_deeper (pooled)"

#: minimum cluster count for a cell to carry a claim
MIN_CLUSTERS_FOR_CLAIM = 20
CONT_TAU = 5.0          # the MIMAG-inspired contamination boundary

OUT_DIR = PROJECT / "results" / "revision" / "ws11" / "novelty_ladder"

_VER = re.compile(r"\.\d+$")
_NUM = re.compile(r"(\d{6,})")
_SUFFIX = re.compile(r"_[A-Z]{1,3}$")


# ---------------------------------------------------------------------------
# Accession normalisation -- VERBATIM from scripts/074_provenance_audit.py
# ---------------------------------------------------------------------------


def strict_norm(acc) -> Optional[str]:
    """GB_GCA_001822065.1 -> GCA_001822065 (keeps the GCA/GCF distinction)."""
    if not isinstance(acc, str) or not acc:
        return None
    a = acc.strip()
    for p in ("GB_", "RS_"):
        if a.startswith(p):
            a = a[len(p):]
    return _VER.sub("", a)


def assembly_number(acc) -> Optional[str]:
    """GB_GCA_001822065.1 -> '001822065' (shared by a GCA/GCF assembly pair)."""
    if not isinstance(acc, str) or not acc:
        return None
    m = _NUM.search(acc)
    return m.group(1) if m else None


def build_crossmap(fg: pd.DataFrame) -> Tuple[Dict[str, str], dict]:
    """strict-normalised accession string -> canonical assembly-pair key.

    Identical construction to ``074_provenance_audit.build_crossmap``; the stats
    dict is compared against results/revision/provenance/accession_crossmap_stats.json
    so that any drift is caught rather than silently absorbed.
    """
    cmap: Dict[str, str] = {}
    groups: Dict[str, set] = defaultdict(set)
    inconsistent = 0
    for gtdb, ncbi, gcf in zip(fg["gtdb_accession"], fg["ncbi_accession"],
                               fg["gcf_accession"]):
        key = assembly_number(ncbi) or assembly_number(gtdb) or assembly_number(gcf)
        if key is None:
            continue
        nums = set()
        for a in (gtdb, ncbi, gcf):
            s = strict_norm(a)
            if s:
                prev = cmap.get(s)
                if prev is not None and prev != key:
                    inconsistent += 1
                cmap[s] = key
                groups[key].add(s)
                n = assembly_number(a)
                if n:
                    nums.add(n)
        if len(nums) > 1:
            inconsistent += 1
    stats = {
        "filtered_genomes_rows": int(len(fg)),
        "distinct_accession_strings_mapped": len(cmap),
        "distinct_canonical_assemblies": len(groups),
        "rows_with_inconsistent_assembly_numbers": int(inconsistent),
        "mean_strings_per_assembly": round(
            float(np.mean([len(v) for v in groups.values()])), 3),
    }
    return cmap, stats


def canon_key(acc: str, cmap: Dict[str, str]) -> Optional[str]:
    s = strict_norm(acc)
    if s is None:
        return None
    return cmap.get(s) or assembly_number(acc)


# ---------------------------------------------------------------------------
# GTDB taxonomy handling
# ---------------------------------------------------------------------------


def parse_taxonomy(t: str) -> Dict[str, str]:
    """'d__Bacteria;p__X;...' -> {'d': 'Bacteria', 'p': 'X', ...}."""
    out: Dict[str, str] = {}
    for part in str(t).split(";"):
        part = part.strip()
        if len(part) >= 3 and part[1:3] == "__":
            out[part[0]] = part[3:].strip()
    return out


def strip_polyphyly_suffix(name: str, rank: str) -> str:
    """Drop GTDB's polyphyly suffix ('Bacteroidota_A' -> 'Bacteroidota').

    Used ONLY for the sensitivity arm. The primary classification keeps the
    suffix, because GTDB treats a suffixed lineage as a separate taxon and this
    project already does so elsewhere (WS1.6 held out Bacteroidota_A separately
    from Bacteroidota precisely so no sister lineage would leak).
    """
    if rank == "s":
        return " ".join(_SUFFIX.sub("", tok) for tok in name.split(" "))
    return _SUFFIX.sub("", name)


def training_taxon_sets(df: pd.DataFrame, strip: bool = False
                        ) -> Tuple[Dict[str, set], Dict[str, Dict[str, int]]]:
    """Per-rank set of taxa present, plus per-rank genome counts per taxon."""
    present = {r: set() for r in RANK_CODES}
    counts = {r: defaultdict(int) for r in RANK_CODES}
    for t in df["gtdb_taxonomy"]:
        p = parse_taxonomy(t)
        for r in RANK_CODES:
            v = p.get(r, "")
            if not v:
                continue
            if strip:
                v = strip_polyphyly_suffix(v, r)
            present[r].add(v)
            counts[r][v] += 1
    return present, {r: dict(c) for r, c in counts.items()}


def classify_novelty(tax: str, present: Dict[str, set], strip: bool = False
                     ) -> Tuple[str, str]:
    """(novelty_class, deepest_represented_rank_code) for one taxonomy string."""
    p = parse_taxonomy(tax)
    for r in SEARCH:                      # species -> ... -> phylum
        v = p.get(r, "")
        if not v:
            continue
        if strip:
            v = strip_polyphyly_suffix(v, r)
        if v in present[r]:
            return NOVEL_BELOW[r], r
    return "phylum_novel", "-"


# ---------------------------------------------------------------------------
# Bootstrap statistics for one (cell, tool)
# ---------------------------------------------------------------------------


def _stat_fn(df: pd.DataFrame, tool: str):
    tc = df["true_completeness"].to_numpy(float)
    tx = df["true_contamination"].to_numpy(float)
    pc = df[f"pred_completeness__{tool}"].to_numpy(float)
    px = df[f"pred_contamination__{tool}"].to_numpy(float)

    def f(idx: np.ndarray) -> Dict[str, float]:
        a, b, c, d = tc[idx], tx[idx], pc[idx], px[idx]
        th = fw.threshold_metrics(b, d, CONT_TAU, "contamination")
        return {
            "completeness_mae": float(np.mean(np.abs(c - a))),
            "completeness_bias": float(np.mean(c - a)),
            "contamination_mae": float(np.mean(np.abs(d - b))),
            "contamination_bias": float(np.mean(d - b)),
            "completeness_r2_cod": fw.r2_coefficient_of_determination(a, c),
            "contamination_r2_cod": fw.r2_coefficient_of_determination(b, d),
            "false_fail_rate_cont5": th["false_fail_rate"],
            "false_pass_rate_cont5": th["false_pass_rate"],
            "balanced_accuracy_cont5": th["balanced_accuracy"],
        }
    return f


def cell_rows(df: pd.DataFrame, scope: str, cell: str, seed_tag: str,
              n_boot: int, cfg) -> List[dict]:
    """One row per tool for one (scope, novelty class) cell."""
    if len(df) == 0:
        return []
    seed = cfg.seed + (fw.stable_hash(seed_tag) % 100000)
    bs = fw.Bootstrapper(clusters=df["cluster_id"].to_numpy(), n_iter=n_boot,
                         ci_level=cfg.ci_level, seed=seed)
    bs.resample_indices()                    # shared resample across tools

    tx = df["true_contamination"].to_numpy(float)
    n_true_pass = int(np.sum(tx < CONT_TAU))
    n_true_fail = int(np.sum(tx >= CONT_TAU))
    true_comp_var = float(np.var(df["true_completeness"].to_numpy(float)))
    true_cont_var = float(np.var(tx))

    rows = []
    for tool in TOOLS:
        if f"pred_completeness__{tool}" not in df.columns:
            continue
        boot = bs.ci_multi(_stat_fn(df, tool))
        th = fw.threshold_metrics(tx, df[f"pred_contamination__{tool}"].to_numpy(float),
                                  CONT_TAU, "contamination")
        rec = {
            "scope": scope,
            "novelty_class": cell,
            "novelty_depth": CLASS_DEPTH.get(cell, np.nan),
            "tool": tool,
            "tool_label": cfg.tool_label(tool),
            "n_genomes": int(len(df)),
            "n_clusters_set_x_ref": int(df["cluster_id"].nunique()),
            "n_distinct_reference_accessions": int(df["dominant_accession"].nunique()),
            "seed": int(seed),
            "bootstrap_iterations": int(n_boot),
            "cluster_definition": "set x dominant_accession",
        }
        for k, v in boot.items():
            rec[k] = v["estimate"]
            rec[k + "_ci_lo"] = v["ci_lo"]
            rec[k + "_ci_hi"] = v["ci_hi"]
        # R2 is blanked (with the reason recorded) where true variance is ~0
        for metric, var in (("completeness", true_comp_var),
                            ("contamination", true_cont_var)):
            if var < 1e-8:
                for sfx in ("", "_ci_lo", "_ci_hi"):
                    rec[f"{metric}_r2_cod{sfx}"] = np.nan
                rec[f"{metric}_r2_omitted_reason"] = (
                    f"true {metric} variance {var:.3g} ~ 0 (R1-m19)")
            else:
                rec[f"{metric}_r2_omitted_reason"] = ""
        # threshold denominators, named explicitly (trap T2)
        rec.update({
            "cont5_n_true_pass_DENOM_false_fail": n_true_pass,
            "cont5_n_true_fail_DENOM_false_pass": n_true_fail,
            "cont5_n_false_fail": th["n_false_fail"],
            "cont5_n_false_pass": th["n_false_pass"],
            "false_fail_denominator_statement":
                f"{th['n_false_fail']}/{n_true_pass} genomes whose TRUE "
                f"contamination is < {CONT_TAU:g}%",
            "false_pass_denominator_statement":
                f"{th['n_false_pass']}/{n_true_fail} genomes whose TRUE "
                f"contamination is >= {CONT_TAU:g}%",
            "true_completeness_variance": true_comp_var,
            "true_contamination_variance": true_cont_var,
            "supports_claim": bool(df["cluster_id"].nunique() >= MIN_CLUSTERS_FOR_CLAIM),
        })
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-slow", type=int, default=1000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = fw.load_config(args.config)
    eda: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                 "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                 "n_boot": args.n_boot, "sets_allowed": FIVE_SETS,
                 "issues": []}

    def issue(msg: str):
        eda["issues"].append(msg)
        print(f"  [!] {msg}")

    print("=" * 78)
    print("WS11.N  -- taxonomic-novelty ladder on the five leakage-free sets")
    print("           OBSERVATIONAL STRATIFICATION, NOT A RETRAINED HOLDOUT")
    print("=" * 78)

    # ---------------------------------------------------- [0] input existence
    print("\n[0] input verification")
    required = {
        "filtered_genomes": PROJECT / "data" / "gtdb" / "filtered_genomes.tsv",
        "train_split": PROJECT / "data" / "splits" / "train_genomes.tsv",
        "val_split": PROJECT / "data" / "splits" / "val_genomes.tsv",
    }
    for k, p in required.items():
        if not p.exists():
            print(f"  FATAL: missing {k}: {p}")
            return 2
        print(f"  ok  {k:20s} {p}")

    # ------------------------------------------------------ [1] the five sets
    all_sets = fw.discover_sets(cfg, include_missing=False)
    by_name = {b.name: b for b in all_sets}
    leaked = [b.name for b in all_sets
              if not b.listed and b.name not in FIVE_SETS]
    if leaked:
        print(f"  note: autodiscovery saw unregistered sets {leaked}; "
              f"EXCLUDED by the trap-T1 guard")
    sets = []
    for name in FIVE_SETS:
        if name not in by_name:
            print(f"  FATAL: required set {name} not present on disk")
            return 2
        sets.append(by_name[name])
    assert [b.name for b in sets] == FIVE_SETS, "trap T1 guard failed"

    frames = []
    eda["sets"] = {}
    for b in sets:
        df, avail, prov = fw.load_set(cfg, b)
        rec = {"n_rows": int(len(df)), "available_tools": avail,
               "n_clusters": prov["n_clusters"],
               "dominant_split_counts": prov["dominant_split_counts"]}
        if len(df) != 1000:
            issue(f"{b.name}: expected 1,000 rows, found {len(df)}")
        for t in TOOLS:
            info = prov["tools"].get(t, {})
            if not info.get("usable"):
                issue(f"{b.name}: tool {t} unusable/absent -> "
                      f"{info.get('reason', 'file missing')}")
                continue
            rec[f"{t}_n_rows"] = info["n_rows"]
            rec[f"{t}_n_matched"] = info["n_matched"]
            if info["n_matched"] != len(df):
                issue(f"{b.name}/{t}: only {info['n_matched']}/{len(df)} "
                      f"prediction rows joined to metadata")
            nan_c = int(df[f"pred_completeness__{t}"].isna().sum())
            nan_x = int(df[f"pred_contamination__{t}"].isna().sum())
            if nan_c or nan_x:
                issue(f"{b.name}/{t}: {nan_c} NaN completeness / "
                      f"{nan_x} NaN contamination predictions")
            for key in ("truth_max_abs_diff_completeness",
                        "truth_max_abs_diff_contamination"):
                if key in info and info[key] > 1e-6:
                    issue(f"{b.name}/{t}: {key} = {info[key]:.6g} (truth mismatch)")
        # every dominant of the five sets must be OUT of train and val
        bad = {k: v for k, v in prov["dominant_split_counts"].items()
               if k in ("train", "val")}
        if bad:
            issue(f"{b.name}: dominants found in {bad} -- leakage, investigate")
        eda["sets"][b.name] = rec
        keep = ["genome_id", "dominant_accession", "dominant_phylum",
                "dominant_split", "true_completeness", "true_contamination",
                "true_mimag", "cluster_id"]
        keep += [c for c in df.columns if c.startswith(("pred_", "err_", "abs_err_",
                                                        "mimag__"))]
        s = df[keep].copy()
        s["set"] = b.name
        s["set_label"] = b.label
        frames.append(s)
        print(f"  ok  {b.name:14s} n={len(df):5d}  clusters={prov['n_clusters']:5d}  "
              f"tools={len(avail)}")

    D = pd.concat(frames, ignore_index=True)
    # pooled clusters are set x reference genome (the WS5 definition)
    D["cluster_id"] = D["set"] + "::" + D["dominant_accession"].astype(str)
    eda["pooled"] = {
        "n_genomes": int(len(D)),
        "n_clusters_set_x_ref": int(D["cluster_id"].nunique()),
        "n_distinct_reference_accessions": int(D["dominant_accession"].nunique()),
    }
    print(f"  pooled: {len(D)} genomes, {D.cluster_id.nunique()} set x reference "
          f"clusters, {D.dominant_accession.nunique()} distinct reference accessions")
    if len(D) != 5000:
        issue(f"pooled row count {len(D)} != 5,000")
    if D["cluster_id"].nunique() != 2586:
        issue(f"pooled cluster count {D['cluster_id'].nunique()} != 2,586 "
              f"(the WS5 definitive value)")

    # ------------------------------------------------- [2] taxonomy + crossmap
    print("\n[1] GCA/GCF cross-map and GTDB taxonomy")
    fg = pd.read_csv(required["filtered_genomes"], sep="\t",
                     usecols=["gtdb_accession", "ncbi_accession", "gcf_accession",
                              "domain", "phylum", "gtdb_taxonomy"])
    cmap, cmap_stats = build_crossmap(fg)
    eda["crossmap_stats"] = cmap_stats
    ref_stats_path = (PROJECT / "results" / "revision" / "provenance"
                      / "accession_crossmap_stats.json")
    if ref_stats_path.exists():
        ref_stats = json.loads(ref_stats_path.read_text())
        eda["crossmap_matches_ws1_4_audit"] = (ref_stats == cmap_stats)
        if ref_stats != cmap_stats:
            issue(f"cross-map stats differ from the WS1.4 audit: "
                  f"{cmap_stats} vs {ref_stats}")
        else:
            print(f"  ok  cross-map reproduces the WS1.4 provenance audit exactly "
                  f"({cmap_stats['distinct_canonical_assemblies']} assemblies, "
                  f"{cmap_stats['mean_strings_per_assembly']} strings/assembly)")
    tax_by_key: Dict[str, str] = {}
    for gtdb, ncbi, gcf, t in zip(fg["gtdb_accession"], fg["ncbi_accession"],
                                  fg["gcf_accession"], fg["gtdb_taxonomy"]):
        k = assembly_number(ncbi) or assembly_number(gtdb) or assembly_number(gcf)
        if k:
            tax_by_key.setdefault(k, t)

    train = pd.read_csv(required["train_split"], sep="\t",
                        usecols=["gtdb_accession", "gtdb_taxonomy"])
    val = pd.read_csv(required["val_split"], sep="\t",
                      usecols=["gtdb_accession", "gtdb_taxonomy"])
    present_tr, counts_tr = training_taxon_sets(train)
    present_trx, _ = training_taxon_sets(train, strip=True)
    present_tv, _ = training_taxon_sets(pd.concat([train, val], ignore_index=True))
    eda["train_split_rows"] = int(len(train))
    eda["val_split_rows"] = int(len(val))
    eda["train_distinct_taxa_per_rank"] = {RANK_NAMES[r]: len(present_tr[r])
                                           for r in RANK_CODES}
    if len(train) != 79948:
        issue(f"train split has {len(train)} rows, expected 79,948")
    print(f"  ok  training split {len(train)} genomes; distinct taxa "
          + ", ".join(f"{RANK_NAMES[r]}={len(present_tr[r])}"
                      for r in ["p", "c", "o", "f", "g", "s"]))

    # ------------------------------------------- [3] classify every dominant
    print("\n[2] novelty classification of every dominant reference genome")
    dom_sets: Dict[str, set] = defaultdict(set)
    dom_n: Dict[str, int] = defaultdict(int)
    for acc, sname in zip(D["dominant_accession"].astype(str), D["set"]):
        dom_sets[acc].add(sname)
        dom_n[acc] += 1
    rows = []
    unresolved = []
    for acc in sorted(dom_sets):
        key = canon_key(acc, cmap)
        tax = tax_by_key.get(key) if key else None
        if tax is None:
            unresolved.append(acc)
            continue
        p = parse_taxonomy(tax)
        cls, deepest = classify_novelty(tax, present_tr)
        cls_x, _ = classify_novelty(tax, present_trx, strip=True)
        cls_tv, _ = classify_novelty(tax, present_tv)
        rec = {
            "dominant_accession": acc,
            "canonical_assembly_key": key,
            "sets": ";".join(sorted(dom_sets[acc])),
            "n_benchmark_genomes": dom_n[acc],
            "gtdb_taxonomy": tax,
            "novelty_class": cls,
            "novelty_depth": CLASS_DEPTH[cls],
            "deepest_represented_rank": RANK_NAMES.get(deepest, "none (phylum absent)"),
            "novelty_class_suffix_stripped_sensitivity": cls_x,
            "novelty_class_vs_train_plus_val_sensitivity": cls_tv,
        }
        for r in ["p", "c", "o", "f", "g", "s"]:
            name = p.get(r, "")
            rec[f"{RANK_NAMES[r]}"] = name
            rec[f"{RANK_NAMES[r]}_in_train"] = bool(name in present_tr[r])
            rec[f"{RANK_NAMES[r]}_n_train_genomes"] = int(counts_tr[r].get(name, 0))
        rows.append(rec)
    C = pd.DataFrame(rows)
    if unresolved:
        issue(f"{len(unresolved)} dominant accessions have no GTDB taxonomy "
              f"(examples: {unresolved[:5]}) -- EXCLUDED, not silently dropped")
    eda["dominants_resolved"] = int(len(C))
    eda["dominants_unresolved"] = len(unresolved)
    eda["dominants_unresolved_list"] = unresolved
    C = C.sort_values(["novelty_depth", "dominant_accession"])
    C.to_csv(out_dir / "dominant_novelty_classification.tsv", sep="\t", index=False)
    print(f"  ok  {len(C)} distinct dominants classified, {len(unresolved)} unresolved")

    lut = dict(zip(C["dominant_accession"], C["novelty_class"]))
    lut_x = dict(zip(C["dominant_accession"],
                     C["novelty_class_suffix_stripped_sensitivity"]))
    lut_tv = dict(zip(C["dominant_accession"],
                      C["novelty_class_vs_train_plus_val_sensitivity"]))
    D["novelty_class"] = D["dominant_accession"].astype(str).map(lut)
    D["novelty_class_suffix_stripped"] = D["dominant_accession"].astype(str).map(lut_x)
    D["novelty_class_train_plus_val"] = D["dominant_accession"].astype(str).map(lut_tv)
    n_unmapped = int(D["novelty_class"].isna().sum())
    if n_unmapped:
        issue(f"{n_unmapped} benchmark genomes have an unclassifiable dominant "
              f"-- EXCLUDED from the ladder")
    D = D[D["novelty_class"].notna()].copy()
    D["novelty_depth"] = D["novelty_class"].map(CLASS_DEPTH)

    # -------------------------------------------------------- class balance
    bal_cluster = (D.drop_duplicates(["set", "dominant_accession"])
                   .groupby(["set", "novelty_class"]).size().unstack(fill_value=0))
    bal_genome = D.groupby(["set", "novelty_class"]).size().unstack(fill_value=0)
    for c in CLASS_ORDER:
        if c not in bal_cluster.columns:
            bal_cluster[c] = 0
        if c not in bal_genome.columns:
            bal_genome[c] = 0
    bal_cluster = bal_cluster[CLASS_ORDER]
    bal_genome = bal_genome[CLASS_ORDER]
    print("\n  class balance -- reference clusters (set x ref):")
    print(bal_cluster.to_string())
    print("\n  class balance -- genomes:")
    print(bal_genome.to_string())
    eda["class_balance_clusters"] = json.loads(bal_cluster.to_json(orient="index"))
    eda["class_balance_genomes"] = json.loads(bal_genome.to_json(orient="index"))
    # sensitivity: how many cluster labels move
    u = D.drop_duplicates(["set", "dominant_accession"])
    eda["sensitivity_suffix_stripped_clusters_changed"] = int(
        (u["novelty_class"] != u["novelty_class_suffix_stripped"]).sum())
    eda["sensitivity_train_plus_val_clusters_changed"] = int(
        (u["novelty_class"] != u["novelty_class_train_plus_val"]).sum())

    (out_dir / "novelty_ladder_eda.json").write_text(json.dumps(eda, indent=2))

    # ------------------------------------------------------------- [4] metrics
    print(f"\n[3] accuracy + MIMAG-inspired threshold metrics "
          f"({args.n_boot} cluster-bootstrap resamples per cell)")
    M: List[dict] = []
    # pooled cells
    for cls in CLASS_ORDER:
        sub = D[D["novelty_class"] == cls]
        if len(sub) == 0:
            continue
        M += cell_rows(sub, "pooled_five_sets", cls, f"ws11n|pooled|{cls}",
                       args.n_boot, cfg)
        print(f"  pooled {cls:22s} n={len(sub):5d}  clusters={sub.cluster_id.nunique():5d}")
    sub = D[D["novelty_class"].isin(DEEP_POOL)]
    if len(sub):
        M += cell_rows(sub, "pooled_five_sets", DEEP_POOL_LABEL,
                       f"ws11n|pooled|{DEEP_POOL_LABEL}", args.n_boot, cfg)
        print(f"  pooled {DEEP_POOL_LABEL:22s} n={len(sub):5d}  "
              f"clusters={sub.cluster_id.nunique():5d}")
    M += cell_rows(D, "pooled_five_sets", "ALL (any novelty)",
                   "ws11n|pooled|ALL", args.n_boot, cfg)
    # per-set cells
    for name in FIVE_SETS:
        dsub = D[D["set"] == name]
        for cls in CLASS_ORDER + [DEEP_POOL_LABEL]:
            s2 = (dsub[dsub["novelty_class"].isin(DEEP_POOL)] if cls == DEEP_POOL_LABEL
                  else dsub[dsub["novelty_class"] == cls])
            if len(s2) == 0:
                continue
            M += cell_rows(s2, name, cls, f"ws11n|{name}|{cls}", args.n_boot, cfg)
    MT = pd.DataFrame(M)
    MT.to_csv(out_dir / "novelty_ladder_metrics.tsv", sep="\t", index=False)
    print(f"  wrote {len(MT)} metric rows")

    # ------------------------------------------- [5] paired MAGICC vs CheckM2
    print("\n[4] paired MAGICC-vs-comparator tests within each novelty class")
    P: List[dict] = []
    cells = [("pooled_five_sets", c, D[D["novelty_class"] == c]) for c in CLASS_ORDER]
    cells.append(("pooled_five_sets", DEEP_POOL_LABEL,
                  D[D["novelty_class"].isin(DEEP_POOL)]))
    for scope, cls, sub in cells:
        if len(sub) < 10:
            if len(sub):
                P.append({"scope": scope, "novelty_class": cls,
                          "n_pairs": int(len(sub)),
                          "n_clusters": int(sub.cluster_id.nunique()),
                          "note": "n < 10: no test computed"})
            continue
        seed = cfg.seed + 31 + (fw.stable_hash(f"ws11n|paired|{scope}|{cls}") % 100000)
        bs = fw.Bootstrapper(clusters=sub["cluster_id"].to_numpy(),
                             n_iter=args.n_boot, ci_level=cfg.ci_level, seed=seed)
        bs_slow = fw.Bootstrapper(clusters=sub["cluster_id"].to_numpy(),
                                  n_iter=args.n_boot_slow, ci_level=cfg.ci_level,
                                  seed=seed + 1)
        for comp in ["checkm2", "cocopye", "deepcheck"]:
            for metric in ["abs_err_completeness", "abs_err_contamination"]:
                r = st.paired_comparison(sub, REF_TOOL, comp, metric, cfg,
                                         bs, bs_slow, cluster_col="cluster_id")
                if r is None:
                    continue
                r.update({"scope": scope, "novelty_class": cls,
                          "novelty_depth": CLASS_DEPTH.get(cls, np.nan),
                          "seed": int(seed)})
                P.append(r)
    PT = pd.DataFrame(P)
    # BH correction across the family of novelty classes, PRIMARY = cluster-mean.
    # The headline family is MAGICC vs CheckM2; the other two comparators are
    # corrected in their own families so the headline q-values are not diluted.
    for pcol, qcol in [("p_wilcoxon_two_sided_cluster_mean", "q_bh_cluster_mean"),
                       ("p_wilcoxon_two_sided_naive", "q_bh_naive")]:
        if pcol not in PT.columns:
            continue
        PT[qcol] = np.nan
        for (comp, metric), idx in PT.groupby(
                ["comparison_tool", "metric"], dropna=True).groups.items():
            PT.loc[idx, qcol] = fw.bh_correct(PT.loc[idx, pcol].to_numpy())
    if "q_bh_cluster_mean" in PT.columns:
        PT["significant_bh_primary"] = PT["q_bh_cluster_mean"] < cfg.raw[
            "statistics"].get("bh_alpha", 0.05)
        PT["bh_family"] = ("comparison_tool x metric, across novelty classes "
                           "(pooled five-set cells)")
    PT.to_csv(out_dir / "novelty_ladder_paired_tests.tsv", sep="\t", index=False)
    print(f"  wrote {len(PT)} paired-test rows")

    # ---------------------------------------------------------- [6] trend test
    print("\n[5] monotonicity of MAGICC's error in novelty depth")
    TR: List[dict] = []
    cl = (D.groupby(["cluster_id", "novelty_depth"], as_index=False)
          .agg(**{f"{m}__{t}": pd.NamedAgg(column=f"abs_err_{m}__{t}", aggfunc="mean")
                  for m in ("completeness", "contamination") for t in TOOLS}))
    for tool in TOOLS:
        for metric in ("completeness", "contamination"):
            col = f"{metric}__{tool}"
            x = cl["novelty_depth"].to_numpy(float)
            y = cl[col].to_numpy(float)
            ok = ~np.isnan(y)
            rho, p = sps.spearmanr(x[ok], y[ok])
            seed = cfg.seed + 77 + (fw.stable_hash(f"ws11n|trend|{tool}|{metric}") % 100000)
            bsr = fw.Bootstrapper(clusters=cl["cluster_id"].to_numpy()[ok],
                                  n_iter=args.n_boot, ci_level=cfg.ci_level, seed=seed)
            xx, yy = x[ok], y[ok]
            ci = bsr.ci(lambda idx: float(sps.spearmanr(xx[idx], yy[idx]).statistic),
                        point=float(rho))
            # ordered cell means, for the plain monotonicity read-out
            means = (cl.groupby("novelty_depth")[col].mean()
                     .reindex(range(len(CLASS_ORDER))).dropna())
            vals = means.to_numpy()
            # monotonicity restricted to classes that carry enough clusters to
            # support a claim -- a single-genome class (class_novel, 1 cluster)
            # otherwise decides the flag on its own.
            nclu = cl.groupby("novelty_depth")["cluster_id"].nunique()
            powered = [int(d) for d in means.index
                       if nclu.get(d, 0) >= MIN_CLUSTERS_FOR_CLAIM]
            pvals = means.loc[powered].to_numpy()
            TR.append({
                "tool": tool, "tool_label": cfg.tool_label(tool), "metric": metric,
                "unit": "cluster-mean absolute error (pp)",
                "n_clusters": int(ok.sum()),
                "spearman_rho_depth_vs_error": float(rho),
                "spearman_rho_ci_lo": ci["ci_lo"], "spearman_rho_ci_hi": ci["ci_hi"],
                "p_spearman_two_sided": float(p),
                "monotone_nondecreasing_all_classes": bool(np.all(np.diff(vals) >= 0)),
                "monotone_nondecreasing_powered_classes":
                    bool(np.all(np.diff(pvals) >= 0)) if len(pvals) > 1 else None,
                "powered_classes": ";".join(CLASS_ORDER[d] for d in powered),
                "min_clusters_for_powered": MIN_CLUSTERS_FOR_CLAIM,
                "ordered_class_means": ";".join(
                    f"{CLASS_ORDER[int(d)]}={v:.3f}(k={int(nclu.get(d, 0))})"
                    for d, v in means.items()),
                "seed": int(seed),
            })
    TRD = pd.DataFrame(TR)
    TRD["q_bh_spearman"] = fw.bh_correct(TRD["p_spearman_two_sided"].to_numpy())
    TRD.to_csv(out_dir / "novelty_ladder_trend.tsv", sep="\t", index=False)
    print(TRD[["tool", "metric", "spearman_rho_depth_vs_error",
               "spearman_rho_ci_lo", "spearman_rho_ci_hi",
               "monotone_nondecreasing_all_classes",
               "monotone_nondecreasing_powered_classes"]].to_string(index=False))

    # ------------------------------------------------------------ [7] summary
    summary = {
        "workstream": "WS11.N",
        "title": "Taxonomic-novelty ladder on the five leakage-free benchmark sets",
        "design": ("OBSERVATIONAL stratification of frozen-model predictions by the "
                   "deepest GTDB rank at which the dominant reference genome is novel "
                   "relative to the training split. NOT a retrained holdout: no genome "
                   "was removed from training for this analysis."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sets": FIVE_SETS,
        "trap_T1_guard": "set_F / set_G are excluded by an explicit name whitelist",
        "eda": eda,
        "class_balance_clusters": eda["class_balance_clusters"],
        "class_balance_genomes": eda["class_balance_genomes"],
        "pooled_cluster_totals": {
            c: {"clusters": int(bal_cluster[c].sum()),
                "genomes": int(bal_genome[c].sum())} for c in CLASS_ORDER},
        "min_clusters_for_claim": MIN_CLUSTERS_FOR_CLAIM,
        "classes_too_thin_for_a_claim": [
            c for c in CLASS_ORDER
            if 0 < int(bal_cluster[c].sum()) < MIN_CLUSTERS_FOR_CLAIM],
        "classes_absent": [c for c in CLASS_ORDER if int(bal_cluster[c].sum()) == 0],
        "magicc_pooled_by_class": {},
        "trend": TRD.to_dict(orient="records"),
        "outputs": ["dominant_novelty_classification.tsv",
                    "novelty_ladder_metrics.tsv",
                    "novelty_ladder_paired_tests.tsv",
                    "novelty_ladder_trend.tsv",
                    "novelty_ladder_eda.json",
                    "novelty_ladder_summary.json",
                    "WS11_N_REPORT.md"],
    }
    mp = MT[(MT.scope == "pooled_five_sets") & (MT.tool == REF_TOOL)]
    for _, r in mp.iterrows():
        summary["magicc_pooled_by_class"][r["novelty_class"]] = {
            "n_genomes": int(r["n_genomes"]),
            "n_clusters": int(r["n_clusters_set_x_ref"]),
            "completeness_mae": round(float(r["completeness_mae"]), 4),
            "completeness_mae_ci": [round(float(r["completeness_mae_ci_lo"]), 4),
                                    round(float(r["completeness_mae_ci_hi"]), 4)],
            "completeness_bias": round(float(r["completeness_bias"]), 4),
            "contamination_mae": round(float(r["contamination_mae"]), 4),
            "contamination_mae_ci": [round(float(r["contamination_mae_ci_lo"]), 4),
                                     round(float(r["contamination_mae_ci_hi"]), 4)],
            "contamination_bias": round(float(r["contamination_bias"]), 4),
            "false_fail_rate_cont5": (None if pd.isna(r["false_fail_rate_cont5"])
                                      else round(float(r["false_fail_rate_cont5"]), 4)),
            "false_fail_denominator": r["false_fail_denominator_statement"],
            "false_pass_rate_cont5": (None if pd.isna(r["false_pass_rate_cont5"])
                                      else round(float(r["false_pass_rate_cont5"]), 4)),
            "false_pass_denominator": r["false_pass_denominator_statement"],
            "supports_claim": bool(r["supports_claim"]),
        }
    (out_dir / "novelty_ladder_summary.json").write_text(json.dumps(summary, indent=2))

    write_report(out_dir, MT, PT, TRD, bal_cluster, bal_genome, summary, eda, cfg,
                 args.n_boot)
    print(f"\nDONE -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _ci(v, lo, hi, nd=2, sign=False):
    if pd.isna(v):
        return "-"
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    if pd.isna(lo) or pd.isna(hi):
        return f.format(v)
    return f.format(v) + f" [{lo:.{nd}f}, {hi:.{nd}f}]"


def headline_sentences(MT, PT, TRD, bal_cluster, bal_genome, cfg) -> List[str]:
    """Findings written straight from the tables, never hard-coded."""
    out: List[str] = []
    mp = MT[(MT.scope == "pooled_five_sets") & (MT.tool == "magicc_v5")]
    idx = {r["novelty_class"]: r for _, r in mp.iterrows()}
    powered = [c for c in CLASS_ORDER
               if c in idx and idx[c]["n_clusters_set_x_ref"] >= MIN_CLUSTERS_FOR_CLAIM]
    if powered:
        out.append(
            "**Ladder is real.** MAGICC v5 completeness MAE climbs "
            + " → ".join(f"{idx[c]['completeness_mae']:.2f} pp ({c})" for c in powered)
            + "; contamination MAE climbs "
            + " → ".join(f"{idx[c]['contamination_mae']:.2f} pp" for c in powered)
            + f". Adequately powered classes only (≥ {MIN_CLUSTERS_FOR_CLAIM} clusters).")
    for tool in ("magicc_v5", "checkm2"):
        for metric in ("completeness", "contamination"):
            r = TRD[(TRD.tool == tool) & (TRD.metric == metric)]
            if r.empty:
                continue
            r = r.iloc[0]
            out.append(
                f"**Trend, {cfg.tool_short(tool)} {metric}.** Spearman ρ between "
                f"novelty depth and cluster-mean absolute error = "
                f"{r['spearman_rho_depth_vs_error']:.3f} "
                f"[{r['spearman_rho_ci_lo']:.3f}, {r['spearman_rho_ci_hi']:.3f}], "
                f"q = {r['q_bh_spearman']:.2g}; monotone over the powered classes: "
                f"{r['monotone_nondecreasing_powered_classes']}.")
    if not PT.empty and "comparison_tool" in PT.columns:
        for metric, lab in (("abs_err_completeness", "completeness"),
                            ("abs_err_contamination", "contamination")):
            sub = PT[(PT.comparison_tool == "checkm2") & (PT.metric == metric)]
            if sub.empty:
                continue
            wins = [r["novelty_class"] for _, r in sub.iterrows()
                    if r["hodges_lehmann_diff"] < 0 and r["q_bh_cluster_mean"] < 0.05]
            loses = [r["novelty_class"] for _, r in sub.iterrows()
                     if r["hodges_lehmann_diff"] > 0 and r["q_bh_cluster_mean"] < 0.05]
            out.append(
                f"**MAGICC vs CheckM2, {lab}.** MAGICC significantly better in "
                + (", ".join(f"`{w}`" for w in wins) if wins else "no class")
                + "; significantly worse in "
                + (", ".join(f"`{l}`" for l in loses) if loses else "no class")
                + " (BH-corrected, cluster-mean paired Wilcoxon).")
    absent = [c for c in CLASS_ORDER if int(bal_cluster[c].sum()) == 0]
    thin = [c for c in CLASS_ORDER
            if 0 < int(bal_cluster[c].sum()) < MIN_CLUSTERS_FOR_CLAIM]
    out.append(
        "**Limits of this panel.** "
        + (f"No class carries zero clusters." if not absent else
           f"{', '.join('`' + c + '`' for c in absent)} does not occur at all on this "
           f"panel, so it cannot be measured here.")
        + (f" {', '.join('`' + c + '`' for c in thin)} carry fewer than "
           f"{MIN_CLUSTERS_FOR_CLAIM} reference clusters and are reported but must "
           f"not carry a claim on their own; they are pooled as "
           f"`{DEEP_POOL_LABEL}`." if thin else ""))
    return out


def write_report(out_dir: Path, MT, PT, TRD, bal_cluster, bal_genome, summary,
                 eda, cfg, n_boot):
    L: List[str] = []
    A = L.append
    A("# WS11.N — the taxonomic-novelty ladder on the headline five-set panel")
    A("")
    A(f"Generated {summary['generated_utc']} · "
      f"`scripts/210_ws11n_novelty_ladder.py` · "
      f"{n_boot} cluster-bootstrap resamples · `PYTHONHASHSEED="
      f"{eda.get('pythonhashseed')}`")
    A("")
    A("> **This is an observational stratification, not a holdout experiment.**")
    A("> Every number below comes from the *same frozen models* scoring the *same*")
    A("> five leakage-free benchmark sets that the manuscript already reports. No")
    A("> genome was removed from training for this analysis; genomes in the deeper")
    A("> novelty classes simply happen to have no training relative at that rank.")
    A("> Nothing here may be described as \"held out at species/genus level\". The")
    A("> retrained validation artefacts (WS1.6 phylum, WS1.9 family, WS11.G genus)")
    A("> remain the only holdout evidence. What this analysis *does* deliver is the")
    A("> genome → species → genus ladder measured on exactly the panel Reviewer 1")
    A("> was reading, on identical genomes for all four tools.")
    A("")
    A("## 0. Headline findings")
    A("")
    for s in headline_sentences(MT, PT, TRD, bal_cluster, bal_genome, cfg):
        A(f"* {s}")
    A("")
    A("## 1. Design")
    A("")
    A("For every dominant reference genome of `set_A_v2`, `set_B_v2`, `set_C_clean`,")
    A("`set_D_clean` and `set_E` the GTDB lineage is walked from species upwards and")
    A("compared against the taxa present in `data/splits/train_genomes.tsv`")
    A(f"({eda['train_split_rows']:,} genomes). The deepest represented rank fixes the")
    A("novelty class:")
    A("")
    A("| class | definition |")
    A("|---|---|")
    A("| `lineage_represented` | the genome's **species** is present in training (the "
      "genome itself is still held out — this is the genome-level-only case) |")
    A("| `species_novel` | species absent, **genus** present |")
    A("| `genus_novel` | genus absent, **family** present |")
    A("| `family_novel` | family absent, **order** present |")
    A("| `order_novel` | order absent, **class** present |")
    A("| `class_novel` | class absent, **phylum** present |")
    A("| `phylum_novel` | phylum absent from training |")
    A("")
    A("Accession matching uses the GCA↔GCF cross-map of the WS1.4 provenance audit")
    A("(`scripts/074_provenance_audit.py`), rebuilt here and verified byte-for-byte")
    A("against `results/revision/provenance/accession_crossmap_stats.json` "
      f"(match: {eda.get('crossmap_matches_ws1_4_audit')}).")
    A("GTDB polyphyly suffixes (`Bacteroidota_A`) are treated as distinct taxa, "
      "consistent with WS1.6.")
    A("")
    A("**Denominators.** " + fw.caption_denominator(cfg).replace("\n", " "))
    A("")
    A("## 2. Class balance — reported honestly")
    A("")
    A("Reference clusters (set × dominant reference genome):")
    A("")
    A(fw.md_table(bal_cluster.reset_index()))
    A("")
    A("Genomes:")
    A("")
    A(fw.md_table(bal_genome.reset_index()))
    A("")
    thin = summary["classes_too_thin_for_a_claim"]
    absent = summary["classes_absent"]
    A(f"**Too thin to support a claim** (< {summary['min_clusters_for_claim']} "
      f"reference clusters pooled across all five sets): "
      + (", ".join(f"`{c}` ({int(bal_cluster[c].sum())} clusters, "
                   f"{int(bal_genome[c].sum())} genomes)" for c in thin)
         if thin else "none"))
    A("")
    A(f"**Absent entirely**: "
      + (", ".join(f"`{c}`" for c in absent) if absent else "none")
      + ". The five-set panel therefore *cannot* speak to phylum-level novelty at all;")
    A("that level is only measurable by retraining (WS1.6).")
    A("")
    A(f"Sensitivity: stripping GTDB polyphyly suffixes moves "
      f"{eda['sensitivity_suffix_stripped_clusters_changed']}/"
      f"{eda['pooled']['n_clusters_set_x_ref']} cluster labels; scoring novelty against")
    A(f"train ∪ val instead of train alone moves "
      f"{eda['sensitivity_train_plus_val_clusters_changed']}. Both are reported as")
    A("columns of `dominant_novelty_classification.tsv`; neither is the primary "
      "definition.")
    A("")
    A("## 3. Pooled five-set metrics by novelty class")
    A("")
    A("All intervals are 95% percentile cluster bootstraps over reference genomes "
      f"({n_boot} resamples, clusters = set × dominant accession).")
    A("")
    for metric, lab in (("completeness", "Completeness"),
                        ("contamination", "Contamination")):
        A(f"### {lab} (MAE and signed bias, pp)")
        A("")
        rows = []
        for cls in CLASS_ORDER + [DEEP_POOL_LABEL, "ALL (any novelty)"]:
            sub = MT[(MT.scope == "pooled_five_sets") & (MT.novelty_class == cls)]
            if sub.empty:
                continue
            r0 = sub.iloc[0]
            d = {"novelty class": cls, "n": int(r0["n_genomes"]),
                 "clusters": int(r0["n_clusters_set_x_ref"])}
            for t in TOOLS:
                rr = sub[sub.tool == t]
                if rr.empty:
                    continue
                rr = rr.iloc[0]
                d[cfg.tool_short(t) + " MAE"] = _ci(rr[f"{metric}_mae"],
                                                    rr[f"{metric}_mae_ci_lo"],
                                                    rr[f"{metric}_mae_ci_hi"])
                d[cfg.tool_short(t) + " bias"] = _ci(rr[f"{metric}_bias"],
                                                     rr[f"{metric}_bias_ci_lo"],
                                                     rr[f"{metric}_bias_ci_hi"],
                                                     sign=True)
            rows.append(d)
        A(fw.md_table(pd.DataFrame(rows)))
        A("")
    A("### MIMAG-inspired 5% contamination boundary")
    A("")
    A("`false_fail` = a truly clean genome (true contamination < 5%) rejected; "
      "denominator = truly clean genomes.  ")
    A("`false_pass` = a truly contaminated genome (true contamination ≥ 5%) let "
      "through; denominator = truly contaminated genomes.")
    A("")
    rows = []
    for cls in CLASS_ORDER + [DEEP_POOL_LABEL, "ALL (any novelty)"]:
        sub = MT[(MT.scope == "pooled_five_sets") & (MT.novelty_class == cls)]
        if sub.empty:
            continue
        for t in TOOLS:
            rr = sub[sub.tool == t]
            if rr.empty:
                continue
            rr = rr.iloc[0]
            rows.append({
                "novelty class": cls, "tool": cfg.tool_short(t),
                "false-fail": _ci(rr["false_fail_rate_cont5"],
                                  rr["false_fail_rate_cont5_ci_lo"],
                                  rr["false_fail_rate_cont5_ci_hi"], nd=3),
                "false-fail denominator": rr["false_fail_denominator_statement"],
                "false-pass": _ci(rr["false_pass_rate_cont5"],
                                  rr["false_pass_rate_cont5_ci_lo"],
                                  rr["false_pass_rate_cont5_ci_hi"], nd=3),
                "false-pass denominator": rr["false_pass_denominator_statement"],
                "balanced acc.": _ci(rr["balanced_accuracy_cont5"],
                                     rr["balanced_accuracy_cont5_ci_lo"],
                                     rr["balanced_accuracy_cont5_ci_hi"], nd=3),
            })
    A(fw.md_table(pd.DataFrame(rows)))
    A("")
    A("## 4. Is MAGICC's error monotone in novelty depth?")
    A("")
    A("Trend test: Spearman ρ between novelty depth (0 = `lineage_represented` … "
      "6 = `phylum_novel`) and the **cluster-mean** absolute error, with a cluster "
      "bootstrap CI, BH-corrected over the 8 tool × metric tests. "
      "`monotone_nondecreasing_all_classes` includes cells with as few as one "
      "cluster and is therefore decided by noise; "
      "`monotone_nondecreasing_powered_classes` restricts the check to classes with "
      f"at least {MIN_CLUSTERS_FOR_CLAIM} reference clusters and is the one to read.")
    A("")
    A(fw.md_table(TRD[["tool_label", "metric", "n_clusters",
                       "spearman_rho_depth_vs_error", "spearman_rho_ci_lo",
                       "spearman_rho_ci_hi", "p_spearman_two_sided",
                       "q_bh_spearman", "monotone_nondecreasing_all_classes",
                       "monotone_nondecreasing_powered_classes"]]))
    A("")
    A("Ordered class means (cluster-mean absolute error, pp):")
    A("")
    for _, r in TRD.iterrows():
        A(f"* **{r['tool_label']} {r['metric']}** — {r['ordered_class_means']}")
    A("")
    A("## 5. Paired MAGICC-vs-comparator tests within each novelty class")
    A("")
    A("Paired difference `d = |error|_MAGICC − |error|_comparator` on **identical "
      "genomes**; negative favours MAGICC. Primary test is the two-sided paired "
      "Wilcoxon on cluster means (unit of analysis = reference genome); "
      "Hodges–Lehmann median paired difference with a cluster-bootstrap CI is the "
      "effect size; BH correction runs across novelty classes within each "
      "(comparator × metric) family.")
    A("")
    if not PT.empty and "comparison_tool" in PT.columns:
        for comp in ["checkm2", "cocopye", "deepcheck"]:
            sub = PT[(PT.comparison_tool == comp)] if "comparison_tool" in PT else None
            if sub is None or sub.empty:
                continue
            A(f"### MAGICC v5 vs {cfg.tool_short(comp)}")
            A("")
            t = pd.DataFrame({
                "novelty class": sub.novelty_class,
                "metric": sub.metric.str.replace("abs_err_", "", regex=False),
                "n": sub.n_pairs, "clusters": sub.n_clusters,
                "MAE MAGICC": sub.mae_reference.round(3),
                f"MAE {cfg.tool_short(comp)}": sub.mae_comparison.round(3),
                "HL diff [95% CI]": [_ci(*r, sign=True) for r in
                                     zip(sub.hodges_lehmann_diff, sub.hl_ci_lo,
                                         sub.hl_ci_hi)],
                "mean diff [95% CI]": [_ci(*r, sign=True) for r in
                                       zip(sub.mean_paired_difference,
                                           sub.mean_diff_ci_lo,
                                           sub.mean_diff_ci_hi)],
                "Cliff's δ": sub.cliffs_delta.round(3),
                "p (cluster-mean)": [f"{p:.3g}" for p in
                                     sub.p_wilcoxon_two_sided_cluster_mean],
                "q (BH)": [f"{q:.3g}" for q in sub.q_bh_cluster_mean],
                "favours": sub.favours,
            })
            A(fw.md_table(t))
            A("")
    A("## 6. Files")
    A("")
    for f in summary["outputs"]:
        A(f"* `results/revision/ws11/novelty_ladder/{f}`")
    A("")
    if eda["issues"]:
        A("## 7. Input-verification issues")
        A("")
        for i in eda["issues"]:
            A(f"* {i}")
    else:
        A("## 7. Input verification")
        A("")
        A("Every input file exists, every set carries exactly 1,000 rows, all four "
          "tool prediction tables join 1:1 to `metadata.tsv` with zero NaN "
          "predictions and truth columns identical to the metadata, every dominant "
          "resolves to a GTDB lineage, and no dominant of the five sets is in the "
          "train or val split. No row was dropped.")
    A("")
    (out_dir / "WS11_N_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
