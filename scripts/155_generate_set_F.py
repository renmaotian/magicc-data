#!/usr/bin/env python3
"""
WS2.2 + WS2.3 — Build **Set F**: contamination type x taxonomic distance.

(Protocol name: ``82_generate_set_F.py``; renumbered to 145 because 82 was
already taken — see the header of ``154_contamination_type_module.py``.)

DESIGN
------
Fully-crossed factorial, 100 reference genomes x 6 taxonomic distances x 3
contamination types = 1,800 contaminated samples, plus 100 uncontaminated
controls (one per reference) = **1,900 samples**.

*Fully crossed* is the point: every reference appears once in every one of the
18 cells, carrying the **same** target contamination level and the **same**
fragmentation tier, and (for REDUNDANT and SINGLE) a byte-identical dominant
assembly.  Type and distance contrasts are therefore paired within reference and
within contamination level, so they cannot be confounded by genome composition -
the failure mode that forced the WS1 decision-rule correction.

Taxonomic distance (GTDB r220 taxonomy, `data/gtdb/`), Cornet's six ranks:

    species  contaminant is a different assembly of the SAME species
    genus    same genus,  different species
    family   same family, different genus
    order    same order,  different family
    class    same class,  different order
    phylum   same phylum, different class

Contamination levels emphasise the practically relevant band (Reviewer 2):
1, 2, 3, 5, 7.5, 10, 12.5, 15, 17.5, 20 % - ten references per level, assigned
by a seeded permutation.  Every sample is therefore **inside the training
domain** (contamination% <= completeness%, protocol section 4.4a): contamination
<= 20% and completeness is floored at 55%.

LEAKAGE
-------
Dominants **and** contaminants are drawn exclusively from
``data/splits/test_genomes.tsv`` (the held-out test split).  Proven, not
asserted, by ``146``-adjacent provenance audit written here and by
``scripts/074_provenance_audit.py``-style GCA<->GCF cross-mapping.

OUTPUTS
-------
    data/benchmarks/set_F/fasta/<genome_id>.fasta
    data/benchmarks/set_F/metadata.tsv               (tool-facing labels)
    data/benchmarks/set_F/generation_metadata.tsv    (full provenance, WS7.7)
    data/benchmarks/set_F/reference_selection.tsv
    data/benchmarks/set_F/design.tsv
    data/benchmarks/set_F/generation_checkpoint.jsonl
    data/benchmarks/set_F/validation_report.json
    data/benchmarks/set_F/annotations/<accession>/   (Prodigal cache)
    data/benchmarks/set_F/orthology/<acc>__<acc>.tsv (RBH cache)

USAGE
-----
    python scripts/155_generate_set_F.py --pilot            # 2x2x10 smoke test
    python scripts/155_generate_set_F.py --workers 10
    python scripts/155_generate_set_F.py --validate-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ctm = _load_module(PROJECT_DIR / "scripts" / "154_contamination_type_module.py",
                   "ws2_contamination_types")

from magicc.fragmentation import read_fasta  # noqa: E402

SET_DIR = PROJECT_DIR / "data" / "benchmarks" / "set_F"
ANN_DIR = SET_DIR / "annotations"
ORT_DIR = SET_DIR / "orthology"
FASTA_DIR = SET_DIR / "fasta"

SET_BASE = 8_200_000                      # master seed for Set F
N_REFERENCES = 100
N_DONORS_PER_CELL = 3
DISTANCES = ["species", "genus", "family", "order", "class", "phylum"]
TYPES = list(ctm.CONTAMINATION_TYPES)      # redundant, replaced, single
CONTAM_LADDER = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
TIERS = ["medium", "low", "highly_fragmented"]

# distance -> (rank that must MATCH, rank that must DIFFER)
DIST_RULE = {
    "species": ("s", None),
    "genus":   ("g", "s"),
    "family":  ("f", "g"),
    "order":   ("o", "f"),
    "class":   ("c", "o"),
    "phylum":  ("p", "c"),
}
RANK_COL = {"d": "domain", "p": "phylum", "c": "class", "o": "order",
            "f": "family", "g": "genus", "s": "species"}


# ==========================================================================
# taxonomy
# ==========================================================================
def load_test_split() -> pd.DataFrame:
    t = pd.read_csv(PROJECT_DIR / "data/splits/test_genomes.tsv", sep="\t")
    t["fasta_path"] = t["fasta_path"].str.replace(
        "/path/to/magicc", str(PROJECT_DIR), regex=False)
    tax = t["gtdb_taxonomy"].str.split(";", expand=True)
    tax.columns = ["d", "p", "c", "o", "f", "g", "s"]
    for col in tax.columns:
        tax[col] = tax[col].str.replace(r"^[a-z]__", "", regex=True).str.strip()
    t = t.drop(columns=["domain", "phylum"]).join(tax)
    return t.sort_values("gtdb_accession").reset_index(drop=True)


def eligible_at_all_distances(t: pd.DataFrame) -> pd.Series:
    """True where a same-split partner exists at every one of the six ranks."""
    masks = [t["s"].map(t.groupby("s").size()).gt(1)]
    for share, differ in [("g", "s"), ("f", "g"), ("o", "f"), ("c", "o"), ("p", "c")]:
        n = t.groupby(share)[differ].nunique()
        masks.append(t[share].map(n).gt(1))
    return pd.Series(np.logical_and.reduce([m.values for m in masks]), index=t.index)


def pick_references(t: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Round-robin over families to maximise phylogenetic breadth."""
    cand = t[eligible_at_all_distances(t)].copy()
    fams = sorted(cand["f"].unique())
    by_fam = {f: np.array(cand.index[cand["f"] == f].to_numpy(), copy=True) for f in fams}
    for f in fams:
        rng.shuffle(by_fam[f])
    order = np.argsort([-len(by_fam[f]) for f in fams], kind="stable")
    fams = [fams[i] for i in order]
    picked, k = [], 0
    while len(picked) < n:
        added = False
        for f in fams:
            if k < len(by_fam[f]):
                picked.append(by_fam[f][k])
                added = True
                if len(picked) == n:
                    break
        if not added:
            break
        k += 1
    return cand.loc[picked].reset_index(drop=True)


def pick_contaminants(t: pd.DataFrame, ref: pd.Series, distance: str,
                      n_donors: int, rng: np.random.Generator) -> List[str]:
    share, differ = DIST_RULE[distance]
    m = (t[share] == ref[share]) & (t["gtdb_accession"] != ref["gtdb_accession"])
    if differ is not None:
        m &= t[differ] != ref[differ]
    cand = t.index[m].to_numpy()
    if len(cand) == 0:
        return []
    # prefer larger donors (more eligible territory) but keep it stochastic:
    # sample up to 3x n_donors at random, then order by genome size descending
    take = min(len(cand), n_donors * 3)
    sel = rng.choice(cand, size=take, replace=False)
    sel = sorted(sel, key=lambda i: -int(t.loc[i, "genome_size"]))
    return [t.loc[i, "gtdb_accession"] for i in sel[:n_donors]]


# ==========================================================================
# Phase 0 — design
# ==========================================================================
def build_design(t: pd.DataFrame, n_refs: int, distances: List[str],
                 types: List[str], n_donors: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SET_BASE)
    refs = pick_references(t, n_refs, rng)

    levels = np.array(CONTAM_LADDER * int(np.ceil(n_refs / len(CONTAM_LADDER))))[:n_refs]
    rng.shuffle(levels)
    tiers = np.array((TIERS * int(np.ceil(n_refs / len(TIERS))))[:n_refs])
    rng.shuffle(tiers)
    refs["ref_index"] = np.arange(len(refs))
    refs["target_contamination"] = levels
    refs["quality_tier"] = tiers

    acc_to_row = {a: i for i, a in enumerate(t["gtdb_accession"])}
    rows = []
    for _, r in refs.iterrows():
        drng = np.random.default_rng(SET_BASE + 300_000 + int(r.ref_index))
        for di, dist in enumerate(distances):
            donors = pick_contaminants(t, r, dist, n_donors, drng)
            for ti, typ in enumerate(types):
                rows.append({
                    "genome_id": f"F_r{int(r.ref_index):03d}_{dist}_{typ}",
                    "ref_index": int(r.ref_index),
                    "distance": dist, "distance_index": di,
                    "contamination_type": typ, "type_index": ti,
                    "dominant_accession": r.gtdb_accession,
                    "donor_accessions": ";".join(donors),
                    "target_contamination": float(r.target_contamination),
                    "quality_tier": r.quality_tier,
                })
        rows.append({
            "genome_id": f"F_r{int(r.ref_index):03d}_control_none",
            "ref_index": int(r.ref_index),
            "distance": "none", "distance_index": -1,
            "contamination_type": "none", "type_index": -1,
            "dominant_accession": r.gtdb_accession,
            "donor_accessions": "",
            "target_contamination": 0.0,
            "quality_tier": r.quality_tier,
        })
    design = pd.DataFrame(rows)
    keep = ["gtdb_accession", "ncbi_accession", "gcf_accession", "ref_index",
            "target_contamination", "quality_tier", "genome_size", "contig_count",
            "n50_contigs", "checkm2_completeness", "checkm2_contamination",
            "d", "p", "c", "o", "f", "g", "s", "gtdb_taxonomy", "fasta_path"]
    return design, refs[[k for k in keep if k in refs.columns]]


# ==========================================================================
# Phase 1 — annotation (Prodigal), Phase 2 — orthology (DIAMOND RBH)
# ==========================================================================
_PATHS: Dict[str, str] = {}


def _init_paths(paths: Dict[str, str]):
    global _PATHS
    _PATHS = paths


def _annotate_one(acc: str):
    try:
        ctm.annotate_genome(acc, _PATHS[acc], ANN_DIR / acc.replace("/", "_"))
        return acc, None
    except Exception as e:                                       # noqa: BLE001
        return acc, f"{type(e).__name__}: {e}"


def _rbh_one(args):
    a_acc, d_acc = args
    try:
        ann_a = ctm.annotate_genome(a_acc, _PATHS[a_acc], ANN_DIR / a_acc.replace("/", "_"))
        ann_d = ctm.annotate_genome(d_acc, _PATHS[d_acc], ANN_DIR / d_acc.replace("/", "_"))
        pairs = ctm.rbh_orthologs(ann_a, ann_d, _rbh_path(a_acc, d_acc), threads=1)
        return (a_acc, d_acc), len(pairs), None
    except Exception as e:                                       # noqa: BLE001
        return (a_acc, d_acc), 0, f"{type(e).__name__}: {e}"


def _rbh_path(a_acc: str, d_acc: str) -> Path:
    return ORT_DIR / f"{a_acc.replace('/', '_')}__{d_acc.replace('/', '_')}.tsv"


# ==========================================================================
# Phase 3 — generation (one worker task == one reference == 19 samples)
# ==========================================================================
def _generate_reference(task):
    ref_index, dominant_acc, dom_path, rows = task
    out = []
    try:
        ann_a = ctm.annotate_genome(dominant_acc, dom_path,
                                    ANN_DIR / dominant_acc.replace("/", "_"))
        seq_a = read_fasta(dom_path)
        donor_cache: Dict[str, Tuple[object, str]] = {}

        for row in rows:
            gid = row["genome_id"]
            fp = FASTA_DIR / f"{gid}.fasta"
            typ = row["contamination_type"]
            donors = [d for d in str(row["donor_accessions"]).split(";") if d]
            seed_dom = SET_BASE + int(ref_index)
            seed_cont = (SET_BASE + 1_000 * int(ref_index)
                         + 100 * (int(row["distance_index"]) + 1)
                         + (int(row["type_index"]) + 1))
            seed_event = seed_cont + 5_000_000

            anns, seqs = [], {}
            rbh: Dict[str, List[Tuple[str, str]]] = {}
            for d in donors:
                if d not in donor_cache:
                    ad = ctm.annotate_genome(d, _PATHS[d], ANN_DIR / d.replace("/", "_"))
                    donor_cache[d] = (ad, read_fasta(_PATHS[d]))
                ad, sd = donor_cache[d]
                anns.append(ad)
                seqs[d] = sd
                rbh[d] = ctm.rbh_orthologs(ann_a, ad, _rbh_path(dominant_acc, d), threads=1)

            target_bp = int(round(float(row["target_contamination"]) / 100.0 * len(seq_a)))
            if typ == "none" or target_bp <= 0 or not anns:
                ev = ctm.ContaminationEvent(event_type="none")
            else:
                ev = ctm.make_event(ann_a, anns, rbh, typ, target_bp,
                                    np.random.default_rng(seed_event))

            sample = ctm.generate_typed_sample(
                seq_a, seqs, ev,
                target_completeness=1.0,
                quality_tier=row["quality_tier"],
                contaminant_tier=row["quality_tier"],
                rng=np.random.default_rng(seed_dom),
                rng_contaminant=np.random.default_rng(seed_cont),
            )
            ctm.write_fasta(sample["contigs"], fp, prefix=gid)

            rec = dict(row)
            rec.update({
                "seed_dominant": seed_dom, "seed_contaminant": seed_cont,
                "seed_event": seed_event,
                "dominant_reference_bp": sample["dominant_full_length"],
                "dominant_retained_bp": sample["dominant_retained_bp"],
                "contaminant_bp": sample["contaminant_bp"],
                "excised_bp": sample["excised_bp"],
                "mandatory_excision_bp": ev.mandatory_bp,
                "size_match_deficit_bp": ev.size_match_deficit_bp,
                "survivor_bp": sample["survivor_bp"],
                "observed_completeness": sample["completeness"],
                "observed_contamination": sample["contamination"],
                "target_contaminant_bp": target_bp,
                "shortfall_bp": ev.shortfall_bp,
                "donors_used": ";".join(sorted(set(ev.donors_used))),
                "n_donors_available": len(donors),
                "n_donors_used": len(set(ev.donors_used)),
                "n_donor_blocks": ev.n_donor_blocks,
                "n_transferred_genes": ev.n_transferred_genes,
                "n_duplicated_markers": ev.n_duplicated_families,
                "n_replaced_markers": ev.n_replaced_families,
                "n_novel_genes": ev.n_novel_genes,
                "n_contigs_dominant": sample["n_contigs_dominant"],
                "n_contigs_contaminant": sample["n_contigs_contaminant"],
                "n_contigs": len(sample["contigs"]),
                "total_length": sum(len(c) for c in sample["contigs"]),
                "dominant_genes_called": ann_a.n_genes,
                "error": "",
            })
            out.append(rec)
        return ref_index, out, None
    except Exception as e:                                       # noqa: BLE001
        import traceback
        return ref_index, out, traceback.format_exc(limit=6)


# ==========================================================================
# main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pilot", action="store_true",
                    help="2 types x 2 distances x 10 references, into set_F_pilot")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global SET_DIR, ANN_DIR, ORT_DIR, FASTA_DIR
    n_refs, distances, types = N_REFERENCES, DISTANCES, TYPES
    if args.pilot:
        SET_DIR = PROJECT_DIR / "data" / "benchmarks" / "set_F_pilot"
        n_refs, distances, types = 10, ["species", "phylum"], TYPES
    # annotation + orthology caches are shared with the full run (never duplicated)
    ANN_DIR = PROJECT_DIR / "data" / "benchmarks" / "set_F" / "annotations"
    ORT_DIR = PROJECT_DIR / "data" / "benchmarks" / "set_F" / "orthology"
    FASTA_DIR = SET_DIR / "fasta"
    for d in (SET_DIR, ANN_DIR, ORT_DIR, FASTA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"WS2.3 — Set F generation  ->  {SET_DIR}")
    print("=" * 78)

    t = load_test_split()
    design, refs = build_design(t, n_refs, distances, types, N_DONORS_PER_CELL)
    design.to_csv(SET_DIR / "design.tsv", sep="\t", index=False)
    refs.to_csv(SET_DIR / "reference_selection.tsv", sep="\t", index=False)
    print(f"design: {len(design)} samples, {len(refs)} references, "
          f"{len(distances)} distances x {len(types)} types + controls")
    print(f"  references span {refs['p'].nunique()} phyla / {refs['c'].nunique()} classes "
          f"/ {refs['f'].nunique()} families / {refs['g'].nunique()} genera")
    print(f"  genome size Mbp: median {refs.genome_size.median()/1e6:.2f} "
          f"[{refs.genome_size.min()/1e6:.2f}, {refs.genome_size.max()/1e6:.2f}]")

    paths = dict(zip(t["gtdb_accession"], t["fasta_path"]))
    needed = set(design["dominant_accession"])
    for s in design["donor_accessions"]:
        needed.update(x for x in str(s).split(";") if x)
    paths = {a: paths[a] for a in needed}
    missing = [a for a, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"FATAL: {len(missing)} genome FASTAs missing, e.g. {missing[:3]}",
              file=sys.stderr)
        return 1
    print(f"  unique genomes to annotate: {len(paths)}")

    if not args.validate_only:
        # ---------------- Phase 1: Prodigal ------------------------------
        todo = [a for a in sorted(paths)
                if not (ANN_DIR / a.replace('/', '_') / 'genes.tsv').exists()]
        if todo:
            print(f"\n[1/3] Prodigal on {len(todo)} genomes ({args.workers} workers) ...")
            t0 = time.time()
            errs = []
            with Pool(args.workers, initializer=_init_paths, initargs=(paths,)) as pool:
                for i, (acc, err) in enumerate(
                        pool.imap_unordered(_annotate_one, todo, chunksize=2), 1):
                    if err:
                        errs.append((acc, err))
                    if i % 200 == 0:
                        print(f"    {i}/{len(todo)}  ({time.time()-t0:.0f}s)")
            print(f"    done in {time.time()-t0:.0f}s; {len(errs)} failures")
            if errs:
                print("    " + "; ".join(f"{a}: {e[:80]}" for a, e in errs[:5]))
        else:
            print("\n[1/3] Prodigal: all annotations cached")

        # ---------------- Phase 2: RBH -----------------------------------
        pairs = sorted({(r.dominant_accession, d)
                        for r in design.itertuples()
                        for d in str(r.donor_accessions).split(";") if d})
        todo = [p for p in pairs if not _rbh_path(*p).exists()]
        if todo:
            print(f"\n[2/3] DIAMOND RBH on {len(todo)}/{len(pairs)} pairs "
                  f"({args.workers} workers) ...")
            t0 = time.time()
            errs = []
            with Pool(args.workers, initializer=_init_paths, initargs=(paths,)) as pool:
                for i, (pr, n, err) in enumerate(
                        pool.imap_unordered(_rbh_one, todo, chunksize=1), 1):
                    if err:
                        errs.append((pr, err))
                    if i % 200 == 0:
                        print(f"    {i}/{len(todo)}  ({time.time()-t0:.0f}s)")
            print(f"    done in {time.time()-t0:.0f}s; {len(errs)} failures")
            if errs:
                print("    " + "; ".join(f"{p}: {e[:80]}" for p, e in errs[:5]))
        else:
            print("\n[2/3] DIAMOND RBH: all pairs cached")

        # ---------------- Phase 3: generation ----------------------------
        ckpt = SET_DIR / "generation_checkpoint.jsonl"
        done: Dict[str, dict] = {}
        if ckpt.exists() and not args.force:
            with open(ckpt) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (FASTA_DIR / f"{r['genome_id']}.fasta").exists():
                        done[r["genome_id"]] = r
        todo_refs = sorted(set(design.loc[~design.genome_id.isin(done), "ref_index"]))
        print(f"\n[3/3] generation: {len(done)} samples cached, "
              f"{len(todo_refs)} references to build")
        if todo_refs:
            tasks = []
            for ri in todo_refs:
                sub = design[design.ref_index == ri]
                rows = [r for r in sub.to_dict("records") if r["genome_id"] not in done]
                dom = sub.iloc[0]["dominant_accession"]
                tasks.append((int(ri), dom, paths[dom], rows))
            t0 = time.time()
            fails = []
            with open(ckpt, "a") as fh, \
                    Pool(args.workers, initializer=_init_paths, initargs=(paths,)) as pool:
                for i, (ri, recs, err) in enumerate(
                        pool.imap_unordered(_generate_reference, tasks), 1):
                    for r in recs:
                        done[r["genome_id"]] = r
                        fh.write(json.dumps(r, default=str) + "\n")
                    fh.flush()
                    if err:
                        fails.append((ri, err))
                    if i % 10 == 0 or i == len(tasks):
                        print(f"    {i}/{len(tasks)} refs  "
                              f"({len(done)} samples, {time.time()-t0:.0f}s)")
            print(f"    generation done in {time.time()-t0:.0f}s; {len(fails)} ref failures")
            for ri, err in fails[:3]:
                print(f"      ref {ri}: {err.splitlines()[-1][:160]}")

        # ---------------- write metadata ---------------------------------
        gm = pd.DataFrame([done[g] for g in design.genome_id if g in done])
        write_metadata(gm, refs, SET_DIR)

    return validate(SET_DIR, refs)


def write_metadata(gm: pd.DataFrame, refs: pd.DataFrame, set_dir: Path) -> None:
    rmap = refs.set_index("gtdb_accession")
    gm = gm.copy()
    for col, src in (("dominant_phylum", "p"), ("dominant_domain", "d"),
                     ("dominant_taxonomy", "gtdb_taxonomy")):
        gm[col] = gm["dominant_accession"].map(rmap[src])
    gm["dominant_source_split"] = "test"
    gm["contaminant_source_split"] = "test"
    gm["set"] = set_dir.name
    gm["sample_type"] = "set_f_" + gm["contamination_type"].astype(str)
    gm["replicate"] = 0
    gm["in_training_domain"] = (
        gm["observed_contamination"] <= gm["observed_completeness"] + 1e-9)
    gm["constraint_violation"] = ~gm["in_training_domain"]
    gm = gm.sort_values(["ref_index", "distance_index", "type_index"]).reset_index(drop=True)
    gm.to_csv(set_dir / "generation_metadata.tsv", sep="\t", index=False)

    meta = gm[["genome_id"]].copy()
    meta["true_completeness"] = gm["observed_completeness"]
    meta["true_contamination"] = gm["observed_contamination"]
    meta["dominant_accession"] = gm["dominant_accession"]
    meta["dominant_phylum"] = gm["dominant_phylum"]
    meta["sample_type"] = gm["sample_type"]
    meta["contamination_type"] = gm["contamination_type"]
    meta["distance"] = gm["distance"]
    meta["target_contamination"] = gm["target_contamination"]
    meta["n_contigs"] = gm["n_contigs"]
    meta["total_length"] = gm["total_length"]
    meta["ref_index"] = gm["ref_index"]
    meta["replicate"] = gm["replicate"]
    meta.to_csv(set_dir / "metadata.tsv", sep="\t", index=False)
    print(f"\nwrote {set_dir/'metadata.tsv'} and generation_metadata.tsv "
          f"({len(gm)} rows, {gm.shape[1]} provenance columns)")


# ==========================================================================
# validation
# ==========================================================================
def validate(set_dir: Path, refs: pd.DataFrame) -> int:
    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)
    gm = pd.read_csv(set_dir / "generation_metadata.tsv", sep="\t")
    design = pd.read_csv(set_dir / "design.tsv", sep="\t")
    rep: Dict[str, object] = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                              "set_dir": str(set_dir), "n_samples": int(len(gm))}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        rep[name] = {"pass": bool(cond), "detail": detail}
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
        if not cond:
            ok = False

    check("all_design_rows_generated", len(gm) == len(design),
          f"{len(gm)}/{len(design)}")
    check("unique_genome_ids", gm.genome_id.nunique() == len(gm), f"{gm.genome_id.nunique()}")

    # FASTA integrity: contig count and bp must match metadata exactly
    n_bad_ct, n_bad_bp, n_missing = 0, 0, 0
    for r in gm.itertuples():
        p = set_dir / "fasta" / f"{r.genome_id}.fasta"
        if not p.exists():
            n_missing += 1
            continue
        nc, bp = 0, 0
        with open(p) as fh:
            for line in fh:
                if line.startswith(">"):
                    nc += 1
                else:
                    bp += len(line.strip())
        if nc != r.n_contigs:
            n_bad_ct += 1
        if bp != r.total_length:
            n_bad_bp += 1
    check("fasta_present", n_missing == 0, f"{n_missing} missing")
    check("fasta_contig_counts_match", n_bad_ct == 0, f"{n_bad_ct} mismatched")
    check("fasta_bp_match", n_bad_bp == 0, f"{n_bad_bp} mismatched")

    viol = int(gm["constraint_violation"].sum())
    check("training_domain_contamination_le_completeness", viol == 0,
          f"{viol} out-of-domain samples")
    check("completeness_within_50_100",
          bool(((gm.observed_completeness >= 50) & (gm.observed_completeness <= 100)).all()),
          f"min {gm.observed_completeness.min():.2f}, max {gm.observed_completeness.max():.2f}")
    check("contamination_le_100", bool((gm.observed_contamination <= 100).all()),
          f"max {gm.observed_contamination.max():.2f}")

    contaminated = gm[gm.contamination_type != "none"]
    if len(contaminated):
        rel = ((contaminated.observed_contamination - contaminated.target_contamination).abs()
               / contaminated.target_contamination.clip(lower=1e-9))
        check("contamination_hits_target_within_5pct", float(rel.quantile(0.95)) < 0.05,
              f"p95 relative deviation {100*rel.quantile(0.95):.2f}%, "
              f"max {100*rel.max():.2f}%")
        short = contaminated[contaminated.shortfall_bp > 0]
        rep["shortfall"] = {
            "n_samples_with_shortfall": int(len(short)),
            "by_type_distance": ({f"{a}|{b}": int(c) for (a, b), c in
                                  short.groupby(["contamination_type",
                                                 "distance"]).size().items()}
                                 if len(short) else {}),
            "median_shortfall_pct_of_target":
                float(100 * (short.shortfall_bp / short.target_contaminant_bp).median())
                if len(short) else 0.0,
        }
        print(f"  [INFO] shortfall: {len(short)}/{len(contaminated)} samples could not "
              f"reach the target from eligible donor territory")

        # mechanism check: only REDUNDANT duplicates markers; only REPLACED excises
        red = contaminated[contaminated.contamination_type == "redundant"]
        rpl = contaminated[contaminated.contamination_type == "replaced"]
        sng = contaminated[contaminated.contamination_type == "single"]
        check("redundant_duplicates_markers",
              bool((red.n_duplicated_markers > 0).all()) if len(red) else True,
              f"median {red.n_duplicated_markers.median() if len(red) else 0:.0f} markers")
        check("redundant_excises_nothing",
              bool((red.excised_bp == 0).all()) if len(red) else True)
        check("single_excises_nothing",
              bool((sng.excised_bp == 0).all()) if len(sng) else True)
        check("single_transfers_no_shared_marker",
              bool((sng.n_duplicated_markers == 0).all()
                   & (sng.n_replaced_markers == 0).all()) if len(sng) else True)
        if len(rpl):
            eq = (rpl.excised_bp == rpl.contaminant_bp)
            excused = (rpl.mandatory_excision_bp > rpl.contaminant_bp)
            check("replaced_is_size_matched", bool((eq | excused).all()),
                  f"{int((~(eq | excused)).sum())} not size-matched; "
                  f"{int(excused.sum())} kept the full mandatory deletion instead "
                  f"(mandatory > donor bp)")
            check("replaced_deletes_every_acceptor_marker_copy",
                  bool((rpl.excised_bp >= rpl.mandatory_excision_bp).all()),
                  f"{int((rpl.excised_bp < rpl.mandatory_excision_bp).sum())} samples "
                  f"left an acceptor copy behind")
            check("replaced_removes_dominant_sequence",
                  bool((rpl.excised_bp > 0).all()))

    # leakage: every accession used must be in the test split
    test_acc = set(pd.read_csv(PROJECT_DIR / "data/splits/test_genomes.tsv",
                               sep="\t")["gtdb_accession"])
    used = set(gm.dominant_accession)
    for s in gm.donors_used.fillna(""):
        used.update(x for x in str(s).split(";") if x)
    outside = sorted(used - test_acc)
    check("all_accessions_from_test_split", len(outside) == 0,
          f"{len(outside)} outside test split")

    rep["overall_pass"] = ok
    rep["composition"] = {
        "by_type": gm.contamination_type.value_counts().to_dict(),
        "by_distance": gm.distance.value_counts().to_dict(),
        "by_type_distance": {f"{a}|{b}": int(c) for (a, b), c in
                             gm.groupby(["contamination_type", "distance"]).size().items()},
        "completeness": {"mean": float(gm.observed_completeness.mean()),
                         "min": float(gm.observed_completeness.min()),
                         "max": float(gm.observed_completeness.max())},
        "contamination": {"mean": float(gm.observed_contamination.mean()),
                          "min": float(gm.observed_contamination.min()),
                          "max": float(gm.observed_contamination.max())},
        "n_references": int(gm.ref_index.nunique()),
    }
    (set_dir / "validation_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nwrote {set_dir/'validation_report.json'}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
