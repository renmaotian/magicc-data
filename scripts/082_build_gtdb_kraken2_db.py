#!/usr/bin/env python3
"""
WS4.3 (CONDITIONAL -- STOOD DOWN, NOT RUN) -- build a Kraken2 database from this
project's own GTDB reference set, to decide the capacity-vs-novelty question.

=============================================================================
STATUS: STOOD DOWN 2026-07-26. THIS SCRIPT WAS NEVER EXECUTED AT FULL SCALE.
=============================================================================
Verified only at small scale: a 300-genome test build completed successfully
(exit 0, 4,923 sequences, 274 Mbp -> 470 MB hash, 3 min at 6 threads), which
confirms the taxonomy/library/seqid2taxid.map construction and the kraken2-build
invocation are correct. The full build was deliberately NOT run.

Decision and rationale (auditable):
  * The withdrawal of the manuscript's Kraken2 claim does NOT depend on the
    capacity-vs-novelty verdict. It rests on four already-complete findings:
      1. the metric collapses to 0.00% median contamination under a minimal
         evidence requirement (>=0.5% confidence, >=2 informative k-mers);
      2. the matched-cohort comparison shows no difference between cohorts
         (Hodges-Lehmann +0.80 pp, 95% CI [-2.96, +5.24]; Cliff's delta +0.034),
         with 72.1% of agreed-clean MAGs exceeding the 5% threshold;
      3. the V5-vs-metric correlation dissolves under novelty control
         (rho +0.201, p=0.017 -> partial rho +0.110, p=0.199);
      4. the database-independent SCG assay finds the cohorts indistinguishable,
         with the surviving association favouring CheckM2, not V5.
  * The mechanism is already established by the positive-control calibration in
    scripts/83: reference genomes from sparse lineages that we KNOW are in the
    reference set show informative-k-mer density 0.00049, against the cohort's
    0.00050. That is the explanatory result.
  * A GTDB-level index would only refine a characterization of something no
    longer being claimed, at ~4.5 h of heavily contended CPU, while the true
    critical path lies elsewhere (GPU holdout retrain; real-data acquisition).

Consequence to state in the paper: we did NOT attempt a GTDB-level Kraken2
index, so the distinction between database CAPACITY and genuine TAXONOMIC
NOVELTY remains UNRESOLVED (scripts/81 reports
verdict = 'undecided_needs_gtdb_level_db'). This does not affect the withdrawal,
which rests on the four findings above rather than on any positive claim about
novelty.

Original rationale for the design follows.

Why a custom database is required
---------------------------------
The published Kraken2 metric was computed against ``k2_standard_08gb``, a
*capped* build. Re-testing with the full uncapped ``k2_standard`` removes the
capacity objection, but **not** the coverage objection: k2_standard is
RefSeq-derived (archaea, bacteria, viral, plasmid, human, UniVec_Core) and
contains essentially no MAG-derived references. The 141-MAG cohort is dominated
by uncultivated, MAG-derived lineages -- exactly the material RefSeq lacks and
GTDB has. A low informative-k-mer density against RefSeq therefore licenses only
"no RefSeq reference exists", never "genuinely novel".

Building from ``data/genomes/`` answers a sharper and more useful question than
any generic database: it measures whether these MAGs are novel **relative to the
exact reference set MAGICC was trained on** -- which is the quantity that bears
on whether V5's contamination estimates can be trusted for them.

Reference set
-------------
``data/gtdb/selected_100k_genomes.tsv``: 100,000 GTDB genomes (110 phyla,
22,721 distinct species-level lineages), FASTAs under
``data/genomes/<accession>/<accession>_*_genomic.fna`` (~361 GB).

How it works
------------
``kraken2-build`` normally derives sequence->taxid from NCBI accession2taxid,
which does not apply to GTDB taxonomy. Instead we construct a self-contained
taxonomy from the GTDB lineage strings and supply the mapping directly:

  1. Parse GTDB lineages into a rank tree (kingdom/phylum/class/order/family/
     genus/species) and emit NCBI-format ``taxonomy/nodes.dmp`` and
     ``taxonomy/names.dmp`` with synthetic taxids.
  2. Populate ``library/`` with **symlinks** to the existing FASTAs (no 361 GB
     copy) and write ``seqid2taxid.map`` mapping every sequence ID to its
     genome's GTDB species taxid.  ``build_kraken2_db.sh`` uses a pre-existing
     ``seqid2taxid.map`` verbatim and skips accession lookup.
  3. Run ``kraken2-build --build`` (uncapped -- capping is the very thing under
     test).

Cost warning
------------
Two full passes over the library (estimate_capacity, then build_db): ~720 GB of
reads, and the hash is held in RAM. Expect several hours and a few hundred GB of
RAM. Run only when the conditional trigger fires, and not alongside other
disk-heavy jobs.

Subsetting: decision and rationale (2026-07-26)
-----------------------------------------------
``--one-per-species`` cuts input ~4.4x (22,721 vs 100,000 genomes; ~4.5 h vs
~20 h at 16 threads) but LOWERS measured k-mer density, which biases the
analysis *towards* concluding novelty -- the conclusion that happens to suit the
narrative. That bias is the reason it is not silently default.

**Decision: use ``--one-per-species`` TOGETHER WITH internal positive controls.**
Rationale:

  * GTDB species representatives ARE the standard GTDB reference set -- most
    tools that consume GTDB use exactly that level -- so a 22,721-genome
    species-representative database is legitimately "GTDB-level" and can carry
    ``is_gtdb_level: True`` honestly.
  * The additional ~77,000 genomes buy strain-level resolution, which is not the
    quantity at issue. The question is whether these MAGs have *any*
    phylum-to-genus-level relative in GTDB; species representatives answer that.
  * The positive controls (``scripts/83_kmer_density_controls.py``) neutralise
    the bias entirely: judging the 141 MAGs' informative-k-mer density against
    density measured on genomes we *know* the database represents converts an
    absolute threshold into a relative one. That is the methodologically correct
    comparison regardless, and is stronger than the full build without controls.
  * The full 100k build is reserved for a genuinely ambiguous outcome -- i.e. the
    141 sitting *between* the positive controls and the floor rather than clearly
    at one end.

Usage
-----
    # prepare only (fast; inspect before committing to the build)
    python scripts/82_build_gtdb_kraken2_db.py --prepare-only

    # prepare and build
    python scripts/82_build_gtdb_kraken2_db.py --threads 16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
GENOME_TSV = PROJECT_DIR / 'data' / 'gtdb' / 'selected_100k_genomes.tsv'
GENOME_DIR = PROJECT_DIR / 'data' / 'genomes'
DB_DIR = PROJECT_DIR / 'tools' / 'kraken2_db_gtdb'

RANK_PREFIX = OrderedDict([
    ('d__', 'kingdom'), ('p__', 'phylum'), ('c__', 'class'),
    ('o__', 'order'), ('f__', 'family'), ('g__', 'genus'), ('s__', 'species'),
])
FIRST_TAXID = 1_000_000


# ---------------------------------------------------------------------------
def parse_lineage(s: str) -> List[Tuple[str, str]]:
    """'d__Bacteria;p__X;...' -> [('kingdom','Bacteria'), ('phylum','X'), ...]"""
    out = []
    for part in s.split(';'):
        part = part.strip()
        for pref, rank in RANK_PREFIX.items():
            if part.startswith(pref):
                name = part[len(pref):].strip()
                if name:
                    out.append((rank, name))
                break
    return out


def build_taxonomy(rows: List[Dict[str, str]]
                   ) -> Tuple[Dict[str, int], List[Tuple[int, int, str, str]]]:
    """
    Returns (accession -> species taxid, nodes) where nodes is a list of
    (taxid, parent_taxid, rank, name).
    """
    next_id = FIRST_TAXID
    # key: tuple of lineage names up to that rank -> taxid
    node_id: Dict[Tuple[str, ...], int] = {}
    nodes: List[Tuple[int, int, str, str]] = [(1, 1, 'no rank', 'root')]
    acc2taxid: Dict[str, int] = {}
    n_bad = 0

    for r in rows:
        lin = parse_lineage(r.get('gtdb_taxonomy', ''))
        if not lin:
            n_bad += 1
            continue
        parent = 1
        key: Tuple[str, ...] = ()
        for rank, name in lin:
            key = key + (name,)
            if key not in node_id:
                next_id += 1
                node_id[key] = next_id
                nodes.append((next_id, parent, rank, name))
            parent = node_id[key]
        acc = r.get('ncbi_accession') or r.get('gtdb_accession', '')
        if acc:
            acc2taxid[acc] = parent   # deepest rank reached (species)
    if n_bad:
        print(f"  WARNING: {n_bad} rows had no parsable GTDB lineage")
    return acc2taxid, nodes


def write_taxonomy(nodes: List[Tuple[int, int, str, str]], tax_dir: Path) -> None:
    tax_dir.mkdir(parents=True, exist_ok=True)
    with open(tax_dir / 'nodes.dmp', 'w') as f:
        for taxid, parent, rank, _ in nodes:
            # NCBI nodes.dmp: tax_id | parent | rank | embl code | division id | ...
            f.write(f"{taxid}\t|\t{parent}\t|\t{rank}\t|\t\t|\t0\t|\t1\t|\t11\t|"
                    f"\t1\t|\t0\t|\t1\t|\t0\t|\t0\t|\t\t|\n")
    with open(tax_dir / 'names.dmp', 'w') as f:
        for taxid, _, _, name in nodes:
            f.write(f"{taxid}\t|\t{name}\t|\t\t|\tscientific name\t|\n")


# ---------------------------------------------------------------------------
def find_fasta(acc: str) -> Optional[Path]:
    d = GENOME_DIR / acc
    if not d.is_dir():
        return None
    for pat in ('*_genomic.fna', '*.fna', '*.fa', '*.fasta'):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def scan_headers(args: Tuple[str, str, int]) -> Tuple[str, List[Tuple[str, int]], int]:
    """Return (accession, [(seqid, taxid)], total_bp) for one genome."""
    acc, path, taxid = args
    pairs: List[Tuple[str, int]] = []
    total = 0
    try:
        with open(path) as f:
            for line in f:
                if line.startswith('>'):
                    sid = line[1:].split()[0] if line[1:].split() else ''
                    if sid:
                        pairs.append((sid, taxid))
                else:
                    total += len(line) - 1
    except OSError as e:
        print(f"  WARN {acc}: {e}")
    return acc, pairs, total


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--db-dir', default=str(DB_DIR))
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--one-per-species', action='store_true',
                    help='cut input ~4.4x by keeping one genome per GTDB species. '
                         'NOTE: lowers measured k-mer density, biasing the analysis '
                         'TOWARDS a novelty conclusion. Off by default.')
    ap.add_argument('--limit', type=int, default=0, help='debug: cap genomes')
    ap.add_argument('--prepare-only', action='store_true',
                    help='write taxonomy/library/seqid2taxid.map, print the build '
                         'command, and stop')
    ap.add_argument('--kraken2-build',
                    default='/path/to/conda/envs/kraken2_env/bin/kraken2-build')
    args = ap.parse_args()

    db = Path(args.db_dir)
    lib = db / 'library'
    tax = db / 'taxonomy'
    lib.mkdir(parents=True, exist_ok=True)
    tax.mkdir(parents=True, exist_ok=True)

    with open(GENOME_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    print(f"reference genomes in metadata: {len(rows):,}")

    if args.one_per_species:
        seen = set()
        keep = []
        for r in rows:
            lin = r.get('gtdb_taxonomy', '')
            if lin in seen:
                continue
            seen.add(lin)
            keep.append(r)
        print(f"  --one-per-species: {len(keep):,} kept "
              f"(BIASES TOWARDS NOVELTY -- must be stated)")
        rows = keep
    if args.limit:
        rows = rows[:args.limit]

    print("Building GTDB taxonomy ...")
    acc2taxid, nodes = build_taxonomy(rows)
    write_taxonomy(nodes, tax)
    ranks: Dict[str, int] = {}
    for _, _, rank, _ in nodes:
        ranks[rank] = ranks.get(rank, 0) + 1
    print(f"  taxonomy nodes: {len(nodes):,}  " +
          '  '.join(f'{k}={v:,}' for k, v in ranks.items() if k != 'no rank'))
    print(f"  wrote {tax / 'nodes.dmp'} and {tax / 'names.dmp'}")

    # ---- library symlinks + seqid2taxid.map ----
    print("Locating FASTAs and linking into library/ ...")
    jobs = []
    missing = 0
    for r in rows:
        acc = r.get('ncbi_accession') or r.get('gtdb_accession', '')
        taxid = acc2taxid.get(acc)
        fa = find_fasta(acc) if acc else None
        if fa is None or taxid is None:
            missing += 1
            continue
        link = lib / f'{acc}.fna'
        if not link.exists():
            try:
                link.symlink_to(fa.resolve())
            except FileExistsError:
                pass
        jobs.append((acc, str(fa), taxid))
    print(f"  linked {len(jobs):,} genomes   missing/unmapped: {missing:,}")

    print(f"Scanning sequence headers with {args.threads} worker(s) "
          f"(one full read of the library) ...")
    t0 = time.time()
    n_seq = 0
    total_bp = 0
    with open(db / 'seqid2taxid.map', 'w') as out:
        with Pool(processes=max(1, args.threads)) as pool:
            for i, (acc, pairs, bp) in enumerate(
                    pool.imap_unordered(scan_headers, jobs, chunksize=16)):
                for sid, taxid in pairs:
                    out.write(f'{sid}\t{taxid}\n')
                n_seq += len(pairs)
                total_bp += bp
                if (i + 1) % 10000 == 0:
                    print(f"  {i + 1:,}/{len(jobs):,} genomes  "
                          f"{n_seq:,} sequences  {total_bp / 1e9:.1f} Gbp  "
                          f"({time.time() - t0:.0f}s)")
    print(f"  seqid2taxid.map: {n_seq:,} sequences, {total_bp / 1e9:.1f} Gbp total "
          f"({time.time() - t0:.0f}s)")

    manifest = {
        'reference_metadata': str(GENOME_TSV),
        'n_genomes_in_metadata': len(rows),
        'n_genomes_linked': len(jobs),
        'n_genomes_missing': missing,
        'n_sequences': n_seq,
        'total_bp': total_bp,
        'total_gbp': round(total_bp / 1e9, 2),
        'n_taxonomy_nodes': len(nodes),
        'rank_counts': ranks,
        'one_per_species': args.one_per_species,
        'subset_bias_note': (
            'one_per_species lowers informative k-mer density and therefore biases '
            'the capacity-vs-novelty analysis TOWARDS a novelty conclusion. This is '
            'neutralised by the internal positive controls in '
            'scripts/83_kmer_density_controls.py, which convert the absolute density '
            'threshold into a relative comparison against genomes the database is '
            'known to represent.'
            if args.one_per_species else 'full reference set used; no subset bias'),
        'is_gtdb_level': True,
        'is_gtdb_level_justification': (
            'GTDB species representatives are the standard GTDB reference level and '
            'are what most GTDB-consuming tools use. The question here is whether '
            'these MAGs have ANY phylum-to-genus-level relative in GTDB, which '
            'species representatives answer; the remaining ~77k genomes add only '
            'strain-level resolution.'),
        'requires_positive_controls': bool(args.one_per_species),
        'db_dir': str(db),
        'uncapped': True,
        'note': ('No --max-db-size: capping is the artifact under test, so the '
                 'database must be uncapped.'),
    }
    (db / 'build_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {db / 'build_manifest.json'}")

    build_cmd = [args.kraken2_build, '--build', '--db', str(db),
                 '--threads', str(args.threads)]
    print(f"\nBuild command:\n  {' '.join(build_cmd)}")
    if args.prepare_only:
        print("\n--prepare-only: stopping before the build.")
        return 0

    print(f"\nRunning kraken2-build (this is the multi-hour step) ...")
    t0 = time.time()
    log = db / 'kraken2_build.log'
    with open(log, 'w') as lf:
        proc = subprocess.run(build_cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=str(db.parent))
    dt = time.time() - t0
    print(f"  kraken2-build exit={proc.returncode} in {dt / 60:.1f} min "
          f"(log: {log})")
    if proc.returncode != 0:
        print(log.read_text()[-3000:], file=sys.stderr)
        return proc.returncode

    for f in ('hash.k2d', 'opts.k2d', 'taxo.k2d'):
        p = db / f
        print(f"  {f}: {p.stat().st_size:,} bytes" if p.is_file()
              else f"  {f}: MISSING")
    manifest['build_seconds'] = round(dt, 1)
    manifest['hash_k2d_bytes'] = ((db / 'hash.k2d').stat().st_size
                                  if (db / 'hash.k2d').is_file() else None)
    manifest['build_command'] = ' '.join(build_cmd)
    (db / 'build_manifest.json').write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
