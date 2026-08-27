#!/usr/bin/env python3
"""
WS2.1 — Contamination-type event generators (redundant / replaced / single),
following Cornet & Baurain 2022 and Cornet et al. 2024 (CRACOT) semantics.

NOTE ON SCRIPT NUMBERING
------------------------
The v6 protocol names this module ``81_contamination_type_module.py``.  Numbers
81-83 were already taken by the Kraken2 database-capacity work
(``81_kraken2_db_capacity_comparison.py``, ``82_build_gtdb_kraken2_db.py``,
``83_kmer_density_controls.py``) before WS2 started, so WS2 uses 144-147:

    144_contamination_type_module.py   <- protocol 81
    145_generate_set_F.py              <- protocol 82
    146_run_tools_set_F.sh             <- protocol 83
    147_analyze_contamination_types.py <- protocol 84


==========================================================================
OPERATIONAL DEFINITIONS  (stated explicitly; a reviewer who knows the Cornet
papers will check these)
==========================================================================

SOURCE SEMANTICS (what the papers actually say)
----------------------------------------------
Cornet & Baurain (2022) *Genome Biology* 23:60, "Contamination detection in
genomic data: more is not enough", Supplementary Note 1 ("Creation of chimeric
genomes").  Two genomes (a "master"/acceptor and a "sub"/donor) are annotated
with Prodigal; OrthoFinder infers orthologous groups; groups holding exactly one
sequence from each genome (i.e. single-copy orthologues) are randomly sampled.
Two contamination flavours are then built at 5% of total proteins each:

  * "non-redundant": genes in the master genome are **replaced** by the
    corresponding sequences of the second genome;
  * "redundant": genes of the master genome are **duplicated** by genes from
    the sub-genome (the master keeps its own copy).

Cornet et al. (2024) *Applied Microbiology* 4(1):9 (the CRACOT framework;
preprint bioRxiv 2022.11.14.516442) generalises this to **three** basic event
types and to **six taxonomic ranks** (phylum -> species).  Verbatim from the
CRACOT Results and Methods (preprint text, verified 2026-07-27):

  * "The first type is redundant contamination that occurs when the contaminant
    sequence is redundant with an homologous genomic sequence of the expected
    organism.  The second type is replaced contamination that is similar to the
    first one, but with the genuine sequence of the expected organism lacking
    from its genome.  The third type is single contamination that occurs when
    the contaminant sequence has naturally no homologous sequence within the
    genome of the expected organism."
  * Orthology: "Proteins were then predicted with Prodigal V2.6.3 ... Finally,
    OrthoFinder V2.5.4 ... was used for orthologous inference."  "Common
    proteins were defined as proteins present in only one copy for both the main
    and slave genome in the OG while single proteins were singletons of the
    slave genome."
  * "Duplicated contamination events were fished from the pool of common OGs,
    and the corresponding gene sequences of the slave genome were added to the
    end of the last contig of the master genome (with a serie of five 'N' added
    to either side of the gene).  Replaced contamination events were also fished
    from the pool of common OGs but slave genes replaced the genuine genes
    within the main genome.  Single contamination events were fished from the
    pool of singletons of the slave organism, and the corresponding gene
    sequences were added to the end of the last contig of the master genome."
  * Pairing: "for one specific taxonomic rank, ranging from phylum to species.
    When a rank is selected, the two genomes should belong to the same taxon at
    this rank, but have a different taxonomy starting with the next rank."
    (NCBI Taxonomy; 705 reference genomes from class Clostridia and genus
    Lactobacillus.)

Their headline result: "only redundant contamination is accurately estimated".
CheckM "overestimated the redundant CL ... but quite logically, does not detect
replaced or single contamination"; CheckM2 "largely overestimated the redundant
CL at all ranks and, as for CheckM, underdetected replaced and single
contamination events".

Third prior-work anchor - GUNC (Orakov et al. 2021, *Genome Biology* 22:178) -
independently reported the same phenomenon with a bp-level simulation that is
closer to ours than CRACOT's gene-level one, and is the actual source of the
"size-matched" phrasing (do NOT attribute it to Cornet):

    "Non-redundant contamination was simulated by replacing part of an acceptor
     genome by a size-matched contaminant fraction of a donor genome; to
     simulate redundant contamination, surplus donor fragments were added to
     complete acceptor genomes.  We observed that CheckM systematically
     overestimated completeness and underestimated contamination for genomes
     with simulated non-redundant contamination, largely independent of the
     taxonomic level of source genome divergence."

Our contribution is quantification of that underestimation for a k-mer/deep-
learning estimator versus three modern tools, at speed and resolved across six
ranks in the practically relevant 0-20% band - not the discovery of the
phenomenon (protocol section 8, rule 5).

OUR OPERATIONAL DEFINITIONS (the exact thing implemented here)
--------------------------------------------------------------
MAGICC's ground truth is expressed in **base pairs against the dominant
genome's full reference length** (completeness = retained dominant bp / dominant
full reference bp; contamination = contaminant bp / dominant full reference bp).
Cornet's framework is expressed in **genes**.  We therefore implement bp-level
analogues that preserve the *mechanism* that distinguishes the three types, and
we state the adaptation rather than hiding it.

Let A be the acceptor (dominant) reference and D the donor (contaminant)
reference.  Genes are called with Prodigal 2.6.3 (`-p single`).  Orthology
between A and D is defined operationally by **reciprocal best DIAMOND blastp
hits** (RBH; e-value <= 1e-5, >= 30% identity, >= 50% query coverage) - the
standard cheap surrogate for the single-copy orthologous groups that
OrthoFinder returns in the source papers.  This gives, exactly as in CRACOT:

    common genes  = donor genes with an RBH partner in the acceptor
    singletons    = donor genes with no RBH partner in the acceptor

A *donor block* is a maximal run of consecutive same-class donor genes on one
donor contig, together with the intergenic sequence between them (blocks break
at the first gene of the other class, and at contig ends).  Blocks are the unit
of contamination: they are realistic contiguous stretches of foreign DNA, not
bare CDSs.

  REDUNDANT   Donor blocks built from **common genes** are ADDED to the acceptor
              assembly.  The acceptor is left untouched, so every transferred
              marker is now present twice - once from A, once from D.  This is
              the only type that produces the marker-duplication signal that
              CheckM/CheckM2-style estimators are built on.
              (= CRACOT "redundant"/"duplicated"; = GUNC "adding surplus donor
              fragments to complete acceptor genomes".)

  REPLACED    Donor blocks built from **common genes** are ADDED, and a
              **size-matched** amount of acceptor sequence is simultaneously
              DELETED - the deletion is anchored on the acceptor's own RBH
              partners of exactly those transferred genes and expanded around
              them until the deleted bp equals the added bp.  Each transferred
              marker therefore ends up present exactly ONCE, but sourced from
              the wrong organism, and the total assembly size is unchanged.
              Ground-truth completeness falls by the replaced fraction; this is
              real missing dominant sequence, not a labelling artefact.
              (= CRACOT "replaced" ("slave genes replaced the genuine genes
              within the main genome"); = GUNC "replacing part of an acceptor
              genome by a size-matched contaminant fraction of a donor genome".)

  SINGLE      Donor blocks built from **singletons** - donor genes with no
              acceptor counterpart - are ADDED.  The acceptor is untouched.
              No marker is duplicated and no marker is replaced: the assembly
              simply gains foreign accessory DNA.
              (= CRACOT "single": "fished from the pool of singletons of the
              slave organism".)

DELIBERATE DEVIATIONS FROM THE SOURCE PAPERS, AND WHY
-----------------------------------------------------
1. RBH instead of OrthoFinder orthogroups.  OrthoFinder on ~700 genomes x 6
   pairings is not affordable inside this revision, and RBH is the accepted
   surrogate for one-to-one orthology between two genomes.  Consequence: RBH is
   slightly conservative (misses some orthologues), which inflates the singleton
   pool.  That biases the *single* class towards including a few genes that are
   in fact divergent orthologues - i.e. it makes "single" marginally easier for
   marker-based tools to catch, so any underestimation we measure for "single"
   is a lower bound on the true underestimation.  Direction of bias stated.
2. Blocks (contiguous multi-gene segments + intergenic DNA), not bare CDSs.
   Contaminating DNA in a real MAG arrives as contigs, and the tools under test
   consume nucleotide FASTA.  Transferring bare CDSs would create an assembly
   no assembler could produce and would be unfair to every tool.
2b. Donor material is emitted as its OWN contigs, fragmented on the same
   empirical contig-length model as the dominant and then shuffled into the
   assembly - it is NOT appended to the end of the acceptor's last contig with
   'N' padding as CRACOT does.  Appending would place foreign and native genes
   on one physical contig, which (a) no assembler produces and (b) would
   systematically penalise contig-aware detectors (GUNC) while flattering
   contig-agnostic ones.  This deviation is closer to GUNC's fragment-level
   simulation and is the fairer test.
3. Contamination expressed in bp, not in % of proteins, because that is the
   denominator the manuscript, its ground truth and all four tools use
   (protocol section 9.2 / R1-M4).  Both are reported per sample.
4. The acceptor reference is treated as one concatenated sequence (as
   ``magicc.fragmentation.read_fasta`` already does everywhere in this project),
   so the surviving pieces after a REPLACED excision are re-concatenated before
   fragmentation.  This creates the same class of artificial junction that the
   existing production generator already creates between reference contigs; it
   is not new to this module.

MODULE REUSE (WS2 requirement: no parallel implementation)
----------------------------------------------------------
All fragmentation, completeness, coverage-dropout, GC-biased-loss and repeat-
exclusion behaviour is imported directly from ``magicc.fragmentation`` - the
same production code path Sets A-E used - so Set F assemblies are structurally
identical to the rest of the benchmark suite.

``magicc.contamination.fragment_contaminant`` is deliberately NOT used, and this
is a semantic requirement rather than a preference: that function fragments the
whole contaminant genome and then samples *random* contigs until a bp target is
met.  Set F must transfer *designated* sequence - blocks built specifically from
common genes (redundant/replaced) or from singletons (single) - because the
identity of the transferred genes is the entire experimental variable.  Random
contig sampling would mix the three classes and destroy the contrast.
``split_into_contigs`` therefore cuts an already-chosen block, and it does so
with ``magicc.fragmentation.QUALITY_TIERS``, i.e. the same empirical contig-
length model, so contig structure remains comparable across sets.  The ground
truth (contaminant bp / dominant full reference bp) is the production
convention, unchanged.

The module is import-safe: ``145_generate_set_F.py`` imports it.  Running it
directly executes a self-test on a small pair of real test-split genomes.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from magicc.fragmentation import (  # noqa: E402
    QUALITY_TIERS,
    apply_completeness,
    apply_coverage_dropout,
    apply_gc_biased_loss,
    apply_repeat_exclusion,
    fragment_genome,
    generate_contig_lengths,
    read_fasta,
    simulate_fragmentation,
)

# MAGICC V5 predicts completeness in [50, 100] only, and was trained where
# contamination% <= completeness% (protocol section 4.4a).  Set F keeps every
# sample inside that domain; this floor is enforced, not hoped for.
MIN_COMPLETENESS_PCT = 55.0

# --------------------------------------------------------------------------
# external binaries
# --------------------------------------------------------------------------
PRODIGAL_BIN = os.environ.get("MAGICC_PRODIGAL", "prodigal")
# diamond ships in the checkm2_py39 / gunc_env conda environments
_DIAMOND_CANDIDATES = [
    os.environ.get("MAGICC_DIAMOND", ""),
    "/path/to/conda/envs/gunc_env/bin/diamond",
    "/path/to/conda/envs/checkm2_py39/bin/diamond",
    "diamond",
]

CONTAMINATION_TYPES = ("redundant", "replaced", "single")

# RBH thresholds
RBH_EVALUE = 1e-5
RBH_MIN_PIDENT = 30.0
RBH_MIN_QCOV = 50.0


def diamond_bin() -> str:
    for c in _DIAMOND_CANDIDATES:
        if not c:
            continue
        p = shutil.which(c) if os.path.sep not in c else (c if os.path.exists(c) else None)
        if p:
            return p
    raise RuntimeError("diamond binary not found; set MAGICC_DIAMOND")


# ==========================================================================
# 1. Annotation
# ==========================================================================
@dataclass
class GenomeAnnotation:
    """Prodigal genes mapped onto the *concatenated* genome coordinate system.

    ``read_fasta`` concatenates the FASTA's contigs in file order, so a gene at
    (contig c, 1-based start s, end e) sits at [offset(c)+s-1, offset(c)+e) in
    the concatenated 0-based half-open coordinate system used everywhere here.
    """

    accession: str
    fasta_path: str
    total_bp: int
    contig_names: List[str]
    contig_lens: List[int]
    contig_offsets: List[int]
    gene_ids: List[str]
    gene_contig: np.ndarray          # index into contig_names
    gene_start: np.ndarray           # concatenated, 0-based inclusive
    gene_end: np.ndarray             # concatenated, 0-based exclusive
    faa_path: str

    @property
    def n_genes(self) -> int:
        return len(self.gene_ids)


def _parse_fasta_contigs(path: str) -> Tuple[List[str], List[int]]:
    names, lens = [], []
    cur_name, cur_len = None, 0
    opener = open
    if str(path).endswith(".gz"):
        import gzip

        opener = gzip.open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_name is not None:
                    names.append(cur_name)
                    lens.append(cur_len)
                cur_name = line[1:].split()[0]
                cur_len = 0
            else:
                cur_len += len(line.strip())
    if cur_name is not None:
        names.append(cur_name)
        lens.append(cur_len)
    keep = [i for i, L in enumerate(lens) if L > 0]
    return [names[i] for i in keep], [lens[i] for i in keep]


def annotate_genome(accession: str, fasta_path: str, out_dir: Path,
                    force: bool = False) -> GenomeAnnotation:
    """Run Prodigal (cached) and return genes in concatenated coordinates."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faa = out_dir / "proteins.faa"
    genes_tsv = out_dir / "genes.tsv"
    contigs_tsv = out_dir / "contigs.tsv"

    if force or not (faa.exists() and genes_tsv.exists() and contigs_tsv.exists()):
        names, lens = _parse_fasta_contigs(fasta_path)
        with open(contigs_tsv + ".tmp" if False else str(contigs_tsv) + ".tmp", "w") as fh:
            fh.write("contig\tlength\n")
            for n, L in zip(names, lens):
                fh.write(f"{n}\t{L}\n")
        os.replace(str(contigs_tsv) + ".tmp", contigs_tsv)

        with tempfile.TemporaryDirectory(dir=str(out_dir)) as td:
            tmp_faa = Path(td) / "p.faa"
            tmp_gff = Path(td) / "p.gff"
            cmd = [PRODIGAL_BIN, "-i", str(fasta_path), "-a", str(tmp_faa),
                   "-f", "gff", "-o", str(tmp_gff), "-p", "single", "-q"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not tmp_faa.exists():
                # very small / unusual genomes: fall back to metagenomic mode
                cmd[-2:] = ["meta", "-q"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"prodigal failed for {accession}: {r.stderr[:400]}")
            rows = []
            with open(tmp_gff) as fh:
                for line in fh:
                    if line.startswith("#") or not line.strip():
                        continue
                    f = line.rstrip("\n").split("\t")
                    if len(f) < 9 or f[2] != "CDS":
                        continue
                    attrs = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
                    gid = f"{f[0]}_{attrs.get('ID', '0').split('_')[-1]}"
                    rows.append((gid, f[0], int(f[3]), int(f[4]), f[6]))
            with open(str(genes_tsv) + ".tmp", "w") as fh:
                fh.write("gene_id\tcontig\tstart\tend\tstrand\n")
                for gid, c, s, e, st in rows:
                    fh.write(f"{gid}\t{c}\t{s}\t{e}\t{st}\n")
            os.replace(str(genes_tsv) + ".tmp", genes_tsv)
            shutil.move(str(tmp_faa), str(faa))

    # ---- load
    names, lens = [], []
    with open(contigs_tsv) as fh:
        next(fh)
        for line in fh:
            n, L = line.rstrip("\n").split("\t")
            names.append(n)
            lens.append(int(L))
    offsets, acc = [], 0
    for L in lens:
        offsets.append(acc)
        acc += L
    idx_of = {n: i for i, n in enumerate(names)}

    gids, gc, gs, ge = [], [], [], []
    with open(genes_tsv) as fh:
        next(fh)
        for line in fh:
            gid, c, s, e, _st = line.rstrip("\n").split("\t")
            if c not in idx_of:
                continue
            ci = idx_of[c]
            gids.append(gid)
            gc.append(ci)
            gs.append(offsets[ci] + int(s) - 1)
            ge.append(offsets[ci] + int(e))
    order = np.argsort(np.asarray(gs, dtype=np.int64), kind="stable")
    gids = [gids[i] for i in order]
    return GenomeAnnotation(
        accession=accession, fasta_path=str(fasta_path), total_bp=acc,
        contig_names=names, contig_lens=lens, contig_offsets=offsets,
        gene_ids=gids,
        gene_contig=np.asarray(gc, dtype=np.int64)[order],
        gene_start=np.asarray(gs, dtype=np.int64)[order],
        gene_end=np.asarray(ge, dtype=np.int64)[order],
        faa_path=str(faa),
    )


# ==========================================================================
# 2. Orthology: reciprocal best DIAMOND blastp hits
# ==========================================================================
def _diamond_best_hits(query_faa: str, subject_faa: str, threads: int,
                       tmpdir: str) -> Dict[str, Tuple[str, float]]:
    db = os.path.join(tmpdir, "subj.dmnd")
    dbin = diamond_bin()
    subprocess.run([dbin, "makedb", "--in", subject_faa, "-d", db,
                    "--threads", str(threads), "--quiet"],
                   check=True, capture_output=True, text=True)
    out = os.path.join(tmpdir, "hits.tsv")
    subprocess.run(
        [dbin, "blastp", "-q", query_faa, "-d", db, "-o", out,
         "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore", "qcovhsp",
         "--max-target-seqs", "1", "--evalue", str(RBH_EVALUE),
         "--threads", str(threads), "--quiet"],
        check=True, capture_output=True, text=True)
    best: Dict[str, Tuple[str, float]] = {}
    with open(out) as fh:
        for line in fh:
            q, s, pid, ev, bits, qcov = line.rstrip("\n").split("\t")
            if float(pid) < RBH_MIN_PIDENT or float(qcov) < RBH_MIN_QCOV:
                continue
            b = float(bits)
            if q not in best or b > best[q][1]:
                best[q] = (s, b)
    return best


def rbh_orthologs(ann_a: GenomeAnnotation, ann_b: GenomeAnnotation,
                  cache_path: Path, threads: int = 1,
                  force: bool = False) -> List[Tuple[str, str]]:
    """Reciprocal-best-hit one-to-one orthologues between A (acceptor) and B (donor).

    Cached as a two-column TSV so the whole pipeline is resumable.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        pairs = []
        with open(cache_path) as fh:
            next(fh)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        a2b = _diamond_best_hits(ann_a.faa_path, ann_b.faa_path, threads, td)
    with tempfile.TemporaryDirectory() as td:
        b2a = _diamond_best_hits(ann_b.faa_path, ann_a.faa_path, threads, td)
    pairs = []
    for qa, (sb, _) in a2b.items():
        back = b2a.get(sb)
        if back is not None and back[0] == qa:
            pairs.append((qa, sb))
    pairs.sort()
    tmp = str(cache_path) + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("acceptor_gene\tdonor_gene\n")
        for a, b in pairs:
            fh.write(f"{a}\t{b}\n")
    os.replace(tmp, cache_path)
    return pairs


# ==========================================================================
# 3. Donor blocks
# ==========================================================================
def build_donor_blocks(ann: GenomeAnnotation, eligible_mask: np.ndarray,
                       pad: int = 200) -> List[Tuple[int, int, List[int]]]:
    """Maximal runs of consecutive eligible donor genes on one contig.

    Returns [(start, end, [gene indices]), ...] in concatenated coordinates.
    A block is padded by ``pad`` bp on each side but never crosses a contig
    boundary and never swallows a neighbouring ineligible gene.
    """
    blocks: List[Tuple[int, int, List[int]]] = []
    n = ann.n_genes
    if n == 0:
        return blocks
    i = 0
    while i < n:
        if not eligible_mask[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n and eligible_mask[j + 1]
               and ann.gene_contig[j + 1] == ann.gene_contig[i]):
            j += 1
        ci = int(ann.gene_contig[i])
        lo_bound = ann.contig_offsets[ci]
        hi_bound = ann.contig_offsets[ci] + ann.contig_lens[ci]
        # do not swallow the previous / next ineligible gene
        if i - 1 >= 0 and ann.gene_contig[i - 1] == ci:
            lo_bound = max(lo_bound, int(ann.gene_end[i - 1]))
        if j + 1 < n and ann.gene_contig[j + 1] == ci:
            hi_bound = min(hi_bound, int(ann.gene_start[j + 1]))
        s = max(lo_bound, int(ann.gene_start[i]) - pad)
        e = min(hi_bound, int(ann.gene_end[j]) + pad)
        if e > s:
            blocks.append((s, e, list(range(i, j + 1))))
        i = j + 1
    return blocks


# ==========================================================================
# 4. Interval helpers
# ==========================================================================
def merge_intervals(iv: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not iv:
        return []
    s = sorted(iv)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total_bp(iv: Sequence[Tuple[int, int]]) -> int:
    return int(sum(b - a for a, b in iv))


def extract(seq: str, iv: Sequence[Tuple[int, int]]) -> List[str]:
    return [seq[a:b] for a, b in iv]


def complement_intervals(iv: Sequence[Tuple[int, int]], total: int) -> List[Tuple[int, int]]:
    out, pos = [], 0
    for a, b in merge_intervals(iv):
        if a > pos:
            out.append((pos, a))
        pos = max(pos, b)
    if pos < total:
        out.append((pos, total))
    return out


# ==========================================================================
# 5. Contaminant fragment splitting (bp-preserving)
# ==========================================================================
def split_into_contigs(seq: str, tier: str, rng: np.random.Generator) -> List[str]:
    """Cut a donor block into contigs of tier-appropriate length. bp preserved."""
    L = len(seq)
    if L == 0:
        return []
    _cmin, _cmax, n50_lo, n50_hi, min_bp = QUALITY_TIERS[tier]
    if L <= min_bp:
        return [seq]
    target_n50 = int(rng.integers(n50_lo, n50_hi + 1))
    mu = np.log(min(target_n50, max(L, 2)))
    sigma = float(rng.uniform(0.8, 1.2))
    out, pos = [], 0
    while pos < L:
        piece = int(max(min_bp, rng.lognormal(mu, sigma)))
        if L - (pos + piece) < min_bp:
            piece = L - pos
        piece = min(piece, L - pos)
        out.append(seq[pos:pos + piece])
        pos += piece
    return [c for c in out if c]


# ==========================================================================
# 6. The three event generators
# ==========================================================================
def _fragment_with_floor(sequence: str, target_completeness: float, quality_tier: str,
                         rng: np.random.Generator, min_completeness: float
                         ) -> Dict[str, object]:
    """``simulate_fragmentation`` with an explicit completeness floor.

    Identical code path to ``magicc.fragmentation.simulate_fragmentation`` (same
    contig-length model, same coverage/GC/repeat biases, same completeness
    filter); the only difference is that the 0.50 floor hardcoded there becomes
    a parameter, so a REPLACED sample whose acceptor already lost ``r`` of its
    sequence can still be held above MAGICC's 50% prediction floor.
    """
    full = len(sequence)
    if full == 0:
        return {"contigs": [], "completeness": 0.0, "quality_tier": quality_tier}
    lengths = generate_contig_lengths(full, quality_tier, rng)
    contigs = fragment_genome(sequence, lengths, rng)
    pre_bias = list(contigs)
    contigs = apply_coverage_dropout(contigs, rng)
    contigs = apply_gc_biased_loss(contigs, rng)
    contigs = apply_repeat_exclusion(contigs, rng)
    contigs, actual = apply_completeness(contigs, target_completeness, full, rng,
                                         min_completeness=min_completeness)
    if actual < min_completeness:
        contigs, actual = apply_completeness(pre_bias, target_completeness, full, rng,
                                             min_completeness=min_completeness)
    return {"contigs": contigs, "completeness": actual, "quality_tier": quality_tier}


@dataclass
class ContaminationEvent:
    """Everything a single (acceptor, donor set, type, target) event produced."""

    event_type: str
    donor_intervals: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    acceptor_excised: List[Tuple[int, int]] = field(default_factory=list)
    # spans of the acceptor's OWN copies of every replaced marker: these must be
    # removed in full, otherwise the sample is redundant contamination wearing a
    # "replaced" label. Never truncated by the size-matching step.
    acceptor_mandatory: List[Tuple[int, int]] = field(default_factory=list)
    mandatory_bp: int = 0
    size_match_deficit_bp: int = 0
    donor_bp: int = 0
    excised_bp: int = 0
    n_donor_blocks: int = 0
    n_transferred_genes: int = 0
    n_duplicated_families: int = 0    # markers now present twice (redundant)
    n_replaced_families: int = 0      # markers now present once, wrong organism
    n_novel_genes: int = 0            # donor genes with no acceptor counterpart
    shortfall_bp: int = 0             # target not reachable from eligible territory
    donors_used: List[str] = field(default_factory=list)


def _select_blocks(blocks: List[Tuple[int, int, List[int]]], target_bp: int,
                   rng: np.random.Generator, gene_end: np.ndarray,
                   min_keep: int = 200
                   ) -> Tuple[List[Tuple[int, int]], List[int], int]:
    """Shuffle blocks, take until target_bp, trim the last one to hit it exactly."""
    if target_bp <= 0 or not blocks:
        return [], [], target_bp
    order = rng.permutation(len(blocks))
    chosen: List[Tuple[int, int]] = []
    genes: List[int] = []
    got = 0
    for k in order:
        s, e, gidx = blocks[int(k)]
        need = target_bp - got
        if need <= 0:
            break
        L = e - s
        if L <= need:
            chosen.append((s, e))
            genes.extend(gidx)
            got += L
        else:
            if need >= min_keep:
                chosen.append((s, s + need))
                got += need
                # only genes FULLY inside the trimmed block count as transferred;
                # a half-transferred gene is neither duplicated nor replaced
                cut = s + need
                genes.extend([g for g in gidx if int(gene_end[g]) <= cut])
            break
    return chosen, genes, max(0, target_bp - got)


def make_event(
    acceptor: GenomeAnnotation,
    donors: Sequence[GenomeAnnotation],
    rbh_by_donor: Dict[str, List[Tuple[str, str]]],
    event_type: str,
    target_contaminant_bp: int,
    rng: np.random.Generator,
) -> ContaminationEvent:
    """Build one contamination event of the requested Cornet type.

    ``rbh_by_donor[donor.accession]`` is the list of (acceptor_gene, donor_gene)
    reciprocal-best-hit pairs for that donor.
    """
    if event_type not in CONTAMINATION_TYPES:
        raise ValueError(f"unknown contamination type {event_type!r}")

    ev = ContaminationEvent(event_type=event_type)
    remaining = int(target_contaminant_bp)

    for donor in donors:
        if remaining <= 0:
            break
        pairs = rbh_by_donor.get(donor.accession, [])
        donor_gene_index = {g: i for i, g in enumerate(donor.gene_ids)}
        acceptor_gene_index = {g: i for i, g in enumerate(acceptor.gene_ids)}
        donor_is_common = np.zeros(donor.n_genes, dtype=bool)
        partner_of = {}
        for a_gene, d_gene in pairs:
            di = donor_gene_index.get(d_gene)
            ai = acceptor_gene_index.get(a_gene)
            if di is None or ai is None:
                continue
            donor_is_common[di] = True
            partner_of[di] = ai

        eligible = donor_is_common if event_type in ("redundant", "replaced") \
            else ~donor_is_common
        blocks = build_donor_blocks(donor, eligible)
        chosen, genes, shortfall = _select_blocks(blocks, remaining, rng, donor.gene_end)
        if not chosen:
            continue

        got = total_bp(chosen)
        ev.donor_intervals.setdefault(donor.accession, []).extend(chosen)
        ev.donors_used.append(donor.accession)
        ev.donor_bp += got
        ev.n_donor_blocks += len(chosen)
        ev.n_transferred_genes += len(genes)
        remaining -= got

        if event_type == "redundant":
            ev.n_duplicated_families += len(genes)
        elif event_type == "single":
            ev.n_novel_genes += len(genes)
        else:  # replaced
            partners = sorted({partner_of[g] for g in genes if g in partner_of})
            ev.n_replaced_families += len(partners)
            ev.acceptor_mandatory.extend(
                (int(acceptor.gene_start[g]), int(acceptor.gene_end[g]))
                for g in partners)

    ev.shortfall_bp = max(0, remaining)
    ev.donor_intervals = {k: merge_intervals(v) for k, v in ev.donor_intervals.items()}
    ev.donor_bp = int(sum(total_bp(v) for v in ev.donor_intervals.values()))

    if event_type == "replaced":
        ev.acceptor_mandatory = merge_intervals(ev.acceptor_mandatory)
        ev.mandatory_bp = total_bp(ev.acceptor_mandatory)
        ev.acceptor_excised = _size_match_excision(
            acceptor, ev.acceptor_mandatory, ev.donor_bp, rng)
        ev.excised_bp = total_bp(ev.acceptor_excised)
        ev.size_match_deficit_bp = ev.donor_bp - ev.excised_bp
    return ev


def _size_match_excision(acceptor: GenomeAnnotation,
                         mandatory: List[Tuple[int, int]], need_bp: int,
                         rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Acceptor territory deleted by a REPLACED event, exactly ``need_bp`` long.

    Invariant, and the whole point of the type: **every** acceptor copy of a
    replaced marker is deleted in full.  ``mandatory`` (the union of those gene
    spans) is therefore never truncated.  The remaining bp needed for size
    matching are taken by symmetric growth around each mandatory interval within
    its own contig, and then, if still short, from spare acceptor territory.
    If ``mandatory`` alone already exceeds ``need_bp`` the replacement cannot be
    size-matched without leaving a duplicate behind; we keep the mandatory
    deletion and record the deficit rather than silently corrupting the type.
    """
    mandatory = merge_intervals(mandatory)
    have = total_bp(mandatory)
    if have >= need_bp:
        return mandatory

    out = list(mandatory)
    extra = need_bp - have
    per = int(np.ceil(extra / (2 * max(1, len(mandatory)))))
    ci_of = _contig_index_lookup(acceptor)
    grown = []
    for s, e in mandatory:
        lo, hi = ci_of(s)
        grown.append((max(lo, s - per), min(hi, e + per)))
    out = merge_intervals(out + [g for g in grown if g[1] > g[0]])

    if total_bp(out) < need_bp:
        free = [f for f in complement_intervals(out, acceptor.total_bp) if f[1] > f[0]]
        rng.shuffle(free)
        need = need_bp - total_bp(out)
        for a, b in free:
            if need <= 0:
                break
            take = min(b - a, need)
            out.append((a, a + take))
            need -= take
        out = merge_intervals(out)

    # trim the overshoot, but only from territory that is NOT mandatory
    over = total_bp(out) - need_bp
    if over > 0:
        out = _trim_non_mandatory(out, mandatory, over)
    return merge_intervals(out)


def _contig_index_lookup(acceptor: GenomeAnnotation):
    starts = np.asarray(acceptor.contig_offsets, dtype=np.int64)
    lens = np.asarray(acceptor.contig_lens, dtype=np.int64)

    def f(pos: int) -> Tuple[int, int]:
        i = int(np.searchsorted(starts, pos, side="right") - 1)
        i = max(0, min(i, len(starts) - 1))
        return int(starts[i]), int(starts[i] + lens[i])

    return f


def _trim_non_mandatory(iv: List[Tuple[int, int]], mandatory: List[Tuple[int, int]],
                        over: int) -> List[Tuple[int, int]]:
    """Remove ``over`` bp from ``iv``, never touching ``mandatory`` territory."""
    mand = merge_intervals(mandatory)
    out: List[Tuple[int, int]] = []
    for a, b in merge_intervals(iv):
        pieces = _subtract(a, b, mand)          # non-mandatory parts of this interval
        keep_mand = [(max(a, m0), min(b, m1)) for m0, m1 in mand
                     if m1 > a and m0 < b]
        for p0, p1 in pieces:
            if over <= 0:
                out.append((p0, p1))
                continue
            take = min(p1 - p0, over)
            over -= take
            if p1 - p0 - take > 0:
                out.append((p0 + take, p1))
        out.extend(k for k in keep_mand if k[1] > k[0])
    return merge_intervals(out)


def _subtract(a: int, b: int, blocks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out, pos = [], a
    for m0, m1 in blocks:
        if m1 <= a or m0 >= b:
            continue
        if m0 > pos:
            out.append((pos, min(m0, b)))
        pos = max(pos, min(m1, b))
    if pos < b:
        out.append((pos, b))
    return [(x, y) for x, y in out if y > x]


# ==========================================================================
# 7. Full sample assembly
# ==========================================================================
def generate_typed_sample(
    acceptor_seq: str,
    donor_seqs: Dict[str, str],
    event: ContaminationEvent,
    target_completeness: float,
    quality_tier: str,
    contaminant_tier: str,
    rng: np.random.Generator,
    rng_contaminant: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Assemble one Set-F sample from a prepared contamination event.

    Ground truth uses MAGICC's production convention:
        completeness  = retained dominant bp / dominant FULL reference bp
        contamination = contaminant bp       / dominant FULL reference bp

    ``rng`` drives the DOMINANT assembly and ``rng_contaminant`` the contaminant
    contigs.  Keeping them separate is what makes the Set F factorial paired:
    with the same ``rng`` seed, the REDUNDANT and SINGLE samples of a given
    (reference, distance) cell carry a byte-identical dominant assembly, so the
    only thing that varies between cells is the contamination itself.
    """
    if rng_contaminant is None:
        rng_contaminant = rng
    full_bp = len(acceptor_seq)

    # --- acceptor side -----------------------------------------------------
    if event.acceptor_excised:
        keep = complement_intervals(event.acceptor_excised, full_bp)
        survivor = "".join(extract(acceptor_seq, keep))
    else:
        survivor = acceptor_seq
    survivor_frac = len(survivor) / full_bp if full_bp else 0.0

    effective_target = min(target_completeness, survivor_frac)
    tgt_on_survivor = (min(1.0, effective_target / survivor_frac)
                       if survivor_frac > 0 else 0.0)
    # floor expressed relative to the survivor, so that the FINAL completeness
    # (measured against the full reference) never drops below MIN_COMPLETENESS_PCT
    floor_on_survivor = (min(0.999, (MIN_COMPLETENESS_PCT / 100.0) / survivor_frac)
                         if survivor_frac > 0 else 0.5)

    dom = _fragment_with_floor(survivor, tgt_on_survivor, quality_tier, rng,
                               floor_on_survivor)
    dominant_contigs = dom["contigs"]
    dominant_bp = sum(len(c) for c in dominant_contigs)

    # --- contaminant side --------------------------------------------------
    contaminant_contigs: List[str] = []
    for acc in sorted(event.donor_intervals):
        seq = donor_seqs[acc]
        for block in extract(seq, event.donor_intervals[acc]):
            contaminant_contigs.extend(
                split_into_contigs(block, contaminant_tier, rng_contaminant))
    contaminant_bp = sum(len(c) for c in contaminant_contigs)

    # --- merge -------------------------------------------------------------
    all_contigs = dominant_contigs + contaminant_contigs
    if len(all_contigs) > 1:
        idx = np.arange(len(all_contigs))
        rng_contaminant.shuffle(idx)
        all_contigs = [all_contigs[i] for i in idx]

    return {
        "contigs": all_contigs,
        "dominant_contigs": dominant_contigs,
        "contaminant_contigs": contaminant_contigs,
        "completeness": 100.0 * dominant_bp / full_bp if full_bp else 0.0,
        "contamination": 100.0 * contaminant_bp / full_bp if full_bp else 0.0,
        "dominant_full_length": full_bp,
        "dominant_retained_bp": dominant_bp,
        "contaminant_bp": contaminant_bp,
        "excised_bp": event.excised_bp,
        "survivor_bp": len(survivor),
        "quality_tier": dom["quality_tier"],
        "n_contigs_dominant": len(dominant_contigs),
        "n_contigs_contaminant": len(contaminant_contigs),
    }


def write_fasta(contigs: Sequence[str], path: Path, prefix: str = "contig") -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        for i, c in enumerate(contigs):
            fh.write(f">{prefix}_{i + 1} length={len(c)}\n")
            for j in range(0, len(c), 80):
                fh.write(c[j:j + 80] + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ==========================================================================
# self-test
# ==========================================================================
def _self_test() -> int:
    import pandas as pd

    print("=" * 74)
    print("144_contamination_type_module — self-test")
    print("=" * 74)
    t = pd.read_csv(PROJECT_DIR / "data/splits/test_genomes.tsv", sep="\t")
    t["fasta_path"] = t["fasta_path"].str.replace(
        "/path/to/home/projects/magicc2", str(PROJECT_DIR), regex=False)
    tax = t["gtdb_taxonomy"].str.split(";", expand=True)
    t["sp"] = tax[6].str.replace("^s__", "", regex=True)
    t["gn"] = tax[5].str.replace("^g__", "", regex=True)
    # a species with >=2 test genomes and a congeneric different species
    vc = t["sp"].value_counts()
    sp = vc[vc >= 2].index[0]
    sub = t[t["sp"] == sp]
    a = sub.iloc[0]
    b = sub.iloc[1]
    print(f"acceptor {a.gtdb_accession} ({a.genome_size/1e6:.2f} Mbp)")
    print(f"donor    {b.gtdb_accession} ({b.genome_size/1e6:.2f} Mbp)  same species")

    work = Path(tempfile.mkdtemp(prefix="ws2_selftest_"))
    try:
        ann_a = annotate_genome(a.gtdb_accession, a.fasta_path, work / "A")
        ann_b = annotate_genome(b.gtdb_accession, b.fasta_path, work / "B")
        print(f"  genes: acceptor {ann_a.n_genes}, donor {ann_b.n_genes}")
        pairs = rbh_orthologs(ann_a, ann_b, work / "rbh.tsv", threads=2)
        print(f"  RBH orthologues: {len(pairs)} "
              f"({100*len(pairs)/max(1,ann_b.n_genes):.1f}% of donor genes are 'common')")

        seq_a = read_fasta(ann_a.fasta_path)
        seq_b = read_fasta(ann_b.fasta_path)
        assert len(seq_a) == ann_a.total_bp, "concatenated length mismatch (acceptor)"
        assert len(seq_b) == ann_b.total_bp, "concatenated length mismatch (donor)"

        target = int(0.10 * len(seq_a))
        ok = True
        for et in CONTAMINATION_TYPES:
            rng = np.random.default_rng(1234)
            ev = make_event(ann_a, [ann_b], {ann_b.accession: pairs}, et, target, rng)
            s = generate_typed_sample(seq_a, {ann_b.accession: seq_b}, ev,
                                      target_completeness=1.0,
                                      quality_tier="low", contaminant_tier="low",
                                      rng=np.random.default_rng(99))
            bp_check = sum(len(c) for c in s["contigs"])
            print(f"  [{et:9s}] donor_bp={ev.donor_bp:>8d} shortfall={ev.shortfall_bp:>8d} "
                  f"excised={ev.excised_bp:>8d} | comp={s['completeness']:6.2f}% "
                  f"cont={s['contamination']:6.2f}% contigs={len(s['contigs'])} "
                  f"bp={bp_check}")
            if et == "replaced" and ev.donor_bp > 0 and ev.excised_bp != ev.donor_bp:
                if ev.mandatory_bp <= ev.donor_bp:
                    print("     FAIL: replacement is not size-matched")
                    ok = False
                else:
                    print(f"     note: mandatory deletion {ev.mandatory_bp} exceeds "
                          f"donor {ev.donor_bp}; kept mandatory, deficit recorded")
            if et == "replaced":
                # THE defining invariant: no acceptor copy of a replaced marker
                # may survive the excision.
                kept = complement_intervals(ev.acceptor_excised, len(seq_a))
                surviving = 0
                for m0, m1 in ev.acceptor_mandatory:
                    for k0, k1 in kept:
                        if k0 < m1 and m0 < k1:
                            surviving += 1
                            break
                print(f"     replaced-marker acceptor copies surviving: {surviving}"
                      f" / {len(ev.acceptor_mandatory)}")
                if surviving:
                    print("     FAIL: acceptor copies of replaced markers survive")
                    ok = False
            if et != "replaced" and ev.excised_bp != 0:
                print("     FAIL: non-replaced event excised acceptor sequence")
                ok = False
            if bp_check != s["dominant_retained_bp"] + s["contaminant_bp"]:
                print("     FAIL: contig bp != dominant+contaminant bp")
                ok = False
            if s["contamination"] > s["completeness"] + 1e-9:
                print("     FAIL: out of training domain (contamination% > completeness%)")
                ok = False

        # determinism
        rng = np.random.default_rng(7)
        e1 = make_event(ann_a, [ann_b], {ann_b.accession: pairs}, "replaced", target, rng)
        rng = np.random.default_rng(7)
        e2 = make_event(ann_a, [ann_b], {ann_b.accession: pairs}, "replaced", target, rng)
        if e1.donor_intervals != e2.donor_intervals or e1.acceptor_excised != e2.acceptor_excised:
            print("  FAIL: event generation is not deterministic under a fixed seed")
            ok = False
        else:
            print("  determinism under fixed seed: OK")
        print("SELF-TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(_self_test())
