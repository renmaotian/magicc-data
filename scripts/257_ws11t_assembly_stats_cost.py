#!/usr/bin/env python3
"""
WS11.T (T3 support) — direct micro-benchmark of the ONE line that differs
between the V3 and V5 feature-extraction workers.

    V3 worker:  assembly_feats = compute_assembly_stats(contigs, log10, kmer_counts)
    V5 worker:  assembly_feats = compute_assembly_stats(log10, kmer_counts)

`magicc/kmer_counter.py` and `magicc/fragmentation.py` are BYTE-IDENTICAL
between git 471eb28 (V3) and HEAD (V5), so FASTA reading and k-mer counting are
literally the same code in both arms; the assembly-statistics call is the only
per-genome difference.  This script times each component separately on the same
genomes, single-threaded, so the end-to-end intervention arm can be checked
against a bottom-up cost model.

Writes results/revision/speed_v3/assembly_stats_microbench.tsv
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path("/path/to/magicc")
OLD = PROJECT / "results" / "revision" / "speed"
NEW = PROJECT / "results" / "revision" / "speed_v3"
V3TREE = NEW / "v3_code" / "magicc"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    sys.path.insert(0, str(PROJECT))
    from magicc.cli import read_fasta_contigs           # noqa: E402
    from magicc.kmer_counter import KmerCounter         # noqa: E402

    as_v5 = load_module("as_v5", PROJECT / "magicc" / "assembly_stats.py")
    as_v3 = load_module("as_v3", V3TREE / "assembly_stats.py")
    assert as_v3.N_FEATURES == 26 and as_v5.N_FEATURES == 7

    genomes = [l.strip() for l in
               (OLD / "inputs" / "set_E_100.txt").read_text().splitlines() if l.strip()]
    kc = KmerCounter(str(PROJECT / "magicc" / "data" / "selected_kmers.txt"))

    # JIT + page-cache warm-up on the first genome, not timed
    c0 = read_fasta_contigs(genomes[0])
    k0 = kc.count_contigs(c0)
    as_v3.compute_assembly_stats(c0, kc.total_kmer_count(k0), k0)
    as_v5.compute_assembly_stats(kc.total_kmer_count(k0), k0)

    t_read = t_kmer = t_v3 = t_v5 = t_gc = 0.0
    total_bp = 0
    n = 0
    for g in genomes:
        t0 = time.perf_counter(); contigs = read_fasta_contigs(g); t1 = time.perf_counter()
        counts = kc.count_contigs(contigs);                        t2 = time.perf_counter()
        log10 = kc.total_kmer_count(counts)
        as_v3.compute_assembly_stats(contigs, log10, counts);      t3 = time.perf_counter()
        as_v5.compute_assembly_stats(log10, counts);               t4 = time.perf_counter()
        # the encode + Numba GC scan alone, i.e. the dominant term inside V3
        t5 = time.perf_counter()
        for contig in contigs:
            if contig:
                as_v3._compute_gc_from_bytes(
                    np.frombuffer(contig.encode("ascii"), dtype=np.uint8))
        t6 = time.perf_counter()

        t_read += t1 - t0
        t_kmer += t2 - t1
        t_v3 += t3 - t2
        t_v5 += t4 - t3
        t_gc += t6 - t5
        total_bp += sum(len(c) for c in contigs)
        n += 1

    manifest = json.loads((OLD / "inputs" / "input_manifest.json").read_text())
    bp_full = manifest["set_E_full"]["total_sequence_bp"]
    scale = bp_full / total_bp

    rows = [
        ("read_fasta_contigs (identical code in V3 and V5)", t_read),
        ("k-mer counting (identical code in V3 and V5)", t_kmer),
        ("compute_assembly_stats V3 (26 features, needs contigs)", t_v3),
        ("compute_assembly_stats V5 (7 features, kmer_counts only)", t_v5),
        ("  of which: per-contig .encode('ascii') + Numba GC scan", t_gc),
        ("V3 MINUS V5 (the one-line difference)", t_v3 - t_v5),
    ]
    out = ["component\ttotal_s_over_%d_genomes\tms_per_genome\tus_per_Mbp\t"
           "projected_s_over_set_E_full_%dbp" % (n, bp_full)]
    for label, tt in rows:
        out.append(f"{label}\t{tt:.4f}\t{tt/n*1000:.3f}\t"
                   f"{tt/(total_bp/1e6)*1e6:.1f}\t{tt*scale:.2f}")
    out.append(f"# genomes={n}  measured_bp={total_bp}  set_E_full_bp={bp_full}  "
               f"scale_factor={scale:.4f}  single-threaded, warm page cache")
    (NEW / "assembly_stats_microbench.tsv").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
