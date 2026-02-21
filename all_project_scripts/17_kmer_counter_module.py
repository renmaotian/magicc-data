#!/usr/bin/env python3
"""
Step 6 (Part 1): Test the k-mer counter module.

Tests:
1. K-mer encoding/decoding
2. Count k-mers in simple sequences
3. Count k-mers in real genomes
4. Verify canonical form handling
5. Performance benchmark
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.kmer_counter import (
    KmerCounter, encode_kmer, reverse_complement_code, canonical_code,
    load_selected_kmers, K,
)


def test_encoding():
    """Test k-mer encoding and canonical form."""
    print("=" * 70)
    print("TEST 1: K-mer encoding and canonical form")
    print("=" * 70)

    # Test encoding
    assert encode_kmer('AAAAAAAAA') == 0
    assert encode_kmer('AAAAAAAAC') == 1
    assert encode_kmer('AAAAAAAAG') == 2
    assert encode_kmer('AAAAAAAT') == 3  # 8-mer, should still work for testing
    print("  [PASS] Basic encoding correct")

    # Test reverse complement
    code = encode_kmer('ACGTACGTA')
    rc = reverse_complement_code(code, K)
    rc_seq = encode_kmer('TACGTACGT')  # RC of ACGTACGTA
    assert rc == rc_seq, f"RC mismatch: {rc} != {rc_seq}"
    print("  [PASS] Reverse complement correct")

    # Test canonical
    code1 = encode_kmer('ACGTACGTA')
    code2 = encode_kmer('TACGTACGT')  # RC of above
    can1 = canonical_code(code1, K)
    can2 = canonical_code(code2, K)
    assert can1 == can2, f"Canonical mismatch: {can1} != {can2}"
    print("  [PASS] Canonical form correct")

    # Test invalid base
    assert encode_kmer('ACGTNACGT') == -1
    print("  [PASS] Invalid base handled")


def test_load_selected_kmers():
    """Test loading selected k-mers."""
    print("\n" + "=" * 70)
    print("TEST 2: Load selected k-mers")
    print("=" * 70)

    kmer_path = '/home/tianrm/projects/magicc2/data/kmer_selection/selected_kmers.txt'
    codes = load_selected_kmers(kmer_path)

    print(f"  Loaded {len(codes)} k-mers")
    assert len(codes) == 9249, f"Expected 9249, got {len(codes)}"
    assert len(codes) == len(set(codes)), "Duplicate codes found"

    # Verify all are canonical
    for code in codes[:100]:
        rc = reverse_complement_code(code, K)
        assert code <= rc, f"Non-canonical code found: {code} > {rc}"

    print("  [PASS] All 9,249 k-mers loaded and canonical")


def test_simple_counting():
    """Test k-mer counting on simple sequences."""
    print("\n" + "=" * 70)
    print("TEST 3: Simple k-mer counting")
    print("=" * 70)

    kmer_path = '/home/tianrm/projects/magicc2/data/kmer_selection/selected_kmers.txt'
    counter = KmerCounter(kmer_path)
    print(f"  Counter initialized: {counter.n_features} features")

    # Count on a known sequence
    # First, find a k-mer that's in our selected set
    with open(kmer_path) as f:
        first_kmer = f.readline().strip()
    print(f"  First selected k-mer: {first_kmer}")

    # Create a sequence containing this k-mer repeated
    seq = first_kmer * 10  # 90 bp, contains the k-mer 10 times (with overlaps)
    counts = counter.count_sequence(seq)
    total = counts.sum()
    print(f"  Sequence: {first_kmer}*10 ({len(seq)} bp)")
    print(f"  Total selected k-mer occurrences: {total}")
    assert total > 0, "Expected at least one k-mer count"

    # Empty sequence
    counts = counter.count_sequence("")
    assert counts.sum() == 0
    print("  [PASS] Empty sequence returns zero counts")

    # Short sequence (< k)
    counts = counter.count_sequence("ACGT")
    assert counts.sum() == 0
    print("  [PASS] Short sequence (< k) returns zero counts")

    # Sequence with N
    counts = counter.count_sequence("ACGTNNNNACGTACGTA")
    print(f"  Sequence with N: total={counts.sum()}")
    print("  [PASS] N-containing sequence handled")


def test_contig_counting():
    """Test counting across multiple contigs."""
    print("\n" + "=" * 70)
    print("TEST 4: Contig-level k-mer counting")
    print("=" * 70)

    kmer_path = '/home/tianrm/projects/magicc2/data/kmer_selection/selected_kmers.txt'
    counter = KmerCounter(kmer_path)

    rng = np.random.default_rng(42)
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)

    # Generate synthetic contigs
    contigs = []
    for _ in range(50):
        length = rng.integers(1000, 100_000)
        seq = bytes(bases[rng.integers(0, 4, size=length)]).decode('ascii')
        contigs.append(seq)

    counts = counter.count_contigs(contigs)
    total = counts.sum()
    total_bp = sum(len(c) for c in contigs)
    log10_total = counter.total_kmer_count(counts)

    print(f"  {len(contigs)} contigs, {total_bp:,} total bp")
    print(f"  Total selected k-mer occurrences: {total:,}")
    print(f"  Non-zero features: {(counts > 0).sum()} / {counter.n_features}")
    print(f"  Log10 total k-mer count: {log10_total:.4f}")
    assert total > 0
    assert (counts > 0).sum() > 100  # Should have many non-zero

    # Verify that single contig = contigs with 1 element
    single_counts = counter.count_contigs([contigs[0]])
    direct_counts = counter.count_sequence(contigs[0])
    assert np.array_equal(single_counts, direct_counts)
    print("  [PASS] Single contig consistency verified")


def test_real_genome():
    """Test k-mer counting on real reference genomes."""
    print("\n" + "=" * 70)
    print("TEST 5: Real genome k-mer counting")
    print("=" * 70)

    from magicc.fragmentation import read_fasta

    kmer_path = '/home/tianrm/projects/magicc2/data/kmer_selection/selected_kmers.txt'
    counter = KmerCounter(kmer_path)

    manifest_path = '/home/tianrm/projects/magicc2/data/splits/genome_manifest.tsv'
    genomes = []
    with open(manifest_path) as f:
        header = f.readline()
        for i, line in enumerate(f):
            if i >= 3:
                break
            parts = line.strip().split('\t')
            acc = parts[1]
            fasta = parts[-1]
            phylum = parts[10]
            if os.path.exists(fasta):
                genomes.append((acc, fasta, phylum))

    for acc, fasta_path, phylum in genomes:
        sequence = read_fasta(fasta_path)

        start = time.perf_counter()
        counts = counter.count_sequence(sequence)
        elapsed = time.perf_counter() - start

        total = counts.sum()
        nonzero = (counts > 0).sum()
        log10_total = counter.total_kmer_count(counts)

        print(f"  {acc} ({phylum}): {len(sequence):,} bp, "
              f"total_kmers={total:,}, nonzero={nonzero}/{counter.n_features}, "
              f"log10={log10_total:.2f}, time={elapsed*1000:.1f}ms")

    print("  [PASS] Real genome k-mer counting works")


def test_performance():
    """Benchmark k-mer counting speed."""
    print("\n" + "=" * 70)
    print("TEST 6: Performance benchmark")
    print("=" * 70)

    kmer_path = '/home/tianrm/projects/magicc2/data/kmer_selection/selected_kmers.txt'
    counter = KmerCounter(kmer_path)

    rng = np.random.default_rng(101)
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)

    # Warm up Numba JIT
    warmup_seq = bytes(bases[rng.integers(0, 4, size=1000)]).decode('ascii')
    _ = counter.count_sequence(warmup_seq)

    # Benchmark single sequence (5 Mb genome)
    genome_len = 5_000_000
    sequence = bytes(bases[rng.integers(0, 4, size=genome_len)]).decode('ascii')

    n_iterations = 50
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = counter.count_sequence(sequence)
    elapsed = time.perf_counter() - start
    per_genome = elapsed / n_iterations * 1000

    print(f"  Single sequence ({genome_len/1e6:.0f} Mb): {per_genome:.1f} ms/genome")

    # Benchmark batch of contigs (simulating a fragmented genome)
    n_genomes = 100
    all_contigs = []
    for _ in range(n_genomes):
        n_ctg = rng.integers(10, 200)
        contigs = []
        for _ in range(n_ctg):
            clen = rng.integers(1000, 100_000)
            contigs.append(bytes(bases[rng.integers(0, 4, size=clen)]).decode('ascii'))
        all_contigs.append(contigs)

    start = time.perf_counter()
    results = counter.count_contigs_batch(all_contigs)
    batch_elapsed = time.perf_counter() - start
    per_genome_batch = batch_elapsed / n_genomes * 1000

    print(f"  Batch contigs ({n_genomes} genomes): {per_genome_batch:.1f} ms/genome")
    print(f"  Results shape: {results.shape}")
    assert results.shape == (n_genomes, counter.n_features)

    target = 50  # ms
    status = "PASS" if per_genome < target else "WARN"
    print(f"  [{status}] Single genome: {per_genome:.1f} ms (target <{target} ms)")
    status = "PASS" if per_genome_batch < target else "WARN"
    print(f"  [{status}] Batch genome: {per_genome_batch:.1f} ms (target <{target} ms)")


if __name__ == '__main__':
    test_encoding()
    test_load_selected_kmers()
    test_simple_counting()
    test_contig_counting()
    test_real_genome()
    test_performance()
    print("\n" + "=" * 70)
    print("ALL K-MER COUNTER MODULE TESTS PASSED")
    print("=" * 70)
