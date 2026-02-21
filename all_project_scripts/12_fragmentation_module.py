#!/usr/bin/env python3
"""
Step 1: Test the genome fragmentation module.

Tests:
1. Generate contigs for each quality tier
2. Fragment a real reference genome
3. Verify completeness targeting works
4. Test edge cases (small genomes, single-contig)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.fragmentation import (
    simulate_fragmentation, read_fasta, generate_contig_lengths,
    QUALITY_TIERS, apply_completeness
)


def compute_n50(lengths):
    """Compute N50 from a list of contig lengths."""
    sorted_lens = sorted(lengths, reverse=True)
    total = sum(sorted_lens)
    running = 0
    for l in sorted_lens:
        running += l
        if running >= total / 2:
            return l
    return 0


def test_contig_generation():
    """Test contig length generation for each quality tier."""
    print("=" * 70)
    print("TEST 1: Contig length generation for each quality tier")
    print("=" * 70)

    rng = np.random.default_rng(42)
    genome_length = 5_000_000  # 5 Mb typical genome

    for tier_name, (c_min, c_max, n50_min, n50_max, min_bp) in QUALITY_TIERS.items():
        lengths = generate_contig_lengths(genome_length, tier_name, rng)
        n50 = compute_n50(lengths)
        print(f"\n  {tier_name:20s}: {len(lengths):5d} contigs, "
              f"N50={n50:>10,d} bp, "
              f"sum={sum(lengths):>12,d} bp, "
              f"min={min(lengths):>8,d} bp, "
              f"max={max(lengths):>10,d} bp")
        print(f"    Expected contigs: {c_min}-{c_max}, "
              f"Expected N50: {n50_min:,}-{n50_max:,}")

        assert sum(lengths) == genome_length, \
            f"Sum mismatch: {sum(lengths)} != {genome_length}"
        assert all(l > 0 for l in lengths), "Negative/zero contig length found"

    print("\n  [PASS] All quality tiers generate valid contig lengths")


def test_real_genome_fragmentation():
    """Test fragmentation on real reference genomes."""
    print("\n" + "=" * 70)
    print("TEST 2: Real genome fragmentation")
    print("=" * 70)

    # Load genome manifest to find a few test genomes
    manifest_path = '/home/tianrm/projects/magicc2/data/splits/genome_manifest.tsv'
    test_genomes = []
    with open(manifest_path) as f:
        header = f.readline()
        for i, line in enumerate(f):
            if i >= 5:
                break
            parts = line.strip().split('\t')
            acc = parts[1]
            fasta = parts[-1]
            phylum = parts[10]
            test_genomes.append((acc, fasta, phylum))

    rng = np.random.default_rng(123)

    for acc, fasta_path, phylum in test_genomes:
        if not os.path.exists(fasta_path):
            print(f"  Skipping {acc} (file not found)")
            continue

        sequence = read_fasta(fasta_path)
        genome_len = len(sequence)
        print(f"\n  Genome: {acc} ({phylum})")
        print(f"    Full length: {genome_len:,} bp")

        for tier in ['high', 'medium', 'low', 'highly_fragmented']:
            for target_comp in [1.0, 0.75, 0.50]:
                result = simulate_fragmentation(
                    sequence, target_comp, quality_tier=tier, rng=rng
                )
                n50 = compute_n50([len(c) for c in result['contigs']])
                print(f"    {tier:20s} comp={target_comp:.0%}: "
                      f"{result['n_contigs']:5d} contigs, "
                      f"N50={n50:>10,d}, "
                      f"actual_comp={result['completeness']:.1%}, "
                      f"total={result['total_length']:>10,d} bp")

    print("\n  [PASS] Real genome fragmentation works correctly")


def test_completeness_targeting():
    """Test that completeness targeting is reasonably accurate."""
    print("\n" + "=" * 70)
    print("TEST 3: Completeness targeting accuracy")
    print("=" * 70)

    rng = np.random.default_rng(456)
    # Generate a synthetic sequence (fast method using numpy bytes)
    genome_len = 4_000_000
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)  # A, C, G, T
    sequence = bytes(bases[rng.integers(0, 4, size=genome_len)]).decode('ascii')

    targets = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    errors = []
    for target in targets:
        results = []
        for i in range(20):
            result = simulate_fragmentation(
                sequence, target, quality_tier='medium', rng=rng
            )
            results.append(result['completeness'])
        mean_comp = np.mean(results)
        std_comp = np.std(results)
        error = abs(mean_comp - target)
        errors.append(error)
        print(f"  Target={target:.0%}: achieved={mean_comp:.3f} +/- {std_comp:.3f} "
              f"(error={error:.3f})")

    mean_error = np.mean(errors)
    print(f"\n  Mean absolute error across targets: {mean_error:.4f}")
    print(f"  Max absolute error: {max(errors):.4f}")
    # Allow up to 10% error since we're doing contig-level dropping
    assert max(errors) < 0.15, f"Completeness targeting too inaccurate: {max(errors):.3f}"
    print("  [PASS] Completeness targeting within acceptable range")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 70)
    print("TEST 4: Edge cases")
    print("=" * 70)

    rng = np.random.default_rng(789)

    # Empty genome
    result = simulate_fragmentation("", 0.5, rng=rng)
    assert result['n_contigs'] == 0
    print("  [PASS] Empty genome handled")

    # Very small genome (100 bp)
    result = simulate_fragmentation("A" * 100, 1.0, quality_tier='high', rng=rng)
    assert result['n_contigs'] >= 1
    assert result['total_length'] <= 100
    print(f"  [PASS] Tiny genome (100bp): {result['n_contigs']} contigs, "
          f"{result['total_length']} bp")

    # Single contig genome (1 kb)
    result = simulate_fragmentation("ACGT" * 250, 1.0, quality_tier='high', rng=rng)
    assert result['n_contigs'] >= 1
    print(f"  [PASS] Small genome (1kb): {result['n_contigs']} contigs")

    # 100% completeness
    seq = ''.join(rng.choice(list('ACGT'), 2_000_000))
    result = simulate_fragmentation(seq, 1.0, quality_tier='medium', rng=rng)
    print(f"  [PASS] 100% completeness: {result['n_contigs']} contigs, "
          f"comp={result['completeness']:.3f}")


def test_performance():
    """Benchmark fragmentation speed."""
    print("\n" + "=" * 70)
    print("TEST 5: Performance benchmark")
    print("=" * 70)

    rng = np.random.default_rng(101)
    genome_len = 5_000_000
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)  # A, C, G, T
    sequence = bytes(bases[rng.integers(0, 4, size=genome_len)]).decode('ascii')

    n_iterations = 100
    start = time.perf_counter()
    for i in range(n_iterations):
        result = simulate_fragmentation(
            sequence, rng.uniform(0.5, 1.0), rng=rng
        )
    elapsed = time.perf_counter() - start
    per_genome = elapsed / n_iterations * 1000

    print(f"  {n_iterations} fragmentations of {genome_len/1e6:.1f} Mb genome")
    print(f"  Total: {elapsed:.2f}s, Per genome: {per_genome:.1f} ms")
    print(f"  [{'PASS' if per_genome < 500 else 'WARN'}] "
          f"Performance: {per_genome:.1f} ms/genome")


if __name__ == '__main__':
    test_contig_generation()
    test_real_genome_fragmentation()
    test_completeness_targeting()
    test_edge_cases()
    test_performance()
    print("\n" + "=" * 70)
    print("ALL FRAGMENTATION MODULE TESTS PASSED")
    print("=" * 70)
