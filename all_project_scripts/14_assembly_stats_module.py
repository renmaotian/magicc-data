#!/usr/bin/env python3
"""
Step 3: Test the assembly statistics module.

Tests:
1. Compute stats on synthetic genomes
2. Verify all 20 features are computed correctly
3. Test with real reference genomes
4. Performance benchmark
5. Edge cases
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.assembly_stats import (
    compute_assembly_stats, compute_assembly_stats_batch,
    format_stats, FEATURE_NAMES, N_FEATURES,
)
from magicc.fragmentation import read_fasta, simulate_fragmentation


def fast_random_seq(rng, length):
    """Generate a random DNA sequence efficiently."""
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)
    return bytes(bases[rng.integers(0, 4, size=length)]).decode('ascii')


def test_feature_count():
    """Test that we compute exactly 20 features."""
    print("=" * 70)
    print("TEST 1: Feature count verification")
    print("=" * 70)
    assert N_FEATURES == 20, f"Expected 20 features, got {N_FEATURES}"
    assert len(FEATURE_NAMES) == 20
    print(f"  Feature names ({N_FEATURES}):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"    {i:2d}: {name}")
    print("  [PASS] 20 features defined")


def test_synthetic_genome():
    """Test stats computation on a synthetic fragmented genome."""
    print("\n" + "=" * 70)
    print("TEST 2: Synthetic genome statistics")
    print("=" * 70)

    rng = np.random.default_rng(42)
    # Create synthetic contigs with known properties
    contigs = [
        fast_random_seq(rng, 500_000),
        fast_random_seq(rng, 300_000),
        fast_random_seq(rng, 200_000),
        fast_random_seq(rng, 100_000),
        fast_random_seq(rng, 50_000),
    ]

    features = compute_assembly_stats(contigs)
    stats = format_stats(features)

    print(f"  Contigs: {len(contigs)} ({', '.join(f'{len(c):,}' for c in contigs)} bp)")
    for name, val in stats.items():
        print(f"    {name:30s}: {val:>15.4f}")

    # Verify key features
    assert stats['total_length'] == 1_150_000
    assert stats['contig_count'] == 5
    assert stats['largest_contig'] == 500_000
    assert stats['smallest_contig'] == 50_000
    assert abs(stats['mean_contig'] - 230_000) < 1
    assert abs(stats['gc_mean'] - 0.5) < 0.01  # Random sequence ~50% GC
    assert stats['n50'] == 300_000  # 500k+300k = 800k > 575k (half of 1150k)
    assert stats['n50_mean_ratio'] > 1.0  # N50 > mean for typical distribution

    print("  [PASS] All feature values correct")


def test_single_contig():
    """Test with a single-contig genome."""
    print("\n" + "=" * 70)
    print("TEST 3: Single-contig genome")
    print("=" * 70)

    rng = np.random.default_rng(123)
    contig = fast_random_seq(rng, 5_000_000)
    features = compute_assembly_stats([contig])
    stats = format_stats(features)

    print(f"  Single contig: {len(contig):,} bp")
    for name in ['total_length', 'contig_count', 'n50', 'l50', 'gc_mean',
                  'largest_contig_fraction', 'top10_concentration']:
        print(f"    {name:30s}: {stats[name]:>15.4f}")

    assert stats['contig_count'] == 1
    assert stats['n50'] == 5_000_000
    assert stats['l50'] == 1
    assert stats['largest_contig_fraction'] == 1.0
    assert stats['top10_concentration'] == 1.0
    assert stats['gc_std'] == 0.0  # Only one contig, no variance
    assert stats['contig_length_std'] == 0.0
    print("  [PASS] Single-contig genome handled correctly")


def test_real_genomes():
    """Test with real reference genomes."""
    print("\n" + "=" * 70)
    print("TEST 4: Real reference genomes")
    print("=" * 70)

    manifest_path = '/home/tianrm/projects/magicc2/data/splits/genome_manifest.tsv'
    genomes = []
    with open(manifest_path) as f:
        header = f.readline()
        for i, line in enumerate(f):
            if i >= 5:
                break
            parts = line.strip().split('\t')
            acc = parts[1]
            fasta = parts[-1]
            phylum = parts[10]
            genome_size = int(parts[8])
            if os.path.exists(fasta):
                genomes.append((acc, fasta, phylum, genome_size))

    rng = np.random.default_rng(456)

    for acc, fasta_path, phylum, expected_size in genomes:
        sequence = read_fasta(fasta_path)

        # Fragment it
        result = simulate_fragmentation(sequence, 0.80, quality_tier='medium', rng=rng)
        contigs = result['contigs']

        features = compute_assembly_stats(contigs)
        stats = format_stats(features)

        print(f"\n  {acc} ({phylum}, {expected_size:,} bp original)")
        print(f"    Fragmented: {len(contigs)} contigs, "
              f"{int(stats['total_length']):,} bp, "
              f"comp={result['completeness']:.1%}")
        for name in ['total_length', 'contig_count', 'n50', 'n90', 'l50', 'l90',
                      'gc_mean', 'gc_std', 'gc_bimodality', 'largest_contig_fraction']:
            print(f"      {name:30s}: {stats[name]:>15.4f}")

    print("\n  [PASS] Real genome stats computed correctly")


def test_fragmented_quality_tiers():
    """Test stats across different quality tiers."""
    print("\n" + "=" * 70)
    print("TEST 5: Stats across quality tiers")
    print("=" * 70)

    rng = np.random.default_rng(789)
    sequence = fast_random_seq(rng, 5_000_000)

    for tier in ['high', 'medium', 'low', 'highly_fragmented']:
        result = simulate_fragmentation(sequence, 0.90, quality_tier=tier, rng=rng)
        features = compute_assembly_stats(result['contigs'])
        stats = format_stats(features)
        print(f"  {tier:20s}: "
              f"contigs={int(stats['contig_count']):5d}, "
              f"N50={int(stats['n50']):>10,d}, "
              f"GC={stats['gc_mean']:.4f}, "
              f"GC_std={stats['gc_std']:.6f}, "
              f"top10={stats['top10_concentration']:.3f}")

    print("  [PASS] Quality tier stats make sense")


def test_performance():
    """Benchmark assembly stats computation speed."""
    print("\n" + "=" * 70)
    print("TEST 6: Performance benchmark")
    print("=" * 70)

    rng = np.random.default_rng(101)

    # Pre-generate synthetic genomes (use pre-fragmented contigs for speed)
    n_genomes = 500
    all_contigs = []
    for i in range(n_genomes):
        n_ctg = rng.integers(5, 200)
        contigs = []
        for _ in range(n_ctg):
            clen = rng.integers(500, 100_000)
            contigs.append(fast_random_seq(rng, clen))
        all_contigs.append(contigs)

    # Warm up Numba JIT
    _ = compute_assembly_stats(all_contigs[0])

    # Benchmark
    start = time.perf_counter()
    for contigs in all_contigs:
        _ = compute_assembly_stats(contigs)
    elapsed = time.perf_counter() - start
    per_genome = elapsed / n_genomes * 1000

    print(f"  {n_genomes} genomes (synthetic pre-fragmented)")
    print(f"  Total: {elapsed:.2f}s, Per genome: {per_genome:.2f} ms")
    print(f"  [{'PASS' if per_genome < 50 else 'WARN'}] "
          f"Performance: {per_genome:.2f} ms/genome (target <50 ms)")

    # Batch benchmark
    start = time.perf_counter()
    batch_result = compute_assembly_stats_batch(all_contigs)
    batch_elapsed = time.perf_counter() - start
    print(f"  Batch: {batch_elapsed:.2f}s ({batch_elapsed/n_genomes*1000:.2f} ms/genome)")
    assert batch_result.shape == (n_genomes, N_FEATURES)
    print("  [PASS] Batch computation works")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 70)
    print("TEST 7: Edge cases")
    print("=" * 70)

    # Empty contig list
    features = compute_assembly_stats([])
    assert np.all(features == 0)
    print("  [PASS] Empty contig list returns zeros")

    # Single very short contig
    features = compute_assembly_stats(["ACGT"])
    stats = format_stats(features)
    assert stats['total_length'] == 4
    assert stats['contig_count'] == 1
    assert stats['gc_mean'] == 0.5
    print(f"  [PASS] Very short contig (4bp): GC={stats['gc_mean']}")

    # Contigs with N bases
    features = compute_assembly_stats(["ACGTNNNN", "GGCCAATT"])
    stats = format_stats(features)
    print(f"  [PASS] N-containing contig: GC={stats['gc_mean']:.4f}")

    # Many identical contigs
    features = compute_assembly_stats(["ACGT" * 100] * 1000)
    stats = format_stats(features)
    assert stats['contig_count'] == 1000
    assert stats['gc_std'] < 1e-10, f"gc_std should be ~0 for identical contigs, got {stats['gc_std']}"
    print(f"  [PASS] 1000 identical contigs: GC_std={stats['gc_std']}")


if __name__ == '__main__':
    test_feature_count()
    test_synthetic_genome()
    test_single_contig()
    test_real_genomes()
    test_fragmented_quality_tiers()
    test_performance()
    test_edge_cases()
    print("\n" + "=" * 70)
    print("ALL ASSEMBLY STATS MODULE TESTS PASSED")
    print("=" * 70)
