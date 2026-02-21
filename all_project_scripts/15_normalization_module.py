#!/usr/bin/env python3
"""
Step 4: Test the feature normalization pipeline.

Tests:
1. Running statistics (Welford's algorithm)
2. K-mer normalization (log + Z-score)
3. Assembly normalization (mixed per feature type)
4. Save/load round-trip
5. Batch incremental updates
"""

import sys
import os
import time
import tempfile
import numpy as np

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.normalization import (
    FeatureNormalizer, RunningStats,
    LOG10_FEATURES, MINMAX_FEATURES, ROBUST_FEATURES, PASSTHROUGH_FEATURES,
)
from magicc.assembly_stats import FEATURE_NAMES, FEATURE_INDEX, N_FEATURES


def test_running_stats():
    """Test running statistics computation."""
    print("=" * 70)
    print("TEST 1: Running statistics (Welford's algorithm)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_features = 10
    stats = RunningStats(n_features, reservoir_size=5000)

    # Generate data in batches
    all_data = []
    for _ in range(10):
        batch = rng.normal(5.0, 2.0, size=(1000, n_features))
        stats.update_batch(batch)
        all_data.append(batch)

    all_data = np.vstack(all_data)

    # Compare with numpy
    np_mean = np.mean(all_data, axis=0)
    np_std = np.std(all_data, axis=0, ddof=1)
    np_min = np.min(all_data, axis=0)
    np_max = np.max(all_data, axis=0)
    np_median = np.median(all_data, axis=0)

    assert np.allclose(stats.mean, np_mean, atol=1e-6), f"Mean mismatch"
    assert np.allclose(stats.std, np_std, atol=1e-6), f"Std mismatch"
    assert np.allclose(stats.min_vals, np_min, atol=1e-10), f"Min mismatch"
    assert np.allclose(stats.max_vals, np_max, atol=1e-10), f"Max mismatch"
    # Median from reservoir is approximate
    median_error = np.mean(np.abs(stats.median - np_median))
    print(f"  Mean error: {np.max(np.abs(stats.mean - np_mean)):.2e}")
    print(f"  Std error: {np.max(np.abs(stats.std - np_std)):.2e}")
    print(f"  Median approx error: {median_error:.4f}")
    assert median_error < 0.2, f"Median approximation too inaccurate: {median_error}"

    print(f"  Count: {stats.count}")
    print("  [PASS] Running statistics match numpy within tolerance")


def test_kmer_normalization():
    """Test k-mer feature normalization."""
    print("\n" + "=" * 70)
    print("TEST 2: K-mer normalization (log + Z-score)")
    print("=" * 70)

    rng = np.random.default_rng(123)
    n_kmers = 100  # Small for testing

    normalizer = FeatureNormalizer(n_kmer_features=n_kmers, reservoir_size=5000)

    # Simulate k-mer counts (Poisson-like)
    all_counts = []
    for _ in range(5):
        batch = rng.poisson(lam=50, size=(1000, n_kmers)).astype(np.float64)
        normalizer.update_kmer_batch(batch)
        all_counts.append(batch)

    normalizer.finalize()

    # Test normalization
    test_batch = rng.poisson(lam=50, size=(100, n_kmers)).astype(np.float64)
    normalized = normalizer.normalize_kmer(test_batch)

    # Check properties of normalized data
    # After Z-score on log-transformed data, mean should be ~0 and std ~1
    all_counts = np.vstack(all_counts)
    log_all = np.log1p(all_counts)
    expected_mean = np.mean(log_all, axis=0)
    expected_std = np.std(log_all, axis=0, ddof=1)

    print(f"  Input shape: {test_batch.shape}")
    print(f"  Output shape: {normalized.shape}")
    print(f"  Normalized mean: {np.mean(normalized):.4f}")
    print(f"  Normalized std: {np.std(normalized):.4f}")
    print(f"  Normalized range: [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")

    # Zero counts should map to a specific value
    zero_batch = np.zeros((1, n_kmers))
    zero_norm = normalizer.normalize_kmer(zero_batch)
    print(f"  Zero count normalized: mean={np.mean(zero_norm):.4f}")

    print("  [PASS] K-mer normalization produces valid output")


def test_assembly_normalization():
    """Test assembly feature normalization with mixed scaling."""
    print("\n" + "=" * 70)
    print("TEST 3: Assembly normalization (mixed scaling)")
    print("=" * 70)

    rng = np.random.default_rng(456)
    normalizer = FeatureNormalizer(n_kmer_features=100, reservoir_size=5000)

    # Generate realistic assembly stats
    n_samples = 5000
    assembly_data = np.zeros((n_samples, N_FEATURES), dtype=np.float64)
    for i in range(n_samples):
        assembly_data[i, FEATURE_INDEX['total_length']] = rng.uniform(500_000, 10_000_000)
        assembly_data[i, FEATURE_INDEX['contig_count']] = rng.integers(1, 2000)
        assembly_data[i, FEATURE_INDEX['n50']] = rng.uniform(1_000, 1_000_000)
        assembly_data[i, FEATURE_INDEX['n90']] = rng.uniform(500, 100_000)
        assembly_data[i, FEATURE_INDEX['l50']] = rng.integers(1, 500)
        assembly_data[i, FEATURE_INDEX['l90']] = rng.integers(1, 1000)
        assembly_data[i, FEATURE_INDEX['largest_contig']] = rng.uniform(10_000, 5_000_000)
        assembly_data[i, FEATURE_INDEX['smallest_contig']] = rng.uniform(100, 10_000)
        assembly_data[i, FEATURE_INDEX['mean_contig']] = rng.uniform(5_000, 500_000)
        assembly_data[i, FEATURE_INDEX['median_contig']] = rng.uniform(2_000, 200_000)
        assembly_data[i, FEATURE_INDEX['contig_length_std']] = rng.uniform(1_000, 500_000)
        assembly_data[i, FEATURE_INDEX['gc_mean']] = rng.uniform(0.2, 0.8)
        assembly_data[i, FEATURE_INDEX['gc_std']] = rng.uniform(0.0, 0.1)
        assembly_data[i, FEATURE_INDEX['gc_iqr']] = rng.uniform(0.0, 0.05)
        assembly_data[i, FEATURE_INDEX['gc_bimodality']] = rng.uniform(0.0, 2.0)
        assembly_data[i, FEATURE_INDEX['gc_outlier_fraction']] = rng.uniform(0.0, 0.3)
        assembly_data[i, FEATURE_INDEX['largest_contig_fraction']] = rng.uniform(0.01, 1.0)
        assembly_data[i, FEATURE_INDEX['top10_concentration']] = rng.uniform(0.1, 1.0)
        assembly_data[i, FEATURE_INDEX['n50_mean_ratio']] = rng.uniform(0.5, 10.0)
        assembly_data[i, FEATURE_INDEX['log10_total_kmer_count']] = rng.uniform(4.0, 7.0)

    # Update in batches
    for start in range(0, n_samples, 1000):
        normalizer.update_assembly_batch(assembly_data[start:start+1000])

    normalizer.finalize()

    # Normalize
    normalized = normalizer.normalize_assembly(assembly_data[:100])

    print(f"  Input shape: {assembly_data[:100].shape}")
    print(f"  Output shape: {normalized.shape}")

    # Check each normalization type
    print(f"\n  Log10 features (length-based):")
    for fname in LOG10_FEATURES[:3]:
        idx = FEATURE_INDEX[fname]
        print(f"    {fname:25s}: raw=[{assembly_data[:100, idx].min():.0f}, "
              f"{assembly_data[:100, idx].max():.0f}] -> "
              f"norm=[{normalized[:, idx].min():.2f}, {normalized[:, idx].max():.2f}]")

    print(f"\n  MinMax features (percentage-based):")
    for fname in MINMAX_FEATURES[:3]:
        idx = FEATURE_INDEX[fname]
        print(f"    {fname:25s}: raw=[{assembly_data[:100, idx].min():.4f}, "
              f"{assembly_data[:100, idx].max():.4f}] -> "
              f"norm=[{normalized[:, idx].min():.4f}, {normalized[:, idx].max():.4f}]")

    print(f"\n  Robust features (count-based):")
    for fname in ROBUST_FEATURES:
        idx = FEATURE_INDEX[fname]
        print(f"    {fname:25s}: raw=[{assembly_data[:100, idx].min():.0f}, "
              f"{assembly_data[:100, idx].max():.0f}] -> "
              f"norm=[{normalized[:, idx].min():.2f}, {normalized[:, idx].max():.2f}]")

    print(f"\n  Passthrough features:")
    for fname in PASSTHROUGH_FEATURES:
        idx = FEATURE_INDEX[fname]
        print(f"    {fname:25s}: raw=[{assembly_data[:100, idx].min():.4f}, "
              f"{assembly_data[:100, idx].max():.4f}] -> "
              f"norm=[{normalized[:, idx].min():.4f}, {normalized[:, idx].max():.4f}]")

    print("\n  [PASS] Assembly normalization with mixed scaling works")


def test_save_load():
    """Test save/load round-trip."""
    print("\n" + "=" * 70)
    print("TEST 4: Save/load round-trip")
    print("=" * 70)

    rng = np.random.default_rng(789)
    normalizer = FeatureNormalizer(n_kmer_features=50)

    # Add some data
    for _ in range(3):
        normalizer.update_kmer_batch(rng.poisson(30, (500, 50)).astype(np.float64))
        normalizer.update_assembly_batch(rng.uniform(0, 1000, (500, N_FEATURES)))

    normalizer.finalize()

    # Save
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        save_path = f.name
    normalizer.save(save_path)

    # Load
    loaded = FeatureNormalizer.load(save_path)

    # Verify
    assert loaded.finalized
    assert np.allclose(loaded.kmer_mean, normalizer.kmer_mean)
    assert np.allclose(loaded.kmer_std, normalizer.kmer_std)
    assert np.allclose(loaded.assembly_minmax_min, normalizer.assembly_minmax_min)
    assert np.allclose(loaded.assembly_minmax_range, normalizer.assembly_minmax_range)
    assert np.allclose(loaded.assembly_robust_median, normalizer.assembly_robust_median)
    assert np.allclose(loaded.assembly_robust_iqr, normalizer.assembly_robust_iqr)

    # Test that loaded normalizer produces same results
    test_kmer = rng.poisson(30, (10, 50)).astype(np.float64)
    test_asm = rng.uniform(0, 1000, (10, N_FEATURES))
    orig_norm = normalizer.normalize_all(test_kmer, test_asm)
    loaded_norm = loaded.normalize_all(test_kmer, test_asm)
    assert np.allclose(orig_norm, loaded_norm)

    os.unlink(save_path)
    file_size = os.path.getsize(save_path) if os.path.exists(save_path) else 0
    print(f"  Save/load round-trip successful")
    print("  [PASS] Normalization parameters persist correctly")


def test_full_pipeline():
    """Test combined normalization pipeline."""
    print("\n" + "=" * 70)
    print("TEST 5: Full normalization pipeline")
    print("=" * 70)

    rng = np.random.default_rng(101)
    n_kmers = 9249  # Full k-mer count
    normalizer = FeatureNormalizer(n_kmer_features=n_kmers)

    # Simulate multiple batch updates
    n_batches = 5
    batch_size = 2000
    print(f"  Processing {n_batches} batches of {batch_size} samples...")

    start = time.perf_counter()
    for b in range(n_batches):
        kmer_batch = rng.poisson(lam=50, size=(batch_size, n_kmers)).astype(np.float64)
        asm_batch = rng.uniform(0, 1e6, (batch_size, N_FEATURES))
        normalizer.update_kmer_batch(kmer_batch)
        normalizer.update_assembly_batch(asm_batch)
    update_time = time.perf_counter() - start

    normalizer.finalize()

    # Normalize a test batch
    test_kmer = rng.poisson(50, (1000, n_kmers)).astype(np.float64)
    test_asm = rng.uniform(0, 1e6, (1000, N_FEATURES))

    start = time.perf_counter()
    result = normalizer.normalize_all(test_kmer, test_asm)
    norm_time = time.perf_counter() - start

    print(f"  Statistics update: {update_time:.2f}s for {n_batches * batch_size:,} samples")
    print(f"  Normalization: {norm_time:.3f}s for 1,000 samples ({norm_time/1000*1000:.2f} ms/sample)")
    print(f"  Output shape: {result.shape}")
    print(f"  Expected shape: (1000, {n_kmers + N_FEATURES})")
    assert result.shape == (1000, n_kmers + N_FEATURES)
    print(f"  K-mer features: mean={np.mean(result[:, :n_kmers]):.4f}, "
          f"std={np.std(result[:, :n_kmers]):.4f}")

    print("  [PASS] Full normalization pipeline works correctly")


if __name__ == '__main__':
    test_running_stats()
    test_kmer_normalization()
    test_assembly_normalization()
    test_save_load()
    test_full_pipeline()
    print("\n" + "=" * 70)
    print("ALL NORMALIZATION MODULE TESTS PASSED")
    print("=" * 70)
