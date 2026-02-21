#!/usr/bin/env python3
"""
Step 5: Test HDF5 feature storage.

Tests:
1. Initialize storage with pre-allocated arrays
2. Write batch data
3. Read batch data and verify
4. Performance benchmarking (read/write speed)
5. File size estimation
"""

import sys
import os
import time
import tempfile
import numpy as np

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.storage import FeatureStore, METADATA_DTYPE, DEFAULT_N_KMER, DEFAULT_N_ASSEMBLY


def test_initialize():
    """Test storage initialization."""
    print("=" * 70)
    print("TEST 1: Storage initialization")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        tmp_path = f.name

    try:
        with FeatureStore(tmp_path, mode='w') as store:
            store.initialize(
                splits={'train': 10_000, 'val': 1_000, 'test': 1_000},
                chunk_size=1_000,
            )

            info = store.get_all_info()
            for split, details in info.items():
                print(f"    {split}: {details}")

            assert info['train']['n_total'] == 10_000
            assert info['val']['n_total'] == 1_000
            assert info['train']['kmer_shape'] == (10_000, DEFAULT_N_KMER)

        file_size = os.path.getsize(tmp_path)
        print(f"\n  Empty file size: {file_size / 1e6:.1f} MB")
        print("  [PASS] Storage initialized correctly")
    finally:
        os.unlink(tmp_path)


def test_write_read():
    """Test write and read operations."""
    print("\n" + "=" * 70)
    print("TEST 2: Write/read batch data")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        tmp_path = f.name

    rng = np.random.default_rng(42)

    try:
        # Create and write
        with FeatureStore(tmp_path, mode='w') as store:
            store.initialize(
                splits={'train': 5_000, 'val': 500},
                chunk_size=1_000,
            )

            # Write 3 batches to train
            for batch_idx in range(3):
                batch_size = 1_000
                kmer = rng.random((batch_size, DEFAULT_N_KMER)).astype(np.float32)
                asm = rng.random((batch_size, DEFAULT_N_ASSEMBLY)).astype(np.float32)
                labels = rng.random((batch_size, 2)).astype(np.float32) * 100
                metadata = np.zeros(batch_size, dtype=METADATA_DTYPE)
                metadata['completeness'] = labels[:, 0]
                metadata['contamination'] = labels[:, 1]
                metadata['batch_id'] = batch_idx

                n_written = store.write_batch('train', kmer, asm, labels, metadata)
                print(f"    Batch {batch_idx}: wrote {n_written} samples")

            info = store.get_split_info('train')
            print(f"    Train: {info['n_written']}/{info['n_total']} written")
            assert info['n_written'] == 3_000

        # Read back
        with FeatureStore(tmp_path, mode='r') as store:
            kmer_r, asm_r, labels_r, meta_r = store.read_batch('train', 0, 100)
            print(f"\n    Read back: kmer={kmer_r.shape}, asm={asm_r.shape}, "
                  f"labels={labels_r.shape}")
            assert kmer_r.shape == (100, DEFAULT_N_KMER)
            assert asm_r.shape == (100, DEFAULT_N_ASSEMBLY)
            assert labels_r.shape == (100, 2)

        file_size = os.path.getsize(tmp_path)
        print(f"    File size after 3 batches: {file_size / 1e6:.1f} MB")
        print("  [PASS] Write/read works correctly")
    finally:
        os.unlink(tmp_path)


def test_performance():
    """Benchmark read/write performance."""
    print("\n" + "=" * 70)
    print("TEST 3: Performance benchmarking")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        tmp_path = f.name

    rng = np.random.default_rng(123)
    n_samples = 50_000
    batch_size = 10_000

    try:
        # Write benchmark
        with FeatureStore(tmp_path, mode='w') as store:
            store.initialize(
                splits={'train': n_samples},
                chunk_size=batch_size,
            )

            write_start = time.perf_counter()
            for batch_idx in range(n_samples // batch_size):
                kmer = rng.random((batch_size, DEFAULT_N_KMER)).astype(np.float32)
                asm = rng.random((batch_size, DEFAULT_N_ASSEMBLY)).astype(np.float32)
                labels = rng.random((batch_size, 2)).astype(np.float32) * 100
                metadata = np.zeros(batch_size, dtype=METADATA_DTYPE)
                metadata['completeness'] = labels[:, 0]
                metadata['contamination'] = labels[:, 1]
                metadata['batch_id'] = batch_idx

                store.write_batch('train', kmer, asm, labels, metadata)

            write_elapsed = time.perf_counter() - write_start
            print(f"  Write: {n_samples:,} samples in {write_elapsed:.2f}s "
                  f"({n_samples/write_elapsed:.0f} samples/sec)")

        file_size = os.path.getsize(tmp_path)
        print(f"  File size: {file_size / 1e6:.1f} MB "
              f"({file_size / n_samples:.0f} bytes/sample)")

        # Read benchmark
        with FeatureStore(tmp_path, mode='r') as store:
            read_start = time.perf_counter()
            for batch_idx in range(n_samples // batch_size):
                start = batch_idx * batch_size
                end = start + batch_size
                kmer_r, asm_r, labels_r, meta_r = store.read_batch('train', start, end)
            read_elapsed = time.perf_counter() - read_start
            print(f"  Read:  {n_samples:,} samples in {read_elapsed:.2f}s "
                  f"({n_samples/read_elapsed:.0f} samples/sec)")

        # Estimate full dataset size
        bytes_per_sample = file_size / n_samples
        full_size_gb = bytes_per_sample * 1_000_000 / 1e9
        print(f"\n  Estimated full dataset (1M samples): {full_size_gb:.1f} GB")
        print("  [PASS] Performance benchmarking complete")
    finally:
        os.unlink(tmp_path)


def test_real_storage_init():
    """Initialize the actual storage file for the project."""
    print("\n" + "=" * 70)
    print("TEST 4: Initialize actual project storage")
    print("=" * 70)

    storage_dir = '/home/tianrm/projects/magicc2/data/features'
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(storage_dir, 'magicc_features.h5')

    if os.path.exists(storage_path):
        print(f"  Storage already exists: {storage_path}")
        print(f"  Size: {os.path.getsize(storage_path) / 1e6:.1f} MB")
        # Verify it's valid
        with FeatureStore(storage_path, mode='r') as store:
            info = store.get_all_info()
            for split, details in info.items():
                print(f"    {split}: {details['n_written']}/{details['n_total']} written")
        print("  [PASS] Existing storage verified")
        return

    with FeatureStore(storage_path, mode='w') as store:
        store.initialize(
            splits={
                'train': 800_000,
                'val': 100_000,
                'test': 100_000,
            },
            chunk_size=10_000,
            compression='gzip',
            compression_opts=1,
        )
        info = store.get_all_info()
        for split, details in info.items():
            print(f"    {split}: kmer={details['kmer_shape']}, "
                  f"asm={details['assembly_shape']}, chunks={details['chunks']}")

    file_size = os.path.getsize(storage_path)
    print(f"  Storage file created: {storage_path}")
    print(f"  Initial size: {file_size / 1e6:.1f} MB")
    print("  [PASS] Project storage initialized")


if __name__ == '__main__':
    test_initialize()
    test_write_read()
    test_performance()
    test_real_storage_init()
    print("\n" + "=" * 70)
    print("ALL STORAGE MODULE TESTS PASSED")
    print("=" * 70)
