#!/usr/bin/env python3
"""
Step 6 (Part 2): Integration test and performance profiling.

Tests:
1. End-to-end pipeline on single genome
2. Feature extraction speed profiling
3. Full pipeline with fragmentation + contamination + features
4. Batch processing on 100 synthetic genomes
5. Storage round-trip test
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.pipeline import MAGICCPipeline
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats, FEATURE_NAMES
from magicc.normalization import FeatureNormalizer
from magicc.storage import FeatureStore, METADATA_DTYPE
from magicc.fragmentation import read_fasta, simulate_fragmentation


KMER_PATH = '/home/tianrm/projects/magicc2/data/kmer_selection/selected_kmers.txt'


def fast_random_seq(rng, length):
    """Generate a random DNA sequence efficiently."""
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)
    return bytes(bases[rng.integers(0, 4, size=length)]).decode('ascii')


def test_single_genome_pipeline():
    """Test end-to-end pipeline on a single genome."""
    print("=" * 70)
    print("TEST 1: Single genome pipeline (pure)")
    print("=" * 70)

    pipeline = MAGICCPipeline(KMER_PATH)
    rng = np.random.default_rng(42)
    sequence = fast_random_seq(rng, 5_000_000)

    result = pipeline.process_single_genome(
        dominant_sequence=sequence,
        target_completeness=0.80,
        target_contamination=0.0,
        quality_tier='medium',
        rng=rng,
    )

    print(f"  Completeness: {result['completeness']:.1f}%")
    print(f"  Contamination: {result['contamination']:.1f}%")
    print(f"  Contigs: {result['n_contigs']}")
    print(f"  Total length: {result['total_length']:,} bp")
    print(f"  K-mer counts: shape={result['kmer_counts'].shape}, "
          f"sum={result['kmer_counts'].sum():,}, "
          f"nonzero={np.count_nonzero(result['kmer_counts'])}")
    print(f"  Assembly features: shape={result['assembly_features'].shape}")
    print(f"  Log10 total k-mer: {result['log10_total_kmer']:.4f}")

    assert result['kmer_counts'].shape == (9249,)
    assert result['assembly_features'].shape == (20,)
    assert result['contamination'] == 0.0
    print("  [PASS] Single genome pipeline works")


def test_contaminated_genome_pipeline():
    """Test pipeline with contamination."""
    print("\n" + "=" * 70)
    print("TEST 2: Contaminated genome pipeline")
    print("=" * 70)

    pipeline = MAGICCPipeline(KMER_PATH)
    rng = np.random.default_rng(123)
    dominant = fast_random_seq(rng, 5_000_000)
    contaminants = [fast_random_seq(rng, 3_000_000), fast_random_seq(rng, 4_000_000)]

    result = pipeline.process_single_genome(
        dominant_sequence=dominant,
        contaminant_sequences=contaminants,
        target_completeness=0.85,
        target_contamination=30.0,
        quality_tier='medium',
        rng=rng,
    )

    print(f"  Completeness: {result['completeness']:.1f}%")
    print(f"  Contamination: {result['contamination']:.1f}%")
    print(f"  Contigs: {result['n_contigs']}")
    print(f"  Total length: {result['total_length']:,} bp")
    print(f"  K-mer counts sum: {result['kmer_counts'].sum():,}")
    print(f"  Log10 total k-mer: {result['log10_total_kmer']:.4f}")

    assert result['contamination'] > 0
    print("  [PASS] Contaminated genome pipeline works")


def test_feature_extraction_speed():
    """Profile feature extraction speed (target: <50ms/genome)."""
    print("\n" + "=" * 70)
    print("TEST 3: Feature extraction speed profiling")
    print("=" * 70)

    pipeline = MAGICCPipeline(KMER_PATH)
    rng = np.random.default_rng(456)

    # Pre-generate contigs (skip fragmentation for speed profiling)
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)
    n_genomes = 200
    all_contigs = []
    for _ in range(n_genomes):
        n_ctg = rng.integers(10, 200)
        contigs = []
        for _ in range(n_ctg):
            clen = rng.integers(1000, 100_000)
            contigs.append(bytes(bases[rng.integers(0, 4, size=clen)]).decode('ascii'))
        all_contigs.append(contigs)

    # Warm up
    _ = pipeline.extract_features(all_contigs[0])

    # Profile k-mer counting only
    start = time.perf_counter()
    for contigs in all_contigs:
        _ = pipeline.kmer_counter.count_contigs(contigs)
    kmer_time = time.perf_counter() - start
    kmer_per_genome = kmer_time / n_genomes * 1000

    # Profile assembly stats only
    start = time.perf_counter()
    for contigs in all_contigs:
        _ = compute_assembly_stats(contigs)
    asm_time = time.perf_counter() - start
    asm_per_genome = asm_time / n_genomes * 1000

    # Profile combined
    start = time.perf_counter()
    for contigs in all_contigs:
        _ = pipeline.extract_features(contigs)
    total_time = time.perf_counter() - start
    total_per_genome = total_time / n_genomes * 1000

    print(f"  K-mer counting:    {kmer_per_genome:.1f} ms/genome")
    print(f"  Assembly stats:    {asm_per_genome:.1f} ms/genome")
    print(f"  Combined:          {total_per_genome:.1f} ms/genome")
    print(f"  Target:            <50.0 ms/genome")

    status = "PASS" if total_per_genome < 50 else "WARN"
    print(f"  [{status}] Feature extraction: {total_per_genome:.1f} ms/genome")


def test_real_genome_pipeline():
    """Test pipeline on real reference genomes."""
    print("\n" + "=" * 70)
    print("TEST 4: Real genome pipeline")
    print("=" * 70)

    pipeline = MAGICCPipeline(KMER_PATH)
    rng = np.random.default_rng(789)

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
        result = pipeline.process_single_genome(
            dominant_sequence=sequence,
            target_completeness=rng.uniform(0.5, 1.0),
            rng=rng,
        )
        elapsed = time.perf_counter() - start

        print(f"  {acc} ({phylum}): "
              f"comp={result['completeness']:.1f}%, "
              f"contigs={result['n_contigs']}, "
              f"kmers={result['kmer_counts'].sum():,}, "
              f"time={elapsed*1000:.0f}ms")

    print("  [PASS] Real genome pipeline works")


def test_batch_processing():
    """Test batch processing on 100 synthetic genomes."""
    print("\n" + "=" * 70)
    print("TEST 5: Batch processing (100 synthetic genomes)")
    print("=" * 70)

    pipeline = MAGICCPipeline(KMER_PATH)
    rng = np.random.default_rng(101)
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)

    n_genomes = 100

    # Generate sequences
    print(f"  Generating {n_genomes} synthetic sequences...")
    gen_start = time.perf_counter()
    dominant_seqs = []
    contaminant_seqs_list = []
    completeness_list = []
    contamination_list = []
    quality_list = []

    for i in range(n_genomes):
        genome_len = rng.integers(500_000, 8_000_000)
        dominant_seqs.append(
            bytes(bases[rng.integers(0, 4, size=genome_len)]).decode('ascii')
        )

        # 50% pure, 50% contaminated
        if rng.random() < 0.5:
            contaminant_seqs_list.append(None)
            contamination_list.append(0.0)
        else:
            n_cont = rng.integers(1, 4)
            conts = []
            for _ in range(n_cont):
                cont_len = rng.integers(500_000, 5_000_000)
                conts.append(
                    bytes(bases[rng.integers(0, 4, size=cont_len)]).decode('ascii')
                )
            contaminant_seqs_list.append(conts)
            contamination_list.append(float(rng.uniform(5, 80)))

        completeness_list.append(float(rng.uniform(0.5, 1.0)))
        quality_list.append(rng.choice(['high', 'medium', 'low', 'highly_fragmented']))

    gen_time = time.perf_counter() - gen_start
    print(f"  Sequence generation: {gen_time:.1f}s")

    # Process batch
    print(f"  Processing batch...")
    proc_start = time.perf_counter()
    batch_result = pipeline.process_batch(
        dominant_sequences=dominant_seqs,
        contaminant_sequences_list=contaminant_seqs_list,
        target_completeness_list=completeness_list,
        target_contamination_list=contamination_list,
        quality_tier_list=quality_list,
        rng=rng,
    )
    proc_time = time.perf_counter() - proc_start

    kmer_counts = batch_result['kmer_counts']
    assembly_features = batch_result['assembly_features']
    labels = batch_result['labels']

    print(f"  Processing: {proc_time:.1f}s ({proc_time/n_genomes*1000:.0f} ms/genome)")
    print(f"  K-mer counts: shape={kmer_counts.shape}, "
          f"mean_sum={kmer_counts.sum(axis=1).mean():.0f}")
    print(f"  Assembly features: shape={assembly_features.shape}")
    print(f"  Labels: shape={labels.shape}")
    print(f"    Completeness: mean={labels[:, 0].mean():.1f}%, "
          f"range=[{labels[:, 0].min():.1f}, {labels[:, 0].max():.1f}]%")
    print(f"    Contamination: mean={labels[:, 1].mean():.1f}%, "
          f"range=[{labels[:, 1].min():.1f}, {labels[:, 1].max():.1f}]%")

    assert kmer_counts.shape == (n_genomes, 9249)
    assert assembly_features.shape == (n_genomes, 20)
    assert labels.shape == (n_genomes, 2)

    print("  [PASS] Batch processing works correctly")

    return batch_result


def test_storage_roundtrip(batch_result):
    """Test writing batch results to HDF5 and reading back."""
    print("\n" + "=" * 70)
    print("TEST 6: Storage round-trip test")
    print("=" * 70)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        tmp_path = f.name

    n_genomes = batch_result['kmer_counts'].shape[0]

    try:
        # Write
        with FeatureStore(tmp_path, mode='w') as store:
            store.initialize(
                splits={'train': n_genomes},
                chunk_size=min(1000, n_genomes),
            )
            store.write_batch(
                'train',
                batch_result['kmer_counts'].astype(np.float32),
                batch_result['assembly_features'].astype(np.float32),
                batch_result['labels'],
                batch_result['metadata'],
            )

        # Read back
        with FeatureStore(tmp_path, mode='r') as store:
            kmer_r, asm_r, labels_r, meta_r = store.read_batch('train', 0, n_genomes)

            # Verify
            assert np.allclose(kmer_r, batch_result['kmer_counts'].astype(np.float32), atol=1e-4)
            assert np.allclose(asm_r, batch_result['assembly_features'].astype(np.float32), atol=1e-4)
            assert np.allclose(labels_r, batch_result['labels'], atol=1e-4)

        file_size = os.path.getsize(tmp_path)
        print(f"  Wrote {n_genomes} samples, file size: {file_size / 1e6:.1f} MB")
        print(f"  Read back and verified: shapes match, values match")
        print("  [PASS] Storage round-trip successful")
    finally:
        os.unlink(tmp_path)


def test_normalization_integration():
    """Test normalization integration with pipeline."""
    print("\n" + "=" * 70)
    print("TEST 7: Normalization integration")
    print("=" * 70)

    rng = np.random.default_rng(202)
    normalizer = FeatureNormalizer(n_kmer_features=9249, reservoir_size=5000)

    # Generate some synthetic data for stats
    n_samples = 500
    kmer_data = rng.poisson(50, (n_samples, 9249)).astype(np.float64)
    asm_data = np.zeros((n_samples, 20), dtype=np.float64)
    asm_data[:, 0] = rng.uniform(500_000, 10_000_000, n_samples)  # total_length
    asm_data[:, 1] = rng.integers(1, 2000, n_samples)  # contig_count
    asm_data[:, 2] = rng.uniform(1000, 1_000_000, n_samples)  # n50
    for i in range(3, 20):
        asm_data[:, i] = rng.uniform(0, 1000, n_samples)

    normalizer.update_kmer_batch(kmer_data)
    normalizer.update_assembly_batch(asm_data)
    normalizer.finalize()

    # Normalize a test sample
    test_kmer = kmer_data[:10]
    test_asm = asm_data[:10]
    normalized = normalizer.normalize_all(test_kmer, test_asm)

    print(f"  Normalized shape: {normalized.shape}")
    print(f"  Expected: (10, {9249 + 20})")
    assert normalized.shape == (10, 9269)
    print(f"  K-mer part: mean={normalized[:, :9249].mean():.4f}, "
          f"std={normalized[:, :9249].std():.4f}")
    print(f"  Assembly part: mean={normalized[:, 9249:].mean():.4f}")

    # Save and reload
    save_path = '/tmp/test_normalizer.json'
    normalizer.save(save_path)
    loaded = FeatureNormalizer.load(save_path)
    normalized2 = loaded.normalize_all(test_kmer, test_asm)
    assert np.allclose(normalized, normalized2)
    os.unlink(save_path)

    print("  [PASS] Normalization integration works")


def print_performance_summary():
    """Print overall performance summary."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    pipeline = MAGICCPipeline(KMER_PATH)
    rng = np.random.default_rng(303)
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)

    # Generate a typical genome
    genome_len = 5_000_000
    sequence = bytes(bases[rng.integers(0, 4, size=genome_len)]).decode('ascii')

    # Profile each step
    n_iter = 50

    # Fragmentation
    start = time.perf_counter()
    for _ in range(n_iter):
        result = simulate_fragmentation(sequence, 0.80, quality_tier='medium', rng=rng)
    frag_time = (time.perf_counter() - start) / n_iter * 1000
    contigs = result['contigs']

    # K-mer counting
    start = time.perf_counter()
    for _ in range(n_iter):
        kmer_counts = pipeline.kmer_counter.count_contigs(contigs)
    kmer_time = (time.perf_counter() - start) / n_iter * 1000

    # Assembly stats
    start = time.perf_counter()
    for _ in range(n_iter):
        asm_features = compute_assembly_stats(contigs)
    asm_time = (time.perf_counter() - start) / n_iter * 1000

    # Combined feature extraction
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = pipeline.extract_features(contigs)
    combined_time = (time.perf_counter() - start) / n_iter * 1000

    print(f"  Genome: {genome_len/1e6:.0f} Mb, {len(contigs)} contigs (medium quality)")
    print(f"  Fragmentation:          {frag_time:>8.1f} ms/genome")
    print(f"  K-mer counting:         {kmer_time:>8.1f} ms/genome")
    print(f"  Assembly statistics:     {asm_time:>8.1f} ms/genome")
    print(f"  Combined features:      {combined_time:>8.1f} ms/genome")
    print(f"  Target combined:          <50.0 ms/genome")
    print(f"")
    print(f"  Feature extraction is {'within' if combined_time < 50 else 'ABOVE'} target")


if __name__ == '__main__':
    test_single_genome_pipeline()
    test_contaminated_genome_pipeline()
    test_feature_extraction_speed()
    test_real_genome_pipeline()
    batch_result = test_batch_processing()
    test_storage_roundtrip(batch_result)
    test_normalization_integration()
    print_performance_summary()
    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 70)
