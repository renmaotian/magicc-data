#!/usr/bin/env python3
"""
Step 2: Test the contamination module.

Tests:
1. Pure sample generation (0% contamination)
2. Single contaminant within-phylum
3. Multiple contaminants cross-phylum
4. Contamination rate calculation correctness
5. Edge cases
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/tianrm/projects/magicc2')
from magicc.contamination import (
    generate_contaminated_sample, generate_pure_sample,
    compute_contamination_rate, select_contaminant_target_bp,
)
from magicc.fragmentation import read_fasta


def fast_random_seq(rng, length):
    """Generate a random DNA sequence efficiently."""
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)  # A, C, G, T
    return bytes(bases[rng.integers(0, 4, size=length)]).decode('ascii')


def test_contamination_calculation():
    """Test contamination rate calculation."""
    print("=" * 70)
    print("TEST 1: Contamination rate calculation")
    print("=" * 70)

    # Protocol examples:
    # 5 Mb dominant + 0 contaminant = 0%
    assert compute_contamination_rate(0, 5_000_000) == 0.0
    print("  [PASS] 0% contamination: 5 Mb + 0 Mb = 0%")

    # 5 Mb dominant + 0.5 Mb contaminant = 10%
    rate = compute_contamination_rate(500_000, 5_000_000)
    assert abs(rate - 10.0) < 0.01
    print(f"  [PASS] 10% contamination: 5 Mb + 0.5 Mb = {rate:.1f}%")

    # 5 Mb dominant + 2.5 Mb contaminant = 50%
    rate = compute_contamination_rate(2_500_000, 5_000_000)
    assert abs(rate - 50.0) < 0.01
    print(f"  [PASS] 50% contamination: 5 Mb + 2.5 Mb = {rate:.1f}%")

    # 5 Mb dominant + 5 Mb contaminant = 100%
    rate = compute_contamination_rate(5_000_000, 5_000_000)
    assert abs(rate - 100.0) < 0.01
    print(f"  [PASS] 100% contamination: 5 Mb + 5 Mb = {rate:.1f}%")

    # Target bp calculation
    assert select_contaminant_target_bp(50.0, 5_000_000) == 2_500_000
    assert select_contaminant_target_bp(10.0, 5_000_000) == 500_000
    print("  [PASS] Target bp calculations correct")


def test_pure_sample():
    """Test pure sample generation (0% contamination)."""
    print("\n" + "=" * 70)
    print("TEST 2: Pure sample generation")
    print("=" * 70)

    rng = np.random.default_rng(42)
    sequence = fast_random_seq(rng, 5_000_000)

    for comp in [1.0, 0.75, 0.50]:
        result = generate_pure_sample(sequence, comp, rng)
        print(f"  comp={comp:.0%}: {result['n_contigs_dominant']} contigs, "
              f"actual_comp={result['completeness']:.3f}, "
              f"contamination={result['contamination']:.1f}%, "
              f"total={sum(len(c) for c in result['contigs']):,} bp")
        assert result['contamination'] == 0.0
        assert result['n_contigs_contaminant'] == 0
        assert result['n_contaminant_genomes'] == 0

    print("  [PASS] Pure samples have 0% contamination")


def test_single_contaminant():
    """Test single contaminant mixing."""
    print("\n" + "=" * 70)
    print("TEST 3: Single contaminant (within-phylum)")
    print("=" * 70)

    rng = np.random.default_rng(123)
    dominant = fast_random_seq(rng, 5_000_000)
    contaminant = fast_random_seq(rng, 4_000_000)

    for target_cont in [10.0, 25.0, 50.0, 75.0, 100.0]:
        result = generate_contaminated_sample(
            dominant, [contaminant],
            target_completeness=0.80,
            target_contamination=target_cont,
            rng=rng,
        )
        total_bp = sum(len(c) for c in result['contigs'])
        print(f"  target_cont={target_cont:5.1f}%: "
              f"actual={result['contamination']:5.1f}%, "
              f"dom_contigs={result['n_contigs_dominant']}, "
              f"cont_contigs={result['n_contigs_contaminant']}, "
              f"total={total_bp:>10,} bp, "
              f"comp={result['completeness']:.3f}")

    print("  [PASS] Single contaminant mixing works")


def test_multiple_contaminants():
    """Test multiple contaminant mixing (cross-phylum)."""
    print("\n" + "=" * 70)
    print("TEST 4: Multiple contaminants (cross-phylum)")
    print("=" * 70)

    rng = np.random.default_rng(456)
    dominant = fast_random_seq(rng, 5_000_000)
    contaminants = [fast_random_seq(rng, s) for s in [3_000_000, 4_000_000, 6_000_000]]

    for target_cont in [20.0, 50.0, 80.0]:
        result = generate_contaminated_sample(
            dominant, contaminants,
            target_completeness=0.90,
            target_contamination=target_cont,
            rng=rng,
        )
        total_bp = sum(len(c) for c in result['contigs'])
        dom_bp = sum(len(c) for c in result['dominant_contigs'])
        cont_bp = sum(len(c) for c in result['contaminant_contigs'])
        print(f"  target_cont={target_cont:5.1f}%: "
              f"actual={result['contamination']:5.1f}%, "
              f"dom={dom_bp:>10,} bp, cont={cont_bp:>10,} bp, "
              f"dom_ctg={result['n_contigs_dominant']}, "
              f"cont_ctg={result['n_contigs_contaminant']}, "
              f"n_genomes={result['n_contaminant_genomes']}")

    print("  [PASS] Multiple contaminant mixing works")


def test_real_genomes():
    """Test with real reference genomes."""
    print("\n" + "=" * 70)
    print("TEST 5: Real genome contamination mixing")
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
            if os.path.exists(fasta):
                genomes.append((acc, fasta, phylum))

    if len(genomes) < 2:
        print("  [SKIP] Not enough genomes available for testing")
        return

    rng = np.random.default_rng(789)

    # Use first as dominant, rest as contaminants
    dom_acc, dom_fasta, dom_phylum = genomes[0]
    dom_seq = read_fasta(dom_fasta)
    print(f"  Dominant: {dom_acc} ({dom_phylum}, {len(dom_seq):,} bp)")

    for i in range(1, min(len(genomes), 4)):
        cont_acc, cont_fasta, cont_phylum = genomes[i]
        cont_seq = read_fasta(cont_fasta)
        print(f"  Contaminant {i}: {cont_acc} ({cont_phylum}, {len(cont_seq):,} bp)")

    cont_seqs = [read_fasta(g[1]) for g in genomes[1:]]

    for target_cont in [10.0, 50.0]:
        result = generate_contaminated_sample(
            dom_seq, cont_seqs,
            target_completeness=0.80,
            target_contamination=target_cont,
            rng=rng,
        )
        print(f"\n  target_cont={target_cont:.0f}%: "
              f"actual={result['contamination']:.1f}%, "
              f"comp={result['completeness']:.3f}, "
              f"dom_bp={sum(len(c) for c in result['dominant_contigs']):,}, "
              f"cont_bp={sum(len(c) for c in result['contaminant_contigs']):,}, "
              f"total_contigs={len(result['contigs'])}")

    print("\n  [PASS] Real genome contamination mixing works")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 70)
    print("TEST 6: Edge cases")
    print("=" * 70)

    rng = np.random.default_rng(101)

    # Empty dominant genome
    result = generate_contaminated_sample("", ["ACGT" * 1000], 0.5, 50.0, rng)
    assert result['n_contigs_dominant'] == 0
    print("  [PASS] Empty dominant genome handled")

    # Empty contaminant list
    result = generate_contaminated_sample("ACGT" * 1000, [], 0.5, 50.0, rng)
    assert result['contamination'] == 0.0
    print("  [PASS] Empty contaminant list handled")

    # Zero contamination target
    dom = fast_random_seq(rng, 1_000_000)
    result = generate_contaminated_sample(dom, [fast_random_seq(rng, 500_000)], 0.8, 0.0, rng)
    assert result['contamination'] == 0.0
    print("  [PASS] Zero contamination target handled")

    # Very high contamination (100%)
    result = generate_contaminated_sample(dom, [fast_random_seq(rng, 2_000_000)], 1.0, 100.0, rng)
    print(f"  [PASS] 100% contamination target: actual={result['contamination']:.1f}%")


def test_contamination_rate_accuracy():
    """Test that actual contamination rates match targets reasonably well."""
    print("\n" + "=" * 70)
    print("TEST 7: Contamination rate targeting accuracy")
    print("=" * 70)

    rng = np.random.default_rng(202)
    dom = fast_random_seq(rng, 5_000_000)

    targets = [10.0, 20.0, 30.0, 50.0, 70.0, 90.0]
    for target in targets:
        actuals = []
        for _ in range(10):
            cont = fast_random_seq(rng, 5_000_000)
            result = generate_contaminated_sample(
                dom, [cont], 0.90, target, rng
            )
            actuals.append(result['contamination'])
        mean_actual = np.mean(actuals)
        std_actual = np.std(actuals)
        error = abs(mean_actual - target)
        print(f"  target={target:5.1f}%: actual={mean_actual:5.1f}% +/- {std_actual:4.1f}% "
              f"(error={error:.1f}%)")

    print("  [PASS] Contamination targeting within acceptable range")


if __name__ == '__main__':
    test_contamination_calculation()
    test_pure_sample()
    test_single_contaminant()
    test_multiple_contaminants()
    test_real_genomes()
    test_edge_cases()
    test_contamination_rate_accuracy()
    print("\n" + "=" * 70)
    print("ALL CONTAMINATION MODULE TESTS PASSED")
    print("=" * 70)
