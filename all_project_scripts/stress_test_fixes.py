#!/usr/bin/env python3
"""
Stress test: verify completeness and contamination ranges after fixes.

Generates 1000 synthetic genomes with random parameters and checks:
- All completeness values are in [50%, 100%]
- All contamination values are in [0%, 100%]
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, '/home/tianrm/projects/magicc2')

from magicc.fragmentation import simulate_fragmentation, _warm_numba_fragmentation
from magicc.contamination import generate_contaminated_sample, generate_pure_sample

# Warm up Numba
_warm_numba_fragmentation()

N_TESTS = 1000
rng = np.random.default_rng(42)

# Pre-generate genome sequences efficiently using numpy
_BASES = np.array([65, 67, 71, 84], dtype=np.uint8)  # A, C, G, T

def random_genome(size, rng):
    """Generate a random genome string efficiently."""
    indices = rng.integers(0, 4, size=size)
    return _BASES[indices].tobytes().decode('ascii')

print(f"Generating {N_TESTS} synthetic genomes...", flush=True)
start = time.time()

completeness_values = []
contamination_values = []
errors = []

quality_tiers = ['high', 'medium', 'low', 'highly_fragmented']

for i in range(N_TESTS):
    try:
        # Use smaller genomes (50kb-1Mb) for speed while still exercising the logic
        genome_size = int(rng.integers(50_000, 1_000_000))
        dominant_seq = random_genome(genome_size, rng)

        target_completeness = float(rng.uniform(0.5, 1.0))
        target_contamination = float(rng.uniform(0.0, 100.0))
        quality_tier = rng.choice(quality_tiers)

        # 50% pure, 50% contaminated
        if i % 2 == 0:
            # Pure sample
            sample = generate_pure_sample(
                dominant_sequence=dominant_seq,
                target_completeness=target_completeness,
                rng=np.random.default_rng(rng.integers(0, 2**32)),
                quality_tier=quality_tier,
            )
        else:
            # Contaminated sample with high target contamination to stress-test the cap
            n_contaminants = int(rng.integers(1, 4))
            contaminant_seqs = []
            for _ in range(n_contaminants):
                csize = int(rng.integers(30_000, 800_000))
                contaminant_seqs.append(random_genome(csize, rng))

            sample = generate_contaminated_sample(
                dominant_sequence=dominant_seq,
                contaminant_sequences=contaminant_seqs,
                target_completeness=target_completeness,
                target_contamination=target_contamination,
                rng=np.random.default_rng(rng.integers(0, 2**32)),
                dominant_quality_tier=quality_tier,
            )

        comp = sample['completeness'] * 100.0  # Convert fraction to percentage
        contam = sample['contamination']

        completeness_values.append(comp)
        contamination_values.append(contam)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{N_TESTS} done ({elapsed:.1f}s)", flush=True)

    except Exception as e:
        errors.append((i, str(e)))
        import traceback
        print(f"  ERROR at sample {i}: {e}", flush=True)
        traceback.print_exc()

elapsed = time.time() - start
print(f"\nCompleted {len(completeness_values)} tests in {elapsed:.1f}s "
      f"({len(completeness_values)/elapsed:.1f} samples/sec)", flush=True)
print(f"Errors: {len(errors)}", flush=True)

comp_arr = np.array(completeness_values)
contam_arr = np.array(contamination_values)

print(f"\n{'='*60}")
print(f"COMPLETENESS RESULTS:")
print(f"  Min:    {comp_arr.min():.2f}%")
print(f"  Max:    {comp_arr.max():.2f}%")
print(f"  Mean:   {comp_arr.mean():.2f}%")
print(f"  Median: {np.median(comp_arr):.2f}%")
print(f"  < 50%:  {(comp_arr < 50.0).sum()} samples")
print(f"  >= 50%: {(comp_arr >= 50.0).sum()} samples")
print(f"  > 100%: {(comp_arr > 100.0).sum()} samples")

print(f"\nCONTAMINATION RESULTS:")
print(f"  Min:    {contam_arr.min():.2f}%")
print(f"  Max:    {contam_arr.max():.2f}%")
print(f"  Mean:   {contam_arr.mean():.2f}%")
print(f"  Median: {np.median(contam_arr):.2f}%")
print(f"  < 0%:   {(contam_arr < 0.0).sum()} samples")
print(f"  > 100%: {(contam_arr > 100.0).sum()} samples")

print(f"\n{'='*60}")
comp_ok = (comp_arr >= 50.0).all() and (comp_arr <= 100.01).all()
contam_ok = (contam_arr >= 0.0).all() and (contam_arr <= 100.01).all()

if comp_ok:
    print("COMPLETENESS: PASS (all values in [50%, 100%])")
else:
    print("COMPLETENESS: FAIL")
    if (comp_arr < 50.0).any():
        bad = comp_arr[comp_arr < 50.0]
        print(f"  {len(bad)} samples below 50%: min={bad.min():.2f}%")
    if (comp_arr > 100.01).any():
        bad = comp_arr[comp_arr > 100.01]
        print(f"  {len(bad)} samples above 100%: max={bad.max():.2f}%")

if contam_ok:
    print("CONTAMINATION: PASS (all values in [0%, 100%])")
else:
    print("CONTAMINATION: FAIL")
    if (contam_arr < 0.0).any():
        print(f"  {(contam_arr < 0.0).sum()} samples below 0%")
    if (contam_arr > 100.01).any():
        bad = contam_arr[contam_arr > 100.01]
        print(f"  {len(bad)} samples above 100%: max={bad.max():.2f}%")

print(f"{'='*60}")
if comp_ok and contam_ok:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
sys.exit(0 if (comp_ok and contam_ok) else 1)
