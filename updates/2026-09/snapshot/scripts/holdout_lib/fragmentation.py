"""
Genome Fragmentation Module for MAGICC.

Simulates realistic MAG fragmentation patterns including:
- Coverage-based dropout (log-normal coverage distribution)
- Contig-level removal (binning failures)
- GC-biased loss (extreme GC regions)
- Repeat region exclusion (repetitive elements)

Quality tiers:
- High:              10-50 contigs,    N50 100-500 kb, min contig 5 kb
- Medium:            50-200 contigs,   N50 20-100 kb,  min contig 2 kb
- Low:               200-500 contigs,  N50 5-20 kb,    min contig 1 kb
- Highly fragmented: 500-2000 contigs, N50 1-5 kb,     min contig 500 bp
"""

import numpy as np
import numba
from numba import njit, types
from typing import List, Tuple, Optional, Dict, Any


# ============================================================================
# Numba-accelerated helper functions
# ============================================================================

# Lookup tables for character classification (ASCII)
_GC_LOOKUP = np.zeros(256, dtype=np.int8)
_GC_LOOKUP[ord('G')] = 1
_GC_LOOKUP[ord('C')] = 1
_GC_LOOKUP[ord('g')] = 1
_GC_LOOKUP[ord('c')] = 1

_VALID_LOOKUP = np.zeros(256, dtype=np.int8)
for _c in b'ACGTacgt':
    _VALID_LOOKUP[_c] = 1


@njit(cache=True)
def _compute_gc_count(seq_bytes, gc_lookup, valid_lookup):
    """Count GC and total valid bases using lookup table."""
    gc = 0
    total = 0
    for i in range(len(seq_bytes)):
        b = seq_bytes[i]
        gc += gc_lookup[b]
        total += valid_lookup[b]
    return gc, total


@njit(cache=True)
def _compute_repeat_score(seq_bytes):
    """Compute repeat score (homopolymer fraction + max dinucleotide frequency).

    Operates on uppercase ASCII byte array.
    """
    n = len(seq_bytes)
    if n < 2:
        return 0.0

    # Count homopolymer bases (runs of >=6 identical chars)
    homopolymer_bases = 0
    run_len = 1
    for i in range(1, n):
        if seq_bytes[i] == seq_bytes[i - 1]:
            run_len += 1
        else:
            if run_len >= 6:
                homopolymer_bases += run_len
            run_len = 1
    if run_len >= 6:
        homopolymer_bases += run_len

    # Count dinucleotide frequencies using a flat array
    # Encode each dinucleotide as first_byte * 256 + second_byte
    # But that's too large. Use a simpler approach: just 16 possible dinucs (ACGT x ACGT)
    # Map A=0, C=1, G=2, T=3, other=-1
    dinuc_counts = np.zeros(16, dtype=np.int64)
    n_dinucs = 0
    for i in range(n - 1):
        b1 = seq_bytes[i]
        b2 = seq_bytes[i + 1]
        # Map to 0-3
        if b1 == 65:  # A
            v1 = 0
        elif b1 == 67:  # C
            v1 = 1
        elif b1 == 71:  # G
            v1 = 2
        elif b1 == 84:  # T
            v1 = 3
        else:
            continue
        if b2 == 65:
            v2 = 0
        elif b2 == 67:
            v2 = 1
        elif b2 == 71:
            v2 = 2
        elif b2 == 84:
            v2 = 3
        else:
            continue
        dinuc_counts[v1 * 4 + v2] += 1
        n_dinucs += 1

    max_dinuc_freq = 0.0
    if n_dinucs > 0:
        for j in range(16):
            freq = dinuc_counts[j] / n_dinucs
            if freq > max_dinuc_freq:
                max_dinuc_freq = freq

    return homopolymer_bases / n + max_dinuc_freq


def _warm_numba_fragmentation():
    """Warm up Numba JIT for fragmentation functions."""
    dummy = np.frombuffer(b'ACGTACGTACGTACGT', dtype=np.uint8)
    _compute_gc_count(dummy, _GC_LOOKUP, _VALID_LOOKUP)
    _compute_repeat_score(dummy)

# Quality tier definitions: (contig_min, contig_max, n50_min, n50_max, min_contig_bp)
QUALITY_TIERS = {
    'high':               (10,   50,   100_000, 500_000, 5_000),
    'medium':             (50,   200,  20_000,  100_000, 2_000),
    'low':                (200,  500,  5_000,   20_000,  1_000),
    'highly_fragmented':  (500,  2000, 1_000,   5_000,   500),
}


def select_quality_tier(rng: np.random.Generator) -> str:
    """Select a random quality tier with equal probability."""
    return rng.choice(list(QUALITY_TIERS.keys()))


def generate_contig_lengths(
    genome_length: int,
    quality_tier: str,
    rng: np.random.Generator,
    sigma_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    Generate contig lengths from a log-normal distribution matching a quality tier.

    Algorithm (from protocol):
    1. Sample target N50 from quality-appropriate range (uniform distribution)
    2. Generate contig lengths from log-normal with mu=log(N50), sigma=0.8-1.2
    3. Adjust to match genome length

    Parameters
    ----------
    genome_length : int
        Total length of the reference genome in bp.
    quality_tier : str
        One of 'high', 'medium', 'low', 'highly_fragmented'.
    rng : np.random.Generator
        Random number generator for reproducibility.
    sigma_range : tuple
        Range for log-normal sigma parameter.

    Returns
    -------
    np.ndarray
        Array of contig lengths (integers, sum equals genome_length).
    """
    if genome_length <= 0:
        return np.array([], dtype=np.int64)

    contig_min, contig_max, n50_min, n50_max, min_contig_bp = QUALITY_TIERS[quality_tier]

    # Sample target N50 uniformly from tier range
    target_n50 = rng.integers(n50_min, n50_max + 1)

    # Clamp N50 to genome length (for very small genomes)
    target_n50 = min(target_n50, genome_length)

    # If genome is too small for fragmentation, return single contig
    if genome_length <= min_contig_bp:
        return np.array([genome_length], dtype=np.int64)

    # Log-normal parameters
    mu = np.log(target_n50)
    sigma = rng.uniform(sigma_range[0], sigma_range[1])

    # Generate enough contig lengths to cover genome
    # Over-generate and then trim
    max_attempts = 50
    for attempt in range(max_attempts):
        # Estimate how many contigs we need
        expected_mean = np.exp(mu + sigma**2 / 2)
        n_estimate = max(int(genome_length / expected_mean * 1.5), contig_min * 2)
        n_estimate = max(n_estimate, 10)

        raw_lengths = rng.lognormal(mean=mu, sigma=sigma, size=n_estimate)
        raw_lengths = np.maximum(raw_lengths, min_contig_bp).astype(np.int64)

        # Sort descending and accumulate
        raw_lengths = np.sort(raw_lengths)[::-1]
        cumsum = np.cumsum(raw_lengths)

        # Find where cumulative sum exceeds genome length
        idx = np.searchsorted(cumsum, genome_length)
        if idx < len(raw_lengths):
            contigs = raw_lengths[:idx + 1].copy()
            # Adjust last contig to match genome length exactly
            contigs[-1] = genome_length - cumsum[idx - 1] if idx > 0 else genome_length
            if contigs[-1] < 1:
                # Drop last and adjust previous
                contigs = contigs[:-1]
                if len(contigs) == 0:
                    contigs = np.array([genome_length], dtype=np.int64)
                else:
                    contigs[-1] += (genome_length - contigs.sum())
            break
        else:
            # Need more contigs, reduce sigma or increase n_estimate
            sigma *= 0.9
            if attempt == max_attempts - 1:
                # Fallback: equal-size contigs
                n_contigs = max(contig_min, genome_length // target_n50)
                n_contigs = min(n_contigs, contig_max)
                base_size = genome_length // n_contigs
                remainder = genome_length % n_contigs
                contigs = np.full(n_contigs, base_size, dtype=np.int64)
                contigs[:remainder] += 1

    # Ensure sum is correct
    diff = genome_length - contigs.sum()
    if diff != 0:
        contigs[0] += diff

    # Filter out zero or negative contigs
    contigs = contigs[contigs > 0]

    # If only 1 contig after filtering but tier expects more, that's fine for small genomes
    return contigs


def fragment_genome(
    sequence: str,
    contig_lengths: np.ndarray,
    rng: np.random.Generator,
) -> List[str]:
    """
    Fragment a genome sequence at random positions to create contigs of specified lengths.

    Parameters
    ----------
    sequence : str
        Full reference genome sequence (concatenated contigs if multi-contig).
    contig_lengths : np.ndarray
        Target contig lengths. Must sum to len(sequence).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    list of str
        List of contig sequences.
    """
    genome_len = len(sequence)
    if genome_len == 0:
        return []

    total_target = contig_lengths.sum()
    if total_target != genome_len:
        # Rescale contig lengths proportionally
        scale = genome_len / total_target
        contig_lengths = (contig_lengths * scale).astype(np.int64)
        diff = genome_len - contig_lengths.sum()
        if diff != 0:
            contig_lengths[0] += diff

    # Generate random cut points
    n_contigs = len(contig_lengths)
    if n_contigs <= 1:
        return [sequence]

    # Random permutation of positions to start each contig
    # We shuffle the contig order to randomize which parts of genome get which length
    shuffled_lengths = contig_lengths.copy()
    rng.shuffle(shuffled_lengths)

    # Create contigs by sequential fragmentation at random start
    # First, pick a random start offset to avoid bias toward beginning
    start_offset = rng.integers(0, genome_len)

    contigs = []
    pos = 0
    for length in shuffled_lengths:
        if length <= 0:
            continue
        # Circular extraction with offset
        actual_start = (start_offset + pos) % genome_len
        actual_end = actual_start + length
        if actual_end <= genome_len:
            contigs.append(sequence[actual_start:actual_end])
        else:
            # Wrap around
            contigs.append(sequence[actual_start:] + sequence[:actual_end - genome_len])
        pos += length

    return contigs


def compute_contig_gc(contig: str) -> float:
    """Compute GC content of a contig using Numba-accelerated lookup."""
    if len(contig) == 0:
        return 0.5
    seq_bytes = np.frombuffer(contig.encode('ascii'), dtype=np.uint8)
    gc, total = _compute_gc_count(seq_bytes, _GC_LOOKUP, _VALID_LOOKUP)
    if total == 0:
        return 0.5
    return gc / total


def apply_coverage_dropout(
    contigs: List[str],
    rng: np.random.Generator,
    coverage_mean: float = 30.0,
    coverage_sigma: float = 1.0,
    coverage_threshold: float = 5.0,
) -> List[str]:
    """
    Simulate coverage-based dropout using log-normal distribution.

    Assigns each contig a simulated coverage from a log-normal distribution.
    Drops contigs with coverage below the threshold.

    Parameters
    ----------
    contigs : list of str
        Input contigs.
    rng : np.random.Generator
        Random number generator.
    coverage_mean : float
        Mean of underlying normal distribution for log-normal coverage.
    coverage_sigma : float
        Sigma of underlying normal distribution for log-normal coverage.
    coverage_threshold : float
        Minimum coverage to retain a contig.

    Returns
    -------
    list of str
        Contigs surviving coverage filter.
    """
    if len(contigs) == 0:
        return contigs

    # Log-normal coverage for each contig
    log_coverages = rng.normal(np.log(coverage_mean), coverage_sigma, size=len(contigs))
    coverages = np.exp(log_coverages)

    surviving = [c for c, cov in zip(contigs, coverages) if cov >= coverage_threshold]

    # Ensure at least one contig survives
    if len(surviving) == 0:
        # Keep the one with highest coverage
        best_idx = np.argmax(coverages)
        surviving = [contigs[best_idx]]

    return surviving


def apply_gc_biased_loss(
    contigs: List[str],
    rng: np.random.Generator,
    gc_loss_strength: float = 0.3,
) -> List[str]:
    """
    Preferentially drop contigs with extreme GC content.

    Parameters
    ----------
    contigs : list of str
        Input contigs.
    rng : np.random.Generator
        Random number generator.
    gc_loss_strength : float
        Base probability of dropping extreme GC contigs (0-1).

    Returns
    -------
    list of str
        Contigs surviving GC-biased filter.
    """
    if len(contigs) <= 1:
        return contigs

    # Vectorized GC computation: concatenate all contigs, compute once
    n_contigs = len(contigs)
    gc_values = np.empty(n_contigs, dtype=np.float64)
    for i, c in enumerate(contigs):
        gc_values[i] = compute_contig_gc(c)

    gc_mean = np.mean(gc_values)
    gc_std = np.std(gc_values)

    if gc_std < 1e-10:
        return contigs

    # Probability of dropping increases with distance from mean GC
    z_scores = np.abs(gc_values - gc_mean) / gc_std
    drop_probs = gc_loss_strength * (1 - np.exp(-z_scores))

    # Random survival
    rand_vals = rng.random(n_contigs)
    mask = rand_vals >= drop_probs
    surviving = [contigs[i] for i in range(n_contigs) if mask[i]]

    if len(surviving) == 0:
        # Keep largest contig
        idx = max(range(n_contigs), key=lambda i: len(contigs[i]))
        surviving = [contigs[idx]]

    return surviving


def apply_repeat_exclusion(
    contigs: List[str],
    rng: np.random.Generator,
    repeat_loss_prob: float = 0.15,
    low_complexity_threshold: float = 0.6,
) -> List[str]:
    """
    Higher probability of losing contigs with repetitive/low-complexity regions.

    Approximates repeat regions by detecting low-complexity sequences
    (high frequency of any single dinucleotide or homopolymer runs).
    Uses Numba-accelerated repeat score computation.

    Parameters
    ----------
    contigs : list of str
        Input contigs.
    rng : np.random.Generator
        Random number generator.
    repeat_loss_prob : float
        Base probability of dropping a repeat-rich contig.
    low_complexity_threshold : float
        Fraction of sequence that must be low-complexity to be considered repeat-rich.

    Returns
    -------
    list of str
        Contigs surviving repeat filter.
    """
    if len(contigs) <= 1:
        return contigs

    n_contigs = len(contigs)

    # Pre-compute repeat scores for contigs >= 100 bp using Numba
    repeat_scores = np.zeros(n_contigs, dtype=np.float64)
    contig_lengths = np.empty(n_contigs, dtype=np.int64)
    for i, contig in enumerate(contigs):
        contig_lengths[i] = len(contig)
        if len(contig) >= 100:
            seq_bytes = np.frombuffer(contig.upper().encode('ascii'), dtype=np.uint8)
            repeat_scores[i] = _compute_repeat_score(seq_bytes)

    # Determine which contigs to potentially drop
    high_repeat_mask = (repeat_scores > low_complexity_threshold) & (contig_lengths >= 100)
    rand_vals = rng.random(n_contigs)
    drop_mask = high_repeat_mask & (rand_vals < repeat_loss_prob)

    surviving = [contigs[i] for i in range(n_contigs) if not drop_mask[i]]

    if len(surviving) == 0:
        idx = max(range(n_contigs), key=lambda i: len(contigs[i]))
        surviving = [contigs[idx]]

    return surviving


def apply_completeness(
    contigs: List[str],
    target_completeness: float,
    genome_full_length: int,
    rng: np.random.Generator,
    min_completeness: float = 0.50,
) -> Tuple[List[str], float]:
    """
    Apply completeness by dropping contigs to approach the target genome fraction.

    Strategy for fine-grained control:
    1. Sort contigs by size (smallest first) so dropping is granular.
    2. Shuffle within size groups to maintain some randomness.
    3. Greedily add contigs until target length is reached.
    4. Safety net: if actual completeness < min_completeness (default 50%),
       add back contigs (smallest first from the dropped set) until the
       minimum completeness threshold is met. This ensures the protocol's
       50-100% completeness range is always respected.

    Parameters
    ----------
    contigs : list of str
        Input contigs.
    target_completeness : float
        Target completeness as fraction (0.5 to 1.0).
    genome_full_length : int
        Full reference genome length (for computing actual completeness).
    rng : np.random.Generator
        Random number generator.
    min_completeness : float
        Hard floor for completeness (default 0.50). If after contig dropping
        the actual completeness falls below this, contigs are added back.

    Returns
    -------
    tuple of (list of str, float)
        (Surviving contigs, actual completeness achieved).
    """
    if target_completeness >= 1.0:
        actual = sum(len(c) for c in contigs) / genome_full_length if genome_full_length > 0 else 1.0
        return contigs, actual

    if len(contigs) == 0:
        return contigs, 0.0

    target_length = int(target_completeness * genome_full_length)
    current_length = sum(len(c) for c in contigs)

    if current_length <= target_length:
        actual = current_length / genome_full_length if genome_full_length > 0 else 0.0
        return contigs, actual

    # Sort contigs by length (smallest first) for finer-grained dropping,
    # then shuffle within the order to add randomness
    indices = np.arange(len(contigs))
    lengths = np.array([len(c) for c in contigs])
    sorted_indices = indices[np.argsort(lengths)]  # smallest first
    rng.shuffle(sorted_indices)  # randomize order but with small contigs mixed in

    kept_indices = []
    kept_length = 0
    dropped_indices = []
    for idx in sorted_indices:
        c = contigs[idx]
        if kept_length + len(c) <= target_length:
            kept_indices.append(idx)
            kept_length += len(c)
        elif kept_length >= target_length * 0.95:
            # Close enough to target
            dropped_indices.append(idx)
        else:
            # Adding this contig would exceed target, but we're not close enough
            overshoot = (kept_length + len(c)) - target_length
            undershoot = target_length - kept_length
            if overshoot < undershoot * 2:
                kept_indices.append(idx)
                kept_length += len(c)
            else:
                dropped_indices.append(idx)
                continue

    # Collect any remaining indices as dropped
    kept_set = set(kept_indices)
    for idx in sorted_indices:
        if idx not in kept_set and idx not in set(dropped_indices):
            dropped_indices.append(idx)

    if len(kept_indices) == 0:
        # At minimum keep the smallest contig
        smallest_idx = int(np.argmin(lengths))
        kept_indices = [smallest_idx]
        kept_length = lengths[smallest_idx]

    # Safety net: enforce minimum completeness (protocol specifies 50-100%)
    # If actual completeness dropped below the floor, add back contigs
    # (smallest first from the dropped set) until we meet the minimum.
    min_length = int(min_completeness * genome_full_length)
    if kept_length < min_length and len(dropped_indices) > 0:
        # Sort dropped contigs by size (smallest first) for minimal overshoot
        dropped_lengths = np.array([lengths[i] for i in dropped_indices])
        add_back_order = np.argsort(dropped_lengths)  # smallest dropped first
        for oi in add_back_order:
            didx = dropped_indices[oi]
            kept_indices.append(didx)
            kept_length += lengths[didx]
            if kept_length >= min_length:
                break

    kept = [contigs[i] for i in kept_indices]
    actual_completeness = kept_length / genome_full_length if genome_full_length > 0 else 0.0
    return kept, actual_completeness


def simulate_fragmentation(
    sequence: str,
    target_completeness: float,
    quality_tier: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    apply_biases: bool = True,
) -> Dict[str, Any]:
    """
    Full fragmentation pipeline for a single genome.

    Parameters
    ----------
    sequence : str
        Full reference genome sequence (concatenated if multi-contig).
    target_completeness : float
        Target completeness as fraction (0.5 to 1.0).
    quality_tier : str or None
        Quality tier. If None, randomly selected.
    rng : np.random.Generator or None
        Random number generator. Created from seed if None.
    seed : int or None
        Random seed (used only if rng is None).
    apply_biases : bool
        Whether to apply coverage dropout, GC-biased loss, and repeat exclusion
        before completeness filtering.

    Returns
    -------
    dict with keys:
        'contigs': list of str - final contig sequences
        'completeness': float - actual completeness achieved
        'quality_tier': str - quality tier used
        'n_contigs': int - number of contigs
        'total_length': int - total bp in contigs
        'genome_full_length': int - original genome length
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    genome_full_length = len(sequence)

    if genome_full_length == 0:
        return {
            'contigs': [],
            'completeness': 0.0,
            'quality_tier': quality_tier or 'high',
            'n_contigs': 0,
            'total_length': 0,
            'genome_full_length': 0,
        }

    # Select quality tier if not specified
    if quality_tier is None:
        quality_tier = select_quality_tier(rng)

    # Step 1: Generate contig lengths
    contig_lengths = generate_contig_lengths(genome_full_length, quality_tier, rng)

    # Step 2: Fragment genome at random positions
    contigs = fragment_genome(sequence, contig_lengths, rng)

    # Step 3: Apply realistic biases (before completeness filtering)
    # Keep pre-bias contigs for safety net recovery if completeness drops too low.
    all_contigs_before_bias = list(contigs) if apply_biases else None
    if apply_biases:
        contigs = apply_coverage_dropout(contigs, rng)
        contigs = apply_gc_biased_loss(contigs, rng)
        contigs = apply_repeat_exclusion(contigs, rng)

    # Step 4: Apply completeness by dropping contigs
    contigs, actual_completeness = apply_completeness(
        contigs, target_completeness, genome_full_length, rng
    )

    total_length = sum(len(c) for c in contigs)

    # Step 5: Enforce minimum completeness floor (protocol: 50-100%)
    # If biases + completeness dropping resulted in < 50% completeness,
    # re-run apply_completeness on the full pre-bias contig set (which has
    # all the original fragmented contigs before any were dropped by biases).
    # This guarantees we can always reach at least 50% as long as the genome
    # was fragmented from a full reference.
    MIN_COMPLETENESS = 0.50
    min_length = int(MIN_COMPLETENESS * genome_full_length)

    if total_length < min_length and all_contigs_before_bias is not None:
        # Use apply_completeness on the full pre-bias set with the original target,
        # which has the min_completeness safety net built in.
        contigs, actual_completeness = apply_completeness(
            all_contigs_before_bias, target_completeness, genome_full_length, rng
        )
        total_length = sum(len(c) for c in contigs)

    return {
        'contigs': contigs,
        'completeness': actual_completeness,
        'quality_tier': quality_tier,
        'n_contigs': len(contigs),
        'total_length': total_length,
        'genome_full_length': genome_full_length,
    }


def load_original_contigs(fasta_path: str) -> List[str]:
    """
    Load the original contigs from a FASTA file without any fragmentation.

    Returns each contig/scaffold as a separate sequence string, preserving
    the natural contig structure of the reference genome. For "complete"
    sample type, this provides realistic contig structures (typically 1-100
    contigs as they exist in the reference FASTA).

    Parameters
    ----------
    fasta_path : str
        Path to FASTA file (may be gzipped).

    Returns
    -------
    list of str
        List of contig sequences (uppercase).
    """
    if fasta_path.endswith('.gz'):
        import gzip
        with gzip.open(fasta_path, 'rt') as f:
            content = f.read()
    else:
        with open(fasta_path, 'r') as f:
            content = f.read()

    contigs = []
    current_parts = []
    for line in content.split('\n'):
        if line and line[0] == '>':
            if current_parts:
                contigs.append(''.join(current_parts).upper())
                current_parts = []
        elif line:
            current_parts.append(line.strip())
    if current_parts:
        contigs.append(''.join(current_parts).upper())

    # Filter out empty contigs
    contigs = [c for c in contigs if len(c) > 0]
    return contigs


def read_fasta(fasta_path: str) -> str:
    """
    Read a FASTA file and return concatenated sequence (all contigs joined).

    Optimized: reads entire file at once and filters header lines.

    Parameters
    ----------
    fasta_path : str
        Path to FASTA file (may be gzipped).

    Returns
    -------
    str
        Concatenated uppercase sequence.
    """
    if fasta_path.endswith('.gz'):
        import gzip
        with gzip.open(fasta_path, 'rt') as f:
            content = f.read()
    else:
        with open(fasta_path, 'r') as f:
            content = f.read()

    # Fast path: join all non-header lines, strip whitespace
    parts = []
    for line in content.split('\n'):
        if line and line[0] != '>':
            parts.append(line.strip())
    return ''.join(parts).upper()
