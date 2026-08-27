"""
Assembly Statistics Module for MAGICC.

Computes all 26 assembly statistics features in a single pass:
- 11 contig length stats: total_length, contig_count, n50, n90, l50, l90,
  largest_contig, smallest_contig, mean_contig, median_contig, contig_length_std
- 4 base composition: gc_mean, gc_std, gc_iqr, gc_bimodality
- 4 distributional: gc_outlier_fraction, largest_contig_fraction,
  top10_concentration, n50_mean_ratio
- 1 k-mer (legacy): log10_total_kmer_count (placeholder, filled during k-mer counting)
- 6 k-mer summary features (new): total_kmer_sum, unique_kmer_count,
  duplicate_kmer_count, kmer_entropy, unique_kmer_ratio, duplicate_kmer_ratio

Uses Numba JIT compilation for performance-critical parts.
"""

import numpy as np
import numba as nb
from typing import List, Dict, Optional, Tuple

# Feature names in order
FEATURE_NAMES = [
    # 11 contig length stats
    'total_length',
    'contig_count',
    'n50',
    'n90',
    'l50',
    'l90',
    'largest_contig',
    'smallest_contig',
    'mean_contig',
    'median_contig',
    'contig_length_std',
    # 4 base composition
    'gc_mean',
    'gc_std',
    'gc_iqr',
    'gc_bimodality',
    # 4 distributional
    'gc_outlier_fraction',
    'largest_contig_fraction',
    'top10_concentration',
    'n50_mean_ratio',
    # 1 k-mer (legacy)
    'log10_total_kmer_count',
    # 6 new k-mer summary features
    'total_kmer_sum',         # sum of all 9,249 k-mer raw counts
    'unique_kmer_count',      # number of k-mers with count > 0
    'duplicate_kmer_count',   # number of k-mers with count > 1
    'kmer_entropy',           # Shannon entropy of k-mer count distribution
    'unique_kmer_ratio',      # unique_kmer_count / 9249
    'duplicate_kmer_ratio',   # duplicate_kmer_count / unique_kmer_count
]

N_FEATURES = len(FEATURE_NAMES)  # 26
assert N_FEATURES == 26

# Feature name -> index mapping
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


@nb.njit(cache=True)
def _compute_gc_from_bytes(seq_bytes: np.ndarray) -> float:
    """Compute GC content from a byte array of DNA sequence (Numba-accelerated)."""
    gc = 0
    total = 0
    for i in range(len(seq_bytes)):
        b = seq_bytes[i]
        # G=71, C=67, A=65, T=84, g=103, c=99, a=97, t=116
        if b == 71 or b == 67 or b == 103 or b == 99:
            gc += 1
            total += 1
        elif b == 65 or b == 84 or b == 97 or b == 116:
            total += 1
        # N and other chars are ignored
    if total == 0:
        return 0.5
    return gc / total


@nb.njit(cache=True)
def _compute_nx_lx(sorted_lengths: np.ndarray, total_length: int, fraction: float) -> Tuple:
    """
    Compute Nx and Lx metrics from sorted (descending) contig lengths.

    Parameters
    ----------
    sorted_lengths : np.ndarray
        Contig lengths sorted in descending order.
    total_length : int
        Total assembly length.
    fraction : float
        Fraction (e.g., 0.5 for N50, 0.9 for N90).

    Returns
    -------
    tuple of (nx, lx)
        nx = contig length at which fraction of assembly is covered
        lx = number of contigs needed to cover fraction
    """
    threshold = total_length * fraction
    running = 0
    for i in range(len(sorted_lengths)):
        running += sorted_lengths[i]
        if running >= threshold:
            return sorted_lengths[i], i + 1
    return sorted_lengths[-1], len(sorted_lengths)


@nb.njit(cache=True)
def _compute_bimodality(values: np.ndarray) -> float:
    """
    Compute bimodality coefficient: (skewness^2 + 1) / kurtosis.

    Uses excess kurtosis + 3 for the denominator.
    Returns 0 if kurtosis is zero or less.
    """
    n = len(values)
    if n < 4:
        return 0.0

    mean = 0.0
    for i in range(n):
        mean += values[i]
    mean /= n

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        d = values[i] - mean
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2
    m2 /= n
    m3 /= n
    m4 /= n

    if m2 < 1e-20:
        return 0.0

    # Skewness
    skewness = m3 / (m2 ** 1.5)

    # Kurtosis (excess kurtosis + 3 = regular kurtosis)
    kurtosis = m4 / (m2 ** 2)

    if kurtosis < 1e-20:
        return 0.0

    return (skewness * skewness + 1.0) / kurtosis


def compute_assembly_stats(
    contigs: List[str],
    log10_total_kmer_count: float = 0.0,
    kmer_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute all 26 assembly statistics features.

    Parameters
    ----------
    contigs : list of str
        List of contig sequences.
    log10_total_kmer_count : float
        Pre-computed log10 total k-mer count (filled during k-mer counting step).
    kmer_counts : np.ndarray or None
        Raw k-mer counts array of shape (n_kmer_features,). If provided,
        used to compute the 6 new k-mer summary features.

    Returns
    -------
    np.ndarray
        Array of 26 features in order defined by FEATURE_NAMES.
    """
    features = np.zeros(N_FEATURES, dtype=np.float64)

    n_contigs = len(contigs)
    if n_contigs == 0:
        return features

    # Compute contig lengths and GC content per contig
    lengths = np.empty(n_contigs, dtype=np.int64)
    gc_values = np.empty(n_contigs, dtype=np.float64)

    for i, contig in enumerate(contigs):
        lengths[i] = len(contig)
        if len(contig) > 0:
            seq_bytes = np.frombuffer(contig.encode('ascii'), dtype=np.uint8)
            gc_values[i] = _compute_gc_from_bytes(seq_bytes)
        else:
            gc_values[i] = 0.5

    # Sort lengths descending for N50/N90 computation
    sorted_lengths = np.sort(lengths)[::-1]
    total_length = int(lengths.sum())

    # === 11 Contig length stats ===
    features[0] = total_length                          # total_length
    features[1] = n_contigs                             # contig_count

    n50, l50 = _compute_nx_lx(sorted_lengths, total_length, 0.5)
    n90, l90 = _compute_nx_lx(sorted_lengths, total_length, 0.9)
    features[2] = n50                                   # n50
    features[3] = n90                                   # n90
    features[4] = l50                                   # l50
    features[5] = l90                                   # l90
    features[6] = sorted_lengths[0]                     # largest_contig
    features[7] = sorted_lengths[-1]                    # smallest_contig
    mean_length = total_length / n_contigs
    features[8] = mean_length                           # mean_contig
    features[9] = float(np.median(lengths))             # median_contig
    features[10] = float(np.std(lengths))               # contig_length_std

    # === 4 Base composition ===
    # Weight GC by contig length for overall mean
    length_weights = lengths.astype(np.float64) / total_length
    gc_mean = float(np.sum(gc_values * length_weights))
    features[11] = gc_mean                              # gc_mean

    # GC std across contigs (weighted)
    gc_var = float(np.sum(length_weights * (gc_values - gc_mean) ** 2))
    gc_std = np.sqrt(gc_var)
    features[12] = gc_std                               # gc_std

    # GC IQR
    if n_contigs >= 4:
        q75, q25 = np.percentile(gc_values, [75, 25])
        gc_iqr = q75 - q25
    else:
        gc_iqr = gc_std * 1.35  # Approximate IQR from std for small samples
    features[13] = gc_iqr                               # gc_iqr

    # GC bimodality: (skewness^2 + 1) / kurtosis
    features[14] = _compute_bimodality(gc_values)       # gc_bimodality

    # === 4 Distributional ===
    # GC outlier fraction: fraction of total length in contigs with |GC - mean| > 2*std
    if gc_std > 1e-10:
        outlier_mask = np.abs(gc_values - gc_mean) > 2 * gc_std
        gc_outlier_fraction = float(lengths[outlier_mask].sum()) / total_length
    else:
        gc_outlier_fraction = 0.0
    features[15] = gc_outlier_fraction                  # gc_outlier_fraction

    # Largest contig fraction
    features[16] = float(sorted_lengths[0]) / total_length  # largest_contig_fraction

    # Top 10% contig concentration
    n_top10 = max(1, int(np.ceil(n_contigs * 0.1)))
    top10_length = float(sorted_lengths[:n_top10].sum())
    features[17] = top10_length / total_length          # top10_concentration

    # N50/mean ratio
    if mean_length > 0:
        features[18] = n50 / mean_length                # n50_mean_ratio
    else:
        features[18] = 0.0

    # === 1 K-mer (legacy) ===
    features[19] = log10_total_kmer_count               # log10_total_kmer_count

    # === 6 New k-mer summary features ===
    if kmer_counts is not None:
        total_kmer_sum = float(kmer_counts.sum())
        features[20] = total_kmer_sum                    # total_kmer_sum

        unique_count = int(np.count_nonzero(kmer_counts))
        features[21] = float(unique_count)               # unique_kmer_count

        duplicate_count = int(np.sum(kmer_counts > 1))
        features[22] = float(duplicate_count)            # duplicate_kmer_count

        # Shannon entropy of k-mer count distribution
        if total_kmer_sum > 0:
            probs = kmer_counts.astype(np.float64) / total_kmer_sum
            # Only compute for non-zero entries to avoid log(0)
            nonzero_mask = probs > 0
            entropy = -float(np.sum(probs[nonzero_mask] * np.log2(probs[nonzero_mask])))
        else:
            entropy = 0.0
        features[23] = entropy                           # kmer_entropy

        n_total_kmers = len(kmer_counts)
        features[24] = unique_count / n_total_kmers if n_total_kmers > 0 else 0.0  # unique_kmer_ratio

        features[25] = duplicate_count / unique_count if unique_count > 0 else 0.0  # duplicate_kmer_ratio
    # else: features[20:26] remain 0.0

    return features


def compute_assembly_stats_batch(
    batch_contigs: List[List[str]],
    log10_kmer_counts: Optional[np.ndarray] = None,
    batch_kmer_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute assembly statistics for a batch of genomes.

    Parameters
    ----------
    batch_contigs : list of list of str
        List of contig lists, one per genome.
    log10_kmer_counts : np.ndarray or None
        Array of log10 total k-mer counts, one per genome.
    batch_kmer_counts : np.ndarray or None
        Raw k-mer counts, shape (n_genomes, n_kmer_features). If provided,
        used to compute the 6 new k-mer summary features.

    Returns
    -------
    np.ndarray
        Array of shape (n_genomes, 26).
    """
    n = len(batch_contigs)
    result = np.zeros((n, N_FEATURES), dtype=np.float64)

    for i in range(n):
        kmer_count = float(log10_kmer_counts[i]) if log10_kmer_counts is not None else 0.0
        kc = batch_kmer_counts[i] if batch_kmer_counts is not None else None
        result[i] = compute_assembly_stats(batch_contigs[i], kmer_count, kc)

    return result


def format_stats(features: np.ndarray) -> Dict[str, float]:
    """Convert feature array to named dictionary."""
    return {name: float(features[i]) for i, name in enumerate(FEATURE_NAMES)}
