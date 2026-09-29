"""
Assembly Statistics Module for MAGICC.

Computes 7 k-mer summary features:
- 1 k-mer (legacy): log10_total_kmer_count (placeholder, filled during k-mer counting)
- 6 k-mer summary features: total_kmer_sum, unique_kmer_count,
  duplicate_kmer_count, kmer_entropy, unique_kmer_ratio, duplicate_kmer_ratio

Assembly statistics (contig lengths, GC content, distributional stats) have been
removed to prevent overfitting to specific fragmentation patterns in training data.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

# Feature names in order (7 k-mer summary features only)
FEATURE_NAMES = [
    # 1 k-mer (legacy)
    'log10_total_kmer_count',
    # 6 k-mer summary features
    'total_kmer_sum',         # sum of all 9,249 k-mer raw counts
    'unique_kmer_count',      # number of k-mers with count > 0
    'duplicate_kmer_count',   # number of k-mers with count > 1
    'kmer_entropy',           # Shannon entropy of k-mer count distribution
    'unique_kmer_ratio',      # unique_kmer_count / 9249
    'duplicate_kmer_ratio',   # duplicate_kmer_count / unique_kmer_count
]

N_FEATURES = len(FEATURE_NAMES)  # 7
assert N_FEATURES == 7

# Feature name -> index mapping
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def compute_assembly_stats(
    log10_total_kmer_count: float = 0.0,
    kmer_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute all 7 k-mer summary features.

    Parameters
    ----------
    log10_total_kmer_count : float
        Pre-computed log10 total k-mer count (filled during k-mer counting step).
    kmer_counts : np.ndarray or None
        Raw k-mer counts array of shape (n_kmer_features,). If provided,
        used to compute the 6 k-mer summary features.

    Returns
    -------
    np.ndarray
        Array of 7 features in order defined by FEATURE_NAMES.
    """
    features = np.zeros(N_FEATURES, dtype=np.float64)

    # log10_total_kmer_count
    features[0] = log10_total_kmer_count

    # 6 k-mer summary features
    if kmer_counts is not None:
        total_kmer_sum = float(kmer_counts.sum())
        features[1] = total_kmer_sum                     # total_kmer_sum

        unique_count = int(np.count_nonzero(kmer_counts))
        features[2] = float(unique_count)                # unique_kmer_count

        duplicate_count = int(np.sum(kmer_counts > 1))
        features[3] = float(duplicate_count)             # duplicate_kmer_count

        # Shannon entropy of k-mer count distribution
        if total_kmer_sum > 0:
            probs = kmer_counts.astype(np.float64) / total_kmer_sum
            # Only compute for non-zero entries to avoid log(0)
            nonzero_mask = probs > 0
            entropy = -float(np.sum(probs[nonzero_mask] * np.log2(probs[nonzero_mask])))
        else:
            entropy = 0.0
        features[4] = entropy                            # kmer_entropy

        n_total_kmers = len(kmer_counts)
        features[5] = unique_count / n_total_kmers if n_total_kmers > 0 else 0.0  # unique_kmer_ratio

        features[6] = duplicate_count / unique_count if unique_count > 0 else 0.0  # duplicate_kmer_ratio
    # else: features[1:7] remain 0.0

    return features


def compute_assembly_stats_batch(
    log10_kmer_counts: Optional[np.ndarray] = None,
    batch_kmer_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute k-mer summary statistics for a batch of genomes.

    Parameters
    ----------
    log10_kmer_counts : np.ndarray or None
        Array of log10 total k-mer counts, one per genome.
    batch_kmer_counts : np.ndarray or None
        Raw k-mer counts, shape (n_genomes, n_kmer_features). If provided,
        used to compute the 6 k-mer summary features.

    Returns
    -------
    np.ndarray
        Array of shape (n_genomes, 7).
    """
    if batch_kmer_counts is not None:
        n = batch_kmer_counts.shape[0]
    elif log10_kmer_counts is not None:
        n = len(log10_kmer_counts)
    else:
        raise ValueError("Must provide at least log10_kmer_counts or batch_kmer_counts")

    result = np.zeros((n, N_FEATURES), dtype=np.float64)

    for i in range(n):
        kmer_count = float(log10_kmer_counts[i]) if log10_kmer_counts is not None else 0.0
        kc = batch_kmer_counts[i] if batch_kmer_counts is not None else None
        result[i] = compute_assembly_stats(kmer_count, kc)

    return result


def format_stats(features: np.ndarray) -> Dict[str, float]:
    """Convert feature array to named dictionary."""
    return {name: float(features[i]) for i, name in enumerate(FEATURE_NAMES)}
