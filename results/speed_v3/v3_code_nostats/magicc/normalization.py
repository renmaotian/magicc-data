"""
Feature Normalization Pipeline for MAGICC.

Implements streaming/incremental normalization with running statistics:
- K-mer features: log(count + 1) -> Z-score standardization
- Assembly statistics:
  - Log10 for length-based features (total_length, n50, n90, largest_contig,
    smallest_contig, mean_contig, median_contig, contig_length_std,
    total_kmer_sum)
  - Min-max scaling for percentage-based features (gc_mean, gc_std, gc_iqr,
    gc_bimodality, gc_outlier_fraction, largest_contig_fraction,
    top10_concentration, n50_mean_ratio, kmer_entropy,
    unique_kmer_ratio, duplicate_kmer_ratio)
  - Robust scaling (median + IQR) for count features (contig_count, l50, l90,
    unique_kmer_count, duplicate_kmer_count)
  - log10_total_kmer_count is already log-transformed (pass through)
"""

import numpy as np
import json
from typing import Optional, Dict, Any
from pathlib import Path

from magicc.assembly_stats import FEATURE_NAMES, FEATURE_INDEX, N_FEATURES

# Classify assembly features by normalization type
LOG10_FEATURES = [
    'total_length', 'n50', 'n90', 'largest_contig', 'smallest_contig',
    'mean_contig', 'median_contig', 'contig_length_std',
    'total_kmer_sum',
]
MINMAX_FEATURES = [
    'gc_mean', 'gc_std', 'gc_iqr', 'gc_bimodality',
    'gc_outlier_fraction', 'largest_contig_fraction',
    'top10_concentration', 'n50_mean_ratio',
    'kmer_entropy', 'unique_kmer_ratio', 'duplicate_kmer_ratio',
]
ROBUST_FEATURES = [
    'contig_count', 'l50', 'l90',
    'unique_kmer_count', 'duplicate_kmer_count',
]
PASSTHROUGH_FEATURES = ['log10_total_kmer_count']

# Indices
LOG10_INDICES = [FEATURE_INDEX[f] for f in LOG10_FEATURES]
MINMAX_INDICES = [FEATURE_INDEX[f] for f in MINMAX_FEATURES]
ROBUST_INDICES = [FEATURE_INDEX[f] for f in ROBUST_FEATURES]
PASSTHROUGH_INDICES = [FEATURE_INDEX[f] for f in PASSTHROUGH_FEATURES]


class RunningStats:
    """
    Streaming statistics tracker using Welford's algorithm and reservoir sampling.

    Tracks: mean, variance, min, max, count, and approximate median/quartiles
    via a reservoir sample.
    """

    def __init__(self, n_features: int, reservoir_size: int = 10000):
        self.n_features = n_features
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)
        self.min_vals = np.full(n_features, np.inf, dtype=np.float64)
        self.max_vals = np.full(n_features, -np.inf, dtype=np.float64)
        self.reservoir_size = reservoir_size
        self.reservoir = np.zeros((reservoir_size, n_features), dtype=np.float64)
        self.reservoir_count = 0

    def update_batch(self, data: np.ndarray):
        """
        Update running statistics with a batch of data.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (batch_size, n_features).
        """
        batch_size = data.shape[0]
        if batch_size == 0:
            return

        # Update min/max
        batch_min = np.nanmin(data, axis=0)
        batch_max = np.nanmax(data, axis=0)
        self.min_vals = np.minimum(self.min_vals, batch_min)
        self.max_vals = np.maximum(self.max_vals, batch_max)

        # Welford's online algorithm for mean and variance (batched)
        for i in range(batch_size):
            self.count += 1
            delta = data[i] - self.mean
            self.mean += delta / self.count
            delta2 = data[i] - self.mean
            self.m2 += delta * delta2

        # Reservoir sampling for median/quartile approximation
        for i in range(batch_size):
            if self.reservoir_count < self.reservoir_size:
                self.reservoir[self.reservoir_count] = data[i]
                self.reservoir_count += 1
            else:
                # Replace with probability reservoir_size / total_count
                j = np.random.randint(0, self.count)
                if j < self.reservoir_size:
                    self.reservoir[j] = data[i]

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros(self.n_features)
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    @property
    def median(self) -> np.ndarray:
        if self.reservoir_count == 0:
            return np.zeros(self.n_features)
        return np.median(self.reservoir[:self.reservoir_count], axis=0)

    @property
    def q25(self) -> np.ndarray:
        if self.reservoir_count == 0:
            return np.zeros(self.n_features)
        return np.percentile(self.reservoir[:self.reservoir_count], 25, axis=0)

    @property
    def q75(self) -> np.ndarray:
        if self.reservoir_count == 0:
            return np.zeros(self.n_features)
        return np.percentile(self.reservoir[:self.reservoir_count], 75, axis=0)

    @property
    def iqr(self) -> np.ndarray:
        return self.q75 - self.q25

    def to_dict(self) -> Dict[str, Any]:
        return {
            'count': int(self.count),
            'mean': self.mean.tolist(),
            'variance': self.variance.tolist(),
            'std': self.std.tolist(),
            'min': self.min_vals.tolist(),
            'max': self.max_vals.tolist(),
            'median': self.median.tolist(),
            'q25': self.q25.tolist(),
            'q75': self.q75.tolist(),
            'iqr': self.iqr.tolist(),
        }


class FeatureNormalizer:
    """
    Feature normalizer for MAGICC with streaming statistics.

    Handles two feature types:
    1. K-mer features (n_kmer_features): log(count+1) -> Z-score
    2. Assembly statistics (20 features): mixed normalization per feature type

    Parameters
    ----------
    n_kmer_features : int
        Number of k-mer features.
    reservoir_size : int
        Size of reservoir for median/quartile approximation.
    """

    def __init__(self, n_kmer_features: int = 9249, reservoir_size: int = 50000):
        self.n_kmer_features = n_kmer_features
        self.n_assembly_features = N_FEATURES  # 20

        # Running stats for k-mer features (after log transform)
        self.kmer_stats = RunningStats(n_kmer_features, reservoir_size)

        # Running stats for assembly features (after initial transforms)
        self.assembly_stats = RunningStats(N_FEATURES, reservoir_size)

        # Finalized parameters (set after all batches processed)
        self.finalized = False
        self.kmer_mean = None
        self.kmer_std = None
        self.assembly_log10_offset = 1.0  # Add 1 before log10 to handle zeros
        self.assembly_minmax_min = None
        self.assembly_minmax_range = None
        self.assembly_robust_median = None
        self.assembly_robust_iqr = None

    def update_kmer_batch(self, kmer_counts: np.ndarray):
        """
        Update k-mer normalization statistics with a batch.

        Parameters
        ----------
        kmer_counts : np.ndarray
            Raw k-mer counts, shape (batch_size, n_kmer_features).
        """
        # Apply log(count + 1) transform
        log_counts = np.log1p(kmer_counts.astype(np.float64))
        self.kmer_stats.update_batch(log_counts)

    def update_assembly_batch(self, assembly_features: np.ndarray):
        """
        Update assembly normalization statistics with a batch.

        Applies initial transforms (log10 for length features) before tracking stats.

        Parameters
        ----------
        assembly_features : np.ndarray
            Raw assembly features, shape (batch_size, 20).
        """
        # Apply log10 transform to length-based features
        transformed = assembly_features.copy()
        for idx in LOG10_INDICES:
            transformed[:, idx] = np.log10(transformed[:, idx] + self.assembly_log10_offset)

        self.assembly_stats.update_batch(transformed)

    def finalize(self):
        """
        Finalize normalization parameters from accumulated statistics.

        Must be called after all batches have been processed and before
        applying normalization.
        """
        # K-mer: Z-score parameters
        self.kmer_mean = self.kmer_stats.mean.copy()
        self.kmer_std = self.kmer_stats.std.copy()
        # Avoid division by zero
        self.kmer_std[self.kmer_std < 1e-10] = 1.0

        # Assembly: min-max parameters for percentage features
        self.assembly_minmax_min = np.zeros(N_FEATURES)
        self.assembly_minmax_range = np.ones(N_FEATURES)
        for idx in MINMAX_INDICES:
            self.assembly_minmax_min[idx] = self.assembly_stats.min_vals[idx]
            range_val = self.assembly_stats.max_vals[idx] - self.assembly_stats.min_vals[idx]
            self.assembly_minmax_range[idx] = range_val if range_val > 1e-10 else 1.0

        # Assembly: robust scaling parameters for count features
        self.assembly_robust_median = self.assembly_stats.median.copy()
        self.assembly_robust_iqr = self.assembly_stats.iqr.copy()
        # Avoid division by zero
        self.assembly_robust_iqr[self.assembly_robust_iqr < 1e-10] = 1.0

        self.finalized = True

    def normalize_kmer(self, kmer_counts: np.ndarray) -> np.ndarray:
        """
        Normalize k-mer features: log(count+1) -> Z-score.

        Parameters
        ----------
        kmer_counts : np.ndarray
            Raw k-mer counts, shape (batch_size, n_kmer_features) or (n_kmer_features,).

        Returns
        -------
        np.ndarray
            Normalized features, same shape as input.
        """
        assert self.finalized, "Must call finalize() before normalizing"
        log_counts = np.log1p(kmer_counts.astype(np.float64))
        return (log_counts - self.kmer_mean) / self.kmer_std

    def normalize_assembly(self, assembly_features: np.ndarray) -> np.ndarray:
        """
        Normalize assembly features with mixed normalization.

        Parameters
        ----------
        assembly_features : np.ndarray
            Raw assembly features, shape (batch_size, 20) or (20,).

        Returns
        -------
        np.ndarray
            Normalized features, same shape as input.
        """
        assert self.finalized, "Must call finalize() before normalizing"

        is_1d = assembly_features.ndim == 1
        if is_1d:
            features = assembly_features.reshape(1, -1).copy()
        else:
            features = assembly_features.copy()

        result = np.zeros_like(features, dtype=np.float64)

        # Log10 for length-based features
        for idx in LOG10_INDICES:
            result[:, idx] = np.log10(features[:, idx] + self.assembly_log10_offset)

        # Min-max scaling for percentage-based features
        for idx in MINMAX_INDICES:
            result[:, idx] = (features[:, idx] - self.assembly_minmax_min[idx]) / \
                             self.assembly_minmax_range[idx]

        # Robust scaling for count features
        for idx in ROBUST_INDICES:
            result[:, idx] = (features[:, idx] - self.assembly_robust_median[idx]) / \
                             self.assembly_robust_iqr[idx]

        # Passthrough for log10_total_kmer_count (already log-transformed)
        for idx in PASSTHROUGH_INDICES:
            result[:, idx] = features[:, idx]

        if is_1d:
            return result[0]
        return result

    def normalize_all(
        self,
        kmer_counts: np.ndarray,
        assembly_features: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize and concatenate k-mer and assembly features.

        Parameters
        ----------
        kmer_counts : np.ndarray
            Raw k-mer counts, shape (batch_size, n_kmer_features).
        assembly_features : np.ndarray
            Raw assembly features, shape (batch_size, 20).

        Returns
        -------
        np.ndarray
            Concatenated normalized features, shape (batch_size, n_kmer + 20).
        """
        kmer_norm = self.normalize_kmer(kmer_counts)
        asm_norm = self.normalize_assembly(assembly_features)
        return np.concatenate([kmer_norm, asm_norm], axis=-1)

    def save(self, path: str):
        """Save normalization parameters to JSON file."""
        params = {
            'n_kmer_features': self.n_kmer_features,
            'n_assembly_features': self.n_assembly_features,
            'finalized': self.finalized,
            'kmer_stats': self.kmer_stats.to_dict(),
            'assembly_stats': self.assembly_stats.to_dict(),
        }
        if self.finalized:
            params['kmer_mean'] = self.kmer_mean.tolist()
            params['kmer_std'] = self.kmer_std.tolist()
            params['assembly_log10_offset'] = self.assembly_log10_offset
            params['assembly_minmax_min'] = self.assembly_minmax_min.tolist()
            params['assembly_minmax_range'] = self.assembly_minmax_range.tolist()
            params['assembly_robust_median'] = self.assembly_robust_median.tolist()
            params['assembly_robust_iqr'] = self.assembly_robust_iqr.tolist()

        with open(path, 'w') as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FeatureNormalizer':
        """Load normalization parameters from JSON file."""
        with open(path) as f:
            params = json.load(f)

        normalizer = cls(
            n_kmer_features=params['n_kmer_features'],
        )

        if params.get('finalized', False):
            normalizer.kmer_mean = np.array(params['kmer_mean'])
            normalizer.kmer_std = np.array(params['kmer_std'])
            normalizer.assembly_log10_offset = params['assembly_log10_offset']
            normalizer.assembly_minmax_min = np.array(params['assembly_minmax_min'])
            normalizer.assembly_minmax_range = np.array(params['assembly_minmax_range'])
            normalizer.assembly_robust_median = np.array(params['assembly_robust_median'])
            normalizer.assembly_robust_iqr = np.array(params['assembly_robust_iqr'])
            normalizer.finalized = True

        # Restore running stats
        kmer_dict = params.get('kmer_stats', {})
        normalizer.kmer_stats.count = kmer_dict.get('count', 0)
        if 'mean' in kmer_dict:
            normalizer.kmer_stats.mean = np.array(kmer_dict['mean'])

        asm_dict = params.get('assembly_stats', {})
        normalizer.assembly_stats.count = asm_dict.get('count', 0)
        if 'mean' in asm_dict:
            normalizer.assembly_stats.mean = np.array(asm_dict['mean'])

        return normalizer
