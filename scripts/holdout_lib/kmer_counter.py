"""
K-mer Counter Module for MAGICC.

Counts occurrences of pre-selected canonical 9-mers directly on genome sequences.
Uses Numba JIT compilation for maximum performance.

The selected k-mers are loaded from data/kmer_selection/selected_kmers.txt
(9,249 canonical 9-mers, one per line).
"""

import numpy as np
import numba as nb
from typing import List, Optional, Dict
from pathlib import Path


# K-mer encoding: A=0, C=1, G=2, T=3
# Bases that are not A/C/G/T are treated as invalid (skip that k-mer window)
_BASE_MAP = np.full(256, 255, dtype=np.uint8)
_BASE_MAP[ord('A')] = 0
_BASE_MAP[ord('C')] = 1
_BASE_MAP[ord('G')] = 2
_BASE_MAP[ord('T')] = 3
_BASE_MAP[ord('a')] = 0
_BASE_MAP[ord('c')] = 1
_BASE_MAP[ord('g')] = 2
_BASE_MAP[ord('t')] = 3

K = 9  # K-mer length
N_POSSIBLE = 4 ** K  # 262144 possible 9-mers


def encode_kmer(kmer: str) -> int:
    """Encode a k-mer string to integer."""
    val = 0
    for c in kmer.upper():
        base = int(_BASE_MAP[ord(c)])
        if base == 255:
            return -1
        val = val * 4 + base
    return val


def reverse_complement_code(code: int, k: int = K) -> int:
    """Get the reverse complement encoding of a k-mer code."""
    code = int(code)
    rc = 0
    for _ in range(k):
        rc = rc * 4 + (3 - (code & 3))
        code >>= 2
    return rc


def canonical_code(code: int, k: int = K) -> int:
    """Get the canonical (minimum of forward and reverse complement) k-mer code."""
    rc = reverse_complement_code(code, k)
    return min(code, rc)


def load_selected_kmers(path: str) -> np.ndarray:
    """
    Load selected k-mers and return their canonical integer codes.

    Parameters
    ----------
    path : str
        Path to selected_kmers.txt (one k-mer per line).

    Returns
    -------
    np.ndarray
        Sorted array of canonical k-mer codes (int32).
    """
    codes = []
    with open(path) as f:
        for line in f:
            kmer = line.strip()
            if kmer:
                code = encode_kmer(kmer)
                assert code >= 0, f"Invalid k-mer: {kmer}"
                codes.append(code)

    codes = np.array(sorted(codes), dtype=np.int32)
    return codes


def build_kmer_index(selected_codes: np.ndarray) -> np.ndarray:
    """
    Build a lookup table mapping canonical k-mer codes to feature indices.

    Parameters
    ----------
    selected_codes : np.ndarray
        Sorted array of selected canonical k-mer codes.

    Returns
    -------
    np.ndarray
        Array of size N_POSSIBLE where entry[code] = feature_index or -1.
    """
    index = np.full(N_POSSIBLE, -1, dtype=np.int32)
    for i, code in enumerate(selected_codes):
        index[code] = i
    return index


@nb.njit(cache=True)
def _count_kmers_single(
    seq_bytes: np.ndarray,
    kmer_index: np.ndarray,
    n_features: int,
    k: int = 9,
) -> np.ndarray:
    """
    Count selected k-mers in a single sequence (Numba-accelerated).

    Parameters
    ----------
    seq_bytes : np.ndarray
        Sequence as uint8 byte array.
    kmer_index : np.ndarray
        Lookup table: code -> feature_index (or -1).
    n_features : int
        Number of selected k-mers.
    k : int
        K-mer length.

    Returns
    -------
    np.ndarray
        Raw counts for each selected k-mer.
    """
    counts = np.zeros(n_features, dtype=np.int64)
    seq_len = len(seq_bytes)

    if seq_len < k:
        return counts

    # Base encoding: A=0, C=1, G=2, T=3
    base_map = np.full(256, np.uint8(255), dtype=np.uint8)
    base_map[65] = 0   # A
    base_map[67] = 1   # C
    base_map[71] = 2   # G
    base_map[84] = 3   # T
    base_map[97] = 0   # a
    base_map[99] = 1   # c
    base_map[103] = 2  # g
    base_map[116] = 3  # t

    mask = (1 << (2 * k)) - 1  # Mask for k bases
    rc_shift = 2 * (k - 1)

    fwd = 0   # Forward hash
    rev = 0   # Reverse complement hash
    valid = 0  # Number of valid bases in current window

    for i in range(seq_len):
        base = base_map[seq_bytes[i]]
        if base == 255:
            # Invalid base, reset window
            valid = 0
            fwd = 0
            rev = 0
            continue

        # Update forward hash (shift left and add new base)
        fwd = ((fwd << 2) | base) & mask

        # Update reverse complement hash (shift right and add complement at top)
        rc_base = 3 - base
        rev = (rev >> 2) | (rc_base << rc_shift)

        valid += 1

        if valid >= k:
            # Canonical = min(forward, reverse complement)
            canonical = fwd if fwd <= rev else rev

            # Look up in index
            idx = kmer_index[canonical]
            if idx >= 0:
                counts[idx] += 1

    return counts


class KmerCounter:
    """
    Fast k-mer counter for selected canonical 9-mers.

    Parameters
    ----------
    selected_kmers_path : str
        Path to file with selected k-mers (one per line).
    """

    def __init__(self, selected_kmers_path: str):
        self.selected_codes = load_selected_kmers(selected_kmers_path)
        self.n_features = len(self.selected_codes)
        self.kmer_index = build_kmer_index(self.selected_codes)

        # Load k-mer strings for reference
        self.kmer_strings = []
        with open(selected_kmers_path) as f:
            for line in f:
                kmer = line.strip()
                if kmer:
                    self.kmer_strings.append(kmer)

    def count_sequence(self, sequence: str) -> np.ndarray:
        """
        Count selected k-mers in a single DNA sequence.

        Parameters
        ----------
        sequence : str
            DNA sequence.

        Returns
        -------
        np.ndarray
            Raw k-mer counts, shape (n_features,).
        """
        if len(sequence) == 0:
            return np.zeros(self.n_features, dtype=np.int64)

        seq_bytes = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        return _count_kmers_single(seq_bytes, self.kmer_index, self.n_features, K)

    def count_contigs(self, contigs: List[str]) -> np.ndarray:
        """
        Count selected k-mers across all contigs of a genome.

        Parameters
        ----------
        contigs : list of str
            List of contig sequences.

        Returns
        -------
        np.ndarray
            Raw k-mer counts, shape (n_features,).
        """
        total_counts = np.zeros(self.n_features, dtype=np.int64)
        for contig in contigs:
            if len(contig) >= K:
                seq_bytes = np.frombuffer(contig.encode('ascii'), dtype=np.uint8)
                total_counts += _count_kmers_single(
                    seq_bytes, self.kmer_index, self.n_features, K
                )
        return total_counts

    def count_contigs_batch(
        self,
        batch_contigs: List[List[str]],
    ) -> np.ndarray:
        """
        Count k-mers for a batch of genomes.

        Parameters
        ----------
        batch_contigs : list of list of str
            List of contig lists, one per genome.

        Returns
        -------
        np.ndarray
            K-mer counts, shape (n_genomes, n_features).
        """
        n = len(batch_contigs)
        result = np.zeros((n, self.n_features), dtype=np.int64)
        for i in range(n):
            result[i] = self.count_contigs(batch_contigs[i])
        return result

    def total_kmer_count(self, counts: np.ndarray) -> float:
        """Compute log10 of total k-mer count."""
        total = counts.sum()
        if total <= 0:
            return 0.0
        return np.log10(total)
