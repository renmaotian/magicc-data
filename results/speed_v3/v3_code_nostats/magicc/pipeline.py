"""
Integrated Pipeline Module for MAGICC.

Combines all modules into a single pipeline that processes a genome end-to-end:
1. Fragmentation (simulate MAG assembly artifacts)
2. Contamination (mix genomes)
3. K-mer counting (count selected canonical 9-mers)
4. Assembly statistics (26 features: 20 original + 6 new k-mer summary)
5. Feature normalization
6. Storage (write to HDF5)
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any, Tuple

from magicc.fragmentation import simulate_fragmentation, read_fasta, load_original_contigs
from magicc.contamination import generate_contaminated_sample, generate_pure_sample
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats, N_FEATURES as N_ASSEMBLY_FEATURES
from magicc.normalization import FeatureNormalizer
from magicc.storage import FeatureStore, METADATA_DTYPE


class MAGICCPipeline:
    """
    End-to-end pipeline for synthetic genome generation and feature extraction.

    Parameters
    ----------
    selected_kmers_path : str
        Path to selected k-mers file.
    normalizer : FeatureNormalizer or None
        Pre-initialized normalizer. If None, normalization statistics will
        be collected but features won't be normalized.
    """

    def __init__(
        self,
        selected_kmers_path: str,
        normalizer: Optional[FeatureNormalizer] = None,
    ):
        self.kmer_counter = KmerCounter(selected_kmers_path)
        self.normalizer = normalizer
        self.n_kmer_features = self.kmer_counter.n_features
        self.n_assembly_features = N_ASSEMBLY_FEATURES
        self.n_total_features = self.n_kmer_features + self.n_assembly_features

    def extract_features(
        self,
        contigs: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract k-mer and assembly features from a set of contigs.

        This is the core feature extraction function that should be fast
        (target: <50ms per genome).

        Parameters
        ----------
        contigs : list of str
            Contig sequences.

        Returns
        -------
        tuple of (kmer_counts, assembly_features, log10_total_kmer)
            kmer_counts: np.ndarray of shape (n_kmer_features,) - raw counts
            assembly_features: np.ndarray of shape (26,)
            log10_total_kmer: float
        """
        # K-mer counting
        kmer_counts = self.kmer_counter.count_contigs(contigs)
        log10_total = self.kmer_counter.total_kmer_count(kmer_counts)

        # Assembly statistics (pass log10 total kmer count AND raw kmer counts
        # for the 6 new k-mer summary features)
        assembly_features = compute_assembly_stats(contigs, log10_total, kmer_counts)

        return kmer_counts, assembly_features, log10_total

    def process_single_genome(
        self,
        dominant_sequence: str,
        contaminant_sequences: Optional[List[str]] = None,
        target_completeness: float = 1.0,
        target_contamination: float = 0.0,
        quality_tier: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a single genome through the full pipeline.

        Steps:
        1. Fragment and/or contaminate
        2. Extract features
        3. Return features + metadata

        Parameters
        ----------
        dominant_sequence : str
            Full reference genome sequence.
        contaminant_sequences : list of str or None
            Contaminant genome sequences (for contaminated samples).
        target_completeness : float
            Target completeness (0.5 to 1.0).
        target_contamination : float
            Target contamination rate (0-100%).
        quality_tier : str or None
            Fragmentation quality tier.
        rng : np.random.Generator or None
            Random number generator.
        seed : int or None
            Random seed.

        Returns
        -------
        dict with keys:
            'kmer_counts': np.ndarray (n_kmer_features,)
            'assembly_features': np.ndarray (26,)
            'completeness': float
            'contamination': float
            'n_contigs': int
            'total_length': int
            'dominant_full_length': int
            'quality_tier': str
            'log10_total_kmer': float
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        # Step 1: Fragmentation and contamination
        if contaminant_sequences and target_contamination > 0:
            sample = generate_contaminated_sample(
                dominant_sequence=dominant_sequence,
                contaminant_sequences=contaminant_sequences,
                target_completeness=target_completeness,
                target_contamination=target_contamination,
                rng=rng,
                dominant_quality_tier=quality_tier,
            )
            contigs = sample['contigs']
            actual_completeness = sample['completeness']
            actual_contamination = sample['contamination']
            used_quality_tier = sample['dominant_quality_tier']
        else:
            sample = generate_pure_sample(
                dominant_sequence=dominant_sequence,
                target_completeness=target_completeness,
                rng=rng,
                quality_tier=quality_tier,
            )
            contigs = sample['contigs']
            actual_completeness = sample['completeness']
            actual_contamination = 0.0
            used_quality_tier = sample['dominant_quality_tier']

        # Step 2: Feature extraction
        kmer_counts, assembly_features, log10_total = self.extract_features(contigs)

        return {
            'kmer_counts': kmer_counts,
            'assembly_features': assembly_features,
            'completeness': actual_completeness * 100,  # Convert to percentage
            'contamination': actual_contamination,
            'n_contigs': len(contigs),
            'total_length': sum(len(c) for c in contigs),
            'dominant_full_length': len(dominant_sequence),
            'quality_tier': used_quality_tier,
            'log10_total_kmer': log10_total,
        }

    def process_batch(
        self,
        dominant_sequences: List[str],
        contaminant_sequences_list: Optional[List[Optional[List[str]]]] = None,
        target_completeness_list: Optional[List[float]] = None,
        target_contamination_list: Optional[List[float]] = None,
        quality_tier_list: Optional[List[Optional[str]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Process a batch of genomes.

        Parameters
        ----------
        dominant_sequences : list of str
            Dominant genome sequences.
        contaminant_sequences_list : list of (list of str or None) or None
            Contaminant sequences for each sample.
        target_completeness_list : list of float or None
            Target completeness values.
        target_contamination_list : list of float or None
            Target contamination rates.
        quality_tier_list : list of str or None
            Quality tiers.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        dict with keys:
            'kmer_counts': np.ndarray (batch_size, n_kmer_features)
            'assembly_features': np.ndarray (batch_size, 26)
            'labels': np.ndarray (batch_size, 2)
            'metadata': np.ndarray structured (batch_size,)
        """
        if rng is None:
            rng = np.random.default_rng()

        batch_size = len(dominant_sequences)

        # Default values
        if contaminant_sequences_list is None:
            contaminant_sequences_list = [None] * batch_size
        if target_completeness_list is None:
            target_completeness_list = [1.0] * batch_size
        if target_contamination_list is None:
            target_contamination_list = [0.0] * batch_size
        if quality_tier_list is None:
            quality_tier_list = [None] * batch_size

        # Allocate output arrays
        kmer_counts = np.zeros((batch_size, self.n_kmer_features), dtype=np.int64)
        assembly_features = np.zeros((batch_size, self.n_assembly_features), dtype=np.float64)
        labels = np.zeros((batch_size, 2), dtype=np.float32)
        metadata = np.zeros(batch_size, dtype=METADATA_DTYPE)

        for i in range(batch_size):
            result = self.process_single_genome(
                dominant_sequence=dominant_sequences[i],
                contaminant_sequences=contaminant_sequences_list[i],
                target_completeness=target_completeness_list[i],
                target_contamination=target_contamination_list[i],
                quality_tier=quality_tier_list[i],
                rng=rng,
            )

            kmer_counts[i] = result['kmer_counts']
            assembly_features[i] = result['assembly_features']
            labels[i, 0] = result['completeness']
            labels[i, 1] = result['contamination']
            metadata[i]['completeness'] = result['completeness']
            metadata[i]['contamination'] = result['contamination']
            metadata[i]['quality_tier'] = result['quality_tier'].encode('utf-8')
            metadata[i]['genome_full_length'] = result['dominant_full_length']

        return {
            'kmer_counts': kmer_counts,
            'assembly_features': assembly_features,
            'labels': labels,
            'metadata': metadata,
        }
