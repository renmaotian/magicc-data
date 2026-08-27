"""
HDF5 Feature Storage Module for MAGICC.

Manages storage for 1,000,000 samples x 9,275 features (9,249 k-mer + 26 assembly).
Uses HDF5 for storage with optimized chunk sizes for batch read/write.

Decision: HDF5 over Zarr because:
- Better support for concurrent read/write
- More mature and stable
- h5py is already installed and well-tested
- Direct memory mapping via h5py
- Better compression options (gzip, lzf)
- Zarr v3 API still evolving

Storage layout:
- /train/kmer_features     (800_000, 9249) float32
- /train/assembly_features (800_000, 26)   float32
- /train/labels            (800_000, 2)    float32  [completeness, contamination]
- /train/metadata          (800_000,)      structured array
- Similar for /val and /test
"""

import numpy as np
import h5py
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Any


# Default configuration
DEFAULT_N_KMER = 9249
DEFAULT_N_ASSEMBLY = 26
DEFAULT_SPLITS = {
    'train': 800_000,
    'val': 100_000,
    'test': 100_000,
}
DEFAULT_BATCH_SIZE = 10_000
DEFAULT_CHUNK_SIZE = 10_000  # Samples per chunk (matches batch write size)


# Metadata dtype
METADATA_DTYPE = np.dtype([
    ('completeness', np.float32),
    ('contamination', np.float32),
    ('dominant_phylum', 'S64'),       # string, max 64 chars
    ('sample_type', 'S32'),           # e.g., 'pure', 'within_phylum', 'cross_phylum'
    ('quality_tier', 'S20'),          # e.g., 'high', 'medium', 'low'
    ('dominant_accession', 'S30'),    # GCA/GCF accession
    ('genome_full_length', np.int64),
    ('n_contaminants', np.int32),
    ('batch_id', np.int32),
])


class FeatureStore:
    """
    HDF5-based feature storage for MAGICC training data.

    Parameters
    ----------
    path : str
        Path to HDF5 file.
    n_kmer_features : int
        Number of k-mer features.
    n_assembly_features : int
        Number of assembly features.
    mode : str
        File mode: 'w' for create, 'r' for read, 'a' for append/update.
    """

    def __init__(
        self,
        path: str,
        n_kmer_features: int = DEFAULT_N_KMER,
        n_assembly_features: int = DEFAULT_N_ASSEMBLY,
        mode: str = 'r',
    ):
        self.path = path
        self.n_kmer_features = n_kmer_features
        self.n_assembly_features = n_assembly_features
        self.n_total_features = n_kmer_features + n_assembly_features
        self.mode = mode
        self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self):
        """Open the HDF5 file."""
        self._file = h5py.File(self.path, self.mode)

    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def initialize(
        self,
        splits: Optional[Dict[str, int]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        compression: str = 'gzip',
        compression_opts: int = 1,
    ):
        """
        Initialize HDF5 storage with pre-allocated arrays.

        Parameters
        ----------
        splits : dict
            Dictionary mapping split name to number of samples.
        chunk_size : int
            Chunk size for HDF5 datasets (in samples).
        compression : str
            Compression algorithm.
        compression_opts : int
            Compression level.
        """
        if splits is None:
            splits = DEFAULT_SPLITS

        assert self._file is not None, "File not opened"
        assert self.mode in ('w', 'a'), "File must be opened in write mode"

        for split_name, n_samples in splits.items():
            grp = self._file.create_group(split_name)

            # K-mer features
            grp.create_dataset(
                'kmer_features',
                shape=(n_samples, self.n_kmer_features),
                dtype=np.float32,
                chunks=(min(chunk_size, n_samples), self.n_kmer_features),
                compression=compression,
                compression_opts=compression_opts,
                fillvalue=0.0,
            )

            # Assembly features
            grp.create_dataset(
                'assembly_features',
                shape=(n_samples, self.n_assembly_features),
                dtype=np.float32,
                chunks=(min(chunk_size, n_samples), self.n_assembly_features),
                compression=compression,
                compression_opts=compression_opts,
                fillvalue=0.0,
            )

            # Labels [completeness, contamination]
            grp.create_dataset(
                'labels',
                shape=(n_samples, 2),
                dtype=np.float32,
                chunks=(min(chunk_size, n_samples), 2),
                compression=compression,
                compression_opts=compression_opts,
                fillvalue=0.0,
            )

            # Metadata
            grp.create_dataset(
                'metadata',
                shape=(n_samples,),
                dtype=METADATA_DTYPE,
                chunks=(min(chunk_size, n_samples),),
                compression=compression,
                compression_opts=compression_opts,
            )

            # Write counter
            grp.attrs['n_written'] = 0
            grp.attrs['n_total'] = n_samples

            print(f"  Initialized {split_name}: {n_samples:,} samples, "
                  f"kmer=({n_samples},{self.n_kmer_features}), "
                  f"asm=({n_samples},{self.n_assembly_features})")

        # Store configuration
        self._file.attrs['n_kmer_features'] = self.n_kmer_features
        self._file.attrs['n_assembly_features'] = self.n_assembly_features
        self._file.attrs['n_total_features'] = self.n_total_features
        self._file.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')

    def write_batch(
        self,
        split: str,
        kmer_features: np.ndarray,
        assembly_features: np.ndarray,
        labels: np.ndarray,
        metadata: np.ndarray,
        batch_offset: Optional[int] = None,
    ) -> int:
        """
        Write a batch of features to storage.

        Parameters
        ----------
        split : str
            Split name ('train', 'val', 'test').
        kmer_features : np.ndarray
            K-mer features, shape (batch_size, n_kmer_features).
        assembly_features : np.ndarray
            Assembly features, shape (batch_size, n_assembly_features).
        labels : np.ndarray
            Labels [completeness, contamination], shape (batch_size, 2).
        metadata : np.ndarray
            Structured metadata array, shape (batch_size,).
        batch_offset : int or None
            Starting index. If None, appends after last written.

        Returns
        -------
        int
            Number of samples written.
        """
        assert self._file is not None, "File not opened"
        grp = self._file[split]

        batch_size = kmer_features.shape[0]
        if batch_offset is None:
            batch_offset = int(grp.attrs['n_written'])

        end = batch_offset + batch_size
        n_total = int(grp.attrs['n_total'])
        assert end <= n_total, \
            f"Batch would exceed storage: {end} > {n_total}"

        grp['kmer_features'][batch_offset:end] = kmer_features.astype(np.float32)
        grp['assembly_features'][batch_offset:end] = assembly_features.astype(np.float32)
        grp['labels'][batch_offset:end] = labels.astype(np.float32)
        grp['metadata'][batch_offset:end] = metadata

        grp.attrs['n_written'] = max(int(grp.attrs['n_written']), end)

        return batch_size

    def read_batch(
        self,
        split: str,
        start: int,
        end: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read a batch of features from storage.

        Parameters
        ----------
        split : str
            Split name.
        start : int
            Starting index.
        end : int
            Ending index (exclusive).

        Returns
        -------
        tuple of (kmer_features, assembly_features, labels, metadata)
        """
        assert self._file is not None, "File not opened"
        grp = self._file[split]

        kmer = grp['kmer_features'][start:end]
        assembly = grp['assembly_features'][start:end]
        labels = grp['labels'][start:end]
        metadata = grp['metadata'][start:end]

        return kmer, assembly, labels, metadata

    def get_split_info(self, split: str) -> Dict[str, Any]:
        """Get information about a split."""
        assert self._file is not None, "File not opened"
        grp = self._file[split]
        return {
            'n_total': int(grp.attrs['n_total']),
            'n_written': int(grp.attrs['n_written']),
            'kmer_shape': grp['kmer_features'].shape,
            'assembly_shape': grp['assembly_features'].shape,
            'labels_shape': grp['labels'].shape,
            'chunks': grp['kmer_features'].chunks,
            'compression': grp['kmer_features'].compression,
        }

    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all splits."""
        assert self._file is not None, "File not opened"
        info = {}
        for split in self._file.keys():
            info[split] = self.get_split_info(split)
        return info
