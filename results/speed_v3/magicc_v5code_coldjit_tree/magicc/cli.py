#!/usr/bin/env python3
"""
MAGICC Command-Line Interface

Production-ready CLI for genome quality assessment.
Takes a directory of genome FASTA files, a single FASTA file, or a text file
listing genome paths as input, runs the full pipeline (read FASTA -> k-mer
counting -> assembly stats -> normalization -> ONNX inference), and outputs a
TSV with predictions.

Input FASTA files may be plain text or gzip-compressed (``.fasta.gz``,
``.fa.gz``, ``.fna.gz``, ...); compression is detected from the file contents,
so predictions are identical either way.

Usage:
    python -m magicc predict --input /path/to/genomes --output predictions.tsv
    python -m magicc predict --input genome.fasta --output predictions.tsv
    python -m magicc predict --input genome.fasta.gz --output predictions.tsv
    python -m magicc predict --input /path/to/genomes --output predictions.tsv --threads 8
    python -m magicc predict --input /path/to/genomes --extension auto -o out.tsv
    python -m magicc predict --input-list genome_paths.txt --output predictions.tsv
"""

import argparse
import gzip
import logging
import os
import sys
import time
import json
import zlib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------------------------
# Resolve default resource paths -- check package data first, then project layout
# ---------------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent          # magicc/
_PROJECT_DIR = _PACKAGE_DIR.parent                       # magicc2/ (dev layout)
_USER_DATA_DIR = Path.home() / '.magicc'

MODEL_URL = "https://github.com/renmaotian/magicc/raw/main/models/magicc_v5.onnx"
MODEL_FILENAME = "magicc_v5.onnx"

def _resolve_data_path(*candidates: str) -> str:
    """Return the first candidate path that exists, or the last one as default."""
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[-1]

def _ensure_model() -> str:
    """Ensure the ONNX model is available, downloading from GitHub if needed."""
    # Check package data, project layout, then user cache
    candidates = [
        _PACKAGE_DIR / 'data' / MODEL_FILENAME,
        _PROJECT_DIR / 'models' / MODEL_FILENAME,
        _USER_DATA_DIR / MODEL_FILENAME,
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 1_000_000:  # >1MB sanity check
            return str(p)

    # Download to user cache
    dest = _USER_DATA_DIR / MODEL_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MAGICC model to {dest} ...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, str(dest))
    print(f"Download complete ({dest.stat().st_size / 1e6:.1f} MB)")
    return str(dest)

# Package data paths (installed via pip) vs project layout paths (development)
DEFAULT_NORM_PATH = _resolve_data_path(
    str(_PACKAGE_DIR / 'data' / 'normalization_params.json'),
    str(_PROJECT_DIR / 'data' / 'features' / 'normalization_params.json'),
)
DEFAULT_KMER_PATH = _resolve_data_path(
    str(_PACKAGE_DIR / 'data' / 'selected_kmers.txt'),
    str(_PROJECT_DIR / 'data' / 'kmer_selection' / 'selected_kmers.txt'),
)

# Logging
logger = logging.getLogger('magicc')


# ---------------------------------------------------------------------------
# FASTA I/O  (plain text and gzip)
# ---------------------------------------------------------------------------
#: FASTA extensions recognised for genome discovery and for deriving genome
#: names.  Each may additionally carry a ``.gz`` suffix.  Ordered longest-first
#: so that e.g. ``.fasta`` is stripped before ``.fa`` is considered.
FASTA_EXTENSIONS: Tuple[str, ...] = (
    '.fasta', '.fna', '.ffn', '.fas', '.fsa', '.seq', '.fa',
)

#: gzip member header magic bytes (RFC 1952).
GZIP_MAGIC = b'\x1f\x8b'


class FastaReadError(Exception):
    """Raised when a FASTA file cannot be read (missing, unreadable, corrupt gzip)."""


def _is_gzipped(fasta_path: str) -> bool:
    """
    Return True if *fasta_path* is a gzip stream.

    Detection is by magic bytes rather than by file name, so gzip files with a
    non-standard name are handled and a mis-named plain-text file is not
    mistakenly treated as compressed.  Falls back to the file extension if the
    file cannot be opened (the subsequent open will raise the real error).
    """
    try:
        with open(fasta_path, 'rb') as f:
            return f.read(2) == GZIP_MAGIC
    except OSError:
        return fasta_path.lower().endswith('.gz')


def open_fasta(fasta_path: str):
    """
    Open a FASTA file for text reading, transparently decompressing gzip input.

    The encoding is pinned to UTF-8 with strict error handling so that results
    do not depend on the caller's locale, and so that binary/garbage input is
    rejected rather than silently mangled.

    Returns a file object suitable for use as a context manager.  Callers should
    wrap use in ``try/except FastaReadError`` via :func:`read_fasta_contigs`.
    """
    if _is_gzipped(fasta_path):
        return gzip.open(fasta_path, 'rt', encoding='utf-8')
    return open(fasta_path, 'r', encoding='utf-8')


def read_fasta_contigs(fasta_path: str) -> List[str]:
    """
    Read a FASTA file and return a list of contig sequences.

    Handles:
    - Plain-text and gzip-compressed input (detected from file contents)
    - Multi-line FASTA
    - Mixed case (uppercased)
    - CRLF line endings
    - Empty files / files with no valid sequences (returns empty list)

    Raises
    ------
    FastaReadError
        If the file is missing, unreadable, or a corrupt/truncated gzip stream.
        An empty but readable file is *not* an error -- it yields ``[]``.
    """
    contigs: List[str] = []
    current_parts: List[str] = []
    try:
        with open_fasta(fasta_path) as f:
            for line in f:
                line = line.rstrip('\n\r')
                if line.startswith('>'):
                    if current_parts:
                        seq = ''.join(current_parts).upper()
                        if seq:
                            contigs.append(seq)
                        current_parts = []
                else:
                    current_parts.append(line.strip())
        if current_parts:
            seq = ''.join(current_parts).upper()
            if seq:
                contigs.append(seq)
    except (OSError, EOFError, zlib.error, UnicodeDecodeError) as e:
        # OSError covers missing/unreadable files and gzip.BadGzipFile;
        # EOFError/zlib.error cover truncated and corrupt gzip streams;
        # UnicodeDecodeError covers binary input mistaken for FASTA.
        raise FastaReadError(f"Failed to read {fasta_path}: {e}") from e
    return contigs


def validate_fasta(fasta_path: str) -> bool:
    """Quick validation that a file looks like FASTA (gzip-aware)."""
    try:
        with open_fasta(fasta_path) as f:
            first_line = ''
            for line in f:
                first_line = line.strip()
                if first_line:
                    break
            return first_line.startswith('>')
    except Exception:
        return False


def genome_name_from_path(fasta_path: str) -> str:
    """
    Derive a genome name from a FASTA path by stripping the compression suffix
    and then a recognised FASTA extension.

    ``genome_0.fasta`` -> ``genome_0``
    ``genome_0.fasta.gz`` -> ``genome_0``
    ``GCA_000009265.1.fna.gz`` -> ``GCA_000009265.1``

    Files with no recognised FASTA extension fall back to stripping the final
    suffix, matching the historical ``pathlib.Path.stem`` behaviour.
    """
    name = os.path.basename(fasta_path)
    if name.lower().endswith('.gz'):
        name = name[:-3]
    lowered = name.lower()
    for ext in FASTA_EXTENSIONS:
        if lowered.endswith(ext) and len(name) > len(ext):
            return name[:-len(ext)]
    return Path(name).stem


# ---------------------------------------------------------------------------
# Feature extraction (designed to work with multiprocessing)
# ---------------------------------------------------------------------------
# These module-level variables are set by _init_worker so that each worker
# process has its own KmerCounter instance (Numba-compiled, not picklable).
_worker_kmer_counter = None
_worker_kmer_path = None


def _init_worker(kmer_path: str):
    """Initializer for multiprocessing workers -- creates per-process KmerCounter."""
    global _worker_kmer_counter, _worker_kmer_path
    _worker_kmer_path = kmer_path
    from magicc.kmer_counter import KmerCounter
    _worker_kmer_counter = KmerCounter(kmer_path)
    # Warm up Numba JIT with a tiny sequence
    _worker_kmer_counter.count_sequence("ACGTACGTACGTACGTACGT" * 10)


def _extract_features_worker(args: Tuple[str, str]) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Worker function for parallel feature extraction.

    Parameters
    ----------
    args : (genome_name, fasta_path)

    Returns
    -------
    (genome_name, kmer_counts, assembly_features) or None on failure.
    """
    genome_name, fasta_path = args

    from magicc.assembly_stats import compute_assembly_stats

    try:
        contigs = read_fasta_contigs(fasta_path)
    except FastaReadError as e:
        logger.warning("Skipping %s: %s", genome_name, e)
        return None
    if not contigs:
        logger.warning("Skipping %s: no valid contigs", genome_name)
        return None

    kmer_counts = _worker_kmer_counter.count_contigs(contigs)
    log10_total = _worker_kmer_counter.total_kmer_count(kmer_counts)
    assembly_feats = compute_assembly_stats(log10_total, kmer_counts)

    return (genome_name, kmer_counts.astype(np.float32), assembly_feats.astype(np.float32))


# ---------------------------------------------------------------------------
# Discovery of genome files
# ---------------------------------------------------------------------------
#: Sentinel values for ``--extension`` meaning "any recognised FASTA extension".
AUTO_EXTENSIONS = ('auto', 'any', '*')


def _matches_extension(filename: str, extension: str) -> bool:
    """
    Test whether *filename* should be treated as a genome FASTA.

    ``extension`` is either one of :data:`AUTO_EXTENSIONS` (accept any name in
    :data:`FASTA_EXTENSIONS`, with or without a ``.gz`` suffix), or a literal
    suffix.  A literal suffix also matches its gzip form, so the historical
    default ``.fasta`` now picks up ``.fasta.gz`` as well.
    """
    if extension.lower() in AUTO_EXTENSIONS:
        stem = filename[:-3] if filename.lower().endswith('.gz') else filename
        lowered = stem.lower()
        return any(lowered.endswith(ext) and len(stem) > len(ext)
                   for ext in FASTA_EXTENSIONS)
    return filename.endswith(extension) or filename.endswith(extension + '.gz')


def discover_genomes(input_path: str, extension: str = '.fasta') -> List[Tuple[str, str]]:
    """
    Discover genome FASTA files from an input path.

    Parameters
    ----------
    input_path : str
        A single FASTA file (used as-is, regardless of *extension*) or a
        directory to scan.
    extension : str
        Suffix filter for directory scans.  A literal suffix (e.g. ``.fasta``,
        ``.fna``) also matches the gzip form (``.fasta.gz``).  Pass ``auto`` to
        accept any recognised FASTA extension, compressed or not.

    Returns
    -------
    list of (genome_name, fasta_path), sorted by file name.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return [(genome_name_from_path(str(input_path)), str(input_path))]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Ensure a bare extension starts with a dot ('fasta' -> '.fasta').  Compound
    # suffixes that already contain a dot (e.g. NCBI's '_genomic.fna') are left
    # alone so they match as written.
    if (extension.lower() not in AUTO_EXTENSIONS
            and not extension.startswith('.') and '.' not in extension):
        extension = '.' + extension

    genomes = []
    for entry in sorted(input_path.iterdir()):
        if entry.is_file() and _matches_extension(entry.name, extension):
            genomes.append((genome_name_from_path(entry.name), str(entry)))

    return genomes


def discover_genomes_from_list(list_path: str) -> List[Tuple[str, str]]:
    """
    Read a text file of genome paths (one per line) and resolve them.

    Format
    ------
    * one genome path per line, plain or gzip-compressed, extensions may differ
    * blank lines are ignored
    * lines whose first non-whitespace character is ``#`` are comments
      (``#`` is *not* treated as a comment character inside a path)
    * leading/trailing whitespace is stripped; ``~`` is expanded
    * relative paths are resolved against the current working directory first,
      then against the directory containing the list file

    Returns
    -------
    list of (genome_name, absolute_fasta_path) in list-file order, with
    duplicate paths removed (first occurrence kept).

    Raises
    ------
    FileNotFoundError
        If the list file itself is missing, or if any listed path does not
        exist or is not a regular file.  All offending entries are reported
        together with their line numbers.
    ValueError
        If the list file contains no genome paths.
    """
    list_file = Path(list_path)
    if not list_file.exists():
        raise FileNotFoundError(f"Genome list file not found: {list_path}")
    if not list_file.is_file():
        raise FileNotFoundError(
            f"Genome list file not found: {list_path} is a directory, not a file"
        )

    try:
        raw_lines = list_file.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError as e:
        raise FileNotFoundError(f"Could not read genome list file {list_path}: {e}") from e

    list_dir = list_file.parent
    entries: List[Tuple[int, str]] = []
    for lineno, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        entries.append((lineno, stripped))

    if not entries:
        raise ValueError(
            f"Genome list file {list_path} contains no genome paths "
            "(only blank lines and/or '#' comments)"
        )

    genomes: List[Tuple[str, str]] = []
    seen = set()
    missing: List[str] = []
    not_files: List[str] = []

    for lineno, entry in entries:
        candidate = Path(os.path.expanduser(entry))
        if candidate.is_absolute():
            resolved = candidate
        else:
            cwd_candidate = Path.cwd() / candidate
            list_candidate = list_dir / candidate
            if cwd_candidate.exists():
                resolved = cwd_candidate
            elif list_candidate.exists():
                resolved = list_candidate
            else:
                resolved = cwd_candidate

        abs_path = os.path.abspath(str(resolved))

        if not os.path.exists(abs_path):
            missing.append(f"  line {lineno}: {entry}")
            continue
        if not os.path.isfile(abs_path):
            not_files.append(f"  line {lineno}: {entry} (not a file)")
            continue

        if abs_path in seen:
            logger.warning(
                "Genome list %s line %d: duplicate path, skipping: %s",
                list_path, lineno, entry,
            )
            continue
        seen.add(abs_path)
        genomes.append((genome_name_from_path(abs_path), abs_path))

    if missing or not_files:
        problems = missing + not_files
        raise FileNotFoundError(
            f"Genome list {list_path}: {len(problems)} path(s) could not be used:\n"
            + "\n".join(problems)
            + "\nPaths may be absolute, or relative to the current directory "
              "or to the list file."
        )

    # Warn on duplicate genome names (different files, same derived name)
    name_counts: Dict[str, int] = {}
    for name, _ in genomes:
        name_counts[name] = name_counts.get(name, 0) + 1
    dupes = sorted(n for n, c in name_counts.items() if c > 1)
    if dupes:
        logger.warning(
            "Genome list %s: %d duplicate genome name(s) (output rows will repeat): %s",
            list_path, len(dupes), ', '.join(dupes[:10]),
        )

    return genomes


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------
def predict(
    input_path: Optional[str] = None,
    output_path: str = None,
    model_path: str = None,
    norm_path: str = DEFAULT_NORM_PATH,
    kmer_path: str = DEFAULT_KMER_PATH,
    threads: int = 1,
    batch_size: int = 64,
    extension: str = '.fasta',
    input_list: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full MAGICC prediction pipeline.

    Parameters
    ----------
    input_path : str, optional
        Directory of genome FASTA files, or path to a single FASTA file.
        Plain and gzip-compressed FASTA are both accepted.
        Mutually exclusive with *input_list*.
    output_path : str
        Path to write the output TSV.
    model_path : str
        Path to the ONNX model.
    norm_path : str
        Path to the normalization parameters JSON.
    kmer_path : str
        Path to the selected k-mers file.
    threads : int
        Number of threads for parallel feature extraction.
    batch_size : int
        Batch size for ONNX inference.
    extension : str
        File extension filter for directory discovery.  A literal suffix also
        matches its gzip form; ``auto`` accepts any recognised FASTA extension.
    input_list : str, optional
        Text file listing genome paths, one per line (``#`` comments and blank
        lines allowed).  Mutually exclusive with *input_path*.

    Returns
    -------
    dict with timing and summary information.
    """
    import onnxruntime as ort
    from magicc.normalization import FeatureNormalizer

    t_total_start = time.time()

    # ------------------------------------------------------------------
    # Input source: exactly one of input_path / input_list
    # ------------------------------------------------------------------
    if input_path is not None and input_list is not None:
        raise ValueError(
            "--input and --input-list are mutually exclusive; provide only one. "
            "Use --input for a directory or single FASTA file, --input-list for "
            "a text file of genome paths."
        )
    if input_path is None and input_list is None:
        raise ValueError(
            "No input specified: provide --input (directory or FASTA file) or "
            "--input-list (text file of genome paths)."
        )
    if output_path is None:
        raise ValueError("No output specified: provide --output.")

    # ------------------------------------------------------------------
    # Resolve model path (auto-download if needed)
    # ------------------------------------------------------------------
    if model_path is None:
        model_path = _ensure_model()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for label, path in [('Model', model_path), ('Normalization params', norm_path),
                        ('K-mer list', kmer_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # ------------------------------------------------------------------
    # Discover genomes
    # ------------------------------------------------------------------
    if input_list is not None:
        logger.info("Reading genome list: %s", input_list)
        genomes = discover_genomes_from_list(input_list)
        if not genomes:
            raise RuntimeError(f"No genome files listed in {input_list}")
    else:
        logger.info("Discovering genome files...")
        genomes = discover_genomes(input_path, extension)
        if not genomes:
            hint = ''
            if os.path.isdir(input_path):
                present = discover_genomes(input_path, 'auto')
                if present:
                    exts = sorted({
                        os.path.basename(p)[len(n):] for n, p in present
                    })
                    hint = (
                        f" FASTA-like files with extension(s) {', '.join(exts)} "
                        f"are present -- re-run with --extension <ext> or "
                        f"--extension auto."
                    )
            raise RuntimeError(
                f"No genome files found at {input_path} with extension "
                f"'{extension}'.{hint}"
            )
    logger.info("Found %d genome(s)", len(genomes))

    # ------------------------------------------------------------------
    # Load resources
    # ------------------------------------------------------------------
    logger.info("Loading ONNX model: %s", model_path)
    sess_options = ort.SessionOptions()
    # ONNX inference uses 1 thread regardless (lightweight step)
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path, sess_options, providers=['CPUExecutionProvider']
    )
    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    logger.info("Loading normalization parameters: %s", norm_path)
    normalizer = FeatureNormalizer.load(norm_path)

    # ------------------------------------------------------------------
    # Feature extraction (parallelized)
    # ------------------------------------------------------------------
    n_genomes = len(genomes)
    if threads <= 0:
        threads = cpu_count() or 1
    effective_threads = min(threads, n_genomes)
    logger.info(
        "Extracting features for %d genomes using %d thread(s)...",
        n_genomes, effective_threads,
    )

    t_feat_start = time.time()

    if effective_threads <= 1:
        # Single-threaded: init worker in main process
        _init_worker(kmer_path)
        results_raw = []
        for idx, (gname, gpath) in enumerate(genomes):
            res = _extract_features_worker((gname, gpath))
            results_raw.append(res)
            if (idx + 1) % 100 == 0 or (idx + 1) == n_genomes:
                elapsed = time.time() - t_feat_start
                speed = (idx + 1) / elapsed
                logger.info(
                    "  Feature extraction: %d/%d (%.1f genomes/s, %.1f ms/genome)",
                    idx + 1, n_genomes, speed, 1000.0 / max(speed, 0.001),
                )
    else:
        # Multi-threaded using multiprocessing.Pool
        with Pool(
            processes=effective_threads,
            initializer=_init_worker,
            initargs=(kmer_path,),
        ) as pool:
            results_raw = []
            for idx, res in enumerate(
                pool.imap(
                    _extract_features_worker,
                    genomes,
                    chunksize=max(1, n_genomes // (effective_threads * 4)),
                )
            ):
                results_raw.append(res)
                if (idx + 1) % 100 == 0 or (idx + 1) == n_genomes:
                    elapsed = time.time() - t_feat_start
                    speed = (idx + 1) / elapsed
                    logger.info(
                        "  Feature extraction: %d/%d (%.1f genomes/s, %.1f ms/genome)",
                        idx + 1, n_genomes, speed, 1000.0 / max(speed, 0.001),
                    )

    t_feat_end = time.time()
    feat_time = t_feat_end - t_feat_start

    # Collect valid results
    valid_names = []
    kmer_list = []
    assembly_list = []
    skipped = 0
    for res in results_raw:
        if res is None:
            skipped += 1
            continue
        gname, kmer_counts, assembly_feats = res
        valid_names.append(gname)
        kmer_list.append(kmer_counts)
        assembly_list.append(assembly_feats)

    n_valid = len(valid_names)
    if n_valid == 0:
        raise RuntimeError("No valid genomes after feature extraction")

    if skipped > 0:
        logger.warning("Skipped %d genome(s) (empty or invalid)", skipped)

    logger.info(
        "Feature extraction complete: %d genomes in %.1fs (%.1f ms/genome)",
        n_valid, feat_time, feat_time / n_valid * 1000,
    )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    logger.info("Normalizing features...")
    t_norm_start = time.time()

    kmer_array = np.stack(kmer_list)           # (n, 9249)
    assembly_array = np.stack(assembly_list)    # (n, 7)

    kmer_norm = normalizer.normalize_kmer(kmer_array).astype(np.float32)
    assembly_norm = normalizer.normalize_assembly(assembly_array).astype(np.float32)

    t_norm_end = time.time()
    norm_time = t_norm_end - t_norm_start
    logger.info("Normalization: %.3fs", norm_time)

    # ------------------------------------------------------------------
    # ONNX Inference
    # ------------------------------------------------------------------
    logger.info("Running ONNX inference (batch_size=%d)...", batch_size)
    t_infer_start = time.time()

    predictions = np.zeros((n_valid, 2), dtype=np.float32)
    for batch_start in range(0, n_valid, batch_size):
        batch_end = min(batch_start + batch_size, n_valid)
        feed = {
            input_names[0]: kmer_norm[batch_start:batch_end],
            input_names[1]: assembly_norm[batch_start:batch_end],
        }
        result = session.run([output_name], feed)
        predictions[batch_start:batch_end] = result[0]

    t_infer_end = time.time()
    infer_time = t_infer_end - t_infer_start
    logger.info(
        "Inference complete: %.3fs (%.2f ms/genome)",
        infer_time, infer_time / n_valid * 1000,
    )

    # ------------------------------------------------------------------
    # Write output TSV
    # ------------------------------------------------------------------
    logger.info("Writing predictions to %s", output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("genome_name\tpred_completeness\tpred_contamination\n")
        for i, gname in enumerate(valid_names):
            comp = float(predictions[i, 0])
            cont = float(predictions[i, 1])
            f.write(f"{gname}\t{comp:.4f}\t{cont:.4f}\n")

    t_total_end = time.time()
    total_time = t_total_end - t_total_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    speed_genomes_per_min = n_valid / total_time * 60 if total_time > 0 else 0
    speed_per_thread = speed_genomes_per_min / max(effective_threads, 1)

    summary = {
        'n_genomes': n_valid,
        'n_skipped': skipped,
        'threads': effective_threads,
        'batch_size': batch_size,
        'feature_extraction_s': feat_time,
        'normalization_s': norm_time,
        'inference_s': infer_time,
        'total_time_s': total_time,
        'genomes_per_min': speed_genomes_per_min,
        'genomes_per_min_per_thread': speed_per_thread,
        'ms_per_genome': total_time / n_valid * 1000,
        'output_path': output_path,
    }

    logger.info("="*60)
    logger.info("MAGICC Prediction Summary")
    logger.info("="*60)
    logger.info("  Genomes processed: %d", n_valid)
    if skipped > 0:
        logger.info("  Genomes skipped:   %d", skipped)
    logger.info("  Threads:           %d", effective_threads)
    logger.info("  Feature extraction: %.1fs", feat_time)
    logger.info("  Normalization:      %.3fs", norm_time)
    logger.info("  ONNX inference:     %.3fs", infer_time)
    logger.info("  Total time:         %.1fs", total_time)
    logger.info("  Speed:              %.0f genomes/min", speed_genomes_per_min)
    logger.info("  Speed/thread:       %.0f genomes/min/thread", speed_per_thread)
    logger.info("  Output:             %s", output_path)
    logger.info("="*60)

    return summary


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the MAGICC CLI."""
    parser = argparse.ArgumentParser(
        prog='magicc',
        description=(
            'MAGICC - Metagenome-Assembled Genome Inference of '
            'Completeness and Contamination.\n\n'
            'Ultra-fast genome quality assessment using core gene '
            'k-mer profiles and deep learning.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # predict sub-command
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict completeness and contamination for genome(s)',
        description=(
            'Run the MAGICC prediction pipeline on genome FASTA file(s).\n\n'
            'Input may be plain or gzip-compressed FASTA (.fasta, .fa, .fna, '
            '.fas, .ffn and their .gz forms). Compression is detected from the\n'
            'file contents, so predictions are identical for X.fasta and '
            'X.fasta.gz.\n\n'
            'Genomes can be supplied as a directory, a single file (--input), '
            'or a text file\nlisting one genome path per line (--input-list).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i', default=None,
        help='Path to a directory of genome FASTA files or a single FASTA file '
             '(plain or .gz)',
    )
    input_group.add_argument(
        '--input-list', '-I', default=None,
        help='Text file listing genome paths, one per line. Blank lines and '
             'lines starting with "#" are ignored; paths may be absolute or '
             'relative and may mix compressed and uncompressed files',
    )
    predict_parser.add_argument(
        '--output', '-o', required=True,
        help='Output TSV file path for predictions',
    )
    predict_parser.add_argument(
        '--threads', '-t', type=int, default=0,
        help='Number of threads for parallel feature extraction (default: 0 = all CPUs)',
    )
    predict_parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for ONNX inference (default: 64)',
    )
    predict_parser.add_argument(
        '--extension', '-x', default='.fasta',
        help='Genome file extension filter for directory input (default: '
             '.fasta). The matching gzip form is always accepted too, so '
             '".fasta" also matches ".fasta.gz". Use "auto" to accept any '
             'recognised FASTA extension, compressed or not',
    )
    predict_parser.add_argument(
        '--model', default=None,
        help='Path to ONNX model (default: auto-download from GitHub)',
    )
    predict_parser.add_argument(
        '--normalization', default=DEFAULT_NORM_PATH,
        help=f'Path to normalization params JSON (default: {DEFAULT_NORM_PATH})',
    )
    predict_parser.add_argument(
        '--kmers', default=DEFAULT_KMER_PATH,
        help=f'Path to selected k-mers file (default: {DEFAULT_KMER_PATH})',
    )
    predict_parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress progress output (only errors)',
    )
    predict_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose debug output',
    )

    return parser


def main(argv: Optional[List[str]] = None):
    """Main entry point for the MAGICC CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'predict':
        # Set up logging
        if args.quiet:
            log_level = logging.WARNING
        elif args.verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        try:
            summary = predict(
                input_path=args.input,
                input_list=args.input_list,
                output_path=args.output,
                model_path=args.model,
                norm_path=args.normalization,
                kmer_path=args.kmers,
                threads=args.threads,
                batch_size=args.batch_size,
                extension=args.extension,
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            sys.exit(1)


if __name__ == '__main__':
    main()
