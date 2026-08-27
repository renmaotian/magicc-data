#!/usr/bin/env python3
"""
MAGICC Command-Line Interface

Production-ready CLI for genome quality assessment.
Takes a directory of genome FASTA files (or a single FASTA file) as input,
runs the full pipeline (read FASTA -> k-mer counting -> assembly stats ->
normalization -> ONNX inference), and outputs a TSV with predictions.

Usage:
    python -m magicc predict --input /path/to/genomes --output predictions.tsv
    python -m magicc predict --input genome.fasta --output predictions.tsv
    python -m magicc predict --input /path/to/genomes --output predictions.tsv --threads 8
"""

import argparse
import logging
import os
import sys
import time
import json
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

MODEL_URL = "https://github.com/renmaotian/magicc/raw/main/models/magicc_v3.onnx"
MODEL_FILENAME = "magicc_v3.onnx"

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
# FASTA I/O
# ---------------------------------------------------------------------------
def read_fasta_contigs(fasta_path: str) -> List[str]:
    """
    Read a FASTA file and return a list of contig sequences.

    Handles:
    - Multi-line FASTA
    - Mixed case (uppercased)
    - Empty files (returns empty list)
    - Files with no valid sequences
    """
    contigs = []
    current_parts = []
    try:
        with open(fasta_path, 'r') as f:
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
    except Exception as e:
        logger.warning("Failed to read %s: %s", fasta_path, e)
    return contigs


def validate_fasta(fasta_path: str) -> bool:
    """Quick validation that a file looks like FASTA."""
    try:
        with open(fasta_path, 'r') as f:
            first_line = ''
            for line in f:
                first_line = line.strip()
                if first_line:
                    break
            return first_line.startswith('>')
    except Exception:
        return False


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

    contigs = read_fasta_contigs(fasta_path)
    if not contigs:
        logger.warning("Skipping %s: no valid contigs", genome_name)
        return None

    kmer_counts = _worker_kmer_counter.count_contigs(contigs)
    log10_total = _worker_kmer_counter.total_kmer_count(kmer_counts)
    assembly_feats = compute_assembly_stats(contigs, log10_total, kmer_counts)

    return (genome_name, kmer_counts.astype(np.float32), assembly_feats.astype(np.float32))


# ---------------------------------------------------------------------------
# Discovery of genome files
# ---------------------------------------------------------------------------
def discover_genomes(input_path: str, extension: str) -> List[Tuple[str, str]]:
    """
    Discover genome FASTA files from input path.

    Returns list of (genome_name, fasta_path) tuples, sorted by name.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        name = input_path.stem
        return [(name, str(input_path))]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension

    genomes = []
    for entry in sorted(input_path.iterdir()):
        if entry.is_file() and entry.name.endswith(extension):
            name = entry.stem
            genomes.append((name, str(entry)))

    return genomes


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------
def predict(
    input_path: str,
    output_path: str,
    model_path: str = None,
    norm_path: str = DEFAULT_NORM_PATH,
    kmer_path: str = DEFAULT_KMER_PATH,
    threads: int = 1,
    batch_size: int = 64,
    extension: str = '.fasta',
) -> Dict[str, Any]:
    """
    Run the full MAGICC prediction pipeline.

    Parameters
    ----------
    input_path : str
        Directory of genome FASTA files, or path to a single FASTA file.
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
        File extension filter for genome discovery.

    Returns
    -------
    dict with timing and summary information.
    """
    import onnxruntime as ort
    from magicc.normalization import FeatureNormalizer

    t_total_start = time.time()

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
    logger.info("Discovering genome files...")
    genomes = discover_genomes(input_path, extension)
    if not genomes:
        raise RuntimeError(
            f"No genome files found at {input_path} with extension '{extension}'"
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
    assembly_array = np.stack(assembly_list)    # (n, 26)

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
        description='Run the MAGICC prediction pipeline on genome FASTA file(s).',
    )
    predict_parser.add_argument(
        '--input', '-i', required=True,
        help='Path to a directory of genome FASTA files or a single FASTA file',
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
        help='Genome file extension filter (default: .fasta)',
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
