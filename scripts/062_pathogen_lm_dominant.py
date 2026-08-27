#!/usr/bin/env python3
"""
Script 62: Pathogen Synthetic Genome — L. monocytogenes DOMINANT, S. enterica CONTAMINANT

EXACT completeness & contamination, no fragmentation.
Each synthetic genome has exactly 1 or 2 contigs:
  - 1 contig for the dominant portion (L.m. whole genome or contiguous substring)
  - 1 contig for the contaminant portion (S.e. contiguous substring)

Completeness = dominant_bp / dominant_genome_full_size (L.m.)
Contamination = contaminant_bp / dominant_genome_full_size (L.m.)

5 configurations x 10 replicates (seeds 0-9).
Both MAGICC V5 and CheckM2 are run.

Model: models/magicc_v5.onnx
"""

import sys
import os
import signal
import json
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

# Collapse-safe: ignore SIGHUP
signal.signal(signal.SIGHUP, signal.SIG_IGN)

sys.path.insert(0, '/path/to/magicc-legacy')

import onnxruntime as ort
from magicc.kmer_counter import KmerCounter
from magicc.assembly_stats import compute_assembly_stats
from magicc.normalization import FeatureNormalizer

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/path/to/magicc-legacy')
DATA_DIR = PROJECT_DIR / 'data'

SELECTED_KMERS_PATH = str(DATA_DIR / 'kmer_selection' / 'selected_kmers.txt')
NORMALIZATION_PATH = str(DATA_DIR / 'features' / 'normalization_params.json')
ONNX_MODEL_PATH = str(PROJECT_DIR / 'models' / 'magicc_v5.onnx')

# L. monocytogenes = DOMINANT, S. enterica = CONTAMINANT
LM_GENOME = DATA_DIR / 'genomes' / 'GCF_000021185.1' / 'GCF_000021185.1_ASM2118v1_genomic.fna'
SE_GENOME = DATA_DIR / 'genomes' / 'GCF_001302605.1' / 'GCF_001302605.1_ASM130260v1_genomic.fna'

OUTPUT_DIR = DATA_DIR / 'benchmarks' / 'pathogen_analysis_v5' / 'exact_synthetic_lm_dominant'
FASTA_DIR = OUTPUT_DIR / 'fastas'  # All 50 FASTAs for CheckM2 batch

CHECKM2_DB = '/path/to/magicc/tools/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd'
CHECKM2_ENV = 'checkm2_py39'
CHECKM2_THREADS = 43

N_REPLICATES = 10
REPLICATE_SEEDS = list(range(10))  # seeds 0-9

# 5 genome configurations: (name, dominant_comp_frac, contaminant_frac_of_dominant)
# Dominant = L.m., Contaminant = S.e.
GENOME_CONFIGS = [
    ("100lm_5se",   1.00, 0.05),
    ("100lm_20se",  1.00, 0.20),
    ("100lm_50se",  1.00, 0.50),
    ("100lm_100se", 1.00, 1.00),
    ("60lm_40se",   0.60, 0.40),
]


# ============================================================================
# FASTA I/O
# ============================================================================
def read_fasta_contigs(fasta_path: str) -> List[str]:
    """Read FASTA file and return list of contig sequences."""
    contigs = []
    current_parts = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_parts:
                    contigs.append(''.join(current_parts).upper())
                    current_parts = []
            else:
                current_parts.append(line.strip())
    if current_parts:
        contigs.append(''.join(current_parts).upper())
    return [c for c in contigs if len(c) > 0]


def read_full_sequence(fasta_path: str) -> str:
    """Read FASTA and return all contigs concatenated into one sequence."""
    contigs = read_fasta_contigs(fasta_path)
    return ''.join(contigs)


def write_fasta(path: str, contigs: List[Tuple[str, str]]):
    """Write contigs as FASTA. contigs = list of (header, sequence)."""
    with open(path, 'w') as f:
        for header, seq in contigs:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


# ============================================================================
# Synthetic genome generation — EXACT, no fragmentation
# ============================================================================
def generate_exact_synthetic(
    dominant_seq: str,
    contaminant_seq: str,
    dominant_full_size: int,
    comp_frac: float,
    cont_frac: float,
    seed: int,
) -> Tuple[List[str], int, int]:
    """
    Generate synthetic genome with EXACT completeness and contamination.
    No fragmentation — output has 1 or 2 contigs.

    Parameters
    ----------
    dominant_seq : str
        Full concatenated dominant genome sequence (L.m.).
    contaminant_seq : str
        Full concatenated contaminant genome sequence (S.e.).
    dominant_full_size : int
        Total bp of the dominant genome (for computing fractions).
    comp_frac : float
        Completeness fraction (e.g. 0.60 for 60%).
    cont_frac : float
        Contamination fraction of dominant size (e.g. 0.50 for 50%).
    seed : int
        Random seed for selecting start positions.

    Returns
    -------
    contigs : list of str
        1 or 2 contig sequences.
    dominant_bp : int
        Exact bp in dominant contig.
    contaminant_bp : int
        Exact bp in contaminant contig (0 if no contamination).
    """
    rng = np.random.default_rng(seed)

    # --- Dominant portion ---
    if comp_frac >= 1.0:
        # 100% completeness: use full genome sequence as-is
        dominant_contig = dominant_seq
    else:
        # X% completeness: take a random contiguous substring
        target_bp = int(round(dominant_full_size * comp_frac))
        max_start = len(dominant_seq) - target_bp
        assert max_start >= 0, (
            f"Dominant sequence ({len(dominant_seq)} bp) shorter than "
            f"target ({target_bp} bp)"
        )
        start = rng.integers(0, max_start + 1)  # inclusive of max_start
        dominant_contig = dominant_seq[start:start + target_bp]

    dominant_bp = len(dominant_contig)

    # --- Contaminant portion ---
    contaminant_bp = 0
    contaminant_contig = None

    if cont_frac > 0:
        target_cont_bp = int(round(dominant_full_size * cont_frac))

        # If contaminant genome is shorter than needed, duplicate it
        cont_pool = contaminant_seq
        if len(cont_pool) < target_cont_bp:
            repeats = (target_cont_bp // len(contaminant_seq)) + 1
            cont_pool = contaminant_seq * repeats

        # Take a random contiguous substring from the pool
        max_start = len(cont_pool) - target_cont_bp
        start = rng.integers(0, max_start + 1)
        contaminant_contig = cont_pool[start:start + target_cont_bp]
        contaminant_bp = len(contaminant_contig)

    # Build contig list
    contigs = [dominant_contig]
    if contaminant_contig is not None:
        contigs.append(contaminant_contig)

    return contigs, dominant_bp, contaminant_bp


# ============================================================================
# Feature extraction + ONNX inference
# ============================================================================
def extract_and_predict(
    contigs: List[str],
    kmer_counter: KmerCounter,
    normalizer: FeatureNormalizer,
    session,
) -> Tuple[float, float]:
    """Extract features from contigs and run ONNX inference. Returns (comp, cont)."""
    kmer_counts = kmer_counter.count_contigs(contigs)
    log10_total = kmer_counter.total_kmer_count(kmer_counts)
    assembly_feats = compute_assembly_stats(log10_total, kmer_counts)

    kmer_norm = normalizer.normalize_kmer(
        kmer_counts.astype(np.float32).reshape(1, -1)
    ).astype(np.float32)
    asm_norm = normalizer.normalize_assembly(
        assembly_feats.astype(np.float32).reshape(1, -1)
    ).astype(np.float32)

    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name
    feed = {input_names[0]: kmer_norm, input_names[1]: asm_norm}
    result = session.run([output_name], feed)
    pred = result[0][0]
    return float(pred[0]), float(pred[1])


# ============================================================================
# CheckM2 runner
# ============================================================================
def run_checkm2(fasta_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Run CheckM2 on all FASTAs in fasta_dir. Returns DataFrame with results."""
    checkm2_out = output_dir / 'checkm2_output'
    checkm2_out.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"conda run -n {CHECKM2_ENV} "
        f"env CHECKM2DB={CHECKM2_DB} "
        f"checkm2 predict --threads {CHECKM2_THREADS} "
        f"-x .fasta "
        f"--input {fasta_dir} "
        f"--output-directory {checkm2_out} "
        f"--force"
    )
    print(f"\nRunning CheckM2:\n  {cmd}\n")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"CheckM2 completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"CheckM2 STDERR:\n{result.stderr}")
        print(f"CheckM2 STDOUT:\n{result.stdout}")
        raise RuntimeError(f"CheckM2 failed with return code {result.returncode}")

    # Parse CheckM2 output
    quality_report = checkm2_out / 'quality_report.tsv'
    if not quality_report.exists():
        raise FileNotFoundError(f"CheckM2 quality report not found: {quality_report}")

    df = pd.read_csv(quality_report, sep='\t')
    print(f"CheckM2 results: {len(df)} genomes assessed")
    return df


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("Script 62: Pathogen Synthetic Genome — L.m. DOMINANT, S.e. CONTAMINANT")
    print("=" * 80)
    print(f"Model: {ONNX_MODEL_PATH}")
    print(f"No fragmentation. 1-2 contigs per synthetic genome.")
    print(f"Replicates per config: {N_REPLICATES} (seeds {REPLICATE_SEEDS})")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FASTA_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load model, normalizer, kmer counter
    # ----------------------------------------------------------------
    print("Loading ONNX model...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider']
    )
    print(f"  Inputs: {[(i.name, i.shape) for i in session.get_inputs()]}")

    print("Loading normalizer...")
    normalizer = FeatureNormalizer.load(NORMALIZATION_PATH)
    print(f"  K-mer features: {normalizer.n_kmer_features}, "
          f"Assembly features: {normalizer.n_assembly_features}")

    print("Loading k-mer counter...")
    kmer_counter = KmerCounter(SELECTED_KMERS_PATH)
    print(f"  {kmer_counter.n_features} k-mers loaded")

    # Warm up Numba JIT
    print("Warming up Numba JIT...")
    dummy = ["ACGTACGTACGTACGTACGT" * 100]
    kmer_counter.count_contigs(dummy)

    # ----------------------------------------------------------------
    # Read reference genomes
    # ----------------------------------------------------------------
    print("\nReading reference genomes...")
    lm_seq = read_full_sequence(str(LM_GENOME))
    se_seq = read_full_sequence(str(SE_GENOME))
    lm_size = len(lm_seq)
    se_size = len(se_seq)
    print(f"  L. monocytogenes (DOMINANT):    {lm_size:>10,} bp ({lm_size/1e6:.3f} Mbp)")
    print(f"  S. enterica (CONTAMINANT):      {se_size:>10,} bp ({se_size/1e6:.3f} Mbp)")

    # ----------------------------------------------------------------
    # Phase 1: Generate all FASTAs + run MAGICC V5
    # ----------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Phase 1: Generate synthetic genomes + MAGICC V5 predictions")
    print(f"{'='*80}")

    all_config_results = []
    all_replicate_rows = []

    for config_name, comp_frac, cont_frac in GENOME_CONFIGS:
        print(f"\n{'─'*70}")
        print(f"Config: {config_name}  "
              f"(L.m. {comp_frac*100:.0f}% comp + S.e. {cont_frac*100:.0f}% cont)")
        print(f"{'─'*70}")

        # Compute expected bp
        expected_dom_bp = lm_size if comp_frac >= 1.0 else int(round(lm_size * comp_frac))
        expected_cont_bp = int(round(lm_size * cont_frac))
        print(f"  Expected dominant bp:     {expected_dom_bp:>10,}")
        print(f"  Expected contaminant bp:  {expected_cont_bp:>10,}")
        print(f"  Expected completeness:    {expected_dom_bp / lm_size * 100:.6f}%")
        print(f"  Expected contamination:   {expected_cont_bp / lm_size * 100:.6f}%")

        pred_comps = []
        pred_conts = []
        replicate_details = []

        for rep_idx, seed in enumerate(REPLICATE_SEEDS):
            contigs, dom_bp, cont_bp = generate_exact_synthetic(
                lm_seq, se_seq, lm_size, comp_frac, cont_frac, seed
            )

            true_comp = dom_bp / lm_size * 100.0
            true_cont = cont_bp / lm_size * 100.0

            # Verify exact counts
            assert dom_bp == expected_dom_bp, (
                f"Rep {rep_idx}: dominant {dom_bp} != expected {expected_dom_bp}"
            )
            assert cont_bp == expected_cont_bp, (
                f"Rep {rep_idx}: contaminant {cont_bp} != expected {expected_cont_bp}"
            )

            # Run V5 inference
            pred_comp, pred_cont = extract_and_predict(
                contigs, kmer_counter, normalizer, session
            )

            pred_comps.append(pred_comp)
            pred_conts.append(pred_cont)

            # Save FASTA for ALL replicates (for CheckM2 batch)
            fasta_name = f"{config_name}_rep{rep_idx}.fasta"
            fasta_path = FASTA_DIR / fasta_name
            fasta_entries = [("dominant_Lm", contigs[0])]
            if len(contigs) > 1:
                fasta_entries.append(("contaminant_Se", contigs[1]))
            write_fasta(str(fasta_path), fasta_entries)

            rep_detail = {
                'config': config_name,
                'replicate': rep_idx,
                'seed': seed,
                'dominant_bp': dom_bp,
                'contaminant_bp': cont_bp,
                'n_contigs': len(contigs),
                'true_completeness': round(true_comp, 6),
                'true_contamination': round(true_cont, 6),
                'pred_completeness': round(pred_comp, 4),
                'pred_contamination': round(pred_cont, 4),
            }
            replicate_details.append(rep_detail)
            all_replicate_rows.append(rep_detail.copy())

            print(f"  Rep {rep_idx} (seed={seed}): "
                  f"dom={dom_bp:,}bp cont={cont_bp:,}bp | "
                  f"V5 comp={pred_comp:.2f}% cont={pred_cont:.2f}%")

        pred_comps = np.array(pred_comps)
        pred_conts = np.array(pred_conts)

        true_comp_exact = expected_dom_bp / lm_size * 100.0
        true_cont_exact = expected_cont_bp / lm_size * 100.0

        config_result = {
            'config': config_name,
            'comp_frac': comp_frac,
            'cont_frac': cont_frac,
            'dominant_bp': expected_dom_bp,
            'contaminant_bp': expected_cont_bp,
            'dominant_genome_full_size': lm_size,
            'true_completeness': round(true_comp_exact, 6),
            'true_contamination': round(true_cont_exact, 6),
            'v5_comp_mean': round(float(pred_comps.mean()), 4),
            'v5_comp_std': round(float(pred_comps.std()), 4),
            'v5_cont_mean': round(float(pred_conts.mean()), 4),
            'v5_cont_std': round(float(pred_conts.std()), 4),
            'replicates': replicate_details,
        }
        all_config_results.append(config_result)

        print(f"\n  Summary for {config_name}:")
        print(f"    True comp: {true_comp_exact:.4f}%  |  True cont: {true_cont_exact:.4f}%")
        print(f"    V5 comp:  {pred_comps.mean():.2f} +/- {pred_comps.std():.2f}%")
        print(f"    V5 cont:  {pred_conts.mean():.2f} +/- {pred_conts.std():.2f}%")

    # ----------------------------------------------------------------
    # Phase 2: Run CheckM2 on all FASTAs
    # ----------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Phase 2: Run CheckM2 on all 50 synthetic genomes")
    print(f"{'='*80}")

    checkm2_df = run_checkm2(FASTA_DIR, OUTPUT_DIR)

    # Parse CheckM2 results and merge with config results
    # CheckM2 "Name" column is the filename without extension
    checkm2_map = {}
    for _, row in checkm2_df.iterrows():
        name = row['Name']
        checkm2_map[name] = {
            'comp': row['Completeness'],
            'cont': row['Contamination'],
        }

    # Add CheckM2 results to replicate rows
    for row in all_replicate_rows:
        fasta_name = f"{row['config']}_rep{row['replicate']}"
        ckm2 = checkm2_map.get(fasta_name, {})
        row['checkm2_completeness'] = ckm2.get('comp', np.nan)
        row['checkm2_contamination'] = ckm2.get('cont', np.nan)

    # Compute CheckM2 per-config mean/std
    for config_result in all_config_results:
        config_name = config_result['config']
        ckm2_comps = []
        ckm2_conts = []
        for rep_idx in range(N_REPLICATES):
            fasta_name = f"{config_name}_rep{rep_idx}"
            ckm2 = checkm2_map.get(fasta_name, {})
            if 'comp' in ckm2:
                ckm2_comps.append(ckm2['comp'])
                ckm2_conts.append(ckm2['cont'])

        if ckm2_comps:
            config_result['checkm2_comp_mean'] = round(float(np.mean(ckm2_comps)), 4)
            config_result['checkm2_comp_std'] = round(float(np.std(ckm2_comps)), 4)
            config_result['checkm2_cont_mean'] = round(float(np.mean(ckm2_conts)), 4)
            config_result['checkm2_cont_std'] = round(float(np.std(ckm2_conts)), 4)
        else:
            config_result['checkm2_comp_mean'] = None
            config_result['checkm2_comp_std'] = None
            config_result['checkm2_cont_mean'] = None
            config_result['checkm2_cont_std'] = None

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    tsv_path = OUTPUT_DIR / 'replicate_summary.tsv'
    pd.DataFrame(all_replicate_rows).to_csv(tsv_path, sep='\t', index=False)
    print(f"\nSaved replicate summary: {tsv_path}")

    # JSON details (remove replicates list for cleaner JSON, keep in separate file)
    json_path = OUTPUT_DIR / 'config_summary.json'
    json_results = []
    for r in all_config_results:
        jr = {k: v for k, v in r.items() if k != 'replicates'}
        json_results.append(jr)
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved config summary: {json_path}")

    json_detail_path = OUTPUT_DIR / 'replicate_details.json'
    with open(json_detail_path, 'w') as f:
        json.dump(all_config_results, f, indent=2)
    print(f"Saved replicate details: {json_detail_path}")

    # ----------------------------------------------------------------
    # Final summary table
    # ----------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print("FINAL RESULTS — EXACT Synthetic Pathogen Analysis")
    print("L. monocytogenes (DOMINANT) + S. enterica (CONTAMINANT)")
    print(f"{'='*100}")
    print(f"L. monocytogenes genome: {lm_size:,} bp  |  S. enterica genome: {se_size:,} bp")
    print()

    # BP verification table
    print("BP Verification:")
    print(f"  {'Config':<16} {'Dom BP':>12} {'Cont BP':>12} "
          f"{'True Comp%':>12} {'True Cont%':>12}")
    print(f"  {'-'*64}")
    for r in all_config_results:
        print(f"  {r['config']:<16} {r['dominant_bp']:>12,} {r['contaminant_bp']:>12,} "
              f"{r['true_completeness']:>12.4f} {r['true_contamination']:>12.4f}")

    # Results table
    print(f"\n{'Config':<16} {'True':>6} {'True':>6} "
          f"{'V5 Comp':>20} {'V5 Cont':>20} "
          f"{'CkM2 Comp':>20} {'CkM2 Cont':>20}")
    print(f"{'':16} {'Comp':>6} {'Cont':>6} "
          f"{'(mean+/-sd)':>20} {'(mean+/-sd)':>20} "
          f"{'(mean+/-sd)':>20} {'(mean+/-sd)':>20}")
    print("-" * 116)
    for r in all_config_results:
        v5c = f"{r['v5_comp_mean']:.2f}+/-{r['v5_comp_std']:.2f}"
        v5t = f"{r['v5_cont_mean']:.2f}+/-{r['v5_cont_std']:.2f}"
        if r['checkm2_comp_mean'] is not None:
            cm2c = f"{r['checkm2_comp_mean']:.2f}+/-{r['checkm2_comp_std']:.2f}"
            cm2t = f"{r['checkm2_cont_mean']:.2f}+/-{r['checkm2_cont_std']:.2f}"
        else:
            cm2c = "N/A"
            cm2t = "N/A"
        print(f"{r['config']:<16} {r['true_completeness']:>6.1f} {r['true_contamination']:>6.1f} "
              f"{v5c:>20} {v5t:>20} "
              f"{cm2c:>20} {cm2t:>20}")

    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("Done.")


if __name__ == '__main__':
    main()
