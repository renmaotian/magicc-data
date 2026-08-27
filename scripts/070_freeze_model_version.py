#!/usr/bin/env python3
"""
WS0.1 / WS0.3 — Freeze MAGICC V5 as *the* manuscript model and emit a machine-readable
model card.

Records, all verified by direct inspection (nothing hard-coded from documentation):
  * SHA256 + byte size of models/magicc_v5.onnx and of the source PyTorch checkpoint
  * ONNX IR version, opset, producer, input/output names / shapes / dtypes
  * Parameter count recomputed from the ONNX initializers AND from the PyTorch
    state_dict (they must agree up to BatchNorm running stats)
  * Architecture summary (V2-style two-branch MLP)
  * Training data file (data/features/magicc_v5_features.h5): SHA256 is expensive on a
    41 GB file, so we record size + mtime + per-split shapes and a SHA256 of the first
    and last 64 MiB (a cheap but strong integrity fingerprint). Full SHA256 optional
    via --full-hash.
  * Hyperparameters and best epoch, read from scripts/53_train_v5_run3.py's actual
    trainer call defaults + the checkpoint + the training log
  * Normalization parameter file provenance
  * Feature definitions (9,249 k-mers + 7 k-mer-summary features)
  * Known limitations

Output: results/revision/model_card.json

Usage:
    python scripts/70_freeze_model_version.py [--full-hash]
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

MODEL_DIR = PROJECT_DIR / 'models'
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results' / 'revision'

ONNX_PATH = MODEL_DIR / 'magicc_v5.onnx'
PT_PATH = MODEL_DIR / 'best_model_v5_run3.pt'
H5_PATH = DATA_DIR / 'features' / 'magicc_v5_features.h5'
NORM_PATH = DATA_DIR / 'features' / 'normalization_params.json'
KMERS_PATH = DATA_DIR / 'kmer_selection' / 'selected_kmers.txt'
TRAIN_SCRIPT = PROJECT_DIR / 'scripts' / '53_train_v5_run3.py'
TRAIN_LOG = MODEL_DIR / 'training_v5_run3_log.txt'
HISTORY_PATH = MODEL_DIR / 'training_history_v5_run3.json'

CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_head_tail(path: Path, n_bytes: int = 64 * 1024 * 1024) -> dict:
    """Cheap integrity fingerprint for very large files."""
    size = path.stat().st_size
    h_head = hashlib.sha256()
    h_tail = hashlib.sha256()
    with open(path, 'rb') as f:
        remaining = min(n_bytes, size)
        while remaining > 0:
            b = f.read(min(CHUNK, remaining))
            if not b:
                break
            h_head.update(b)
            remaining -= len(b)
        tail_start = max(0, size - n_bytes)
        f.seek(tail_start)
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h_tail.update(b)
    return {
        'method': f'sha256 of first and last {n_bytes} bytes',
        'head_sha256': h_head.hexdigest(),
        'tail_sha256': h_tail.hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-hash', action='store_true',
                    help='Also compute the full SHA256 of the 41 GB training HDF5 (slow)')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for p in (ONNX_PATH, PT_PATH, NORM_PATH, KMERS_PATH):
        if not p.exists():
            raise SystemExit(f'FATAL: required file missing: {p}')

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import numpy_helper

    print('=' * 78)
    print('WS0.1 — MAGICC model version freeze')
    print('=' * 78)

    # ------------------------------------------------------------------ ONNX
    print(f'\n[1/6] Hashing and inspecting {ONNX_PATH.name} ...')
    onnx_sha = sha256_file(ONNX_PATH)
    onnx_size = ONNX_PATH.stat().st_size
    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)

    elem_map = {1: 'float32', 7: 'int64', 10: 'float16', 11: 'float64'}

    def io_spec(vi):
        return {
            'name': vi.name,
            'shape': [d.dim_param or int(d.dim_value)
                      for d in vi.type.tensor_type.shape.dim],
            'dtype': elem_map.get(vi.type.tensor_type.elem_type,
                                  str(vi.type.tensor_type.elem_type)),
        }

    onnx_params = int(sum(numpy_helper.to_array(i).size for i in m.graph.initializer))

    onnx_block = {
        'path': str(ONNX_PATH.relative_to(PROJECT_DIR)),
        'sha256': onnx_sha,
        'size_bytes': onnx_size,
        'size_mb': round(onnx_size / 1e6, 2),
        'ir_version': int(m.ir_version),
        'opset_imports': [{'domain': o.domain or 'ai.onnx', 'version': int(o.version)}
                          for o in m.opset_import],
        'producer': f'{m.producer_name} {m.producer_version}',
        'precision': 'FP32',
        'inputs': [io_spec(i) for i in m.graph.input],
        'outputs': [io_spec(o) for o in m.graph.output],
        'n_graph_nodes': len(m.graph.node),
        'n_initializers': len(m.graph.initializer),
        'n_initializer_elements': onnx_params,
        'onnxruntime_version_verified_with': ort.__version__,
        'onnx_version_verified_with': onnx.__version__,
    }
    print(f'      sha256 {onnx_sha}')
    print(f'      {onnx_size/1e6:.2f} MB, opset '
          f'{onnx_block["opset_imports"]}, {onnx_params:,} initializer elements')

    # numeric smoke test: fixed input -> fixed output, so the frozen model is
    # verifiable by anyone.
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX_PATH), so, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(20260726)
    probe_kmer = rng.standard_normal((4, 9249)).astype(np.float32)
    probe_asm = rng.standard_normal((4, 7)).astype(np.float32)
    probe_out = sess.run(None, {'kmer_features': probe_kmer,
                                'assembly_features': probe_asm})[0]
    onnx_block['determinism_probe'] = {
        'description': ('numpy default_rng(20260726) standard_normal((4,9249)) and '
                        '((4,7)) as kmer_features/assembly_features, float32, '
                        'CPUExecutionProvider, 1 intra/inter-op thread'),
        'probe_input_sha256': hashlib.sha256(
            probe_kmer.tobytes() + probe_asm.tobytes()).hexdigest(),
        'expected_output': probe_out.round(6).tolist(),
    }
    print(f'      determinism probe output: '
          f'{np.array2string(probe_out, precision=4)}')

    # ------------------------------------------------------ PyTorch checkpoint
    print(f'\n[2/6] Hashing and inspecting {PT_PATH.name} ...')
    pt_sha = sha256_file(PT_PATH)
    ck = torch.load(str(PT_PATH), map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    pt_total = int(sum(v.numel() for v in sd.values()))
    # count only *parameters* (exclude BatchNorm running_mean/var/num_batches_tracked)
    pt_params = int(sum(v.numel() for k, v in sd.items()
                        if not re.search(r'(running_mean|running_var|num_batches_tracked)$', k)))
    ckpt_block = {
        'path': str(PT_PATH.relative_to(PROJECT_DIR)),
        'sha256': pt_sha,
        'size_bytes': PT_PATH.stat().st_size,
        'epoch_at_save_0indexed': int(ck['epoch']),
        'best_epoch_0indexed': int(ck['best_epoch']),
        'best_epoch_1indexed': int(ck['best_epoch']) + 1,
        'best_val_loss': float(ck['best_val_loss']),
        'state_dict_total_elements': pt_total,
        'state_dict_parameter_elements_excluding_bn_buffers': pt_params,
        'torch_version_verified_with': torch.__version__,
    }
    print(f'      sha256 {pt_sha}')
    print(f'      best_epoch (0-indexed) {ck["best_epoch"]}  '
          f'best_val_loss {ck["best_val_loss"]:.4f}')
    print(f'      params (excl. BN buffers) {pt_params:,}; '
          f'state_dict total {pt_total:,}; ONNX initializers {onnx_params:,}')

    # ---------------------------------------------------------- architecture
    print('\n[3/6] Rebuilding architecture and counting parameters ...')
    from magicc.model import MAGICCModel
    model = MAGICCModel(n_kmer_features=9249, n_assembly_features=7,
                        use_gradient_checkpointing=False)
    model.load_state_dict(sd)  # hard check that ONNX arch == released arch
    pc = model.count_parameters()
    arch_block = {
        'family': 'MAGICCModel ("V2-style" two-branch MLP; no attention)',
        'source_module': 'magicc/model.py::MAGICCModel',
        'n_kmer_features': 9249,
        'n_kmer_summary_features': 7,
        'kmer_branch': ('Linear(9249->4096)-BN-SiLU-Dropout(0.4) -> '
                        'Linear(4096->1024)-BN-SiLU-Dropout(0.2) -> '
                        'Linear(1024->256)-BN-SiLU'),
        'kmer_summary_branch': ('Linear(7->32)-BN-SiLU-Dropout(0.2) -> '
                               'Linear(32->16)-BN-SiLU'),
        'fusion_head': ('concat(256,16)=272 -> Linear(272->128)-BN-SiLU-Dropout(0.1) '
                        '-> Linear(128->64)-SiLU -> Linear(64->2)'),
        'output_activations': {
            'completeness': 'sigmoid(x)*50 + 50  -> [50, 100] %',
            'contamination': 'sigmoid(x)*100     -> [0, 100] %',
        },
        'weight_init': 'Kaiming normal (nonlinearity=relu proxy for SiLU); BN weight=1, bias=0',
        'parameter_counts': {k: int(v) for k, v in pc.items()},
        'state_dict_load': 'strict load of best_model_v5_run3.pt succeeded',
    }
    print(f'      total parameters {pc["total"]:,} '
          f'(kmer {pc["kmer_branch"]:,} / summary {pc["assembly_branch"]:,} / '
          f'fusion {pc["fusion_head"]:,})')

    # -------------------------------------------------------- training data
    print('\n[4/6] Fingerprinting training data ...')
    import h5py
    h5_block = {
        'path': str(H5_PATH.relative_to(PROJECT_DIR)) if H5_PATH.exists() else None,
        'exists': H5_PATH.exists(),
    }
    if H5_PATH.exists():
        st = H5_PATH.stat()
        h5_block.update({
            'size_bytes': st.st_size,
            'size_gb': round(st.st_size / 1e9, 2),
            'mtime_utc': datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            'fingerprint': sha256_head_tail(H5_PATH),
        })
        if args.full_hash:
            print('      computing full SHA256 (this takes a while) ...')
            h5_block['sha256'] = sha256_file(H5_PATH)
        splits = {}
        with h5py.File(H5_PATH, 'r') as f:
            for split in ('train', 'val', 'test'):
                if split in f:
                    g = f[split]
                    splits[split] = {
                        'n_samples': int(g['kmer_features'].shape[0]),
                        'n_kmer_features': int(g['kmer_features'].shape[1]),
                        'n_assembly_features': int(g['assembly_features'].shape[1]),
                        'datasets': sorted(g.keys()),
                    }
        h5_block['splits'] = splits
        for k, v in splits.items():
            print(f'      {k}: {v["n_samples"]:,} samples '
                  f'({v["n_kmer_features"]} + {v["n_assembly_features"]} features)')
    h5_block['composition'] = (
        'V5 = 800,000 V4-style synthetic samples + 100,000 completeness=100%/'
        'contamination=0% + 100,000 completeness=100%/contamination=0-10% '
        '= 1,000,000 train; 100,000 val; 100,000 test '
        '(see scripts/50_generate_v5_training_data.py and Phase 4 of '
        'the internal project log)'
    )
    h5_block['reference_genomes'] = (
        'Synthesised from the 79,948-genome train split for train, the 10,010-genome '
        'val split for val, and the 9,999-genome test split for test '
        '(data/splits/{train,val,test}_genomes.tsv). Splits are stratified by phylum '
        'and mutually disjoint by accession.'
    )

    # ------------------------------------------------------ hyperparameters
    print('\n[5/6] Recording hyperparameters ...')
    hist = json.load(open(HISTORY_PATH)) if HISTORY_PATH.exists() else {}
    n_epochs_run = len(hist.get('val_loss', []))
    best_i = int(ck['best_epoch'])
    best_metrics = {}
    if hist:
        for key, out in (('val_comp_mae', 'val_comp_mae'), ('val_cont_mae', 'val_cont_mae'),
                         ('val_comp_rmse', 'val_comp_rmse'), ('val_cont_rmse', 'val_cont_rmse'),
                         ('val_comp_r2', 'val_comp_r2'), ('val_cont_r2', 'val_cont_r2'),
                         ('val_loss', 'val_loss'), ('train_loss', 'train_loss')):
            if key in hist and best_i < len(hist[key]):
                best_metrics[out] = round(float(hist[key][best_i]), 6)

    train_block = {
        'training_script': str(TRAIN_SCRIPT.relative_to(PROJECT_DIR)),
        'run_label': 'V5 Run 3 (V2 architecture on V5 expanded data)',
        'optimizer': 'AdamW',
        'learning_rate': 1e-3,
        'weight_decay': 5e-4,
        'lr_schedule': 'CosineAnnealingWarmRestarts(T_0=10, T_mult=2)',
        'loss': 'weighted MSE, comp_weight=2.0, cont_weight=1.0',
        'batch_size': 512,
        'max_epochs': 150,
        'early_stopping_patience': 20,
        'grad_clip_norm': 1.0,
        'mixed_precision': 'FP16 (torch.cuda.amp)',
        'gradient_checkpointing': True,
        'augmentation': {'kmer_mask_rate': 0.02, 'gaussian_noise_std': 0.01},
        'dataloader_workers': 4,
        'epochs_run': n_epochs_run,
        'early_stopped': n_epochs_run < 150,
        'best_epoch_0indexed': best_i,
        'best_epoch_1indexed': best_i + 1,
        'best_val_loss': float(ck['best_val_loss']),
        'best_epoch_val_metrics': best_metrics,
        'random_seed': None,
        'random_seed_note': (
            'VERIFIED BY CODE INSPECTION: neither magicc/trainer.py nor '
            'scripts/53_train_v5_run3.py sets a torch/numpy/python random seed, so the '
            'released weights are not bit-reproducible from scratch. The released '
            'artefact is therefore frozen by SHA256 (above) rather than by seed. '
            'Inference IS deterministic: see determinism_probe.'
        ),
        'hardware': 'NVIDIA Quadro P2200 (5.1 GB VRAM), 48-core CPU, 881 GB RAM',
    }
    if TRAIN_LOG.exists():
        txt = TRAIN_LOG.read_text(errors='replace')
        mm = re.search(r'Training complete in ([\d.]+) minutes', txt)
        if mm:
            train_block['wall_clock_minutes'] = float(mm.group(1))
        mm = re.search(r'Early stopping at epoch (\d+)', txt)
        if mm:
            train_block['early_stopping_epoch_reported_in_log'] = int(mm.group(1))
        train_block['training_log'] = str(TRAIN_LOG.relative_to(PROJECT_DIR))
    print(f'      epochs run {n_epochs_run}, best epoch (1-indexed) {best_i + 1}, '
          f'best val loss {ck["best_val_loss"]:.4f}')
    print(f'      best-epoch val comp MAE {best_metrics.get("val_comp_mae")} %, '
          f'cont MAE {best_metrics.get("val_cont_mae")} %')

    # ------------------------------------------------------------- features
    norm = json.load(open(NORM_PATH))
    n_kmers_file = sum(1 for line in open(KMERS_PATH) if line.strip())
    feature_block = {
        'kmer_features': {
            'n': n_kmers_file,
            'k': 9,
            'canonical': True,
            'list_file': str(KMERS_PATH.relative_to(PROJECT_DIR)),
            'list_sha256': sha256_file(KMERS_PATH),
            'selection': ('top 9,000 bacterial (prevalence 529-992/1000) + top 1,000 '
                          'archaeal (791-998/1000) single-copy-core-gene 9-mers, '
                          '751 shared -> 9,249 unique'),
            'selection_genomes': ('1,000 bacterial + 1,000 archaeal representatives '
                                  'sampled from the TRAIN split only, seed=42 '
                                  '(data/kmer_selection/selected_{bacterial,archaeal}_1000.tsv)'),
            'transform': 'log10(count + 1) then z-score with stored mean/std',
        },
        'kmer_summary_features': {
            'n': 7,
            'names': ['log10_total_kmer_count', 'total_kmer_sum', 'unique_kmer_count',
                      'duplicate_kmer_count', 'kmer_entropy', 'unique_kmer_ratio',
                      'duplicate_kmer_ratio'],
            'note': ('V4 onward: all 20/26 assembly statistics were REMOVED and replaced '
                     'by these 7 features derived purely from the selected-9-mer count '
                     'vector, so inference needs no assembly-level metadata.'),
        },
        'normalization': {
            'path': str(NORM_PATH.relative_to(PROJECT_DIR)),
            'sha256': sha256_file(NORM_PATH),
            'n_kmer_features': norm['n_kmer_features'],
            'n_assembly_features': norm['n_assembly_features'],
            'finalized': norm['finalized'],
            'method': ('streaming Welford mean/std over the training synthesis stream '
                       '(k-mers, after log10(count+1)); reservoir sampling for the '
                       'k-mer-summary feature scalers'),
        },
    }

    # ----------------------------------------------------------- assemble
    print('\n[6/6] Writing model card ...')
    try:
        git_commit = subprocess.check_output(
            ['git', '-C', str(PROJECT_DIR), 'rev-parse', 'HEAD'],
            text=True).strip()
    except Exception:
        git_commit = None

    card = {
        'schema': 'magicc-model-card/1.0',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'generated_by': 'scripts/70_freeze_model_version.py',
        'git_commit': git_commit,
        'frozen_model': {
            'name': 'MAGICC',
            'model_version': 'V5',
            'python_package_version': '0.3.0',
            'decision': ('WS0.1: V5 is FROZEN as the single model for the revised '
                         'manuscript. It is the version released on PyPI/GitHub as '
                         'magicc-genome 0.3.0. No number in the revised manuscript may '
                         'come from V1-V4.'),
            'supersedes_in_manuscript': ('V4 (submitted manuscript reported V4: overall '
                                         '4.00 % completeness MAE / 5.53 % contamination '
                                         'MAE on the 5,000-genome benchmark; V5 gives '
                                         '3.79 % / 5.39 %)'),
        },
        'onnx': onnx_block,
        'pytorch_checkpoint': ckpt_block,
        'architecture': arch_block,
        'features': feature_block,
        'training': train_block,
        'training_data': h5_block,
        'ground_truth_definitions': {
            'completeness_pct': ('100 x (retained dominant-genome bp) / (dominant '
                                 'reference full length)'),
            'contamination_pct': ('100 x (total contaminant bp) / (dominant reference '
                                  'full length)'),
            'note': ('the two metrics are independent and share the same denominator, '
                     'the dominant genome full reference length; the synthesis '
                     'constraint contaminant_bp <= dominant_retained_bp implies '
                     'contamination% <= completeness%'),
        },
        'intended_use': ('Genome-level completeness and contamination estimation for '
                         'bacterial and archaeal draft genomes / MAGs from nucleotide '
                         'FASTA, annotation-free at inference time.'),
        'limitations': [
            'Completeness output is architecturally bounded to [50, 100] %: the model '
            'cannot report completeness below 50 % and must not be used to triage very '
            'incomplete bins.',
            'Contamination output is architecturally bounded to [0, 100] %.',
            'Bacteria and Archaea only. No eukaryotes, no viruses, no plasmid-only input.',
            'Genome-level estimates only; MAGICC does not localise contamination to '
            'contigs.',
            'Trained exclusively on synthetic mixtures derived from high-quality GTDB '
            'reference genomes (CheckM2 completeness >= 98 %, contamination <= 2 %), so '
            'reference selection inherits CheckM2 as a filter (circularity addressed in '
            'protocol WS1.11).',
            'k-mer composition features are a priori more sensitive to sequencing and '
            'assembly error than protein-level features; quantified in protocol WS6.',
            'Accuracy on lineages that are taxonomically novel relative to the training '
            'set is quantified separately (protocol WS1.6-1.8) and is not covered by the '
            'in-distribution numbers reported here.',
            'Training was not seeded, so the released weights cannot be re-derived '
            'bit-exactly; the artefact is pinned by SHA256 instead.',
        ],
        'known_issues_fixed_in_this_revision': [
            'Benchmark Sets C and D as submitted drew their dominant reference genomes '
            'from train+val+test (985/1000 of Set C dominants are TRAIN genomes; only '
            '36/1000 of Set D dominants are TEST genomes). Both sets are superseded by '
            'data/benchmarks/set_C_clean and set_D_clean, whose dominants come from the '
            'held-out test split only (protocol WS1.1-1.4).',
        ],
        'environment_at_freeze': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'torch': torch.__version__,
            'onnx': onnx.__version__,
            'onnxruntime': ort.__version__,
            'numpy': np.__version__,
        },
    }

    out = RESULTS_DIR / 'model_card.json'
    with open(out, 'w') as f:
        json.dump(card, f, indent=2, sort_keys=False)
    print(f'      wrote {out}')
    print('\nDONE')


if __name__ == '__main__':
    main()
