#!/usr/bin/env python3
"""
WS1.6 / WS1.9 step 3 - Retrain MAGICC with the held-out taxon panel removed.

Level is selected by MAGICC_HOLDOUT_LEVEL (phylum | family); all artifact paths
are level-suffixed, so the family model is models/magicc_holdout_family.onnx and
cannot collide with models/magicc_holdout_phylum.onnx.

THE HOLDOUT MODELS ARE VALIDATION ARTIFACTS ONLY. The released model remains
trained on all data.

Configuration is byte-for-byte the V5 winning recipe (scripts/53_train_v5_run3.py):
  MAGICCModel (V2 architecture, NO attention - attention was shown to hurt on V5 data)
  9,249 k-mer + 7 k-mer-summary features, 42,400,946 parameters
  AdamW lr 1e-3, weight_decay 5e-4
  CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
  WeightedMSE comp_weight=2.0 cont_weight=1.0
  batch 512, FP16 AMP + gradient checkpointing, grad clip 1.0
  augmentation: 2% k-mer masking + Gaussian noise sigma=0.01
  max 150 epochs, early stopping patience 20, checkpoint every 10 epochs

The ONLY difference from production V5 is the training/validation DATA, which comes
from data/features/holdout_phylum_features.h5 (script 121): identical size (1M/100K/100K)
and identical sample-type composition, but built from a genome pool with the 10 panel
phyla removed from BOTH the dominant and the contaminant role.

Artifacts are written to models/holdout/ so nothing in models/ is overwritten;
the final model is copied to models/best_model_holdout_phylum.pt and exported to
models/magicc_holdout_phylum.onnx (FP32, opset 17).

Usage
  python scripts/122_train_holdout_phylum.py                       # train (+ auto-export)
  python scripts/122_train_holdout_phylum.py --smoke                # 2-epoch E2E test
  python scripts/122_train_holdout_phylum.py --resume-auto          # resume latest ckpt
  python scripts/122_train_holdout_phylum.py --export-only
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from holdout_lib import config as C
from holdout_lib.model import MAGICCModel, build_model
from holdout_lib.seeding import DEFAULT_SEED, SeededTrainer, seed_everything

OUT = C.HOLDOUT_TRAIN_OUT


def export_onnx(pt_path, onnx_path, logf=print, h5=None):
    import onnx
    import onnxruntime as ort

    logf(f'\nExporting {pt_path} -> {onnx_path} (FP32, opset 17)')
    model = MAGICCModel(n_kmer_features=C.N_KMER_FEATURES,
                        n_assembly_features=C.N_SUMMARY_FEATURES,
                        use_gradient_checkpointing=False)
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    logf(f'  best_epoch={ck.get("best_epoch", ck.get("epoch"))} '
         f'best_val_loss={ck.get("best_val_loss")}')
    logf(f'  parameters={sum(p.numel() for p in model.parameters()):,}')

    torch.onnx.export(
        model, (torch.randn(1, C.N_KMER_FEATURES), torch.randn(1, C.N_SUMMARY_FEATURES)),
        str(onnx_path), export_params=True, opset_version=17, do_constant_folding=True,
        input_names=['kmer_features', 'assembly_features'], output_names=['predictions'],
        dynamic_axes={'kmer_features': {0: 'batch_size'},
                      'assembly_features': {0: 'batch_size'},
                      'predictions': {0: 'batch_size'}})
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    logf(f'  ONNX checker passed, {os.path.getsize(onnx_path)/1e6:.1f} MB')

    rng = np.random.default_rng(42)
    tk = rng.standard_normal((1000, C.N_KMER_FEATURES), dtype=np.float32)
    ta = rng.standard_normal((1000, C.N_SUMMARY_FEATURES), dtype=np.float32)
    with torch.no_grad():
        pt = model(torch.from_numpy(tk), torch.from_numpy(ta)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    on = sess.run(None, {'kmer_features': tk, 'assembly_features': ta})[0]
    mad = float(np.max(np.abs(pt - on)))
    logf(f'  max abs diff PyTorch vs ONNX (1000 random samples): {mad:.3e}')
    logf(f'  mean abs diff: {float(np.mean(np.abs(pt-on))):.3e}')
    ok = mad < 1e-4
    logf(f'  VERIFICATION {"PASSED" if ok else "FAILED"} (tolerance 1e-4)')

    # Also verify on real holdout test features
    import h5py
    with h5py.File(h5 or C.HOLDOUT_H5, 'r') as f:
        rk = f['test']['kmer_features'][:512].astype(np.float32)
        ra = f['test']['assembly_features'][:512].astype(np.float32)
    with torch.no_grad():
        pt2 = model(torch.from_numpy(rk), torch.from_numpy(ra)).numpy()
    on2 = sess.run(None, {'kmer_features': rk, 'assembly_features': ra})[0]
    mad2 = float(np.max(np.abs(pt2 - on2)))
    logf(f'  max abs diff on 512 real holdout-test samples: {mad2:.3e}')
    return {'onnx_path': str(onnx_path), 'max_abs_diff_random': mad,
            'max_abs_diff_real': mad2, 'verified': bool(ok and mad2 < 1e-4),
            'size_mb': round(os.path.getsize(onnx_path) / 1e6, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--resume-auto', action='store_true')
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--max-epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--deterministic', action='store_true',
                    help='force cuDNN determinism (slower; off by default)')
    a = ap.parse_args()

    out = OUT if not a.smoke else C.MODELS_DIR / f'holdout_{C.PANEL_LEVEL}_smoke'
    out.mkdir(parents=True, exist_ok=True)
    h5 = C.HOLDOUT_H5 if not a.smoke else C.HOLDOUT_DIR / 'smoke_features.h5'
    best_pt = out / 'best_model.pt'

    if a.export_only:
        r = export_onnx(best_pt, C.HOLDOUT_ONNX)
        (C.RESULTS_DIR / 'onnx_export_verification.json').write_text(json.dumps(r, indent=2))
        print(json.dumps(r, indent=2))
        return

    assert h5.exists(), f'missing {h5}'
    if not torch.cuda.is_available():
        sys.exit('ERROR: CUDA unavailable')

    seed_info = seed_everything(a.seed, deterministic_algorithms=a.deterministic)
    print('\nSEEDING (production V5 was trained with NO seed set - see R1 comment)')
    print(f'  seed = {a.seed}')
    for s in seed_info['seeded']:
        print(f'    + {s}')
    print(f'  cudnn.benchmark={seed_info["cudnn_benchmark"]} '
          f'cudnn.deterministic={seed_info["cudnn_deterministic"]}')
    print('  residual nondeterminism:')
    for s in seed_info['residual_nondeterminism']:
        print(f'    - {s}')
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'PyTorch {torch.__version__} / CUDA {torch.version.cuda}')

    import h5py
    with h5py.File(h5, 'r') as f:
        print(f'\ndata: {h5}')
        print(f'  normalized={f.attrs.get("normalized")}  '
              f'version={f.attrs.get("version")}')
        for s in ['train', 'val', 'test']:
            print(f'  {s}: {f[s]["kmer_features"].shape} '
                  f'asm={f[s]["assembly_features"].shape}')
        panel = json.loads(f.attrs['panel_taxa']) if 'panel_taxa' in f.attrs else (
            json.loads(f.attrs['panel_phyla']) if 'panel_phyla' in f.attrs else [])
        print(f'  panel_level: {f.attrs.get("panel_level", "phylum")}')
        print(f'  panel_taxa ({len(panel)}): {panel}')
        assert set(panel) == set(C.PANEL_TAXA), (
            'HDF5 panel does not match the configured panel - wrong '
            'MAGICC_HOLDOUT_LEVEL or a stale HDF5')
        assert bool(f.attrs.get('normalized', False)), 'HDF5 not normalized yet'

    model = build_model(n_kmer_features=C.N_KMER_FEATURES,
                        n_assembly_features=C.N_SUMMARY_FEATURES,
                        use_gradient_checkpointing=True, device='cuda')
    p = model.count_parameters()
    print(f'\nMAGICCModel (V2 arch, no attention): {p["total"]:,} params '
          f'(kmer {p["kmer_branch"]:,} / asm {p["assembly_branch"]:,} / '
          f'fusion {p["fusion_head"]:,})')
    assert p['total'] == 42_400_946, f'param mismatch: {p["total"]}'

    trainer = SeededTrainer(
        seed=a.seed,
        model=model, h5_path=str(h5), output_dir=str(out),
        lr=a.lr, weight_decay=5e-4, comp_weight=2.0, cont_weight=1.0,
        batch_size=a.batch_size, num_workers=a.num_workers,
        max_epochs=2 if a.smoke else a.max_epochs,
        patience=a.patience, grad_clip_norm=1.0, t_0=10, t_mult=2,
        mask_rate=0.02, noise_std=0.01, checkpoint_every=1 if a.smoke else 10,
        device='cuda')

    resume = a.resume
    if a.resume_auto and resume is None:
        cks = sorted(glob.glob(str(out / 'checkpoint_epoch_*.pt')),
                     key=lambda x: int(re.search(r'(\d+)\.pt$', x).group(1)))
        if cks:
            resume = cks[-1]
            print(f'auto-resuming from {resume}')
    if resume:
        trainer.load_checkpoint(resume)

    t0 = time.time()
    best = trainer.train()
    wall = time.time() - t0
    print(f'\ntraining wall time {wall/3600:.2f} h')

    for src, dst in [('best_model.pt', C.HOLDOUT_BEST_PT),
                     ('training_history.json', C.HOLDOUT_HISTORY),
                     ('training_curves.png',
                      C.MODELS_DIR / f'training_curves_holdout_{C.PANEL_LEVEL}.png')]:
        s = out / src
        if s.exists() and not a.smoke:
            shutil.copy2(s, dst)
            print(f'  {s} -> {dst}')

    # record seeds inside the history JSON as well
    hist_path = out / 'training_history.json'
    if hist_path.exists():
        hj = json.loads(hist_path.read_text())
        hj = hj if isinstance(hj, dict) else {'history': hj}
        hj['seeding'] = seed_info
        hj['data'] = str(h5)
        hj['panel_level'] = C.PANEL_LEVEL
        hj['panel_taxa'] = C.PANEL_TAXA
        hj['panel_phyla'] = C.PANEL_PHYLA
        hist_path.write_text(json.dumps(hj, indent=2))

    summary = {'wall_hours': round(wall / 3600, 3), 'best_metrics': best,
               'seeding': seed_info,
               'config': {'arch': 'MAGICCModel V2 (no attention)',
                          'params': p['total'], 'lr': a.lr, 'weight_decay': 5e-4,
                          'batch_size': a.batch_size, 'comp_weight': 2.0,
                          'cont_weight': 1.0, 'scheduler': 'CosineAnnealingWarmRestarts'
                          ' T_0=10 T_mult=2', 'mask_rate': 0.02, 'noise_std': 0.01,
                          'patience': a.patience, 'max_epochs': a.max_epochs,
                          'amp': 'fp16', 'grad_checkpointing': True},
               'data': str(h5), 'panel_level': C.PANEL_LEVEL,
               'panel_taxa': C.PANEL_TAXA, 'panel_phyla': C.PANEL_PHYLA,
               'panel_groups': {g: d['taxa'] for g, d in C.PANEL_GROUPS.items()},
               'status': ('VALIDATION ARTIFACT ONLY - the released MAGICC model '
                          'remains trained on all data')}
    print('\n' + '=' * 70)
    print('HOLDOUT MODEL - BEST VALIDATION METRICS (in-distribution, panel-free val)')
    print('=' * 70)
    for k, v in best.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    print('\nProduction V5 reference (val = full-phylum): comp_mae 3.795  cont_mae 4.144'
          '  comp_r2 0.838  cont_r2 0.933')

    if not a.smoke:
        summary['onnx'] = export_onnx(best_pt, C.HOLDOUT_ONNX)
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (C.RESULTS_DIR / 'training_summary.json').write_text(json.dumps(summary, indent=2))
        print(f'\nwrote {C.RESULTS_DIR / "training_summary.json"}')
    else:
        print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
