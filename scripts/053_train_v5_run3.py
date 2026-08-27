#!/usr/bin/env python3
"""
Train MAGICC V5 Run 3: V2 architecture (MAGICCModel, no attention) on V5 expanded data.

Purpose: Isolate the effect of the expanded V5 training data from the V3 architecture.
If Run 3 beats V4, the new data helps and we can test V3 attention on top.
If Run 3 does NOT beat V4, the data alone doesn't explain differences.

Configuration (identical to V4 training):
- Model: MAGICCModel (V2 arch, no attention) via build_model()
- Data: V5 expanded (1M train, 100K val, 100K test)
- n_kmer_features=9249, n_assembly_features=7
- AdamW, lr=1e-3, weight_decay=5e-4
- WeightedMSE: comp_weight=2.0, cont_weight=1.0
- CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
- batch_size=512, max_epochs=150, patience=20
- grad_clip_norm=1.0, FP16, gradient checkpointing
- mask_rate=0.02, noise_std=0.01
- checkpoint_every=10

Usage:
    python scripts/53_train_v5_run3.py
    python scripts/53_train_v5_run3.py --resume models/checkpoint_v5r3_epoch_050.pt
"""

import sys
import os
import argparse
import shutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import build_model
from magicc.trainer import MAGICCTrainer


def main():
    parser = argparse.ArgumentParser(description='Train MAGICC V5 Run 3 (V2 arch + V5 data)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    # Paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_path = os.path.join(project_dir, 'data', 'features', 'magicc_v5_features.h5')
    output_dir = os.path.join(project_dir, 'models')

    # Verify V5 HDF5 exists
    if not os.path.exists(h5_path):
        print(f"ERROR: V5 HDF5 not found at {h5_path}")
        sys.exit(1)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Verify V5 HDF5 data
    print(f"\nVerifying V5 HDF5 data: {h5_path}")
    import h5py
    with h5py.File(h5_path, 'r') as f:
        for split in ['train', 'val', 'test']:
            grp = f[split]
            n = grp['kmer_features'].shape[0]
            n_kmer = grp['kmer_features'].shape[1]
            n_asm = grp['assembly_features'].shape[1]
            print(f"  {split}: {n:,} samples, kmer={n_kmer}, asm={n_asm}")
        print("  V5 HDF5 verified OK")

    # Build V2 model (MAGICCModel, no attention) -- same as V4
    print("\nBuilding MAGICCModel (V2 arch, no attention)...")
    model = build_model(
        n_kmer_features=9249,
        n_assembly_features=7,
        use_gradient_checkpointing=True,
        device=device,
    )
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} total")
    print(f"  K-mer branch: {params['kmer_branch']:,}")
    print(f"  Assembly branch: {params['assembly_branch']:,}")
    print(f"  Fusion head: {params['fusion_head']:,}")

    # Create trainer -- identical config to V4 (scripts/20_train_model.py)
    trainer = MAGICCTrainer(
        model=model,
        h5_path=h5_path,
        output_dir=output_dir,
        lr=args.lr,
        weight_decay=5e-4,
        comp_weight=2.0,
        cont_weight=1.0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip_norm=1.0,
        t_0=10,
        t_mult=2,
        mask_rate=0.02,
        noise_std=0.01,
        checkpoint_every=10,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Print training config
    print(f"\nV5 Run 3 Training Config:")
    print(f"  Data: V5 expanded (1M train, 100K val, 100K test)")
    print(f"  Model: MAGICCModel (V2 arch, no attention)")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: 5e-4")
    print(f"  LR schedule: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)")
    print(f"  Loss weights: comp=2.0, cont=1.0")
    print(f"  Augmentation: mask_rate=0.02, noise_std=0.01")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Gradient clipping: max_norm=1.0")
    print(f"  Mixed precision: FP16")
    print(f"  Gradient checkpointing: enabled")

    # Train!
    best_metrics = trainer.train()

    # Copy best_model.pt to best_model_v5_run3.pt
    src = os.path.join(output_dir, 'best_model.pt')
    dst = os.path.join(output_dir, 'best_model_v5_run3.pt')
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"\nCopied best model to: {dst}")
    else:
        print(f"\nWARNING: {src} not found, cannot copy to {dst}")

    # Also copy training curves and history with run3 suffix
    for fname, dst_fname in [
        ('training_curves.png', 'training_curves_v5_run3.png'),
        ('training_history.json', 'training_history_v5_run3.json'),
    ]:
        src_f = os.path.join(output_dir, fname)
        dst_f = os.path.join(output_dir, dst_fname)
        if os.path.exists(src_f):
            shutil.copy2(src_f, dst_f)

    # Print final results
    print("\n" + "=" * 70)
    print("V5 RUN 3 FINAL RESULTS (V2 arch + V5 data)")
    print("=" * 70)
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Compare with V4 and previous V5 runs
    print("\n" + "-" * 70)
    print("COMPARISON: V4 vs V5R1 vs V5R2 vs V5R3")
    print("-" * 70)
    v4_results = {
        'comp_mae': 3.86, 'cont_mae': 4.13,
        'comp_rmse': 6.38, 'cont_rmse': 7.07,
        'comp_r2': 0.837, 'cont_r2': 0.934,
    }
    v5r1_results = {
        'comp_mae': 3.96, 'cont_mae': 4.20,
        'comp_rmse': 6.54, 'cont_rmse': 7.21,
        'comp_r2': 0.828, 'cont_r2': 0.931,
    }
    v5r2_results = {
        'comp_mae': 4.05, 'cont_mae': 4.34,
        'comp_rmse': 6.67, 'cont_rmse': 7.40,
        'comp_r2': 0.822, 'cont_r2': 0.927,
    }

    print(f"  {'Metric':<12} {'V4':>8} {'V5R1':>8} {'V5R2':>8} {'V5R3':>8} {'R3 vs V4':>12}")
    print(f"  {'-'*60}")
    for metric in ['comp_mae', 'cont_mae', 'comp_rmse', 'cont_rmse', 'comp_r2', 'cont_r2']:
        v4_val = v4_results.get(metric, float('nan'))
        v5r1_val = v5r1_results.get(metric, float('nan'))
        v5r2_val = v5r2_results.get(metric, float('nan'))
        v5r3_val = best_metrics.get(metric, float('nan'))
        delta = v5r3_val - v4_val
        if metric.endswith('r2'):
            better = "BETTER" if v5r3_val > v4_val else "WORSE"
        else:
            better = "BETTER" if v5r3_val < v4_val else "WORSE"
        print(f"  {metric:<12} {v4_val:>8.4f} {v5r1_val:>8.4f} {v5r2_val:>8.4f} "
              f"{v5r3_val:>8.4f} {delta:>+8.4f} {better}")

    # Decision logic for Run 4
    comp_mae_r3 = best_metrics.get('comp_mae', float('inf'))
    cont_mae_r3 = best_metrics.get('cont_mae', float('inf'))
    beats_v4 = comp_mae_r3 < 3.86

    print(f"\n  V5R3 comp_mae = {comp_mae_r3:.4f} (V4 = 3.86)")
    if beats_v4:
        print("  VERDICT: V5R3 BEATS V4 -- V5 data helps! Proceed to Run 4 with V3 attention.")
    else:
        print("  VERDICT: V5R3 does NOT beat V4 -- data alone insufficient.")
        print("  Proceed to Run 4 with V3 attention to test if architecture helps.")

    print(f"\nBest model saved to: {dst}")
    print(f"Training curves: {output_dir}/training_curves_v5_run3.png")


if __name__ == '__main__':
    main()
