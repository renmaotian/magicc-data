#!/usr/bin/env python3
"""
Train MAGICC V5 model with attention mechanism on expanded V5 training dataset.

V5 uses:
- MAGICCModelV3 architecture (SE attention in k-mer branch + cross-attention fusion)
- Expanded training data: 1M samples (800K original + 100K pure complete + 100K low-contamination)
- Same val/test sets as V4 (100K each)
- 7 assembly features, 9249 k-mer features

Training config (based on V3 Run 3 best + V4):
- AdamW, lr=1e-3, weight_decay=5e-4
- WeightedMSE: comp_weight=2.0, cont_weight=1.0
- CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- Batch size 512, max 150 epochs, patience 20
- Gradient clipping max_norm=1.0, FP16, gradient checkpointing
- Data augmentation: mask_rate=0.02, noise_std=0.01

Usage:
    python scripts/051_train_v5_model.py
    python scripts/051_train_v5_model.py --resume models/best_model.pt
"""

import sys
import os
import argparse
import time
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import build_model_v3
from magicc.trainer import MAGICCTrainer


def main():
    parser = argparse.ArgumentParser(description='Train MAGICC V5 attention model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--se-reduction', type=int, default=16,
                        help='SE block reduction ratio')
    parser.add_argument('--n-attn-heads', type=int, default=4,
                        help='Number of cross-attention heads')
    parser.add_argument('--n-attn-groups', type=int, default=16,
                        help='Number of groups for k-mer token splitting')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--t0', type=int, default=10,
                        help='CosineAnnealingWarmRestarts T_0')
    parser.add_argument('--t-mult', type=int, default=2,
                        help='CosineAnnealingWarmRestarts T_mult')
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

    # Build V3 model with attention
    print("\nBuilding MAGICCModelV3 (SE attention + cross-attention fusion)...")
    model = build_model_v3(
        n_kmer_features=9249,
        n_assembly_features=7,
        use_gradient_checkpointing=True,
        device=device,
        se_reduction=args.se_reduction,
        n_attn_heads=args.n_attn_heads,
        n_attn_groups=args.n_attn_groups,
    )
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} total")
    print(f"  K-mer branch (with SE): {params['kmer_branch']:,}")
    print(f"  Assembly branch: {params['assembly_branch']:,}")
    print(f"  Fusion head (with cross-attn): {params['fusion_head']:,}")

    # GPU memory profile
    torch.cuda.reset_peak_memory_stats()
    model.train()
    dummy_kmer = torch.randn(args.batch_size, 9249, device=device)
    dummy_asm = torch.randn(args.batch_size, 7, device=device)
    with torch.amp.autocast('cuda'):
        out = model(dummy_kmer, dummy_asm)
        loss = out.mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"\nGPU memory profile (batch_size={args.batch_size}):")
    print(f"  Peak memory: {peak:.0f} MB / {gpu_mem*1000:.0f} MB ({peak/gpu_mem/10:.1f}%)")
    del dummy_kmer, dummy_asm, out, loss
    torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)

    # Create trainer
    trainer = MAGICCTrainer(
        model=model,
        h5_path=h5_path,
        output_dir=output_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        comp_weight=2.0,
        cont_weight=1.0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip_norm=1.0,
        t_0=args.t0,
        t_mult=args.t_mult,
        mask_rate=0.02,
        noise_std=0.01,
        checkpoint_every=10,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Print training config
    print(f"\nV5 Training config:")
    print(f"  Data: V5 expanded (1M train, 100K val, 100K test)")
    print(f"  Model: MAGICCModelV3 (SE attention + cross-attention)")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LR schedule: CosineAnnealingWarmRestarts(T_0={args.t0}, T_mult={args.t_mult})")
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

    # Print final results
    print("\n" + "=" * 70)
    print("V5 FINAL RESULTS")
    print("=" * 70)
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Compare with V4 results
    print("\n" + "-" * 70)
    print("V4 -> V5 COMPARISON")
    print("-" * 70)
    v4_results = {
        'comp_mae': 3.86, 'cont_mae': 4.13,
        'comp_rmse': 6.38, 'cont_rmse': 7.07,
        'comp_r2': 0.837, 'cont_r2': 0.934,
    }

    for metric in ['comp_mae', 'cont_mae', 'comp_rmse', 'cont_rmse', 'comp_r2', 'cont_r2']:
        v5_val = best_metrics.get(metric, float('nan'))
        v4_val = v4_results.get(metric, float('nan'))
        delta = v5_val - v4_val
        if metric.endswith('r2'):
            better = "BETTER" if v5_val > v4_val else "WORSE"
        else:
            better = "BETTER" if v5_val < v4_val else "WORSE"
        print(f"  {metric}: V5={v5_val:.4f} vs V4={v4_val:.4f} (delta={delta:+.4f}) {better}")

    print(f"\nBest model saved to: {output_dir}/best_model.pt")
    print(f"Training curves saved to: {output_dir}/training_curves.png")
    print(f"Training history saved to: {output_dir}/training_history.json")

    # Save V5 config for reproducibility
    v5_config = {
        'model_version': 'V5',
        'architecture': 'MAGICCModelV3 (SE attention + cross-attention)',
        'data': 'V5 expanded: 1M train (800K orig + 100K pure complete + 100K low-contam)',
        'n_kmer_features': 9249,
        'n_assembly_features': 7,
        'params': params,
        'training_config': {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'patience': args.patience,
            'se_reduction': args.se_reduction,
            'n_attn_heads': args.n_attn_heads,
            'n_attn_groups': args.n_attn_groups,
            't_0': args.t0,
            't_mult': args.t_mult,
            'comp_weight': 2.0,
            'cont_weight': 1.0,
            'mask_rate': 0.02,
            'noise_std': 0.01,
            'grad_clip_norm': 1.0,
            'mixed_precision': 'FP16',
            'gradient_checkpointing': True,
        },
        'best_metrics': best_metrics,
        'v4_comparison': v4_results,
    }
    config_path = os.path.join(output_dir, 'v5_config.json')
    with open(config_path, 'w') as f:
        json.dump(v5_config, f, indent=2)
    print(f"V5 config saved to: {config_path}")


if __name__ == '__main__':
    main()
