#!/usr/bin/env python3
"""
Phase 5, V3: Train the MAGICC attention-based model.

Uses MAGICCModelV3 with:
- SE (Squeeze-and-Excitation) attention in k-mer branch
- Cross-attention at fusion stage (assembly queries k-mer)
- Same training config as V2: comp_weight=2.0, cont_weight=1.0,
  weight_decay=5e-4, mask_rate=0.02

Usage:
    python scripts/23_train_model_v3.py [--lr 5e-4] [--resume CHECKPOINT_PATH]
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
    parser = argparse.ArgumentParser(description='Train MAGICC V3 attention model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4, lower than V2 for attention)')
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
    h5_path = os.path.join(project_dir, 'data', 'features', 'magicc_features.h5')
    output_dir = os.path.join(project_dir, 'models')

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

    # Build V3 model with attention
    print("\nBuilding MAGICCModelV3 (attention-based)...")
    model = build_model_v3(
        n_kmer_features=9249,
        n_assembly_features=26,
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
    dummy_asm = torch.randn(args.batch_size, 26, device=device)
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

    # Create trainer (reuses existing MAGICCTrainer -- same interface)
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

    # Train!
    print(f"\nTraining config:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LR schedule: CosineAnnealingWarmRestarts(T_0={args.t0}, T_mult={args.t_mult})")
    print(f"  Loss weights: comp=2.0, cont=1.0")
    print(f"  Augmentation: mask_rate=0.02, noise_std=0.01")

    best_metrics = trainer.train()

    # Print final results
    print("\n" + "=" * 70)
    print("V3 FINAL RESULTS")
    print("=" * 70)
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Compare with V2 targets
    print("\n" + "-" * 70)
    print("TARGET COMPARISON")
    print("-" * 70)
    v2_results = {
        'comp_mae': 3.93, 'cont_mae': 4.49,
        'comp_rmse': 6.56, 'cont_rmse': 7.41,
        'comp_r2': 0.828, 'cont_r2': 0.945,
    }
    targets = {'comp_mae': 2.5, 'cont_mae': 3.0, 'comp_r2': 0.90}

    for metric in ['comp_mae', 'cont_mae', 'comp_rmse', 'cont_rmse', 'comp_r2', 'cont_r2']:
        v3_val = best_metrics.get(metric, float('nan'))
        v2_val = v2_results.get(metric, float('nan'))
        target = targets.get(metric, None)
        delta = v3_val - v2_val
        target_str = f" (target: {target})" if target else ""
        if metric.endswith('r2'):
            better = "BETTER" if v3_val > v2_val else "WORSE"
            met = " MET!" if (target and v3_val >= target) else (" NOT MET" if target else "")
        else:
            better = "BETTER" if v3_val < v2_val else "WORSE"
            met = " MET!" if (target and v3_val <= target) else (" NOT MET" if target else "")
        print(f"  {metric}: V3={v3_val:.4f} vs V2={v2_val:.4f} (delta={delta:+.4f}) {better}{target_str}{met}")

    print(f"\nBest model saved to: {output_dir}/best_model.pt")
    print(f"Training curves saved to: {output_dir}/training_curves.png")
    print(f"Training history saved to: {output_dir}/training_history.json")

    # Save V3 config for reproducibility
    v3_config = {
        'model_version': 'V3',
        'architecture': 'MAGICCModelV3 (SE attention + cross-attention)',
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
        },
        'best_metrics': best_metrics,
        'v2_comparison': v2_results,
    }
    config_path = os.path.join(output_dir, 'v3_config.json')
    with open(config_path, 'w') as f:
        json.dump(v3_config, f, indent=2)
    print(f"V3 config saved to: {config_path}")


if __name__ == '__main__':
    main()
