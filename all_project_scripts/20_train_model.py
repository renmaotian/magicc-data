#!/usr/bin/env python3
"""
Phase 5, Step 3: Train the MAGICC multi-branch fusion neural network.

Usage:
    python scripts/20_train_model.py [--resume CHECKPOINT_PATH]
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import build_model
from magicc.trainer import MAGICCTrainer


def main():
    parser = argparse.ArgumentParser(description='Train MAGICC model')
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

    # Build model with gradient checkpointing enabled
    model = build_model(
        n_kmer_features=9249,
        n_assembly_features=26,
        use_gradient_checkpointing=True,
        device=device,
    )
    params = model.count_parameters()
    print(f"\nModel parameters: {params['total']:,} total")
    print(f"  K-mer branch: {params['kmer_branch']:,}")
    print(f"  Assembly branch: {params['assembly_branch']:,}")
    print(f"  Fusion head: {params['fusion_head']:,}")

    # Create trainer
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

    # Train!
    best_metrics = trainer.train()

    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nBest model saved to: {output_dir}/best_model.pt")
    print(f"Training curves saved to: {output_dir}/training_curves.png")
    print(f"Training history saved to: {output_dir}/training_history.json")


if __name__ == '__main__':
    main()
