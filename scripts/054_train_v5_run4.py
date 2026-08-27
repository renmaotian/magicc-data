#!/usr/bin/env python3
"""
Train MAGICC V5 Run 4: V3 architecture (SE attention + cross-attention) on V5 data.

Run 3 showed V5 data helps (beats V4 with V2 arch). Run 4 tests if V3 attention
adds further improvement on top of the better data.

Configuration (same as Run 3 / V4, but with V3 architecture):
- Model: MAGICCModelV3 (SE attention + cross-attention) via build_model_v3()
- Data: V5 expanded (1M train, 100K val, 100K test)
- AdamW, lr=1e-3, weight_decay=5e-4
- CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
- batch_size=512, max_epochs=150, patience=20
"""

import sys
import os
import argparse
import shutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import build_model_v3
from magicc.trainer import MAGICCTrainer


def main():
    parser = argparse.ArgumentParser(description='Train MAGICC V5 Run 4 (V3 attn + V5 data)')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_path = os.path.join(project_dir, 'data', 'features', 'magicc_v5_features.h5')
    output_dir = os.path.join(project_dir, 'models')

    if not os.path.exists(h5_path):
        print(f"ERROR: V5 HDF5 not found at {h5_path}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

    # Build V3 model (SE attention + cross-attention fusion)
    print("\nBuilding MAGICCModelV3 (SE attention + cross-attention)...")
    model = build_model_v3(
        n_kmer_features=9249,
        n_assembly_features=7,
        use_gradient_checkpointing=True,
        device=device,
    )
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} total")

    # Create trainer -- same config as V4/R3
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

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print(f"\nV5 Run 4: V3 attention arch + V5 data + lr=1e-3 + CosineRestart")
    best_metrics = trainer.train()

    # Save with run4 suffix
    src = os.path.join(output_dir, 'best_model.pt')
    dst = os.path.join(output_dir, 'best_model_v5_run4.pt')
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"\nCopied best model to: {dst}")

    for fname, dst_fname in [
        ('training_curves.png', 'training_curves_v5_run4.png'),
        ('training_history.json', 'training_history_v5_run4.json'),
    ]:
        src_f = os.path.join(output_dir, fname)
        dst_f = os.path.join(output_dir, dst_fname)
        if os.path.exists(src_f):
            shutil.copy2(src_f, dst_f)

    # Print results
    print("\n" + "=" * 70)
    print("V5 RUN 4 FINAL RESULTS (V3 attention + V5 data)")
    print("=" * 70)
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    refs = {
        'V4':  {'comp_mae': 3.86, 'cont_mae': 4.13, 'comp_r2': 0.837, 'cont_r2': 0.934},
        'V5R3': {'comp_mae': 3.795, 'cont_mae': 4.144, 'comp_r2': 0.838, 'cont_r2': 0.933},
    }
    print(f"  {'Metric':<12} {'V4':>8} {'V5R3':>8} {'V5R4':>8} {'R4 vs R3':>12}")
    print(f"  {'-'*48}")
    for metric in ['comp_mae', 'cont_mae', 'comp_r2', 'cont_r2']:
        v4_val = refs['V4'][metric]
        r3_val = refs['V5R3'][metric]
        r4_val = best_metrics.get(metric, float('nan'))
        delta = r4_val - r3_val
        better = "BETTER" if (delta < 0 if not metric.endswith('r2') else delta > 0) else "WORSE"
        print(f"  {metric:<12} {v4_val:>8.4f} {r3_val:>8.4f} {r4_val:>8.4f} {delta:>+8.4f} {better}")


if __name__ == '__main__':
    main()
