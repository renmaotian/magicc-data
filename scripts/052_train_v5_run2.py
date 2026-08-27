#!/usr/bin/env python3
"""
Train MAGICC V5 Run 2 with improved LR schedule (ReduceLROnPlateau).

V5 Run 1 used CosineAnnealingWarmRestarts which caused training disruption
at restart boundaries. Run 2 fixes this with:
- ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6)
- Linear warmup for first 5 epochs (1e-5 -> 5e-4)
- Lower initial LR: 5e-4 (vs 1e-3 in Run 1)
- Longer early stopping patience: 30 (vs 20 in Run 1)
- Max epochs: 200

Architecture: MAGICCModelV3 (SE attention + cross-attention fusion)
Data: V5 expanded (1M train, 100K val, 100K test)

Usage:
    python scripts/52_train_v5_run2.py
    python scripts/52_train_v5_run2.py --resume models/best_model_v5_run2.pt
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import build_model_v3, MAGICCModelV3
from magicc.trainer import HDF5Dataset, WeightedMSELoss


def get_warmup_lr(epoch: int, warmup_epochs: int, warmup_start_lr: float,
                  target_lr: float) -> float:
    """Linear warmup LR from warmup_start_lr to target_lr over warmup_epochs."""
    if epoch >= warmup_epochs:
        return target_lr
    return warmup_start_lr + (target_lr - warmup_start_lr) * epoch / warmup_epochs


def set_lr(optimizer, lr: float):
    """Set learning rate for all parameter groups."""
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def validate(model, val_loader, criterion, device):
    """Run validation and return metrics dict."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for kmer, asm, labels in val_loader:
            kmer = kmer.to(device, non_blocking=True)
            asm = asm.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda'):
                pred = model(kmer, asm)
                loss = criterion(pred, labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    comp_pred, cont_pred = preds[:, 0], preds[:, 1]
    comp_true, cont_true = targets[:, 0], targets[:, 1]

    comp_mae = np.mean(np.abs(comp_pred - comp_true))
    cont_mae = np.mean(np.abs(cont_pred - cont_true))
    comp_rmse = np.sqrt(np.mean((comp_pred - comp_true) ** 2))
    cont_rmse = np.sqrt(np.mean((cont_pred - cont_true) ** 2))

    comp_ss_res = np.sum((comp_pred - comp_true) ** 2)
    comp_ss_tot = np.sum((comp_true - comp_true.mean()) ** 2)
    comp_r2 = 1 - comp_ss_res / comp_ss_tot if comp_ss_tot > 0 else 0.0

    cont_ss_res = np.sum((cont_pred - cont_true) ** 2)
    cont_ss_tot = np.sum((cont_true - cont_true.mean()) ** 2)
    cont_r2 = 1 - cont_ss_res / cont_ss_tot if cont_ss_tot > 0 else 0.0

    return {
        'val_loss': total_loss / n_batches,
        'comp_mae': float(comp_mae),
        'cont_mae': float(cont_mae),
        'comp_rmse': float(comp_rmse),
        'cont_rmse': float(cont_rmse),
        'comp_r2': float(comp_r2),
        'cont_r2': float(cont_r2),
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss,
                    best_epoch, history, output_dir, filename):
    """Save checkpoint to disk."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'history': history,
    }
    path = output_dir / filename
    torch.save(state, path)
    return path


def plot_training_curves(history, output_dir):
    """Save training curves plot."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Weighted MSE Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['val_comp_mae'], label='Completeness')
    axes[0, 1].plot(epochs, history['val_cont_mae'], label='Contamination')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (%)')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['val_comp_rmse'], label='Completeness')
    axes[0, 2].plot(epochs, history['val_cont_rmse'], label='Contamination')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('RMSE (%)')
    axes[0, 2].set_title('Validation RMSE')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['val_comp_r2'], label='Completeness')
    axes[1, 0].plot(epochs, history['val_cont_r2'], label='Contamination')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R-squared')
    axes[1, 0].set_title('Validation R-squared')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['lr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, history['epoch_time'])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Time (s)')
    axes[1, 2].set_title('Epoch Duration')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'training_curves_v5_run2.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train MAGICC V5 Run 2')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Target learning rate after warmup (default: 5e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of linear warmup epochs')
    parser.add_argument('--warmup-start-lr', type=float, default=1e-5,
                        help='Warmup starting LR')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--plateau-patience', type=int, default=10,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('--plateau-factor', type=float, default=0.5,
                        help='ReduceLROnPlateau factor')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_path = os.path.join(project_dir, 'data', 'features', 'magicc_v5_features.h5')
    output_dir = Path(os.path.join(project_dir, 'models'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also log to file
    log_path = output_dir / 'training_v5_run2_log.txt'

    if not os.path.exists(h5_path):
        print(f"ERROR: V5 HDF5 not found at {h5_path}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9 if hasattr(
        torch.cuda.get_device_properties(0), 'total_mem') else \
        torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Verify HDF5
    print(f"\nVerifying V5 HDF5: {h5_path}")
    import h5py
    with h5py.File(h5_path, 'r') as f:
        for split in ['train', 'val', 'test']:
            grp = f[split]
            n = grp['kmer_features'].shape[0]
            n_kmer = grp['kmer_features'].shape[1]
            n_asm = grp['assembly_features'].shape[1]
            print(f"  {split}: {n:,} samples, kmer={n_kmer}, asm={n_asm}")

    # Build model
    print("\nBuilding MAGICCModelV3 (SE attention + cross-attention fusion)...")
    model = build_model_v3(
        n_kmer_features=9249,
        n_assembly_features=7,
        use_gradient_checkpointing=True,
        device=device,
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
    with autocast('cuda'):
        out = model(dummy_kmer, dummy_asm)
        loss = out.mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"\nGPU memory profile (batch_size={args.batch_size}):")
    print(f"  Peak memory: {peak:.0f} MB / {gpu_mem*1000:.0f} MB ({peak/gpu_mem/10:.1f}%)")
    del dummy_kmer, dummy_asm, out, loss
    torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)

    # Loss
    criterion = WeightedMSELoss(comp_weight=2.0, cont_weight=1.0)

    # Optimizer - start with warmup_start_lr, will ramp up during warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.warmup_start_lr,  # Start low, warmup will increase
        weight_decay=args.weight_decay,
    )

    # LR Scheduler: ReduceLROnPlateau (only applied after warmup)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=args.min_lr,
        verbose=True,
    )

    # Mixed precision
    scaler = GradScaler('cuda')

    # Datasets
    train_dataset = HDF5Dataset(
        h5_path, split='train', augment=True,
        mask_rate=0.02, noise_std=0.01,
    )
    val_dataset = HDF5Dataset(
        h5_path, split='val', augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Training state
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_comp_mae': [],
        'val_cont_mae': [],
        'val_comp_rmse': [],
        'val_cont_rmse': [],
        'val_comp_r2': [],
        'val_cont_r2': [],
        'lr': [],
        'epoch_time': [],
    }
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 0

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\nLoading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        best_val_loss = ckpt['best_val_loss']
        best_epoch = ckpt['best_epoch']
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        epochs_without_improvement = start_epoch - best_epoch
        print(f"  Resumed from epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f} at epoch {best_epoch}")

    # Print training config
    print(f"\n{'='*70}")
    print(f"MAGICC V5 Run 2 Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: MAGICCModelV3 (SE attention + cross-attention)")
    print(f"Parameters: {params['total']:,}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches/epoch: {len(train_loader)}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Early stopping patience: {args.patience}")
    print(f"LR schedule: ReduceLROnPlateau(factor={args.plateau_factor}, "
          f"patience={args.plateau_patience}, min_lr={args.min_lr})")
    print(f"Warmup: {args.warmup_epochs} epochs ({args.warmup_start_lr} -> {args.lr})")
    print(f"Initial LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Loss weights: comp=2.0, cont=1.0")
    print(f"Augmentation: mask_rate=0.02, noise_std=0.01")
    print(f"Mixed precision: FP16")
    print(f"Gradient checkpointing: enabled")
    print(f"Gradient clipping: max_norm=1.0")
    print(f"{'='*70}\n")

    total_start = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        epoch_start = time.time()

        # --- Learning rate warmup ---
        if epoch < args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, args.warmup_epochs,
                                       args.warmup_start_lr, args.lr)
            set_lr(optimizer, warmup_lr)

        # --- Train one epoch ---
        model.train()
        total_loss = 0.0
        n_batches = 0

        for kmer, asm, labels in train_loader:
            kmer = kmer.to(device, non_blocking=True)
            asm = asm.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                pred = model(kmer, asm)
                loss = criterion(pred, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        # --- Validate ---
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics['val_loss']

        # --- LR scheduler step (only after warmup) ---
        if epoch >= args.warmup_epochs:
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_comp_mae'].append(val_metrics['comp_mae'])
        history['val_cont_mae'].append(val_metrics['cont_mae'])
        history['val_comp_rmse'].append(val_metrics['comp_rmse'])
        history['val_cont_rmse'].append(val_metrics['cont_rmse'])
        history['val_comp_r2'].append(val_metrics['comp_r2'])
        history['val_cont_r2'].append(val_metrics['cont_r2'])
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        # Check for improvement
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress
        log_line = (f"Epoch {epoch+1:3d}/{args.max_epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"MAE comp: {val_metrics['comp_mae']:.2f}% "
                    f"cont: {val_metrics['cont_mae']:.2f}% | "
                    f"R2 comp: {val_metrics['comp_r2']:.4f} "
                    f"cont: {val_metrics['cont_r2']:.4f} | "
                    f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
                    f"{' *BEST*' if is_best else ''}")
        print(log_line, flush=True)

        # Save checkpoints
        if is_best:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            best_val_loss, best_epoch, history, output_dir,
                            'best_model_v5_run2.pt')
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            best_val_loss, best_epoch, history, output_dir,
                            f'checkpoint_v5r2_epoch_{epoch+1:03d}.pt')
            plot_training_curves(history, output_dir)

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    total_time = time.time() - total_start

    # Final summary
    print(f"\n{'='*70}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    print(f"{'='*70}")

    # Final plot
    plot_training_curves(history, output_dir)

    # Save training history
    history_path = output_dir / 'training_history_v5_run2.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Get best metrics from history
    best_idx = best_epoch - start_epoch
    if 0 <= best_idx < len(history['val_loss']):
        best_metrics = {
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'comp_mae': history['val_comp_mae'][best_idx],
            'cont_mae': history['val_cont_mae'][best_idx],
            'comp_rmse': history['val_comp_rmse'][best_idx],
            'cont_rmse': history['val_cont_rmse'][best_idx],
            'comp_r2': history['val_comp_r2'][best_idx],
            'cont_r2': history['val_cont_r2'][best_idx],
            'total_time_min': total_time / 60,
        }
    else:
        best_metrics = {'best_val_loss': best_val_loss, 'best_epoch': best_epoch + 1}

    # Print final results
    print("\n" + "=" * 70)
    print("V5 RUN 2 FINAL RESULTS")
    print("=" * 70)
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Compare with V4 and V5 Run 1
    print("\n" + "-" * 70)
    print("COMPARISON: V4 vs V5R1 vs V5R2")
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

    print(f"  {'Metric':<12} {'V4':>8} {'V5R1':>8} {'V5R2':>8} {'V5R2 vs V4':>12}")
    print(f"  {'-'*48}")
    for metric in ['comp_mae', 'cont_mae', 'comp_rmse', 'cont_rmse', 'comp_r2', 'cont_r2']:
        v4_val = v4_results.get(metric, float('nan'))
        v5r1_val = v5r1_results.get(metric, float('nan'))
        v5r2_val = best_metrics.get(metric, float('nan'))
        delta = v5r2_val - v4_val
        if metric.endswith('r2'):
            better = "BETTER" if v5r2_val > v4_val else "WORSE"
        else:
            better = "BETTER" if v5r2_val < v4_val else "WORSE"
        print(f"  {metric:<12} {v4_val:>8.4f} {v5r1_val:>8.4f} {v5r2_val:>8.4f} "
              f"{delta:>+8.4f} {better}")

    print(f"\nBest model saved to: {output_dir}/best_model_v5_run2.pt")
    print(f"Training curves: {output_dir}/training_curves_v5_run2.png")
    print(f"Training history: {history_path}")

    # Save V5 Run 2 config
    v5r2_config = {
        'model_version': 'V5_Run2',
        'architecture': 'MAGICCModelV3 (SE attention + cross-attention)',
        'data': 'V5 expanded: 1M train, 100K val, 100K test',
        'n_kmer_features': 9249,
        'n_assembly_features': 7,
        'params': params,
        'training_config': {
            'lr': args.lr,
            'warmup_epochs': args.warmup_epochs,
            'warmup_start_lr': args.warmup_start_lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'patience': args.patience,
            'lr_schedule': 'ReduceLROnPlateau',
            'plateau_factor': args.plateau_factor,
            'plateau_patience': args.plateau_patience,
            'min_lr': args.min_lr,
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
        'v5r1_comparison': v5r1_results,
    }
    config_path = output_dir / 'v5_run2_config.json'
    with open(config_path, 'w') as f:
        json.dump(v5r2_config, f, indent=2)
    print(f"Config saved to: {config_path}")


if __name__ == '__main__':
    main()
