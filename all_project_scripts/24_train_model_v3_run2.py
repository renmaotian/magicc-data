#!/usr/bin/env python3
"""
V3 Run 2: Train with warmup + cosine decay (no restarts), lower LR.

Changes from Run 1:
- LR schedule: linear warmup (5 epochs) + cosine decay to 1e-6
- Lower peak LR: 3e-4 (vs 5e-4)
- More patience (25 vs 20) since no restarts to disrupt learning
- Optionally initialize from Run 1 pretrained weights

Usage:
    python scripts/24_train_model_v3_run2.py [--pretrained models/best_model_v3_run1.pt]
"""

import sys
import os
import argparse
import time
import json
import math
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magicc.model import build_model_v3
from magicc.trainer import MAGICCTrainer, HDF5Dataset, WeightedMSELoss
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class WarmupCosineScheduler:
    """Linear warmup + cosine decay scheduler."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._last_epoch = 0

    def step(self, epoch):
        self._last_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = self.min_lr / self.base_lrs[0] + (1 - self.min_lr / self.base_lrs[0]) * 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale if epoch < self.warmup_epochs else base_lr * scale

    def state_dict(self):
        return {'last_epoch': self._last_epoch}

    def load_state_dict(self, state_dict):
        self._last_epoch = state_dict.get('last_epoch', 0)


def main():
    parser = argparse.ArgumentParser(description='Train MAGICC V3 Run 2')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained V3 weights (model state dict only)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_path = os.path.join(project_dir, 'data', 'features', 'magicc_features.h5')
    output_dir = os.path.join(project_dir, 'models')
    os.makedirs(output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

    # Build model
    print("\nBuilding MAGICCModelV3...")
    model = build_model_v3(
        n_kmer_features=9249,
        n_assembly_features=26,
        use_gradient_checkpointing=True,
        device=device,
    )
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")

    # Load pretrained weights if specified
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}...")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded from epoch {ckpt['epoch']+1}, val_loss={ckpt['best_val_loss']:.4f}")

    # Datasets
    train_dataset = HDF5Dataset(
        h5_path, split='train', augment=True, mask_rate=0.02, noise_std=0.01)
    val_dataset = HDF5Dataset(
        h5_path, split='val', augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False)

    # Loss, optimizer, scheduler
    criterion = WeightedMSELoss(comp_weight=2.0, cont_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.max_epochs, min_lr=1e-6)
    scaler = GradScaler('cuda')

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'val_comp_mae': [], 'val_cont_mae': [],
        'val_comp_rmse': [], 'val_cont_rmse': [],
        'val_comp_r2': [], 'val_cont_r2': [],
        'lr': [], 'epoch_time': [],
    }
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    print(f"\nTraining config:")
    print(f"  LR: {args.lr} with warmup ({args.warmup_epochs} epochs) + cosine decay")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Loss weights: comp=2.0, cont=1.0")
    print(f"  Patience: {args.patience}")
    print(f"  Max epochs: {args.max_epochs}")

    print(f"\n{'='*70}")
    print(f"MAGICC V3 Training (Run 2)")
    print(f"{'='*70}")
    print(f"Training: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")
    print(f"Batches/epoch: {len(train_loader)}, Batch size: {args.batch_size}")
    print(f"{'='*70}\n")

    total_start = time.time()

    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        scheduler.step(epoch)

        # ---- Train ----
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

        # ---- Validate ----
        model.eval()
        val_loss_total = 0.0
        val_n = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for kmer, asm, labels in val_loader:
                kmer = kmer.to(device, non_blocking=True)
                asm = asm.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast('cuda'):
                    pred = model(kmer, asm)
                    vloss = criterion(pred, labels)
                val_loss_total += vloss.item()
                val_n += 1
                all_preds.append(pred.cpu())
                all_targets.append(labels.cpu())

        val_loss = val_loss_total / val_n
        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        comp_mae = np.mean(np.abs(preds[:, 0] - targets[:, 0]))
        cont_mae = np.mean(np.abs(preds[:, 1] - targets[:, 1]))
        comp_rmse = np.sqrt(np.mean((preds[:, 0] - targets[:, 0]) ** 2))
        cont_rmse = np.sqrt(np.mean((preds[:, 1] - targets[:, 1]) ** 2))

        ss_res_c = np.sum((preds[:, 0] - targets[:, 0]) ** 2)
        ss_tot_c = np.sum((targets[:, 0] - targets[:, 0].mean()) ** 2)
        comp_r2 = 1 - ss_res_c / ss_tot_c if ss_tot_c > 0 else 0.0

        ss_res_t = np.sum((preds[:, 1] - targets[:, 1]) ** 2)
        ss_tot_t = np.sum((targets[:, 1] - targets[:, 1].mean()) ** 2)
        cont_r2 = 1 - ss_res_t / ss_tot_t if ss_tot_t > 0 else 0.0

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_comp_mae'].append(float(comp_mae))
        history['val_cont_mae'].append(float(cont_mae))
        history['val_comp_rmse'].append(float(comp_rmse))
        history['val_cont_rmse'].append(float(cont_rmse))
        history['val_comp_r2'].append(float(comp_r2))
        history['val_cont_r2'].append(float(cont_r2))
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best model
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
            torch.save(state, os.path.join(output_dir, 'best_model.pt'))
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:3d}/{args.max_epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"MAE comp: {comp_mae:.2f}% cont: {cont_mae:.2f}% | "
              f"R2 comp: {comp_r2:.4f} cont: {cont_r2:.4f} | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
              f"{' *BEST*' if is_best else ''}")

        if is_best:
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
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
            torch.save(state, os.path.join(output_dir, f'checkpoint_epoch_{epoch:03d}.pt'))

            # Plot curves
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            ep = range(1, len(history['train_loss']) + 1)
            axes[0, 0].plot(ep, history['train_loss'], label='Train')
            axes[0, 0].plot(ep, history['val_loss'], label='Val')
            axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
            axes[0, 1].plot(ep, history['val_comp_mae'], label='Comp')
            axes[0, 1].plot(ep, history['val_cont_mae'], label='Cont')
            axes[0, 1].set_title('MAE (%)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
            axes[0, 2].plot(ep, history['val_comp_rmse'], label='Comp')
            axes[0, 2].plot(ep, history['val_cont_rmse'], label='Cont')
            axes[0, 2].set_title('RMSE (%)'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
            axes[1, 0].plot(ep, history['val_comp_r2'], label='Comp')
            axes[1, 0].plot(ep, history['val_cont_r2'], label='Cont')
            axes[1, 0].set_title('R-squared'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
            axes[1, 1].plot(ep, history['lr'])
            axes[1, 1].set_title('LR'); axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)
            axes[1, 2].plot(ep, history['epoch_time'])
            axes[1, 2].set_title('Time (s)'); axes[1, 2].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best epoch: {best_epoch+1}, best val_loss: {best_val_loss:.4f}")
    print(f"{'='*70}")

    # Final curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ep = range(1, len(history['train_loss']) + 1)
    axes[0, 0].plot(ep, history['train_loss'], label='Train')
    axes[0, 0].plot(ep, history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(ep, history['val_comp_mae'], label='Comp')
    axes[0, 1].plot(ep, history['val_cont_mae'], label='Cont')
    axes[0, 1].set_title('MAE (%)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].plot(ep, history['val_comp_rmse'], label='Comp')
    axes[0, 2].plot(ep, history['val_cont_rmse'], label='Cont')
    axes[0, 2].set_title('RMSE (%)'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].plot(ep, history['val_comp_r2'], label='Comp')
    axes[1, 0].plot(ep, history['val_cont_r2'], label='Cont')
    axes[1, 0].set_title('R-squared'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(ep, history['lr'])
    axes[1, 1].set_title('LR'); axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].plot(ep, history['epoch_time'])
    axes[1, 2].set_title('Time (s)'); axes[1, 2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Report best metrics
    best_idx = best_epoch
    print(f"\nBest epoch {best_epoch+1} metrics:")
    for metric in ['val_comp_mae', 'val_cont_mae', 'val_comp_rmse', 'val_cont_rmse', 'val_comp_r2', 'val_cont_r2']:
        print(f"  {metric}: {history[metric][best_idx]:.4f}")

    # V2 comparison
    v2_results = {
        'val_comp_mae': 3.93, 'val_cont_mae': 4.49,
        'val_comp_rmse': 6.56, 'val_cont_rmse': 7.41,
        'val_comp_r2': 0.828, 'val_cont_r2': 0.945,
    }
    print(f"\n{'='*70}")
    print("V3 Run 2 vs V2 Comparison")
    print(f"{'='*70}")
    for metric in ['val_comp_mae', 'val_cont_mae', 'val_comp_rmse', 'val_cont_rmse', 'val_comp_r2', 'val_cont_r2']:
        v3 = history[metric][best_idx]
        v2 = v2_results[metric]
        delta = v3 - v2
        if metric.endswith('r2'):
            better = "BETTER" if v3 > v2 else "WORSE"
        else:
            better = "BETTER" if v3 < v2 else "WORSE"
        print(f"  {metric}: V3={v3:.4f} vs V2={v2:.4f} (delta={delta:+.4f}) {better}")

    # Save config
    config = {
        'model': 'MAGICCModelV3', 'run': 2,
        'schedule': 'warmup_cosine_decay',
        'lr': args.lr, 'warmup_epochs': args.warmup_epochs,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'pretrained': args.pretrained,
        'best_epoch': best_epoch + 1,
        'best_val_loss': best_val_loss,
        'params': params,
    }
    with open(os.path.join(output_dir, 'v3_run2_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
