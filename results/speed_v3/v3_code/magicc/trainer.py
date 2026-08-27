"""
MAGICC Model Trainer.

Handles:
- HDF5 dataset loading with data augmentation
- Mixed-precision training (FP16) with gradient checkpointing
- Weighted MSE loss: L = 2.0 * MSE(comp) + 1.0 * MSE(cont)
- AdamW optimizer with cosine annealing warm restarts
- Early stopping, gradient clipping, checkpointing
- Training/validation logging and curve plotting
"""

import os
import time
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple


class HDF5Dataset(Dataset):
    """
    PyTorch Dataset that reads from HDF5 file.

    The HDF5 data is already normalized -- features are used directly.
    Supports data augmentation: random k-mer masking and Gaussian noise injection.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 file.
    split : str
        Split name ('train', 'val', 'test').
    augment : bool
        Whether to apply data augmentation (only for training).
    mask_rate : float
        Fraction of k-mer features to randomly zero out (augmentation).
    noise_std : float
        Standard deviation of Gaussian noise to inject (augmentation).
    """

    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        augment: bool = False,
        mask_rate: float = 0.02,
        noise_std: float = 0.01,
    ):
        self.h5_path = h5_path
        self.split = split
        self.augment = augment
        self.mask_rate = mask_rate
        self.noise_std = noise_std

        # Read entire dataset into memory for fast access
        # 800k * (9249 + 20 + 2) * 4 bytes ~ 29.7 GB -- fits in 881 GB RAM
        print(f"Loading {split} data from HDF5 into memory...")
        t0 = time.time()
        with h5py.File(h5_path, 'r') as f:
            grp = f[split]
            self.kmer_features = grp['kmer_features'][:]     # (n, 9249)
            self.assembly_features = grp['assembly_features'][:]  # (n, 20)
            self.labels = grp['labels'][:]                   # (n, 2)
        elapsed = time.time() - t0
        self.n_samples = self.kmer_features.shape[0]
        self.n_kmer = self.kmer_features.shape[1]
        print(f"  Loaded {self.n_samples:,} samples in {elapsed:.1f}s "
              f"(kmer: {self.kmer_features.shape}, asm: {self.assembly_features.shape})")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kmer = self.kmer_features[idx].copy()     # (9249,)
        asm = self.assembly_features[idx].copy()   # (20,)
        labels = self.labels[idx].copy()           # (2,)

        # Data augmentation (training only)
        if self.augment:
            # Random k-mer masking: zero out mask_rate fraction of k-mer features
            mask = np.random.random(self.n_kmer) < self.mask_rate
            kmer[mask] = 0.0

            # Gaussian noise injection on both feature types
            kmer += np.random.normal(0, self.noise_std, self.n_kmer).astype(np.float32)
            asm += np.random.normal(0, self.noise_std, asm.shape[0]).astype(np.float32)

        return (
            torch.from_numpy(kmer),
            torch.from_numpy(asm),
            torch.from_numpy(labels),
        )


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss: L = comp_weight * MSE(completeness) + cont_weight * MSE(contamination).

    Parameters
    ----------
    comp_weight : float
        Weight for completeness loss component (default: 2.0).
    cont_weight : float
        Weight for contamination loss component (default: 1.0).
    """

    def __init__(self, comp_weight: float = 2.0, cont_weight: float = 1.0):
        super().__init__()
        self.comp_weight = comp_weight
        self.cont_weight = cont_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        comp_loss = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
        cont_loss = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
        return self.comp_weight * comp_loss + self.cont_weight * cont_loss


class MAGICCTrainer:
    """
    Trainer for the MAGICC multi-branch fusion model.

    Parameters
    ----------
    model : nn.Module
        The MAGICCModel to train.
    h5_path : str
        Path to HDF5 features file.
    output_dir : str
        Directory for saving checkpoints, logs, and plots.
    lr : float
        Initial learning rate.
    weight_decay : float
        AdamW weight decay.
    comp_weight : float
        Weight for completeness loss.
    cont_weight : float
        Weight for contamination loss.
    batch_size : int
        Training batch size.
    num_workers : int
        DataLoader workers.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience (epochs without improvement).
    grad_clip_norm : float
        Maximum gradient norm for clipping.
    t_0 : int
        Cosine annealing T_0 parameter.
    t_mult : int
        Cosine annealing T_mult parameter.
    mask_rate : float
        K-mer masking rate for augmentation.
    noise_std : float
        Gaussian noise std for augmentation.
    checkpoint_every : int
        Save checkpoint every N epochs.
    device : str
        Device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        h5_path: str,
        output_dir: str = 'models',
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        comp_weight: float = 2.0,
        cont_weight: float = 1.0,
        batch_size: int = 512,
        num_workers: int = 4,
        max_epochs: int = 150,
        patience: int = 20,
        grad_clip_norm: float = 1.0,
        t_0: int = 10,
        t_mult: int = 2,
        mask_rate: float = 0.05,
        noise_std: float = 0.01,
        checkpoint_every: int = 10,
        device: str = 'cuda',
    ):
        self.model = model
        self.h5_path = h5_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Training config
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_every = checkpoint_every

        # Loss
        self.criterion = WeightedMSELoss(comp_weight=comp_weight, cont_weight=cont_weight)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR Scheduler: Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=t_0,
            T_mult=t_mult,
        )

        # Mixed precision
        self.scaler = GradScaler('cuda')

        # Datasets
        self.train_dataset = HDF5Dataset(
            h5_path, split='train', augment=True,
            mask_rate=mask_rate, noise_std=noise_std,
        )
        self.val_dataset = HDF5Dataset(
            h5_path, split='val', augment=False,
        )

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size * 2,  # Larger batch for validation (no grads)
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Training state
        self.history = {
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
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.start_epoch = 0

    def train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch, return average training loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for kmer, asm, labels in self.train_loader:
            kmer = kmer.to(self.device, non_blocking=True)
            asm = asm.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with autocast('cuda'):
                pred = self.model(kmer, asm)
                loss = self.criterion(pred, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        # Step LR scheduler
        self.scheduler.step(epoch + 1)

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation, return metrics dict."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for kmer, asm, labels in self.val_loader:
            kmer = kmer.to(self.device, non_blocking=True)
            asm = asm.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast('cuda'):
                pred = self.model(kmer, asm)
                loss = self.criterion(pred, labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())

        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        # Compute metrics
        comp_pred, cont_pred = preds[:, 0], preds[:, 1]
        comp_true, cont_true = targets[:, 0], targets[:, 1]

        comp_mae = np.mean(np.abs(comp_pred - comp_true))
        cont_mae = np.mean(np.abs(cont_pred - cont_true))
        comp_rmse = np.sqrt(np.mean((comp_pred - comp_true) ** 2))
        cont_rmse = np.sqrt(np.mean((cont_pred - cont_true) ** 2))

        # R-squared
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

    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = None):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }
        if filename is None:
            filename = f'checkpoint_epoch_{epoch:03d}.pt'
        path = self.output_dir / filename
        torch.save(state, path)

        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(state, best_path)
            print(f"  -> Saved best model (val_loss={self.best_val_loss:.4f})")

    def load_checkpoint(self, path: str):
        """Load checkpoint for resuming training."""
        print(f"Loading checkpoint from {path}...")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.best_val_loss = ckpt['best_val_loss']
        self.best_epoch = ckpt['best_epoch']
        self.history = ckpt['history']
        self.start_epoch = ckpt['epoch'] + 1
        self.epochs_without_improvement = self.start_epoch - self.best_epoch
        print(f"  Resumed from epoch {self.start_epoch}, "
              f"best_val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch}")

    def plot_training_curves(self):
        """Save training curves plot."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Weighted MSE Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE
        axes[0, 1].plot(epochs, self.history['val_comp_mae'], label='Completeness')
        axes[0, 1].plot(epochs, self.history['val_cont_mae'], label='Contamination')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (%)')
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # RMSE
        axes[0, 2].plot(epochs, self.history['val_comp_rmse'], label='Completeness')
        axes[0, 2].plot(epochs, self.history['val_cont_rmse'], label='Contamination')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('RMSE (%)')
        axes[0, 2].set_title('Validation RMSE')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # R-squared
        axes[1, 0].plot(epochs, self.history['val_comp_r2'], label='Completeness')
        axes[1, 0].plot(epochs, self.history['val_cont_r2'], label='Contamination')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R-squared')
        axes[1, 0].set_title('Validation R-squared')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 1].plot(epochs, self.history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        # Epoch time
        axes[1, 2].plot(epochs, self.history['epoch_time'])
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (s)')
        axes[1, 2].set_title('Epoch Duration')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {plot_path}")

    def train(self) -> Dict[str, float]:
        """
        Run full training loop.

        Returns
        -------
        dict
            Best validation metrics.
        """
        print(f"\n{'='*70}")
        print(f"MAGICC Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_dataset):,}")
        print(f"Validation samples: {len(self.val_dataset):,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batches/epoch: {len(self.train_loader)}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Early stopping patience: {self.patience}")
        print(f"Mixed precision: True (FP16)")
        print(f"Gradient checkpointing: {self.model.use_gradient_checkpointing}")
        print(f"{'='*70}\n")

        total_start = time.time()

        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_comp_mae'].append(val_metrics['comp_mae'])
            self.history['val_cont_mae'].append(val_metrics['cont_mae'])
            self.history['val_comp_rmse'].append(val_metrics['comp_rmse'])
            self.history['val_cont_rmse'].append(val_metrics['cont_rmse'])
            self.history['val_comp_r2'].append(val_metrics['comp_r2'])
            self.history['val_cont_r2'].append(val_metrics['cont_r2'])
            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Print progress
            print(f"Epoch {epoch+1:3d}/{self.max_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"MAE comp: {val_metrics['comp_mae']:.2f}% cont: {val_metrics['cont_mae']:.2f}% | "
                  f"R2 comp: {val_metrics['comp_r2']:.4f} cont: {val_metrics['cont_r2']:.4f} | "
                  f"LR: {current_lr:.2e} | {epoch_time:.1f}s"
                  f"{' *BEST*' if is_best else ''}")

            # Save checkpoints
            if is_best:
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % self.checkpoint_every == 0:
                self.save_checkpoint(epoch, is_best=False)
                self.plot_training_curves()

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"Training complete in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
        print(f"{'='*70}")

        # Final plot
        self.plot_training_curves()

        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Return best metrics
        best_idx = self.best_epoch - self.start_epoch
        if 0 <= best_idx < len(self.history['val_loss']):
            return {
                'best_epoch': self.best_epoch + 1,
                'best_val_loss': self.best_val_loss,
                'comp_mae': self.history['val_comp_mae'][best_idx],
                'cont_mae': self.history['val_cont_mae'][best_idx],
                'comp_rmse': self.history['val_comp_rmse'][best_idx],
                'cont_rmse': self.history['val_cont_rmse'][best_idx],
                'comp_r2': self.history['val_comp_r2'][best_idx],
                'cont_r2': self.history['val_cont_r2'][best_idx],
                'total_time_min': total_time / 60,
            }
        return {'best_val_loss': self.best_val_loss, 'best_epoch': self.best_epoch + 1}
