#!/usr/bin/env python3
"""
Script 38: Run DeepCheck on all 5 v2 benchmark sets.

DeepCheck uses CheckM2's intermediate feature vectors (PKL files from --dbg_vectors):
- Load CheckM2 feature vectors from checkm2_output/ in each set
- Apply MinMaxScaler transform manually (saved parameters at tools/DeepCheck/scaler_params.npz)
- Take first 20,021 features, zero-pad to 20,164, reshape to 142x142
- Run through DeepCheck's ResNet model: tools/DeepCheck/models/best_model.pt
- DeepCheck model bug: forward() only returns x_comp; define ResNetDualOutput to return both
- Scale outputs from [0,1] to [0,100]%

Sets:
1. data/benchmarks/motivating_v2/set_A/ -> deepcheck_predictions.tsv
2. data/benchmarks/motivating_v2/set_B/ -> deepcheck_predictions.tsv
3. data/benchmarks/set_A_v2/ -> deepcheck_predictions.tsv
4. data/benchmarks/set_B_v2/ -> deepcheck_predictions.tsv
5. data/benchmarks/set_E/ -> deepcheck_predictions.tsv

Usage:
    conda run -n magicc2 python scripts/38_run_deepcheck_v2.py
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = "/home/tianrm/projects/magicc2"
DEEPCHECK_DIR = os.path.join(PROJECT_DIR, "tools", "DeepCheck")
BENCHMARK_DIR = os.path.join(PROJECT_DIR, "data", "benchmarks")

# ============================================================
# Define the model with BOTH completeness and contamination output
# (The original model.py only returns x_comp in forward())
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        return out

class ResNetDualOutput(nn.Module):
    """ResNet with both completeness and contamination outputs."""
    def __init__(self, block, layers, num_classes=1):
        super(ResNetDualOutput, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.attention = SelfAttention(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 100)
        self.fc1 = nn.Linear(100, num_classes)  # completeness
        self.fc2 = nn.Linear(100, num_classes)  # contamination

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x_comp = self.fc1(x)
        x_cont = self.fc2(x)
        return x_comp, x_cont


class DeepCheckDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].reshape(142, 142)
        sample = torch.FloatTensor(sample).unsqueeze(0)
        return sample


def load_checkm2_features(checkm2_output_dir):
    """Load and concatenate CheckM2 feature vector PKL files."""
    pkl_files = sorted([f for f in os.listdir(checkm2_output_dir) if f.endswith('.pkl')])
    if not pkl_files:
        raise FileNotFoundError(f"No PKL files found in {checkm2_output_dir}")

    dfs = []
    for pkl in pkl_files:
        df = pd.read_pickle(os.path.join(checkm2_output_dir, pkl))
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(full_df)} genomes from {len(pkl_files)} PKL files")
    print(f"  Feature shape: {full_df.shape}")
    return full_df


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load DeepCheck scaler parameters
    scaler_params_path = os.path.join(DEEPCHECK_DIR, "scaler_params.npz")
    scaler_params = np.load(scaler_params_path)
    scaler_scale = scaler_params['scale']
    scaler_min = scaler_params['min_val']
    print(f"Loaded scaler params: scale shape={scaler_scale.shape}, min shape={scaler_min.shape}")

    # Load DeepCheck model
    model = ResNetDualOutput(ResidualBlock, [2, 2, 2, 2])
    state_dict = torch.load(
        os.path.join(DEEPCHECK_DIR, "models", "best_model.pt"),
        map_location='cpu', weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded DeepCheck model: {sum(p.numel() for p in model.parameters()):,} parameters")

    N_THREADS = 1
    BATCH_SIZE = 64

    SETS = [
        {'name': 'Motivating A', 'path': 'motivating_v2/set_A'},
        {'name': 'Motivating B', 'path': 'motivating_v2/set_B'},
        {'name': 'Benchmark A_v2', 'path': 'set_A_v2'},
        {'name': 'Benchmark B_v2', 'path': 'set_B_v2'},
        {'name': 'Set E', 'path': 'set_E'},
    ]

    all_results = []

    for s in SETS:
        set_dir = os.path.join(BENCHMARK_DIR, s['path'])
        checkm2_output_dir = os.path.join(set_dir, "checkm2_output")
        metadata_path = os.path.join(set_dir, "metadata.tsv")
        output_path = os.path.join(set_dir, "deepcheck_predictions.tsv")

        print(f"\n{'='*60}")
        print(f"{s['name']} ({s['path']})")
        print(f"{'='*60}")

        # Check if already done (resumable)
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path, sep="\t")
            metadata = pd.read_csv(metadata_path, sep="\t")
            if len(existing) >= len(metadata) and 'pred_completeness' in existing.columns:
                n_valid = existing['pred_completeness'].notna().sum()
                if n_valid >= len(metadata):
                    print(f"  SKIPPING: Output already exists with {n_valid} valid predictions")
                    valid = existing.dropna(subset=['pred_completeness', 'pred_contamination'])
                    comp_mae = np.mean(np.abs(valid['true_completeness'] - valid['pred_completeness']))
                    cont_mae = np.mean(np.abs(valid['true_contamination'] - valid['pred_contamination']))
                    ws = valid['wall_clock_s'].iloc[0] if 'wall_clock_s' in valid.columns else 0
                    print(f"  Existing: comp MAE={comp_mae:.2f}%, cont MAE={cont_mae:.2f}%, time={ws:.1f}s")
                    all_results.append({'name': s['name'], 'n': len(valid), 'wall_s': ws,
                                        'comp_mae': comp_mae, 'cont_mae': cont_mae})
                    continue

        # Load CheckM2 feature vectors
        try:
            features_df = load_checkm2_features(checkm2_output_dir)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        # Extract names and feature matrix
        names = features_df['Name'].values
        feature_matrix = features_df.iloc[:, 1:].values.astype(float)
        print(f"  Feature matrix shape: {feature_matrix.shape}")

        # Scale features using manual MinMaxScaler transform
        scaled_features = feature_matrix * scaler_scale + scaler_min
        scaled_features = scaled_features[:, :20021]
        print(f"  Scaled features shape: {scaled_features.shape}")

        # Zero-pad to 20,164 (142*142 = 20,164)
        n_samples = scaled_features.shape[0]
        padded_features = np.zeros((n_samples, 20164), dtype=np.float32)
        padded_features[:, :20021] = scaled_features
        print(f"  Padded features shape: {padded_features.shape}")

        # Create dataset and dataloader
        dataset = DeepCheckDataset(padded_features)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Run inference
        all_comp = []
        all_cont = []
        start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                comp_pred, cont_pred = model(batch)
                # DeepCheck outputs are in [0, 1] scale (labels were divided by 100 during training)
                all_comp.extend((comp_pred.squeeze() * 100).cpu().numpy().tolist())
                all_cont.extend((cont_pred.squeeze() * 100).cpu().numpy().tolist())

        wall_clock_s = time.time() - start_time
        print(f"  Inference time: {wall_clock_s:.2f}s ({len(names)/wall_clock_s:.1f} genomes/sec)")

        # Create results dataframe
        results = pd.DataFrame({
            'genome_id': names,
            'pred_completeness': all_comp,
            'pred_contamination': all_cont
        })

        # Load metadata and merge
        metadata = pd.read_csv(metadata_path, sep="\t")
        merged = metadata.merge(results, on='genome_id', how='left')
        merged['wall_clock_s'] = wall_clock_s
        merged['n_threads'] = N_THREADS

        # Check for missing predictions
        n_missing = merged['pred_completeness'].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} genomes missing predictions")

        # Save
        merged.to_csv(output_path, sep="\t", index=False)
        print(f"  Saved: {output_path}")

        # Quick accuracy summary
        valid = merged.dropna(subset=['pred_completeness', 'pred_contamination'])
        comp_mae = np.mean(np.abs(valid['true_completeness'] - valid['pred_completeness']))
        cont_mae = np.mean(np.abs(valid['true_contamination'] - valid['pred_contamination']))
        genomes_per_min = len(valid) / (wall_clock_s / 60) if wall_clock_s > 0 else 0

        print(f"\n  --- {s['name']} Summary ---")
        print(f"  Genomes: {len(valid)}")
        print(f"  Wall-clock: {wall_clock_s:.2f}s")
        print(f"  Completeness MAE: {comp_mae:.2f}%")
        print(f"  Contamination MAE: {cont_mae:.2f}%")
        print(f"  Speed: {genomes_per_min:.1f} genomes/min ({N_THREADS} thread)")

        all_results.append({'name': s['name'], 'n': len(valid), 'wall_s': wall_clock_s,
                            'comp_mae': comp_mae, 'cont_mae': cont_mae})

    # Overall summary
    print(f"\n\n{'='*60}")
    print("OVERALL DEEPCHECK v2 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Set':<20} {'N':>5} {'Time(s)':>8} {'Comp MAE':>10} {'Cont MAE':>10}")
    print('-'*60)
    for r in all_results:
        print(f"{r['name']:<20} {r['n']:>5} {r['wall_s']:>8.1f} {r['comp_mae']:>9.2f}% {r['cont_mae']:>9.2f}%")

    if all_results:
        total_n = sum(r['n'] for r in all_results)
        total_t = sum(r['wall_s'] for r in all_results)
        # Weighted average MAE
        avg_comp = sum(r['comp_mae']*r['n'] for r in all_results) / total_n
        avg_cont = sum(r['cont_mae']*r['n'] for r in all_results) / total_n
        print(f"\nTotal: {total_n} genomes, {total_t:.1f}s inference")
        print(f"Weighted avg comp MAE: {avg_comp:.2f}%")
        print(f"Weighted avg cont MAE: {avg_cont:.2f}%")
        print(f"NOTE: DeepCheck inference time does NOT include CheckM2 feature extraction time")


if __name__ == "__main__":
    main()
