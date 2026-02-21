#!/usr/bin/env python3
"""
Run DeepCheck on motivating benchmark sets A and B.
Uses CheckM2 intermediate feature vectors (PKL files).
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
BENCHMARK_DIR = os.path.join(PROJECT_DIR, "data", "benchmarks", "motivating")

sys.path.insert(0, DEEPCHECK_DIR)


# ============================================================
# DeepCheck Model (with both comp and cont output)
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
    pkl_files = sorted([f for f in os.listdir(checkm2_output_dir) if f.endswith('.pkl')])
    if not pkl_files:
        raise FileNotFoundError(f"No PKL files found in {checkm2_output_dir}")
    dfs = []
    for pkl in pkl_files:
        df = pd.read_pickle(os.path.join(checkm2_output_dir, pkl))
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(full_df)} genomes from {len(pkl_files)} PKL files")
    return full_df


def main():
    device = torch.device("cpu")
    print(f"DeepCheck Motivating Benchmark Runner")
    print(f"Using device: {device}")

    # Load scaler
    scaler_params_path = os.path.join(DEEPCHECK_DIR, "scaler_params.npz")
    scaler_params = np.load(scaler_params_path)
    scaler_scale = scaler_params['scale']
    scaler_min = scaler_params['min_val']
    print(f"Loaded scaler params: scale shape={scaler_scale.shape}")

    # Load model
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

    for set_name in ["A", "B"]:
        set_dir = os.path.join(BENCHMARK_DIR, f"set_{set_name}")
        checkm2_output_dir = os.path.join(set_dir, "checkm2_output")
        metadata_path = os.path.join(set_dir, "metadata.tsv")
        output_path = os.path.join(set_dir, "deepcheck_predictions.tsv")

        print(f"\n{'='*60}")
        print(f"Set {set_name}")
        print(f"{'='*60}")

        # Check if already done
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path, sep="\t")
            metadata = pd.read_csv(metadata_path, sep="\t")
            if len(existing) >= len(metadata):
                print(f"  SKIPPING: Output already exists with {len(existing)} results")
                continue

        # Load CheckM2 feature vectors
        try:
            features_df = load_checkm2_features(checkm2_output_dir)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        names = features_df['Name'].values
        feature_matrix = features_df.iloc[:, 1:].values.astype(float)
        print(f"  Feature matrix shape: {feature_matrix.shape}")

        # Scale
        scaled_features = feature_matrix * scaler_scale + scaler_min
        scaled_features = scaled_features[:, :20021]

        # Zero-pad
        n_samples = scaled_features.shape[0]
        padded_features = np.zeros((n_samples, 20164), dtype=np.float32)
        padded_features[:, :20021] = scaled_features

        dataset = DeepCheckDataset(padded_features)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        all_comp = []
        all_cont = []
        start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                comp_pred, cont_pred = model(batch)
                all_comp.extend((comp_pred.squeeze() * 100).cpu().numpy().tolist())
                all_cont.extend((cont_pred.squeeze() * 100).cpu().numpy().tolist())

        wall_clock_s = time.time() - start_time
        print(f"  Inference time: {wall_clock_s:.2f}s ({len(names)/wall_clock_s:.1f} genomes/sec)")

        results = pd.DataFrame({
            'genome_id': names,
            'pred_completeness': all_comp,
            'pred_contamination': all_cont
        })

        metadata = pd.read_csv(metadata_path, sep="\t")
        merged = metadata.merge(results, on='genome_id', how='left')
        merged['wall_clock_s'] = wall_clock_s
        merged['n_threads'] = N_THREADS

        n_missing = merged['pred_completeness'].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} genomes missing predictions")

        merged.to_csv(output_path, sep="\t", index=False)
        print(f"  Saved: {output_path}")

        valid = merged.dropna(subset=['pred_completeness', 'pred_contamination'])
        comp_mae = np.mean(np.abs(valid['true_completeness'] - valid['pred_completeness']))
        cont_mae = np.mean(np.abs(valid['true_contamination'] - valid['pred_contamination']))
        genomes_per_min = len(valid) / (wall_clock_s / 60) if wall_clock_s > 0 else 0

        print(f"\n  --- Set {set_name} Summary ---")
        print(f"  Genomes: {len(valid)}")
        print(f"  Completeness MAE: {comp_mae:.2f}%")
        print(f"  Contamination MAE: {cont_mae:.2f}%")
        print(f"  Speed: {genomes_per_min:.1f} genomes/min (inference only, {N_THREADS} thread)")


if __name__ == "__main__":
    main()
