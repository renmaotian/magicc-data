#!/usr/bin/env python3
"""
Memory benchmark: DeepCheck inference on Set E (1,000 genomes).

Uses pre-computed CheckM2 feature vectors from Set E.
Inference only, 1 thread (CPU).

Run with: /usr/bin/time -v conda run -n magicc2 python scripts/44_memory_benchmark_deepcheck.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = "/home/tianrm/projects/magicc2"
DEEPCHECK_DIR = os.path.join(PROJECT_DIR, "tools", "DeepCheck")
SET_E_DIR = os.path.join(PROJECT_DIR, "data", "benchmarks", "set_E")

# Limit to 1 thread
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ============================================================
# Model definition (exact copy from script 28c)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        self.fc1 = nn.Linear(100, num_classes)
        self.fc2 = nn.Linear(100, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = [block(self.in_channels, out_channels, stride, downsample)]
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
        comp = self.fc1(x)
        cont = self.fc2(x)
        return comp, cont


class DeepCheckDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].reshape(142, 142)
        sample = torch.FloatTensor(sample).unsqueeze(0)
        return sample


def main():
    t0 = time.time()
    print("DeepCheck Memory Benchmark on Set E")
    print(f"  Using 1 thread (inference only)")

    # Load model
    model_path = os.path.join(DEEPCHECK_DIR, "models", "best_model.pt")
    model = ResNetDualOutput(ResidualBlock, [2, 2, 2, 2])
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load scaler
    scaler_path = os.path.join(DEEPCHECK_DIR, "scaler_params.npz")
    scaler_params = np.load(scaler_path)
    scaler_scale = scaler_params['scale']
    scaler_min = scaler_params['min_val']
    print(f"  Scaler loaded: {scaler_scale.shape[0]} features")

    # Load features
    checkm2_dir = os.path.join(SET_E_DIR, "checkm2_output")
    pkl_files = sorted([f for f in os.listdir(checkm2_dir) if f.endswith('.pkl')])
    dfs = []
    for pkl in pkl_files:
        df = pd.read_pickle(os.path.join(checkm2_dir, pkl))
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(full_df)} feature vectors from {len(pkl_files)} PKL files")

    names = full_df['Name'].values
    feature_matrix = full_df.iloc[:, 1:].values.astype(float)

    # Scale
    scaled_features = feature_matrix * scaler_scale + scaler_min
    scaled_features = scaled_features[:, :20021]

    # Pad to 20164 (142*142)
    n_samples = scaled_features.shape[0]
    padded_features = np.zeros((n_samples, 20164), dtype=np.float32)
    padded_features[:, :20021] = scaled_features

    # Create dataset
    dataset = DeepCheckDataset(padded_features)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Inference
    print(f"  Running inference...")
    t_infer = time.time()
    all_comp = []
    all_cont = []
    with torch.no_grad():
        for batch in loader:
            comp, cont = model(batch)
            all_comp.extend((comp.squeeze() * 100).cpu().numpy().tolist())
            all_cont.extend((cont.squeeze() * 100).cpu().numpy().tolist())

    infer_time = time.time() - t_infer
    print(f"  Inference time: {infer_time:.1f}s ({len(names)} genomes)")
    print(f"  Speed: {len(names)/infer_time*60:.0f} genomes/min")

    # Merge with metadata
    metadata = pd.read_csv(os.path.join(SET_E_DIR, "metadata.tsv"), sep='\t')
    results = pd.DataFrame({
        'genome_id': names,
        'pred_completeness': all_comp,
        'pred_contamination': all_cont
    })
    merged = metadata.merge(results, on='genome_id', how='left')

    # Quick accuracy
    valid = merged.dropna(subset=['pred_completeness', 'pred_contamination'])
    comp_mae = np.mean(np.abs(valid['true_completeness'] - valid['pred_completeness']))
    cont_mae = np.mean(np.abs(valid['true_contamination'] - valid['pred_contamination']))
    print(f"  Completeness MAE: {comp_mae:.2f}%")
    print(f"  Contamination MAE: {cont_mae:.2f}%")

    out_path = os.path.join(PROJECT_DIR, "results", "memory_benchmarks", "deepcheck_predictions.tsv")
    merged.to_csv(out_path, sep='\t', index=False)
    print(f"  Saved: {out_path}")

    total = time.time() - t0
    print(f"  Total time: {total:.1f}s")


if __name__ == '__main__':
    main()
