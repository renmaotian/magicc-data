#!/usr/bin/env python3
"""
WS8.1 — DeepCheck inference, thread-parameterised.

Model definition copied verbatim from scripts/44_memory_benchmark_deepcheck.py
(which itself copies scripts/28c). The only changes are (a) the thread count and
feature directory are arguments, and (b) the timer prints a phase breakdown.

DeepCheck cannot read FASTA: it consumes CheckM2's pickled feature vectors, so
its *end-to-end* cost is CheckM2's cost plus this. Both are reported.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True, help="dir containing CheckM2 feature_vectors_*.pkl")
ap.add_argument("--output", required=True)
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()

# Thread pinning must happen before torch import.
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import torch                                                      # noqa: E402
import torch.nn as nn                                             # noqa: E402
from torch.utils.data import Dataset, DataLoader                  # noqa: E402

torch.set_num_threads(args.threads)

DEEPCHECK_DIR = "/path/to/magicc/tools/DeepCheck"


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, h * w)
        a = self.softmax(torch.bmm(q, k))
        v = self.value_conv(x).view(b, -1, h * w)
        return torch.bmm(v, a.permute(0, 2, 1)).view(b, c, h, w)


class ResNetDualOutput(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
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
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.attention(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(self.dropout(x))
        return self.fc1(x), self.fc2(x)


class DeepCheckDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx].reshape(142, 142)).unsqueeze(0)


def main() -> int:
    t0 = time.time()
    model = ResNetDualOutput(ResidualBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(os.path.join(DEEPCHECK_DIR, "models", "best_model.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    sp = np.load(os.path.join(DEEPCHECK_DIR, "scaler_params.npz"))
    t_model = time.time() - t0

    t1 = time.time()
    pkls = sorted(f for f in os.listdir(args.features) if f.endswith(".pkl"))
    if not pkls:
        print(f"ERROR: no feature_vectors_*.pkl in {args.features}", file=sys.stderr)
        return 1
    full = pd.concat([pd.read_pickle(os.path.join(args.features, p)) for p in pkls],
                     ignore_index=True)
    names = full["Name"].values
    feat = full.iloc[:, 1:].values.astype(float)
    scaled = (feat * sp["scale"] + sp["min_val"])[:, :20021]
    padded = np.zeros((scaled.shape[0], 20164), dtype=np.float32)
    padded[:, :20021] = scaled
    t_feat = time.time() - t1

    t2 = time.time()
    loader = DataLoader(DeepCheckDataset(padded), batch_size=64, shuffle=False, num_workers=0)
    comps, conts = [], []
    with torch.no_grad():
        for batch in loader:
            c, x = model(batch)
            comps.extend((c.squeeze() * 100).cpu().numpy().tolist())
            conts.extend((x.squeeze() * 100).cpu().numpy().tolist())
    t_infer = time.time() - t2

    pd.DataFrame({"genome_id": names,
                  "pred_completeness": comps,
                  "pred_contamination": conts}).to_csv(args.output, sep="\t", index=False)

    print(f"threads={args.threads} n={len(names)} model_load_s={t_model:.2f} "
          f"feature_load_s={t_feat:.2f} inference_s={t_infer:.2f} "
          f"total_s={time.time()-t0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
