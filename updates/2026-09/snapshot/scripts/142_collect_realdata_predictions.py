#!/usr/bin/env python3
"""
WS3 Track A — collect MAGICC V5, CheckM2, CoCoPyE and DeepCheck predictions for a
real-data cohort into one tidy table.

Parsing conventions are taken verbatim from scripts/91_parse_competitor_clean_cd.py so
that real-data and synthetic-benchmark numbers are directly comparable:
  * CheckM2  — quality_report.tsv, columns Completeness / Contamination
  * CoCoPyE  — cocopye_raw_output.csv, stage-3 (marker + neural network) with stage-2
               fallback, scaled 0-1 -> 0-100 %
  * DeepCheck— CheckM2's --dbg_vectors PKLs -> vendored ResNet (dual-output work-around
               for the upstream forward() bug), CPU only

Usage:
    python scripts/142_collect_realdata_predictions.py --dir results/.../meslier \
        --truth results/.../meslier/bin_truth.tsv --id-col bin_id --out predictions.tsv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

signal.signal(signal.SIGHUP, signal.SIG_IGN)

ROOT = Path("/media/Data_1/tianrm/projects/magicc2")
DEEPCHECK_DIR = ROOT / "tools" / "DeepCheck"
SCRIPT_38 = ROOT / "scripts" / "38_run_deepcheck_v2.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def parse_checkm2(d: Path):
    qr = d / "checkm2_output" / "quality_report.tsv"
    if not qr.is_file():
        return None
    c = pd.read_csv(qr, sep="\t")
    return pd.DataFrame({"genome_id": c["Name"].astype(str),
                         "checkm2_completeness": c["Completeness"].astype(float),
                         "checkm2_contamination": c["Contamination"].astype(float)})


def parse_cocopye(d: Path):
    raw = d / "cocopye_raw_output.csv"
    if not raw.is_file():
        return None
    c = pd.read_csv(raw)
    if "3_completeness" in c.columns:
        comp = c["3_completeness"].fillna(c.get("2_completeness")) * 100.0
        cont = c["3_contamination"].fillna(c.get("2_contamination")) * 100.0
    elif "2_completeness" in c.columns:
        comp = c["2_completeness"] * 100.0
        cont = c["2_contamination"] * 100.0
    else:
        return None
    return pd.DataFrame({"genome_id": c["bin"].astype(str),
                         "cocopye_completeness": comp.astype(float),
                         "cocopye_contamination": cont.astype(float)})


def parse_deepcheck(d: Path, torch_threads=4):
    ck = d / "checkm2_output"
    pkls = sorted(ck.glob("*.pkl"))
    if not pkls or not SCRIPT_38.is_file():
        return None
    import torch
    m38 = load_module(SCRIPT_38, "dc38")
    sp = np.load(DEEPCHECK_DIR / "scaler_params.npz")
    model = m38.ResNetDualOutput(m38.ResidualBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(DEEPCHECK_DIR / "models" / "best_model.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    feats = pd.concat([pd.read_pickle(p) for p in pkls], ignore_index=True)
    names = feats["Name"].astype(str).values
    fm = feats.iloc[:, 1:].values.astype(float)
    scaled = (fm * sp["scale"] + sp["min_val"])[:, :20021]
    padded = np.zeros((scaled.shape[0], 20164), dtype=np.float32)
    padded[:, :20021] = scaled
    from torch.utils.data import DataLoader
    torch.set_num_threads(torch_threads)
    comp, cont = [], []
    with torch.no_grad():
        for batch in DataLoader(m38.DeepCheckDataset(padded), batch_size=64,
                                shuffle=False, num_workers=0):
            cp, cx = model(batch)
            comp.extend((cp.squeeze() * 100).cpu().numpy().tolist())
            cont.extend((cx.squeeze() * 100).cpu().numpy().tolist())
    return pd.DataFrame({"genome_id": names,
                         "deepcheck_completeness": comp,
                         "deepcheck_contamination": cont})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results dir holding tool outputs")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--id-col", required=True, help="truth column holding genome_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--magicc", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    truth = pd.read_csv(args.truth, sep="\t")
    truth["genome_id"] = truth[args.id_col].astype(str)
    n0 = len(truth)

    mg = Path(args.magicc or (d / "magicc_v5_predictions.tsv"))
    m = pd.read_csv(mg, sep="\t")
    m["genome_id"] = m["genome_id"].astype(str)
    out = truth.merge(m[["genome_id", "magicc_completeness", "magicc_contamination"]],
                      on="genome_id", how="left")

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "dir": str(d), "n_truth_rows": n0,
              "magicc_matched": int(out["magicc_completeness"].notna().sum())}

    for name, fn in (("checkm2", parse_checkm2), ("cocopye", parse_cocopye),
                     ("deepcheck", parse_deepcheck)):
        try:
            t = fn(d)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED — {type(e).__name__}: {e}")
            report[name] = f"failed: {e}"
            continue
        if t is None:
            print(f"  {name}: absent")
            report[name] = "absent"
            continue
        t = t.drop_duplicates("genome_id")
        out = out.merge(t, on="genome_id", how="left")
        col = f"{name}_completeness"
        report[f"{name}_matched"] = int(out[col].notna().sum())
        print(f"  {name}: {report[f'{name}_matched']}/{n0} matched")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out} ({len(out)} rows)")
    with open(str(args.out).replace(".tsv", "_merge_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
