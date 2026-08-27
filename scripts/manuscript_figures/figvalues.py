#!/usr/bin/env python3
"""Numeric ledger shared by the figure builders.

Every figure script flattens the values it plotted into a TSV under
``figures/_values/``.  ``verify_no_drift.py`` diffs those TSVs against the same
ledger produced by the frozen ``resubmission/`` scripts, which is how the
rebuild proves that recolouring and recomposing changed no number.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def flatten(prefix, obj, rows) -> None:
    """Recursively append (key, value) pairs for every scalar reachable in obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}|{k}", v, rows)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten(f"{prefix}[{i}]", v, rows)
    elif isinstance(obj, (bool, np.bool_)):
        rows.append((prefix, str(bool(obj))))
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        rows.append((prefix, float(obj)))
    elif isinstance(obj, np.ndarray):
        for i, v in enumerate(obj.ravel()):
            rows.append((f"{prefix}[{i}]", float(v)))
    elif isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            s = obj[c]
            if np.issubdtype(s.dtype, np.number):
                for i, v in zip(obj.index, s.values):
                    rows.append((f"{prefix}|{c}|{i}", float(v)))
    elif isinstance(obj, pd.Series):
        for i, v in obj.items():
            try:
                rows.append((f"{prefix}|{i}", float(v)))
            except (TypeError, ValueError):
                pass
    elif isinstance(obj, str):
        rows.append((prefix, obj))


def write_values(fig_dir: str, name: str, rows) -> str:
    out_dir = os.path.join(fig_dir, "_values")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.tsv")
    pd.DataFrame(rows, columns=["key", "value"]).to_csv(path, sep="\t", index=False)
    print(f"  value ledger -> {os.path.basename(path)} ({len(rows)} entries)")
    return path


def dump(fig_dir: str, name: str, obj) -> str:
    rows = []
    flatten(name, obj, rows)
    return write_values(fig_dir, name, rows)
