#!/usr/bin/env python
"""
101_metrics_framework.py
========================
Reusable, config-driven analysis framework for the MAGICC Nature Communications
major revision, workstream WS5 (MIMAG-centric metrics, threshold analysis,
signed errors, clustered statistics, distribution plots).

This module is BOTH a library (imported by 102-105) and a CLI:

    python scripts/101_metrics_framework.py --selftest    # validation suite (run first)
    python scripts/101_metrics_framework.py --inventory   # write the data inventory

Design goals
------------
*   Nothing about the benchmark sets or tools is hard-coded. Everything comes
    from ``scripts/config_revision_metrics.yaml``. New benchmark sets are added
    by appending to that file (or are auto-discovered on disk).
*   Every statistic that is reported with a confidence interval goes through one
    generic bootstrap engine that supports **cluster resampling** (resample
    clusters of correlated observations, not individual observations).
*   All statistical primitives (cluster bootstrap, Cliff's delta, matched-pairs
    rank-biserial correlation, Hodges-Lehmann estimator, Benjamini-Hochberg)
    are validated by ``--selftest`` against hand-checkable cases, brute-force
    reference implementations, a coverage simulation, and SciPy where available.
*   Read-only on data/. Writes only under ``results/revision/metrics/``.

Author: Stage A revision analysis (WS5.1-5.7)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Limit BLAS threads *before* numpy is imported so concurrent agents are not starved.
_CFG_PATH_DEFAULT = Path(__file__).resolve().parent / "config_revision_metrics.yaml"


def _preset_blas_threads(n: str = "1") -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, n)


_preset_blas_threads()

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import yaml                             # noqa: E402
from scipy import stats as sps          # noqa: E402

__all__ = [
    "load_config", "Config", "BenchmarkSet", "discover_sets", "load_set",
    "mimag_classify", "confusion_matrix", "classification_metrics",
    "threshold_metrics", "contamination_bin", "phylum_group",
    "encode_labels", "cm_from_codes", "metrics_from_cm",
    "Bootstrapper", "bootstrap_ci", "cliffs_delta", "rank_biserial_paired",
    "hodges_lehmann_paired", "bh_correct", "mae", "rmse", "bias",
    "stable_hash",
    "r2_coefficient_of_determination", "r2_pearson_squared",
    "simulate_cvd", "palette_cvd_report", "save_figure", "caption_denominator",
    "md_table", "init_matplotlib",
]

# ---------------------------------------------------------------------------
# Deterministic seed derivation
# ---------------------------------------------------------------------------


def stable_hash(s: str) -> int:
    """Process-independent, platform-independent hash of a string.

    Python's built-in ``hash()`` for str/bytes is salted per interpreter process
    (PYTHONHASHSEED), so seeds derived from it differ between runs and the
    bootstrap confidence intervals are NOT reproducible. Every derived seed in
    scripts 102-106 therefore goes through this function instead. CRC-32 is not
    a cryptographic hash, but it is fully specified, stable across Python
    versions and platforms, and only ever used to spread seeds.
    """
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    raw: dict
    path: Path

    # convenience accessors -------------------------------------------------
    @property
    def project_root(self) -> Path:
        return Path(self.raw["project_root"])

    @property
    def benchmark_root(self) -> Path:
        return self.project_root / self.raw["benchmark_root"]

    @property
    def splits_dir(self) -> Path:
        return self.project_root / self.raw["splits_dir"]

    @property
    def out_dir(self) -> Path:
        d = self.project_root / self.raw["output_root"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def fig_dir(self) -> Path:
        d = self.out_dir / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def seed(self) -> int:
        return int(self.raw["random_seed"])

    @property
    def n_jobs(self) -> int:
        return int(self.raw.get("n_jobs", 8))

    @property
    def n_boot(self) -> int:
        return int(self.raw["bootstrap"]["n_iter"])

    @property
    def n_boot_slow(self) -> int:
        return int(self.raw["bootstrap"]["n_iter_slow"])

    @property
    def ci_level(self) -> float:
        return float(self.raw["bootstrap"]["ci_level"])

    @property
    def cluster_col(self) -> str:
        return self.raw["bootstrap"]["cluster_default"]

    @property
    def mimag_classes(self) -> List[str]:
        return list(self.raw["mimag"]["classes"])

    @property
    def tools(self) -> Dict[str, dict]:
        return self.raw["tools"]

    def tool_label(self, tool: str) -> str:
        return self.tools.get(tool, {}).get("label", tool)

    def tool_short(self, tool: str) -> str:
        return self.tools.get(tool, {}).get("short", tool)

    def tool_role(self, tool: str) -> str:
        return self.tools.get(tool, {}).get("role", "unknown")

    def tool_colour(self, tool: str) -> str:
        return self.raw["palette"]["tools"].get(tool, "#000000")

    def tool_marker(self, tool: str) -> str:
        return self.raw["palette"].get("markers", {}).get(tool, "o")


def load_config(path: os.PathLike | str | None = None) -> Config:
    p = Path(path) if path else _CFG_PATH_DEFAULT
    with open(p) as fh:
        raw = yaml.safe_load(fh)
    _preset_blas_threads(str(raw.get("blas_threads", 1)))
    return Config(raw=raw, path=p)


# ---------------------------------------------------------------------------
# Benchmark-set discovery and loading
# ---------------------------------------------------------------------------

REQUIRED_METADATA_COLS = ("genome_id", "true_completeness", "true_contamination")


DEFAULT_TIER_BY_STATUS = {
    "current": "primary",
    "superseded_leaky": "reported",
    "legacy_superseded": "secondary",
    "motivating": "secondary",
}


@dataclass
class BenchmarkSet:
    name: str
    directory: Path
    label: str
    status: str
    design: str
    relatedness_rule: str
    tier: str = "primary"
    listed: bool = True
    present: bool = False
    superseded_by: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.tsv"

    @property
    def labels_path(self) -> Path:
        return self.directory / "labels.npy"


def discover_sets(cfg: Config, include_missing: bool = True,
                  tiers: Optional[Sequence[str]] = None) -> List[BenchmarkSet]:
    """Resolve the configured sets plus anything new found on disk.

    ``tiers`` optionally restricts the result to the given analysis tiers
    (``primary`` / ``reported`` / ``secondary``).
    """
    out: List[BenchmarkSet] = []
    listed_dirs = set()
    for entry in cfg.raw.get("sets", []):
        d = cfg.benchmark_root / entry["dir"]
        listed_dirs.add(entry["dir"])
        status = entry.get("status", "current")
        bs = BenchmarkSet(
            name=entry["name"], directory=d,
            label=entry.get("label", entry["name"]),
            status=status,
            design=entry.get("design", ""),
            relatedness_rule=entry.get("relatedness_rule", "from_sample_type"),
            tier=entry.get("tier", DEFAULT_TIER_BY_STATUS.get(status, "primary")),
            listed=True,
            present=(d / "metadata.tsv").exists(),
            superseded_by=entry.get("superseded_by"),
        )
        if bs.present or include_missing:
            out.append(bs)

    ad = cfg.raw.get("autodiscover", {})
    if ad.get("enabled", False):
        found = []
        for pat in ad.get("patterns", []):
            found.extend(sorted(cfg.benchmark_root.glob(pat)))
        for d in found:
            rel = str(d.relative_to(cfg.benchmark_root))
            if rel in listed_dirs or not d.is_dir():
                continue
            if not (d / "metadata.tsv").exists():
                continue
            st = ad.get("default_status", "current")
            out.append(BenchmarkSet(
                name=rel.replace("/", "__"), directory=d, label=rel,
                status=st,
                design="auto-discovered on disk; not described in the config",
                relatedness_rule=ad.get("default_relatedness_rule", "from_sample_type"),
                tier=DEFAULT_TIER_BY_STATUS.get(st, "primary"),
                listed=False, present=True,
            ))
    if tiers is not None:
        out = [b for b in out if b.tier in tiers]
    return out


_SPLIT_MAP_CACHE: Optional[Dict[str, Tuple[str, str, str]]] = None


def _accession_variants(acc: str) -> List[str]:
    """All plausible spellings of one GTDB/NCBI assembly accession.

    GTDB prefixes RefSeq accessions with ``RS_`` and GenBank accessions with
    ``GB_``. GCA_<num>.<v> and GCF_<num>.<v> denote the *same* assembly
    deposited in GenBank vs RefSeq, so the numeric part is the reliable key.
    """
    acc = str(acc)
    bare = re.sub(r"^(RS_|GB_)", "", acc)
    flip = (bare.replace("GCA_", "GCF_") if bare.startswith("GCA_")
            else bare.replace("GCF_", "GCA_") if bare.startswith("GCF_") else bare)
    cands = [acc, bare, flip]
    for b in (bare, flip):
        cands += ["RS_" + b, "GB_" + b]
    seen, uniq = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def split_provenance_map(cfg: Config) -> Dict[str, Tuple[str, str, str]]:
    """accession-variant -> (split, domain, phylum) for every curated genome."""
    global _SPLIT_MAP_CACHE
    if _SPLIT_MAP_CACHE is not None:
        return _SPLIT_MAP_CACHE
    m: Dict[str, Tuple[str, str, str]] = {}
    for split in ("train", "val", "test"):
        p = cfg.splits_dir / f"{split}_genomes.tsv"
        if not p.exists():
            continue
        d = pd.read_csv(p, sep="\t",
                        usecols=["gtdb_accession", "ncbi_accession", "domain", "phylum"])
        for gtdb, ncbi, dom, phy in zip(d.gtdb_accession, d.ncbi_accession,
                                        d.domain, d.phylum):
            val = (split, str(dom), str(phy))
            for k in _accession_variants(gtdb):
                m.setdefault(k, val)
            if isinstance(ncbi, str) and ncbi:
                for k in _accession_variants(ncbi):
                    m.setdefault(k, val)
    _SPLIT_MAP_CACHE = m
    return m


def _lookup_split(acc: str, m: Dict[str, Tuple[str, str, str]]) -> Tuple[str, str, str]:
    for k in _accession_variants(acc):
        if k in m:
            return m[k]
    return ("absent", "unknown", "unknown")


def resolve_relatedness(df: pd.DataFrame, rule: str) -> pd.Series:
    """Contamination-source relatedness per row.

    Rows with zero true contamination are always ``none (uncontaminated)``.
    Otherwise the rule is applied:
      * ``from_sample_type`` -- look for within_phylum / cross_phylum in
        ``sample_type`` (this is how the generators name the two designs).
      * ``fixed:<value>``    -- constant value taken from the generator source.
    """
    n = len(df)
    if rule.startswith("fixed:"):
        base = pd.Series([rule.split(":", 1)[1]] * n, index=df.index, dtype=object)
    else:
        st = (df["sample_type"].astype(str) if "sample_type" in df.columns
              else pd.Series([""] * n, index=df.index, dtype=object))
        base = pd.Series(["unknown"] * n, index=df.index, dtype=object)
        base[st.str.contains("within_phylum", na=False)] = "within_phylum"
        base[st.str.contains("cross_phylum", na=False)] = "cross_phylum"
    base = base.astype(object)
    base[df["true_contamination"].to_numpy() <= 0] = "none (uncontaminated)"
    return base


def load_set(cfg: Config, bs: BenchmarkSet) -> Tuple[pd.DataFrame, List[str], dict]:
    """Load one benchmark set into a wide frame.

    Returns ``(frame, available_tools, provenance)``.

    Frame columns: metadata columns, plus for every available tool
    ``pred_completeness__<tool>`` / ``pred_contamination__<tool>`` and the derived
    ``err_*``/``abs_err_*`` columns, plus ``relatedness``, ``dominant_split``,
    ``dominant_domain``, ``true_mimag``, ``true_cont_bin``.
    """
    md = pd.read_csv(bs.metadata_path, sep="\t")
    missing = [c for c in REQUIRED_METADATA_COLS if c not in md.columns]
    if missing:
        raise ValueError(f"{bs.name}: metadata.tsv missing required columns {missing}")

    prov: dict = {"set": bs.name, "n_metadata": int(len(md)),
                  "metadata_columns": list(md.columns)}

    # ground-truth cross-check against labels.npy ---------------------------
    if bs.labels_path.exists():
        lab = np.load(bs.labels_path)
        if lab.shape[0] == len(md):
            prov["labels_max_abs_diff_completeness"] = float(
                np.abs(lab[:, 0] - md.true_completeness.to_numpy()).max())
            prov["labels_max_abs_diff_contamination"] = float(
                np.abs(lab[:, 1] - md.true_contamination.to_numpy()).max())
        else:
            prov["labels_shape_mismatch"] = f"{lab.shape} vs metadata n={len(md)}"

    df = md.copy()

    # cluster column --------------------------------------------------------
    ccol = cfg.cluster_col
    if ccol not in df.columns:
        warnings.warn(f"{bs.name}: cluster column '{ccol}' absent; each row is its "
                      f"own cluster (statistics fall back to iid resampling)")
        df[ccol] = df["genome_id"].astype(str)
    df["cluster_id"] = df[ccol].astype(str)

    # provenance of the dominant genome -------------------------------------
    smap = split_provenance_map(cfg)
    trip = [_lookup_split(a, smap) for a in df["dominant_accession"].astype(str)] \
        if "dominant_accession" in df.columns else [("absent", "unknown", "unknown")] * len(df)
    df["dominant_split"] = [t[0] for t in trip]
    df["dominant_domain"] = [t[1] for t in trip]
    if "dominant_phylum" not in df.columns:
        df["dominant_phylum"] = [t[2] for t in trip]

    # derived strata --------------------------------------------------------
    df["relatedness"] = resolve_relatedness(df, bs.relatedness_rule)
    df["true_mimag"] = mimag_classify(df.true_completeness.to_numpy(),
                                      df.true_contamination.to_numpy(), cfg)
    df["true_cont_bin"] = contamination_bin(df.true_contamination.to_numpy(), cfg)

    # predictions -----------------------------------------------------------
    available: List[str] = []
    prov["tools"] = {}
    for tool, tcfg in cfg.tools.items():
        p = bs.directory / tcfg["file"]
        if not p.exists():
            prov["tools"][tool] = {"present": False}
            continue
        pr = pd.read_csv(p, sep="\t")
        idc = "genome_id" if "genome_id" in pr.columns else (
            "genome_name" if "genome_name" in pr.columns else None)
        if idc is None or "pred_completeness" not in pr.columns:
            prov["tools"][tool] = {"present": True, "usable": False,
                                   "reason": "no genome id / pred_completeness column",
                                   "columns": list(pr.columns)}
            continue
        sub = pr[[idc, "pred_completeness", "pred_contamination"]].rename(
            columns={idc: "genome_id"})
        info = {"present": True, "usable": True, "file": tcfg["file"],
                "n_rows": int(len(pr)), "n_unique_ids": int(sub.genome_id.nunique()),
                "columns": list(pr.columns),
                "n_nan_completeness": int(sub.pred_completeness.isna().sum()),
                "n_nan_contamination": int(sub.pred_contamination.isna().sum())}
        # ground-truth consistency check when the tool file carries truth columns
        if "true_completeness" in pr.columns:
            chk = df[["genome_id", "true_completeness", "true_contamination"]].merge(
                pr[[idc, "true_completeness", "true_contamination"]].rename(
                    columns={idc: "genome_id"}), on="genome_id", suffixes=("_md", "_pr"))
            info["truth_max_abs_diff_completeness"] = float(
                np.abs(chk.true_completeness_md - chk.true_completeness_pr).max())
            info["truth_max_abs_diff_contamination"] = float(
                np.abs(chk.true_contamination_md - chk.true_contamination_pr).max())
        n_before = len(df)
        df = df.merge(sub.rename(columns={
            "pred_completeness": f"pred_completeness__{tool}",
            "pred_contamination": f"pred_contamination__{tool}"}),
            on="genome_id", how="left")
        assert len(df) == n_before, f"{bs.name}/{tool}: merge duplicated rows"
        info["n_matched"] = int(df[f"pred_completeness__{tool}"].notna().sum())
        prov["tools"][tool] = info
        available.append(tool)

    # errors and predicted MIMAG class --------------------------------------
    for tool in available:
        for metric, tcol in (("completeness", "true_completeness"),
                             ("contamination", "true_contamination")):
            pc = f"pred_{metric}__{tool}"
            df[f"err_{metric}__{tool}"] = df[pc] - df[tcol]
            df[f"abs_err_{metric}__{tool}"] = (df[pc] - df[tcol]).abs()
        df[f"mimag__{tool}"] = mimag_classify(
            df[f"pred_completeness__{tool}"].to_numpy(),
            df[f"pred_contamination__{tool}"].to_numpy(), cfg)

    prov["n_clusters"] = int(df["cluster_id"].nunique())
    prov["max_cluster_size"] = int(df["cluster_id"].value_counts().max())
    prov["dominant_split_counts"] = {k: int(v) for k, v in
                                     df["dominant_split"].value_counts().items()}
    prov["available_tools"] = available
    return df, available, prov


# ---------------------------------------------------------------------------
# MIMAG-inspired classification and strata
# ---------------------------------------------------------------------------


def mimag_classify(completeness, contamination, cfg: Config) -> np.ndarray:
    """Hierarchical MIMAG-inspired 3-class label (completeness/contamination only).

    high   : completeness >= high.completeness_min  AND contamination < high.contamination_max
    medium : completeness >= medium.completeness_min AND contamination < medium.contamination_max
    low    : otherwise

    NaN in either input yields ``"na"``.
    """
    m = cfg.raw["mimag"]
    comp = np.asarray(completeness, dtype=float)
    cont = np.asarray(contamination, dtype=float)
    ge = np.greater_equal if m.get("completeness_inclusive", True) else np.greater
    lt = np.less_equal if m.get("contamination_inclusive", False) else np.less

    out = np.full(comp.shape, "low", dtype=object)
    hi = ge(comp, m["high"]["completeness_min"]) & lt(cont, m["high"]["contamination_max"])
    med = ge(comp, m["medium"]["completeness_min"]) & lt(cont, m["medium"]["contamination_max"])
    out[med & ~hi] = "medium"
    out[hi] = "high"
    out[np.isnan(comp) | np.isnan(cont)] = "na"
    return out


def contamination_bin(contamination, cfg: Config) -> np.ndarray:
    spec = cfg.raw["strata"]["contamination_bins"]
    edges = np.asarray(spec["edges"], dtype=float)
    labels = list(spec["labels"])
    x = np.asarray(contamination, dtype=float)
    # right-open bins [e_i, e_{i+1}); the first bin is exactly zero
    idx = np.clip(np.digitize(x, edges[1:], right=False), 0, len(labels) - 1)
    out = np.array([labels[i] for i in idx], dtype=object)
    out[np.isnan(x)] = "na"
    return out


def phylum_group(phyla: Iterable[str], min_n: int) -> np.ndarray:
    s = pd.Series(list(phyla), dtype=object)
    vc = s.value_counts()
    keep = set(vc[vc >= min_n].index)
    return np.array([p if p in keep else f"Other (n<{min_n})" for p in s], dtype=object)


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


def mae(true, pred) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(true, float))))


def rmse(true, pred) -> float:
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sqrt(np.mean(d ** 2)))


def bias(true, pred) -> float:
    """Mean signed error (predicted - true)."""
    return float(np.mean(np.asarray(pred, float) - np.asarray(true, float)))


def r2_coefficient_of_determination(true, pred) -> float:
    """R^2 = 1 - SS_res/SS_tot. NaN when the truth has zero variance."""
    y = np.asarray(true, float)
    yh = np.asarray(pred, float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - yh) ** 2) / ss_tot)


def r2_pearson_squared(true, pred) -> float:
    """Squared Pearson correlation. NOT the coefficient of determination.

    Reported separately because the submitted manuscript's Table S2c mixes the
    two (see the discrepancy note in table_S2_rebuilt).
    """
    y = np.asarray(true, float)
    yh = np.asarray(pred, float)
    if y.std() == 0 or yh.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, yh)[0, 1] ** 2)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def encode_labels(y, labels: Sequence[str]) -> np.ndarray:
    """Map string class labels to integer codes; unknown/NaN -> -1."""
    idx = {l: i for i, l in enumerate(labels)}
    return np.array([idx.get(v, -1) for v in y], dtype=np.int64)


def cm_from_codes(t_code: np.ndarray, p_code: np.ndarray, k: int) -> np.ndarray:
    """Fast confusion matrix from integer codes (rows=true, cols=pred).

    Rows with a negative code in either vector are dropped.
    """
    ok = (t_code >= 0) & (p_code >= 0)
    t, p = t_code[ok], p_code[ok]
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k).astype(np.int64)


def confusion_matrix(y_true, y_pred, labels: Sequence[str]) -> np.ndarray:
    """rows = true class, cols = predicted class, in `labels` order."""
    return cm_from_codes(encode_labels(y_true, labels),
                         encode_labels(y_pred, labels), len(labels))


def metrics_from_cm(cm: np.ndarray, labels: Sequence[str],
                    observed_mask: Optional[np.ndarray] = None) -> dict:
    """Per-class and averaged classification metrics from a confusion matrix.

    ``observed_mask`` fixes which classes enter the ``*_observed`` macro
    averages. Pass the mask derived from the FULL dataset when bootstrapping,
    otherwise the estimator silently changes definition between replicates
    (a class with support 2 is absent from most resamples) and the resulting
    interval is meaningless.
    """
    support = cm.sum(axis=1)
    pred_n = cm.sum(axis=0)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    total = cm.sum()
    out = {"labels": list(labels), "confusion": cm,
           "support": support.astype(int), "predicted_count": pred_n.astype(int),
           "precision": prec, "recall": rec, "f1": f1,
           "accuracy": float(tp.sum() / total) if total else float("nan"),
           "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
           "macro_f1": float(f1.mean()),
           # micro-averaged P = R = F1 = accuracy in single-label multi-class
           "micro_f1": float(tp.sum() / total) if total else float("nan"),
           "balanced_accuracy": float(np.mean(rec[support > 0])) if (support > 0).any() else float("nan"),
           }
    w = support / total if total else support * 0.0
    out["weighted_f1"] = float(np.sum(w * f1))
    # Macro averages restricted to classes that actually occur in the ground
    # truth. Reported alongside the all-class macro average because several
    # benchmark sets contain no genomes of one class BY DESIGN (e.g. Set A has
    # zero low-quality genomes), which drags the 3-class macro F1 down by a
    # factor of 2/3 and is not a property of the tool.
    obs = (support > 0) if observed_mask is None else np.asarray(observed_mask, bool)
    out["n_classes_observed"] = int(obs.sum())
    out["classes_absent_in_truth"] = ",".join(
        l for l, o in zip(labels, obs) if not o)
    if obs.any():
        out["macro_f1_observed"] = float(f1[obs].mean())
        out["macro_precision_observed"] = float(prec[obs].mean())
        out["macro_recall_observed"] = float(rec[obs].mean())
    else:
        out["macro_f1_observed"] = out["macro_precision_observed"] = \
            out["macro_recall_observed"] = float("nan")
    # Cohen's kappa
    pe = float(np.sum((support / total) * (pred_n / total))) if total else float("nan")
    out["cohen_kappa"] = float((out["accuracy"] - pe) / (1 - pe)) if total and pe != 1 else float("nan")
    return out


def classification_metrics(y_true, y_pred, labels: Sequence[str]) -> dict:
    """Per-class precision/recall/F1/support plus macro/micro/weighted summaries.

    Convention (identical to sklearn's ``zero_division=0``): precision and
    recall are 0 when the denominator is 0.
    """
    return metrics_from_cm(confusion_matrix(y_true, y_pred, labels), labels)


def threshold_metrics(true, pred, tau: float, criterion: str) -> dict:
    """QC decision accuracy at a single decision threshold.

    ``criterion='contamination'``: a genome PASSES when the value is ``< tau``.
    ``criterion='completeness'``:  a genome PASSES when the value is ``>= tau``.

    Definitions (denominators stated explicitly):
      false_pass_rate = P(predicted PASS | truly FAIL)  -- a bad genome let
                        through, i.e. 1 - sensitivity for detecting failure.
      false_fail_rate = P(predicted FAIL | truly PASS)  -- a good genome
                        discarded, i.e. 1 - specificity.
      sensitivity     = P(predicted FAIL | truly FAIL)
      specificity     = P(predicted PASS | truly PASS)
    """
    t = np.asarray(true, float)
    p = np.asarray(pred, float)
    ok = ~(np.isnan(t) | np.isnan(p))
    t, p = t[ok], p[ok]
    if criterion == "contamination":
        t_pass, p_pass = t < tau, p < tau
    elif criterion == "completeness":
        t_pass, p_pass = t >= tau, p >= tau
    else:
        raise ValueError(criterion)
    t_fail, p_fail = ~t_pass, ~p_pass

    n = int(t.size)
    n_tp_pass, n_tp_fail = int(t_pass.sum()), int(t_fail.sum())
    false_pass = int((t_fail & p_pass).sum())
    false_fail = int((t_pass & p_fail).sum())
    tp = int((t_fail & p_fail).sum())      # correctly rejected
    tn = int((t_pass & p_pass).sum())      # correctly retained
    d = lambda a, b: float(a / b) if b else float("nan")   # noqa: E731
    return {
        "n": n, "n_true_pass": n_tp_pass, "n_true_fail": n_tp_fail,
        "n_pred_pass": int(p_pass.sum()), "n_pred_fail": int(p_fail.sum()),
        "n_false_pass": false_pass, "n_false_fail": false_fail,
        "false_pass_rate": d(false_pass, n_tp_fail),
        "false_fail_rate": d(false_fail, n_tp_pass),
        "sensitivity": d(tp, n_tp_fail),
        "specificity": d(tn, n_tp_pass),
        "ppv_fail": d(tp, tp + false_fail),
        "npv_pass": d(tn, tn + false_pass),
        "accuracy": d(tp + tn, n),
        "balanced_accuracy": float(np.nanmean([d(tp, n_tp_fail), d(tn, n_tp_pass)])),
        "n_ties_true_at_tau": int(np.sum(t == tau)),
        "n_ties_pred_at_tau": int(np.sum(p == tau)),
    }


# ---------------------------------------------------------------------------
# Bootstrap engine (iid and cluster)
# ---------------------------------------------------------------------------


class Bootstrapper:
    """Percentile bootstrap over row indices, optionally resampling clusters.

    Parameters
    ----------
    clusters
        Cluster label per row. ``None`` (or all-unique labels) gives the
        ordinary iid bootstrap. When clusters are supplied the resampling unit
        is the cluster: ``n_clusters`` clusters are drawn with replacement and
        *all* of their rows are taken, which preserves the within-cluster
        correlation that makes iid intervals too narrow.
    """

    def __init__(self, clusters: Optional[Sequence] = None, n_rows: Optional[int] = None,
                 n_iter: int = 2000, ci_level: float = 0.95, seed: int = 0):
        if clusters is None:
            if n_rows is None:
                raise ValueError("give clusters or n_rows")
            self.groups = [np.array([i]) for i in range(n_rows)]
            self.n_rows = n_rows
            self.clustered = False
        else:
            c = np.asarray(list(clusters), dtype=object)
            self.n_rows = c.size
            order = np.argsort(c.astype(str), kind="stable")
            cs = c[order].astype(str)
            bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
            self.groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            self.clustered = len(self.groups) < self.n_rows
        self.n_iter = int(n_iter)
        self.ci_level = float(ci_level)
        self.rng = np.random.default_rng(seed)
        self._index_cache: Optional[List[np.ndarray]] = None

    @property
    def n_clusters(self) -> int:
        return len(self.groups)

    def resample_indices(self) -> List[np.ndarray]:
        """Cached list of row-index arrays, one per bootstrap replicate."""
        if self._index_cache is None:
            g = self.groups
            k = len(g)
            picks = self.rng.integers(0, k, size=(self.n_iter, k))
            self._index_cache = [np.concatenate([g[j] for j in row]) for row in picks]
        return self._index_cache

    def ci(self, stat_fn: Callable[[np.ndarray], float],
           point: Optional[float] = None) -> dict:
        """Percentile CI of ``stat_fn(row_index_array)``."""
        all_idx = np.arange(self.n_rows)
        est = float(stat_fn(all_idx)) if point is None else float(point)
        vals = np.empty(self.n_iter, dtype=float)
        for b, idx in enumerate(self.resample_indices()):
            try:
                vals[b] = stat_fn(idx)
            except Exception:
                vals[b] = np.nan
        good = vals[~np.isnan(vals)]
        alpha = 1.0 - self.ci_level
        if good.size < 2:
            lo = hi = float("nan")
        else:
            lo, hi = np.percentile(good, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"estimate": est, "ci_lo": float(lo), "ci_hi": float(hi),
                "boot_n_valid": int(good.size), "boot_n_iter": self.n_iter,
                "boot_clustered": bool(self.clustered),
                "boot_n_clusters": self.n_clusters,
                "boot_mean": float(good.mean()) if good.size else float("nan"),
                "boot_sd": float(good.std(ddof=1)) if good.size > 1 else float("nan")}

    def ci_multi(self, stat_fn: Callable[[np.ndarray], Dict[str, float]]
                 ) -> Dict[str, dict]:
        """Bootstrap many statistics in one pass.

        ``stat_fn(row_index_array)`` must return a flat ``{name: value}`` dict
        with the same keys every call.
        """
        point = stat_fn(np.arange(self.n_rows))
        keys = list(point)
        acc = {k: np.full(self.n_iter, np.nan) for k in keys}
        for b, idx in enumerate(self.resample_indices()):
            try:
                r = stat_fn(idx)
            except Exception:
                continue
            for k in keys:
                acc[k][b] = r.get(k, np.nan)
        alpha = 1.0 - self.ci_level
        out = {}
        for k in keys:
            good = acc[k][~np.isnan(acc[k])]
            if good.size < 2:
                lo = hi = float("nan")
            else:
                lo, hi = np.percentile(good, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[k] = {"estimate": float(point[k]), "ci_lo": float(lo), "ci_hi": float(hi),
                      "boot_n_valid": int(good.size), "boot_n_iter": self.n_iter,
                      "boot_clustered": bool(self.clustered),
                      "boot_n_clusters": self.n_clusters}
        return out

    def p_two_sided(self, stat_fn: Callable[[np.ndarray], float],
                    null: float = 0.0) -> Tuple[float, np.ndarray]:
        """Two-sided bootstrap p-value for H0: statistic == ``null``.

        p = 2 * min( (#{theta* <= null} + 1)/(B+1), (#{theta* >= null} + 1)/(B+1) ),
        capped at 1. The +1 correction keeps p strictly positive and finite.
        """
        vals = np.empty(self.n_iter, dtype=float)
        for b, idx in enumerate(self.resample_indices()):
            try:
                vals[b] = stat_fn(idx)
            except Exception:
                vals[b] = np.nan
        good = vals[~np.isnan(vals)]
        B = good.size
        if B == 0:
            return float("nan"), good
        lo = (np.sum(good <= null) + 1) / (B + 1)
        hi = (np.sum(good >= null) + 1) / (B + 1)
        return float(min(1.0, 2 * min(lo, hi))), good


def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float],
                 clusters: Optional[Sequence] = None, n_iter: int = 2000,
                 ci_level: float = 0.95, seed: int = 0) -> dict:
    """Convenience wrapper: bootstrap ``stat`` of a 1-D array of values."""
    v = np.asarray(values, dtype=float)
    bs = Bootstrapper(clusters=clusters, n_rows=v.size, n_iter=n_iter,
                      ci_level=ci_level, seed=seed)
    return bs.ci(lambda idx: stat(v[idx]))


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cliffs_delta(x, y) -> float:
    """Cliff's delta = P(X > Y) - P(X < Y), computed in O(n log n).

    Sign convention: positive means values of ``x`` tend to be LARGER than ``y``.
    For absolute-error comparisons a NEGATIVE delta therefore means ``x`` has
    smaller errors, i.e. ``x`` is the better tool.
    """
    a = np.asarray(x, float)
    b = np.asarray(y, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("nan")
    bs = np.sort(b)
    # number of b strictly less than each a, and number of b <= each a
    less = np.searchsorted(bs, a, side="left")      # b < a
    leq = np.searchsorted(bs, a, side="right")      # b <= a
    greater = m - leq                               # b > a
    return float((less.sum() - greater.sum()) / (n * m))


def _cliffs_delta_bruteforce(x, y) -> float:
    a = np.asarray(x, float)
    b = np.asarray(y, float)
    gt = lt = 0
    for u in a:
        for v in b:
            if u > v:
                gt += 1
            elif u < v:
                lt += 1
    return (gt - lt) / (a.size * b.size)


def rank_biserial_paired(d) -> float:
    """Matched-pairs rank-biserial correlation, the effect size that accompanies
    the Wilcoxon signed-rank test.

    r = (W+ - W-) / (W+ + W-), where W+/W- are the sums of the ranks of |d| over
    positive/negative differences (zero differences are dropped, ties in |d| get
    average ranks). r in [-1, 1]; positive means ``d`` is predominantly positive.
    """
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if d.size == 0:
        return float("nan")
    r = sps.rankdata(np.abs(d))
    wp = r[d > 0].sum()
    wn = r[d < 0].sum()
    tot = wp + wn
    return float((wp - wn) / tot) if tot else float("nan")


_TRIU_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _triu(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached upper-triangle indices (bootstrap calls this thousands of times)."""
    t = _TRIU_CACHE.get(n)
    if t is None:
        if len(_TRIU_CACHE) > 64:
            _TRIU_CACHE.clear()
        t = np.triu_indices(n)
        _TRIU_CACHE[n] = t
    return t


def hodges_lehmann_paired(d) -> float:
    """One-sample Hodges-Lehmann estimator: median of the Walsh averages
    ``(d_i + d_j)/2`` for all ``i <= j``. This is the location estimate that the
    Wilcoxon signed-rank test inverts.
    """
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = d.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(d[0])
    w = d[:, None] + d[None, :]
    iu = _triu(n)
    return float(np.median(w[iu]) * 0.5)


def bh_correct(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR adjustment (monotone q-values).

    NaN p-values are excluded from ``m`` and returned as NaN.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    q[ok] = out
    return q


# ---------------------------------------------------------------------------
# Colour-vision-deficiency verification
# ---------------------------------------------------------------------------

# Machado, Oliveira & Fernandes (2009) severity-1.0 transformation matrices,
# applied in LINEAR RGB.
_CVD_MATRICES = {
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate_cvd(hex_colour: str, kind: str) -> str:
    """Simulate a dichromatic view of one sRGB hex colour."""
    lin = _srgb_to_linear(_hex_to_rgb(hex_colour))
    out = _linear_to_srgb(_CVD_MATRICES[kind] @ lin)
    return "#{:02X}{:02X}{:02X}".format(*(np.round(out * 255).astype(int)))


_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def _hex_to_lab(h: str) -> np.ndarray:
    xyz = _M_RGB2XYZ @ _srgb_to_linear(_hex_to_rgb(h)) / _WHITE_D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def palette_cvd_report(colours: Dict[str, str], min_delta_e: float = 20.0) -> pd.DataFrame:
    """Pairwise CIE76 Delta-E between every pair of palette colours under normal,
    deuteranopic, protanopic and tritanopic vision.

    A pair is flagged when Delta-E falls below ``min_delta_e`` in ANY vision
    type (20 is a conservative "clearly different" threshold for large patches).
    """
    keys = list(colours)
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            rec = {"colour_a": a, "colour_b": b,
                   "hex_a": colours[a], "hex_b": colours[b]}
            worst = np.inf
            for vision in ("normal", "deuteranopia", "protanopia", "tritanopia"):
                ha = colours[a] if vision == "normal" else simulate_cvd(colours[a], vision)
                hb = colours[b] if vision == "normal" else simulate_cvd(colours[b], vision)
                de = float(np.linalg.norm(_hex_to_lab(ha) - _hex_to_lab(hb)))
                rec[f"delta_e_{vision}"] = round(de, 2)
                worst = min(worst, de)
            rec["min_delta_e"] = round(worst, 2)
            rec["flag_too_similar"] = bool(worst < min_delta_e)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values("min_delta_e").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def init_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.constrained_layout.use": True,
    })
    return plt


def save_figure(fig, cfg: Config, name: str, caption: str,
                captions: Optional[List[str]] = None) -> List[Path]:
    """Save PNG + PDF and record the caption."""
    paths = []
    for ext in ("png", "pdf"):
        p = cfg.fig_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    if captions is not None:
        captions.append(f"### {name}\n\n{caption}\n")
    return paths


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    """Minimal GitHub-flavoured markdown table (no `tabulate` dependency)."""
    if df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v).replace("|", "\\|").replace("\n", " ")

    rows = [[fmt(v) for v in rec] for rec in df.itertuples(index=False, name=None)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
              for i in range(len(cols))]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    return "\n".join(out)


def caption_denominator(cfg: Config) -> str:
    d = cfg.raw["denominators"]
    return (f"Denominators: {d['completeness'].strip()}; {d['contamination'].strip()}. "
            f"{d['note'].strip()}")


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def build_inventory(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Row-level inventory of every set x file, plus a per-set summary."""
    file_rows, set_rows, prov_all = [], [], {}
    for bs in discover_sets(cfg, include_missing=True):
        if not bs.present:
            set_rows.append({"set": bs.name, "directory": str(bs.directory),
                             "listed_in_config": bs.listed, "present": False,
                             "status": bs.status, "tier": bs.tier, "label": bs.label,
                             "note": "listed in config but not on disk yet - skipped"})
            file_rows.append({"set": bs.name, "file": "metadata.tsv", "present": False,
                              "n_rows": np.nan, "n_cols": np.nan, "note": "set absent"})
            continue
        df, tools, prov = load_set(cfg, bs)
        prov_all[bs.name] = prov
        set_rows.append({
            "set": bs.name, "directory": str(bs.directory.relative_to(cfg.project_root)),
            "listed_in_config": bs.listed, "present": True, "status": bs.status,
            "tier": bs.tier, "label": bs.label, "design": bs.design,
            "n_genomes": prov["n_metadata"],
            "n_clusters_dominant_accession": prov["n_clusters"],
            "max_cluster_size": prov["max_cluster_size"],
            "tools_available": ",".join(tools),
            "tools_missing": ",".join(t for t in cfg.tools if t not in tools),
            "dominant_split_counts": json.dumps(prov["dominant_split_counts"]),
            "dominant_domain_counts": json.dumps(
                {k: int(v) for k, v in df.dominant_domain.value_counts().items()}),
            "n_phyla": int(df.dominant_phylum.nunique()),
            "relatedness_counts": json.dumps(
                {k: int(v) for k, v in df.relatedness.value_counts().items()}),
            "true_completeness_min": round(float(df.true_completeness.min()), 4),
            "true_completeness_max": round(float(df.true_completeness.max()), 4),
            "true_completeness_var": round(float(df.true_completeness.var()), 6),
            "true_contamination_min": round(float(df.true_contamination.min()), 4),
            "true_contamination_max": round(float(df.true_contamination.max()), 4),
            "true_contamination_var": round(float(df.true_contamination.var()), 6),
            "true_mimag_counts": json.dumps(
                {k: int(v) for k, v in df.true_mimag.value_counts().items()}),
            "labels_npy_max_abs_diff": max(
                prov.get("labels_max_abs_diff_completeness", 0.0),
                prov.get("labels_max_abs_diff_contamination", 0.0)),
            "note": "SUPERSEDED - training-set leakage" if bs.status == "superseded_leaky" else "",
        })
        # metadata row
        file_rows.append({"set": bs.name, "tool": "", "file": "metadata.tsv",
                          "present": True, "n_rows": prov["n_metadata"],
                          "n_cols": len(prov["metadata_columns"]),
                          "columns": ",".join(prov["metadata_columns"]), "note": ""})
        file_rows.append({"set": bs.name, "tool": "", "file": "labels.npy",
                          "present": bs.labels_path.exists(),
                          "n_rows": prov["n_metadata"] if bs.labels_path.exists() else np.nan,
                          "n_cols": 2 if bs.labels_path.exists() else np.nan,
                          "columns": "completeness,contamination",
                          "note": f"max|labels-metadata| = "
                                  f"{max(prov.get('labels_max_abs_diff_completeness', 0.0), prov.get('labels_max_abs_diff_contamination', 0.0)):.2e}"
                          if bs.labels_path.exists() else "absent"})
        for tool, tcfg in cfg.tools.items():
            info = prov["tools"].get(tool, {"present": False})
            note = ""
            if info.get("present") and info.get("usable"):
                if "truth_max_abs_diff_completeness" in info:
                    note = (f"embedded truth agrees with metadata "
                            f"(max |diff| = {max(info['truth_max_abs_diff_completeness'], info['truth_max_abs_diff_contamination']):.2e})")
                else:
                    note = "no embedded truth columns - joined on genome id"
            file_rows.append({
                "set": bs.name, "tool": tool, "file": tcfg["file"],
                "present": bool(info.get("present", False)),
                "n_rows": info.get("n_rows", np.nan),
                "n_cols": len(info.get("columns", [])) or np.nan,
                "columns": ",".join(info.get("columns", [])),
                "n_unique_ids": info.get("n_unique_ids", np.nan),
                "n_matched_to_metadata": info.get("n_matched", np.nan),
                "n_nan_predictions": (info.get("n_nan_completeness", 0) or 0)
                                     + (info.get("n_nan_contamination", 0) or 0),
                "note": note if info.get("present") else "file absent",
            })
    return pd.DataFrame(file_rows), pd.DataFrame(set_rows), prov_all


def write_inventory(cfg: Config) -> None:
    files, sets, prov = build_inventory(cfg)
    files.to_csv(cfg.out_dir / "inventory_files.tsv", sep="\t", index=False)
    sets.to_csv(cfg.out_dir / "inventory_sets.tsv", sep="\t", index=False)
    with open(cfg.out_dir / "inventory_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2, default=str)

    lines = ["# Benchmark data inventory (WS5 Stage A)", "",
             f"Config: `{cfg.path}`  |  benchmark root: `{cfg.raw['benchmark_root']}`", "",
             caption_denominator(cfg), "",
             "## Sets", ""]
    show = ["set", "status", "tier", "n_genomes", "n_clusters_dominant_accession",
            "max_cluster_size", "tools_available", "tools_missing",
            "dominant_split_counts", "note"]
    sub = sets.reindex(columns=show)
    lines.append(md_table(sub))
    lines += ["", "## Files", ""]
    fshow = ["set", "tool", "file", "present", "n_rows", "n_cols",
             "n_matched_to_metadata", "n_nan_predictions", "note"]
    lines.append(md_table(files.reindex(columns=fshow)))
    (cfg.out_dir / "inventory.md").write_text("\n".join(lines) + "\n")
    print(f"[inventory] wrote {cfg.out_dir}/inventory_files.tsv, inventory_sets.tsv, inventory.md")
    print(sub.to_string(index=False))


# ---------------------------------------------------------------------------
# Validation suite (test-driven development for the statistics)
# ---------------------------------------------------------------------------


def selftest(verbose: bool = True) -> List[str]:
    """Assertion suite validating every statistical primitive.

    Returns the log lines; raises AssertionError on the first failure.
    """
    log: List[str] = []

    def ok(msg):
        log.append("PASS  " + msg)
        if verbose:
            print("PASS  " + msg)

    cfg = load_config()

    # ---- MIMAG classification -------------------------------------------
    comp = np.array([100, 95, 90, 89.999, 90, 50, 49.9, 70, 100, 90])
    cont = np.array([0, 4.999, 5.0, 0, 9.999, 9.999, 0, 10.0, 100, 4.9999])
    got = list(mimag_classify(comp, cont, cfg))
    exp = ["high", "high", "medium", "medium", "medium", "medium",
           "low", "low", "low", "high"]
    assert got == exp, (got, exp)
    ok("mimag_classify: hand-checked boundary cases (>=90/<5 high, >=50/<10 medium)")
    assert list(mimag_classify([np.nan], [0], cfg)) == ["na"]
    ok("mimag_classify: NaN -> 'na'")

    # ---- contamination bins ---------------------------------------------
    b = list(contamination_bin([0.0, 0.001, 4.9, 5.0, 9.9, 10.0, 19.9,
                                20.0, 39.9, 40.0, 100.0], cfg))
    exp = ["0 (uncontaminated)", "(0,5)", "(0,5)", "[5,10)", "[5,10)",
           "[10,20)", "[10,20)", "[20,40)", "[20,40)", "[40,100]", "[40,100]"]
    assert b == exp, b
    ok("contamination_bin: hand-checked edges")

    # ---- point metrics ---------------------------------------------------
    assert abs(mae([1, 2, 3], [2, 2, 5]) - (1 + 0 + 2) / 3) < 1e-12
    assert abs(bias([1, 2, 3], [2, 2, 5]) - (1 + 0 + 2) / 3) < 1e-12
    assert abs(rmse([0, 0], [3, 4]) - np.sqrt(12.5)) < 1e-12
    ok("mae / bias / rmse: hand-checked")
    assert abs(r2_coefficient_of_determination([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12
    # y=[1,2,3], yhat=mean -> R2 = 0
    assert abs(r2_coefficient_of_determination([1, 2, 3], [2, 2, 2])) < 1e-12
    assert np.isnan(r2_coefficient_of_determination([5, 5, 5], [1, 2, 3]))
    ok("R^2: perfect=1, mean-predictor=0, constant truth=NaN")
    # R2 vs squared Pearson differ under bias/scale; identical for identity fit
    y = np.array([1., 2., 3., 4.])
    assert abs(r2_pearson_squared(y, 2 * y + 7) - 1.0) < 1e-12
    assert r2_coefficient_of_determination(y, 2 * y + 7) < 0
    ok("r^2 (Pearson) vs R^2 (coeff. of determination) are demonstrably different")
    try:
        from sklearn.metrics import r2_score
        rng = np.random.default_rng(0)
        a, bb = rng.normal(size=50), rng.normal(size=50)
        assert abs(r2_coefficient_of_determination(a, bb) - r2_score(a, bb)) < 1e-12
        ok("R^2 matches sklearn.metrics.r2_score")
    except ImportError:
        log.append("SKIP  sklearn not available for R^2 cross-check")

    # ---- classification metrics -----------------------------------------
    # hand-built 3x3 confusion:
    #            pred high  med  low
    # true high      3       1    1     support 5
    # true med       0       2    2     support 4
    # true low       1       0    3     support 4
    yt = (["high"] * 5) + (["medium"] * 4) + (["low"] * 4)
    yp = (["high"] * 3 + ["medium"] + ["low"]) + (["medium"] * 2 + ["low"] * 2) + \
         (["high"] + ["low"] * 3)
    labels = ["high", "medium", "low"]
    cm = confusion_matrix(yt, yp, labels)
    assert cm.tolist() == [[3, 1, 1], [0, 2, 2], [1, 0, 3]], cm.tolist()
    m = classification_metrics(yt, yp, labels)
    # precision: high 3/4, medium 2/3, low 3/6
    assert np.allclose(m["precision"], [3 / 4, 2 / 3, 3 / 6])
    # recall: high 3/5, medium 2/4, low 3/4
    assert np.allclose(m["recall"], [3 / 5, 2 / 4, 3 / 4])
    f1_exp = [2 * (3 / 4) * (3 / 5) / (3 / 4 + 3 / 5), 2 * (2 / 3) * .5 / (2 / 3 + .5),
              2 * .5 * .75 / (.5 + .75)]
    assert np.allclose(m["f1"], f1_exp)
    assert abs(m["accuracy"] - 8 / 13) < 1e-12
    assert abs(m["micro_f1"] - 8 / 13) < 1e-12
    assert abs(m["macro_f1"] - float(np.mean(f1_exp))) < 1e-12
    ok("classification_metrics: hand-computed 3x3 precision/recall/F1/accuracy")
    tc, pc = encode_labels(yt, labels), encode_labels(yp, labels)
    assert np.array_equal(cm_from_codes(tc, pc, 3), cm)
    m_fast = metrics_from_cm(cm_from_codes(tc, pc, 3), labels)
    assert np.allclose(m_fast["f1"], m["f1"]) and abs(m_fast["accuracy"] - m["accuracy"]) < 1e-15
    assert np.array_equal(cm_from_codes(np.array([-1, 0]), np.array([0, 0]), 3),
                          np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    ok("cm_from_codes / metrics_from_cm: vectorised path matches, unknown codes dropped")
    try:
        from sklearn.metrics import (precision_recall_fscore_support,
                                     balanced_accuracy_score, cohen_kappa_score)
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels,
                                                    zero_division=0)
        assert np.allclose(m["precision"], p) and np.allclose(m["recall"], r)
        assert np.allclose(m["f1"], f) and np.array_equal(m["support"], s)
        assert abs(m["balanced_accuracy"] - balanced_accuracy_score(yt, yp)) < 1e-12
        assert abs(m["cohen_kappa"] - cohen_kappa_score(yt, yp)) < 1e-12
        ok("classification_metrics matches sklearn (P/R/F1/support/balanced acc/kappa)")
    except ImportError:
        log.append("SKIP  sklearn not available for classification cross-check")

    # ---- threshold metrics ----------------------------------------------
    # contamination criterion at tau=5: pass <=> value < 5
    true = [1.0, 2.0, 6.0, 7.0, 20.0]     # pass, pass, fail, fail, fail
    pred = [1.0, 6.0, 1.0, 7.0, 30.0]     # pass, fail, pass, fail, fail
    t = threshold_metrics(true, pred, 5.0, "contamination")
    assert t["n_true_pass"] == 2 and t["n_true_fail"] == 3
    assert t["n_false_pass"] == 1 and t["n_false_fail"] == 1
    assert abs(t["false_pass_rate"] - 1 / 3) < 1e-12
    assert abs(t["false_fail_rate"] - 1 / 2) < 1e-12
    assert abs(t["sensitivity"] - 2 / 3) < 1e-12 and abs(t["specificity"] - 1 / 2) < 1e-12
    assert abs(t["accuracy"] - 3 / 5) < 1e-12
    ok("threshold_metrics (contamination, tau=5): hand-checked false-pass/false-fail")
    # completeness criterion at tau=90: pass <=> value >= 90
    t2 = threshold_metrics([95, 80, 91, 50], [85, 95, 92, 40], 90.0, "completeness")
    assert t2["n_true_pass"] == 2 and t2["n_true_fail"] == 2
    assert t2["n_false_fail"] == 1 and t2["n_false_pass"] == 1
    ok("threshold_metrics (completeness, tau=90): hand-checked")

    # ---- Cliff's delta ---------------------------------------------------
    assert abs(cliffs_delta([1, 2, 3], [4, 5, 6]) + 1.0) < 1e-12
    assert abs(cliffs_delta([4, 5, 6], [1, 2, 3]) - 1.0) < 1e-12
    assert abs(cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-12
    assert abs(cliffs_delta([1, 2], [1, 3]) + 0.25) < 1e-12
    ok("cliffs_delta: hand-checked (-1, +1, 0, -0.25 with ties)")
    rng = np.random.default_rng(7)
    for _ in range(200):
        n, mm = rng.integers(1, 25), rng.integers(1, 25)
        a = rng.integers(0, 6, size=n).astype(float)
        b = rng.integers(0, 6, size=mm).astype(float)
        assert abs(cliffs_delta(a, b) - _cliffs_delta_bruteforce(a, b)) < 1e-12
    ok("cliffs_delta: 200 random cases match the O(nm) brute force (ties included)")

    # ---- matched-pairs rank-biserial -------------------------------------
    # d = [1, 2, -3] -> |d| ranks 1,2,3 ; W+=1+2=3, W-=3 -> r=0
    assert abs(rank_biserial_paired([1, 2, -3])) < 1e-12
    assert abs(rank_biserial_paired([1, 2, 3]) - 1.0) < 1e-12
    assert abs(rank_biserial_paired([-1, -2, -3]) + 1.0) < 1e-12
    # zeros dropped
    assert abs(rank_biserial_paired([0, 0, 5]) - 1.0) < 1e-12
    # d=[3,-1,-2]: ranks |3|=3,|1|=1,|2|=2 ; W+=3, W-=3 -> 0
    assert abs(rank_biserial_paired([3, -1, -2])) < 1e-12
    ok("rank_biserial_paired: hand-checked (0, +1, -1, zero-dropping)")
    # relation to scipy's Wilcoxon statistic: W- = min-sum branch
    d = np.array([2.0, -1.0, 4.0, -3.0, 5.0])
    r = sps.rankdata(np.abs(d))
    wp, wn = r[d > 0].sum(), r[d < 0].sum()
    st = sps.wilcoxon(d, alternative="two-sided", zero_method="wilcox").statistic
    assert abs(st - min(wp, wn)) < 1e-12
    assert abs(rank_biserial_paired(d) - (wp - wn) / (wp + wn)) < 1e-12
    ok("rank_biserial_paired: W+/W- consistent with scipy.stats.wilcoxon statistic")

    # ---- Hodges-Lehmann --------------------------------------------------
    # d=[1,2,4]: Walsh averages (i<=j) = 1,1.5,2.5,2,3,4 -> sorted 1,1.5,2,2.5,3,4
    # median = (2+2.5)/2 = 2.25
    assert abs(hodges_lehmann_paired([1, 2, 4]) - 2.25) < 1e-12
    assert abs(hodges_lehmann_paired([5]) - 5.0) < 1e-12
    assert abs(hodges_lehmann_paired([-2, 2]) - 0.0) < 1e-12
    ok("hodges_lehmann_paired: hand-enumerated Walsh averages")
    # HL is the value that makes the signed-rank test maximally non-significant
    rng = np.random.default_rng(3)
    d = rng.normal(4.0, 1.0, size=40)
    hl = hodges_lehmann_paired(d)
    p_at_hl = sps.wilcoxon(d - hl, alternative="two-sided").pvalue
    for shift in (hl - 0.5, hl + 0.5):
        assert sps.wilcoxon(d - shift, alternative="two-sided").pvalue <= p_at_hl + 1e-12
    ok("hodges_lehmann_paired: maximises the signed-rank p-value (test inversion)")

    # ---- Benjamini-Hochberg ---------------------------------------------
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    q = bh_correct(p)
    assert np.allclose(q, [0.05, 0.05, 0.05, 0.05, 0.05]), q
    q2 = bh_correct([0.001, 0.5, 0.9])
    assert np.allclose(q2, [0.003, 0.75, 0.9]), q2
    assert np.isnan(bh_correct([0.01, np.nan])[1])
    ok("bh_correct: hand-computed step-up q-values, monotone, NaN-safe")
    rng = np.random.default_rng(11)
    pv = rng.uniform(size=97)
    assert np.allclose(bh_correct(pv), sps.false_discovery_control(pv, method="bh"))
    ok("bh_correct matches scipy.stats.false_discovery_control on 97 random p-values")

    # ---- Deterministic seed derivation -----------------------------------
    # Hard-coded CRC-32 values: if these ever change, every previously reported
    # confidence interval becomes irreproducible, so they are pinned.
    assert stable_hash("set_C_clean") == 2775168523, stable_hash("set_C_clean")
    assert stable_hash("set_A_v2") == 1098507440, stable_hash("set_A_v2")
    assert stable_hash("") == 0
    assert all(0 <= stable_hash(s) < 2 ** 32 for s in
               ("set_E", "POOLED_leakage_free_5_sets", "magicc_v5"))
    # and it must NOT agree with the salted builtin (that is the whole point)
    _sub = __import__("subprocess")
    _outs = {_sub.run([sys.executable, "-c", "print(hash('set_C_clean'))"],
                      capture_output=True, text=True,
                      env={**os.environ, "PYTHONHASHSEED": "random"}).stdout.strip()
             for _ in range(8)}
    assert len(_outs) > 1, ("builtin hash() appears unsalted in this environment; "
                            "stable_hash is still required for portability")
    ok("stable_hash: CRC-32 seed derivation is process-independent "
       "(builtin hash() verified salted, so it must not be used for seeds)")

    # ---- Bootstrap: iid vs cluster --------------------------------------
    v = np.arange(20, dtype=float)
    # singleton clusters must reproduce the iid bootstrap exactly (same RNG path)
    b_iid = Bootstrapper(n_rows=20, n_iter=500, seed=42)
    b_sing = Bootstrapper(clusters=[f"c{i:02d}" for i in range(20)], n_iter=500, seed=42)
    assert not b_sing.clustered
    r1 = b_iid.ci(lambda idx: v[idx].mean())
    r2 = b_sing.ci(lambda idx: v[idx].mean())
    assert abs(r1["ci_lo"] - r2["ci_lo"]) < 1e-12 and abs(r1["ci_hi"] - r2["ci_hi"]) < 1e-12
    ok("Bootstrapper: singleton clusters reproduce the iid bootstrap exactly")

    # fully redundant clusters: 10 clusters x 5 identical copies. The cluster
    # bootstrap of the mean must equal the iid bootstrap of the 10 cluster values.
    base = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    vv = np.repeat(base, 5)
    cl = np.repeat([f"g{i}" for i in range(10)], 5)
    bc = Bootstrapper(clusters=cl, n_iter=4000, seed=1)
    bi = Bootstrapper(n_rows=10, n_iter=4000, seed=1)
    rc = bc.ci(lambda idx: vv[idx].mean())
    ri = bi.ci(lambda idx: base[idx].mean())
    assert bc.clustered and bc.n_clusters == 10
    assert abs(rc["ci_lo"] - ri["ci_lo"]) < 1e-9 and abs(rc["ci_hi"] - ri["ci_hi"]) < 1e-9
    ok("Bootstrapper: perfectly redundant clusters == iid bootstrap of cluster means")

    # naive bootstrap on the duplicated data is far too narrow
    bn = Bootstrapper(n_rows=50, n_iter=4000, seed=1)
    rn = bn.ci(lambda idx: vv[idx].mean())
    w_naive = rn["ci_hi"] - rn["ci_lo"]
    w_clust = rc["ci_hi"] - rc["ci_lo"]
    assert w_clust > 1.8 * w_naive, (w_clust, w_naive)
    ok(f"Bootstrapper: cluster CI is {w_clust / w_naive:.2f}x wider than the naive CI "
       f"on 5x-redundant data (sqrt(5)=2.24 expected)")

    # coverage simulation: clustered data, true mean 0
    rng = np.random.default_rng(2024)
    n_g, k = 40, 10
    cov_c = cov_i = 0
    reps = 300
    for _ in range(reps):
        g_eff = rng.normal(0, 1.0, size=n_g)                 # between-cluster
        x = (np.repeat(g_eff, k) + rng.normal(0, 0.5, size=n_g * k))
        cl2 = np.repeat(np.arange(n_g), k)
        rr = Bootstrapper(clusters=cl2, n_iter=400, seed=int(rng.integers(1e9))
                          ).ci(lambda idx: x[idx].mean())
        cov_c += (rr["ci_lo"] <= 0 <= rr["ci_hi"])
        rr2 = Bootstrapper(n_rows=x.size, n_iter=400, seed=int(rng.integers(1e9))
                           ).ci(lambda idx: x[idx].mean())
        cov_i += (rr2["ci_lo"] <= 0 <= rr2["ci_hi"])
    cc, ci_ = cov_c / reps, cov_i / reps
    assert cc > 0.90, f"cluster bootstrap coverage {cc:.3f} too low"
    assert ci_ < 0.60, f"iid bootstrap coverage {ci_:.3f} unexpectedly high"
    ok(f"Bootstrapper coverage simulation (ICC=0.8, 40 clusters x 10): "
       f"cluster {cc:.1%} (nominal 95%), naive iid {ci_:.1%} -> clustering matters")

    # bootstrap two-sided p-value sanity
    rng = np.random.default_rng(5)
    z = rng.normal(0.0, 1.0, size=300)
    pz = Bootstrapper(n_rows=300, n_iter=2000, seed=9).p_two_sided(
        lambda idx: z[idx].mean())[0]
    z2 = z + 1.0
    pz2 = Bootstrapper(n_rows=300, n_iter=2000, seed=9).p_two_sided(
        lambda idx: z2[idx].mean())[0]
    assert pz > 0.2 and pz2 < 0.002, (pz, pz2)
    ok(f"Bootstrapper.p_two_sided: null p={pz:.3f}, strong-effect p={pz2:.2e}")

    # ---- CVD palette -----------------------------------------------------
    # a pure red/green pair must be flagged, Okabe-Ito blue/orange must not
    bad = palette_cvd_report({"red": "#FF0000", "green": "#00CC00"})
    assert bool(bad.flag_too_similar.iloc[0]), bad.to_dict()
    good = palette_cvd_report({"blue": "#0072B2", "orange": "#E69F00"})
    assert not bool(good.flag_too_similar.iloc[0]), good.to_dict()
    ok("palette_cvd_report: flags red/green, accepts Okabe-Ito blue/orange")
    rep = palette_cvd_report(cfg.raw["palette"]["tools"])
    flagged = rep[rep.flag_too_similar]
    assert flagged.empty, "configured tool palette has CVD-unsafe pairs:\n" + flagged.to_string()
    ok(f"configured tool palette is CVD-safe: min Delta-E over all pairs and all "
       f"vision types = {rep.min_delta_e.min():.1f} (>= 20)")
    repm = palette_cvd_report(cfg.raw["palette"]["mimag"])
    assert repm[repm.flag_too_similar].empty
    ok(f"configured MIMAG palette is CVD-safe (min Delta-E = {repm.min_delta_e.min():.1f})")

    log.append("")
    log.append(f"ALL {sum(1 for l in log if l.startswith('PASS'))} CHECKS PASSED")
    if verbose:
        print(f"\nALL {sum(1 for l in log if l.startswith('PASS'))} CHECKS PASSED")
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--selftest", action="store_true", help="run the validation suite")
    ap.add_argument("--inventory", action="store_true", help="write the data inventory")
    ap.add_argument("--cvd-report", action="store_true",
                    help="write the colour-vision-deficiency palette report")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if not any([args.selftest, args.inventory, args.cvd_report]):
        ap.print_help()
        return 0
    if args.selftest:
        log = selftest()
        (cfg.out_dir / "selftest_report.txt").write_text("\n".join(log) + "\n")
        print(f"[selftest] wrote {cfg.out_dir}/selftest_report.txt")
    if args.cvd_report:
        rows = []
        for group in ("tools", "mimag"):
            r = palette_cvd_report(cfg.raw["palette"][group])
            r.insert(0, "palette_group", group)
            rows.append(r)
        rep = pd.concat(rows, ignore_index=True)
        rep.to_csv(cfg.out_dir / "palette_cvd_report.tsv", sep="\t", index=False)
        sim = pd.DataFrame([
            {"palette_group": g, "key": k, "hex": v,
             "deuteranopia": simulate_cvd(v, "deuteranopia"),
             "protanopia": simulate_cvd(v, "protanopia"),
             "tritanopia": simulate_cvd(v, "tritanopia")}
            for g in ("tools", "mimag") for k, v in cfg.raw["palette"][g].items()])
        sim.to_csv(cfg.out_dir / "palette_cvd_simulated.tsv", sep="\t", index=False)
        print(f"[cvd] wrote palette_cvd_report.tsv (min Delta-E = {rep.min_delta_e.min():.1f}, "
              f"{int(rep.flag_too_similar.sum())} flagged pairs)")
    if args.inventory:
        write_inventory(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
