#!/usr/bin/env python3
"""
Phase 6 - Comprehensive Benchmarking Analysis and Publication-Quality Figures

Performs full comparative analysis of MAGICC vs CheckM2 vs CoCoPyE vs DeepCheck
across benchmark sets A-D and the 100K test set.

Outputs:
  results/accuracy_metrics.tsv
  results/mimag_classification.tsv
  results/statistical_tests.tsv
  results/tool_comparison_summary.tsv
  results/figures/fig1_accuracy_scatter.png
  results/figures/fig2_set_comparison_bars.png
  results/figures/fig3_bland_altman.png
  results/figures/fig4_speed_comparison.png
  results/figures/fig5_magicc_100k_test.png
  results/figures/fig6_per_set_detail.png
"""

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = Path('/home/tianrm/projects/magicc2')
DATA_DIR = PROJECT_DIR / 'data'
BENCHMARK_DIR = DATA_DIR / 'benchmarks'
RESULTS_DIR = PROJECT_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'

TOOLS = ['MAGICC', 'CheckM2', 'CoCoPyE', 'DeepCheck']
TOOL_FILES = {
    'MAGICC': 'magicc_predictions.tsv',
    'CheckM2': 'checkm2_predictions.tsv',
    'CoCoPyE': 'cocopye_predictions.tsv',
    'DeepCheck': 'deepcheck_predictions.tsv',
}
SETS = ['A', 'B', 'C', 'D']

# Color scheme
COLORS = {
    'MAGICC': '#2196F3',
    'CheckM2': '#FF9800',
    'CoCoPyE': '#4CAF50',
    'DeepCheck': '#9C27B0',
}

# Set colors for scatter plots
SET_COLORS = {
    'A': '#E91E63',   # pink
    'B': '#3F51B5',   # indigo
    'C': '#009688',   # teal
    'D': '#FF5722',   # deep orange
}

# Sample type colors for 100K test
SAMPLE_TYPE_COLORS = {
    'pure': '#2196F3',
    'complete': '#4CAF50',
    'within_phylum': '#FF9800',
    'cross_phylum': '#F44336',
    'reduced_genome': '#9C27B0',
    'archaeal': '#795548',
}

N_BOOTSTRAP = 1000
N_CPUS = min(multiprocessing.cpu_count(), 48)
RNG = np.random.default_rng(42)

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})


# ============================================================================
# Data Loading
# ============================================================================
def load_predictions():
    """Load all prediction files into a nested dict: data[set][tool] = DataFrame."""
    data = {}
    for set_name in SETS:
        data[set_name] = {}
        for tool in TOOLS:
            fpath = BENCHMARK_DIR / f'set_{set_name}' / TOOL_FILES[tool]
            if fpath.exists():
                df = pd.read_csv(fpath, sep='\t')
                data[set_name][tool] = df
            else:
                print(f"  WARNING: Missing {fpath}")
                data[set_name][tool] = None
    return data


def load_test_100k():
    """Load 100K test set predictions."""
    fpath = BENCHMARK_DIR / 'test_100k' / 'magicc_predictions.tsv'
    if fpath.exists():
        return pd.read_csv(fpath, sep='\t')
    else:
        print(f"  WARNING: Missing {fpath}")
        return None


# ============================================================================
# Metric Computation
# ============================================================================
def compute_metrics(true_vals, pred_vals):
    """Compute MAE, RMSE, R2 (Pearson), outlier rate."""
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    tv, pv = true_vals[mask], pred_vals[mask]
    if len(tv) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'outlier_rate': np.nan, 'n': 0}

    errors = np.abs(tv - pv)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((tv - pv) ** 2))

    if np.std(tv) > 1e-10 and np.std(pv) > 1e-10:
        r2 = np.corrcoef(tv, pv)[0, 1] ** 2
    else:
        r2 = np.nan

    outlier_rate = np.mean(errors > 20.0) * 100

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'outlier_rate': outlier_rate, 'n': len(tv)}


def mimag_classify(comp, cont):
    """Classify genomes into MIMAG tiers."""
    tiers = np.full(len(comp), 'low', dtype=object)
    medium = (comp >= 50) & (cont < 10)
    high = (comp >= 90) & (cont < 5)
    tiers[medium] = 'medium'
    tiers[high] = 'high'
    return tiers


def bootstrap_mae(true_vals, pred_vals, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Bootstrap 95% CI for MAE."""
    rng = np.random.default_rng(seed)
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    tv, pv = true_vals[mask], pred_vals[mask]
    n = len(tv)
    if n == 0:
        return np.nan, np.nan, np.nan

    errors = np.abs(tv - pv)
    mae_obs = np.mean(errors)

    # Vectorized bootstrap
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_maes = np.mean(errors[indices], axis=1)

    ci_lo = np.percentile(boot_maes, 2.5)
    ci_hi = np.percentile(boot_maes, 97.5)
    return mae_obs, ci_lo, ci_hi


# ============================================================================
# Part A: Accuracy Metrics Table
# ============================================================================
def compute_accuracy_metrics(data):
    """Compute accuracy metrics per set, per tool."""
    print("\n" + "=" * 70)
    print("PART A: ACCURACY METRICS TABLE")
    print("=" * 70)

    rows = []
    for set_name in SETS:
        for tool in TOOLS:
            df = data[set_name].get(tool)
            if df is None:
                continue

            tc = df['true_completeness'].values
            tt = df['true_contamination'].values
            pc = df['pred_completeness'].values
            pt = df['pred_contamination'].values

            mc = compute_metrics(tc, pc)
            mt = compute_metrics(tt, pt)

            rows.append({
                'set': set_name,
                'tool': tool,
                'n_genomes': mc['n'],
                'comp_mae': mc['mae'],
                'comp_rmse': mc['rmse'],
                'comp_r2': mc['r2'],
                'comp_outlier_rate': mc['outlier_rate'],
                'cont_mae': mt['mae'],
                'cont_rmse': mt['rmse'],
                'cont_r2': mt['r2'],
                'cont_outlier_rate': mt['outlier_rate'],
            })

    metrics_df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / 'accuracy_metrics.tsv'
    metrics_df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    print(f"  Saved: {output_path}")

    # Print table
    print(f"\n  {'Set':<4} {'Tool':<10} {'N':>6} | {'Comp MAE':>9} {'Comp RMSE':>10} {'Comp R2':>8} {'Comp Out%':>9} | {'Cont MAE':>9} {'Cont RMSE':>10} {'Cont R2':>8} {'Cont Out%':>9}")
    print("  " + "-" * 110)
    for _, row in metrics_df.iterrows():
        print(f"  {row['set']:<4} {row['tool']:<10} {row['n_genomes']:>6.0f} | "
              f"{row['comp_mae']:>8.2f}% {row['comp_rmse']:>9.2f}% {row['comp_r2']:>8.4f} {row['comp_outlier_rate']:>8.2f}% | "
              f"{row['cont_mae']:>8.2f}% {row['cont_rmse']:>9.2f}% {row['cont_r2']:>8.4f} {row['cont_outlier_rate']:>8.2f}%")

    return metrics_df


# ============================================================================
# Part B: MIMAG Classification
# ============================================================================
def compute_mimag_classification(data):
    """Compute MIMAG classification F1 scores for Sets C and D."""
    print("\n" + "=" * 70)
    print("PART B: MIMAG CLASSIFICATION (Sets C, D)")
    print("=" * 70)

    rows = []
    for set_name in ['C', 'D']:
        for tool in TOOLS:
            df = data[set_name].get(tool)
            if df is None:
                continue

            tc = df['true_completeness'].values
            tt = df['true_contamination'].values
            pc = df['pred_completeness'].values
            pt = df['pred_contamination'].values

            true_tiers = mimag_classify(tc, tt)
            pred_tiers = mimag_classify(pc, pt)

            f1_macro = f1_score(true_tiers, pred_tiers, average='macro', labels=['high', 'medium', 'low'])
            f1_weighted = f1_score(true_tiers, pred_tiers, average='weighted', labels=['high', 'medium', 'low'])

            # Per-tier F1
            f1_per = {}
            for tier in ['high', 'medium', 'low']:
                binary_true = (true_tiers == tier).astype(int)
                binary_pred = (pred_tiers == tier).astype(int)
                if binary_true.sum() > 0:
                    f1_per[tier] = f1_score(binary_true, binary_pred)
                else:
                    f1_per[tier] = np.nan

            # Confusion matrix
            cm = confusion_matrix(true_tiers, pred_tiers, labels=['high', 'medium', 'low'])

            rows.append({
                'set': set_name,
                'tool': tool,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'f1_high': f1_per['high'],
                'f1_medium': f1_per['medium'],
                'f1_low': f1_per['low'],
                'n_true_high': int(np.sum(true_tiers == 'high')),
                'n_true_medium': int(np.sum(true_tiers == 'medium')),
                'n_true_low': int(np.sum(true_tiers == 'low')),
            })

            print(f"\n  Set {set_name}, {tool}: F1_macro={f1_macro:.4f}, F1_weighted={f1_weighted:.4f}")
            print(f"    F1_high={f1_per['high']:.4f}, F1_medium={f1_per['medium']:.4f}, F1_low={f1_per['low']:.4f}")
            print(f"    Confusion matrix (rows=true, cols=pred) [high, medium, low]:")
            for i, tier in enumerate(['high', 'medium', 'low']):
                print(f"      {tier:>6}: {cm[i]}")

    mimag_df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / 'mimag_classification.tsv'
    mimag_df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    print(f"\n  Saved: {output_path}")

    return mimag_df


# ============================================================================
# Part C: Statistical Tests
# ============================================================================
def compute_statistical_tests(data):
    """Paired Wilcoxon signed-rank tests and bootstrap CIs."""
    print("\n" + "=" * 70)
    print("PART C: STATISTICAL TESTS")
    print("=" * 70)

    rows = []
    competitors = ['CheckM2', 'CoCoPyE', 'DeepCheck']

    for set_name in SETS:
        magicc_df = data[set_name].get('MAGICC')
        if magicc_df is None:
            continue

        for comp_tool in competitors:
            comp_df = data[set_name].get(comp_tool)
            if comp_df is None:
                continue

            # Align by genome_id
            merged = magicc_df.merge(comp_df, on='genome_id', suffixes=('_magicc', f'_{comp_tool.lower()}'))

            for metric_name, true_col, magicc_col, comp_col in [
                ('completeness', 'true_completeness_magicc',
                 'pred_completeness_magicc', f'pred_completeness_{comp_tool.lower()}'),
                ('contamination', 'true_contamination_magicc',
                 'pred_contamination_magicc', f'pred_contamination_{comp_tool.lower()}'),
            ]:
                true_vals = merged[true_col].values
                magicc_preds = merged[magicc_col].values
                comp_preds = merged[comp_col].values

                magicc_errors = np.abs(true_vals - magicc_preds)
                comp_errors = np.abs(true_vals - comp_preds)

                # Paired Wilcoxon signed-rank test
                diffs = comp_errors - magicc_errors  # positive = MAGICC better
                nonzero = diffs != 0
                if np.sum(nonzero) > 10:
                    try:
                        stat, p_value = stats.wilcoxon(diffs[nonzero], alternative='greater')
                    except Exception:
                        stat, p_value = np.nan, np.nan
                else:
                    stat, p_value = np.nan, np.nan

                # Bootstrap CIs for MAE
                mae_m, ci_lo_m, ci_hi_m = bootstrap_mae(true_vals, magicc_preds)
                mae_c, ci_lo_c, ci_hi_c = bootstrap_mae(true_vals, comp_preds)

                magicc_mean_err = np.mean(magicc_errors)
                comp_mean_err = np.mean(comp_errors)
                improvement = comp_mean_err - magicc_mean_err

                rows.append({
                    'set': set_name,
                    'metric': metric_name,
                    'comparison': f'MAGICC vs {comp_tool}',
                    'magicc_mae': magicc_mean_err,
                    'magicc_ci_lo': ci_lo_m,
                    'magicc_ci_hi': ci_hi_m,
                    'competitor_mae': comp_mean_err,
                    'competitor_ci_lo': ci_lo_c,
                    'competitor_ci_hi': ci_hi_c,
                    'improvement': improvement,
                    'wilcoxon_stat': stat,
                    'wilcoxon_p': p_value,
                    'significant_p005': p_value < 0.05 if not np.isnan(p_value) else False,
                    'n_samples': len(merged),
                })

                sig = "*" if (not np.isnan(p_value) and p_value < 0.05) else ""
                print(f"  Set {set_name} {metric_name:>15} MAGICC vs {comp_tool:<10}: "
                      f"MAGICC={magicc_mean_err:.2f}% [{ci_lo_m:.2f},{ci_hi_m:.2f}], "
                      f"{comp_tool}={comp_mean_err:.2f}% [{ci_lo_c:.2f},{ci_hi_c:.2f}], "
                      f"diff={improvement:+.2f}%, p={p_value:.2e}{sig}")

    stat_df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / 'statistical_tests.tsv'
    stat_df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    print(f"\n  Saved: {output_path}")

    return stat_df


# ============================================================================
# Part D: Figures
# ============================================================================

def _combine_data(data, tools=TOOLS):
    """Combine all sets into one DataFrame per tool."""
    combined = {tool: [] for tool in tools}
    for set_name in SETS:
        for tool in tools:
            df = data[set_name].get(tool)
            if df is not None:
                df_copy = df.copy()
                df_copy['set'] = set_name
                combined[tool].append(df_copy)
    for tool in tools:
        if combined[tool]:
            combined[tool] = pd.concat(combined[tool], ignore_index=True)
        else:
            combined[tool] = None
    return combined


def fig1_accuracy_scatter(data):
    """Figure 1: 2x4 scatter grid - predicted vs true for all tools, comp and cont."""
    print("\n  Generating Figure 1: Accuracy scatter plots...")

    combined = _combine_data(data)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for col, tool in enumerate(TOOLS):
        cdf = combined[tool]
        if cdf is None:
            continue

        for row, (metric, true_col, pred_col, label) in enumerate([
            ('comp', 'true_completeness', 'pred_completeness', 'Completeness (%)'),
            ('cont', 'true_contamination', 'pred_contamination', 'Contamination (%)'),
        ]):
            ax = axes[row, col]

            # Plot each set with different colors
            for set_name in SETS:
                subset = cdf[cdf['set'] == set_name]
                if len(subset) == 0:
                    continue
                ax.scatter(
                    subset[true_col], subset[pred_col],
                    c=SET_COLORS[set_name], s=6, alpha=0.4,
                    label=f'Set {set_name}', rasterized=True,
                    edgecolors='none'
                )

            # Diagonal reference line
            lims = [0, 100]
            ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, zorder=10)

            # R2 annotation
            true_vals = cdf[true_col].values
            pred_vals = cdf[pred_col].values
            mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
            if np.std(true_vals[mask]) > 0 and np.std(pred_vals[mask]) > 0:
                r2 = np.corrcoef(true_vals[mask], pred_vals[mask])[0, 1] ** 2
                mae = np.mean(np.abs(true_vals[mask] - pred_vals[mask]))
                ax.text(0.05, 0.92, f'R$^2$={r2:.3f}\nMAE={mae:.1f}%',
                        transform=ax.transAxes, fontsize=8, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax.set_xlim(-2, 102)
            ax.set_ylim(-2, 102)

            if row == 0:
                ax.set_title(tool, fontweight='bold', color=COLORS[tool])
            if col == 0:
                ax.set_ylabel(f'Predicted {label}')
            if row == 1:
                ax.set_xlabel(f'True {label}')

            # Only add legend to first subplot
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc='lower right', markerscale=2)

    plt.suptitle('Predicted vs True: All Tools on Benchmark Sets A-D', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_accuracy_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig1_accuracy_scatter.png'}")


def fig2_set_comparison_bars(data):
    """Figure 2: Grouped bar chart of MAE per set per tool with bootstrap CIs."""
    print("\n  Generating Figure 2: Set comparison bar chart...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (metric, true_col, pred_col, title) in enumerate([
        ('comp', 'true_completeness', 'pred_completeness', 'Completeness MAE (%)'),
        ('cont', 'true_contamination', 'pred_contamination', 'Contamination MAE (%)'),
    ]):
        ax = axes[ax_idx]

        x = np.arange(len(SETS))
        width = 0.18
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

        for i, tool in enumerate(TOOLS):
            maes = []
            ci_los = []
            ci_his = []
            for set_name in SETS:
                df = data[set_name].get(tool)
                if df is not None:
                    tv = df[true_col].values
                    pv = df[pred_col].values
                    mae_obs, ci_lo, ci_hi = bootstrap_mae(tv, pv)
                    maes.append(mae_obs)
                    ci_los.append(mae_obs - ci_lo)
                    ci_his.append(ci_hi - mae_obs)
                else:
                    maes.append(0)
                    ci_los.append(0)
                    ci_his.append(0)

            bars = ax.bar(
                x + offsets[i], maes, width,
                color=COLORS[tool], alpha=0.85, label=tool,
                yerr=[ci_los, ci_his], capsize=2, error_kw={'linewidth': 0.8}
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f'Set {s}' for s in SETS])
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.suptitle('Mean Absolute Error by Benchmark Set and Tool', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_set_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig2_set_comparison_bars.png'}")


def fig3_bland_altman(data):
    """Figure 3: Bland-Altman plots for MAGICC vs CheckM2."""
    print("\n  Generating Figure 3: Bland-Altman plots...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Combine all sets
    magicc_all = []
    checkm2_all = []
    for set_name in SETS:
        mdf = data[set_name].get('MAGICC')
        cdf = data[set_name].get('CheckM2')
        if mdf is not None and cdf is not None:
            merged = mdf.merge(cdf, on='genome_id', suffixes=('_m', '_c'))
            magicc_all.append(merged)
    if not magicc_all:
        print("    No data for Bland-Altman plots")
        return

    merged_all = pd.concat(magicc_all, ignore_index=True)

    for ax_idx, (metric, pred_m, pred_c, true_col, title) in enumerate([
        ('comp', 'pred_completeness_m', 'pred_completeness_c', 'true_completeness_m', 'Completeness'),
        ('cont', 'pred_contamination_m', 'pred_contamination_c', 'true_contamination_m', 'Contamination'),
    ]):
        ax = axes[ax_idx]

        # Bland-Altman: compare errors (MAGICC error vs CheckM2 error)
        # Or compare predictions directly
        magicc_vals = merged_all[pred_m].values
        checkm2_vals = merged_all[pred_c].values

        mean_vals = (magicc_vals + checkm2_vals) / 2
        diff_vals = magicc_vals - checkm2_vals  # positive = MAGICC predicts higher

        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        ax.scatter(mean_vals, diff_vals, s=4, alpha=0.3, c='#607D8B',
                   rasterized=True, edgecolors='none')

        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=1.2, label=f'Mean: {mean_diff:.2f}%')
        ax.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', linewidth=0.8,
                   label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}%')
        ax.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', linewidth=0.8,
                   label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}%')
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)

        ax.set_xlabel(f'Mean of MAGICC and CheckM2 {title} (%)')
        ax.set_ylabel(f'MAGICC - CheckM2 {title} (%)')
        ax.set_title(f'Bland-Altman: {title}', fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.2, linewidth=0.5)

    plt.suptitle('Bland-Altman Agreement: MAGICC vs CheckM2', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_bland_altman.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig3_bland_altman.png'}")


def fig4_speed_comparison(data):
    """Figure 4: Speed comparison bar chart (log scale)."""
    print("\n  Generating Figure 4: Speed comparison...")

    # Compute average speed per tool across all sets
    speed_data = {}
    for tool in TOOLS:
        speeds = []
        for set_name in SETS:
            df = data[set_name].get(tool)
            if df is not None and 'wall_clock_s' in df.columns and 'n_threads' in df.columns:
                wall = df['wall_clock_s'].iloc[0]
                threads = df['n_threads'].iloc[0]
                n = len(df)
                speed_per_thread = (n / wall * 60) / threads
                speeds.append(speed_per_thread)
        if speeds:
            speed_data[tool] = {
                'mean': np.mean(speeds),
                'min': np.min(speeds),
                'max': np.max(speeds),
                'speeds': speeds,
            }

    fig, ax = plt.subplots(figsize=(7, 5))

    tools_order = list(speed_data.keys())
    x = np.arange(len(tools_order))
    means = [speed_data[t]['mean'] for t in tools_order]
    colors = [COLORS[t] for t in tools_order]

    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, tool) in enumerate(zip(bars, tools_order)):
        val = means[i]
        # Get thread info
        if tool == 'MAGICC':
            thread_label = '1 thread'
        elif tool == 'CheckM2':
            thread_label = '32 threads'
        elif tool == 'CoCoPyE':
            thread_label = '48 threads'
        elif tool == 'DeepCheck':
            thread_label = '1 thread*'

        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                thread_label,
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(tools_order, fontweight='bold')
    ax.set_ylabel('Genomes/min/thread (log scale)')
    ax.set_title('Speed Comparison: Genomes per Minute per Thread', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Add footnote for DeepCheck
    ax.text(0.5, -0.12,
            '*DeepCheck: inference-only speed; requires CheckM2 feature extraction',
            transform=ax.transAxes, fontsize=8, ha='center', style='italic', color='gray')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig4_speed_comparison.png'}")


def fig5_magicc_100k_test(test_df):
    """Figure 5: MAGICC 100K test set scatter plots."""
    print("\n  Generating Figure 5: MAGICC 100K test set...")

    if test_df is None:
        print("    No 100K test data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax_idx, (true_col, pred_col, title) in enumerate([
        ('true_completeness', 'pred_completeness', 'Completeness (%)'),
        ('true_contamination', 'pred_contamination', 'Contamination (%)'),
    ]):
        ax = axes[ax_idx]

        # Plot each sample type
        sample_types = test_df['sample_type'].unique()
        for st in sorted(sample_types):
            mask = test_df['sample_type'] == st
            subset = test_df[mask]
            color = SAMPLE_TYPE_COLORS.get(st, '#999999')
            ax.scatter(
                subset[true_col], subset[pred_col],
                c=color, s=2, alpha=0.15, label=st.replace('_', ' ').title(),
                rasterized=True, edgecolors='none'
            )

        # Diagonal
        ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5)

        # Metrics annotation
        tv = test_df[true_col].values
        pv = test_df[pred_col].values
        mae = np.mean(np.abs(tv - pv))
        if np.std(tv) > 0 and np.std(pv) > 0:
            r2 = np.corrcoef(tv, pv)[0, 1] ** 2
        else:
            r2 = np.nan

        ax.text(0.05, 0.92, f'MAE={mae:.2f}%\nR$^2$={r2:.4f}\nn={len(test_df):,}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax.set_xlabel(f'True {title}')
        ax.set_ylabel(f'Predicted {title}')
        ax.set_title(f'MAGICC V3: {title}', fontweight='bold')
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)

        if ax_idx == 0:
            # Create legend with larger markers
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels, fontsize=8, loc='lower right',
                               markerscale=5, handletextpad=0.3, framealpha=0.9)

    plt.suptitle('MAGICC V3 Performance on 100,000 Test Genomes', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_magicc_100k_test.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig5_magicc_100k_test.png'}")


def fig6_per_set_detail(data):
    """Figure 6: 4x2 grid - per set, all tools overlaid."""
    print("\n  Generating Figure 6: Per-set detail plots...")

    markers = {
        'MAGICC': 'o',
        'CheckM2': 's',
        'CoCoPyE': '^',
        'DeepCheck': 'D',
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, set_name in enumerate(SETS):
        for row, (true_col, pred_col, ylabel) in enumerate([
            ('true_completeness', 'pred_completeness', 'Predicted Completeness (%)'),
            ('true_contamination', 'pred_contamination', 'Predicted Contamination (%)'),
        ]):
            ax = axes[row, col]

            for tool in TOOLS:
                df = data[set_name].get(tool)
                if df is None:
                    continue

                ax.scatter(
                    df[true_col], df[pred_col],
                    c=COLORS[tool], s=10, alpha=0.45,
                    marker=markers[tool], label=tool,
                    rasterized=True, edgecolors='none'
                )

            # Diagonal
            ax.plot([0, 100], [0, 100], 'k--', linewidth=0.8, alpha=0.5)
            ax.set_xlim(-2, 102)
            ax.set_ylim(-2, 102)

            if row == 0:
                ax.set_title(f'Set {set_name}', fontweight='bold')
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel(f'True {ylabel.split("Predicted ")[1]}')

            if row == 0 and col == 0:
                # Legend with unique handles
                handles = [Line2D([0], [0], marker=markers[t], color='w',
                                  markerfacecolor=COLORS[t], markersize=7, label=t)
                           for t in TOOLS]
                ax.legend(handles=handles, fontsize=7, loc='lower right')

    plt.suptitle('Tool Comparison by Benchmark Set (Completeness and Contamination)', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_per_set_detail.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig6_per_set_detail.png'}")


# ============================================================================
# Part E: Summary Table
# ============================================================================
def compute_summary_table(data, metrics_df):
    """Create overall tool comparison summary."""
    print("\n" + "=" * 70)
    print("PART E: TOOL COMPARISON SUMMARY")
    print("=" * 70)

    rows = []
    combined = _combine_data(data)

    for tool in TOOLS:
        cdf = combined[tool]
        if cdf is None:
            continue

        tc = cdf['true_completeness'].values
        tt = cdf['true_contamination'].values
        pc = cdf['pred_completeness'].values
        pt = cdf['pred_contamination'].values

        mc = compute_metrics(tc, pc)
        mt = compute_metrics(tt, pt)

        # Speed: average genomes/min/thread across sets
        speeds = []
        for set_name in SETS:
            df = data[set_name].get(tool)
            if df is not None and 'wall_clock_s' in df.columns:
                wall = df['wall_clock_s'].iloc[0]
                threads = df['n_threads'].iloc[0]
                n = len(df)
                speeds.append((n / wall * 60) / threads)
        avg_speed = np.mean(speeds) if speeds else np.nan

        # Total wall-clock for all sets
        total_wall = 0
        total_genomes = 0
        total_threads = 1
        for set_name in SETS:
            df = data[set_name].get(tool)
            if df is not None and 'wall_clock_s' in df.columns:
                total_wall += df['wall_clock_s'].iloc[0]
                total_genomes += len(df)
                total_threads = df['n_threads'].iloc[0]

        rows.append({
            'tool': tool,
            'n_genomes_total': mc['n'],
            'comp_mae_overall': mc['mae'],
            'comp_rmse_overall': mc['rmse'],
            'comp_r2_overall': mc['r2'],
            'comp_outlier_rate': mc['outlier_rate'],
            'cont_mae_overall': mt['mae'],
            'cont_rmse_overall': mt['rmse'],
            'cont_r2_overall': mt['r2'],
            'cont_outlier_rate': mt['outlier_rate'],
            'avg_speed_genomes_per_min_per_thread': avg_speed,
            'n_threads': total_threads,
            'total_wall_clock_s': total_wall,
        })

    summary_df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / 'tool_comparison_summary.tsv'
    summary_df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    print(f"  Saved: {output_path}")

    # Print formatted table
    print(f"\n  {'Tool':<10} {'N':>6} | {'Comp MAE':>9} {'Comp RMSE':>10} {'Comp R2':>8} | {'Cont MAE':>9} {'Cont RMSE':>10} {'Cont R2':>8} | {'Speed/min/thr':>14} {'Threads':>8}")
    print("  " + "-" * 105)
    for _, row in summary_df.iterrows():
        print(f"  {row['tool']:<10} {row['n_genomes_total']:>6.0f} | "
              f"{row['comp_mae_overall']:>8.2f}% {row['comp_rmse_overall']:>9.2f}% {row['comp_r2_overall']:>8.4f} | "
              f"{row['cont_mae_overall']:>8.2f}% {row['cont_rmse_overall']:>9.2f}% {row['cont_r2_overall']:>8.4f} | "
              f"{row['avg_speed_genomes_per_min_per_thread']:>13.1f} {row['n_threads']:>8.0f}")

    return summary_df


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK ANALYSIS AND FIGURE GENERATION")
    print("=" * 70)

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    print("\nLoading prediction data...")
    data = load_predictions()
    test_df = load_test_100k()

    # Verify data loaded
    for set_name in SETS:
        for tool in TOOLS:
            df = data[set_name].get(tool)
            status = f"{len(df)} rows" if df is not None else "MISSING"
            print(f"  Set {set_name}, {tool}: {status}")
    if test_df is not None:
        print(f"  Test 100K: {len(test_df)} rows")

    # Part A: Accuracy Metrics
    metrics_df = compute_accuracy_metrics(data)

    # Part B: MIMAG Classification
    mimag_df = compute_mimag_classification(data)

    # Part C: Statistical Tests
    stat_df = compute_statistical_tests(data)

    # Part D: Figures
    print("\n" + "=" * 70)
    print("PART D: PUBLICATION-QUALITY FIGURES")
    print("=" * 70)

    fig1_accuracy_scatter(data)
    fig2_set_comparison_bars(data)
    fig3_bland_altman(data)
    fig4_speed_comparison(data)
    fig5_magicc_100k_test(test_df)
    fig6_per_set_detail(data)

    # Part E: Summary Table
    summary_df = compute_summary_table(data, metrics_df)

    # Final timing
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\nOutput files:")
    print(f"  {RESULTS_DIR / 'accuracy_metrics.tsv'}")
    print(f"  {RESULTS_DIR / 'mimag_classification.tsv'}")
    print(f"  {RESULTS_DIR / 'statistical_tests.tsv'}")
    print(f"  {RESULTS_DIR / 'tool_comparison_summary.tsv'}")
    for fig_name in ['fig1_accuracy_scatter.png', 'fig2_set_comparison_bars.png',
                     'fig3_bland_altman.png', 'fig4_speed_comparison.png',
                     'fig5_magicc_100k_test.png', 'fig6_per_set_detail.png']:
        print(f"  {FIGURES_DIR / fig_name}")


if __name__ == '__main__':
    main()
