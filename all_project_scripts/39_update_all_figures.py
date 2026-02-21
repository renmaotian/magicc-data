#!/usr/bin/env python3
"""
Script 39: Update ALL figures using v2 benchmark data.

Data sources:
- Motivating: data/benchmarks/motivating_v2/set_{A,B}/ (CheckM2, CoCoPyE, DeepCheck + MAGICC)
- Benchmark: data/benchmarks/set_{A_v2,B_v2}/ + set_E/ (all 4 tools)
- Sets C, D: data/benchmarks/set_{C,D}/ (Patescibacteria, Archaea -- unchanged)
- MAGICC 100K test: data/benchmarks/test_100k/magicc_predictions.tsv

Figures generated (results/figures/):
  fig_motivating_completeness.png  -- Motivating Set A: 3 panels (CheckM2, CoCoPyE, DeepCheck)
  fig_motivating_contamination.png -- Motivating Set B: 3 panels, KEY figure
  fig_motivating_speed.png         -- Speed bar chart (3 tools, log scale)
  fig1_accuracy_scatter.png        -- 2x4 scatter (comp/cont x 4 tools)
  fig2_set_comparison_bars.png     -- Grouped bar MAE with bootstrap CI
  fig3_bland_altman.png            -- MAGICC vs CheckM2 Bland-Altman
  fig4_speed_comparison.png        -- Speed bar chart including MAGICC
  fig5_magicc_100k_test.png        -- MAGICC 100K test scatter
  fig6_per_set_detail.png          -- 5 sets x 2 metrics, all 4 tools

Tables generated (results/):
  accuracy_metrics_v2.tsv
  tool_comparison_summary_v2.tsv
  motivating_summary_v2.tsv

Usage:
    conda run -n magicc2 python scripts/39_update_all_figures.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

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
MOTIVATING_TOOLS = ['CheckM2', 'CoCoPyE', 'DeepCheck']

TOOL_FILES = {
    'MAGICC': 'magicc_predictions.tsv',
    'CheckM2': 'checkm2_predictions.tsv',
    'CoCoPyE': 'cocopye_predictions.tsv',
    'DeepCheck': 'deepcheck_predictions.tsv',
}

# Benchmark sets for fig1-fig6 (v2 + existing C/D)
BENCHMARK_SETS = OrderedDict([
    ('A_v2', {'dir': 'set_A_v2', 'label': 'Set A (Completeness)'}),
    ('B_v2', {'dir': 'set_B_v2', 'label': 'Set B (Contamination)'}),
    ('C', {'dir': 'set_C', 'label': 'Set C (Patescibacteria)'}),
    ('D', {'dir': 'set_D', 'label': 'Set D (Archaea)'}),
    ('E', {'dir': 'set_E', 'label': 'Set E (Mixed)'}),
])

# Colors
COLORS = {
    'MAGICC': '#2196F3',
    'CheckM2': '#FF9800',
    'CoCoPyE': '#4CAF50',
    'DeepCheck': '#9C27B0',
}

SET_COLORS = {
    'A_v2': '#E91E63',   # pink
    'B_v2': '#3F51B5',   # indigo
    'C': '#009688',      # teal
    'D': '#FF5722',      # deep orange
    'E': '#795548',      # brown
}

SAMPLE_TYPE_COLORS = {
    'pure': '#2196F3',
    'complete': '#4CAF50',
    'within_phylum': '#FF9800',
    'cross_phylum': '#F44336',
    'reduced_genome': '#9C27B0',
    'archaeal': '#795548',
}

# Font sizes
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 11
TITLE_SIZE = 14
LEGEND_SIZE = 11

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': TICK_SIZE,
    'axes.labelsize': AXIS_LABEL_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ============================================================================
# Data Loading
# ============================================================================
def load_benchmark_predictions():
    """Load all benchmark set predictions: data[set_key][tool] = DataFrame."""
    data = {}
    for set_key, info in BENCHMARK_SETS.items():
        data[set_key] = {}
        set_dir = BENCHMARK_DIR / info['dir']
        for tool in TOOLS:
            fpath = set_dir / TOOL_FILES[tool]
            if fpath.exists():
                df = pd.read_csv(fpath, sep='\t')
                data[set_key][tool] = df
            else:
                print(f"  WARNING: Missing {fpath}")
                data[set_key][tool] = None
    return data


def load_motivating_predictions():
    """Load motivating v2 set predictions."""
    data = {}
    for set_name in ['A', 'B']:
        data[set_name] = {}
        set_dir = BENCHMARK_DIR / 'motivating_v2' / f'set_{set_name}'
        for tool in MOTIVATING_TOOLS:
            fpath = set_dir / TOOL_FILES[tool]
            if fpath.exists():
                df = pd.read_csv(fpath, sep='\t')
                df = df.dropna(subset=['pred_completeness', 'pred_contamination'])
                data[set_name][tool] = df
            else:
                print(f"  WARNING: Missing {fpath}")
                data[set_name][tool] = None
        # Also load MAGICC for speed comparison
        magicc_path = set_dir / TOOL_FILES['MAGICC']
        if magicc_path.exists():
            data[set_name]['MAGICC'] = pd.read_csv(magicc_path, sep='\t')
    return data


def load_test_100k():
    """Load MAGICC 100K test set."""
    fpath = BENCHMARK_DIR / 'test_100k' / 'magicc_predictions.tsv'
    if fpath.exists():
        return pd.read_csv(fpath, sep='\t')
    print(f"  WARNING: Missing {fpath}")
    return None


# ============================================================================
# Metric helpers
# ============================================================================
def compute_metrics(true_vals, pred_vals):
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


def bootstrap_mae(true_vals, pred_vals, n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    tv, pv = true_vals[mask], pred_vals[mask]
    n = len(tv)
    if n == 0:
        return np.nan, np.nan, np.nan
    errors = np.abs(tv - pv)
    mae_obs = np.mean(errors)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_maes = np.mean(errors[indices], axis=1)
    return mae_obs, np.percentile(boot_maes, 2.5), np.percentile(boot_maes, 97.5)


# ============================================================================
# MOTIVATING FIGURES
# ============================================================================

def fig_motivating_completeness(mot_data):
    """Fig M1: Boxplot, 3 panels (CheckM2/CoCoPyE/DeepCheck), x=true comp level, y=pred comp."""
    print("\n  Generating fig_motivating_completeness.png ...")
    completeness_levels = [50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), sharey=True)

    for ax_idx, tool_name in enumerate(MOTIVATING_TOOLS):
        ax = axes[ax_idx]
        color = COLORS[tool_name]
        df = mot_data['A'].get(tool_name)
        if df is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        box_data = []
        for target in completeness_levels:
            mask = df['sample_type'] == f'set_a_comp{target}'
            preds = df.loc[mask, 'pred_completeness'].values
            box_data.append(preds)

        bp = ax.boxplot(box_data, positions=completeness_levels, widths=6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='gray'),
                        capprops=dict(color='gray'))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.plot([45, 105], [45, 105], 'k--', lw=1, alpha=0.5, zorder=0)

        comp_mae = np.mean(np.abs(df['true_completeness'] - df['pred_completeness']))
        ax.text(0.05, 0.95, f'MAE = {comp_mae:.1f}%',
                transform=ax.transAxes, fontsize=ANNOTATION_SIZE,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('True Completeness (%)')
        ax.set_title(tool_name, fontweight='bold', color=color)
        ax.set_xlim(42, 108)
        ax.set_ylim(30, 110)
        ax.set_xticks(completeness_levels)
        ax.set_xticklabels([str(c) for c in completeness_levels], fontsize=10)

    axes[0].set_ylabel('Predicted Completeness (%)')
    fig.suptitle('Completeness Prediction: Existing Tools (Motivating Set A, 0% contamination)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_motivating_completeness.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig_motivating_completeness.png'}")


def fig_motivating_contamination(mot_data):
    """Fig M2: Boxplot, 3 panels, x=true contamination level (0/20/40/60/80%), y=pred cont."""
    print("\n  Generating fig_motivating_contamination.png (KEY FIGURE) ...")
    contamination_levels = [0, 20, 40, 60, 80]

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 4.0), sharey=True)
    x_positions = list(range(len(contamination_levels)))

    for ax_idx, tool_name in enumerate(MOTIVATING_TOOLS):
        ax = axes[ax_idx]
        color = COLORS[tool_name]
        df = mot_data['B'].get(tool_name)
        if df is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        box_data = []
        for target in contamination_levels:
            mask = df['sample_type'] == f'set_b_cont{target}'
            preds = df.loc[mask, 'pred_contamination'].values
            box_data.append(preds)

        bp = ax.boxplot(box_data, positions=x_positions, widths=0.6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='gray'),
                        capprops=dict(color='gray'))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.plot(x_positions, contamination_levels, 'k--', lw=1, alpha=0.5, zorder=0)

        cont_mae = np.mean(np.abs(df['true_contamination'] - df['pred_contamination']))
        # MAE for >20% contamination
        high_mask = df['true_contamination'] > 20
        if high_mask.sum() > 0:
            high_mae = np.mean(np.abs(df.loc[high_mask, 'true_contamination'] - df.loc[high_mask, 'pred_contamination']))
        else:
            high_mae = 0

        ax.text(0.05, 0.95, f'MAE = {cont_mae:.1f}%\nMAE(>20%) = {high_mae:.1f}%',
                transform=ax.transAxes, fontsize=ANNOTATION_SIZE - 1,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('True Contamination (%)')
        ax.set_title(tool_name, fontweight='bold', color=color)
        ax.set_xlim(-0.8, len(contamination_levels) - 0.2)
        ax.set_ylim(-10, 110)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(c) for c in contamination_levels], fontsize=10)

    axes[0].set_ylabel('Predicted Contamination (%)')
    fig.suptitle('Contamination Prediction: Existing Tools (Motivating Set B, 100% completeness)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_motivating_contamination.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig_motivating_contamination.png'}")


def fig_motivating_speed(mot_data):
    """Fig M3: Speed bar chart (genomes/min/thread, log scale), motivating_v2 combined."""
    print("\n  Generating fig_motivating_speed.png ...")

    speed_data = {}
    for tool_name in MOTIVATING_TOOLS:
        total_genomes = 0
        total_wall_clock = 0
        n_threads = 1
        for set_name in ['A', 'B']:
            df = mot_data[set_name].get(tool_name)
            if df is None:
                continue
            if 'wall_clock_s' in df.columns:
                wall_s = df['wall_clock_s'].iloc[0]
            else:
                wall_s = 0
            if 'n_threads' in df.columns:
                n_threads = int(df['n_threads'].iloc[0])
            total_genomes += len(df)
            total_wall_clock += wall_s

        if total_wall_clock > 0:
            gpm = total_genomes / (total_wall_clock / 60)
            gpm_pt = gpm / n_threads
        else:
            gpm = gpm_pt = 0

        speed_data[tool_name] = {
            'total_genomes': total_genomes,
            'total_wall_clock_s': total_wall_clock,
            'n_threads': n_threads,
            'gpm': gpm,
            'gpm_pt': gpm_pt,
        }

    # DeepCheck effective speed = CheckM2 speed
    checkm2_wall = speed_data['CheckM2']['total_wall_clock_s']
    checkm2_threads = speed_data['CheckM2']['n_threads']
    dc_genomes = speed_data['DeepCheck']['total_genomes']
    dc_effective_gpm = dc_genomes / (checkm2_wall / 60) if checkm2_wall > 0 else 0
    dc_effective_gpm_pt = dc_effective_gpm / checkm2_threads
    speed_data['DeepCheck']['effective_gpm_pt'] = dc_effective_gpm_pt
    speed_data['DeepCheck']['effective_threads'] = checkm2_threads

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    tool_names = ['CheckM2', 'CoCoPyE', 'DeepCheck']
    speeds = []
    threads_list = []
    for t in tool_names:
        sd = speed_data[t]
        if t == 'DeepCheck':
            speeds.append(sd['effective_gpm_pt'])
            threads_list.append(sd['effective_threads'])
        else:
            speeds.append(sd['gpm_pt'])
            threads_list.append(sd['n_threads'])

    bars = ax.bar(range(len(tool_names)), speeds,
                  color=[COLORS[t] for t in tool_names],
                  edgecolor='white', linewidth=1.2,
                  width=0.55, alpha=0.85)

    ax.set_yscale('log')
    ax.set_ylabel('Speed (genomes/min/thread)')
    ax.set_xticks(range(len(tool_names)))
    ax.set_xticklabels(tool_names, fontweight='bold')

    for i, (bar, speed, threads) in enumerate(zip(bars, speeds, threads_list)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                f'{speed:.2f}', ha='center', va='bottom', fontsize=ANNOTATION_SIZE, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                f'{threads} thr', ha='center', va='center', fontsize=9, color='white')

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(2.55, 1.15, '1 G/min/thr', fontsize=9, color='gray', alpha=0.7)

    ax.set_ylim(0.1, 3)
    ax.set_title('Processing Speed\n(all tools < 1 genome/min/thread)',
                 fontweight='bold', fontsize=TITLE_SIZE - 1)

    ax.text(0.5, -0.18,
            'DeepCheck requires CheckM2 feature extraction;\neffective end-to-end speed shown',
            transform=ax.transAxes, fontsize=8, ha='center', style='italic', color='gray')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_motivating_speed.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig_motivating_speed.png'}")

    return speed_data


# ============================================================================
# BENCHMARKING FIGURES
# ============================================================================

def _combine_benchmark_data(data, tools=TOOLS):
    """Combine all benchmark sets into one DataFrame per tool."""
    combined = {tool: [] for tool in tools}
    for set_key in BENCHMARK_SETS:
        for tool in tools:
            df = data[set_key].get(tool)
            if df is not None:
                df_copy = df.copy()
                df_copy['set'] = set_key
                combined[tool].append(df_copy)
    for tool in tools:
        if combined[tool]:
            combined[tool] = pd.concat(combined[tool], ignore_index=True)
        else:
            combined[tool] = None
    return combined


def fig1_accuracy_scatter(data):
    """Figure 1: 2x4 grid scatter (comp/cont x 4 tools). Color by set."""
    print("\n  Generating fig1_accuracy_scatter.png ...")
    combined = _combine_benchmark_data(data)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for col, tool in enumerate(TOOLS):
        cdf = combined[tool]
        if cdf is None:
            continue

        for row, (true_col, pred_col, label) in enumerate([
            ('true_completeness', 'pred_completeness', 'Completeness (%)'),
            ('true_contamination', 'pred_contamination', 'Contamination (%)'),
        ]):
            ax = axes[row, col]

            for set_key in BENCHMARK_SETS:
                subset = cdf[cdf['set'] == set_key]
                if len(subset) == 0:
                    continue
                ax.scatter(
                    subset[true_col], subset[pred_col],
                    c=SET_COLORS[set_key], s=6, alpha=0.4,
                    label=f'Set {set_key}', rasterized=True, edgecolors='none'
                )

            ax.plot([0, 100], [0, 100], 'k--', linewidth=0.8, alpha=0.5, zorder=10)

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

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc='lower right', markerscale=2)

    plt.suptitle('Predicted vs True: All Tools on Benchmark Sets A_v2, B_v2, C, D, E',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_accuracy_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig1_accuracy_scatter.png'}")


def fig2_set_comparison_bars(data):
    """Figure 2: Grouped bar MAE per set per tool. Two panels (comp, cont). Bootstrap 95% CI."""
    print("\n  Generating fig2_set_comparison_bars.png ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    set_keys = list(BENCHMARK_SETS.keys())

    for ax_idx, (true_col, pred_col, title) in enumerate([
        ('true_completeness', 'pred_completeness', 'Completeness MAE (%)'),
        ('true_contamination', 'pred_contamination', 'Contamination MAE (%)'),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(set_keys))
        width = 0.18
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

        for i, tool in enumerate(TOOLS):
            maes, ci_los, ci_his = [], [], []
            for set_key in set_keys:
                df = data[set_key].get(tool)
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

            ax.bar(x + offsets[i], maes, width,
                   color=COLORS[tool], alpha=0.85, label=tool,
                   yerr=[ci_los, ci_his], capsize=2, error_kw={'linewidth': 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels([f'Set {s}' for s in set_keys])
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.suptitle('Mean Absolute Error by Benchmark Set and Tool (with 95% CI)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_set_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig2_set_comparison_bars.png'}")


def fig3_bland_altman(data):
    """Figure 3: MAGICC vs CheckM2 Bland-Altman on Sets A_v2+B_v2+C+D+E combined."""
    print("\n  Generating fig3_bland_altman.png ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    merged_all = []
    for set_key in BENCHMARK_SETS:
        mdf = data[set_key].get('MAGICC')
        cdf = data[set_key].get('CheckM2')
        if mdf is not None and cdf is not None:
            merged = mdf.merge(cdf, on='genome_id', suffixes=('_m', '_c'))
            merged_all.append(merged)

    if not merged_all:
        print("    No data for Bland-Altman plots")
        plt.close(fig)
        return

    merged_all = pd.concat(merged_all, ignore_index=True)

    for ax_idx, (pred_m, pred_c, title) in enumerate([
        ('pred_completeness_m', 'pred_completeness_c', 'Completeness'),
        ('pred_contamination_m', 'pred_contamination_c', 'Contamination'),
    ]):
        ax = axes[ax_idx]

        magicc_vals = merged_all[pred_m].values
        checkm2_vals = merged_all[pred_c].values
        mean_vals = (magicc_vals + checkm2_vals) / 2
        diff_vals = magicc_vals - checkm2_vals

        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        ax.scatter(mean_vals, diff_vals, s=4, alpha=0.3, c='#607D8B',
                   rasterized=True, edgecolors='none')

        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=1.2,
                   label=f'Mean: {mean_diff:.2f}%')
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

    plt.suptitle('Bland-Altman Agreement: MAGICC vs CheckM2 (All Benchmark Sets)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_bland_altman.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig3_bland_altman.png'}")


def fig4_speed_comparison(data):
    """Figure 4: Speed bar chart including MAGICC. Log scale."""
    print("\n  Generating fig4_speed_comparison.png ...")

    speed_data = {}
    for tool in TOOLS:
        speeds = []
        for set_key in BENCHMARK_SETS:
            df = data[set_key].get(tool)
            if df is not None and 'wall_clock_s' in df.columns and 'n_threads' in df.columns:
                wall = df['wall_clock_s'].iloc[0]
                threads = df['n_threads'].iloc[0]
                n = len(df)
                if wall > 0:
                    speeds.append((n / wall * 60) / threads)
        if speeds:
            speed_data[tool] = {'mean': np.mean(speeds), 'speeds': speeds}

    # DeepCheck effective end-to-end speed = CheckM2 speed (requires CheckM2 feature extraction)
    if 'DeepCheck' in speed_data and 'CheckM2' in speed_data:
        speed_data['DeepCheck']['mean'] = speed_data['CheckM2']['mean']
        speed_data['DeepCheck']['speeds'] = speed_data['CheckM2']['speeds']

    # Only show 3 tools: MAGICC, CheckM2, CoCoPyE (DeepCheck = CheckM2 speed, redundant)
    tools_to_plot = ['MAGICC', 'CheckM2', 'CoCoPyE']
    tools_order = [t for t in tools_to_plot if t in speed_data]

    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(tools_order))
    means = [speed_data[t]['mean'] for t in tools_order]
    colors = [COLORS[t] for t in tools_order]

    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    thread_labels = {
        'MAGICC': '1 thread',
        'CheckM2': '32 threads',
        'CoCoPyE': '48 threads',
    }

    for i, (bar, tool) in enumerate(zip(bars, tools_order)):
        val = means[i]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                thread_labels.get(tool, ''),
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(tools_order, fontweight='bold')
    ax.set_ylabel('Genomes/min/thread (log scale)')
    ax.set_title('Speed Comparison: Genomes per Minute per Thread', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    ax.text(0.5, -0.10,
            'DeepCheck omitted (requires CheckM2 features; effective speed = CheckM2)',
            transform=ax.transAxes, fontsize=8, ha='center', style='italic', color='gray')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig4_speed_comparison.png'}")


def fig5_magicc_100k_test(test_df):
    """Figure 5: MAGICC 100K test scatter (unchanged, use existing data)."""
    print("\n  Generating fig5_magicc_100k_test.png ...")

    if test_df is None:
        print("    No 100K test data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax_idx, (true_col, pred_col, title) in enumerate([
        ('true_completeness', 'pred_completeness', 'Completeness (%)'),
        ('true_contamination', 'pred_contamination', 'Contamination (%)'),
    ]):
        ax = axes[ax_idx]
        sample_types = test_df['sample_type'].unique()
        for st in sorted(sample_types):
            mask = test_df['sample_type'] == st
            subset = test_df[mask]
            color = SAMPLE_TYPE_COLORS.get(st, '#999999')
            ax.scatter(subset[true_col], subset[pred_col],
                       c=color, s=2, alpha=0.15, label=st.replace('_', ' ').title(),
                       rasterized=True, edgecolors='none')

        ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5)

        tv = test_df[true_col].values
        pv = test_df[pred_col].values
        mae = np.mean(np.abs(tv - pv))
        r2 = np.corrcoef(tv, pv)[0, 1] ** 2 if np.std(tv) > 0 and np.std(pv) > 0 else np.nan

        ax.text(0.05, 0.92, f'MAE={mae:.2f}%\nR$^2$={r2:.4f}\nn={len(test_df):,}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax.set_xlabel(f'True {title}')
        ax.set_ylabel(f'Predicted {title}')
        ax.set_title(f'MAGICC V3: {title}', fontweight='bold')
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)

        if ax_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=8, loc='lower right',
                      markerscale=5, handletextpad=0.3, framealpha=0.9)

    plt.suptitle('MAGICC V3 Performance on 100,000 Test Genomes', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_magicc_100k_test.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig5_magicc_100k_test.png'}")


def fig6_per_set_detail(data):
    """Figure 6: 5 sets x 2 metrics (comp, cont), all 4 tools overlaid."""
    print("\n  Generating fig6_per_set_detail.png ...")

    markers = {'MAGICC': 'o', 'CheckM2': 's', 'CoCoPyE': '^', 'DeepCheck': 'D'}
    set_keys = list(BENCHMARK_SETS.keys())

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for col, set_key in enumerate(set_keys):
        for row, (true_col, pred_col, ylabel) in enumerate([
            ('true_completeness', 'pred_completeness', 'Predicted Completeness (%)'),
            ('true_contamination', 'pred_contamination', 'Predicted Contamination (%)'),
        ]):
            ax = axes[row, col]
            for tool in TOOLS:
                df = data[set_key].get(tool)
                if df is None:
                    continue
                ax.scatter(df[true_col], df[pred_col],
                           c=COLORS[tool], s=10, alpha=0.45,
                           marker=markers[tool], label=tool,
                           rasterized=True, edgecolors='none')

            ax.plot([0, 100], [0, 100], 'k--', linewidth=0.8, alpha=0.5)
            ax.set_xlim(-2, 102)
            ax.set_ylim(-2, 102)

            if row == 0:
                ax.set_title(f'Set {set_key}', fontweight='bold')
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel(f'True {ylabel.split("Predicted ")[1]}')

            if row == 0 and col == 0:
                handles = [Line2D([0], [0], marker=markers[t], color='w',
                                  markerfacecolor=COLORS[t], markersize=7, label=t)
                           for t in TOOLS]
                ax.legend(handles=handles, fontsize=7, loc='lower right')

    plt.suptitle('Tool Comparison by Benchmark Set (5 Sets, Completeness and Contamination)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_per_set_detail.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig6_per_set_detail.png'}")


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def generate_accuracy_metrics_table(data):
    """Per set, per tool accuracy table."""
    print("\n  Generating accuracy_metrics_v2.tsv ...")
    rows = []
    for set_key in BENCHMARK_SETS:
        for tool in TOOLS:
            df = data[set_key].get(tool)
            if df is None:
                continue
            mc = compute_metrics(df['true_completeness'].values, df['pred_completeness'].values)
            mt = compute_metrics(df['true_contamination'].values, df['pred_contamination'].values)
            rows.append({
                'set': set_key,
                'tool': tool,
                'n_genomes': mc['n'],
                'comp_mae': mc['mae'], 'comp_rmse': mc['rmse'],
                'comp_r2': mc['r2'], 'comp_outlier_rate': mc['outlier_rate'],
                'cont_mae': mt['mae'], 'cont_rmse': mt['rmse'],
                'cont_r2': mt['r2'], 'cont_outlier_rate': mt['outlier_rate'],
            })

    metrics_df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'accuracy_metrics_v2.tsv'
    metrics_df.to_csv(out, sep='\t', index=False, float_format='%.4f')
    print(f"    Saved: {out}")

    # Print formatted
    print(f"\n  {'Set':<6} {'Tool':<10} {'N':>6} | {'Comp MAE':>9} {'Comp R2':>8} | {'Cont MAE':>9} {'Cont R2':>8}")
    print("  " + "-" * 70)
    for _, r in metrics_df.iterrows():
        print(f"  {r['set']:<6} {r['tool']:<10} {r['n_genomes']:>6.0f} | "
              f"{r['comp_mae']:>8.2f}% {r['comp_r2']:>8.4f} | "
              f"{r['cont_mae']:>8.2f}% {r['cont_r2']:>8.4f}")

    return metrics_df


def generate_tool_comparison_summary(data):
    """Overall summary across all benchmark sets."""
    print("\n  Generating tool_comparison_summary_v2.tsv ...")
    combined = _combine_benchmark_data(data)
    rows = []

    for tool in TOOLS:
        cdf = combined[tool]
        if cdf is None:
            continue

        mc = compute_metrics(cdf['true_completeness'].values, cdf['pred_completeness'].values)
        mt = compute_metrics(cdf['true_contamination'].values, cdf['pred_contamination'].values)

        # Speed
        speeds = []
        for set_key in BENCHMARK_SETS:
            df = data[set_key].get(tool)
            if df is not None and 'wall_clock_s' in df.columns:
                wall = df['wall_clock_s'].iloc[0]
                threads = df['n_threads'].iloc[0]
                n = len(df)
                if wall > 0:
                    speeds.append((n / wall * 60) / threads)
        avg_speed = np.mean(speeds) if speeds else np.nan

        # Total wall-clock
        total_wall = sum(
            data[sk].get(tool, pd.DataFrame()).get('wall_clock_s', pd.Series([0])).iloc[0]
            for sk in BENCHMARK_SETS if data[sk].get(tool) is not None and 'wall_clock_s' in data[sk].get(tool, pd.DataFrame()).columns
        )

        rows.append({
            'tool': tool,
            'n_genomes_total': mc['n'],
            'comp_mae': mc['mae'], 'comp_rmse': mc['rmse'], 'comp_r2': mc['r2'],
            'cont_mae': mt['mae'], 'cont_rmse': mt['rmse'], 'cont_r2': mt['r2'],
            'avg_speed_genomes_per_min_per_thread': avg_speed,
            'total_wall_clock_s': total_wall,
        })

    summary_df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'tool_comparison_summary_v2.tsv'
    summary_df.to_csv(out, sep='\t', index=False, float_format='%.4f')
    print(f"    Saved: {out}")

    print(f"\n  {'Tool':<10} {'N':>6} | {'Comp MAE':>9} {'Comp R2':>8} | {'Cont MAE':>9} {'Cont R2':>8} | {'Speed/min/thr':>14}")
    print("  " + "-" * 85)
    for _, r in summary_df.iterrows():
        print(f"  {r['tool']:<10} {r['n_genomes_total']:>6.0f} | "
              f"{r['comp_mae']:>8.2f}% {r['comp_r2']:>8.4f} | "
              f"{r['cont_mae']:>8.2f}% {r['cont_r2']:>8.4f} | "
              f"{r['avg_speed_genomes_per_min_per_thread']:>13.1f}")

    return summary_df


def generate_motivating_summary(mot_data):
    """Motivating analysis summary with per-level breakdown."""
    print("\n  Generating motivating_summary_v2.tsv ...")
    rows = []

    # Set A (completeness levels)
    completeness_levels = [50, 60, 70, 80, 90, 100]
    for tool_name in MOTIVATING_TOOLS:
        df = mot_data['A'].get(tool_name)
        if df is None:
            continue
        for target in completeness_levels:
            mask = df['sample_type'] == f'set_a_comp{target}'
            sub = df[mask]
            if len(sub) == 0:
                continue
            comp_mae = np.mean(np.abs(sub['true_completeness'] - sub['pred_completeness']))
            cont_mae = np.mean(np.abs(sub['true_contamination'] - sub['pred_contamination']))
            rows.append({
                'set': 'Motivating A',
                'tool': tool_name,
                'level': f'comp_{target}%',
                'n': len(sub),
                'comp_mae': round(comp_mae, 2),
                'cont_mae': round(cont_mae, 2),
            })
        # Overall
        comp_mae = np.mean(np.abs(df['true_completeness'] - df['pred_completeness']))
        cont_mae = np.mean(np.abs(df['true_contamination'] - df['pred_contamination']))
        wall_s = df['wall_clock_s'].iloc[0] if 'wall_clock_s' in df.columns else 0
        nt = int(df['n_threads'].iloc[0]) if 'n_threads' in df.columns else 1
        gpm_pt = len(df) / (wall_s / 60) / nt if wall_s > 0 else 0
        rows.append({
            'set': 'Motivating A',
            'tool': tool_name,
            'level': 'OVERALL',
            'n': len(df),
            'comp_mae': round(comp_mae, 2),
            'cont_mae': round(cont_mae, 2),
            'wall_clock_s': round(wall_s, 1),
            'genomes_per_min_per_thread': round(gpm_pt, 2),
        })

    # Set B (contamination levels)
    contamination_levels = [0, 20, 40, 60, 80]
    for tool_name in MOTIVATING_TOOLS:
        df = mot_data['B'].get(tool_name)
        if df is None:
            continue
        for target in contamination_levels:
            mask = df['sample_type'] == f'set_b_cont{target}'
            sub = df[mask]
            if len(sub) == 0:
                continue
            comp_mae = np.mean(np.abs(sub['true_completeness'] - sub['pred_completeness']))
            cont_mae = np.mean(np.abs(sub['true_contamination'] - sub['pred_contamination']))
            rows.append({
                'set': 'Motivating B',
                'tool': tool_name,
                'level': f'cont_{target}%',
                'n': len(sub),
                'comp_mae': round(comp_mae, 2),
                'cont_mae': round(cont_mae, 2),
            })
        # Overall
        comp_mae = np.mean(np.abs(df['true_completeness'] - df['pred_completeness']))
        cont_mae = np.mean(np.abs(df['true_contamination'] - df['pred_contamination']))
        wall_s = df['wall_clock_s'].iloc[0] if 'wall_clock_s' in df.columns else 0
        nt = int(df['n_threads'].iloc[0]) if 'n_threads' in df.columns else 1
        gpm_pt = len(df) / (wall_s / 60) / nt if wall_s > 0 else 0
        # MAE >20%
        high_mask = df['true_contamination'] > 20
        high_mae = np.mean(np.abs(df.loc[high_mask, 'true_contamination'] - df.loc[high_mask, 'pred_contamination'])) if high_mask.sum() > 0 else 0
        rows.append({
            'set': 'Motivating B',
            'tool': tool_name,
            'level': 'OVERALL',
            'n': len(df),
            'comp_mae': round(comp_mae, 2),
            'cont_mae': round(cont_mae, 2),
            'cont_mae_gt20': round(high_mae, 2),
            'wall_clock_s': round(wall_s, 1),
            'genomes_per_min_per_thread': round(gpm_pt, 2),
        })

    summary_df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'motivating_summary_v2.tsv'
    summary_df.to_csv(out, sep='\t', index=False)
    print(f"    Saved: {out}")
    return summary_df


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("UPDATE ALL FIGURES (v2 Data)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    bench_data = load_benchmark_predictions()
    mot_data = load_motivating_predictions()
    test_df = load_test_100k()

    # Verify
    for sk, info in BENCHMARK_SETS.items():
        for tool in TOOLS:
            df = bench_data[sk].get(tool)
            status = f"{len(df)} rows" if df is not None else "MISSING"
            print(f"  Benchmark {sk}, {tool}: {status}")
    for s in ['A', 'B']:
        for tool in MOTIVATING_TOOLS + ['MAGICC']:
            df = mot_data[s].get(tool)
            status = f"{len(df)} rows" if df is not None else "MISSING"
            print(f"  Motivating {s}, {tool}: {status}")
    if test_df is not None:
        print(f"  Test 100K: {len(test_df)} rows")

    # === MOTIVATING FIGURES ===
    print("\n" + "=" * 70)
    print("MOTIVATING FIGURES")
    print("=" * 70)
    fig_motivating_completeness(mot_data)
    fig_motivating_contamination(mot_data)
    speed_data = fig_motivating_speed(mot_data)

    # === BENCHMARKING FIGURES ===
    print("\n" + "=" * 70)
    print("BENCHMARKING FIGURES")
    print("=" * 70)
    fig1_accuracy_scatter(bench_data)
    fig2_set_comparison_bars(bench_data)
    fig3_bland_altman(bench_data)
    fig4_speed_comparison(bench_data)
    fig5_magicc_100k_test(test_df)
    fig6_per_set_detail(bench_data)

    # === SUMMARY TABLES ===
    print("\n" + "=" * 70)
    print("SUMMARY TABLES")
    print("=" * 70)
    metrics_df = generate_accuracy_metrics_table(bench_data)
    summary_df = generate_tool_comparison_summary(bench_data)
    mot_summary = generate_motivating_summary(mot_data)

    # === FINAL REPORT ===
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL FIGURES UPDATED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\nFigure files:")
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f"  {f}")
    print(f"\nTable files:")
    for f in sorted(RESULTS_DIR.glob('*.tsv')):
        print(f"  {f}")


if __name__ == '__main__':
    main()
