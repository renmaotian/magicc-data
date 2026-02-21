#!/usr/bin/env python3
"""
Script 43: Update motivating analysis figures to include Set C (realistic set).

Data sources:
- Motivating Set A: data/benchmarks/motivating_v2/set_A/ (controlled completeness)
- Motivating Set B: data/benchmarks/motivating_v2/set_B/ (controlled contamination)
- Motivating Set C: data/benchmarks/motivating_v2/set_C/ (realistic mixed)

Tools: CheckM2, CoCoPyE, DeepCheck (+ MAGICC for speed comparison)

Updated descriptions:
- Set A/B: "Controlled completeness/contamination sets"
- Set C: "Realistic set"

Figures generated (results/figures/):
  fig_motivating_completeness.png  -- Set A: 3 panels (CheckM2, CoCoPyE, DeepCheck)
  fig_motivating_contamination.png -- Set B: 3 panels, KEY figure
  fig_motivating_realistic.png     -- Set C: 3 panels, scatter predicted vs true
  fig_motivating_speed.png         -- Speed bar chart (3 tools, log scale)

Tables generated (results/):
  motivating_summary_v2.tsv (updated with Set C)

Usage:
    conda run -n magicc2 python scripts/43_update_motivating_figures_with_set_c.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

MOTIVATING_TOOLS = ['CheckM2', 'CoCoPyE', 'DeepCheck']
# Motivating analysis shows only competitor tools (no MAGICC) to demonstrate the GAP
ALL_TOOLS = ['CheckM2', 'CoCoPyE', 'DeepCheck']

TOOL_FILES = {
    'CheckM2': 'checkm2_predictions.tsv',
    'CoCoPyE': 'cocopye_predictions.tsv',
    'DeepCheck': 'deepcheck_predictions.tsv',
}

COLORS = {
    'CheckM2': '#FF9800',
    'CoCoPyE': '#4CAF50',
    'DeepCheck': '#9C27B0',
}

# Font sizes
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 11
TITLE_SIZE = 14
LEGEND_SIZE = 11

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
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ============================================================================
# Data Loading
# ============================================================================
def load_motivating_predictions():
    """Load all motivating set predictions (A, B, C)."""
    data = {}
    for set_name in ['A', 'B', 'C']:
        data[set_name] = {}
        set_dir = BENCHMARK_DIR / 'motivating_v2' / f'set_{set_name}'
        for tool in ALL_TOOLS:
            fpath = set_dir / TOOL_FILES[tool]
            if fpath.exists():
                df = pd.read_csv(fpath, sep='\t')
                df = df.dropna(subset=['pred_completeness', 'pred_contamination'])
                data[set_name][tool] = df
            else:
                print(f"  WARNING: Missing {fpath}")
                data[set_name][tool] = None
    return data


# ============================================================================
# Figure M1: Completeness Prediction (Set A) - unchanged
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
    fig.suptitle('Controlled Completeness Set (Set A, 0% contamination)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_motivating_completeness.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig_motivating_completeness.png'}")


# ============================================================================
# Figure M2: Contamination Prediction (Set B) - unchanged
# ============================================================================
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
        high_mask = df['true_contamination'] > 20
        high_mae = np.mean(np.abs(df.loc[high_mask, 'true_contamination'] - df.loc[high_mask, 'pred_contamination'])) if high_mask.sum() > 0 else 0

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
    fig.suptitle('Controlled Contamination Set (Set B, 100% completeness)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_motivating_contamination.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig_motivating_contamination.png'}")


# ============================================================================
# Figure M3: Realistic Set (Set C) - NEW
# ============================================================================
def fig_motivating_realistic(mot_data):
    """Fig M3: Scatter pred vs true for Set C (realistic mixed set).
    2 rows (completeness, contamination) x 3 columns (CheckM2, CoCoPyE, DeepCheck).
    """
    print("\n  Generating fig_motivating_realistic.png (NEW) ...")

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for col_idx, tool_name in enumerate(MOTIVATING_TOOLS):
        color = COLORS[tool_name]
        df = mot_data['C'].get(tool_name)
        if df is None:
            for row in range(2):
                axes[row, col_idx].text(0.5, 0.5, 'No data', transform=axes[row, col_idx].transAxes, ha='center')
            continue

        for row_idx, (true_col, pred_col, label) in enumerate([
            ('true_completeness', 'pred_completeness', 'Completeness'),
            ('true_contamination', 'pred_contamination', 'Contamination'),
        ]):
            ax = axes[row_idx, col_idx]

            tv = df[true_col].values
            pv = df[pred_col].values

            ax.scatter(tv, pv, c=color, s=10, alpha=0.4, edgecolors='none', rasterized=True)
            ax.plot([0, 100], [0, 100], 'k--', lw=1, alpha=0.5, zorder=10)

            mae = np.mean(np.abs(tv - pv))
            rmse = np.sqrt(np.mean((tv - pv)**2))
            if np.std(tv) > 0 and np.std(pv) > 0:
                r2 = np.corrcoef(tv, pv)[0, 1]**2
            else:
                r2 = float('nan')

            ax.text(0.05, 0.95,
                    f'MAE={mae:.1f}%\nRMSE={rmse:.1f}%\nR$^2$={r2:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

            ax.set_xlim(-2, 102)
            ax.set_ylim(-2, 102)

            if row_idx == 0:
                ax.set_title(tool_name, fontweight='bold', color=color)
            if col_idx == 0:
                ax.set_ylabel(f'Predicted {label} (%)')
            if row_idx == 1:
                ax.set_xlabel(f'True {label} (%)')

    fig.suptitle('Realistic Set (Set C): 1,000 genomes, varied completeness & contamination',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig_motivating_realistic.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {FIGURES_DIR / 'fig_motivating_realistic.png'}")


# ============================================================================
# Figure M4: Speed Comparison (updated with Set C data)
# ============================================================================
def fig_motivating_speed(mot_data):
    """Fig M4: Speed bar chart (genomes/min/thread, log scale), all 3 motivating sets combined."""
    print("\n  Generating fig_motivating_speed.png ...")

    speed_data = {}
    for tool_name in MOTIVATING_TOOLS:
        total_genomes = 0
        total_wall_clock = 0
        n_threads = 1
        for set_name in ['A', 'B', 'C']:
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
# Summary Table (updated with Set C)
# ============================================================================
def generate_motivating_summary(mot_data):
    """Motivating analysis summary with per-level breakdown for A/B and overall for C."""
    print("\n  Generating motivating_summary_v2.tsv (updated with Set C) ...")
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

    # Set C (realistic -- overall only, plus per sample_type)
    for tool_name in MOTIVATING_TOOLS:
        df = mot_data['C'].get(tool_name)
        if df is None:
            continue

        # Per sample type
        for st in sorted(df['sample_type'].unique()):
            sub = df[df['sample_type'] == st]
            comp_mae = np.mean(np.abs(sub['true_completeness'] - sub['pred_completeness']))
            cont_mae = np.mean(np.abs(sub['true_contamination'] - sub['pred_contamination']))
            rows.append({
                'set': 'Motivating C',
                'tool': tool_name,
                'level': st.replace('set_c_', ''),
                'n': len(sub),
                'comp_mae': round(comp_mae, 2),
                'cont_mae': round(cont_mae, 2),
            })

        # Overall
        comp_mae = np.mean(np.abs(df['true_completeness'] - df['pred_completeness']))
        cont_mae = np.mean(np.abs(df['true_contamination'] - df['pred_contamination']))
        comp_rmse = np.sqrt(np.mean((df['true_completeness'] - df['pred_completeness'])**2))
        cont_rmse = np.sqrt(np.mean((df['true_contamination'] - df['pred_contamination'])**2))

        tv_comp = df['true_completeness'].values
        pv_comp = df['pred_completeness'].values
        tv_cont = df['true_contamination'].values
        pv_cont = df['pred_contamination'].values

        r2_comp = np.corrcoef(tv_comp, pv_comp)[0,1]**2 if np.std(tv_comp) > 0 and np.std(pv_comp) > 0 else float('nan')
        r2_cont = np.corrcoef(tv_cont, pv_cont)[0,1]**2 if np.std(tv_cont) > 0 and np.std(pv_cont) > 0 else float('nan')

        wall_s = df['wall_clock_s'].iloc[0] if 'wall_clock_s' in df.columns else 0
        nt = int(df['n_threads'].iloc[0]) if 'n_threads' in df.columns else 1
        gpm_pt = len(df) / (wall_s / 60) / nt if wall_s > 0 else 0

        rows.append({
            'set': 'Motivating C',
            'tool': tool_name,
            'level': 'OVERALL',
            'n': len(df),
            'comp_mae': round(comp_mae, 2),
            'cont_mae': round(cont_mae, 2),
            'comp_rmse': round(comp_rmse, 2),
            'cont_rmse': round(cont_rmse, 2),
            'comp_r2': round(r2_comp, 4),
            'cont_r2': round(r2_cont, 4),
            'wall_clock_s': round(wall_s, 1),
            'genomes_per_min_per_thread': round(gpm_pt, 2),
        })

    summary_df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'motivating_summary_v2.tsv'
    summary_df.to_csv(out, sep='\t', index=False)
    print(f"    Saved: {out}")

    # Print just the OVERALL rows
    print("\n  === Motivating Analysis Summary (OVERALL rows) ===")
    overall = summary_df[summary_df['level'] == 'OVERALL']
    for _, r in overall.iterrows():
        extras = ""
        if pd.notna(r.get('comp_r2')) and r.get('comp_r2', 0) > 0:
            extras = f" R2_comp={r['comp_r2']:.4f} R2_cont={r.get('cont_r2', 0):.4f}"
        if pd.notna(r.get('cont_mae_gt20')) and r.get('cont_mae_gt20', 0) > 0:
            extras += f" cont_MAE>20%={r['cont_mae_gt20']:.1f}%"
        print(f"  {r['set']:>13s} {r['tool']:<10s} n={r['n']:>5d}  "
              f"comp_MAE={r['comp_mae']:>5.2f}%  cont_MAE={r['cont_mae']:>5.2f}%  "
              f"wall={r.get('wall_clock_s', 0):>6.0f}s  G/min/thr={r.get('genomes_per_min_per_thread', 0):>5.2f}"
              f"{extras}")

    return summary_df


# ============================================================================
# Accuracy Metrics for all tools on Set C
# ============================================================================
def compute_set_c_metrics(mot_data):
    """Compute and print detailed accuracy metrics for all tools on Set C."""
    print("\n  === SET C ACCURACY METRICS ===")
    print(f"  {'Tool':<12} {'Comp MAE':>10} {'Cont MAE':>10} {'Comp RMSE':>11} {'Cont RMSE':>11} "
          f"{'Comp R2':>8} {'Cont R2':>8} {'Wall(s)':>8} {'G/min/thr':>10}")
    print("  " + "-" * 100)

    metrics = []
    for tool_name in ALL_TOOLS:
        df = mot_data['C'].get(tool_name)
        if df is None:
            print(f"  {tool_name:<12} NO DATA")
            continue

        tv_comp = df['true_completeness'].values
        pv_comp = df['pred_completeness'].values
        tv_cont = df['true_contamination'].values
        pv_cont = df['pred_contamination'].values

        mae_comp = np.mean(np.abs(tv_comp - pv_comp))
        mae_cont = np.mean(np.abs(tv_cont - pv_cont))
        rmse_comp = np.sqrt(np.mean((tv_comp - pv_comp)**2))
        rmse_cont = np.sqrt(np.mean((tv_cont - pv_cont)**2))
        r2_comp = np.corrcoef(tv_comp, pv_comp)[0,1]**2 if np.std(tv_comp) > 0 and np.std(pv_comp) > 0 else float('nan')
        r2_cont = np.corrcoef(tv_cont, pv_cont)[0,1]**2 if np.std(tv_cont) > 0 and np.std(pv_cont) > 0 else float('nan')

        wall_s = df['wall_clock_s'].iloc[0] if 'wall_clock_s' in df.columns else 0
        nt = int(df['n_threads'].iloc[0]) if 'n_threads' in df.columns else 1
        gpm_pt = len(df) / (wall_s / 60) / nt if wall_s > 0 else 0

        # For DeepCheck, effective speed = CheckM2 speed
        effective_wall = wall_s
        effective_threads = nt
        if tool_name == 'DeepCheck':
            checkm2_df = mot_data['C'].get('CheckM2')
            if checkm2_df is not None and 'wall_clock_s' in checkm2_df.columns:
                effective_wall = checkm2_df['wall_clock_s'].iloc[0]
                effective_threads = int(checkm2_df['n_threads'].iloc[0])
                gpm_pt = len(df) / (effective_wall / 60) / effective_threads

        print(f"  {tool_name:<12} {mae_comp:>9.2f}% {mae_cont:>9.2f}% {rmse_comp:>10.2f}% {rmse_cont:>10.2f}% "
              f"{r2_comp:>8.4f} {r2_cont:>8.4f} {effective_wall:>7.0f}s {gpm_pt:>9.2f}")

        metrics.append({
            'tool': tool_name,
            'comp_mae': mae_comp,
            'cont_mae': mae_cont,
            'comp_rmse': rmse_comp,
            'cont_rmse': rmse_cont,
            'comp_r2': r2_comp,
            'cont_r2': r2_cont,
            'wall_clock_s': effective_wall,
            'n_threads': effective_threads,
            'genomes_per_min_per_thread': gpm_pt,
            'n_genomes': len(df),
        })

    return metrics


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("UPDATE MOTIVATING FIGURES (with Set C)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    mot_data = load_motivating_predictions()

    for s in ['A', 'B', 'C']:
        for tool in ALL_TOOLS:
            df = mot_data[s].get(tool)
            status = f"{len(df)} rows" if df is not None else "MISSING"
            print(f"  Motivating {s}, {tool}: {status}")

    # Generate figures
    print("\n" + "=" * 70)
    print("MOTIVATING FIGURES")
    print("=" * 70)
    fig_motivating_completeness(mot_data)
    fig_motivating_contamination(mot_data)
    fig_motivating_realistic(mot_data)
    speed_data = fig_motivating_speed(mot_data)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLES")
    print("=" * 70)
    summary_df = generate_motivating_summary(mot_data)

    # Accuracy metrics for Set C
    print("\n" + "=" * 70)
    print("SET C DETAILED METRICS")
    print("=" * 70)
    set_c_metrics = compute_set_c_metrics(mot_data)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL MOTIVATING FIGURES UPDATED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\nFigure files:")
    for f in sorted(FIGURES_DIR.glob('fig_motivating*')):
        print(f"  {f}")
    print(f"\nTable files:")
    for f in sorted(RESULTS_DIR.glob('motivating*')):
        print(f"  {f}")


if __name__ == '__main__':
    main()
