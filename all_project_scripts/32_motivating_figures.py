#!/usr/bin/env python3
"""
Motivating Analysis Figures for MAGICC Manuscript.

Creates publication-quality figures demonstrating limitations of existing tools
(CheckM2, DeepCheck, CoCoPyE) in contamination detection and speed.

Figures:
  M1: Completeness prediction accuracy (Set A_motiv)
  M2: Contamination prediction accuracy (Set B_motiv) - KEY figure
  M3: Speed comparison bar chart

Output:
  results/figures/fig_motivating_completeness.png
  results/figures/fig_motivating_contamination.png
  results/figures/fig_motivating_speed.png
  results/motivating_summary.tsv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR = '/home/tianrm/projects/magicc2'
BENCHMARK_DIR = os.path.join(PROJECT_DIR, 'data', 'benchmarks', 'motivating')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Tool colors (Nature Methods style)
COLORS = {
    'CheckM2': '#FF9800',   # orange
    'CoCoPyE': '#4CAF50',   # green
    'DeepCheck': '#9C27B0',  # purple
}

# Tool files
TOOLS = {
    'CheckM2': 'checkm2_predictions.tsv',
    'CoCoPyE': 'cocopye_predictions.tsv',
    'DeepCheck': 'deepcheck_predictions.tsv',
}

# Font sizes
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 11
TITLE_SIZE = 14
LEGEND_SIZE = 11

# Matplotlib style
plt.rcParams.update({
    'font.size': TICK_SIZE,
    'axes.labelsize': AXIS_LABEL_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ============================================================================
# Data Loading
# ============================================================================

def load_predictions(set_name):
    """Load all tool predictions for a set."""
    data = {}
    for tool_name, filename in TOOLS.items():
        path = os.path.join(BENCHMARK_DIR, f'set_{set_name}', filename)
        df = pd.read_csv(path, sep='\t')
        df = df.dropna(subset=['pred_completeness', 'pred_contamination'])
        data[tool_name] = df
    return data


# ============================================================================
# Figure M1: Completeness Prediction
# ============================================================================

def figure_m1_completeness():
    """
    X-axis: True completeness level (50%, 60%, 70%, 80%, 90%, 100%)
    Y-axis: Predicted completeness (%)
    3 panels side by side, one per tool, with boxplots and diagonal reference.
    """
    print("\nGenerating Figure M1: Completeness prediction...")
    data = load_predictions('A')

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), sharey=True)

    completeness_levels = [50, 60, 70, 80, 90, 100]

    for ax_idx, (tool_name, df) in enumerate(data.items()):
        ax = axes[ax_idx]
        color = COLORS[tool_name]

        # Collect data per level
        box_data = []
        positions = []
        for target in completeness_levels:
            mask = df['sample_type'] == f'set_a_comp{target}'
            preds = df.loc[mask, 'pred_completeness'].values
            trues = df.loc[mask, 'true_completeness'].values
            box_data.append(preds)
            positions.append(target)

        # Boxplot
        bp = ax.boxplot(box_data, positions=positions, widths=6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='gray'),
                        capprops=dict(color='gray'))

        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Reference diagonal
        ax.plot([45, 105], [45, 105], 'k--', lw=1, alpha=0.5, zorder=0)

        # Compute MAE
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

    fig.suptitle('Completeness Prediction: Existing Tools (Set A, 0% contamination)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig_motivating_completeness.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================================
# Figure M2: Contamination Prediction (KEY FIGURE)
# ============================================================================

def figure_m2_contamination():
    """
    X-axis: True contamination level (0%, 5%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 80%, 100%)
    Y-axis: Predicted contamination (%)
    3 panels side by side, with boxplots and diagonal reference.
    Shows dramatic underestimation at high contamination levels.
    """
    print("\nGenerating Figure M2: Contamination prediction (KEY FIGURE)...")
    data = load_predictions('B')

    contamination_levels = [0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 4.0), sharey=True)

    # Use integer positions 0..10 for uniform spacing, then label with actual contamination levels
    x_positions = list(range(len(contamination_levels)))

    for ax_idx, (tool_name, df) in enumerate(data.items()):
        ax = axes[ax_idx]
        color = COLORS[tool_name]

        box_data = []
        actual_means = []
        for target in contamination_levels:
            mask = df['sample_type'] == f'set_b_cont{target}'
            preds = df.loc[mask, 'pred_contamination'].values
            trues = df.loc[mask, 'true_contamination'].values
            box_data.append(preds)
            actual_means.append(trues.mean())

        # Boxplot with uniformly spaced positions
        bp = ax.boxplot(box_data, positions=x_positions, widths=0.6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='gray'),
                        capprops=dict(color='gray'))

        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Reference diagonal: map actual contamination levels to positions
        # Plot a line connecting (position, target) for each level
        ax.plot(x_positions, contamination_levels, 'k--', lw=1, alpha=0.5, zorder=0)

        # Overall MAE
        cont_mae = np.mean(np.abs(df['true_contamination'] - df['pred_contamination']))

        # Per-level MAEs for key levels
        high_mask = df['true_contamination'] >= 40
        high_mae = np.mean(np.abs(df.loc[high_mask, 'true_contamination'] - df.loc[high_mask, 'pred_contamination']))

        ax.text(0.05, 0.95, f'MAE = {cont_mae:.1f}%\nMAE(>40%) = {high_mae:.1f}%',
                transform=ax.transAxes, fontsize=ANNOTATION_SIZE - 1,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('True Contamination (%)')
        ax.set_title(tool_name, fontweight='bold', color=color)
        ax.set_xlim(-0.8, len(contamination_levels) - 0.2)
        ax.set_ylim(-10, 130)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(c) for c in contamination_levels], fontsize=9, rotation=45)

    axes[0].set_ylabel('Predicted Contamination (%)')

    fig.suptitle('Contamination Prediction: Existing Tools (Set B, 100% completeness)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig_motivating_contamination.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================================
# Figure M3: Speed Comparison
# ============================================================================

def figure_m3_speed():
    """
    Bar chart comparing speed of the 3 tools (genomes/min/thread).
    Uses log scale Y-axis.
    """
    print("\nGenerating Figure M3: Speed comparison...")

    # Collect timing data from prediction files
    speed_data = {}

    for tool_name in TOOLS:
        total_genomes = 0
        total_wall_clock = 0
        n_threads = 1

        for set_name in ['A', 'B']:
            path = os.path.join(BENCHMARK_DIR, f'set_{set_name}', TOOLS[tool_name])
            df = pd.read_csv(path, sep='\t')

            if 'wall_clock_s' in df.columns:
                wall_s = df['wall_clock_s'].iloc[0]
            else:
                wall_s = 0

            if 'n_threads' in df.columns:
                n_threads = int(df['n_threads'].iloc[0])

            total_genomes += len(df)
            total_wall_clock += wall_s

        if total_wall_clock > 0:
            genomes_per_min = total_genomes / (total_wall_clock / 60)
            genomes_per_min_per_thread = genomes_per_min / n_threads
        else:
            genomes_per_min = 0
            genomes_per_min_per_thread = 0

        speed_data[tool_name] = {
            'total_genomes': total_genomes,
            'total_wall_clock_s': total_wall_clock,
            'n_threads': n_threads,
            'genomes_per_min': genomes_per_min,
            'genomes_per_min_per_thread': genomes_per_min_per_thread,
        }

    # For DeepCheck, use CheckM2's wall-clock (since DeepCheck needs CheckM2 features)
    # DeepCheck effective speed = CheckM2 speed
    checkm2_wall = speed_data['CheckM2']['total_wall_clock_s']
    deepcheck_threads = speed_data['CheckM2']['n_threads']  # Uses CheckM2's threads
    deepcheck_total_genomes = speed_data['DeepCheck']['total_genomes']
    deepcheck_effective_gpm = deepcheck_total_genomes / (checkm2_wall / 60) if checkm2_wall > 0 else 0
    deepcheck_effective_gpm_per_thread = deepcheck_effective_gpm / deepcheck_threads

    speed_data['DeepCheck']['effective_genomes_per_min'] = deepcheck_effective_gpm
    speed_data['DeepCheck']['effective_genomes_per_min_per_thread'] = deepcheck_effective_gpm_per_thread
    speed_data['DeepCheck']['effective_n_threads'] = deepcheck_threads
    speed_data['DeepCheck']['effective_wall_clock_s'] = checkm2_wall

    print("\n  Speed Summary:")
    for tool, sd in speed_data.items():
        threads = sd['n_threads']
        gpm = sd['genomes_per_min']
        gpm_pt = sd['genomes_per_min_per_thread']
        print(f"  {tool:12s}: {sd['total_genomes']} genomes, {sd['total_wall_clock_s']:.0f}s, "
              f"{threads} threads, {gpm:.1f} G/min, {gpm_pt:.2f} G/min/thread")
        if 'effective_genomes_per_min_per_thread' in sd:
            print(f"                (effective with CheckM2: {sd['effective_genomes_per_min']:.1f} G/min, "
                  f"{sd['effective_genomes_per_min_per_thread']:.2f} G/min/thread)")

    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    tool_names = ['CheckM2', 'CoCoPyE', 'DeepCheck']
    # Use genomes/min/thread for fair comparison
    # DeepCheck: use effective speed (includes CheckM2 feature extraction)
    speeds = []
    threads_list = []
    for t in tool_names:
        sd = speed_data[t]
        if t == 'DeepCheck':
            speeds.append(sd['effective_genomes_per_min_per_thread'])
            threads_list.append(sd['effective_n_threads'])
        else:
            speeds.append(sd['genomes_per_min_per_thread'])
            threads_list.append(sd['n_threads'])

    bars = ax.bar(range(len(tool_names)), speeds,
                  color=[COLORS[t] for t in tool_names],
                  edgecolor='white', linewidth=1.2,
                  width=0.55, alpha=0.85)

    ax.set_yscale('log')
    ax.set_ylabel('Speed (genomes/min/thread)')
    ax.set_xticks(range(len(tool_names)))
    ax.set_xticklabels(tool_names, fontweight='bold')

    # Annotate bars with speed and thread count
    for i, (bar, speed, threads) in enumerate(zip(bars, speeds, threads_list)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                f'{speed:.2f}',
                ha='center', va='bottom', fontsize=ANNOTATION_SIZE,
                fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                f'{threads} thr',
                ha='center', va='center', fontsize=9,
                fontweight='normal', color='white')

    # Add horizontal reference line at 1 genome/min/thread
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(2.55, 1.15, '1 G/min/thr', fontsize=9, color='gray', alpha=0.7)

    ax.set_ylim(0.1, 3)
    ax.set_title('Processing Speed\n(all tools < 1 genome/min/thread)',
                 fontweight='bold', fontsize=TITLE_SIZE - 1)

    # Note about DeepCheck
    ax.text(0.5, -0.18,
            'DeepCheck requires CheckM2 feature extraction;\neffective end-to-end speed shown',
            transform=ax.transAxes, fontsize=8, ha='center', style='italic', color='gray')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig_motivating_speed.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

    return speed_data


# ============================================================================
# Summary Statistics Table
# ============================================================================

def generate_summary(speed_data):
    """Generate summary TSV with all results."""
    print("\nGenerating summary table...")

    rows = []

    for set_name, set_label in [('A', 'Completeness gradient'), ('B', 'Contamination gradient')]:
        for tool_name in TOOLS:
            path = os.path.join(BENCHMARK_DIR, f'set_{set_name}', TOOLS[tool_name])
            df = pd.read_csv(path, sep='\t')
            valid = df.dropna(subset=['pred_completeness', 'pred_contamination'])

            comp_mae = np.mean(np.abs(valid['true_completeness'] - valid['pred_completeness']))
            cont_mae = np.mean(np.abs(valid['true_contamination'] - valid['pred_contamination']))

            wall_s = valid['wall_clock_s'].iloc[0] if 'wall_clock_s' in valid.columns else 0
            n_threads = int(valid['n_threads'].iloc[0]) if 'n_threads' in valid.columns else 1

            if tool_name == 'DeepCheck':
                # Effective speed includes CheckM2 time
                checkm2_path = os.path.join(BENCHMARK_DIR, f'set_{set_name}', 'checkm2_predictions.tsv')
                ckm2_df = pd.read_csv(checkm2_path, sep='\t')
                effective_wall_s = ckm2_df['wall_clock_s'].iloc[0]
                effective_threads = int(ckm2_df['n_threads'].iloc[0])
                genomes_per_min = len(valid) / (effective_wall_s / 60) if effective_wall_s > 0 else 0
                genomes_per_min_per_thread = genomes_per_min / effective_threads
                wall_s_used = effective_wall_s
                threads_used = effective_threads
            else:
                genomes_per_min = len(valid) / (wall_s / 60) if wall_s > 0 else 0
                genomes_per_min_per_thread = genomes_per_min / n_threads
                wall_s_used = wall_s
                threads_used = n_threads

            rows.append({
                'set': f'Set {set_name} ({set_label})',
                'tool': tool_name,
                'n_genomes': len(valid),
                'comp_mae': round(comp_mae, 2),
                'cont_mae': round(cont_mae, 2),
                'wall_clock_s': round(wall_s_used, 1),
                'n_threads': threads_used,
                'genomes_per_min': round(genomes_per_min, 1),
                'genomes_per_min_per_thread': round(genomes_per_min_per_thread, 2),
            })

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'motivating_summary.tsv')
    summary_df.to_csv(out_path, sep='\t', index=False)
    print(f"  Saved: {out_path}")

    # Print table
    print("\n  === Motivating Analysis Summary ===")
    print(summary_df.to_string(index=False))

    return summary_df


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Motivating Figures Generator")
    print("=" * 70)

    figure_m1_completeness()
    figure_m2_contamination()
    speed_data = figure_m3_speed()
    generate_summary(speed_data)

    print("\n\nAll figures generated successfully.")
    print(f"Output directory: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
