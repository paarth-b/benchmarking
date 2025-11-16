#!/usr/bin/env python
"""
Scientific visualization utilities for benchmarking results.
Uses seaborn's built-in functionality for publication-quality plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import datetime


# Set global style once at import
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams['savefig.dpi'] = 300


def plot_kde_comparison(df, pred_col, truth_col, output_path=None):
    """
    Create KDE plot comparing predicted scores to ground truth.

    Args:
        df: DataFrame with predicted and ground truth scores
        pred_col: Column name for predictions (e.g., 'tmvec_student_score')
        truth_col: Column name for ground truth (e.g., 'tmalign_score')
        output_path: Path to save figure (without extension), optional
    """
    # Reshape data for seaborn
    plot_df = pd.DataFrame({
        'TM-score': pd.concat([df[truth_col], df[pred_col]]),
        'Method': ['TM-align (Ground Truth)'] * len(df) + ['TMvec-Student (Predicted)'] * len(df)
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn's kdeplot with hue
    sns.kdeplot(data=plot_df, x='TM-score', hue='Method', fill=True, alpha=0.5, ax=ax)

    # Add mean lines
    for method, col in [('TM-align (Ground Truth)', truth_col), ('TMvec-Student (Predicted)', pred_col)]:
        mean_val = df[col].mean()
        ax.axvline(mean_val, linestyle='--', linewidth=2,
                   label=f'{method.split()[0]} mean: {mean_val:.3f}')

    ax.set_title('Distribution Comparison: TMvec-Student vs TM-align')
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_path}.pdf', bbox_inches='tight')

    return fig


def plot_runtime_comparison(df, x_col, method_col, runtime_col, output_path=None):
    """
    Create line plot comparing runtimes of different methods.

    Args:
        df: DataFrame with runtime data
        x_col: X-axis column (e.g., 'sequence_length')
        method_col: Column identifying methods
        runtime_col: Runtime column (in seconds)
        output_path: Path to save figure (without extension), optional
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn lineplot with built-in error bands
    sns.lineplot(
        data=df,
        x=x_col,
        y=runtime_col,
        hue=method_col,
        marker='o',
        markersize=8,
        linewidth=2,
        ax=ax
    )

    ax.set_yscale('log')
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel('Runtime (seconds, log scale)')
    ax.set_title('Runtime Comparison Across Methods')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_path}.pdf', bbox_inches='tight')

    return fig


def plot_correlation_scatter(df, x_col, y_col, output_path=None):
    """
    Create scatterplot with Pearson correlation analysis.

    Args:
        df: DataFrame with scores
        x_col: X-axis column (e.g., 'tmalign_score')
        y_col: Y-axis column (e.g., 'tmvec_student_score')
        output_path: Path to save figure (without extension), optional

    Returns:
        Tuple of (figure, statistics dict)
    """
    plot_df = df[[x_col, y_col]].dropna()

    # Calculate statistics
    pearson_r, pearson_p = stats.pearsonr(plot_df[x_col], plot_df[y_col])
    rmse = np.sqrt(np.mean((plot_df[x_col] - plot_df[y_col]) ** 2))
    mae = np.mean(np.abs(plot_df[x_col] - plot_df[y_col]))

    statistics = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'rmse': rmse,
        'mae': mae,
        'n': len(plot_df)
    }

    # Create plot using seaborn's JointGrid
    g = sns.JointGrid(data=plot_df, x=x_col, y=y_col, height=8)

    # Main scatterplot with regression
    g.plot_joint(sns.scatterplot, alpha=0.5, s=20)
    g.plot_joint(sns.regplot, scatter=False, color='red')

    # Marginal distributions
    g.plot_marginals(sns.histplot, kde=True)

    # Add identity line
    lims = [
        min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]),
        max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])
    ]
    g.ax_joint.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='y=x')

    # Add statistics text
    stats_text = (f'Pearson R = {pearson_r:.3f}\n'
                  f'p-value = {pearson_p:.2e}\n'
                  f'RMSE = {rmse:.3f}\n'
                  f'MAE = {mae:.3f}\n'
                  f'n = {len(plot_df):,}')

    g.ax_joint.text(0.05, 0.95, stats_text, transform=g.ax_joint.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    g.ax_joint.set_xlabel(x_col.replace('_', ' ').title())
    g.ax_joint.set_ylabel(y_col.replace('_', ' ').title())
    g.fig.suptitle('Correlation: Predicted vs Ground Truth', y=1.02)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        g.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        g.savefig(f'{output_path}.pdf', bbox_inches='tight')

    return g.fig, statistics


if __name__ == "__main__":
    # Load actual results
    print("Loading CSV files...")
    df_tmalign = pd.read_csv('results/tmalign_similarities.csv').head(100000)
    df_tmvec_student = pd.read_csv('/scratch/akeluska/prot_distill_divide/benchmarking/results/tmvec_student_similarities.csv').head(100000)

    # Merge on sequence IDs
    # Clean seq IDs for matching (remove range info from tmvec_student IDs)
    df_tmvec_student['seq1_clean'] = df_tmvec_student['seq1_id'].str.replace(r'/\d+-\d+', '', regex=True)
    df_tmvec_student['seq2_clean'] = df_tmvec_student['seq2_id'].str.replace(r'/\d+-\d+', '', regex=True)

    df_merged = pd.merge(
        df_tmalign[['seq1_id', 'seq2_id', 'tm_score']],
        df_tmvec_student[['seq1_clean', 'seq2_clean', 'tm_score']],
        left_on=['seq1_id', 'seq2_id'],
        right_on=['seq1_clean', 'seq2_clean'],
        suffixes=('_tmalign', '_tmvec_student')
    )

    # Harmonize column names for plotting helpers
    df_merged = df_merged.rename(columns={
        'tm_score_tmalign': 'score_tmalign',
        'tm_score_tmvec_student': 'score_tmvec_student'
    })

    print(f"Merged {len(df_merged)} pairs\n")
    summary = (
        df_merged[['score_tmalign', 'score_tmvec_student']]
        .describe()
        .rename(columns={
            'score_tmalign': 'tm_align',
            'score_tmvec_student': 'tmvec_student'
        })
    )
    print("Score summary (TM-align vs. TMvec-Student):")
    print(summary.to_string(float_format=lambda x: f"{x:0.6f}"))
    print()

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'figures/{timestamp}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_dir}/\n")

    # Plot 1: KDE
    print("1. KDE comparison")
    fig1 = plot_kde_comparison(df_merged, 'score_tmvec_student', 'score_tmalign',
                               f'{output_dir}/kde_comparison')

    # Plot 2: Correlation
    print("2. Correlation scatter")
    fig2, stats = plot_correlation_scatter(df_merged, 'score_tmalign',
                                          'score_tmvec_student', f'{output_dir}/correlation')

    print(f"\nStats: R={stats['pearson_r']:.3f}, RMSE={stats['rmse']:.3f}, n={stats['n']}")
    print(f"\nDone. Check {output_dir}/ directory")
