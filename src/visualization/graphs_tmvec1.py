#!/usr/bin/env python
"""
Scientific visualization utilities for TMvec-1 benchmarking results.
Uses seaborn's built-in functionality for publication-quality plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import datetime


sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams['savefig.dpi'] = 300


def plot_kde_comparison(df, pred_col, truth_col, output_path=None):
    """
    Create KDE plot comparing predicted scores to ground truth.

    Args:
        df: DataFrame with predicted and ground truth scores
        pred_col: Column name for predictions (e.g., 'tmvec1_score')
        truth_col: Column name for ground truth (e.g., 'tmalign_score')
        output_path: Path to save figure (without extension), optional
    """
    plot_df = pd.DataFrame({
        'TM-score': pd.concat([df[truth_col], df[pred_col]]),
        'Method': ['TM-align (Ground Truth)'] * len(df) + ['TMvec-1 (Predicted)'] * len(df)
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(data=plot_df, x='TM-score', hue='Method', fill=True, alpha=0.5, ax=ax)

    for method, col in [('TM-align (Ground Truth)', truth_col), ('TMvec-1 (Predicted)', pred_col)]:
        mean_val = df[col].mean()
        ax.axvline(mean_val, linestyle='--', linewidth=2,
                   label=f'{method.split()[0]} mean: {mean_val:.3f}')

    ax.set_title('Distribution Comparison: TMvec-1 vs TM-align')
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
        y_col: Y-axis column (e.g., 'tmvec1_score')
        output_path: Path to save figure (without extension), optional

    Returns:
        Tuple of (figure, statistics dict)
    """
    plot_df = df[[x_col, y_col]].dropna()

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

    g = sns.JointGrid(data=plot_df, x=x_col, y=y_col, height=8)

    g.plot_joint(sns.scatterplot, alpha=0.5, s=20)
    g.plot_joint(sns.regplot, scatter=False, color='red')

    g.plot_marginals(sns.histplot, kde=True)

    lims = [
        min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]),
        max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])
    ]
    g.ax_joint.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='y=x')

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


def compute_confusion_matrix(df, truth_col, pred_col, threshold=0.5, output_path=None):
    """
    Compute confusion matrix statistics for binary classification using TM-score threshold.

    Args:
        df: DataFrame with predicted and ground truth scores
        truth_col: Column name for ground truth scores
        pred_col: Column name for predicted scores
        threshold: TM-score threshold for classification (default: 0.5)
        output_path: Path to save CSV (without extension), optional

    Returns:
        DataFrame with confusion matrix statistics
    """
    plot_df = df[[truth_col, pred_col]].dropna()

    y_true = (plot_df[truth_col] >= threshold).astype(int)
    y_pred = (plot_df[pred_col] >= threshold).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    results = pd.DataFrame({
        'metric': ['threshold', 'true_positives', 'true_negatives', 'false_positives',
                  'false_negatives', 'precision', 'recall', 'f1_score', 'accuracy', 'total_samples'],
        'value': [threshold, tp, tn, fp, fn, precision, recall, f1, accuracy, len(plot_df)]
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(f'{output_path}.csv', index=False)

    return results


if __name__ == "__main__":
    print("Loading CSV files...")
    df_tmalign = pd.read_csv('results/tmalign_similarities.csv')
    df_tmvec1 = pd.read_csv('results/tmvec1_similarities.csv').head(100000)

    df_tmvec1['seq1_clean'] = df_tmvec1['seq1_id'].str.replace(r'/\d+-\d+', '', regex=True)
    df_tmvec1['seq2_clean'] = df_tmvec1['seq2_id'].str.replace(r'/\d+-\d+', '', regex=True)

    df_merged = pd.merge(
        df_tmalign[['seq1_id', 'seq2_id', 'tm_score']],
        df_tmvec1[['seq1_clean', 'seq2_clean', 'tm_score']],
        left_on=['seq1_id', 'seq2_id'],
        right_on=['seq1_clean', 'seq2_clean'],
        suffixes=('_tmalign', '_tmvec1')
    )

    print(f"Merged {len(df_merged)} pairs\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'figures/tmvec1_{timestamp}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_dir}/\n")

    print("1. KDE comparison")
    fig1 = plot_kde_comparison(df_merged, 'tm_score_tmvec1', 'tm_score_tmalign',
                               f'{output_dir}/kde_comparison')

    print("2. Correlation scatter")
    fig2, stats = plot_correlation_scatter(df_merged, 'tm_score_tmalign',
                                          'tm_score_tmvec1', f'{output_dir}/correlation')

    print("3. Confusion matrix (threshold=0.5)")
    conf_matrix = compute_confusion_matrix(df_merged, 'tm_score_tmalign', 'tm_score_tmvec1',
                                          threshold=0.5, output_path=f'{output_dir}/confusion_matrix')

    print(f"\nStats: R={stats['pearson_r']:.3f}, RMSE={stats['rmse']:.3f}, n={stats['n']}")
    print(f"Confusion Matrix (threshold=0.5):")
    print(f"  Accuracy: {conf_matrix.loc[conf_matrix['metric'] == 'accuracy', 'value'].iloc[0]:.3f}")
    print(f"  Precision: {conf_matrix.loc[conf_matrix['metric'] == 'precision', 'value'].iloc[0]:.3f}")
    print(f"  Recall: {conf_matrix.loc[conf_matrix['metric'] == 'recall', 'value'].iloc[0]:.3f}")
    print(f"  F1-Score: {conf_matrix.loc[conf_matrix['metric'] == 'f1_score', 'value'].iloc[0]:.3f}")
    print(f"\nDone. Check {output_dir}/ directory")
