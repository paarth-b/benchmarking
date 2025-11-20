#!/usr/bin/env python
"""
Scientific visualization utilities for benchmarking results.
Uses seaborn's built-in functionality for publication-quality plots.
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Set global style once at import
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams['savefig.dpi'] = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TM-Vec1 benchmarking results.")
    parser.add_argument(
        "--tmvec1",
        type=str,
        default=None,
        help="Path to TM-Vec1 similarity CSV (default: results/tmvec1_similarities.csv).",
    )
    parser.add_argument(
        "--tmalign",
        type=str,
        default="results/tmalign_similarities.csv",
        help="Path to TM-align similarity CSV (default: results/tmalign_similarities.csv).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100_000,
        help="Maximum number of TM-Vec1 pairs to load for plotting (default: 100000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store generated figures (default: figures/tmvec1_<timestamp>).",
    )
    parser.add_argument(
        "--results-dir",
        dest="legacy_results_dir",
        type=str,
        default=None,
        help="Deprecated alias for --tmvec1 (kept for backwards compatibility).",
    )
    return parser.parse_args()


def plot_kde_comparison(df, pred_col, truth_col, output_path=None):
    """
    Create KDE plot comparing predicted scores to ground truth.

    Args:
        df: DataFrame with predicted and ground truth scores
        pred_col: Column name for predictions (e.g., 'tmvec1_score')
        truth_col: Column name for ground truth (e.g., 'tmalign_score')
        output_path: Path to save figure (without extension), optional
    """
    # Reshape data for seaborn
    plot_df = pd.DataFrame({
        'TM-score': pd.concat([df[truth_col], df[pred_col]]),
        'Method': ['TM-align (Ground Truth)'] * len(df) + ['TMvec-1 (Predicted)'] * len(df)
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn's kdeplot with hue
    sns.kdeplot(data=plot_df, x='TM-score', hue='Method', fill=True, alpha=0.5, ax=ax)

    # Add mean lines
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
        y_col: Y-axis column (e.g., 'tmvec1_score')
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
    # Drop NA values
    plot_df = df[[truth_col, pred_col]].dropna()

    # Convert to binary classification
    y_true = (plot_df[truth_col] >= threshold).astype(int)
    y_pred = (plot_df[pred_col] >= threshold).astype(int)

    # Compute confusion matrix elements
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # Compute derived metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Create results dataframe
    results = pd.DataFrame({
        'metric': ['threshold', 'true_positives', 'true_negatives', 'false_positives',
                  'false_negatives', 'precision', 'recall', 'f1_score', 'accuracy', 'total_samples'],
        'value': [threshold, tp, tn, fp, fn, precision, recall, f1, accuracy, len(plot_df)]
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(f'{output_path}.csv', index=False)

    return results


def main():
    args = parse_args()

    tmvec1_path = Path(args.tmvec1 or args.legacy_results_dir or 'results/tmvec1_similarities.csv')
    tmalign_path = Path(args.tmalign)

    missing = [
        (label, path) for label, path in
        (("TM-Vec1", tmvec1_path), ("TM-align", tmalign_path))
        if not path.is_file()
    ]
    if missing:
        missing_str = ", ".join(f"{label}: {path}" for label, path in missing)
        raise FileNotFoundError(f"Missing input files -> {missing_str}. Run the similarity scripts first.")

    # Load actual results
    print("Loading CSV files...")
    df_tmalign = pd.read_csv(tmalign_path)
    df_tmvec1 = pd.read_csv(tmvec1_path).head(args.max_pairs)

    # Merge on sequence IDs
    # Clean seq IDs for matching (remove range info from tmvec1 IDs)
    df_tmvec1['seq1_clean'] = df_tmvec1['seq1_id'].str.replace(r'/\d+-\d+', '', regex=True)
    df_tmvec1['seq2_clean'] = df_tmvec1['seq2_id'].str.replace(r'/\d+-\d+', '', regex=True)

    df_merged = pd.merge(
        df_tmalign[['seq1_id', 'seq2_id', 'tm_score']],
        df_tmvec1[['seq1_clean', 'seq2_clean', 'tm_score']],
        left_on=['seq1_id', 'seq2_id'],
        right_on=['seq1_clean', 'seq2_clean'],
        suffixes=('_tmalign', '_tmvec1')
    )

    # Harmonize column names for plotting helpers
    df_merged = df_merged.rename(columns={
        'tm_score_tmalign': 'score_tmalign',
        'tm_score_tmvec1': 'score_tmvec1'
    })

    print(f"Merged {len(df_merged)} pairs\n")
    summary = (
        df_merged[['score_tmalign', 'score_tmvec1']]
        .describe()
        .rename(columns={
            'score_tmalign': 'tm_align',
            'score_tmvec1': 'tmvec1'
        })
    )
    print("Score summary (TM-align vs. TMvec-1):")
    print(summary.to_string(float_format=lambda x: f"{x:0.6f}"))
    print()

    # Create timestamped output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'figures/tmvec1_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_dir}/\n")

    # Plot 1: KDE
    print("1. KDE comparison")
    fig1 = plot_kde_comparison(df_merged, 'score_tmvec1', 'score_tmalign',
                               str(output_dir / 'kde_comparison'))

    # Plot 2: Correlation
    print("2. Correlation scatter")
    fig2, stats = plot_correlation_scatter(
        df_merged,
        'score_tmalign',
        'score_tmvec1',
        str(output_dir / 'correlation'),
    )

    # Confusion Matrix
    print("3. Confusion matrix (threshold=0.5)")
    conf_matrix = compute_confusion_matrix(
        df_merged,
        'score_tmalign',
        'score_tmvec1',
        threshold=0.5,
        output_path=str(output_dir / 'confusion_matrix'),
    )

    print(f"\nStats: R={stats['pearson_r']:.3f}, RMSE={stats['rmse']:.3f}, n={stats['n']}")
    print(f"Confusion Matrix (threshold=0.5):")
    print(f"  Accuracy: {conf_matrix.loc[conf_matrix['metric'] == 'accuracy', 'value'].iloc[0]:.3f}")
    print(f"  Precision: {conf_matrix.loc[conf_matrix['metric'] == 'precision', 'value'].iloc[0]:.3f}")
    print(f"  Recall: {conf_matrix.loc[conf_matrix['metric'] == 'recall', 'value'].iloc[0]:.3f}")
    print(f"  F1-Score: {conf_matrix.loc[conf_matrix['metric'] == 'f1_score', 'value'].iloc[0]:.3f}")
    print(f"\nDone. Check {output_dir}/ directory")


if __name__ == "__main__":
    main()
