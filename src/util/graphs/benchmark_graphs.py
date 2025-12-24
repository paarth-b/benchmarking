#!/usr/bin/env python
"""
Unified benchmarking visualization utility.
Generates publication-quality plots comparing any method against TM-align ground truth.
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
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison plots against TM-align ground truth."
    )
    parser.add_argument(
        "--method-name",
        type=str,
        required=True,
        help="Display name of the method being benchmarked (e.g., 'TMvec-Student', 'Foldseek', 'TMvec-1').",
    )
    parser.add_argument(
        "--prediction-csv",
        type=str,
        required=True,
        help="Path to predictions CSV file (must have seq1_id, seq2_id, tm_score columns).",
    )
    parser.add_argument(
        "--truth-csv",
        type=str,
        default="results/tmalign_similarities.csv",
        help="Path to ground truth TM-align CSV (default: results/tmalign_similarities.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures (default: figures/<method>_<timestamp>).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100_000,
        help="Maximum number of prediction pairs to load (default: 100000).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="TM-score threshold for confusion matrix (default: 0.5).",
    )
    parser.add_argument(
        "--method-id",
        type=str,
        default=None,
        help="Short ID for the method used in filenames (default: derived from method-name).",
    )
    return parser.parse_args()


def plot_kde_comparison(df, pred_col, truth_col, method_name, output_path=None):
    """
    Create KDE plot comparing predicted scores to ground truth.

    Args:
        df: DataFrame with predicted and ground truth scores
        pred_col: Column name for predictions
        truth_col: Column name for ground truth
        method_name: Display name of the method
        output_path: Path to save figure (without extension), optional
    """
    # Reshape data for seaborn
    plot_df = pd.DataFrame({
        'TM-score': pd.concat([df[truth_col], df[pred_col]]),
        'Method': ['TM-align (Ground Truth)'] * len(df) + [f'{method_name} (Predicted)'] * len(df)
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn's kdeplot with hue
    sns.kdeplot(data=plot_df, x='TM-score', hue='Method', fill=True, alpha=0.5, ax=ax)

    # Add mean lines
    for method, col in [('TM-align (Ground Truth)', truth_col), (f'{method_name} (Predicted)', pred_col)]:
        mean_val = df[col].mean()
        ax.axvline(mean_val, linestyle='--', linewidth=2,
                   label=f'{method.split()[0]} mean: {mean_val:.3f}')

    ax.set_title(f'Distribution Comparison: {method_name} vs TM-align')
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_path}.pdf', bbox_inches='tight')

    return fig


def plot_correlation_scatter(df, x_col, y_col, method_name, output_path=None):
    """
    Create scatterplot with Pearson correlation analysis.

    Args:
        df: DataFrame with scores
        x_col: X-axis column (ground truth)
        y_col: Y-axis column (predictions)
        method_name: Display name of the method
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

    g.ax_joint.set_xlabel('TM-align Score (Ground Truth)')
    g.ax_joint.set_ylabel(f'{method_name} Score (Predicted)')
    g.fig.suptitle(f'Correlation: {method_name} vs Ground Truth', y=1.02)

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

    # Derive method ID if not provided
    method_id = args.method_id or args.method_name.lower().replace(' ', '_').replace('-', '_')

    # Validate input files
    pred_path = Path(args.prediction_csv)
    truth_path = Path(args.truth_csv)

    missing = [
        (args.method_name, pred_path),
        ("TM-align", truth_path)
    ]
    missing = [(label, path) for label, path in missing if not path.is_file()]

    if missing:
        missing_str = ", ".join(f"{label}: {path}" for label, path in missing)
        raise FileNotFoundError(
            f"Missing input files -> {missing_str}. Run the benchmark scripts first."
        )

    # Load data
    print("Loading CSV files...")
    df_truth = pd.read_csv(truth_path)

    # Convert tm_score to float (handles scientific notation)
    if 'tm_score' in df_truth.columns:
        df_truth['tm_score'] = pd.to_numeric(df_truth['tm_score'], errors='coerce')

    df_pred = pd.read_csv(pred_path).head(args.max_pairs)

    # Convert tm_score to float (handles scientific notation)
    if 'tm_score' in df_pred.columns:
        df_pred['tm_score'] = pd.to_numeric(df_pred['tm_score'], errors='coerce')

    # Clean sequence IDs for matching (remove range info)
    df_pred['seq1_clean'] = df_pred['seq1_id'].str.replace(r'/\d+-\d+', '', regex=True)
    df_pred['seq2_clean'] = df_pred['seq2_id'].str.replace(r'/\d+-\d+', '', regex=True)

    # Merge on sequence IDs
    df_merged = pd.merge(
        df_truth[['seq1_id', 'seq2_id', 'tm_score']],
        df_pred[['seq1_clean', 'seq2_clean', 'tm_score']],
        left_on=['seq1_id', 'seq2_id'],
        right_on=['seq1_clean', 'seq2_clean'],
        suffixes=('_truth', '_pred')
    )

    # Harmonize column names
    df_merged = df_merged.rename(columns={
        'tm_score_truth': 'score_truth',
        'tm_score_pred': 'score_pred'
    })

    print(f"Merged {len(df_merged)} pairs\n")

    # Print summary statistics
    summary = (
        df_merged[['score_truth', 'score_pred']]
        .describe()
        .rename(columns={
            'score_truth': 'tm_align',
            'score_pred': method_id
        })
    )
    print(f"Score summary (TM-align vs. {args.method_name}):")
    print(summary.to_string(float_format=lambda x: f"{x:0.6f}"))
    print()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'figures/{method_id}_{timestamp}')

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_dir}/\n")

    # Plot 1: KDE comparison
    print("1. KDE comparison")
    plot_kde_comparison(
        df_merged,
        'score_pred',
        'score_truth',
        args.method_name,
        str(output_dir / 'kde_comparison')
    )

    # Plot 2: Correlation scatter
    print("2. Correlation scatter")
    fig, stats_dict = plot_correlation_scatter(
        df_merged,
        'score_truth',
        'score_pred',
        args.method_name,
        str(output_dir / 'correlation')
    )

    # Plot 3: Confusion matrix
    print(f"3. Confusion matrix (threshold={args.threshold})")
    conf_matrix = compute_confusion_matrix(
        df_merged,
        'score_truth',
        'score_pred',
        threshold=args.threshold,
        output_path=str(output_dir / 'confusion_matrix')
    )

    # Print results
    print(f"\nStats: R={stats_dict['pearson_r']:.3f}, RMSE={stats_dict['rmse']:.3f}, n={stats_dict['n']}")
    print(f"Confusion Matrix (threshold={args.threshold}):")
    print(f"  Accuracy: {conf_matrix.loc[conf_matrix['metric'] == 'accuracy', 'value'].iloc[0]:.3f}")
    print(f"  Precision: {conf_matrix.loc[conf_matrix['metric'] == 'precision', 'value'].iloc[0]:.3f}")
    print(f"  Recall: {conf_matrix.loc[conf_matrix['metric'] == 'recall', 'value'].iloc[0]:.3f}")
    print(f"  F1-Score: {conf_matrix.loc[conf_matrix['metric'] == 'f1_score', 'value'].iloc[0]:.3f}")
    print(f"\nDone. Check {output_dir}/ directory")


if __name__ == "__main__":
    main()
