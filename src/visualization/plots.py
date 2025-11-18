#!/usr/bin/env python
"""
Shared plotting utilities for benchmarking results.
Production-quality visualizations using seaborn.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


sns.set_theme(style="whitegrid", context="paper", palette="rocket")
plt.rcParams['savefig.dpi'] = 300


def plot_kde_comparison(df, pred_col, truth_col, method_name, output_path=None):
    """
    Create KDE plot comparing predicted scores to ground truth.

    Args:
        df: DataFrame with predicted and ground truth scores
        pred_col: Column name for predictions
        truth_col: Column name for ground truth
        method_name: Name of prediction method for labels
        output_path: Path to save figure (without extension), optional
    """
    plot_df = pd.DataFrame({
        'TM-score': pd.concat([df[truth_col], df[pred_col]]),
        'Method': ['TM-align (Ground Truth)'] * len(df) + [f'{method_name} (Predicted)'] * len(df)
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(data=plot_df, x='TM-score', hue='Method', fill=True, alpha=0.5, ax=ax)

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


def plot_density_contour(df, x_col, y_col, method_name, output_path=None):
    """
    Create density contour plot with Pearson correlation analysis.

    Args:
        df: DataFrame with scores
        x_col: X-axis column (ground truth)
        y_col: Y-axis column (predictions)
        method_name: Name of prediction method for labels
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

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.kdeplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        fill=True,
        cmap='rocket',
        levels=10,
        thresh=0.05,
        ax=ax
    )

    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        alpha=0.3,
        s=10,
        color='white',
        edgecolor='none',
        ax=ax
    )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'w--', alpha=0.8, linewidth=2, label='y=x')

    stats_text = (f'Pearson R = {pearson_r:.3f}\n'
                  f'p-value = {pearson_p:.2e}\n'
                  f'RMSE = {rmse:.3f}\n'
                  f'MAE = {mae:.3f}\n'
                  f'n = {len(plot_df):,}')

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('TM-align Score (Ground Truth)')
    ax.set_ylabel(f'{method_name} Score (Predicted)')
    ax.set_title(f'Correlation: {method_name} vs TM-align')
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_path}.pdf', bbox_inches='tight')

    return fig, statistics


def plot_confusion_matrix(df, truth_col, pred_col, threshold=0.5, output_path=None):
    """
    Create confusion matrix heatmap for binary classification.

    Args:
        df: DataFrame with predicted and ground truth scores
        truth_col: Column name for ground truth scores
        pred_col: Column name for predicted scores
        threshold: TM-score threshold for classification (default: 0.5)
        output_path: Path to save figure (without extension), optional

    Returns:
        Tuple of (figure, statistics DataFrame)
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

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='rocket_r',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (threshold={threshold})')

    metrics_text = (f'Accuracy: {accuracy:.3f}\n'
                   f'Precision: {precision:.3f}\n'
                   f'Recall: {recall:.3f}\n'
                   f'F1-Score: {f1:.3f}')

    ax.text(1.4, 0.5, metrics_text, transform=ax.transAxes,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    statistics = pd.DataFrame({
        'metric': ['threshold', 'true_positives', 'true_negatives', 'false_positives',
                  'false_negatives', 'precision', 'recall', 'f1_score', 'accuracy', 'total_samples'],
        'value': [threshold, tp, tn, fp, fn, precision, recall, f1, accuracy, len(plot_df)]
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_path}.pdf', bbox_inches='tight')
        statistics.to_csv(f'{output_path}.csv', index=False)

    return fig, statistics


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
