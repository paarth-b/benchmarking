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


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['savefig.dpi'] = 300


def plot_kde_comparison(df, pred_col, truth_col, method_name, output_path=None):
    """
    Create KDE plot comparing predicted scores to ground truth.
    """
    plot_df = pd.DataFrame({
        'TM-score': pd.concat([df[truth_col], df[pred_col]]),
        'Method': ['Ground Truth'] * len(df) + ['Predicted'] * len(df)
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=plot_df, x='TM-score', hue='Method', fill=True, alpha=0.5, ax=ax)

    ax.set_xlabel('TM-score')
    ax.set_ylabel('Density')
    ax.set_title(f'{method_name} vs TM-align')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_path}.pdf', bbox_inches='tight')

    return fig


def plot_density_contour(df, x_col, y_col, method_name, output_path=None):
    """
    Create density hexbin plot with correlation statistics.
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

    fig, ax = plt.subplots(figsize=(6, 6))

    hexbin = ax.hexbin(
        plot_df[x_col],
        plot_df[y_col],
        gridsize=50,
        cmap='rocket',
        mincnt=1,
        edgecolors='none'
    )

    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=1, zorder=10)

    ax.text(0.75, 0.05, f'r = {pearson_r:.3f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='bottom')

    ax.set_xlabel('TM-Score')
    ax.set_ylabel(f'Predicted TM-Score')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('count', rotation=270, labelpad=15)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_path}.pdf', bbox_inches='tight')

    return fig, statistics


def plot_confusion_matrix(df, truth_col, pred_col, threshold=0.5, output_path=None):
    """
    Create confusion matrix heatmap for binary classification.
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

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (threshold={threshold})')
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
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x=x_col,
        y=runtime_col,
        hue=method_col,
        marker='o',
        ax=ax
    )

    ax.set_yscale('log')
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel('Runtime (seconds, log scale)')
    ax.set_title('Runtime Comparison')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{output_path}.pdf', bbox_inches='tight')

    return fig
