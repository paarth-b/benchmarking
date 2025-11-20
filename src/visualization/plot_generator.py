#!/usr/bin/env python
"""
Unified benchmarking visualization tool.
Generates all plots for any benchmarking method.
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from .plots import (
    plot_kde_comparison,
    plot_density_contour,
    plot_confusion_matrix,
    plot_runtime_comparison
)


METHOD_CONFIG = {
    'tmvec1': {
        'name': 'TMvec-1',
        'results_file': 'results/tmvec1_similarities.csv',
        'clean_ids': True
    },
    'student': {
        'name': 'TMvec-Student',
        'results_file': 'results/tmvec_student_similarities.csv',
        'clean_ids': True
    },
    'foldseek': {
        'name': 'Foldseek',
        'results_file': 'results/foldseek_similarities.csv',
        'clean_ids': False
    }
}


def load_and_merge_data(method_key, truth_file='results/tmalign_similarities.csv', max_rows=100000):
    """
    Load prediction results and merge with ground truth.

    Args:
        method_key: Method identifier (tmvec1, tmvec2, student, foldseek)
        truth_file: Path to ground truth TM-align results
        max_rows: Maximum rows to load from prediction file

    Returns:
        Merged DataFrame with both prediction and ground truth scores
    """
    config = METHOD_CONFIG[method_key]

    print(f"Loading {config['name']} results...")
    df_truth = pd.read_csv(truth_file)
    df_pred = pd.read_csv(config['results_file']).head(max_rows)

    if config['clean_ids']:
        df_pred['seq1_clean'] = df_pred['seq1_id'].str.replace(r'/\d+-\d+', '', regex=True)
        df_pred['seq2_clean'] = df_pred['seq2_id'].str.replace(r'/\d+-\d+', '', regex=True)

        df_merged = pd.merge(
            df_truth[['seq1_id', 'seq2_id', 'tm_score']],
            df_pred[['seq1_clean', 'seq2_clean', 'tm_score']],
            left_on=['seq1_id', 'seq2_id'],
            right_on=['seq1_clean', 'seq2_clean'],
            suffixes=('_truth', '_pred')
        )
    else:
        df_merged = pd.merge(
            df_truth[['seq1_id', 'seq2_id', 'tm_score']],
            df_pred[['seq1_id', 'seq2_id', 'tm_score']],
            on=['seq1_id', 'seq2_id'],
            suffixes=('_truth', '_pred')
        )

    print(f"Merged {len(df_merged):,} pairs")
    return df_merged, config['name']


def generate_all_plots(method_key, output_dir=None, threshold=0.5):
    """
    Generate all benchmark plots for a given method.

    Args:
        method_key: Method identifier (tmvec1, tmvec2, student, foldseek)
        output_dir: Output directory (auto-generated if None)
        threshold: Classification threshold for confusion matrix
    """
    df_merged, method_name = load_and_merge_data(method_key)

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = 'tmvec_student' if method_key == 'student' else method_key
        output_dir = f'figures/{folder_name}_{timestamp}'

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_dir}/\n")

    print("1. KDE comparison")
    plot_kde_comparison(
        df_merged,
        'tm_score_pred',
        'tm_score_truth',
        method_name,
        f'{output_dir}/kde_comparison'
    )

    print("2. Density contour plot")
    fig, stats = plot_density_contour(
        df_merged,
        'tm_score_truth',
        'tm_score_pred',
        method_name,
        f'{output_dir}/density_contour'
    )

    print("3. Confusion matrix")
    fig_cm, conf_stats = plot_confusion_matrix(
        df_merged,
        'tm_score_truth',
        'tm_score_pred',
        threshold=threshold,
        output_path=f'{output_dir}/confusion_matrix'
    )

    print(f"\nStatistics:")
    print(f"  Pearson R: {stats['pearson_r']:.3f}")
    print(f"  RMSE: {stats['rmse']:.3f}")
    print(f"  MAE: {stats['mae']:.3f}")
    print(f"  n: {stats['n']:,}")

    print(f"\nConfusion Matrix (threshold={threshold}):")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        value = conf_stats.loc[conf_stats['metric'] == metric, 'value'].iloc[0]
        print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")

    print(f"\nDone! Check {output_dir}/ directory")


def tmvec1(output_dir=None, threshold=0.5):
    """Generate plots for TMvec-1."""
    generate_all_plots('tmvec1', output_dir, threshold)
    

def student(output_dir=None, threshold=0.5):
    """Generate plots for TMvec-Student."""
    generate_all_plots('student', output_dir, threshold)


def foldseek(output_dir=None, threshold=0.5):
    """Generate plots for Foldseek."""
    generate_all_plots('foldseek', output_dir, threshold)


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark visualization plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.visualization.plot_generator tmvec1
  python -m src.visualization.plot_generator foldseek --threshold 0.6
        """
    )
    parser.add_argument(
        'method',
        choices=['tmvec1', 'student', 'foldseek'],
        help='Benchmarking method to visualize'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (auto-generated if not specified)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for confusion matrix (default: 0.5)'
    )

    args = parser.parse_args()

    method_funcs = {
        'tmvec1': tmvec1,
        'student': student,
        'foldseek': foldseek
    }

    method_funcs[args.method](args.output_dir, threshold)


if __name__ == "__main__":
    main()
