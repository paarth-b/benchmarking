#!/usr/bin/env python
"""
Generate a table with combined CATH S100 analysis results.
"""

import pandas as pd


def main():
    methods = ['tmvec1', 'tmvec_student', 'tmalign', 'foldseek']

    dfs = []
    for method in methods[:3]:
        dfs.append(pd.read_csv(f'data/{method}_similarities.csv').rename(columns={'tm_score': method}))

    # Foldseek reports both TM-scores and E-values. We will use E-values.
    dfs.append(pd.read_csv('data/foldseek_similarities.csv').drop(columns='tm_score').rename(columns={'evalue': 'foldseek'}))

    # TM-vec methods have a suffix like "/1-150" after seq_id. Remove it.
    for df in dfs[:2]:
        for i in (1, 2):
            df[f'seq{i}_id'] = df[f'seq{i}_id'].str.split('/').str[0]

    # Remove prefix "cath|4_4_0|".
    for df in dfs:
        for i in (1, 2):
            df[f'seq{i}_id'] = df[f'seq{i}_id'].str.split('|').str[2]

    # Sort and merge seq1 and seq2 IDs.
    for df in dfs:
        df['seq_ids'] = df.apply(lambda row: ','.join(sorted([row['seq1_id'], row['seq2_id']])), axis=1)
        df.drop(['seq1_id', 'seq2_id'], axis=1, inplace=True)
        df.set_index('seq_ids', inplace=True)

    # Combine results.
    df = pd.concat(dfs, axis=1)
    # df.dropna(how='any', inplace=True)

    # Append ground truth.
    truth = pd.read_table('truth.tsv')
    truth['seq_ids'] = truth['a'] + ',' + truth['b']
    truth.set_index('seq_ids', inplace=True)
    truth.drop(columns=['a', 'b'], inplace=True)
    df = pd.concat([truth, df], axis=1)

    # Output.
    df.to_csv('results.tsv', sep='\t')


if __name__ == '__main__':
    main()
