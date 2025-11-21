#!/usr/bin/env python
"""
Generate ground truth matches between CATH S100 proteins.
"""

from itertools import combinations
import pandas as pd


def main():
    # CATH domain classifications were retrieved from:
    # https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list-S100.txt
    # Column definitions were adopted from:
    # https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/README-cath-list-file-format.txt
    levels = ['class', 'architecture', 'topology', 'superfamily']
    columns = levels + ['S35', 'S60', 'S95', 'S100', 'count', 'length', 'resolution']
    df = pd.read_csv('cath/cath-domain-list-S100.txt', sep='\s+', names=columns, index_col=0)

    for level in levels:
        df[level] = df[level].astype(str)
    df['architecture'] = df['class'].str.cat(df['architecture'], sep='.')
    df['topology'] = df['architecture'].str.cat(df['topology'], sep='.')
    df['superfamily'] = df['topology'].str.cat(df['superfamily'], sep='.')

    with open('domain.lst', 'r') as fh:
        domains = fh.read().splitlines()
    df = df.loc[domains]

    with open('truth.tsv', 'w') as fh:
        print('a', 'b', *levels, sep='\t', file=fh)
        for a, b in combinations(domains, 2):
            out = [a, b]
            for level in ('class', 'architecture', 'topology', 'superfamily'):
                out.append(str(int(df.loc[a, level] == df.loc[b, level])))
            print(*out, sep='\t', file=fh)


if __name__ == '__main__':
    main()
