#!/usr/bin/env python
"""
TMvec-2: Generate embeddings and compute TM-score predictions for CATH S100.
Uses Lobster embeddings as input.
"""

import torch

from .embedding_generators import LobsterEmbeddingGenerator
from .tmvec_pipeline import run_tmvec_pipeline


def main():
    fasta_path = "data/fasta/cath-domain-seqs-S100-1k.fa"
    checkpoint_path = "models/tmvec-2/last.ckpt"
    output_path = "results/tmvec2_similarities.csv"

    max_sequences = 1000
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    embedding_generator = LobsterEmbeddingGenerator()
    run_tmvec_pipeline(
        embedding_generator=embedding_generator,
        fasta_path=fasta_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        max_sequences=max_sequences,
        batch_size=batch_size,
        device=device
    )


if __name__ == "__main__":
    main()
