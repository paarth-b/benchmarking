#!/usr/bin/env python
"""
TMvec-1: Generate embeddings and compute TM-score predictions for CATH S100.
"""

import sys
from pathlib import Path

import torch

try:
    # When executed as a package module (python -m src.benchmarks.tmvec_1)
    from .embedding_generators import ProtT5EmbeddingGenerator
    from .tmvec_pipeline import run_tmvec_pipeline
except ImportError:
    # Fallback for direct script execution so src/ stays importable
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from benchmarks.embedding_generators import ProtT5EmbeddingGenerator
    from benchmarks.tmvec_pipeline import run_tmvec_pipeline


def main():
    fasta_path = "data/fasta/cath-domain-seqs-S100-1k.fa"
    checkpoint_path = "models/tm_vec_cath_model.ckpt"
    output_path = "results/tmvec1_similarities.csv"

    max_sequences = 1000
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    embedding_generator = ProtT5EmbeddingGenerator()
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
