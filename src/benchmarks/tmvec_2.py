#!/usr/bin/env python
"""
TMvec-2: Generate embeddings and compute TM-score predictions for CATH S100.
Uses Lobster embeddings as input.
"""

import argparse
import torch

from .embedding_generators import LobsterEmbeddingGenerator
from .tmvec_pipeline import run_tmvec_pipeline


def main():
    parser = argparse.ArgumentParser(description="TMvec-2 TM-score prediction")
    parser.add_argument("--fasta", default="data/fasta/cath-domain-seqs-S100-1k.fa", help="FASTA file path")
    parser.add_argument("--checkpoint", default="models/tmvec-2/last.ckpt", help="Model checkpoint path")
    parser.add_argument("--output", default="results/tmvec2_similarities.csv", help="Output CSV path")
    parser.add_argument("--max-sequences", type=int, default=1000, help="Maximum sequences to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--evalue-threshold", type=float, default=10, help="E-value threshold for filtering pairs")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    embedding_generator = LobsterEmbeddingGenerator()
    run_tmvec_pipeline(
        embedding_generator=embedding_generator,
        fasta_path=args.fasta,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_sequences=args.max_sequences,
        batch_size=args.batch_size,
        device=device,
        evalue_threshold=args.evalue_threshold
    )


if __name__ == "__main__":
    main()
