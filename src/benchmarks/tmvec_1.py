#!/usr/bin/env python
"""
TMvec-1: Generate embeddings and compute TM-score predictions for CATH S100.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import T5EncoderModel, AutoTokenizer

try:
    # When executed as a package module (python -m src.benchmarks.tmvec_1)
    from ..model.tmvec_model import TMScorePredictor, TMVecConfig
    from .embedding_generators import ProtT5EmbeddingGenerator
    from .tmvec_pipeline import run_tmvec_pipeline
except ImportError:
    # Fallback for direct script execution so src/ stays importable
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from model.tmvec_model import TMScorePredictor, TMVecConfig
    from benchmarks.embedding_generators import ProtT5EmbeddingGenerator
    from benchmarks.tmvec_pipeline import run_tmvec_pipeline


def load_sequences(fasta_path, max_sequences=1000):
    """Load protein sequences from FASTA file."""
    sequences = []
    seq_ids = []
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_id is not None:
                    seq_ids.append(current_id)
                    sequences.append(''.join(current_seq))
                    if len(sequences) >= max_sequences:
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None and len(sequences) < max_sequences:
            seq_ids.append(current_id)
            sequences.append(''.join(current_seq))

    print(f"Loaded {len(sequences)} sequences")
    return seq_ids, sequences


def generate_plm_embeddings(sequences, model_name="Rostlab/prot_t5_xl_uniref50", batch_size=16, max_length=512, device='cuda'):
    """Generate ProtT5 embeddings for protein sequences."""
    print(f"Loading ProtT5 model: {model_name}")
    # Use the slow tokenizer to avoid tiktoken conversion issues with the ProtT5 SentencePiece model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    print("Generating ProtT5 embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]

            # Add spaces between amino acids for ProtT5
            batch_seqs_spaced = [" ".join(list(seq)) for seq in batch_seqs]

            encoded = tokenizer(
                batch_seqs_spaced,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeddings = outputs.last_hidden_state

            all_embeddings.append(embeddings.cpu())

    print(f"Generated ProtT5 embeddings: {all_embeddings[0].shape}")
    return all_embeddings


def generate_tmvec_embeddings(plm_embeddings, checkpoint_path, config_path, device='cuda'):
    """Transform ProtT5 embeddings into structure-aware embeddings."""
    print(f"Loading TMvec-1 config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = TMVecConfig(**config_dict)

    print(f"Loading TMvec-1 model from {checkpoint_path}")
    model = TMScorePredictor(config=config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print("Generating TMvec-1 embeddings...")
    all_tmvec_embeddings = []

    with torch.no_grad():
        for batch_embeddings in tqdm(plm_embeddings):
            batch_embeddings = batch_embeddings.to(device)
            batch_size, seq_len = batch_embeddings.shape[:2]

            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            tmvec_emb = model.encode_sequence(batch_embeddings, padding_mask)

            all_tmvec_embeddings.append(tmvec_emb.cpu().numpy())

    tmvec_embeddings = np.concatenate(all_tmvec_embeddings, axis=0)
    print(f"Generated TMvec-1 embeddings: {tmvec_embeddings.shape}")

    return tmvec_embeddings


def calculate_tm_scores(embeddings):
    """Calculate pairwise TM-scores via cosine similarity."""
    print("Calculating pairwise TM-scores...")

    embeddings_tensor = torch.from_numpy(embeddings)
    embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)

    tm_score_matrix = torch.mm(embeddings_norm, embeddings_norm.t()).numpy()

    print(f"Computed {len(embeddings)}x{len(embeddings)} TM-score matrix")
    print(f"Mean: {tm_score_matrix.mean():.4f}, Std: {tm_score_matrix.std():.4f}")

    return tm_score_matrix


def save_results(seq_ids, tm_score_matrix, output_path):
    """Save TM-score matrix as pairwise CSV."""
    print(f"Saving results to {output_path}...")

    pairs = []
    for i in range(len(seq_ids)):
        for j in range(i + 1, len(seq_ids)):
            pairs.append({
                'seq1_id': seq_ids[i],
                'seq2_id': seq_ids[j],
                'tm_score': tm_score_matrix[i, j]
            })

    df = pd.DataFrame(pairs)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} pairwise predictions")


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
        device=device,
        use_v1_model=True
    )


if __name__ == "__main__":
    main()
