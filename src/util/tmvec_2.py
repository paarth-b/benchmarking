#!/usr/bin/env python
"""
TMvec-2: Generate embeddings and compute TM-score predictions for CATH S100.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from lobster.model import LobsterPMLM

from ..model.tmvec_model import TMScorePredictor


def load_sequences(fasta_path, max_sequences=5000):
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


def generate_lobster_embeddings(sequences, batch_size=32, max_length=512, device='cuda'):
    """Generate LOBSTER embeddings for protein sequences."""
    print("Generating LOBSTER embeddings...")
    model = LobsterPMLM("asalam91/lobster_24M")
    tokenizer = model.tokenizer
    model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]

            encoded = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            embeddings = outputs.hidden_states[-1]

            all_embeddings.append(embeddings.cpu())

    print(f"Generated LOBSTER embeddings: {all_embeddings[0].shape}")
    return all_embeddings


def generate_tmvec_embeddings(lobster_embeddings, model_path, device='cuda'):
    """Transform LOBSTER embeddings into structure-aware embeddings."""
    print("Loading TMvec-2 model...")
    model = TMScorePredictor.load_from_checkpoint(model_path, strict=False)
    model.to(device)
    model.eval()

    print("Generating TMvec-2 embeddings...")
    all_tmvec_embeddings = []

    with torch.no_grad():
        for batch_embeddings in tqdm(lobster_embeddings):
            batch_embeddings = batch_embeddings.to(device)
            batch_size, seq_len = batch_embeddings.shape[:2]

            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            tmvec_emb = model.encode_sequence(batch_embeddings, padding_mask)

            all_tmvec_embeddings.append(tmvec_emb.cpu().numpy())

    tmvec_embeddings = np.concatenate(all_tmvec_embeddings, axis=0)
    print(f"Generated TMvec-2 embeddings: {tmvec_embeddings.shape}")

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
    checkpoint_path = "models/tmvec-2/last.ckpt"
    output_path = "results/tmvec2_similarities.csv"

    max_sequences = 1000
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("TMvec-2 TM-Score Prediction")
    print(f"Device: {device}, Max sequences: {max_sequences}")
    print("=" * 80)

    seq_ids, sequences = load_sequences(fasta_path, max_sequences)
    lobster_embeddings = generate_lobster_embeddings(sequences, batch_size, device=device)
    tmvec_embeddings = generate_tmvec_embeddings(lobster_embeddings, checkpoint_path, device)
    tm_score_matrix = calculate_tm_scores(tmvec_embeddings)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results(seq_ids, tm_score_matrix, output_path)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
