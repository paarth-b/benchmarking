#!/usr/bin/env python
"""
Common pipeline for TMvec benchmarking.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.tmvec_1_model import TransformerEncoderModule, TransformerEncoderModuleConfig


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


def generate_tmvec_embeddings(base_embeddings, model_path, device='cuda'):
    """Transform base embeddings into structure-aware embeddings."""
    print("Loading TMvec model...")

    checkpoint = torch.load(model_path, map_location='cpu')
    config = TransformerEncoderModuleConfig(d_model=1024)  # ProtT5 embedding dimension
    model = TransformerEncoderModule(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    print("Generating TMvec embeddings...")
    all_tmvec_embeddings = []

    with torch.no_grad():
        for batch_embeddings in tqdm(base_embeddings):
            batch_embeddings = batch_embeddings.to(device)
            batch_size, seq_len = batch_embeddings.shape[:2]

            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            tmvec_emb = model(batch_embeddings, src_mask=None, src_key_padding_mask=padding_mask)

            all_tmvec_embeddings.append(tmvec_emb.cpu().numpy())

    tmvec_embeddings = np.concatenate(all_tmvec_embeddings, axis=0)
    print(f"Generated TMvec embeddings: {tmvec_embeddings.shape}")

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
    scores = []
    for i in range(len(seq_ids)):
        for j in range(i + 1, len(seq_ids)):
            score = tm_score_matrix[i, j]
            pairs.append({
                'seq1_id': seq_ids[i],
                'seq2_id': seq_ids[j],
                'tm_score': score
            })
            scores.append(score)

    df = pd.DataFrame(pairs)
    df.to_csv(output_path, index=False)

    # Print statistics
    total_pairs = len(pairs)
    scores_array = np.array(scores)
    mean_score = scores_array.mean()
    std_score = scores_array.std()
    min_score = scores_array.min()
    max_score = scores_array.max()

    print("Pairwise TM-score statistics:")
    print(f"  Total pairs: {total_pairs:,}")
    print(f"  Mean / Std: {mean_score:.4f} / {std_score:.4f}")
    print(f"  Min / Max: {min_score:.4f} / {max_score:.4f}")
    print(f"✓ Results saved to {output_path}")


def run_tmvec_pipeline(embedding_generator, fasta_path, checkpoint_path, output_path,
                       max_sequences=1000, batch_size=32, device='cuda'):
    """Run complete TMvec benchmarking pipeline."""
    print("=" * 80)
    print("TMvec TM-Score Prediction")
    print(f"Device: {device}, Max sequences: {max_sequences}")
    print("=" * 80)

    seq_ids, sequences = load_sequences(fasta_path, max_sequences)

    base_embeddings = embedding_generator.generate(sequences, batch_size, device=device)
    tmvec_embeddings = generate_tmvec_embeddings(base_embeddings, checkpoint_path, device)
    tm_score_matrix = calculate_tm_scores(tmvec_embeddings)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results(seq_ids, tm_score_matrix, output_path)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)
