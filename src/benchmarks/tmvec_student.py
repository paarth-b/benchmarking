#!/usr/bin/env python
"""TM-Vec Student: TM-score predictions for CATH and SCOPe."""

import csv
import sys
from pathlib import Path
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model.student_model import StudentModel, encode_sequence


def load_fasta(fasta_path, max_sequences=None):
    """Load sequences from FASTA file."""
    seq_ids, sequences = [], []
    current_id, current_seq = None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id:
                    seq_ids.append(current_id)
                    sequences.append("".join(current_seq))
                    if max_sequences and len(seq_ids) >= max_sequences:
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id and (not max_sequences or len(seq_ids) < max_sequences):
            seq_ids.append(current_id)
            sequences.append("".join(current_seq))

    print(f"Loaded {len(seq_ids)} sequences")
    return seq_ids, sequences


def encode_sequences(sequences, max_length):
    """Tokenize sequences to tensor."""
    return torch.stack([encode_sequence(seq, max_length) for seq in sequences])


def load_model(checkpoint_path, device):
    """Load student model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint

    model = StudentModel()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


def compute_embeddings(model, tokens, batch_size, device):
    """Encode sequences to embeddings."""
    print("Encoding sequences...")
    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, tokens.size(0), batch_size), desc="Encoding"):
            end = min(tokens.size(0), start + batch_size)
            batch = tokens[start:end].to(device)
            embeddings.append(model.seq_encoder(batch).cpu())

    return torch.cat(embeddings, dim=0)


def predict_scores(model, embeddings, seq_ids, output_path, chunk_size, device):
    """Compute pairwise TM-scores and save to CSV."""
    n = embeddings.size(0)
    total_pairs = n * (n - 1) // 2

    print(f"Scoring {total_pairs:,} pairs...")

    predictor = model.tm_predictor.to(device)
    embeddings = embeddings.to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq1_id", "seq2_id", "tm_score"])

        with torch.no_grad():
            with tqdm(total=total_pairs, desc="Predicting") as pbar:
                for i in range(n - 1):
                    emb_i = embeddings[i:i + 1]
                    j = i + 1
                    while j < n:
                        end = min(n, j + chunk_size)
                        emb_j = embeddings[j:end]
                        count = end - j

                        batch_a = emb_i.expand(count, -1)
                        combined = torch.cat([batch_a, emb_j, batch_a * emb_j, torch.abs(batch_a - emb_j)], dim=1)

                        preds = predictor(combined).cpu()

                        rows = [(seq_ids[i], seq_ids[j + k], f"{float(preds[k]):.6f}") for k in range(count)]
                        writer.writerows(rows)

                        pbar.update(count)
                        j = end

    print(f"Saved to {output_path}")


def main():
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"

    if is_scope40:
        fasta = "data/fasta/scope40-2500.fa"
        output = "results/scope40_tmvec_student_similarities.csv"
        max_seq = 2500
    else:
        fasta = "data/fasta/cath-domain-seqs-S100-1k.fa"
        output = "results/tmvec_student_similarities.csv"
        max_seq = 1000

    checkpoint = "binaries/tmvec_student.pt"
    max_length = 600
    embed_batch = 128
    chunk_size = 4096
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"FASTA: {fasta}")
    print(f"Output: {output}")

    seq_ids, sequences = load_fasta(fasta, max_seq)
    tokens = encode_sequences(sequences, max_length)
    model = load_model(checkpoint, device)
    embeddings = compute_embeddings(model, tokens, embed_batch, device)

    del tokens

    predict_scores(model, embeddings, seq_ids, Path(output), chunk_size, device)


if __name__ == "__main__":
    main()
