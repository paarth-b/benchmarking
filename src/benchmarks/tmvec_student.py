#!/usr/bin/env python
"""
TM-Vec Student: Generate pairwise TM-score predictions for large FASTA batches.

This script mirrors the TM-Vec v2 benchmarking pipeline but replaces the
transformer + Lobster stack with the distilled student model defined in
`train.py`. It performs three phases:

1. Load and encode up to N FASTA sequences with the same tokenizer used
   during training (BiLSTM encoder with learned embeddings).
2. Run the student encoder once to obtain latent representations.
3. Score every unique pair (i < j) with the calibrated TM predictor and
   stream the results to CSV to avoid holding ~O(N²) predictions in memory.

Example:
    python -m src.util.tmvec_student \
        --fasta data/fasta/cath-domain-seqs-S100.fa \
        --checkpoint /scratch/akeluska/prot_distill_divide/tmvec_student.pt \
        --output results/tmvec_student_similarities.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

# ------------------------------------------------------------------------------
# Import student model definitions from src/model
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model.student_model import StudentModel, encode_sequence  # type: ignore  # noqa: E402


# ------------------------------------------------------------------------------
# FASTA UTILITIES
# ------------------------------------------------------------------------------
def load_sequences(fasta_path: Path, max_sequences: int | None = None) -> Tuple[List[str], List[str]]:
    """Load sequence IDs and sequences from FASTA."""
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    seq_ids: List[str] = []
    sequences: List[str] = []
    current_id: str | None = None
    current_seq: List[str] = []

    with fasta_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    seq_ids.append(current_id)
                    sequences.append("".join(current_seq))
                    if max_sequences and len(seq_ids) >= max_sequences:
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None and (not max_sequences or len(seq_ids) < max_sequences):
            seq_ids.append(current_id)
            sequences.append("".join(current_seq))

    print(f"Loaded {len(seq_ids)} sequences from {fasta_path}")
    return seq_ids, sequences


def encode_sequences_to_tensor(sequences: Sequence[str], max_length: int) -> torch.Tensor:
    """Tokenize and pad sequences to a uniform tensor."""
    encoded = [encode_sequence(seq, max_length) for seq in sequences]
    return torch.stack(encoded, dim=0)


# ------------------------------------------------------------------------------
# MODEL / EMBEDDING GENERATION
# ------------------------------------------------------------------------------
def load_student_model(checkpoint_path: Path, device: torch.device) -> StudentModel:
    """Instantiate the student model and load weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading TM-Vec Student checkpoint: {checkpoint_path}")
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    except TypeError:
        # Older PyTorch versions do not support weights_only
        checkpoint = torch.load(checkpoint_path, **load_kwargs)

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
    if state_dict is None:
        state_dict = checkpoint

    model = StudentModel()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Student model loaded ({total_params:,} parameters)")
    return model


def compute_sequence_embeddings(
    model: StudentModel,
    token_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode sequences with the student encoder."""
    print("Encoding sequences with TM-Vec Student...")
    embeddings: List[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, token_tensor.size(0), batch_size), desc="Encoding", ncols=100):
            end = min(token_tensor.size(0), start + batch_size)
            batch = token_tensor[start:end].to(device, non_blocking=True)
            batch_emb = model.seq_encoder(batch)
            embeddings.append(batch_emb.detach())

    combined_embeddings = torch.cat(embeddings, dim=0)
    print(f"Encoded embeddings shape: {combined_embeddings.shape}")
    return combined_embeddings


# ------------------------------------------------------------------------------
# PAIRWISE PREDICTION / CSV WRITING
# ------------------------------------------------------------------------------
def predict_pairwise_tm_scores(
    model: StudentModel,
    embeddings: torch.Tensor,
    seq_ids: Sequence[str],
    output_path: Path,
    pair_chunk_size: int,
    buffer_rows: int,
    device: torch.device,
) -> None:
    """Score every unique pair and stream the predictions to CSV."""
    num_sequences = embeddings.size(0)
    if num_sequences < 2:
        print("Fewer than two sequences provided; skipping pairwise scoring.")
        return

    total_pairs = num_sequences * (num_sequences - 1) // 2
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = model.tm_predictor.to(device)
    predictor.eval()

    embeddings = embeddings.to(device)

    pair_sum = 0.0
    pair_sum_sq = 0.0
    min_score = 1.0
    max_score = 0.0

    print(f"Scoring {total_pairs:,} sequence pairs → {output_path}")

    rows_buffer: List[Tuple[str, str, str]] = []

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["seq1_id", "seq2_id", "tm_score"])

        with torch.no_grad():
            progress = tqdm(total=total_pairs, desc="Predicting TM-scores", ncols=100)
            for i in range(num_sequences - 1):
                emb_i = embeddings[i : i + 1]
                j = i + 1
                while j < num_sequences:
                    end = min(num_sequences, j + pair_chunk_size)
                    emb_j = embeddings[j:end]

                    repeat_count = end - j
                    batch_a = emb_i.expand(repeat_count, -1)
                    combined = torch.cat(
                        (
                            batch_a,
                            emb_j,
                            batch_a * emb_j,
                            torch.abs(batch_a - emb_j),
                        ),
                        dim=1,
                    )

                    preds = predictor(combined).detach().cpu()

                    pair_sum += float(preds.sum())
                    pair_sum_sq += float((preds ** 2).sum())
                    min_score = min(min_score, float(preds.min()))
                    max_score = max(max_score, float(preds.max()))

                    batch_rows = [
                        (seq_ids[i], seq_ids[j + offset], f"{float(score):.6f}")
                        for offset, score in enumerate(preds)
                    ]
                    rows_buffer.extend(batch_rows)

                    if len(rows_buffer) >= buffer_rows:
                        writer.writerows(rows_buffer)
                        rows_buffer.clear()

                    progress.update(repeat_count)
                    j = end

            progress.close()

        if rows_buffer:
            writer.writerows(rows_buffer)

    mean_score = pair_sum / total_pairs
    variance = max(pair_sum_sq / total_pairs - mean_score ** 2, 0.0)
    std_score = variance ** 0.5

    print("Pairwise TM-score statistics:")
    print(f"  Total pairs: {total_pairs:,}")
    print(f"  Mean / Std: {mean_score:.4f} / {std_score:.4f}")
    print(f"  Min / Max: {min_score:.4f} / {max_score:.4f}")
    print(f"✓ Results saved to {output_path}")


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TM-Vec Student pairwise inference")
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/fasta/cath-domain-seqs-S100.fa"),
        help="Input FASTA containing sequences to score",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/scratch/akeluska/prot_distill_divide/tmvec_student.pt"),
        help="Path to TM-Vec Student checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tmvec_student_similarities.csv"),
        help="Destination CSV for pairwise TM-score predictions",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=5000,
        help="Maximum number of sequences to load from FASTA",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=900,
        help="Maximum sequence length (tokens) used during training",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=128,
        help="Batch size while generating sequence embeddings",
    )
    parser.add_argument(
        "--pair-chunk-size",
        type=int,
        default=4096,
        help="Number of pairs to score at once (controls memory usage)",
    )
    parser.add_argument(
        "--buffer-rows",
        type=int,
        default=20000,
        help="Number of CSV rows to buffer before flushing to disk",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device (cuda or cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    requested_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but unavailable; falling back to CPU.")
        requested_device = torch.device("cpu")

    print("=" * 80)
    print("TM-Vec Student TM-score Prediction")
    print(f"Device: {requested_device}")
    print(f"FASTA: {args.fasta}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Max sequences: {args.max_sequences}, Max length: {args.max_length}")
    print("=" * 80)

    seq_ids, sequences = load_sequences(args.fasta, args.max_sequences)
    if len(seq_ids) < 2:
        raise ValueError("Need at least two sequences to compute pairwise TM-scores.")

    tokens = encode_sequences_to_tensor(sequences, args.max_length)
    model = load_student_model(args.checkpoint, requested_device)
    embeddings = compute_sequence_embeddings(model, tokens, args.embed_batch_size, requested_device)

    # Free sequence tokens to reclaim CPU memory
    del tokens

    predict_pairwise_tm_scores(
        model=model,
        embeddings=embeddings,
        seq_ids=seq_ids,
        output_path=args.output,
        pair_chunk_size=args.pair_chunk_size,
        buffer_rows=args.buffer_rows,
        device=requested_device,
    )


if __name__ == "__main__":
    main()

