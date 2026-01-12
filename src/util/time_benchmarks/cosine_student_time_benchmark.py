#!/usr/bin/env python
"""
Time benchmark for Cosine Student Model (BiLSTM encoder with cosine similarity).
Measures encoding times and query times for different sequence batch sizes.
"""

import gc
import pandas as pd
import torch
import time
from datetime import datetime
from pathlib import Path
from Bio import SeqIO

# Add src to path for imports
import sys
project_root = Path(__file__).resolve().parents[3]
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.append(str(src_root))

from src.model.student_cos_only_model import StudentModel, encode_sequence

# Set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load sequences
sequence_path = '/Users/aryank/Developer/benchmarking/data/fasta/cath-domain-seqs-S100-1k.fa'
record_ids = []
record_seqs = []

print(f"Loading sequences from {sequence_path}")
with open(sequence_path) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        record_ids.append(record.id)
        record_seqs.append(str(record.seq))

print(f"Loaded {len(record_seqs)} sequences")

# Load the cosine student model
checkpoint_path = "/Users/aryank/Developer/benchmarking/binaries/cosine_student_best_tmvec2_l1.pt"
print(f"Loading Cosine Student model from {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model = StudentModel()
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded ({total_params:,} parameters)")

# Max sequence length used during training
max_length = 600

# Time encoding for different batch sizes
start_time = time.time()
results_df = pd.DataFrame()
encode_df = pd.DataFrame()

print("\nBenchmarking encoding times...")

for encoding_size in [10, 100, 1000, 5000]:
    encoding_seqs = record_seqs[:encoding_size]

    # Encode sequences
    encode_start = time.time()
    encoded_tensors = []
    for seq in encoding_seqs:
        encoded = encode_sequence(seq, max_length)
        encoded_tensors.append(encoded)

    # Stack into batch
    batch_tensor = torch.stack(encoded_tensors, dim=0).to(device)

    # Get embeddings
    with torch.no_grad():
        embeddings = model.seq_encoder(batch_tensor)

    embeddings = embeddings.cpu().numpy()
    encode_seconds = time.time() - encode_start

    print(f"Encoding {encoding_size} sequences: {encode_seconds:.3f}s")

    encode_df = pd.concat([encode_df, pd.DataFrame([{
        "encoding_size": encoding_size,
        "encode_seconds": encode_seconds,
    }])], ignore_index=True)

# Create output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path("results/time_benchmarks") / f"cosine_student_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

# Save encoding times
encode_df.to_csv(output_dir / "encoding_times.csv", index=False)
print(f"Saved encoding times to {output_dir / 'encoding_times.csv'}")

# Now benchmark query times with different database sizes
print("\nBenchmarking query times...")

for database_size in [1000, 5000]:
    # Create database embeddings
    print(f"Building database with {database_size} sequences...")
    db_seqs = record_seqs[:database_size]

    db_encoded_tensors = []
    for seq in db_seqs:
        encoded = encode_sequence(seq, max_length)
        db_encoded_tensors.append(encoded)

    db_batch_tensor = torch.stack(db_encoded_tensors, dim=0).to(device)

    db_build_start = time.time()
    with torch.no_grad():
        db_embeddings = model.seq_encoder(db_batch_tensor)
    db_embeddings = db_embeddings.cpu()
    db_build_seconds = time.time() - db_build_start

    print(f"Database build time: {db_build_seconds:.3f}s")

    # Normalize for cosine similarity
    db_embeddings_norm = torch.nn.functional.normalize(db_embeddings, p=2, dim=1)

    for query_size in [10, 100, 1000]:
        print(f"Querying {query_size} sequences against {database_size} database...")

        # Create query embeddings
        query_seqs = record_seqs[:query_size]
        query_encoded_tensors = []
        for seq in query_seqs:
            encoded = encode_sequence(seq, max_length)
            query_encoded_tensors.append(encoded)

        query_batch_tensor = torch.stack(query_encoded_tensors, dim=0).to(device)

        encode_start = time.time()
        with torch.no_grad():
            query_embeddings = model.seq_encoder(query_batch_tensor)
        query_embeddings = query_embeddings.cpu()
        encode_seconds = time.time() - encode_start

        # Normalize query embeddings
        query_embeddings_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

        # Perform similarity search (cosine similarity)
        search_start = time.time()
        k = 10  # Return top 10 nearest neighbors

        # Compute all pairwise similarities
        similarities = torch.mm(query_embeddings_norm, db_embeddings_norm.t())
        topk_values, topk_indices = torch.topk(similarities, k, dim=1)

        search_seconds = time.time() - search_start

        total_seconds = encode_seconds + db_build_seconds + search_seconds

        print(
            f"Query {query_size} vs {database_size} database: "
            f"(encode {encode_seconds:.3f}s, db_build {db_build_seconds:.3f}s, search {search_seconds:.3f}s, total {total_seconds:.3f}s)"
        )

        results_df = pd.concat([results_df, pd.DataFrame([{
            "query_size": query_size,
            "database_size": database_size,
            "encode_seconds": encode_seconds,
            "db_build_seconds": db_build_seconds,
            "search_seconds": search_seconds,
            "total_seconds": total_seconds,
        }])], ignore_index=True)

# Save query times
results_df.to_csv(output_dir / "query_times.csv", index=False)
print(f"Saved query times to {output_dir / 'query_times.csv'}")

total_benchmark_time = time.time() - start_time
print(".2f")
print(f"Results saved in: {output_dir}")