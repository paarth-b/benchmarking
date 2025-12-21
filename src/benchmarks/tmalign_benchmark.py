#!/usr/bin/env python
"""TMalign benchmark for CATH and SCOPe."""

import subprocess
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_fasta(fasta_path):
    """Extract IDs from FASTA file."""
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                # Handle both formats: >cath|4_4_0|107lA00 and >d12asa_
                parts = line.strip()[1:].split('|')
                ids.append(parts[-1].split('/')[0])
    return ids


def load_structures(domain_ids, pdb_dir, has_extension=True):
    """Load PDB structures from directory."""
    pdb_dir = Path(pdb_dir)
    ext = ".pdb" if has_extension else ""
    return {did: pdb_dir / f"{did}{ext}" for did in domain_ids if (pdb_dir / f"{did}{ext}").exists()}


def run_tmalign(pdb1, pdb2, binary):
    """Run TMalign and return TM-score."""
    try:
        result = subprocess.run([binary, str(pdb1), str(pdb2), "-a", "T"],
                                capture_output=True, text=True, timeout=60)
        for line in result.stdout.split('\n'):
            if line.startswith('TM-score=') and 'average length' in line:
                return float(line.split()[1])
    except:
        pass
    return None


def calculate_scores(structures, binary):
    """Calculate pairwise TM-scores."""
    ids = list(structures.keys())
    pairs = []
    total = len(ids) * (len(ids) - 1) // 2

    with tqdm(total=total, desc="TMalign") as pbar:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                score = run_tmalign(structures[ids[i]], structures[ids[j]], binary)
                if score:
                    pairs.append({'seq1_id': ids[i], 'seq2_id': ids[j], 'tm_score': score})
                pbar.update(1)

    return pairs


def main():
    """Run TMalign benchmark."""
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"

    if is_scope40:
        fasta = "data/fasta/scope40-2500.fa"
        pdb_dir = "data/scope40pdb/pdb"
        output = "results/scope40_tmalign_similarities.csv"
        has_ext = False
    else:
        fasta = "data/fasta/cath-domain-seqs-S100-1k.fa"
        pdb_dir = "data/pdb/cath-s100"
        output = "results/tmalign_similarities.csv"
        has_ext = True

    binary = "binaries/TMalign"

    print(f"Parsing {fasta}...")
    ids = parse_fasta(fasta)
    print(f"Found {len(ids)} sequences")

    print(f"Loading structures from {pdb_dir}...")
    structures = load_structures(ids, pdb_dir, has_ext)
    print(f"Loaded {len(structures)}/{len(ids)} structures")

    if not structures:
        print("Error: No structures found!")
        return

    print("Running TMalign...")
    pairs = calculate_scores(structures, binary)

    if not pairs:
        print("Error: No scores computed!")
        return

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(pairs)
    df.to_csv(output, index=False)

    print(f"\nSaved {len(pairs):,} scores to {output}")
    print(f"Mean: {df['tm_score'].mean():.4f}, Std: {df['tm_score'].std():.4f}")


if __name__ == "__main__":
    main()
