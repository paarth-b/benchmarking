#!/usr/bin/env python
"""
Foldseek Benchmark: Generate pairwise TM-score predictions for protein structures.
"""

from pathlib import Path
import subprocess
import pandas as pd
import tempfile
import shutil
import argparse
from tqdm import tqdm
import os


def get_pdb_files(structure_dir, max_structures=None):
    """Get list of PDB files from structure directory."""
    pdb_files = list(Path(structure_dir).glob("*.pdb"))
    pdb_files.sort()

    if max_structures:
        pdb_files = pdb_files[:max_structures]

    print(f"Found {len(pdb_files)} PDB files")
    return pdb_files


def extract_pdb_ids(pdb_files):
    """Extract CATH IDs from file paths."""
    pdb_ids = []
    for pdb_file in pdb_files:
        # Extract ID from filename (e.g., "107lA00.pdb" -> "cath|4_4_0|107lA00")
        pdb_id = pdb_file.stem
        cath_id = f"cath|4_4_0|{pdb_id}"
        pdb_ids.append(cath_id)
    return pdb_ids


def run_foldseek_all_vs_all_search(structure_dir, output_prefix, threads=32):
    """Run Foldseek all-vs-all search using database approach."""
    print("Running Foldseek all-vs-all search...")

    # Create temporary directory
    tmp_dir = tempfile.mkdtemp()

    try:
        # Create database from structures
        db_path = Path(tmp_dir) / "structures_db"
        cmd_createdb = ["foldseek", "createdb", structure_dir, str(db_path)]
        print(f"Creating database: {' '.join(cmd_createdb)}")
        result = subprocess.run(cmd_createdb, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Database creation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError("Database creation failed")

        # Run all-vs-all search
        aln_path = Path(tmp_dir) / "alignments"
        cmd_search = [
            "foldseek", "search",
            str(db_path), str(db_path),
            str(aln_path), tmp_dir,
            "-a",  # include alignments
            "--threads", str(threads),
            "--max-seqs", "1000000",  # very high limit
            "-e", "10.0"  # very permissive E-value
        ]

        print(f"Running search: {' '.join(cmd_search)}")
        result = subprocess.run(cmd_search, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Foldseek search failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError("Foldseek search failed")

        # Convert to TSV
        tsv_path = f"{output_prefix}.tsv"
        cmd_tsv = [
            "foldseek", "createtsv",
            str(db_path), str(db_path), str(aln_path), tsv_path
        ]

        print(f"Converting to TSV: {' '.join(cmd_tsv)}")
        result = subprocess.run(cmd_tsv, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Foldseek createtsv failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError("Foldseek createtsv failed")

        print("Foldseek all-vs-all search completed")
        return tsv_path

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_foldseek_results(tsv_file, pdb_ids):
    """Parse Foldseek TSV results and extract pairwise TM-scores."""
    print("Parsing Foldseek results...")

    # Read the TSV file
    df = pd.read_csv(tsv_file, sep='\t', header=None,
                     names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen',
                           'qstart', 'qend', 'tstart', 'tend', 'evalue', 'bits', 'alntmscore'])

    print(f"Loaded {len(df)} alignments")

    # When using databases, query and target are indices (0, 1, 2, ...)
    # The order should match the order that createdb processed the files
    # Assuming pdb_ids is in the same order as the files were processed

    # Filter for our PDB IDs and create pairwise results
    pairs = []
    seen_pairs = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query_idx = int(row['query'])
        target_idx = int(row['target'])
        tm_score = row['alntmscore']

        # Map indices to CATH IDs
        if query_idx < len(pdb_ids) and target_idx < len(pdb_ids):
            q_id = pdb_ids[query_idx]
            t_id = pdb_ids[target_idx]

            # Skip self-comparisons and duplicates
            if q_id != t_id:
                pair_key = tuple(sorted([q_id, t_id]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append({
                        'seq1_id': q_id,
                        'seq2_id': t_id,
                        'tm_score': tm_score
                    })

    print(f"Extracted {len(pairs)} unique pairwise comparisons")
    return pairs


def save_results(pairs, output_path):
    """Save pairwise TM-score results as CSV."""
    print(f"Saving results to {output_path}...")

    df = pd.DataFrame(pairs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} pairwise predictions")


def main():
    parser = argparse.ArgumentParser(description="Foldseek protein structure similarity benchmark")
    parser.add_argument("--structure-dir", required=True, help="Directory containing PDB files")
    parser.add_argument("--foldseek-bin", required=True, help="Path to foldseek binary")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads")
    parser.add_argument("--max-structures", type=int, help="Maximum number of structures to process")

    args = parser.parse_args()

    # Set foldseek binary path
    os.environ['PATH'] = f"{Path(args.foldseek_bin).parent}:{os.environ.get('PATH', '')}"

    print("=" * 80)
    print("Foldseek Structure Similarity Benchmark")
    print(f"Structure dir: {args.structure_dir}")
    print(f"Output: {args.output}")
    print(f"Threads: {args.threads}")
    print(f"Max structures: {args.max_structures or 'all'}")
    print("=" * 80)

    # Get PDB files
    pdb_files = get_pdb_files(args.structure_dir, args.max_structures)
    if not pdb_files:
        raise ValueError(f"No PDB files found in {args.structure_dir}")

    pdb_ids = extract_pdb_ids(pdb_files)

    # Create temporary directory for results
    with tempfile.TemporaryDirectory() as tmp_base:
        output_prefix = Path(tmp_base) / "foldseek_results"

        # Run all-vs-all search using database approach
        tsv_file = run_foldseek_all_vs_all_search(args.structure_dir, str(output_prefix), args.threads)

        # Parse results
        pairs = parse_foldseek_results(tsv_file, pdb_ids)

        # Save final results
        save_results(pairs, args.output)

    print("=" * 80)
    print("Foldseek Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
