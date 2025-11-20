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


def run_foldseek_all_vs_all_search(structure_dir, output_prefix, foldseek_bin, threads=32):
    """Run Foldseek all-vs-all search using easy-search with exhaustive mode."""
    print("Running Foldseek all-vs-all search with exhaustive mode...")

    # Create temporary directory
    tmp_dir = tempfile.mkdtemp()

    try:
        # Use easy-search with exhaustive search and TM-score output directly
        # Set very permissive thresholds to get all pairs
        tsv_path = f"{output_prefix}.tsv"
        cmd_easy_search = [
            foldseek_bin, "easy-search",
            structure_dir, structure_dir,
            tsv_path, tmp_dir,
            "--exhaustive-search", "1",  # Skip prefilter, perform all-vs-all alignment
            "--format-output", "query,target,alntmscore,evalue",
            "--threads", str(threads),
            "--gpu", "1",  # Enable GPU acceleration
            "-e", "10",  # Default Foldseek E-value
            "-c", "0.0",  # No coverage threshold (default filters by coverage)
            "--max-seqs", "1000000"  # Very high limit
        ]

        print(f"Running easy-search: {' '.join(cmd_easy_search)}")
        result = subprocess.run(cmd_easy_search, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Foldseek easy-search failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError("Foldseek easy-search failed")

        print("Foldseek all-vs-all search completed")
        return tsv_path

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_foldseek_results(tsv_file, pdb_ids):
    """Parse Foldseek TSV results from easy-search and extract pairwise TM-scores."""
    print("Parsing Foldseek results...")

    # Read the TSV file (easy-search format: query, target, alntmscore, evalue)
    df = pd.read_csv(tsv_file, sep='\t', header=None,
                     names=['query', 'target', 'alntmscore', 'evalue'])

    print(f"Loaded {len(df)} alignments")

    # Create mapping from PDB basename to CATH ID
    basename_to_cath = {}
    for cath_id in pdb_ids:
        # Extract the PDB part: "cath|4_4_0|107lA00" -> "107lA00"
        pdb_part = cath_id.split('|')[-1]
        basename_to_cath[pdb_part] = cath_id

    # Filter for our PDB IDs and create pairwise results
    pairs = []
    seen_pairs = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query_path = row['query']
        target_path = row['target']
        tm_score = row['alntmscore']  # Use alignment-normalized TM-score
        evalue = row['evalue']

        # Extract basename from path (e.g., "/path/to/107lA00.pdb" -> "107lA00")
        q_basename = Path(query_path).stem
        t_basename = Path(target_path).stem

        # Map to CATH IDs
        if q_basename in basename_to_cath and t_basename in basename_to_cath:
            q_id = basename_to_cath[q_basename]
            t_id = basename_to_cath[t_basename]

            # Skip self-comparisons and duplicates
            if q_id != t_id:
                pair_key = tuple(sorted([q_id, t_id]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append({
                        'seq1_id': q_id,
                        'seq2_id': t_id,
                        'tm_score': tm_score,
                        'evalue': evalue
                    })

    print(f"Extracted {len(pairs)} unique pairwise comparisons")
    return pairs


def save_results(pairs, output_path):
    """Save pairwise TM-score results as CSV."""
    print(f"Saving results to {output_path}...")

    df = pd.DataFrame(pairs)

    # Format e-value column in .3g scientific notation
    df['evalue'] = df['evalue'].apply(lambda x: f'{x:.3g}')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"Saved {len(pairs):,} pairwise predictions")


def main():
    parser = argparse.ArgumentParser(description="Foldseek protein structure similarity benchmark")
    parser.add_argument("--structure-dir", required=True, help="Directory containing PDB files")
    parser.add_argument("--foldseek-bin", required=True, help="Path to foldseek binary")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads")
    parser.add_argument("--max-structures", type=int, help="Maximum number of structures to process")

    args = parser.parse_args()

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
        tsv_file = run_foldseek_all_vs_all_search(args.structure_dir, str(output_prefix), args.foldseek_bin, args.threads)

        # Parse results
        pairs = parse_foldseek_results(tsv_file, pdb_ids)

        # Save final results
        save_results(pairs, args.output)

    print("=" * 80)
    print("Foldseek Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
