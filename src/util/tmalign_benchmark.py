#!/usr/bin/env python
"""
TMalign benchmark for CATH S100-1k sequences.

Three-step process:
1. Load or download PDB structures (using pdb_downloader module)
2. Run TMalign on all pairwise combinations
3. Save TM-scores to CSV
"""

import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from pdb_downloader import parse_cath_ids, download_all_structures


def load_existing_structures(domain_ids, pdb_dir):
    """
    Load existing PDB structures from directory.

    Args:
        domain_ids: List of CATH domain IDs
        pdb_dir: Directory containing PDB files

    Returns:
        Dictionary mapping domain_id -> Path to PDB file
    """
    pdb_dir = Path(pdb_dir)
    structures = {}

    for domain_id in domain_ids:
        pdb_path = pdb_dir / f"{domain_id}.pdb"
        if pdb_path.exists():
            structures[domain_id] = pdb_path

    return structures


def run_tmalign(pdb1, pdb2, tmalign_binary):
    """
    Run TMalign and parse TM-score.

    Args:
        pdb1: Path to first PDB file
        pdb2: Path to second PDB file
        tmalign_binary: Path to TMalign executable

    Returns:
        TM-score (float) or None if failed
    """
    try:
        result = subprocess.run(
            [tmalign_binary, str(pdb1), str(pdb2)],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Parse TM-score from output
        # Look for line like: "TM-score= 0.12345 (if normalized by length of Chain_1)"
        for line in result.stdout.split('\n'):
            if line.startswith('TM-score=') and 'Chain_1' in line:
                tm_score = float(line.split()[1])
                return tm_score

        # If we didn't find the expected format, return None
        return None

    except subprocess.TimeoutExpired:
        print(f"Timeout running TMalign on {pdb1.name} vs {pdb2.name}")
        return None
    except Exception as e:
        print(f"Error running TMalign on {pdb1.name} vs {pdb2.name}: {e}")
        return None


def calculate_tmalign_scores(structures, tmalign_binary):
    """
    Calculate pairwise TM-scores using TMalign.

    Args:
        structures: Dictionary mapping domain_id -> Path to PDB file
        tmalign_binary: Path to TMalign executable

    Returns:
        List of dictionaries with seq1_id, seq2_id, and tm_score
    """
    print("Running TMalign on all pairs...")

    domain_ids = list(structures.keys())
    pairs = []
    failed_pairs = 0

    total_pairs = len(domain_ids) * (len(domain_ids) - 1) // 2

    with tqdm(total=total_pairs, desc="TMalign comparisons") as pbar:
        for i in range(len(domain_ids)):
            for j in range(i + 1, len(domain_ids)):
                domain1 = domain_ids[i]
                domain2 = domain_ids[j]

                tm_score = run_tmalign(
                    structures[domain1],
                    structures[domain2],
                    tmalign_binary
                )

                if tm_score is not None:
                    pairs.append({
                        'seq1_id': f"cath|4_4_0|{domain1}",
                        'seq2_id': f"cath|4_4_0|{domain2}",
                        'tm_score': tm_score
                    })
                else:
                    failed_pairs += 1

                pbar.update(1)

    print(f"Computed {len(pairs)} TM-scores ({failed_pairs} failed)")
    return pairs


def save_results(pairs, output_path):
    """
    Save TM-scores to CSV.

    Args:
        pairs: List of dictionaries with seq1_id, seq2_id, and tm_score
        output_path: Path to output CSV file
    """
    if not pairs:
        print("Warning: No TM-scores to save!")
        return

    df = pd.DataFrame(pairs)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} TM-scores to {output_path}")
    print(f"Mean TM-score: {df['tm_score'].mean():.4f}")
    print(f"Std TM-score:  {df['tm_score'].std():.4f}")


def main():
    """
    Main function to run TMalign benchmark.

    Steps:
    1. Parse CATH domain IDs from FASTA
    2. Load existing structures or download missing ones
    3. Run TMalign on all pairs
    4. Save results to CSV
    """
    # Configuration
    fasta_path = "data/fasta/cath-domain-seqs-S100-1k.fa"
    pdb_dir = "data/pdb/cath-s100-1k"
    pdb_cache = "data/pdb/_pdb_cache"
    tmalign_binary = "models/tmalign/TMalign"
    output_path = "results/tmalign_similarities.csv"

    print("=" * 60)
    print("TMalign Benchmark for CATH S100-1k")
    print("=" * 60)
    print()

    # Parse domain IDs
    print("Step 1: Parsing FASTA file...")
    domain_ids = parse_cath_ids(fasta_path)
    print()

    # Load or download structures
    print("Step 2: Loading PDB structures...")
    structures = load_existing_structures(domain_ids, pdb_dir)
    print(f"Found {len(structures)}/{len(domain_ids)} existing structures")

    # Download missing structures if needed
    missing = set(domain_ids) - set(structures.keys())
    if missing:
        print(f"\nDownloading {len(missing)} missing structures...")
        new_structures = download_all_structures(
            list(missing),
            output_dir=pdb_dir,
            pdb_dir=pdb_cache,
            overwrite=False
        )
        structures.update(new_structures)

    if len(structures) < len(domain_ids):
        print(f"\nWarning: Only {len(structures)}/{len(domain_ids)} structures available")
        print(f"Missing {len(domain_ids) - len(structures)} structures")
        missing_ids = set(domain_ids) - set(structures.keys())
        if missing_ids:
            print(f"First few missing: {list(missing_ids)[:5]}")

    if len(structures) == 0:
        print("\nError: No structures available! Cannot proceed.")
        return

    print()

    # Run TMalign
    print("Step 3: Running TMalign on all pairs...")
    pairs = calculate_tmalign_scores(structures, tmalign_binary)
    print()

    if not pairs:
        print("Error: No TM-scores computed! Check TMalign binary and PDB files.")
        return

    # Save results
    print("Step 4: Saving results...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results(pairs, output_path)
    print()

    print("=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
