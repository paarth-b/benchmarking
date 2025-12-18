#!/usr/bin/env python
"""
PDB structure downloader for CATH domains using BioPython.

Downloads PDB structures from the RCSB PDB and extracts specific chains
for CATH domain IDs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from Bio.PDB import PDBList, PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm


class ChainSelect(Select):
    """Select only specific chain(s) from a PDB structure."""

    def __init__(self, chain_id: str):
        self.chain_id = chain_id.upper()

    def accept_chain(self, chain):
        return chain.id == self.chain_id


def parse_cath_ids(fasta_path: str) -> List[str]:
    """
    Extract CATH domain IDs from FASTA file.

    Args:
        fasta_path: Path to CATH FASTA file

    Returns:
        List of CATH domain IDs (e.g., ['107lA00', '108lA00', ...])
    """
    domain_ids = []

    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Format: >cath|4_4_0|107lA00/1-162
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    domain_id = parts[2].split('/')[0]  # Get 107lA00
                    domain_ids.append(domain_id)

    print(f"Parsed {len(domain_ids)} CATH domain IDs")
    return domain_ids


def extract_pdb_code_and_chain(domain_id: str) -> Tuple[str, str]:
    """
    Extract PDB code and chain ID from CATH domain ID.

    CATH domain ID format: 107lA00
    - First 4 chars: PDB code (107l)
    - 5th char: Chain ID (A)
    - Last 2 chars: Domain number (00)

    Args:
        domain_id: CATH domain ID (e.g., '107lA00')

    Returns:
        Tuple of (pdb_code, chain_id) (e.g., ('107l', 'A'))
    """
    if len(domain_id) < 5:
        raise ValueError(f"Invalid CATH domain ID: {domain_id}")

    pdb_code = domain_id[:4].lower()
    chain_id = domain_id[4].upper()

    return pdb_code, chain_id


def download_pdb_structure(
    domain_id: str,
    output_dir: Path,
    pdb_dir: Optional[Path] = None,
    overwrite: bool = False
) -> Optional[Path]:
    """
    Download and extract PDB structure for a CATH domain.

    Uses BioPython's PDBList to download the full PDB file, then extracts
    the specific chain needed for the CATH domain.

    Args:
        domain_id: CATH domain ID (e.g., '107lA00')
        output_dir: Directory to save extracted PDB files
        pdb_dir: Directory for BioPython PDB downloads (temp storage)
        overwrite: Whether to re-download existing files

    Returns:
        Path to extracted PDB file, or None if download failed
    """
    output_path = output_dir / f"{domain_id}.pdb"

    # Skip if already exists
    if output_path.exists() and not overwrite:
        return output_path

    try:
        # Parse domain ID
        pdb_code, chain_id = extract_pdb_code_and_chain(domain_id)

        # Set up PDB download directory
        if pdb_dir is None:
            pdb_dir = output_dir / "_pdb_cache"
        pdb_dir.mkdir(parents=True, exist_ok=True)

        # Download PDB file using BioPython
        pdbl = PDBList(pdb=str(pdb_dir), verbose=False)

        # Suppress BioPython warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pdb_file = pdbl.retrieve_pdb_file(
                pdb_code,
                file_format='pdb',
                pdir=str(pdb_dir)
            )

        if not pdb_file or not Path(pdb_file).exists():
            print(f"Failed to download PDB {pdb_code}")
            return None

        # Parse the structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_code, pdb_file)

        # Check if chain exists
        chain_found = False
        for model in structure:
            if chain_id in [chain.id for chain in model]:
                chain_found = True
                break

        if not chain_found:
            print(f"Chain {chain_id} not found in {pdb_code}")
            return None

        # Extract specific chain
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_path), ChainSelect(chain_id))

        return output_path

    except Exception as e:
        print(f"Error processing {domain_id}: {e}")
        return None


def download_all_structures(
    domain_ids: List[str],
    output_dir: str,
    pdb_dir: Optional[str] = None,
    overwrite: bool = False
) -> Dict[str, Path]:
    """
    Download all CATH domain structures.

    Args:
        domain_ids: List of CATH domain IDs
        output_dir: Directory to save extracted PDB files
        pdb_dir: Directory for BioPython PDB downloads (temp storage)
        overwrite: Whether to re-download existing files

    Returns:
        Dictionary mapping domain_id -> Path to PDB file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pdb_dir:
        pdb_dir = Path(pdb_dir)

    print(f"Downloading structures to {output_dir}...")

    structures = {}
    failed = []

    for domain_id in tqdm(domain_ids, desc="Downloading PDB structures"):
        pdb_path = download_pdb_structure(
            domain_id,
            output_dir,
            pdb_dir=pdb_dir,
            overwrite=overwrite
        )

        if pdb_path:
            structures[domain_id] = pdb_path
        else:
            failed.append(domain_id)

    print(f"\nDownloaded {len(structures)}/{len(domain_ids)} structures")

    if failed:
        print(f"Failed to download {len(failed)} structures:")
        for domain_id in failed[:10]:  # Show first 10 failures
            print(f"  - {domain_id}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return structures


if __name__ == "__main__":
    import sys

    # Configuration (can be overridden by command line args)
    default_fasta = "data/fasta/cath-domain-seqs-S100-1k.fa"
    default_output = "data/pdb/cath-s100"
    default_cache = "data/pdb/_pdb_cache"

    # Parse command line args
    fasta_path = sys.argv[1] if len(sys.argv) > 1 else default_fasta
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output
    pdb_cache = sys.argv[3] if len(sys.argv) > 3 else default_cache

    print("=" * 60)
    print("CATH PDB Structure Downloader")
    print("=" * 60)
    print(f"FASTA file:   {fasta_path}")
    print(f"Output dir:   {output_dir}")
    print(f"PDB cache:    {pdb_cache}")
    print("=" * 60)
    print()

    # Parse domain IDs
    domain_ids = parse_cath_ids(fasta_path)

    # Download structures
    structures = download_all_structures(
        domain_ids,
        output_dir=output_dir,
        pdb_dir=pdb_cache,
        overwrite=False
    )

    # Summary
    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Total domains:      {len(domain_ids)}")
    print(f"Downloaded:         {len(structures)}")
    print(f"Failed:             {len(domain_ids) - len(structures)}")
    print(f"Success rate:       {len(structures)/len(domain_ids)*100:.1f}%")
    print("=" * 60)
