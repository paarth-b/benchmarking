# TM-Vec Student: Structure Similarity Benchmarking

Benchmarking suite for evaluating other TM-score prediction methods on protein structures.

## Overview

This toolkit benchmarks three protein structure similarity methods against TM-Align scores:
- **Foldseek**: Fast structure comparison using 3Di sequences
- **TM-Vec**: Neural network model for TM-score prediction from sequence embeddings
- **TM-Vec Student**: BiLSTM Architecture distilling TM-Vec embeddings to predict from sequence


## Installation

1. Install dependencies using uv or pip:
```bash
uv sync
# or
pip install -r requirements.txt
```

```bash
# then, download the pdb files (by default, it downloads 1k pdb files from CATH) 
python 'src/util/pdb_downloader.py'
```

2. Place required binaries in `binaries/`:
   - `foldseek` - Download from [Foldseek repository](https://github.com/steineggerlab/foldseek)

3. Place model checkpoints in `models/`:
   - TM-Vec model files (e.g., `tm_vec_cath.ckpt`)

## Running Benchmarks

Run benchmarks directly using Python modules:

**Foldseek Benchmark:**
```bash
bash scripts/foldseek.sh
```

**TMalign Benchmark:**
```bash
bash scripts/tmalign.sh
```

**TM-Vec Benchmark:**
```bash
bash scripts/tmvec1.sh
```

**TM-Vec Student Model:**
```bash
bash scripts/tmvec-student.sh
```

## Input Data

### FASTA Files
Protein sequences in FASTA format (e.g., `data/fasta/cath-domain-seqs-S100-1k.fa`):
```
>cath|4_4_0|107lA00
MDPSTPPGVPPGETVSGGDNFTVKKLRKEGWVS...
>cath|4_4_0|108lA00
MKLLPLTALLLLGTVALVAAEAAPLKDVEQSSSQ...
```

### PDB Structures
PDB files in `data/pdb/cath-s100/` directory. The benchmark can automatically download structures using the PDB downloader utility.

## Output Files

All benchmarks generate CSV files in `results/` with pairwise similarity scores:

| query_id | target_id | tm_score |
|----------|-----------|----------|
| cath\|4_4_0\|107lA00 | cath\|4_4_0\|108lA00 | 0.8523 |
| cath\|4_4_0\|107lA00 | cath\|4_4_0\|109lA00 | 0.7234 |

Visualization plots are saved to `figures/`.
