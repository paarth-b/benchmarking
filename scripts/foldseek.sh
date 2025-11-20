#!/bin/bash
#SBATCH --job-name=foldseek-bench
#SBATCH --partition=ghx4              # CUSTOMIZE: your partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72            # CUSTOMIZE: adjust as needed
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --account=beut-dtai-gh        # CUSTOMIZE: your account
#SBATCH --time=12:00:00               # CUSTOMIZE: adjust as needed
#SBATCH --output=logs/%j/%x.out
#SBATCH --error=logs/%j/%x.err
#SBATCH --exclusive

set -e

# Get the repository root directory (parent of scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs/$SLURM_JOB_ID

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

# CUSTOMIZE: Load your cluster's Python/PyTorch module or activate virtual environment
module load python/miniforge3_pytorch/2.7.0  # Replace with your module
# Alternatively: source /path/to/your/venv/bin/activate

FOLDSEEK_BIN=binaries/foldseek
STRUCTURE_DIR=data/pdb/cath-s100-1k
OUTPUT_FILE=results/foldseek_similarities.csv
THREADS=$SLURM_CPUS_PER_TASK

echo "Foldseek binary: $FOLDSEEK_BIN"
echo "Structure dir: $STRUCTURE_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

# Run Foldseek benchmark
echo "Running Foldseek benchmark on CATH S100-1k..."
echo ""

python -m src.benchmarks.foldseek_benchmark \
    --structure-dir "$STRUCTURE_DIR" \
    --foldseek-bin "$FOLDSEEK_BIN" \
    --output "$OUTPUT_FILE" \
    --threads "$THREADS"

echo ""
echo "=========================================="
echo "Foldseek Benchmark Complete!"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  results/foldseek_similarities.csv"
