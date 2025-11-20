#!/bin/bash
#SBATCH --job-name=tm2-bench
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
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# CUSTOMIZE: Load your cluster's Python/PyTorch module or activate virtual environment
module load python/miniforge3_pytorch/2.7.0  # Replace with your module
# Alternatively: source /path/to/your/venv/bin/activate

# Configure PYTHONPATH (adjust LOBSTER_PATH if needed)
export LOBSTER_PATH="${LOBSTER_PATH:-$REPO_ROOT/../lobster}"
export PYTHONPATH="$LOBSTER_PATH/src:$REPO_ROOT:${PYTHONPATH:-}"

echo "Model: TM-Vec2 last.ckpt"
echo "FASTA: data/fasta/cath-domain-seqs-S100.fa (5000 sequences)"
echo "Output: results/tmvec2_similarities.csv"
echo ""

# Run TM-Vec2 predictions
echo "Running TM-Vec 2 predictions on CATH S100..."
echo ""

python -m src.benchmarks.tmvec_2

echo ""
echo "=========================================="
echo "TM-Vec 2 Predictions Complete!"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  results/tmvec2_similarities.csv"