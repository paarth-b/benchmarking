#!/bin/bash
#SBATCH --job-name=tm2-bench
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --account=beut-dtai-gh
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j/%x.out
#SBATCH --error=logs/%j/%x.err
#SBATCH --exclusive

set -e

mkdir -p logs/$SLURM_JOB_ID
cd /u/paarthbatra/git/benchmarking

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Load override module from deltaAI
module load python/miniforge3_pytorch/2.7.0

# Configure PYTHONPATH
export PYTHONPATH="$(pwd)/../lobster/src:$(pwd):${PYTHONPATH:-}"

echo "Model: TM-Vec2 last.ckpt"
echo "FASTA: data/fasta/cath-domain-seqs-S100.fa (5000 sequences)"
echo "Output: results/tmvec2_similarities.csv"
echo ""

# Run TM-Vec2 predictions
echo "Running TM-Vec 2 predictions on CATH S100..."
echo ""

python -m src.util.tmvec_2

echo ""
echo "=========================================="
echo "TM-Vec 2 Predictions Complete!"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  results/tmvec2_similarities.csv"