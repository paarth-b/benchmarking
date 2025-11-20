#!/bin/bash

#SBATCH -A grp_qzhu44
#SBATCH -N 1            # number of nodes
#SBATCH -c 4            # number of cores
#SBATCH -t 1-00:00:00   # time in d-hh:mm:ss
#SBATCH -p public       # partition
#SBATCH -G a100:1
#SBATCH --mem=80G
#SBATCH -q public       # QOS
#SBATCH -o slurm_logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm_logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Change to the directory of our script
cd /scratch/akeluska/prot_distill_divide/benchmarking

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Set hydra's verbosity to full error
export HYDRA_FULL_ERROR=1

# Set HF_HOME to cache directory
export HF_HOME=/scratch/akeluska/.cache/

echo "Model: TM-Vec1 Swiss Model"
echo "Checkpoint: /scratch/akeluska/tm-bench/tmvec_1_models/tm_vec_swiss_model.ckpt"
echo "Config: /scratch/akeluska/tm-bench/tmvec_1_models/tm_vec_swiss_model_params.json"
echo "FASTA: data/fasta/cath-domain-seqs-S100-1k.fa (1000 sequences)"
echo "Output: results/tmvec1_similarities.csv"
echo ""

# Run TM-Vec1 predictions
echo "Running TM-Vec 1 predictions on CATH S100..."
echo ""

uv run python -m src.util.tmvec_1

echo ""
echo "=========================================="
echo "TM-Vec 1 Predictions Complete!"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  results/tmvec1_similarities.csv"

echo "=========================================="
echo "Running the graphing scripts"

uv run python -m src.util.graphs.graphs_tmvec1 \
    --tmvec1 results/tmvec1_similarities.csv \
    --tmalign results/tmalign_similarities.csv

echo "End: $(date)"
echo "=========================================="
