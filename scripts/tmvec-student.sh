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
#SBATCH -e slurm_logs/lurm.%j.err # file to save job's STDERR (%j = JobId)
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

#Load required software
module load mamba/latest

#Activate our enviornment
source activate tmvec_distill

STUDENT_CHECKPOINT=/scratch/akeluska/prot_distill_divide/tmvec_student.pt
FASTA_FILE=/scratch/akeluska/prot_distill_divide/data/fasta/cath-domain-seqs-S100-1k.fa
OUTPUT_FILE=/scratch/akeluska/prot_distill_divide/benchmarking/results/tmvec_student_similarities.csv

echo "Model: TM-Vec Student ${STUDENT_CHECKPOINT}"
echo "FASTA: ${FASTA_FILE} (5000 sequences)"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Run TM-Vec Student predictions
echo "Running TM-Vec Student predictions on CATH S100..."
echo ""

python -m src.util.tmvec_student \
    --fasta "${FASTA_FILE}" \
    --checkpoint "${STUDENT_CHECKPOINT}" \
    --output "${OUTPUT_FILE}" \

echo ""
echo "=========================================="
echo "TM-Vec Student Predictions Complete!"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  ${OUTPUT_FILE}"