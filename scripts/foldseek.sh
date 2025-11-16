#!/bin/bash

#SBATCH -A grp_qzhu44
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH --mem=64G
#SBATCH -q public
#SBATCH -o slurm_logs/slurm.%j.out
#SBATCH -e slurm_logs/slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE

set -euo pipefail
shopt -s nullglob

WORKDIR=/scratch/akeluska/prot_distill_divide/benchmarking
cd "${WORKDIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-32}"
echo "GPU: $(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo 'CPU only')"
echo "Start: $(date)"
echo ""

export HYDRA_FULL_ERROR=1
export HF_HOME=/scratch/akeluska/.cache/

module load mamba/latest
source activate tmvec_distill

FOLDSEEK_BIN=${WORKDIR}/binaries/foldseek
FASTA_FILE=/scratch/akeluska/prot_distill_divide/data/fasta/cath-domain-seqs-S100-1k.fa
STRUCTURE_DIR=/scratch/akeluska/prot_distill_divide/data/pdb/cath-s100-1k
OUTPUT_FILE=${WORKDIR}/results/foldseek_similarities.csv
THREADS=${SLURM_CPUS_PER_TASK:-32}

echo "Foldseek binary: ${FOLDSEEK_BIN}"
echo "FASTA: ${FASTA_FILE}"
echo "Structure dir: ${STRUCTURE_DIR}"
echo "Output: ${OUTPUT_FILE}"
echo ""

python -m src.util.foldseek_benchmark \
    --fasta "${FASTA_FILE}" \
    --structure-dir "${STRUCTURE_DIR}" \
    --foldseek-bin "${FOLDSEEK_BIN}" \
    --output "${OUTPUT_FILE}" \
    --threads "${THREADS}"

echo ""
echo "=========================================="
echo "Foldseek Benchmark Complete!"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  ${OUTPUT_FILE}"
