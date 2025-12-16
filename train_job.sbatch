#!/bin/bash
#SBATCH --job-name=ethan_cifar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=queue1gpu
#SBATCH --time=6-09:59:59
#SBATCH --output=/home/ethan/BabyHypernetworks/logs/train_%j.out
#SBATCH --error=/home/ethan/BabyHypernetworks/logs/train_%j.err

set -euo pipefail

cd /home/ethan/BabyHypernetworks
mkdir -p logs

echo "Starting job ${SLURM_JOB_ID:-unknown} on $(hostname) at $(date)"

source /home/ethan/leo-train-template/.venv/bin/activate

# Launch training with the same resource profile as the interactive salloc command.
srun --cpu-bind=none python -u /home/ethan/BabyHypernetworks/train_vit.py

echo "Job completed at $(date)"

