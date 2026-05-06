#!/bin/bash
#SBATCH -A AccountName
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --gres=gpu:1
#SBATCH --mem=192GB
#SBATCH --time 1-00:00:00
#SBATCH --job-name=nf100.ease

# Exit on error
set -e

# Load the micromamba profile
source ~/.local/share/mamba/etc/profile.d/mamba.sh

# Load the environment variables
source ~/.env

# Go to the experiments directory
cd /path/to/warprec-benchmark-2026/warprec

# Activate the conda environment
micromamba activate warprec

# Set WANDB to offline mode
export WANDB_MODE=offline

# Set OpenBLAS to use the same number of threads as allocated by SLURM
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Start Ray cluster
ray start --head \
    --num-cpus=16 \
    --num-gpus=1

# Start the app
python -m warprec.run -p train -c config/serial/netflix-100m-ease.yml
