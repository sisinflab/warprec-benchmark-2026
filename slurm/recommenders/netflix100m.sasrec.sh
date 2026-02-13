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
#SBATCH --job-name=nf100.sasrec

# Exit on error
set -e

# # Load the micromamba profile
source ~/.local/share/mamba/etc/profile.d/mamba.sh

# Load the environment variables
source ~/.env

# Go to the experiments directory
cd /path/to/warprec-benchmark-2026/recommenders

# Activate the conda environment
micromamba activate recommender-torch

# Start the app
python netflix100m_sasrec.py 

