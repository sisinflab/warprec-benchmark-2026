#!/bin/bash
#SBATCH -A AccountName
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time 1-00:00:00
#SBATCH --job-name=mv32.ease

# Exit on error
set -e

# Load the micromamba profile
source ~/.local/share/mamba/etc/profile.d/mamba.sh

# Load the environment variables
source ~/.env

# Go to the experiments directory
cd /path/to/warprec-benchmark-2026/DaisyRec-v2.0

# Activate the conda environment
micromamba activate daisyrec

# Start the app
python tune.py --optimization_metric=ndcg --hyperopt_trail=6 --algo_name=ease --dataset=ml-32m-custom --prepro=origin --topk=10 --fold_num=1 --epochs=1 --test_size=0.1 --val_size=0.0001 --cand_num=90000 --test_method=rsbr --val_method=rsbr --tune_pack='{"reg": [100, 200, 500, 750, 1000, 2000]}'
