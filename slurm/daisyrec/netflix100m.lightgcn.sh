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
#SBATCH --job-name=nf100.lightgcn

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
python tune.py --optimization_metric=ndcg --hyperopt_trail=6 --algo_name=lightgcn --dataset=nf-100m-custom --prepro=origin --topk=10 --fold_num=1 --epochs=2 --test_size=0.1 --val_size=0.001 --cand_num=18000 --gpu=0 --init_method=default --optimizer=adam --early_stop --loss_type=BPR --test_method=rsbr --val_method=rsbr --sample_method=uniform --tune_pack='{"batch_size": [8192], "lr": [0.001], "factors": [64, 128, 256], "num_ng": [1], "node_dropout": [0], "mess_dropout": [0], "reg_1": [0.001], "reg_2": [0.001], "num_layers": [2, 3]}'
