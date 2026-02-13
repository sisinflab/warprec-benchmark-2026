# WarpRec Reproducibility Package - SIGIR 2026 Resources Track

This repository contains the reproducibility package for the experiments submitted to SIGIR 2026 as a resources track paper. The package includes benchmark scripts for multiple recommendation frameworks across different datasets.

## Overview

The repository includes benchmark implementations for the following frameworks:
- **Cornac**
- **DaisyRec-v2.0**
- **Elliot**
- **RecBole**
- **Recommenders**
- **WarpRec**

Each framework has been configured with isolated conda/micromamba environments to ensure reproducibility.

## Repository Structure

```
warprec-benchmark-2026/
├── cornac/           # Cornac framework with models and scripts
├── DaisyRec-v2.0/    # DaisyRec framework
├── elliot/           # Elliot framework
├── RecBole/          # RecBole framework
├── recommenders/     # Microsoft Recommenders
├── warprec/          # WarpRec framework
├── slurm/            # SLURM job scripts
│   ├── cornac/       # SLURM scripts for Cornac experiments
│   ├── daisyrec/     # SLURM scripts for DaisyRec experiments
│   ├── elliot/       # SLURM scripts for Elliot experiments
│   ├── recbole/      # SLURM scripts for RecBole experiments
│   ├── recommenders/ # SLURM scripts for Recommenders experiments
│   └── warprec/      # SLURM scripts for WarpRec experiments
└── utils/            # Utility scripts
```

## Prerequisites

Before running any experiments, ensure that:

1. **Conda/Micromamba** is installed and configured
2. **Environment files** for each framework are properly set up
3. All required **datasets** are downloaded and placed in the appropriate directories

## Environment Setup

Each framework requires its own isolated environment. Install them using the following pattern:

### Example: Installing Cornac Environment

```bash
cd cornac
micromamba env create -f environment.yml
```

### Example: Installing Elliot Environment

```bash
cd elliot
micromamba env create -f environment.yml
```

Repeat this process for all frameworks you intend to use. Key environments include:
- `cornac` - For Cornac experiments
- `daisyrec` - For DaisyRec experiments
- `elliot` / `elliot-tf` - For Elliot experiments (separate envs for PyTorch/TensorFlow models)
- `recbole` - For RecBole experiments
- `warprec` - For WarpRec experiments

## Running Experiments with SLURM

All experiments are configured to run via SLURM job submission scripts located in the `slurm/` directory.

### SLURM Script Structure

Each SLURM script follows this general pattern:

```bash
#!/bin/bash
#SBATCH -A AccountName
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time 1-00:00:00
#SBATCH --job-name=experiment_name

# Exit on error
set -e

# Load micromamba
source ~/.local/share/mamba/etc/profile.d/mamba.sh

# Navigate to framework directory
cd /path/to/framework

# Activate environment
micromamba activate <environment_name>

# Run experiment
python <experiment_script>.py
```

### Submitting Jobs

Navigate to the SLURM scripts directory and submit jobs using `sbatch`:

```bash
# Example: Submit a Cornac LightGCN experiment on MovieLens 1M
sbatch slurm/cornac/movielens1m.lightgcn.sh

# Example: Submit an Elliot EASE experiment on Netflix 100M
sbatch slurm/elliot/netflix100m.ease.sh

# Example: Submit a RecBole experiment
sbatch slurm/recbole/movielens1m.lightgcn.sh
```

### Available Experiments

Experiments are organized by:
- **Framework**: cornac, daisyrec, elliot, recbole, recommenders, warprec
- **Dataset**: movielens1m, movielens32m, netflix100m
- **Model**: lightgcn, neumf, ease, sasrec, itemknn.

Script naming convention: `<dataset>.<model>.sh`

Example scripts:
- `movielens1m.lightgcn.sh` - LightGCN on MovieLens 1M
- `movielens32m.neumf.sh` - NeuMF on MovieLens 32M
- `netflix100m.ease.sh` - EASE on Netflix 100M

### Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View job output:
```bash
# Output files are typically written to the working directory
tail -f slurm-<job_id>.out
```

Cancel a job:
```bash
scancel <job_id>
```

## Dataset Configuration

Ensure datasets are properly configured before running experiments. Each framework expects datasets in specific locations and formats. Refer to individual framework documentation for dataset preparation instructions.

Common datasets used:
- **MovieLens 1M** - ~1 million ratings
- **MovieLens 32M** - ~32 million ratings  
- **Netflix 100M** - ~100 million ratings

## Resource Requirements

Different experiments have varying resource requirements:

| Dataset | Typical Memory | Typical GPU |
|---------|---------------|-------------|
| MovieLens 1M | 64GB | 1-2 GPU |
| MovieLens 32M | 128-256GB | 1-4 GPU |
| Netflix 100M | 192-256GB | 1-4 GPU |