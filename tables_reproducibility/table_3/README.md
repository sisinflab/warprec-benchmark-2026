# Table 3 — Green AI Profiling of WarpRec on NetflixPrize-100M

Reproduces Table 3 of the paper: per-model RAM/VRAM and energy/emissions profile
of **WarpRec** on **NetflixPrize-100M** for **ItemKNN**, **EASE$^R$**, **NeuMF**,
**LightGCN** and **SASRec**. Runtime numbers (`mean`/`peak` RAM/VRAM, energy,
emissions) come from CodeCarbon plus WarpRec's lifecycle logger; the
`Est. Peak RAM/VRAM` rows come from
[`warprec/warprec/memory_estimator.py`](../../warprec/warprec/memory_estimator.py),
which is a *static, pre-run* estimator — no training is required to obtain
those two rows.

## Layout

```
table_3/
├── README.md                          # this file
├── environment.yaml                   # conda env (table_3_reproducibility_warprec)
├── config/
│   ├── netflix-100m-ease.yml          # EASE^R       (HPO over l2)
│   ├── netflix-100m-itemknn.yml       # ItemKNN      (HPO over k)
│   ├── netflix-100m-lightgcn.yml      # LightGCN     (HPO over embedding_size, n_layers)
│   ├── netflix-100m-neumf.yml         # NeuMF        (HPO over mf_embedding_size, neg_samples)
│   └── netflix-100m-sasrec.yml        # SASRec       (HPO over embedding_size, n_layers)
├── warprec -> ../../warprec/warprec/  # symlink — makes `warprec` importable from here
└── experiments/                       # WarpRec writer outputs (created at run time)
```

The 5 YAMLs all read the same dataset
(`datasets/netflix-prize-100m/ratings_processed.csv`) and write to
`experiments/warprec-benchmark-2026/<dataset_name>/`. CodeCarbon is enabled per
config and writes its CSV into a `codecarbon/` subdirectory of the experiment
folder.

## Prerequisites

* NVIDIA GPU.
* Conda flavour: `micromamba`, `mamba` or `conda`.
* The dataset should be downloaded and saved as a CSV at
  `datasets/netflix-prize-100m/ratings_processed.csv`.

## Environment setup

One env is enough for everything in this folder:

```bash
micromamba env create -f environment.yaml      # creates table_3_reproducibility_warprec
micromamba activate table_3_reproducibility_warprec
```

All commands below assume the env is activated and that the working directory
is `tables_reproducibility/table_3/` — the local `warprec` symlink makes
`python -m warprec.<...>` resolve to the in-tree package.

## 1. Run WarpRec against the configs

Each config drives a full HPO sweep for one model. Pick the model and run the
`train` pipeline:

```bash
# ItemKNN  — 6 trials over k ∈ {50, 100, 200, 500, 1000, 2000}
python -m warprec.run -c config/netflix-100m-itemknn.yml  -p train

# EASE^R   — 6 trials over l2 ∈ {100, 200, 500, 750, 1000, 2000}
python -m warprec.run -c config/netflix-100m-ease.yml     -p train

# NeuMF    — 6 trials (mf_embedding_size × neg_samples)
python -m warprec.run -c config/netflix-100m-neumf.yml    -p train

# LightGCN — 6 trials (embedding_size × n_layers)
python -m warprec.run -c config/netflix-100m-lightgcn.yml -p train

# SASRec   — 6 trials (embedding_size × n_layers)
python -m warprec.run -c config/netflix-100m-sasrec.yml   -p train
```

After each run, look under
`experiments/warprec-benchmark-2026/Netflix100M-<Model>-Serial/` for the
trial-level artefacts:

* `codecarbon/emissions.csv`  — energy (kWh) and emissions (kg CO₂eq) per trial
* `lifecycle/`                — mean/peak RAM and VRAM samples per trial
* `ray_results/`              — per-trial result.json

These populate the `Emissions (c)`, `Energy Consumed (c)`, `Mean/Peak RAM (m)`
and `Mean/Peak VRAM (m)` rows of the table.

## 2. Run the estimator against the configs

[`warprec/warprec/memory_estimator.py`](../../warprec/warprec/memory_estimator.py)
is a stage-by-stage RAM/VRAM estimator that simulates the full driver + Ray
worker pipeline from the YAML alone — it does not train, load the dataset, or
require a GPU. It produces the `Est. Peak RAM Usage (e)` and
`Est. Peak VRAM Usage (e)` rows of the table.

Run it on any of the 5 configs:

```bash
python -m warprec.memory_estimator -c config/netflix-100m-itemknn.yml
python -m warprec.memory_estimator -c config/netflix-100m-ease.yml
python -m warprec.memory_estimator -c config/netflix-100m-neumf.yml
python -m warprec.memory_estimator -c config/netflix-100m-lightgcn.yml
python -m warprec.memory_estimator -c config/netflix-100m-sasrec.yml
```
