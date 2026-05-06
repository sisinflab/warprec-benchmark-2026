"""RecBole training + recommendations export for Table 5 reproducibility.

- Stages canonical splits from data/split/{train,test}.tsv into RecBole atomic
  files at results/recbole_runs/dataset/ml-1m/ml-1m.{train,test}.inter.
- Trains ItemKNN and LightGCN with the user-specified hyperparameters and
  exactly 1 epoch (LightGCN). ItemKNN is non-trainable (ModelType.TRADITIONAL).
- Writes top-10 recommendations per user to
  results/recbole_runs/{ItemKNN,LightGCN}_recs.tsv (raw token IDs, 3 cols).
- Writes the diagonal Train=RecBole / Eval=RecBole metrics to
  results/recbole_runs/metrics.csv.

Run from `tables_reproducibility/table_5/`:
    python frameworks/recbole/run_experiment.py
"""
import csv
import os
from logging import getLogger
from pathlib import Path

import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed
from recbole.utils.case_study import full_sort_topk

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_SPLIT_DIR = ROOT / "data" / "split"
RECBOLE_RUNS_DIR = ROOT / "results" / "recbole_runs"
DATASET_DIR = RECBOLE_RUNS_DIR / "dataset" / "ml-1m"


def prepare_benchmark_splits():
    """Stage data/split/{train,test}.tsv into RecBole's atomic-file layout.

    RecBole expects a header `user_id:token<TAB>item_id:token<TAB>rating:float
    <TAB>timestamp:float` and reads files named `<dataset>.{train,valid,test}.inter`.
    Per the user's setup we have no validation split, so the trainer is invoked
    with `valid_data=None`; we still need a placeholder `.test.inter` for the
    `benchmark_filename: ['train','test','test']` mechanism to work.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    header = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
    for split in ("train", "test"):
        src = DATA_SPLIT_DIR / f"{split}.tsv"
        if not src.is_file():
            raise SystemExit(
                f"[run_experiment] missing {src} — run WarpRec training first."
            )
        dst = DATASET_DIR / f"ml-1m.{split}.inter"
        with src.open("r") as f_in, dst.open("w") as f_out:
            f_out.write(header)
            f_out.write(f_in.read())


BASE_CONFIG = {
    "data_path": str(RECBOLE_RUNS_DIR / "dataset"),
    "checkpoint_dir": str(RECBOLE_RUNS_DIR / "saved"),
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "RATING_FIELD": "rating",
    "TIME_FIELD": "timestamp",
    "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
    "benchmark_filename": ["train", "test", "test"],
    "eval_args": {"group_by": "user", "order": "TO", "mode": "full"},
    "metrics": ["NDCG", "Precision", "Recall", "MRR", "MAP",
                "GiniIndex", "ShannonEntropy"],
    "topk": [10],
    "valid_metric": "NDCG@10",
    "metric_decimal_place": 10,
    "seed": 2020,
    "reproducibility": True,
    "show_progress": True,
    "save_dataset": False,
    "save_dataloaders": False,
}

ITEMKNN_CONFIG = {
    **BASE_CONFIG,
    "epochs": 0,
    "k": 10,
    "shrink": 0.0,
}

LIGHTGCN_CONFIG = {
    **BASE_CONFIG,
    "epochs": 1,
    "eval_step": 999999,
    "stopping_step": 999,
    "train_batch_size": 2048,
    "learning_rate": 0.001,
    "embedding_size": 16,
    "n_layers": 2,
    "reg_weight": 0.001,
}


def run_model(model_name, config_dict):
    config = Config(model=model_name, dataset="ml-1m", config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, _, test_data = data_preparation(config, dataset)

    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    trainer.fit(train_data, valid_data=None, saved=False,
                show_progress=config["show_progress"])

    test_result = trainer.evaluate(test_data, load_best_model=False,
                                   show_progress=config["show_progress"])
    getLogger().info(f"[{model_name}] Test: {test_result}")

    uid_series = np.arange(1, dataset.user_num)
    topk_scores, topk_iids = full_sort_topk(
        uid_series, model, test_data, k=10, device=config["device"]
    )
    return test_result, uid_series, topk_scores, topk_iids, dataset


def save_recs_tsv(filepath, uid_series, topk_iids, topk_scores, dataset):
    external_uids = dataset.id2token(dataset.uid_field, uid_series)
    external_items = dataset.id2token(dataset.iid_field, topk_iids.cpu().numpy())
    scores_np = topk_scores.cpu().numpy()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="") as f:
        for u in range(len(uid_series)):
            for r in range(10):
                f.write(f"{external_uids[u]}\t{external_items[u, r]}\t{scores_np[u, r]}\n")


def save_metrics_csv(filepath, results_by_model):
    all_keys = []
    for result in results_by_model.values():
        for k in result:
            if k not in all_keys:
                all_keys.append(k)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + all_keys)
        for model_name, result in results_by_model.items():
            writer.writerow([model_name] + [result.get(k, "") for k in all_keys])


def main():
    os.chdir(ROOT)
    RECBOLE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    prepare_benchmark_splits()

    results = {}
    for model_name, cfg in [("ItemKNN", ITEMKNN_CONFIG), ("LightGCN", LIGHTGCN_CONFIG)]:
        result, uids, scores, iids, ds = run_model(model_name, cfg)
        results[model_name] = result
        save_recs_tsv(RECBOLE_RUNS_DIR / f"{model_name}_recs.tsv", uids, iids, scores, ds)

    save_metrics_csv(RECBOLE_RUNS_DIR / "metrics.csv", results)
    print(f"Done. RecBole outputs in {RECBOLE_RUNS_DIR}")


if __name__ == "__main__":
    main()
