"""Evaluate recommendation files produced by external frameworks using
RecBole's own evaluator — no retraining.

Expects `recs/<framework>/<model>.tsv` with columns: user_id, item_id, score
(no header). For each file we take the top-K items per user (by score desc),
map external tokens to RecBole internal IDs, and hand a DataStruct to the
Evaluator directly — bypassing the Trainer/Collector pipeline entirely.
"""

import os
import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct

from run_experiment import BASE_CONFIG, prepare_benchmark_splits


K = 10
FRAMEWORKS = ['elliot', 'warprec']
MODELS = ['ItemKNN', 'LightGCN']


# ── Load dataset once; reuse across all files ──────────────────────────────
def load_dataset_and_positives():
    """Return (config, dataset, uid_list, uid2pos).

    - uid_list: 1-D np.ndarray of internal user IDs that have test positives,
      in the canonical order we'll use for the evaluation tensors.
    - uid2pos: dict internal_uid -> set of internal item IDs (ground truth).
    """
    config = Config(model='ItemKNN', dataset='ml-1m', config_dict=BASE_CONFIG)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    _, _, test_data = data_preparation(config, dataset)

    raw = test_data.uid_list
    uid_list = raw.numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)

    uid2pos = {}
    for uid in uid_list:
        positives = test_data.uid2positive_item[int(uid)]
        if positives is None:
            uid2pos[int(uid)] = set()
        else:
            uid2pos[int(uid)] = set(positives.numpy().tolist())
    return config, dataset, uid_list, uid2pos


# ── Read a TSV and build the (n_users, K) tensor of internal item IDs ──────
def load_external_topk(tsv_path, dataset, uid_list, k):
    df = pd.read_csv(
        tsv_path, sep='\t', header=None,
        names=['user', 'item', 'score'],
        dtype={'user': str, 'item': str, 'score': float},
    )
    df = df.sort_values(['user', 'score'], ascending=[True, False], kind='mergesort')

    grouped = defaultdict(list)
    for user, item in zip(df['user'].to_numpy(), df['item'].to_numpy()):
        if len(grouped[user]) < k:
            grouped[user].append(item)

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    n_users = len(uid_list)
    out = np.zeros((n_users, k), dtype=np.int64)

    missing_users = 0
    for row_idx, uid_internal in enumerate(uid_list):
        user_token = str(dataset.id2token(uid_field, int(uid_internal)))
        items_ext = grouped.get(user_token, [])
        if not items_ext:
            missing_users += 1
            continue
        for col_idx, item_token in enumerate(items_ext[:k]):
            try:
                out[row_idx, col_idx] = dataset.token2id(iid_field, str(item_token))
            except ValueError:
                out[row_idx, col_idx] = 0  # unknown item → padding id

    if missing_users:
        print(f"  warning: {missing_users}/{n_users} users had no recs in {tsv_path}")
    return torch.tensor(out, dtype=torch.long)


# ── Build rec.topk: (n_users, k+1) = [hit_mask | pos_count] ────────────────
def build_topk_hits(rec_items, uid_list, uid2pos):
    rec_items_np = rec_items.numpy()
    n_users, k = rec_items_np.shape
    hits = np.zeros((n_users, k), dtype=np.int32)
    pos_counts = np.zeros((n_users, 1), dtype=np.int32)
    for row_idx, uid in enumerate(uid_list):
        positives = uid2pos.get(int(uid), set())
        pos_counts[row_idx, 0] = len(positives)
        if not positives:
            continue
        for col_idx in range(k):
            if int(rec_items_np[row_idx, col_idx]) in positives:
                hits[row_idx, col_idx] = 1
    return torch.tensor(np.concatenate([hits, pos_counts], axis=1), dtype=torch.int)


def evaluate_file(tsv_path, config, dataset, uid_list, uid2pos, k):
    rec_items = load_external_topk(tsv_path, dataset, uid_list, k)
    rec_topk = build_topk_hits(rec_items, uid_list, uid2pos)

    struct = DataStruct()
    struct.set('rec.items', rec_items)
    struct.set('rec.topk', rec_topk)
    struct.set('data.num_items', dataset.num(dataset.iid_field))

    evaluator = Evaluator(config)
    return evaluator.evaluate(struct)


# ── Save metrics CSV (same shape as run_experiment.py output) ──────────────
def save_metrics_csv(filepath, results_by_model):
    all_keys = []
    for result in results_by_model.values():
        for k in result:
            if k not in all_keys:
                all_keys.append(k)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model'] + all_keys)
        for model_name, result in results_by_model.items():
            writer.writerow([model_name] + [result.get(k, '') for k in all_keys])


def main():
    prepare_benchmark_splits()
    config, dataset, uid_list, uid2pos = load_dataset_and_positives()

    out_dir = 'results/external'
    os.makedirs(out_dir, exist_ok=True)

    for framework in FRAMEWORKS:
        results = {}
        for model in MODELS:
            tsv = os.path.join('recs', framework, f'{model}.tsv')
            if not os.path.exists(tsv):
                print(f"skip: {tsv} not found")
                continue
            print(f"Evaluating {tsv}")
            results[model] = evaluate_file(tsv, config, dataset, uid_list, uid2pos, K)
            print(f"  {dict(results[model])}")

        out_path = os.path.join(out_dir, f'{framework}_metrics.csv')
        save_metrics_csv(out_path, results)
        print(f"wrote {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
