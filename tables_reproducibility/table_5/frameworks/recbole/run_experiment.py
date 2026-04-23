import os
import csv
import numpy as np
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from recbole.utils.case_study import full_sort_topk


# ── Write custom splits into the dataset dir in RecBole inter format ────────
def prepare_benchmark_splits():
    """Copy splits/train.tsv and splits/test.tsv into dataset/ml-1m/ as
    ml-1m.train.inter and ml-1m.test.inter, prepending the RecBole header.
    RecBole requires a third split for validation; we reuse the test file
    (valid_data is passed as None to trainer.fit so it is never actually used).
    """
    header = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
    os.makedirs(os.path.join('dataset', 'ml-1m'), exist_ok=True)
    for split in ['train', 'test']:
        src = os.path.join('splits', f'{split}.tsv')
        dst = os.path.join('dataset', 'ml-1m', f'ml-1m.{split}.inter')
        with open(src, 'r') as f_in, open(dst, 'w') as f_out:
            f_out.write(header)
            f_out.write(f_in.read())


# ── Shared data config (same for both models) ──────────────────────────────
BASE_CONFIG = {
    'data_path': 'dataset/',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'TIME_FIELD': 'timestamp',
    'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
    # Use pre-split files; RecBole expects [train, valid, test] so 'test' is
    # listed twice — the middle entry acts as a dummy valid split.
    'benchmark_filename': ['train', 'test', 'test'],
    'eval_args': {
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full',
    },
    'metrics': ['NDCG', 'Precision', 'Recall', 'MRR', 'MAP',
                'GiniIndex', 'ShannonEntropy'],
    'topk': [10],
    'valid_metric': 'NDCG@10',
    'metric_decimal_place': 10,
    'seed': 2020,
    'reproducibility': True,
    'show_progress': True,
    'save_dataset': False,
    'save_dataloaders': False,
    'checkpoint_dir': 'saved',
}

ITEMKNN_CONFIG = {
    **BASE_CONFIG,
    'epochs': 0,          # TraditionalTrainer uses 1 internally
    'k': 10,              # number of neighbours
    'shrink': 0.0,        # cosine similarity (normalize=True + shrink=0)
}

LIGHTGCN_CONFIG = {
    **BASE_CONFIG,
    'epochs': 1,
    'eval_step': 999999,
    'train_batch_size': 2048,
    'learning_rate': 0.001,
    'embedding_size': 16,
}


# ── Train + evaluate one model ──────────────────────────────────────────────
def run_model(model_name, config_dict):
    config = Config(model=model_name, dataset='ml-1m', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(model_name)(config, train_data._dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # Train (valid_data=None → early stopping disabled; saved=False → no checkpoint)
    trainer.fit(train_data, valid_data=None, saved=False,
                show_progress=config['show_progress'])

    # Evaluate on test set using current model state
    test_result = trainer.evaluate(test_data, load_best_model=False,
                                   show_progress=config['show_progress'])
    getLogger().info(f"[{model_name}] Test: {test_result}")

    # Top-10 recommendations for all users (internal ID 0 = padding, skip it)
    uid_series = np.arange(1, dataset.user_num)
    topk_scores, topk_iids = full_sort_topk(
        uid_series, model, test_data, k=10, device=config['device']
    )

    return test_result, uid_series, topk_scores, topk_iids, dataset


# ── Save TSV recommendations ────────────────────────────────────────────────
def save_recs_tsv(filepath, uid_series, topk_iids, topk_scores, dataset):
    external_uids  = dataset.id2token(dataset.uid_field, uid_series)
    external_items = dataset.id2token(dataset.iid_field, topk_iids.cpu().numpy())
    scores_np      = topk_scores.cpu().numpy()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        for u in range(len(uid_series)):
            for r in range(10):
                f.write(f"{external_uids[u]}\t{external_items[u, r]}\t{scores_np[u, r]}\n")


# ── Save metrics CSV ────────────────────────────────────────────────────────
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


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Write splits/train.tsv and splits/test.tsv into the dataset dir
    prepare_benchmark_splits()

    all_results = {}

    # ItemKNN
    knn_result, knn_uids, knn_scores, knn_iids, knn_dataset = run_model(
        'ItemKNN', ITEMKNN_CONFIG
    )
    all_results['ItemKNN'] = knn_result
    save_recs_tsv('results/ItemKNN_recs.tsv', knn_uids, knn_iids, knn_scores, knn_dataset)

    # LightGCN
    lgcn_result, lgcn_uids, lgcn_scores, lgcn_iids, lgcn_dataset = run_model(
        'LightGCN', LIGHTGCN_CONFIG
    )
    all_results['LightGCN'] = lgcn_result
    save_recs_tsv('results/LightGCN_recs.tsv', lgcn_uids, lgcn_iids, lgcn_scores, lgcn_dataset)

    # Metrics
    save_metrics_csv('results/metrics.csv', all_results)
    print("Done. Outputs in results/")
