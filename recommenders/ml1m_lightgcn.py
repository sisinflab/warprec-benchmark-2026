import sys
import os
import pandas as pd
import numpy as np

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.utils.notebook_utils import store_metadata

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (sigir26-resources)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from lifecycle_logger.lifecycle_logger import LifecycleLogger

dataset_name = "Movielens1M"
model_name = "LightGCN"
dataset_path = "../dataset/movielens-1m/ratings.csv"

logger = LifecycleLogger(f"results/{dataset_name}-{model_name}.txt")

logger.start_experiment(model_name)

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '1m'

# Model parameters
EPOCHS = 10
BATCH_SIZE = 8192

SEED = DEFAULT_SEED  # Set None for non-deterministic results

yaml_file = "recommenders/models/deeprec/config/lightgcn.yaml"

logger.start_preprocessing()

# df = movielens.load_pandas_df(size=MOVIELENS_DATA_SIZE)
df = pd.read_csv(dataset_path, sep=',')
df.rename(columns={"user_id": "userID", "item_id": "itemID"}, inplace=True)

df.head()

train, test = python_stratified_split(df, ratio=0.9)

data = ImplicitCF(train=train, test=test, seed=SEED)

logger.end_preprocessing()

for n_layer in [2,3]:
    for embed_size in [64,128,256]:
        
        hparams = prepare_hparams(yaml_file,
                                n_layers=n_layer,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                learning_rate=0.001,
                                eval_epoch=1,
                                top_k=TOP_K,
                                embed_size=embed_size,
                                decay=0.001
                                )

        model = LightGCN(hparams, data, seed=SEED)
    
        with Timer() as train_time:
            logger.start_training()
            model.fit()
            logger.end_training()

        print("Took {} seconds for training.".format(train_time.interval))
        logger.start_evaluation()

        topk_scores = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

        topk_scores.head()

        eval_map = map(test, topk_scores, k=TOP_K)
        eval_ndcg = ndcg_at_k(test, topk_scores, k=TOP_K)
        eval_precision = precision_at_k(test, topk_scores, k=TOP_K)
        eval_recall = recall_at_k(test, topk_scores, k=TOP_K)

        logger.end_evaluation()

        table_str = (
            f"\n{'='*30}\n"
            f"{'METRIC':<15} | {'VALUE':<10}\n"
            f"{'-'*30}\n"
            f"{'MAP':<15} | {eval_map:.4f}\n"
            f"{'NDCG':<15} | {eval_ndcg:.4f}\n"
            f"{'Precision@K':<15} | {eval_precision:.4f}\n"
            f"{'Recall@K':<15} | {eval_recall:.4f}\n"
            f"{'='*30}"
        )

        logger._log(table_str)