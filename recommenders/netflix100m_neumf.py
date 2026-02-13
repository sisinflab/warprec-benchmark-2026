import sys
import os
import pandas as pd
import numpy as np

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
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

dataset_name = "Netflix100M"
model_name = "NeuMF"
dataset_path = "../dataset/netflix-prize-100m/ratings_processed.csv"

logger = LifecycleLogger(f"results/{dataset_name}-{model_name}.txt")

logger.start_experiment(model_name)

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '1m'

# Model parameters
EPOCHS = 2
BATCH_SIZE = 8192

SEED = DEFAULT_SEED  # Set None for non-deterministic results




# df = movielens.load_pandas_df(size=MOVIELENS_DATA_SIZE)


for num_neg in [0,1]:
    logger.start_preprocessing()
    df = pd.read_csv(dataset_path, sep=',')
    df.rename(columns={"user_id": "userID", "item_id": "itemID"}, inplace=True)

    df.head()

    train, test = python_stratified_split(df, ratio=0.9)
    train_file = f"datasets_dumps/train_{model_name}_{dataset_name}_neg{num_neg}.csv"
    test_file = f"datasets_dumps/test_{model_name}_{dataset_name}_neg{num_neg}.csv"
    
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    
    data = NCFDataset(
        train_file=train_file, 
        test_file=test_file,
        n_neg=num_neg,
        seed=SEED
    )

    logger.end_preprocessing()

    for n_factors in [64,128,256]:
        
        model = NCF (
            n_users=data.n_users, 
            n_items=data.n_items,
            model_type="NeuMF",
            n_factors=n_factors,
            layer_sizes=[64,32],
            n_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=1e-3,
            seed=SEED
        )

        with Timer() as train_time:
            logger.start_training()
            model.fit(data)
            logger.end_training()

        print("Took {} seconds for training.".format(train_time.interval))
        logger.start_evaluation()

        with Timer() as test_time:
            users, items, preds = [], [], []
            item = list(train.itemID.unique())
            for user in train.userID.unique():
                user = [user] * len(item) 
                users.extend(user)
                items.extend(item)
                preds.extend(list(model.predict(user, item, is_list=True)))

            all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

            merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
            all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

        print("Took {} seconds for prediction.".format(test_time))
        
        eval_map = map(test, all_predictions, col_prediction='prediction', k=TOP_K)
        eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
        eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
        eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

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