import os
import sys
import pandas as pd 
import torch
from tqdm import tqdm
import numpy as np
from recommenders.utils.timer import Timer
from recommenders.models.sasrec.model import SASREC
from recommenders.models.sasrec.util import SASRecDataSet
from recommenders.models.sasrec.sampler import WarpSampler
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

# --- CONFIGURAZIONE ---
dataset_path = "../dataset/movielens-1m/ratings.csv"
processed_data = "datasets_dumps/processed_SASREC_Movielens1M.txt"
TOP_K = 10
EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
MAX_SEQ_LEN = 20
dataset_name = "Movielens1M"
model_name = "SASRec"

logger = LifecycleLogger(f"results/{dataset_name}-{model_name}.txt")

logger.start_experiment(model_name)

logger.start_preprocessing()

# --- 1. PREPROCESSING ---
df = pd.read_csv(dataset_path, sep=',')
df.rename(columns={"user_id": "userID", "item_id": "itemID"}, inplace=True)

# Mapping ID
user_map = {user: i+1 for i, user in enumerate(df['userID'].unique())}
item_map = {item: i+1 for i, item in enumerate(df['itemID'].unique())}
df["userID"] = df["userID"].map(user_map)
df["itemID"] = df["itemID"].map(item_map)

# Ordinamento temporale e salvataggio
df = df.sort_values(by=["userID", "timestamp"])
df.drop(columns=["timestamp"], inplace=True)
os.makedirs("datasets_dumps", exist_ok=True)
df.to_csv(processed_data, sep="\t", header=False, index=False)

# --- 2. CARICAMENTO DATASET E SPLIT CUSTOM (90/10) ---
data = SASRecDataSet(filename=processed_data, col_sep="\t")
# Caricamento iniziale per popolare usernum e itemnum
data.split(test_size=1, min_interactions=3)

logger.end_preprocessing()



for n_layers in [2,3]:
    for embedding_dim in [64,128,256]:
    
        # --- 3. INIZIALIZZAZIONE MODELLO ---
        # Ora che data.itemnum è popolato, possiamo definire il modello
        model = SASREC(
            item_num=data.itemnum,
            seq_max_len=MAX_SEQ_LEN,
            num_blocks=n_layers,
            embedding_dim=embedding_dim,
            attention_dim=embedding_dim,
            attention_num_heads=8,
            dropout_rate=0.1,
            conv_dims=[512, embedding_dim],
            l2_reg=0.001,
            num_neg_test=100
        )

        # --- 4. TRAINING ---
        sampler = WarpSampler(data.user_train, data.usernum, data.itemnum, 
                            batch_size=BATCH_SIZE, maxlen=MAX_SEQ_LEN, n_workers=3)
        
        logger.start_training()

        print("Inizio Training...")
        with Timer() as train_time:
            history = model.train_model(
                data, 
                sampler, 
                num_epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                learning_rate=LEARNING_RATE, 
                val_epoch=1,  # Saltiamo la validazione interna (evita ZeroDivisionError)
                eval_batch_size=1024,
                verbose=True
            )

        print(f'\nTraining completato in: {train_time.interval/60.0:.2f} mins')
        logger.end_training()

        logger.start_evaluation()
        print("Evaluating on test set...")
        with Timer() as eval_time:
            test_metrics = model.evaluate(data, seed=42, eval_batch_size=1024)
            
        logger.end_evaluation()

        print(f"Evaluation time: {eval_time.interval:.2f}s")
        print(f"\nTest Results:")
        print(f"  NDCG@10: {test_metrics[0]:.6f}")
        print(f"  HR@10:   {test_metrics[1]:.6f}")
        
        logger._log({
            "NDCG@10": test_metrics[0],
            "HR@10": test_metrics[1],
            "Training Time (mins)": train_time.interval/60.0,
            "Evaluation Time (s)": eval_time.interval
        })
        
logger.end_experiment()