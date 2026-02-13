import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import LightGCN, NeuMF, ItemKNN, EASE
from cornac.metrics import NDCG, Precision, Recall, HitRatio, MAP
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (sigir26-resources)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from lifecycle_logger.lifecycle_logger import LifecycleLogger

logger = LifecycleLogger("results/MovieLens1M-LightGCN.txt")
logger.start_experiment("LightGCN")

# load the built-in MovieLens 100K and split the data based on ratio
logger.start_preprocessing()
ml_1m = cornac.datasets.movielens.load_feedback(variant='1M-CUSTOM')
rs = RatioSplit(data=ml_1m, test_size=0.1, seed=42, logger=logger)
logger.end_preprocessing()

# initialize models, here we are comparing: Biased MF, PMF, and BPR
lightgcn_1 = LightGCN(emb_size=64, num_epochs=10, learning_rate=0.001, num_layers=2)
lightgcn_2 = LightGCN(emb_size=128, num_epochs=10, learning_rate=0.001, num_layers=2)
lightgcn_3 = LightGCN(emb_size=256, num_epochs=10, learning_rate=0.001, num_layers=2)
lightgcn_4 = LightGCN(emb_size=64, num_epochs=10, learning_rate=0.001, num_layers=3)
lightgcn_5 = LightGCN(emb_size=128, num_epochs=10, learning_rate=0.001, num_layers=3)
lightgcn_6 = LightGCN(emb_size=256, num_epochs=10, learning_rate=0.001, num_layers=3)
models = [lightgcn_1, lightgcn_2, lightgcn_3, lightgcn_4, lightgcn_5, lightgcn_6]

# define metrics to evaluate the models
metrics = [Precision(k=10), Recall(k=10), NDCG(k=10), HitRatio(k=10), MAP()]

# put it together in an experiment, voilà!
output = cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True, verbose=True).run()

logger._log(output)