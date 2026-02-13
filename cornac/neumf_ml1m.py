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

logger = LifecycleLogger("results/MovieLens1M-NeuMF.txt")
logger.start_experiment("NeuMF")

# load the built-in MovieLens 100K and split the data based on ratio
logger.start_preprocessing()
ml_1m = cornac.datasets.movielens.load_feedback(variant='1M-CUSTOM')
rs = RatioSplit(data=ml_1m, test_size=0.1, seed=42, logger=logger)
logger.end_preprocessing()

# initialize models, here we are comparing: Biased MF, PMF, and BPR
mlp_emb_size = 64
cornac_layers = [mlp_emb_size * 2, mlp_emb_size, mlp_emb_size]
neumf_1 = NeuMF(num_factors=mlp_emb_size, layers=cornac_layers, reg=0.0001, num_epochs=10, batch_size=8192, lr=0.001, num_neg=0, seed=42, backend="pytorch")
neumf_4 = NeuMF(num_factors=mlp_emb_size, layers=cornac_layers, reg=0.0001, num_epochs=10, batch_size=8192, lr=0.001, num_neg=1, seed=42, backend="pytorch")

mlp_emb_size = 128
cornac_layers = [mlp_emb_size * 2, mlp_emb_size, mlp_emb_size]
neumf_2 = NeuMF(num_factors=mlp_emb_size, layers=cornac_layers, reg=0.0001, num_epochs=10, batch_size=8192, lr=0.001, num_neg=0, seed=42, backend="pytorch")
neumf_5 = NeuMF(num_factors=mlp_emb_size, layers=cornac_layers, reg=0.0001, num_epochs=10, batch_size=8192, lr=0.001, num_neg=1, seed=42, backend="pytorch")

mlp_emb_size = 256
cornac_layers = [mlp_emb_size * 2, mlp_emb_size, mlp_emb_size]
neumf_3 = NeuMF(num_factors=mlp_emb_size, layers=cornac_layers, reg=0.0001, num_epochs=10, batch_size=8192, lr=0.001, num_neg=0, seed=42, backend="pytorch")
neumf_6 = NeuMF(num_factors=mlp_emb_size, layers=cornac_layers, reg=0.0001, num_epochs=10, batch_size=8192, lr=0.001, num_neg=1, seed=42, backend="pytorch")

models = [neumf_1, neumf_2, neumf_3, neumf_4, neumf_5, neumf_6]

# define metrics to evaluate the models
metrics = [Precision(k=10), Recall(k=10), NDCG(k=10), HitRatio(k=10), MAP()]

# put it together in an experiment, voilà!
output = cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True, verbose=True).run()

logger._log(output)
