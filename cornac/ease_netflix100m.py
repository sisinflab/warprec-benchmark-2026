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

logger = LifecycleLogger("results/Netflix100M-EASE.txt")
logger.start_experiment("EASE")

# load the built-in MovieLens 100K and split the data based on ratio
logger.start_preprocessing()
ml_1m = cornac.datasets.movielens.load_feedback(variant='100M-CUSTOM')
rs = RatioSplit(data=ml_1m, test_size=0.1, seed=42, logger=logger)
logger.end_preprocessing()

# initialize models, here we are comparing: Biased MF, PMF, and BPR
ease_1 = EASE(lamb=100)
ease_2 = EASE(lamb=200)
ease_3 = EASE(lamb=500)
ease_4 = EASE(lamb=750)
ease_5 = EASE(lamb=1000)
ease_6 = EASE(lamb=2000)
models = [ease_1, ease_2, ease_3, ease_4, ease_5, ease_6]

# define metrics to evaluate the models
metrics = [Precision(k=10), Recall(k=10), NDCG(k=10), HitRatio(k=10), MAP()]

# put it together in an experiment, voilà!
output = cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True, verbose=True).run()

logger._log(output)