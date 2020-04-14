# adapted from https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/xgboost_example.py

import ray
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb
import numpy as np
from hyperopt import hp
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from load_training_data import load_train_test_data


def XGBCallback(env):
    tune.track.log(**dict(env.evaluation_result_list))


def train_XGBoost(config):
    """
    Helper function that trains the XGBoost model with the provided hparams. Called by ray.Tune
    """
    train_set = xgb.DMatrix(train_X, label=train_y)
    val_set = xgb.DMatrix(val_X, label=val_y)
    bst = xgb.train(
        config, train_set, evals=[(val_set, "eval")], callbacks=[XGBCallback])
    preds = bst.predict(val_set)
    pred_labels = np.rint(preds)
    tune.track.log(
        val_auc=sklearn.metrics.roc_auc_score(val_y, pred_labels),
        done=True)

if __name__ == "__main__": 
    train_X, train_y, val_X, val_y, _, _ = load_train_test_data()
        
    # shutdown the service in case if was left on, then initializes it again
    ray.shutdown()
    ray.init(num_cpus=2)

    # defines the search space for the hyperparameters
    space = {
        'max_depth': hp.randint('max_depth', 1, 9),
        'eta': hp.loguniform('eta', -4.0, -1.0),
        'gamma': hp.loguniform('gamma',-8.0, 0.0),
        "grow_policy": hp.choice("grow_policy",["depthwise", "lossguide"]),
        "colsample_bytree": hp.uniform("colsample_bytree",0.3,0.7),
        "min_child_weight": hp.randint("min_child_weight",1,7)      
    }

     # define the hparams search algorithm, in charge of selecting promising values for the next iterations
    algo = HyperOptSearch(
        space, max_concurrent=4, metric="val_auc", mode="max")

    # other parameters needed by XGBoost
    config = {
        "verbosity": 0,
        "num_threads": 2,
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": ["auc", "ams@0", "logloss"]
    }

    # run the analysis
    # the scheduler is directly specified in the run() arguments
    analysis = tune.run(
        train_XGBoost,
        name="xgboost",
        search_alg=algo,
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=20,
        scheduler=ASHAScheduler(metric="val_auc", mode="max"))

    print("Best configuration is", analysis.get_best_config(metric="val_auc"))
    ray.shutdown()