import logging

import ray
from hyperopt import hp
from ray import tune
from ray.tune.integration.keras import TuneReporterCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow import keras

from load_training_data import load_train_test_data
from deep_and_wide import DeepAndWide, deep_and_wide

logging.getLogger("tensorflow").setLevel(logging.ERROR)

def train_nn(config, reporter):
    """
    Function that builds, compiles, and fit the Deep and Wide model. To be called by ray.Tune
    """
    epochs = 80
    model = deep_and_wide(hidden_dim=config["hidden_dim"],
                        activation=config["activation"],
                        dropout=0.3,
                        n_hidden_layers=config["n_hidden_layers"],
                        regularization=config["regularization"])

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate = config["lr"]),
        metrics=["accuracy",keras.metrics.AUC(curve="ROC"),keras.metrics.Precision(),keras.metrics.Recall()])

    history = model.fit(
                train_X,
                train_y,
                epochs=epochs,
                verbose=0,
                validation_data=(val_X, val_y),
                callbacks=[TuneReporterCallback(reporter)])
    
    results = {"val_precision":history.history["val_precision"][-1],
                   "val_recall":history.history["val_recall"][-1],
                   "val_auc":history.history["val_AUC"][-1],
                   "train_auc":history.history["AUC"][-1],
                   "done":True}

    tune.track.log(**results)

if __name__ == "__main__":
    train_X, train_y, val_X, val_y, _, _ = load_train_test_data()
    
    # shutdown the service in case if was left on, then initializes it again
    ray.shutdown()
    ray.init(num_cpus=2)

    # defines the scheduler in charge of allocating resources to each job
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="keras_info/auc",
        mode="max",
        max_t=400,
        grace_period=10)

    # defines the search space for the hyperparameters
    space = {
        'hidden_dim': hp.randint('hidden_dim', 32, 256),
        'regularization': hp.loguniform('regularization', -4.0, -2.0),
        'lr': hp.loguniform('lr',-4.0, -1.0),
        "n_hidden_layers": hp.choice("n_hidden_layers",[1,2,3]),
        "activation": hp.choice("activation",["relu","tanh"]),
        
    }

    # define the hparams search algorithm, in charge of selecting promising values for the next iterations
    algo = HyperOptSearch(
        space, max_concurrent=4, metric="keras_info/auc", mode="max")

    # finally run the hyperparameters tuning job
    analysis = tune.run(
                    train_nn,
                    name="nn_20200413",
                    search_alg=algo,
                    scheduler=sched,
                    stop={
                        "keras_info/auc": 0.99,
                        "training_iteration": 100
                    },
                    num_samples=80,
                    config={
                        "threads": 2,
                        "iterations":100
                    })

    print("Best configuration is", analysis.get_best_config(metric="keras_info/auc"))
    ray.shutdown()