import os
import warnings
from typing import Literal

import hyperopt
from hyperopt import hp

from dengai.models.model_factory import ModelFactory
from dengai.prediction_problem import DengaiPrediction
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.util.pickle import load_pickle

log = logging.getLogger(__name__)


def run_hyperopt(model_type: Literal["xgb"] = "xgb_window", use_cross_validation=False):
    experiment_name = f"{datetime_tag()}-{model_type}"
    result_writer = ResultWriter(os.path.join("results", "hyperopt", experiment_name))
    logging.add_file_logger(result_writer.path("log.txt"))

    pred = DengaiPrediction()
    week_window_size = pred.week_window_size

    if model_type == "xgb_window":
        initial_space = [
            {
                'max_depth': 6,
                'gamma': 0,
                'reg_lambda': 0,
                'colsample_bytree': 1,
                'min_child_weight': 1,
            }
        ]
        space = {
            'max_depth': hp.quniform("max_depth", 3, 18, 1),
            'gamma': hp.uniform('gamma', 0, 9),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 12, 2),
        }

        def create_model(params):
            return ModelFactory.create_xgb_window(window_size=week_window_size,
                verbosity=0,
                max_depth=int(params['max_depth']),
                gamma=params['gamma'],
                reg_lambda=params['reg_lambda'],
                min_child_weight=int(params['min_child_weight']),
                colsample_bytree=params['colsample_bytree']).with_name(model_type)

        hours = 1.5
        warnings.filterwarnings("ignore")
    else:
        raise ValueError(model_type)

    metric_computation = pred.create_metric_computation(use_cross_validation=use_cross_validation)

    def objective(params):
        log.info(f"Evaluating {params}")
        result = metric_computation.compute_metric_value(lambda: create_model(params))
        loss = result.metric_value
        log.info(f"Loss[{metric_computation.metric.name}]={loss}")
        return {'loss': loss, 'status': hyperopt.STATUS_OK}

    trials_file = result_writer.path("trials.pickle")
    logging.getLogger("sensai").setLevel(logging.WARN)
    log.info(f"Starting hyperparameter optimisation for {model_type}")
    hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, timeout=hours*3600, show_progressbar=False,
        trials_save_file=trials_file, points_to_evaluate=initial_space)
    logging.getLogger("sensai").setLevel(logging.INFO)
    trials: hyperopt.Trials = load_pickle(trials_file)
    log.info(f"Best trial: {trials.best_trial}")


if __name__ == '__main__':
    logging.configure()
    log.info("Starting")
    run_hyperopt()
    log.info("Done")
