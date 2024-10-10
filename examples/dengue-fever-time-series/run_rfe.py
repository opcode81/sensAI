import os

from dengai.models.model_factory import ModelFactory
from dengai.prediction_problem import DengaiPrediction
from sensai.feature_selection.rfe import RecursiveFeatureElimination
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag

log = logging.getLogger(__name__)


def run_rfe(model_type="xgb"):
    experiment_name = f"{datetime_tag()}-{model_type}"
    result_writer = ResultWriter(os.path.join("results", "rfe", experiment_name))
    logging.add_file_logger(result_writer.path("log.txt"))

    if model_type == "xgb":
        params = {'colsample_bytree': 0.2867610159931737, 'gamma': 8.62166097248104, 'max_depth': 4,
            'min_child_weight': 12.0, 'reg_lambda': 0.7172266188561198}
        model_factory = lambda: ModelFactory.create_xgb(**params)
    else:
        raise ValueError(model_type)

    metric_computation = DengaiPrediction().create_metric_computation(use_cross_validation=False)
    rfe = RecursiveFeatureElimination(metric_computation)
    result = rfe.run(model_factory, minimise=True)
    result_writer.write_pickle("result", result)

    for i, step in enumerate(result.get_sorted_steps(), start=1):
        log.info(f"Top features #{i}: [loss={step.metric_value}] {step.features}")

    return vars()


if __name__ == '__main__':
    logging.configure()
    log.info("Starting")
    globals().update(run_rfe())
    log.info("Done")
