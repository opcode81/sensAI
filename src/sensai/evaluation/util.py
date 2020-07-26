from typing import Union, Dict, List

import numpy as np

from .eval_stats import EvalStats
from .evaluators import ModelEvaluator
from .vector_model.evaluators.base import VectorModelCrossValidator


# TODO: Extend to inlcude clustering evaluation once the clustering stuff is merged
def computeEvaluationMetricsDict(model, evaluatorOrValidator: Union[ModelEvaluator, VectorModelCrossValidator]) -> Dict[str, float]:
    if isinstance(evaluatorOrValidator, ModelEvaluator):
        evaluator: ModelEvaluator = evaluatorOrValidator
        evaluator.fitModel(model)
        data = evaluator.evalModel(model)
        return data.getEvalStats().getAll()
    elif isinstance(evaluatorOrValidator, VectorModelCrossValidator):
        crossValidator: VectorModelCrossValidator = evaluatorOrValidator
        data = crossValidator.evalModel(model)
        return data.getEvalStatsCollection().aggStats()
    else:
        raise ValueError(f"Unexpected evaluator/validator of type {type(evaluatorOrValidator)}")


def meanStats(evalStatsList: List[EvalStats]):
    """Returns, for a list of EvalStats objects, the mean values of all metrics in a dictionary"""
    dicts = [s.getAll() for s in evalStatsList]
    metrics = dicts[0].keys()
    return {m: np.mean([d[m] for d in dicts]) for m in metrics}
