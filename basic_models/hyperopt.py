import logging
from typing import Dict, Sequence, Any, Callable, Generator

import pandas as pd

from .basic_models_base import VectorModel, VectorModelEvaluator, VectorModelCrossValidator


log = logging.getLogger(__name__)


def iterParamCombinations(hyperParamValues: Dict[str, Sequence[Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Create all possible combinations of values from a dictionary of possible parameter values

    :param hyperParamValues: a mapping from parameter names to lists of possible values
    :return: a dictionary mapping each parameter name to one of the values
    """
    pairs = list(hyperParamValues.items())

    def _iterRecursiveParamCombinations(pairs, i, params):
        """
        Recursive function to create all possible combinations from a list of key-array entries.
        :param pairs: a dictionary of parameter names and their corresponding values
        :param i: the recursive step
        :param params: a dictionary for the iteration results
        """
        if i == len(pairs):
            yield dict(params)
        else:
            paramName, paramValues = pairs[i]
            for paramValue in paramValues:
                params[paramName] = paramValue
                yield from _iterRecursiveParamCombinations(pairs, i+1, params)

    return _iterRecursiveParamCombinations(pairs, 0, {})


class GridSearch:
    def __init__(self, modelFactory: Callable[..., VectorModel], parameterOptions: Dict[str, Sequence[Any]]):
        self.modelFactory = modelFactory
        self.parameterOptions = parameterOptions

    def _run(self, dictProviderFn: Callable[[VectorModel], Dict[str, Any]]):
        df = None
        cols = None
        for i, paramsDict in enumerate(iterParamCombinations(self.parameterOptions)):
            model = self.modelFactory(**paramsDict)
            values = dictProviderFn(model)
            values.update(paramsDict)
            if df is None:
                cols = list(values.keys())
                df = pd.DataFrame(columns=cols)
            df.loc[i] = [values[c] for c in cols]
            log.info(f"Updated grid search result:\n{df.to_string()}")
        return df

    def runWithEvaluator(self, evaluator: VectorModelEvaluator) -> pd.DataFrame:
        def dictFromEvaluator(model):
            evaluator.fitModel(model)
            data = evaluator.evalModel(model)
            return data.getEvalStats().getAll()
        return self._run(dictFromEvaluator)

    def runWithCrossValidator(self, crossValidator: VectorModelCrossValidator) -> pd.DataFrame:
        def dictFromCrossValidator(model):
            data = crossValidator.evalModel(model)
            return data.getEvalStatsCollection().aggStats()
        return self._run(dictFromCrossValidator)

