import logging
from typing import Dict, Sequence, Any, Callable, Generator, Union

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .basic_models_base import VectorModel
from .evaluation import VectorModelEvaluator, VectorModelCrossValidator, computeEvaluationMetricsDict


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
    log = log.getChild(__qualname__)

    def __init__(self, modelFactory: Callable[..., VectorModel], parameterOptions: Dict[str, Sequence[Any]], numProcesses=1,
            csvResultsPath=None):
        self.modelFactory = modelFactory
        self.parameterOptions = parameterOptions
        self.numProcesses = numProcesses
        self.csvResultsPath = csvResultsPath
        self._executor = None

    @classmethod
    def _evalParams(cls, modelFactory, evaluatorOrValidator, **params):
        cls.log.info(f"Evaluating {params}")
        model = modelFactory(**params)
        values = computeEvaluationMetricsDict(model, evaluatorOrValidator)
        values["str(model)"] = str(model)
        values.update(**params)
        return values

    def run(self, evaluatorOrValidator: Union[VectorModelEvaluator, VectorModelCrossValidator]):
        executor = ProcessPoolExecutor(max_workers=self.numProcesses) if self.numProcesses > 1 else ThreadPoolExecutor(max_workers=1)
        futures = []
        for i, paramsDict in enumerate(iterParamCombinations(self.parameterOptions)):
            futures.append(executor.submit(self._evalParams, self.modelFactory, evaluatorOrValidator, **paramsDict))

        df = None
        cols = None
        for i, future in enumerate(futures):
            values = future.result()
            if df is None:
                cols = list(values.keys())
                df = pd.DataFrame(columns=cols)
            df.loc[i] = [values[c] for c in cols]
            log.info(f"Updated grid search result:\n{df.to_string()}")
            if self.csvResultsPath is not None:
                df.to_csv(self.csvResultsPath, index=False)
        return df
