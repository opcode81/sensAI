from abc import ABC, abstractmethod
from typing import List, Union, TypeVar, Dict, Generator, Tuple, Any, Generic

import numpy as np
import pandas as pd

from sensai.models.vector_model import VectorModel
from sensai.util.typing import PandasNamedTuple
from ...eval_stats import EvalStatsCollection, ModelEvaluationData, ModelCrossValidationData, EvalStats, TMetric

PredictionArray = Union[np.ndarray, pd.Series, pd.DataFrame, list]
TVectorModelEvalStats = TypeVar("TVectorModelEvalStats", bound="VectorModelEvalStats")
TVectorModelEvalData = TypeVar("TVectorModelEvalData", bound="VectorModelEvaluationData")
TVectorModel = TypeVar("TVectorModel", bound=VectorModel)
TVectorModelEvalStatsCollection = TypeVar("TVectorModelEvalStatsCollection", bound="VectorModelEvalStatsCollection")


class VectorModelEvalStats(EvalStats[TMetric], ABC):
    """Collects data for the evaluation of a model and computes corresponding metrics"""
    def __init__(self, y_predicted: PredictionArray = None, y_true: PredictionArray = None,
             metrics: List[TMetric] = None):
        self.y_true = []
        self.y_predicted = []
        self.y_true_multidim = None
        self.y_predicted_multidim = None
        if y_predicted is not None:
            self._addAll(y_predicted, y_true)
        super().__init__(metrics)

    def _add(self, y_predicted, y_true):
        """
        Adds a single pair of values to the evaluation

        Parameters:
            y_predicted: the value predicted by the model
            y_true: the true value
        """
        self.y_true.append(y_true)
        self.y_predicted.append(y_predicted)

    def _addAll(self, y_predicted, y_true):
        """
        Adds multiple predicted values and the corresponding ground truth values to the evaluation

        Parameters:
            y_predicted: pandas.Series, array or list of predicted values or, in the case of multi-dimensional models,
                        a pandas DataFrame (multiple series) containing predicted values
            y_true: an object of the same type/shape as y_predicted containing the corresponding ground truth values
        """
        if (isinstance(y_predicted, pd.Series) or isinstance(y_predicted, list) or isinstance(y_predicted, np.ndarray)) \
                and (isinstance(y_true, pd.Series) or isinstance(y_true, list) or isinstance(y_true, np.ndarray)):
            a, b = len(y_predicted), len(y_true)
            if a != b:
                raise Exception(f"Lengths differ (predicted {a}, truth {b})")
        elif isinstance(y_predicted, pd.DataFrame) and isinstance(y_true, pd.DataFrame):  # multiple time series
            # keep track of multidimensional data (to be used later in getEvalStatsCollection)
            y_predicted_multidim = y_predicted.values
            y_true_multidim = y_true.values
            dim = y_predicted_multidim.shape[1]
            if dim != y_true_multidim.shape[1]:
                raise Exception("Dimension mismatch")
            if self.y_true_multidim is None:
                self.y_predicted_multidim = [[] for _ in range(dim)]
                self.y_true_multidim = [[] for _ in range(dim)]
            if len(self.y_predicted_multidim) != dim:
                raise Exception("Dimension mismatch")
            for i in range(dim):
                self.y_predicted_multidim[i].extend(y_predicted_multidim[:, i])
                self.y_true_multidim[i].extend(y_true_multidim[:, i])
            # convert to flat data for this stats object
            y_predicted = y_predicted_multidim.reshape(-1)
            y_true = y_true_multidim.reshape(-1)
        else:
            raise Exception(f"Unhandled data types: {str(type(y_predicted))}, {str(type(y_true))}")
        self.y_true.extend(y_true)
        self.y_predicted.extend(y_predicted)


class VectorModelEvalStatsCollection(EvalStatsCollection[TVectorModelEvalStats], ABC):
    pass


class VectorModelEvaluationData(ModelEvaluationData[TVectorModelEvalStats], ABC):
    def __init__(self, statsDict: Dict[str, TVectorModelEvalStats], inputData: pd.DataFrame, model: VectorModel):
        """
        :param statsDict: a dictionary mapping from output variable name to the evaluation statistics object
        :param inputData: the input data that was used to produce the results
        :param model: the model that was used to produce predictions
        """
        self.evalStatsByVarName = statsDict
        self.predictedVarNames = list(self.evalStatsByVarName.keys())
        super().__init__(inputData, model)

    def getEvalStats(self, predictedVarName=None) -> TVectorModelEvalStats:
        if predictedVarName is None:
            if len(self.evalStatsByVarName) != 1:
                raise Exception(f"Must provide name of predicted variable name, as multiple variables were predicted {list(self.evalStatsByVarName.keys())}")
            else:
                predictedVarName = next(iter(self.evalStatsByVarName.keys()))
        evalStats = self.evalStatsByVarName.get(predictedVarName)
        if evalStats is None:
            raise ValueError(f"No evaluation data present for '{predictedVarName}'; known output variables: {list(self.evalStatsByVarName.keys())}")
        return evalStats

    def getDataFrame(self):
        """
        Returns an DataFrame with all evaluation metrics (one row per output variable)

        :return: a DataFrame containing evaluation metrics
        """
        statsDicts = []
        varNames = []
        for predictedVarName, evalStats in self.evalStatsByVarName.items():
            statsDicts.append(evalStats.getAll())
            varNames.append(predictedVarName)
        df = pd.DataFrame(statsDicts, index=varNames)
        df.index.name = "predictedVar"
        return df

    def iterInputOutputGroundTruthTuples(self, predictedVarName=None) -> Generator[Tuple[PandasNamedTuple, Any, Any], None, None]:
        evalStats = self.getEvalStats(predictedVarName)
        for i, namedTuple in enumerate(self.inputData.itertuples()):
            yield namedTuple, evalStats.y_predicted[i], evalStats.y_true[i]


class VectorModelCrossValidationData(ModelCrossValidationData[TVectorModel, TVectorModelEvalStatsCollection, TVectorModelEvalData],
                                     Generic[TVectorModel, TVectorModelEvalStatsCollection, TVectorModelEvalData, TVectorModelEvalStats]):
    def __init__(self, trainedModels: List[TVectorModel], evalDataList: List[TVectorModelEvalStats],
             predictedVarNames: List[str], testIndicesList=None):
        super().__init__(trainedModels, evalDataList)
        self.predictedVarNames = predictedVarNames
        self.testIndicesList = testIndicesList

    @property
    def modelName(self):
        return self.evalDataList[0].modelName

    @abstractmethod
    def _createEvalStatsCollection(self, l: List[TVectorModelEvalStats]) -> TVectorModelEvalStatsCollection:
        pass

    def getEvalStatsCollection(self, predictedVarName=None) -> TVectorModelEvalStatsCollection:
        if predictedVarName is None:
            if len(self.predictedVarNames) != 1:
                raise Exception("Must provide name of predicted variable")
            else:
                predictedVarName = self.predictedVarNames[0]
        evalStatsList = [evalData.getEvalStats(predictedVarName) for evalData in self.evalDataList]
        return self._createEvalStatsCollection(evalStatsList)

    def iterInputOutputGroundTruthTuples(self, predictedVarName=None) -> Generator[Tuple[PandasNamedTuple, Any, Any], None, None]:
        for evalData in self.evalDataList:
            evalStats = evalData.getEvalStats(predictedVarName)
            for i, namedTuple in enumerate(evalData.inputData.itertuples()):
                yield namedTuple, evalStats.y_predicted[i], evalStats.y_true[i]
