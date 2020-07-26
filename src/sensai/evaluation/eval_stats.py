from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Union, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ..models.base import FitterModel, PredictorModel

TEvalStats = TypeVar("TEvalStats", bound='EvalStats')
TEvalData = TypeVar("TEvalData", bound="ModelEvaluationData")
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound='EvalStatsCollection')
TMetric = TypeVar("TMetric", bound="Metric")
ModelType = Union[FitterModel, PredictorModel]
TModel = TypeVar("TModel", bound=ModelType)


# TODO or not TODO: the inheritance structure here creates a circular dependency of eval stats and metrics.
#  This inhibits separating metrics to a separate module (circular imports) and might be generally undesirable
#  Do we want to keep it this way?
class Metric(Generic[TEvalStats], ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def computeValueForEvalStats(self, evalStats: TEvalStats) -> float:
        pass


class EvalStats(Generic[TMetric]):
    def __init__(self, metrics: List[TMetric] = None):
        if metrics is None:
            raise ValueError("No metrics provided")
        self.metrics = metrics
        self.name = None

    def setName(self, name):
        self.name = name

    def addMetric(self, metric: TMetric):
        self.metrics.append(metric)

    def computeMetricValue(self, metric: TMetric) -> float:
        return metric.computeValueForEvalStats(self)

    def getAll(self) -> Dict[str, float]:
        """Gets a dictionary with all metrics"""
        d = {}
        for metric in self.metrics:
            d[metric.name] = self.computeMetricValue(metric)
        return d

    def __str__(self):
        d = self.getAll()
        return "EvalStats[%s]" % ", ".join([f"{k}={v:4f}" for (k, v) in d.items()])


class EvalStatsCollection(Generic[TEvalStats], ABC):
    def __init__(self, evalStatsList: List[TEvalStats]):
        self.statsList = evalStatsList
        metricsList = [es.getAll() for es in evalStatsList]
        metricNames = sorted(metricsList[0].keys())
        self.metrics = {metric: [d[metric] for d in metricsList] for metric in metricNames}

    def getValues(self, metric):
        return self.metrics[metric]

    def aggStats(self):
        agg = {}
        for metric, values in self.metrics.items():
            agg[f"mean[{metric}]"] = float(np.mean(values))
            agg[f"std[{metric}]"] = float(np.std(values))
        return agg

    def meanStats(self):
        metrics = {metric: np.mean(values) for (metric, values) in self.metrics.items()}
        metrics.update({f"StdDev[{metric}]": np.std(values) for (metric, values) in self.metrics.items()})
        return metrics

    def plotDistribution(self, metric):
        values = self.metrics[metric]
        plt.figure()
        plt.title(metric)
        sns.distplot(values)

    def toDataFrame(self) -> pd.DataFrame:
        """
        :return: a DataFrame with the evaluation metrics from all contained EvalStats objects;
            the EvalStats' name field being used as the index if it is set
        """
        data = dict(self.metrics)
        index = [stats.name for stats in self.statsList]
        if len([n for n in index if n is not None]) == 0:
            index = None
        return pd.DataFrame(data, index=index)

    @abstractmethod
    def getGlobalStats(self) -> TEvalStats:
        pass

    def __str__(self):
        return f"{self.__class__.__name__}[" + \
               ", ".join([f"{key}={self.aggStats()[key]:.4f}" for key in self.metrics]) + "]"


class ModelEvaluationData(ABC, Generic[TEvalStats]):
    def __init__(self, inputData: pd.DataFrame, model: ModelType):
        """
        :param inputData: the input data that was used to produce the results
        :param model: the model that was used to produce predictions
        """
        self.inputData = inputData
        self.modelName = model.getName()

    @abstractmethod
    def getEvalStats(self, *kwargs) -> TEvalStats:
        pass

    def getDataFrame(self) -> pd.DataFrame:
        """
        Returns an DataFrame with all evaluation metrics (one row per output variable)

        :return: a DataFrame containing evaluation metrics
        """
        pass


class ModelCrossValidationData(ABC, Generic[TModel, TEvalStatsCollection, TEvalData]):
    def __init__(self, trainedModels: List[TModel], evalDataList: List[TEvalData]):
        self.trainedModels = trainedModels
        self.evalDataList = evalDataList

    @property
    def modelName(self):
        return self.evalDataList[0].modelName

    @abstractmethod
    def getEvalStatsCollection(self, *kwargs) -> TEvalStatsCollection:
        pass
