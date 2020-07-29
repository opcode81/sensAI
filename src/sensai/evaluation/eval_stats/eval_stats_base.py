from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Union, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sensai import VectorModel

TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)
TMetric = TypeVar("TMetric", bound=Metric)
TVectorModelEvalStats = TypeVar("TVectorModelEvalStats", bound=VectorModelEvalStats)
TVectorModel = TypeVar("TVectorModel", bound=VectorModel)

PredictionArray = Union[np.ndarray, pd.Series, pd.DataFrame, list]


class Metric(Generic[TEvalStats], ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def computeValueForEvalStats(self, evalStats: TEvalStats) -> float:
        pass


class EvalStats(Generic[TMetric]):
    def __init__(self, metrics: List[TMetric]):
        if len(metrics) == 0:
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


TEvalStats = TypeVar("TEvalStats", bound=EvalStats)


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


class VectorModelEvalStats(EvalStats[TMetric], ABC):
    """
    Collects data for the evaluation of a model and computes corresponding metrics
    """
    def __init__(self, y_predicted: PredictionArray, y_true: PredictionArray, metrics: List[TMetric]):
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
            raise Exception(f"Unhandled data types: {type(y_predicted)}, {type(y_true)}")
        self.y_true.extend(y_true)
        self.y_predicted.extend(y_predicted)
