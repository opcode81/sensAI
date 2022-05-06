import numpy as np
import pandas as pd
import seaborn as sns
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from typing import Generic, TypeVar, List, Union, Dict, Sequence, Optional

from ...util.string import ToStringMixin, dictString
from ...vector_model import VectorModel

# Note: in the 2020.2 version of PyCharm passing strings to bound is highlighted as error
# It does not cause runtime errors and the static type checker ignores the bound anyway, so it does not matter for now.
# However, this might cause problems with type checking in the future. Therefore, I moved the definition of TEvalStats
# below the definition of EvalStats. Unfortunately, the dependency in generics between EvalStats and Metric
# does not allow to define both, TMetric and TEvalStats, properly. For now we have to leave it with the bound as string
# and hope for the best in the future
TMetric = TypeVar("TMetric", bound="Metric")
TVectorModel = TypeVar("TVectorModel", bound=VectorModel)

PredictionArray = Union[np.ndarray, pd.Series, pd.DataFrame, list]


class EvalStats(Generic[TMetric], ToStringMixin):
    def __init__(self, metrics: List[TMetric], additionalMetrics: List[TMetric] = None):
        if len(metrics) == 0:
            raise ValueError("No metrics provided")
        self.metrics = metrics
        # Implementations of EvalStats will typically provide default metrics, therefore we include
        # the possibility for passing additional metrics here
        if additionalMetrics is not None:
            self.metrics = self.metrics + additionalMetrics
        self.name = None

    def setName(self, name: str):
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

    def _toStringObjectInfo(self) -> str:
        return dictString(self.getAll())


TEvalStats = TypeVar("TEvalStats", bound=EvalStats)


class Metric(Generic[TEvalStats], ABC):
    name: str

    def __init__(self, name: str = None):
        """
        :param name: the name of the metric; if None use the class' name attribute
        """
        # this raises an attribute error if a subclass does not specify a name as a static attribute nor as parameter
        self.name = name if name is not None else self.__class__.name

    @abstractmethod
    def computeValueForEvalStats(self, evalStats: TEvalStats) -> float:
        pass


class EvalStatsCollection(Generic[TEvalStats], ABC):
    def __init__(self, evalStatsList: List[TEvalStats]):
        self.statsList = evalStatsList
        metricNamesSet = None
        metricsList = []
        for es in evalStatsList:
            metrics = es.getAll()
            currentMetricNamesSet = set(metrics.keys())
            if metricNamesSet is None:
                metricNamesSet = currentMetricNamesSet
            else:
                if metricNamesSet != currentMetricNamesSet:
                    raise Exception(f"Inconsistent set of metrics in evaluation stats collection: Got {metricNamesSet} for one instance, {currentMetricNamesSet} for another")
            metricsList.append(metrics)
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


class PredictionEvalStats(EvalStats[TMetric], ABC):
    """
    Collects data for the evaluation of predicted values (including multi-dimensional predictions)
    and computes corresponding metrics
    """
    def __init__(self, y_predicted: Optional[PredictionArray], y_true: Optional[PredictionArray],
                 metrics: List[TMetric], additionalMetrics: List[TMetric] = None):
        """
        :param y_predicted: sequence of predicted values, or, in case of multi-dimensional predictions, either a data frame with
            one column per dimension or a nested sequence of values
        :param y_true: sequence of ground truth labels of same shape as y_predicted
        :param metrics: list of metrics to be computed on the provided data
        :param additionalMetrics: the metrics to additionally compute. This should only be provided if metrics is None
        """
        self.y_true = []
        self.y_predicted = []
        self.y_true_multidim = None
        self.y_predicted_multidim = None
        if y_predicted is not None:
            self.addAll(y_predicted, y_true)
        super().__init__(metrics, additionalMetrics=additionalMetrics)

    def add(self, y_predicted, y_true):
        """
        Adds a single pair of values to the evaluation
        Parameters:
            y_predicted: the value predicted by the model
            y_true: the true value
        """
        self.y_true.append(y_true)
        self.y_predicted.append(y_predicted)

    def addAll(self, y_predicted: PredictionArray, y_true: PredictionArray):
        """
        :param y_predicted: sequence of predicted values, or, in case of multi-dimensional predictions, either a data frame with
            one column per dimension or a nested sequence of values
        :param y_true: sequence of ground truth labels of same shape as y_predicted
        """
        def isSequence(x):
            return isinstance(x, pd.Series) or isinstance(x, list) or isinstance(x, np.ndarray)

        if isSequence(y_predicted) and isSequence(y_true):
            a, b = len(y_predicted), len(y_true)
            if a != b:
                raise Exception(f"Lengths differ (predicted {a}, truth {b})")
            if a > 0:
                firstItem = y_predicted.iloc[0] if isinstance(y_predicted, pd.Series) else y_predicted[0]
                isNestedSequence = isSequence(firstItem)
                if isNestedSequence:
                    for y_true_i, y_predicted_i in zip(y_true, y_predicted):
                        self.addAll(y_predicted=y_predicted_i, y_true=y_true_i)
                else:
                    self.y_true.extend(y_true)
                    self.y_predicted.extend(y_predicted)
        elif isinstance(y_predicted, pd.DataFrame) and isinstance(y_true, pd.DataFrame):
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
            self.y_true.extend(y_true)
            self.y_predicted.extend(y_predicted)
        else:
            raise Exception(f"Unhandled data types: {type(y_predicted)}, {type(y_true)}")

    def _toStringObjectInfo(self) -> str:
        return f"{super()._toStringObjectInfo()}, N={len(self.y_predicted)}"


def meanStats(evalStatsList: Sequence[EvalStats]) -> Dict[str, float]:
    """
    For a list of EvalStats objects compute the mean values of all metrics in a dictionary.
    Assumes that all provided EvalStats have the same metrics
    """
    dicts = [s.getAll() for s in evalStatsList]
    metrics = dicts[0].keys()
    return {m: np.mean([d[m] for d in dicts]) for m in metrics}


class EvalStatsPlot(Generic[TEvalStats], ABC):
    @abstractmethod
    def createFigure(self, evalStats: TEvalStats, subtitle: str) -> Optional[plt.Figure]:
        """
        :param evalStats: the evaluation stats from which to generate the plot
        :param subtitle: the plot's subtitle
        :return: the figure or None if this plot is not applicable/cannot be created
        """
        pass
