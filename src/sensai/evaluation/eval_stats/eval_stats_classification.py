import numpy as np
import pandas as pd
import sklearn
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import List, Sequence

from .eval_stats_base import PredictionArray, PredictionEvalStats, EvalStatsCollection, Metric
from ...util.plot import plotMatrix


class ClassificationMetric(Metric["ClassificationEvalStats"], ABC):
    requiresProbabilities = False

    def computeValueForEvalStats(self, evalStats: "ClassificationEvalStats"):
        return self.computeValue(evalStats.y_true, evalStats.y_predicted, evalStats.y_predictedClassProbabilities)

    def computeValue(self, y_true, y_predicted, y_predictedClassProbabilities=None):
        if self.requiresProbabilities and y_predictedClassProbabilities is None:
            raise ValueError(f"{self} requires class probabilities")
        return self._computeValue(y_true, y_predicted, y_predictedClassProbabilities)

    @abstractmethod
    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        pass


class ClassificationMetricAccuracy(ClassificationMetric):
    name = "ACC"

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        return accuracy_score(y_true=y_true, y_pred=y_predicted)


class ClassificationMetricGeometricMeanOfTrueClassProbability(ClassificationMetric):
    name = "GeoMeanTrueClassProb"
    requiresProbabilities = True

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        y_predicted_proba_true_class = np.zeros(len(y_true))
        for i in range(len(y_true)):
            trueClass = y_true[i]
            if trueClass not in y_predictedClassProbabilities.columns:
                y_predicted_proba_true_class[i] = 0
            else:
                y_predicted_proba_true_class[i] = y_predictedClassProbabilities[trueClass].iloc[i]
        # the 1e-3 below prevents lp = -inf due to single entries with y_predicted_proba_true_class=0
        lp = np.log(np.maximum(1e-3, y_predicted_proba_true_class))
        return np.exp(lp.sum() / len(lp))


class ClassificationMetricTopNAccuracy(ClassificationMetric):
    requiresProbabilities = True

    def __init__(self, n: int):
        self.n = n
        super().__init__(name=f"Top{n}Accuracy")

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        labels = y_predictedClassProbabilities.columns
        cnt = 0
        for i, rowValues in enumerate(y_predictedClassProbabilities.values.tolist()):
            pairs = sorted(zip(labels, rowValues), key=lambda x: x[1], reverse=True)
            if y_true[i] in (x[0] for x in pairs[:self.n]):
                cnt += 1
        return cnt / len(y_true)


# TODO: make some kwargs to args, infer y_predicted when classProbabilities are passed. We can also infer labels
#  (though not reliably) if they are not passed
class ClassificationEvalStats(PredictionEvalStats["ClassificationMetric"]):
    def __init__(self, y_predicted: PredictionArray = None,
                 y_true: PredictionArray = None,
                 y_predictedClassProbabilities: pd.DataFrame = None,
                 labels: PredictionArray = None,
                 metrics: Sequence["ClassificationMetric"] = None,
                 additionalMetrics: Sequence["ClassificationMetric"] = None):
        """
        :param y_predicted: the predicted class labels
        :param y_true: the true class labels
        :param y_predictedClassProbabilities: a data frame whose columns are the class labels and whose values are probabilities
        :param labels: the list of class labels
        :param metrics: the metrics to compute for evaluation; if None, use default metrics
        :param additionalMetrics: the metrics to additionally compute
        """
        self.labels = labels
        self.y_predictedClassProbabilities = y_predictedClassProbabilities
        self._probabilitiesAvailable = y_predictedClassProbabilities is not None
        if self._probabilitiesAvailable:
            colSet = set(y_predictedClassProbabilities.columns)
            if colSet != set(labels):
                raise ValueError(f"Set of columns in class probabilities data frame ({colSet}) does not correspond to labels ({labels}")
            if len(y_predictedClassProbabilities) != len(y_true):
                raise ValueError("Row count in class probabilities data frame does not match ground truth")

        if metrics is None:
            metrics = [ClassificationMetricAccuracy(), ClassificationMetricGeometricMeanOfTrueClassProbability()]
        metrics = list(metrics)
        if additionalMetrics is not None:
            for m in additionalMetrics:
                if not self._probabilitiesAvailable and m.requiresProbabilities:
                    raise ValueError(f"Additional metric {m} not supported, as class probabilities were not provided")

        super().__init__(y_predicted, y_true, metrics, additionalMetrics=additionalMetrics)

    def getConfusionMatrix(self) -> "ConfusionMatrix":
        return ConfusionMatrix(self.y_true, self.y_predicted)

    def getAccuracy(self):
        return self.computeMetricValue(ClassificationMetricAccuracy())

    def getAll(self):
        """Gets a dictionary with all metrics"""
        d = {}
        for metric in self.metrics:
            if not metric.requiresProbabilities or self._probabilitiesAvailable:
                d[metric.name] = self.computeMetricValue(metric)
        return d

    def plotConfusionMatrix(self, normalize=True, titleAdd: str = None):
        # based on https://scikit-learn.org/0.20/auto_examples/model_selection/plot_confusion_matrix.html
        confusionMatrix = self.getConfusionMatrix()
        return confusionMatrix.plot(normalize=normalize, titleAdd=titleAdd)


class ClassificationEvalStatsCollection(EvalStatsCollection[ClassificationEvalStats]):
    def __init__(self, evalStatsList: List[ClassificationEvalStats]):
        super().__init__(evalStatsList)
        self.globalStats = None

    # TODO once we moved to python 3.8: move to base class and use the new get_args method to infer the generic type at runtime
    #  https://docs.python.org/3/library/typing.html#typing.get_args
    def getGlobalStats(self) -> ClassificationEvalStats:
        """
        Gets an evaluation statistics object that combines the data from all contained eval stats objects
        """
        if self.globalStats is None:
            y_true = np.concatenate([evalStats.y_true for evalStats in self.statsList])
            y_predicted = np.concatenate([evalStats.y_predicted for evalStats in self.statsList])
            self.globalStats = ClassificationEvalStats(y_predicted, y_true)
        return self.globalStats


class ConfusionMatrix:
    def __init__(self, y_true, y_predicted):
        self.labels = sklearn.utils.multiclass.unique_labels(y_true, y_predicted)
        self.confusionMatrix = confusion_matrix(y_true, y_predicted, labels=self.labels)

    def plot(self, normalize=True, titleAdd: str = None):
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix (Counts)'
        return plotMatrix(self.confusionMatrix, title, self.labels, self.labels, 'true class', 'predicted class', normalize=normalize,
            titleAdd=titleAdd)
