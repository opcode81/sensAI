import logging
from abc import abstractmethod, ABC
from typing import Union, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

log = logging.getLogger(__name__)


class EvalStats(ABC):
    """Collects data for the evaluation of a model and computes corresponding metrics"""
    def __init__(self, y_predicted: Union[list, pd.Series, pd.DataFrame] = None,
                 y_true: Union[list, pd.Series, pd.DataFrame] = None):
        self.y_true = []
        self.y_predicted = []
        self.y_true_multidim = None
        self.y_predicted_multidim = None
        self.name = None

        if y_predicted is not None:
            self.addAll(y_predicted, y_true)

    def __str__(self):
        d = self.getAll()
        return "EvalStats[%s]" % ", ".join(["%s=%.4f" % (k, v) for (k, v) in d.items()])

    def setName(self, name):
        self.name = name

    @abstractmethod
    def getAll(self):
        pass

    def add(self, y_predicted, y_true):
        """
        Adds a single pair of values to the evaluation

        Parameters:
            y_predicted: the value predicted by the model
            y_true: the true value
        """
        self.y_true.append(y_true)
        self.y_predicted.append(y_predicted)

    def addAll(self, y_predicted, y_true):
        """
        Adds multiple predicted values and the corresponding ground truth values to the evaluation

        Parameters:
            y_predicted: pandas.Series or list of predicted values or, in the case of multi-dimensional models,
                        a pandas DataFrame (multiple series) containing predicted value
            y_true: an object of the same type/shape as y_predicted containing the corresponding ground truth values
        """
        if (isinstance(y_predicted, pd.Series) or isinstance(y_predicted, list) or isinstance(y_predicted, np.ndarray)) \
                and (isinstance(y_true, pd.Series) or isinstance(y_true, list) or isinstance(y_true, np.ndarray)):
            a, b = len(y_predicted), len(y_true)
            if a != b:
                raise Exception("Lengths differ (predicted %d, truth %d)" % (a, b))
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
            raise Exception("Unhandled data types: %s, %s" % (str(type(y_predicted)), str(type(y_true))))
        self.y_true.extend(y_true)
        self.y_predicted.extend(y_predicted)


class ClassificationEvalStats(EvalStats):
    def __init__(self, y_predicted: Union[list, pd.Series, pd.DataFrame] = None,
            y_true: Union[list, pd.Series, pd.DataFrame] = None,
            y_predictedClassProbabilities: pd.DataFrame = None,
            labels: Union[list, pd.Series, pd.DataFrame, np.ndarray] = None):
        """
        :param y_predicted: the predicted class labels
        :param y_true: the true class labels
        :param y_predictedClassProbabilities: a data frame whose columns are the class labels and whose values are probabilities
        :param labels: the list of class labels
        """
        super().__init__(y_predicted=y_predicted, y_true=y_true)
        self.labels = labels
        self.y_predicted_proba = y_predictedClassProbabilities
        self._probabilitiesAvailable = y_predictedClassProbabilities is not None

        if self._probabilitiesAvailable:
            self.y_predicted_proba_true_class = self._computeProbabilitiesOfTrueClass()

    def _computeProbabilitiesOfTrueClass(self):
        result = []
        for i in range(len(self.y_true)):
            trueClass = self.y_true[i]
            probTrueClass = self.y_predicted_proba[trueClass].iloc[i]
            result.append(probTrueClass)
        return np.array(result)

    def getConfusionMatrix(self):
        return confusion_matrix(y_true=self.y_true, y_pred=self.y_predicted)

    def getAccuracy(self):
        return accuracy_score(y_true=self.y_true, y_pred=self.y_predicted)

    def getAveragedPrecision(self):
        return precision_score(y_true=self.y_true, y_pred=self.y_predicted, average='weighted')

    def getAveragedRecall(self):
        return recall_score(y_true=self.y_true, y_pred=self.y_predicted, average='weighted')

    def getAveragedF1(self):
        return f1_score(y_true=self.y_true, y_pred=self.y_predicted, average='weighted')

    def getGeoMeanTrueClassProbability(self):
        if not self._probabilitiesAvailable:
            return None
        lp = np.log(self.y_predicted_proba_true_class)
        return np.exp(lp.sum() / len(lp))

    def getAll(self):
        """Gets a dictionary with all metrics"""
        d = dict(ACC=self.getAccuracy())
        if self._probabilitiesAvailable:
            d["GeoMeanTrueClassProb"] = self.getGeoMeanTrueClassProbability()
        return d


class RegressionEvalStats(EvalStats):
    """Collects data for the evaluation of a model and computes corresponding metrics"""

    def getAll(self):
        """Gets a dictionary with all metrics"""
        return dict(RRSE=self.getRRSE(), R2=self.getR2(), PCC=self.getCorrelationCoeff(), MAE=self.getMAE(),
                    StdDevAE=self.getStdDevAE(), RMSE=self.getRMSE())

    def getRRSE(self):
        """Gets the root relative squared error"""
        y_predicted = np.array(self.y_predicted)
        y_true = np.array(self.y_true)
        mean_y = np.mean(y_true)
        residuals = y_predicted - y_true
        mean_deviation = y_true - mean_y
        return np.sqrt(np.sum(residuals * residuals) / np.sum(mean_deviation * mean_deviation))

    def getCorrelationCoeff(self):
        """Gets the Pearson correlation coefficient (PCC)"""
        cov = np.cov([self.y_true, self.y_predicted])
        return cov[0][1] / np.sqrt(cov[0][0] * cov[1][1])

    def getR2(self):
        """Gets the R^2 score"""
        rrse = self.getRRSE()
        return 1.0 - rrse*rrse

    def _getErrors(self):
        y_predicted = np.array(self.y_predicted)
        y_true = np.array(self.y_true)
        return y_predicted - y_true

    def _getAbsErrors(self):
        return np.abs(self._getErrors())

    def getMAE(self):
        """Gets the mean absolute error"""
        return np.mean(self._getAbsErrors())

    def getRMSE(self):
        """Gets the root mean squared error"""
        errors = self._getErrors()
        return np.sqrt(np.mean(errors * errors))

    def getStdDevAE(self):
        """Gets the standard deviation of the absolute error"""
        return np.std(self._getAbsErrors())

    def getEvalStatsCollection(self):
        """
        For the case where we collected data on multiple dimensions, obtain a stats collection where
        each object in the collection holds stats on just one dimension
        """
        if self.y_true_multidim is None:
            raise Exception("No multi-dimensional data was collected")
        dim = len(self.y_true_multidim)
        statsList = []
        for i in range(dim):
            stats = RegressionEvalStats()
            stats.addAll(self.y_predicted_multidim[i], self.y_true_multidim[i])
            statsList.append(stats)
        return RegressionEvalStatsCollection(statsList)

    def plotErrorDistribution(self, bins=500, figure=True):
        errors = self._getErrors()
        fig = None
        if figure:
            fig = plt.figure()
        sns.distplot(errors, bins=bins)
        return fig

    def plotScatterGroundTruthPredictions(self, figure=True, **kwargs):
        fig = None
        if figure:
            fig = plt.figure()
        plt.scatter(self.y_true, self.y_predicted, **kwargs)
        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        return fig


def meanStats(evalStatsList):
    """Returns, for a list of EvalStats objects, the mean values of all metrics in a dictionary"""
    dicts = [s.getAll() for s in evalStatsList]
    metrics = dicts[0].keys()
    return {m: np.mean([d[m] for d in dicts]) for m in metrics}


class EvalStatsCollection(ABC):
    def __init__(self, evalStatsList: List[EvalStats]):
        self.statsList = evalStatsList
        metricsList = [es.getAll() for es in evalStatsList]
        metricNames = sorted(metricsList[0].keys())
        self.metrics = {metric: [d[metric] for d in metricsList] for metric in metricNames}

    def getValues(self, metric):
        return self.metrics[metric]

    def aggStats(self):
        agg = {}
        for metric, values in self.metrics.items():
            agg[f"mean[{metric}]"] = np.mean(values)
            agg[f"std[{metric}]"] = np.std(values)
        return agg

    def meanStats(self):
        metrics = {metric: np.mean(values) for (metric, values) in self.metrics.items()}
        metrics.update({"StdDev[%s]" % metric: np.std(values) for (metric, values) in self.metrics.items()})
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

    def __str__(self):
        return f"{self.__class__.__name__}[" + ", ".join(["%s=%.4f" % (key, self.aggStats()[key])
                                                          for key in self.metrics]) + "]"


class RegressionEvalStatsCollection(EvalStatsCollection):
    def __init__(self, evalStatsList: List[RegressionEvalStats]):
        super().__init__(evalStatsList)
        self.globalStats = None

    def getGlobalStats(self):
        """
        Gets an evaluation statistics object that combines the data from all contained eval stats objects
        """
        if self.globalStats is None:
            y_true = np.concatenate([evalStats.y_true for evalStats in self.statsList])
            y_predicted = np.concatenate([evalStats.y_predicted for evalStats in self.statsList])
            self.globalStats = RegressionEvalStats(y_predicted, y_true)
        return self.globalStats

    def plotGroundTruthVsPredictionsPairPlot(self, title=None):
        gstats = self.getGlobalStats()
        df = pd.DataFrame({"groundTruth": gstats.y_true, "prediction": gstats.y_predicted})
        g = sns.pairplot(df)
        if title is not None:
            g.fig.suptitle(title)


class ClassificationEvalStatsCollection(EvalStatsCollection):
    def __init__(self, evalStatsList: List[ClassificationEvalStats]):
        super().__init__(evalStatsList)
