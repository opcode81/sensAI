import logging
import numpy as np
import seaborn as sns
from abc import abstractmethod, ABC
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Sequence, Optional

from .eval_stats_base import PredictionEvalStats, Metric, EvalStatsCollection, PredictionArray

log = logging.getLogger(__name__)


class RegressionMetric(Metric["RegressionEvalStats"], ABC):
    def computeValueForEvalStats(self, evalStats: "RegressionEvalStats"):
        return self.computeValue(np.array(evalStats.y_true), np.array(evalStats.y_predicted))

    @classmethod
    @abstractmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        pass

    @classmethod
    def computeErrors(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return y_predicted - y_true

    @classmethod
    def computeAbsErrors(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return np.abs(cls.computeErrors(y_true, y_predicted))


class RegressionMetricMAE(RegressionMetric):
    name = "MAE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return np.mean(cls.computeAbsErrors(y_true, y_predicted))


class RegressionMetricMSE(RegressionMetric):
    name = "MSE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        residuals = y_predicted - y_true
        return np.sum(residuals * residuals) / len(residuals)


class RegressionMetricRMSE(RegressionMetric):
    name = "RMSE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        errors = cls.computeErrors(y_true, y_predicted)
        return np.sqrt(np.mean(errors * errors))


class RegressionMetricRRSE(RegressionMetric):
    name = "RRSE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        mean_y = np.mean(y_true)
        residuals = y_predicted - y_true
        mean_deviation = y_true - mean_y
        return np.sqrt(np.sum(residuals * residuals) / np.sum(mean_deviation * mean_deviation))


class RegressionMetricR2(RegressionMetric):
    name = "R2"

    def computeValue(self, y_true: np.ndarray, y_predicted: np.ndarray):
        rrse = RegressionMetricRRSE.computeValue(y_true, y_predicted)
        return 1.0 - rrse*rrse


class RegressionMetricPCC(RegressionMetric):
    name = "PCC"

    def computeValue(self, y_true: np.ndarray, y_predicted: np.ndarray):
        cov = np.cov([y_true, y_predicted])
        return cov[0][1] / np.sqrt(cov[0][0] * cov[1][1])


class RegressionMetricStdDevAE(RegressionMetric):
    name = "StdDevAE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return np.std(cls.computeAbsErrors(y_true, y_predicted))


class RegressionMetricMedianAE(RegressionMetric):
    name = "MedianAE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return np.median(cls.computeAbsErrors(y_true, y_predicted))


class RegressionEvalStats(PredictionEvalStats["RegressionMetric"]):
    """
    Collects data for the evaluation of predicted continuous values and computes corresponding metrics
    """
    def __init__(self, y_predicted: PredictionArray, y_true: PredictionArray,
            metrics: Sequence["RegressionMetric"] = None, additionalMetrics: Sequence["RegressionMetric"] = None):
        """
        :param y_predicted: the predicted values
        :param y_true: the true values
        :param metrics: the metrics to compute for evaluation; if None, use default metrics
        :param additionalMetrics: the metrics to additionally compute
        """

        if metrics is None:
            metrics = [RegressionMetricRRSE(), RegressionMetricR2(), RegressionMetricPCC(),
                       RegressionMetricMAE(), RegressionMetricMSE(), RegressionMetricRMSE(),
                       RegressionMetricStdDevAE()]
        metrics = list(metrics)

        super().__init__(y_predicted, y_true, metrics, additionalMetrics=additionalMetrics)

    def getMSE(self):
        return self.computeMetricValue(RegressionMetricMSE())

    def getRRSE(self):
        """Gets the root relative squared error"""
        return self.computeMetricValue(RegressionMetricRRSE())

    def getCorrelationCoeff(self):
        """Gets the Pearson correlation coefficient (PCC)"""
        return self.computeMetricValue(RegressionMetricPCC())

    def getR2(self):
        """Gets the R^2 score"""
        return self.computeMetricValue(RegressionMetricR2())

    def getMAE(self):
        """Gets the mean absolute error"""
        return self.computeMetricValue(RegressionMetricMAE())

    def getRMSE(self):
        """Gets the root mean squared error"""
        return self.computeMetricValue(RegressionMetricRMSE())

    def getStdDevAE(self):
        """Gets the standard deviation of the absolute error"""
        return self.computeMetricValue(RegressionMetricStdDevAE())

    def getEvalStatsCollection(self) -> "RegressionEvalStatsCollection":
        """
        For the case where we collected data on multiple dimensions, obtain a stats collection where
        each object in the collection holds stats on just one dimension
        """
        if self.y_true_multidim is None:
            raise Exception("No multi-dimensional data was collected")
        dim = len(self.y_true_multidim)
        statsList = []
        for i in range(dim):
            stats = RegressionEvalStats(self.y_predicted_multidim[i], self.y_true_multidim[i])
            statsList.append(stats)
        return RegressionEvalStatsCollection(statsList)

    def plotErrorDistribution(self, bins=None, figure=True, titleAdd=None) -> Optional[plt.Figure]:
        """
        :param bins: if None, seaborns default binning will be used
        :param figure: whether to plot in a separate figure and return that figure
        :param titleAdd: a string to add to the title (on a second line)

        :return: the resulting figure object or None
        """
        errors = np.array(self.y_predicted) - np.array(self.y_true)
        fig = None
        title = "Prediction Error Distribution"
        if titleAdd is not None:
            title += "\n" + titleAdd
        if figure:
            fig = plt.figure(title.replace("\n", " "))
        sns.distplot(errors, bins=bins)
        plt.title(title)
        plt.xlabel("error (prediction - ground truth)")
        plt.ylabel("probability density")
        return fig

    def plotScatterGroundTruthPredictions(self, figure=True, titleAdd=None, **kwargs) -> Optional[plt.Figure]:
        """
        :param figure: whether to plot in a separate figure and return that figure
        :param titleAdd: a string to be added to the title in a second line
        :param kwargs: parameters to be passed on to plt.scatter()

        :return:  the resulting figure object or None
        """
        fig = None
        title = "Scatter Plot of Ground Truth vs. Predicted Values"
        if titleAdd is not None:
            title += "\n" + titleAdd
        if figure:
            fig = plt.figure(title.replace("\n", " "))
        y_range = [min(self.y_true), max(self.y_true)]
        plt.scatter(self.y_true, self.y_predicted, **kwargs)
        plt.plot(y_range, y_range, 'k-', lw=2, label="_not in legend", color="r")
        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.title(title)
        return fig

    def plotHeatmapGroundTruthPredictions(self, figure=True, cmap=None, bins=60, titleAdd=None, **kwargs) -> Optional[plt.Figure]:
        """
        :param figure: whether to plot in a separate figure and return that figure
        :param cmap: the colour map to use (see corresponding parameter of plt.imshow); if None use colour map from white to red
        :param bins: how many bins to use for constructing the heatmap
        :param titleAdd: a string to add to the title (on a second line)
        :param kwargs: will be passed to plt.imshow()

        :return:  the resulting figure object or None
        """
        fig = None
        title = "Heat Map of Ground Truth vs. Predicted Values"
        if titleAdd:
            title += "\n" + titleAdd
        if figure:
            fig = plt.figure(title.replace("\n", " "))
        y_range = [min(min(self.y_true), min(self.y_predicted)), max(max(self.y_true), max(self.y_predicted))]
        plt.plot(y_range, y_range, 'k-', lw=0.75, label="_not in legend", color="green", zorder=2)
        heatmap, _, _ = np.histogram2d(self.y_true, self.y_predicted, range=[y_range, y_range], bins=bins)
        extent = [y_range[0], y_range[1], y_range[0], y_range[1]]
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list("whiteToRed", ((0, (1, 1, 1)), (1/len(self.y_predicted), (1.0, 0.95, 0.95)), (1, (0.7, 0, 0))))
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, zorder=1, **kwargs)

        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.title(title)
        return fig


class RegressionEvalStatsCollection(EvalStatsCollection):
    def __init__(self, evalStatsList: List[RegressionEvalStats]):
        super().__init__(evalStatsList)
        self.globalStats = None

    def getGlobalStats(self) -> RegressionEvalStats:
        """
        Gets an evaluation statistics object that combines the data from all contained eval stats objects
        """
        if self.globalStats is None:
            y_true = np.concatenate([evalStats.y_true for evalStats in self.statsList])
            y_predicted = np.concatenate([evalStats.y_predicted for evalStats in self.statsList])
            self.globalStats = RegressionEvalStats(y_predicted, y_true)
        return self.globalStats
