import logging
from abc import abstractmethod, ABC
from typing import List, Sequence, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from . import BinaryClassificationMetric
from .eval_stats_base import PredictionEvalStats, Metric, EvalStatsCollection, PredictionArray, EvalStatsPlot, Array
from ...util import kwarg_if_not_none
from ...util.plot import HistogramPlot
from ...vector_model import VectorRegressionModel, InputOutputData

log = logging.getLogger(__name__)


class RegressionMetric(Metric["RegressionEvalStats"], ABC):
    def compute_value_for_eval_stats(self, eval_stats: "RegressionEvalStats"):
        weights = np.array(eval_stats.weights) if eval_stats.weights is not None else None
        return self.compute_value(np.array(eval_stats.y_true), np.array(eval_stats.y_predicted),
            model=eval_stats.model,
            io_data=eval_stats.ioData,
            **kwarg_if_not_none("weights", weights))

    @abstractmethod
    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray,
            model: VectorRegressionModel = None,
            io_data: InputOutputData = None,
            weights: Optional[np.ndarray] = None):
        pass

    @classmethod
    def compute_errors(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return y_predicted - y_true

    @classmethod
    def compute_abs_errors(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return np.abs(cls.compute_errors(y_true, y_predicted))


class RegressionMetricMAE(RegressionMetric):
    name = "MAE"

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        return mean_absolute_error(y_true, y_predicted, sample_weight=weights)


class RegressionMetricMSE(RegressionMetric):
    name = "MSE"

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        return mean_squared_error(y_true, y_predicted, sample_weight=weights)


class RegressionMetricRMSE(RegressionMetric):
    name = "RMSE"

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        return np.sqrt(mean_squared_error(y_true, y_predicted, sample_weight=weights))


class RegressionMetricRRSE(RegressionMetric):
    name = "RRSE"

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        r2 = r2_score(y_true, y_predicted, sample_weight=weights)
        return np.sqrt(1 - r2)


class RegressionMetricR2(RegressionMetric):
    name = "R2"

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        return r2_score(y_true, y_predicted, sample_weight=weights)


class RegressionMetricPCC(RegressionMetric):
    """
    Pearson's correlation coefficient, aka Pearson's R.
    This metric does not consider sample weights.
    """
    name = "PCC"

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        cov = np.cov([y_true, y_predicted])
        return cov[0][1] / np.sqrt(cov[0][0] * cov[1][1])


class RegressionMetricStdDevAE(RegressionMetric):
    """
    The standard deviation of the absolute error.
    This metric does not consider sample weights.
    """

    name = "StdDevAE"

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        return np.std(self.compute_abs_errors(y_true, y_predicted))


class RegressionMetricMedianAE(RegressionMetric):
    """
    The median absolute error.
    This metric does not consider sample weights.
    """
    name = "MedianAE"

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        return np.median(self.compute_abs_errors(y_true, y_predicted))


class RegressionMetricFromBinaryClassificationMetric(RegressionMetric):
    """
    Supports the computation of binary classification metrics by converting predicted/target values to class labels.
    This metric does not consider sample weights.
    """

    class ClassGenerator(ABC):
        @abstractmethod
        def compute_class(self, predicted_value: float) -> bool:
            """
            Computes the class from the given value

            :param predicted_value: the value predicted by the regressor or regressor target value
            :return: the class
            """
            pass

        @abstractmethod
        def get_metric_qualifier(self) -> str:
            """
            :return: A (short) string which will be added to the original classification metric's name to
                represent the class conversion logic
            """
            pass

    class ClassGeneratorPositiveBeyond(ClassGenerator):
        def __init__(self, min_value_for_positive: float):
            self.min_value_for_positive = min_value_for_positive

        def compute_class(self, predicted_value: float) -> bool:
            return predicted_value >= self.min_value_for_positive

        def get_metric_qualifier(self) -> str:
            return f">={self.min_value_for_positive}"

    def __init__(self, classification_metric: BinaryClassificationMetric,
            class_generator: ClassGenerator):
        """
        :param classification_metric: the classification metric (which shall consider `True` as the positive label)
        :param class_generator: the class generator, which generates `True` and `False` labels from regression values
        """
        super().__init__(name=classification_metric.name + f"[{class_generator.get_metric_qualifier()}]",
            bounds=classification_metric.bounds)
        self.classification_metric = classification_metric
        self.class_generator = class_generator

    def _apply_class_generator(self, y: np.ndarray) -> np.ndarray:
        return np.array([self.class_generator.compute_class(v) for v in y])

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None, weights: Optional[np.ndarray] = None):
        y_true = self._apply_class_generator(y_true)
        y_predicted = self._apply_class_generator(y_predicted)
        return self.classification_metric.compute_value(y_true=y_true, y_predicted=y_predicted)


class HeatMapColorMapFactory(ABC):
    @abstractmethod
    def create_color_map(self, min_sample_weight: float, total_weight: float, num_quantization_levels: int):
        pass


class HeatMapColorMapFactoryWhiteToRed(HeatMapColorMapFactory):
    def create_color_map(self, min_sample_weight: float, total_weight: float, num_quantization_levels: int):
        color_nothing = (1, 1, 1)  # white
        color_min_sample = (1, 0.96, 0.96)  # very slightly red
        color_everything = (0.7, 0, 0)  # dark red
        return LinearSegmentedColormap.from_list("whiteToRed",
            ((0, color_nothing), (min_sample_weight/total_weight, color_min_sample), (1, color_everything)),
            num_quantization_levels)


DEFAULT_REGRESSION_METRICS = (RegressionMetricRRSE(), RegressionMetricR2(), RegressionMetricMAE(),
        RegressionMetricMSE(), RegressionMetricRMSE(), RegressionMetricStdDevAE())


class RegressionEvalStats(PredictionEvalStats["RegressionMetric"]):
    """
    Collects data for the evaluation of predicted continuous values and computes corresponding metrics
    """

    # class members controlling plot appearance, which can be centrally overridden by a user if necessary
    HEATMAP_COLORMAP_FACTORY = HeatMapColorMapFactoryWhiteToRed()
    HEATMAP_DIAGONAL_COLOR = "green"
    HEATMAP_ERROR_BOUNDARY_VALUE = None
    HEATMAP_ERROR_BOUNDARY_COLOR = (0.8, 0.8, 0.8)
    SCATTER_PLOT_POINT_COLOR = (0, 0, 1, 0.05)

    def __init__(self, y_predicted: Optional[PredictionArray] = None, y_true: Optional[PredictionArray] = None,
            metrics: Optional[Sequence["RegressionMetric"]] = None, additional_metrics: Sequence["RegressionMetric"] = None,
            model: VectorRegressionModel = None,
            io_data: InputOutputData = None,
            weights: Optional[Array] = None):
        """
        :param y_predicted: the predicted values
        :param y_true: the true values
        :param metrics: the metrics to compute for evaluation; if None, will use DEFAULT_REGRESSION_METRICS
        :param additional_metrics: the metrics to additionally compute
        :param weights: optional data point weights
        """
        self.model = model
        self.ioData = io_data

        if metrics is None:
            metrics = DEFAULT_REGRESSION_METRICS
        metrics = list(metrics)

        super().__init__(y_predicted, y_true, metrics, additional_metrics=additional_metrics, weights=weights)

    def compute_metric_value(self, metric: RegressionMetric) -> float:
        return metric.compute_value_for_eval_stats(self)

    def compute_mse(self):
        """Computes the mean squared error (MSE)"""
        return self.compute_metric_value(RegressionMetricMSE())

    def compute_rrse(self):
        """Computes the root relative squared error"""
        return self.compute_metric_value(RegressionMetricRRSE())

    def compute_pcc(self):
        """Gets the Pearson correlation coefficient (PCC)"""
        return self.compute_metric_value(RegressionMetricPCC())

    def compute_r2(self):
        """Gets the R^2 score"""
        return self.compute_metric_value(RegressionMetricR2())

    def compute_mae(self):
        """Gets the mean absolute error"""
        return self.compute_metric_value(RegressionMetricMAE())

    def compute_rmse(self):
        """Gets the root mean squared error"""
        return self.compute_metric_value(RegressionMetricRMSE())

    def compute_std_dev_ae(self):
        """Gets the standard deviation of the absolute error"""
        return self.compute_metric_value(RegressionMetricStdDevAE())

    def create_eval_stats_collection(self) -> "RegressionEvalStatsCollection":
        """
        For the case where we collected data on multiple dimensions, obtain a stats collection where
        each object in the collection holds stats on just one dimension
        """
        if self.y_true_multidim is None:
            raise Exception("No multi-dimensional data was collected")
        dim = len(self.y_true_multidim)
        stats_list = []
        for i in range(dim):
            stats = RegressionEvalStats(self.y_predicted_multidim[i], self.y_true_multidim[i])
            stats_list.append(stats)
        return RegressionEvalStatsCollection(stats_list)

    def plot_error_distribution(self, bins="auto", title_add=None) -> Optional[plt.Figure]:
        """
        :param bins: bin specification (see :class:`HistogramPlot`)
        :param title_add: a string to add to the title (on a second line)

        :return: the resulting figure object or None
        """
        errors = np.array(self.y_predicted) - np.array(self.y_true)
        title = "Prediction Error Distribution"
        if title_add is not None:
            title += "\n" + title_add
        if bins == "auto" and len(errors) < 100:
            bins = 10  # seaborn can crash with low number of data points and bins="auto" (tries to allocate vast amounts of memory)
        plot = HistogramPlot(errors, bins=bins, kde=True)
        plot.title(title)
        plot.xlabel("error (prediction - ground truth)")
        plot.ylabel("probability density")
        return plot.fig

    def plot_scatter_ground_truth_predictions(self, figure=True, title_add=None, **kwargs) -> Optional[plt.Figure]:
        """
        :param figure: whether to plot in a separate figure and return that figure
        :param title_add: a string to be added to the title in a second line
        :param kwargs: parameters to be passed on to plt.scatter()

        :return:  the resulting figure object or None
        """
        fig = None
        title = "Scatter Plot of Predicted Values vs. Ground Truth"
        if title_add is not None:
            title += "\n" + title_add
        if figure:
            fig = plt.figure(title.replace("\n", " "))
        y_range = [min(self.y_true), max(self.y_true)]
        plt.scatter(self.y_true, self.y_predicted, c=[self.SCATTER_PLOT_POINT_COLOR], zorder=2, **kwargs)
        plt.plot(y_range, y_range, '-', lw=1, label="_not in legend", color="green", zorder=1)
        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.title(title)
        return fig

    def plot_heatmap_ground_truth_predictions(self, figure=True, cmap=None, bins=60, title_add=None, error_boundary: Optional[float] = None,
            weighted: bool = False, ax: Optional[plt.Axes] = None,
            **kwargs) -> Optional[plt.Figure]:
        """
        :param figure: whether to create a new figure and return that figure (only applies if ax is None)
        :param cmap: the colour map to use (see corresponding parameter of plt.imshow for further information); if None, use factory
            defined in HEATMAP_COLORMAP_FACTORY (which can be centrally set to achieve custom behaviour throughout an application)
        :param bins: how many bins to use for constructing the heatmap
        :param title_add: a string to add to the title (on a second line)
        :param error_boundary: if not None, add two lines (above and below the diagonal) indicating this absolute regression error boundary;
            if None (default), use static member HEATMAP_ERROR_BOUNDARY_VALUE (which is also None by default, but can be centrally set
            to achieve custom behaviour throughout an application)
        :param weighted: whether to consider data point weights
        :param ax: the axis to plot in. If None, use the current axes (which will be the axis of the newly created figure if figure=True).
        :param kwargs: will be passed to plt.imshow()

        :return:  the newly created figure object (if figure=True) or None
        """
        fig = None
        title = "Heat Map of Predicted Values vs. Ground Truth"
        if title_add:
            title += "\n" + title_add
        if figure and ax is None:
            fig = plt.figure(title.replace("\n", " "))
        if ax is None:
            ax = plt.gca()

        y_range = [min(min(self.y_true), min(self.y_predicted)), max(max(self.y_true), max(self.y_predicted))]

        # diagonal
        ax.plot(y_range, y_range, '-', lw=0.75, label="_not in legend", color=self.HEATMAP_DIAGONAL_COLOR, zorder=2)

        # error boundaries
        if error_boundary is None:
            error_boundary = self.HEATMAP_ERROR_BOUNDARY_VALUE
        if error_boundary is not None:
            d = np.array(y_range)
            offs = np.array([error_boundary, error_boundary])
            ax.plot(d, d + offs, '-', lw=0.75, label="_not in legend", color=self.HEATMAP_ERROR_BOUNDARY_COLOR, zorder=2)
            ax.plot(d, d - offs, '-', lw=0.75, label="_not in legend", color=self.HEATMAP_ERROR_BOUNDARY_COLOR, zorder=2)

        # heat map
        weights = None if not weighted else self.weights
        heatmap, _, _ = np.histogram2d(self.y_true, self.y_predicted, range=(y_range, y_range), bins=bins, density=False, weights=weights)
        extent = (y_range[0], y_range[1], y_range[0], y_range[1])
        if cmap is None:
            num_quantization_levels = min(1000, len(self.y_predicted))
            if not weighted:
                min_sample_weight = 1.0
                total_weight = len(self.y_predicted)
            else:
                min_sample_weight = np.min(self.weights)
                total_weight = np.sum(self.weights)
            cmap = self.HEATMAP_COLORMAP_FACTORY.create_color_map(min_sample_weight, total_weight, num_quantization_levels)
        ax.imshow(heatmap.T, extent=extent, origin='lower', interpolation="none", cmap=cmap, zorder=1, **kwargs)

        ax.set_xlabel("ground truth")
        ax.set_ylabel("prediction")
        ax.set_title(title)
        return fig


class RegressionEvalStatsCollection(EvalStatsCollection[RegressionEvalStats, RegressionMetric]):
    def __init__(self, eval_stats_list: List[RegressionEvalStats]):
        super().__init__(eval_stats_list)
        self.globalStats = None

    def get_combined_eval_stats(self) -> RegressionEvalStats:
        if self.globalStats is None:
            y_true = np.concatenate([evalStats.y_true for evalStats in self.statsList])
            y_predicted = np.concatenate([evalStats.y_predicted for evalStats in self.statsList])
            es0 = self.statsList[0]
            self.globalStats = RegressionEvalStats(y_predicted, y_true, metrics=es0.metrics)
        return self.globalStats


class RegressionEvalStatsPlot(EvalStatsPlot[RegressionEvalStats], ABC):
    pass


class RegressionEvalStatsPlotErrorDistribution(RegressionEvalStatsPlot):
    def create_figure(self, eval_stats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return eval_stats.plot_error_distribution(title_add=subtitle)


class RegressionEvalStatsPlotHeatmapGroundTruthPredictions(RegressionEvalStatsPlot):
    def __init__(self, weighted: bool = False):
        self.weighted = weighted

    def is_applicable(self, eval_stats: RegressionEvalStats) -> bool:
        if self.weighted:
            return eval_stats.weights is not None
        else:
            return True

    def create_figure(self, eval_stats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return eval_stats.plot_heatmap_ground_truth_predictions(title_add=subtitle, weighted=self.weighted)


class RegressionEvalStatsPlotScatterGroundTruthPredictions(RegressionEvalStatsPlot):
    def create_figure(self, eval_stats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return eval_stats.plot_scatter_ground_truth_predictions(title_add=subtitle)
