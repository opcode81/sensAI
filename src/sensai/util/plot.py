import logging
from matplotlib.colors import LinearSegmentedColormap
from typing import Sequence, Callable, TypeVar, Type, Tuple, Optional

import matplotlib.ticker as plticker
import matplotlib.figure
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


log = logging.getLogger(__name__)

MATPLOTLIB_DEFAULT_FIGURE_SIZE = (6.4, 4.8)


def plotMatrix(matrix: np.ndarray, title: str, xticklabels: Sequence[str], yticklabels: Sequence[str], xlabel: str,
        ylabel: str, normalize=True, figsize: Tuple[int, int] = (9, 9), titleAdd: str = None) -> matplotlib.figure.Figure:
    """
    :param matrix: matrix whose data to plot, where matrix[i, j] will be rendered at x=i, y=j
    :param title: the plot's title
    :param xticklabels: the labels for the x-axis ticks
    :param yticklabels: the labels for the y-axis ticks
    :param xlabel: the label for the x-axis
    :param ylabel: the label for the y-axis
    :param normalize: whether to normalise the matrix before plotting it (dividing each entry by the sum of all entries)
    :param figsize: an optional size of the figure to be created
    :param titleAdd: an optional second line to add to the title
    :return: the figure object
    """
    matrix = np.transpose(matrix)

    if titleAdd is not None:
        title += f"\n {titleAdd} "

    if normalize:
        matrix = matrix.astype('float') / matrix.sum()
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.set_window_title(title.replace("\n", " "))
    # We want to show all ticks...
    ax.set(xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=xticklabels, yticklabels=yticklabels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel)
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' if normalize else ('.2f' if matrix.dtype.kind == 'f' else 'd')
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


TPlot = TypeVar("TPlot", bound="Plot")


class Plot:
    def __init__(self, draw: Callable[[], None] = None, name=None):
        """
        :param draw: function which returns a matplotlib.Axes object to show
        :param name: name/number of the figure, which determines the window caption; it should be unique, as any plot
            with the same name will have its contents rendered in the same window. By default, figures are number
            sequentially.
        """
        fig, ax = plt.subplots(num=name)
        self.fig: plt.Figure = fig
        self.ax: plt.Axes = ax
        draw()

    def xlabel(self: Type[TPlot], label):
        self.ax.set_xlabel(label)
        return self

    def ylabel(self: Type[TPlot], label) -> TPlot:
        self.ax.set_ylabel(label)
        return self

    def title(self: Type[TPlot], title: str) -> TPlot:
        self.ax.set_title(title)
        return self

    def xlim(self: Type[TPlot], minValue, maxValue) -> TPlot:
        self.ax.set_xlim(minValue, maxValue)
        return self

    def ylim(self: Type[TPlot], minValue, maxValue) -> TPlot:
        self.ax.set_ylim(minValue, maxValue)
        return self

    def save(self, path):
        log.info(f"Saving figure in {path}")
        self.fig.savefig(path)

    def xtick(self, major=None, minor=None):
        """
        Sets a tick on every integer multiple of the given base values.
        The major ticks are labelled, the minor ticks are not.

        :param major: the major tick base value
        :param minor: the minor tick base value
        :return: self
        """
        if major is not None:
            self.xtickMajor(major)
        if minor is not None:
            self.xtickMinor(minor)
        return self

    def xtickMajor(self, base):
        self.ax.xaxis.set_major_locator(plticker.MultipleLocator(base=base))
        return self

    def xtickMinor(self, base):
        self.ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=base))
        return self

    def ytickMajor(self, base):
        self.ax.yaxis.set_major_locator(plticker.MultipleLocator(base=base))
        return self



class ScatterPlot(Plot):
    N_MAX_TRANSPARENCY = 1000
    N_MIN_TRANSPARENCY = 100
    MAX_OPACITY = 0.5
    MIN_OPACITY = 0.05

    def __init__(self, x, y, c=None, c_base: Tuple[float, float, float]=(0, 0, 1), c_opacity=None, x_label=None, y_label=None, **kwargs):
        """
        :param x: the x values; if has name (e.g. pd.Series), will be used as axis label
        :param y: the y values; if has name (e.g. pd.Series), will be used as axis label
        :param c: the colour specification; if None, compose from ``c_base`` and ``c_opacity``
        :param c_base: the base colour as (R, G, B) floats
        :param c_opacity: the opacity; if None, automatically determine from number of data points
        :param x_label:
        :param y_label:
        :param kwargs:
        """
        if c is None:
            if c_base is None:
                c_base = (0, 0, 1)
            if c_opacity is None:
                n = len(x)
                if n > self.N_MAX_TRANSPARENCY:
                    transparency = 1
                elif n < self.N_MIN_TRANSPARENCY:
                    transparency = 0
                else:
                    transparency = (n - self.N_MIN_TRANSPARENCY) / (self.N_MAX_TRANSPARENCY - self.N_MIN_TRANSPARENCY)
                c_opacity = self.MIN_OPACITY + (self.MAX_OPACITY - self.MIN_OPACITY) * (1-transparency)
            c = ((*c_base, c_opacity),)

        assert len(x) == len(y)
        if x_label is None and hasattr(x, "name"):
            x_label = x.name
        if y_label is None and hasattr(y, "name"):
            y_label = y.name

        def draw():
            if x_label is not None:
                plt.xlabel(x_label)
            if x_label is not None:
                plt.ylabel(y_label)
            plt.scatter(x, y, c=c, **kwargs)

        super().__init__(draw)


class HeatMapPlot(Plot):
    DEFAULT_CMAP_FACTORY = lambda numPoints: LinearSegmentedColormap.from_list("whiteToRed", ((0, (1, 1, 1)), (1/numPoints, (1, 0.96, 0.96)), (1, (0.7, 0, 0))), numPoints)

    def __init__(self, x, y, xLabel=None, yLabel=None, bins=60, cmap=None, commonRange=True, diagonal=False,
            diagonalColor="green", **kwargs):
        assert len(x) == len(y)
        if xLabel is None and hasattr(x, "name"):
            xLabel = x.name
        if yLabel is None and hasattr(y, "name"):
            yLabel = y.name

        def draw():
            nonlocal cmap
            x_range = [min(x), max(x)]
            y_range = [min(y), max(y)]
            range = [min(x_range[0], y_range[0]), max(x_range[1], y_range[1])]
            if commonRange:
                x_range = y_range = range
            if diagonal:
                plt.plot(range, range, '-', lw=0.75, label="_not in legend", color=diagonalColor, zorder=2)
            heatmap, _, _ = np.histogram2d(x, y, range=[x_range, y_range], bins=bins, density=False)
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
            if cmap is None:
                cmap = HeatMapPlot.DEFAULT_CMAP_FACTORY(len(x))
            if xLabel is not None:
                plt.xlabel(xLabel)
            if yLabel is not None:
                plt.ylabel(yLabel)
            plt.imshow(heatmap.T, extent=extent, origin='lower', interpolation="none", cmap=cmap, zorder=1, aspect="auto", **kwargs)

        super().__init__(draw)


class HistogramPlot(Plot):
    def __init__(self, values, bins="auto", kde=False, cdf=False, cdfComplementary=False, binwidth=None, stat="probability", xlabel=None,
            **kwargs):

        def draw():
            sns.histplot(values, bins=bins, kde=kde, binwidth=binwidth, stat=stat, **kwargs)
            if cdf:
                if cdfComplementary or stat in ("count", "proportion", "probability"):
                    ecdfStat = "proportion" if stat == "probability" else stat  # same semantics but "probability" not understood by ecdfplot
                    if ecdfStat not in ("count", "proportion"):
                        raise ValueError(f"Complementary cdf (cdfComplementary=True) is only supported for stats 'count', 'proportion' and 'probability, got '{stat}'")
                    sns.ecdfplot(values, stat=ecdfStat, complementary=cdfComplementary, color="orange")
                else:
                    sns.histplot(values, bins=100, stat=stat, element="poly", fill=False, cumulative=True, color="orange")
            if xlabel is not None:
                plt.xlabel(xlabel)

        super().__init__(draw)

        if stat in ("proportion", "probability"):
            yTick = 0.1
        elif stat == "percent":
            yTick = 10
        else:
            yTick = None
        if yTick is not None:
            self.ytickMajor((yTick))
