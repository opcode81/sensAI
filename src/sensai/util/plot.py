import logging
from matplotlib.colors import LinearSegmentedColormap
from typing import Sequence, Callable

import matplotlib.figure
from matplotlib import pyplot as plt
import numpy as np


log = logging.getLogger(__name__)


def plotMatrix(matrix, title, xticklabels: Sequence[str], yticklabels: Sequence[str], xlabel: str, ylabel: str, normalize=True, figsize=(9,9),
        titleAdd: str = None) -> matplotlib.figure.Figure:
    """
    :param matrix: matrix whose data to plot, where matrix[i, j] will be rendered at x=i, y=j
    :param title: the plot's title
    :param xticklabels: the labels for the x-axis ticks
    :param yticklabels: the labels for the y-axis ticks
    :param xlabel: the label for the x-axis
    :param ylabel: the label for the y-axis
    :param normalize: whether to normalise the matrix before plotting it (dividing each entry by the sum of all entries)
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
    fmt = '.2f' if normalize else ('.2f' if matrix.dtype == np.float else 'd')
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


class Plot:
    def __init__(self, draw: Callable[[], plt.Axes] = None, name=None):
        """
        :param draw: function which returns a matplotlib.Axes object to show
        :param name: name/number of the figure, which determines the window caption; it should be unique, as any plot
            with the same name will have its contents rendered in the same window. By default, figures are number
            sequentially.
        """
        self.fig: matplotlib.figure.Figure = plt.figure(name)
        self.ax = draw()

    def xlabel(self, label):
        plt.xlabel(label)
        return self

    def ylabel(self, label):
        plt.ylabel(label)
        return self

    def save(self, path):
        log.info(f"Saving figure in {path}")
        self.fig.savefig(path)


class ScatterPlot(Plot):
    def __init__(self, x, y, **kwargs):
        super().__init__(lambda: plt.scatter(x, y, **kwargs))


class HeatMapPlot(Plot):
    def __init__(self, x, y, bins=60, cmap=None, **kwargs):

        def draw():
            nonlocal cmap
            x_range = [min(x), max(x)]
            y_range = [min(y), max(y)]
            heatmap, _, _ = np.histogram2d(x, y, range=[x_range, y_range], bins=bins)
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
            if cmap is None:
                cmap = LinearSegmentedColormap.from_list("whiteToRed", ((1, 1, 1), (0.7, 0, 0)))
            return plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, zorder=1, aspect="auto", **kwargs)

        super().__init__(draw)