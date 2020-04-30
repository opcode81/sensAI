import logging
import os
from typing import Sequence, Optional, Tuple

import matplotlib.figure
from matplotlib import pyplot as plt
import pandas as pd


_log = logging.getLogger(__name__)


class ResultWriter:
    _log = _log.getChild(__qualname__)

    def __init__(self, resultDir, filenamePrefix=""):
        self.resultDir = resultDir
        os.makedirs(resultDir, exist_ok=True)
        self.filenamePrefix = filenamePrefix

    def childWithAddedPrefix(self, prefix) -> "ResultWriter":
        """
        Creates a derived result writer with an added prefix, i.e. the given prefix is appended to this
        result writer's prefix

        :param prefix: the prefix to append
        :return: a new writer instance
        """
        return ResultWriter(self.resultDir, filenamePrefix=self.filenamePrefix + prefix)

    def path(self, filenameSuffix: str, extensionToAdd=None, validOtherExtensions: Optional[Sequence[str]] = None):
        """
        :param filenameSuffix: the suffix to add (which may or may not already include the file extension "txt", which
            will be added if it is not already present)
        :param extensionToAdd: if not None, the file extension to add (without the leading ".") unless
            the extension to add or one of the extenions in validExtensions is already present
        :param validOtherExtensions: a sequence of valid other extensions (without the "."), only
            relevant if extensionToAdd is specified
        :return: the full path
        """
        if extensionToAdd is not None:
            addExt = True
            validExtensions = set(validOtherExtensions) if validOtherExtensions is not None else set()
            validExtensions.add(extensionToAdd)
            if validExtensions is not None:
                for ext in validExtensions:
                    if filenameSuffix.endswith("." + ext):
                        addExt = False
                        break
            if addExt:
                filenameSuffix += "." + extensionToAdd
        path = os.path.join(self.resultDir, f"{self.filenamePrefix}{filenameSuffix}")
        return path

    def writeTextFile(self, filenameSuffix, content):
        p = self.path(filenameSuffix, extensionToAdd="txt")
        self._log.info(f"Saving text file {p}")
        with open(p, "w") as f:
            f.write(content)
        return p

    def writeDataFrameTextFile(self, filenameSuffix, df: pd.DataFrame):
        p = self.path(filenameSuffix, extensionToAdd="df.txt", validOtherExtensions="txt")
        self._log.info(f"Saving data frame text file {p}")
        with open(p, "w") as f:
            f.write(df.to_string())
        return p

    def writeFigure(self, filenameSuffix, fig, closeFigure=False):
        """
        :param filenameSuffix: the filename suffix, which may or may not include a file extension, valid extensions being {"png", "jpg"}
        :param fig: the figure to save
        :param closeFigure: whether to close the figure after having saved it
        :return: the path to the file that was written
        """
        p = self.path(filenameSuffix, extensionToAdd="png", validOtherExtensions=("jpg",))
        self._log.info(f"Saving figure {p}")
        fig.savefig(p)
        if closeFigure:
            plt.close(fig)
        return p

    def writeFigures(self, figures: Sequence[Tuple[str, matplotlib.figure.Figure]], closeFigures=False):
        for name, fig in figures:
            self.writeFigure(name, fig, closeFigure=closeFigures)