from copy import copy

import pandas as pd


class DataFrameColumnChangeTracker:
    """
    A simple class for keeping track of changes in columns between an initial data frame and some other data frame
    (usually the result of some transformations performed on the initial one).

    Example:

    >>> from sensai.util.pandas import DataFrameColumnChangeTracker
    >>> import pandas as pd

    >>> df = pd.DataFrame({"bar": [1, 2]})
    >>> columnChangeTracker = DataFrameColumnChangeTracker(df)
    >>> df["foo"] = [4, 5]
    >>> columnChangeTracker.trackChange(df)
    >>> columnChangeTracker.getRemovedColumns()
    set()
    >>> columnChangeTracker.getAddedColumns()
    {'foo'}
    """
    def __init__(self, initialDF: pd.DataFrame):
        self.initialColumns = copy(initialDF.columns)
        self.finalColumns = None

    def trackChange(self, changedDF: pd.DataFrame):
        self.finalColumns = copy(changedDF.columns)

    def getRemovedColumns(self):
        self.assertChangeWasTracked()
        return set(self.initialColumns).difference(self.finalColumns)

    def getAddedColumns(self):
        """
        Returns the columns in the last entry of the history that were not present the first one
        """
        self.assertChangeWasTracked()
        return set(self.finalColumns).difference(self.initialColumns)

    def columnChangeString(self):
        """
        Returns a string representation of the change
        """
        self.assertChangeWasTracked()
        if list(self.initialColumns) == list(self.finalColumns):
            return "none"
        removedCols, addedCols = self.getRemovedColumns(), self.getAddedColumns()
        if removedCols == addedCols == set():
            return f"reordered {list(self.finalColumns)}"

        return f"added={list(addedCols)}, removed={list(removedCols)}"

    def assertChangeWasTracked(self):
        if self.finalColumns is None:
            raise Exception(f"No change was tracked yet. "
                            f"Did you forget to call trackChange on the resulting data frame?")
