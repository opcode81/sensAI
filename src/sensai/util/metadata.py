import pandas as pd


class DataFrameHistoryTracker:
    """
    A simple class for keeping track of changes in data frame metadata (like column names, and indices).
    The idea is to call update on the result of methods that transform data frames

    Example:

    >>> from sensai.util.metadata import DataFrameHistoryTracker
    >>> import pandas as pd

    >>> df = pd.DataFrame({"bar": [1, 2]})
    >>> dfHistory = DataFrameHistoryTracker(df, trackIndices=True)
    >>> df["foo"] = [4, 5]
    >>> df.index = ["first", "second"]
    >>> dfHistory.update(df)
    >>> dfHistory.indexHistory
    [RangeIndex(start=0, stop=2, step=1), Index(['first', 'second'], dtype='object')]
    >>> dfHistory.columnsHistory
    [Index(['bar'], dtype='object'), Index(['bar', 'foo'], dtype='object')]
    """
    def __init__(self, df: pd.DataFrame, trackIndices=False):
        self.numUpdates = -1  # after init this will be 0, so init does not count as update
        self.trackIndices = trackIndices
        self.columnsHistory = []
        self.indexHistory = [] if trackIndices else None
        self.update(df)

    def update(self, df: pd.DataFrame):
        self.numUpdates += 1
        self.columnsHistory.append(df.columns)
        if self.trackIndices:
            self.indexHistory.append(df.index)
