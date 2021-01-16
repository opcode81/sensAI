from typing import Callable, Any

import pandas as pd


class DataFrameHistoryTracker:
    """
    A simple class for keeping track of changes in data frame metadata (like column names, and indices).
    The idea is to call update on the result of methods that transform data frames

    Example:

    >>> from sensai.util.metadata import trackDFHistory, DataFrameHistoryTracker
    >>> import pandas as pd

    >>> df = pd.DataFrame({"bar": [1, 2]})
    >>> dfHistory = DataFrameHistoryTracker(df)
    >>> df["foo"] = [4, 5]
    >>> df.index = ["first", "second"]
    >>> dfHistory.update(df)
    >>> dfHistory.indexHistory
    [RangeIndex(start=0, stop=2, step=1), Index(['first', 'second'], dtype='object')]
    >>> dfHistory.columnsHistory
    [Index(['bar'], dtype='object'), Index(['bar', 'foo'], dtype='object')]
    """
    def __init__(self, df: pd.DataFrame):
        self.numUpdates = -1  # after init this will be 0, so init does not count as update
        self.columnsHistory = []
        self.indexHistory = []
        self.update(df)

    def update(self, df: pd.DataFrame):
        self.numUpdates += 1
        self.columnsHistory.append(df.columns)
        self.indexHistory.append(df.index)


def trackDFHistory(dfTransform: Callable[[Any, pd.DataFrame, Any], pd.DataFrame], historyAttributeName="_dfHistory"):
    """
    Decorator for tracking the change of (metadata of) a data frame by instance methods.
    The history of the data frame will be saved to the selected instance attribute. For safety reasons,
    the instance attribute has to exist prior to execution of the annotated method.

    Example:

    >>> from sensai.util.metadata import trackDFHistory, DataFrameHistoryTracker
    >>> import pandas as pd

    >>> class Transformer:
    ...     def __init__(self):
    ...         self._dfHistory = None
    ...
    ...     @property
    ...     def dfHistory(self) -> DataFrameHistoryTracker:
    ...         return self._dfHistory
    ...
    ...     @trackDFHistory
    ...     def apply(self, df):
    ...         return pd.DataFrame({"bar": [1, 2]})

    >>> transformer = Transformer()
    >>> df = transformer.apply(pd.DataFrame({"foo": [1, 2, 3]}))
    >>> transformer.dfHistory.columnsHistory
    [Index(['foo'], dtype='object'), Index(['bar'], dtype='object')]


    """
    def trackedTransform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        if historyAttributeName not in self.__dict__:
            raise Exception(f"Data frame history can be persisted only to existing attributes. "
                            f"No such attribute {historyAttributeName} in instance of {self.__class__.__name__}")
        setattr(self, historyAttributeName, DataFrameHistoryTracker(df))
        df = dfTransform(self, df, *args, **kwargs)
        getattr(self, historyAttributeName).update(df)
        return df
    return trackedTransform
