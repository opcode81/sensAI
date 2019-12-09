from typing import List, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .basic_models_base import DataFrameTransformer, RuleBasedDataFrameTransformer


class DFTOneHotEncoder(DataFrameTransformer):
    def __init__(self, columns: List[str], categories: List[np.ndarray]):
        """
        One hot encode categorical variables
        :param columns: names of original columns that are to be replaced by a list one-hot encoded columns each
        :param categories: numpy arrays containing the possible values of each of the specified columns
        """
        if len(columns) != len(categories):
            raise ValueError(f"Length of categories is not the same as length of columnNamesToProcess")
        self.columnNamesToProcess = columns
        categories = [np.sort(column_categories) for column_categories in categories]
        self.oneHotEncoders = [OneHotEncoder(categories=[categories[i]], sparse=False) for i in range(len(columns))]

    def fit(self, df: pd.DataFrame):
        for encoder, columnName in zip(self.oneHotEncoders, self.columnNamesToProcess):
            encoder.fit(df[[columnName]])

    def apply(self, df: pd.DataFrame):
        df = df.copy()
        for encoder, columnName in zip(self.oneHotEncoders, self.columnNamesToProcess):
            encodedArray = encoder.transform(df[[columnName]])
            df = df.drop(columns=columnName)
            for i in range(encodedArray.shape[1]):
                df["%s_%d" % (columnName, i)] = encodedArray[:, i]
        return df


class DFTColumnFilter(RuleBasedDataFrameTransformer):
    """
    A DataFrame transformer that filters columns by retaining or dropping specified columns
    """

    def __init__(self, keep: Union[str, Sequence[str]] = None, drop: Union[str, Sequence[str]] = None):
        self.keep = [keep] if type(keep) == str else keep
        self.drop = drop

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.keep is not None:
            df = df[self.keep]
        if self.drop is not None:
            df = df.drop(columns=self.drop)
        return df
