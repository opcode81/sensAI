from abc import ABC, abstractmethod
import logging
from typing import Any, Union

import numpy as np
import pandas as pd

from .util.cache import PersistentKeyValueCache


log = logging.getLogger(__name__)


class ColumnGenerator:
    """
    Generates a single column (pd.Series) from an input data frame, which is to have the same index as the input
    """
    def __init__(self, columnName: str):
        """
        :param columnName: the name of the column being generated
        """
        self.columnName = columnName

    def generateColumn(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates a column from the input data frame

        :param df: the input data frame
        :return: the column as a named series, which has the same index as the input
        """
        result = self._generateColumn(df)
        if isinstance(result, pd.Series):
            result.name = self.columnName
        else:
            result = pd.Series(result, index=df.index, name=self.columnName)
        return result

    @abstractmethod
    def _generateColumn(self, df: pd.DataFrame) -> Union[pd.Series, list, np.ndarray]:
        """
        Performs the actual column generation

        :param df: the input data frame
        :return: a list/array of the same length as df or a series with the same index
        """
        pass


class IndexCachedColumnGenerator(ColumnGenerator):
    """
    Decorator for a column generator which adds support for cached column generation where cache keys are given by the input data frame's index.
    Entries not found in the cache are computed by the wrapped column generator.
    """

    log = log.getChild(__qualname__)

    def __init__(self, columnGenerator: ColumnGenerator, cache: PersistentKeyValueCache):
        """
        :param columnGenerator: the column generator with which to generate values for keys not found in the cache
        :param cache: the cache in which to store key-value pairs
        """
        super().__init__(columnGenerator.columnName)
        self.columnGenerator = columnGenerator
        self.cache = cache

    def _generateColumn(self, df: pd.DataFrame) -> pd.Series:
        # compute series of cached values
        cacheValues = [self.cache.get(nt.Index) for nt in df.itertuples()]
        cacheSeries = pd.Series(cacheValues, dtype=object, index=df.index).dropna()

        # compute missing values (if any) via wrapped generator, storing them in the cache
        missingValuesDF = df[~df.index.isin(cacheSeries.index)]
        self.log.info(f"Retrieved {len(cacheSeries)} values from the cache, {len(missingValuesDF)} still to be computed by {self.columnGenerator}")
        if len(missingValuesDF) == 0:
            return cacheSeries
        else:
            missingSeries = self.columnGenerator.generateColumn(missingValuesDF)
            for key, value in missingSeries.iteritems():
                self.cache.set(key, value)
            return pd.concat((cacheSeries, missingSeries))


class ColumnGeneratorCachedByIndex(ColumnGenerator, ABC):
    """
    Base class for column generators, which supports cached column generation, each value being generated independently.
    Cache keys are given by the input data frame's index.
    """

    log = log.getChild(__qualname__)

    def __init__(self, columnName: str, cache: PersistentKeyValueCache):
        """
        :param columnName: the name of the column being generated
        :param cache: the cache in which to store key-value pairs
        """
        super().__init__(columnName)
        self.cache = cache

    def _generateColumn(self, df: pd.DataFrame) -> Union[pd.Series, list, np.ndarray]:
        self.log.info(f"Generating columns with {self}")
        values = []
        cacheHits = 0
        for namedTuple in df.itertuples():
            key = namedTuple.Index
            value = self.cache.get(key)
            if value is None:
                value = self._generateValue(namedTuple)
                self.cache.set(key, value)
            else:
                cacheHits += 1
            values.append(value)
        self.log.info(f"Cached column generation resulted in {cacheHits}/{len(df)} cache hits")
        return values

    @abstractmethod
    def _generateValue(self, namedTuple) -> Any:
        pass