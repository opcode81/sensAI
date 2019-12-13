import logging
import os
from abc import abstractmethod
from typing import Sequence, Tuple, List, Union

import numpy as np
import pandas as pd

from .util import cache

log = logging.getLogger(__name__)


class DistanceMetric:
    """
    Abstract base class for (symmetric) distance metrics
    """

    # Todo or not todo: this forces an unnatural signature on non-caching implementations of DistanceMetric
    @abstractmethod
    def distance(self, idA, valueA, idB, valueB):
        pass

    @abstractmethod
    def __str__(self):
        super().__str__()


class DistanceMatrixDFCache(cache.PersistentKeyValueCache):
    def __init__(self, picklePath):
        self.picklePath = picklePath
        log.info(f"Loading distance dataframe from {picklePath}")
        self.distanceDf = pd.read_pickle(self.picklePath)
        assert isinstance(self.distanceDf, pd.DataFrame)
        log.info(f"Successfully loaded dataframe of shape {self.shape()} from cache. "
                 f"There are {self.numUnfilledEntries()} unfilled entries")
        self._updated = False

    def shape(self):
        nEntries = len(self.distanceDf)
        return nEntries, nEntries

    @staticmethod
    def _assertTuple(key):
        assert isinstance(key, tuple) and len(key) == 2, f"Expected a tuple of two identifiers, instead got {key}"

    def set(self, key: Tuple[Union[str, int], Union[str, int]], value):
        self._assertTuple(key)
        for identifier in key:
            if identifier not in self.distanceDf.columns:
                log.info(f"Adding new column and row for identifier {identifier}")
                self.distanceDf[identifier] = np.nan
                self.distanceDf.loc[identifier] = np.nan
        i1, i2 = key
        log.debug(f"Adding distance value for identifiers {i1}, {i2}")
        self.distanceDf.loc[i1, i2] = self.distanceDf.loc[i2, i1] = value
        self._updated = True

    def save(self):
        log.info(f"Saving new distance matrix to {self.picklePath}")
        os.makedirs(os.path.dirname(self.picklePath), exist_ok=True)
        self.distanceDf.to_pickle(self.picklePath)

    def saveIfUpdated(self):
        if self._updated:
            self.save()

    def get(self, key: Tuple[Union[str, int], Union[str, int]]):
        self._assertTuple(key)
        i1, i2 = key
        try:
            result = self.distanceDf.loc[i1, i2]
        except KeyError:
            return None
        if result is None or np.isnan(result):
            return None
        return result

    def numUnfilledEntries(self):
        return self.distanceDf.isnull().sum().sum()

    def getAllCached(self, identifier: Union[str, int]):
        return self.distanceDf[[identifier]]


class CachedDistanceMetric(DistanceMetric, cache.CachedValueProviderMixin):
    """
    A decorator which provides caching for a distance metric, i.e. the metric is computed only if the
    value for the given pair of identifiers is not found within the persistent cache
    """

    def __init__(self, distanceMetric: DistanceMetric, keyValueCache: cache.PersistentKeyValueCache):
        cache.CachedValueProviderMixin.__init__(self, keyValueCache)
        self.metric = distanceMetric

    def distance(self, idA, valueA, idB, valueB):
        if idB < idA:
            idA, idB, valueA, valueB = idB, idA, valueB, valueA
        return self._provideValue((idA, idB), (valueA, valueB))

    def _computeValue(self, key: Tuple[Union[str, int], Union[str, int]], data):
        idA, idB = key
        valueA, valueB = data
        return self.metric.distance(idA, valueA, idB, valueB)

    def fillCache(self, dfIndexedById: pd.DataFrame):
        """
        Fill cache for all identifiers in the provided dataframe

        Args:
            dfIndexedById: Dataframe that is indexed by identifiers of the members
        """
        for position, idA in enumerate(dfIndexedById.index):
            if position % 10 == 0:
                log.info(f"Processed {round(100 * position / len(dfIndexedById), 2)}%")
            for idB in dfIndexedById.index[position + 1:]:
                valueA, valueB = dfIndexedById.loc[idA], dfIndexedById.loc[idB]
                self.distance(idA, valueA, idB, valueB)
        self._cache.saveIfUpdated()

    def __str__(self):
        return str(self.metric)


class LinearCombinationDistanceMetric(DistanceMetric):
    def __init__(self, metrics: Sequence[Tuple[float, DistanceMetric]]):
        self.metrics = metrics

    def distance(self, idA, valueA, idB, valueB):
        value = 0
        for weight, metric in self.metrics:
            if weight != 0:
                value += metric.distance(idA, valueA, idB, valueB) * weight
        return value

    def __str__(self):
        return f"Linear combination of {[(weight, str(metric)) for weight, metric in self.metrics]}"


class IdentityDistanceMetric(DistanceMetric):
    def __init__(self, keys: Union[str, List[str]]):
        if not isinstance(keys, list):
            keys = [keys]
        assert keys != [], "At least one key has to be provided"
        self.keys = keys

    def distance(self, idA, valueA, idB, valueB):
        for key in self.keys:
            if getattr(valueA, key) != getattr(valueB, key):
                return 1
        return 0

    def __str__(self):
        return f"IdentityDistanceMetric based on keys: {self.keys}"
