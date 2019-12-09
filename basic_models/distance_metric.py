import logging
import os
from abc import abstractmethod
from typing import Sequence, Tuple, Callable, List, Union

import numpy as np
import pandas as pd

from transformations.embeddings.WordEmbeddings import FastTextEmbedding
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


class LinearCombinationDistanceMetric(DistanceMetric):
    def __init__(self, metrics: Sequence[Tuple[float, DistanceMetric]]):
        self.metrics = metrics

    def distance(self, idA, valueA, idB, valueB):
        value = 0
        for weight, metric in self.metrics:
            value += metric.distance(idA, valueA, idB, valueB) * weight
        return value


class WordSetEarthMoverDistanceMetric(DistanceMetric):
    def __init__(self, embeddingFactory: Callable[[], FastTextEmbedding]):
        self.embeddingFactory = embeddingFactory
        self.embedding = None

    def distance(self, idA, valueA, idB, valueB):
        if self.embedding is None:
            log.info("Instantiating embedding from factory")
            self.embedding = self.embeddingFactory()
        return self.embedding.earthMoverDistance(valueA.combined_keywords, valueB.combined_keywords)


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
