import logging
from abc import ABC, abstractmethod
from typing import Union, Set, Dict, Callable, Iterable, Generic, TypeVar
from typing_extensions import Protocol

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from ...util.pickle import PickleSerializingMixin

log = logging.getLogger(__name__)


# TODO or not TODO: at the moment we do not implement predict for clustering models although certain algorithms allow that
class ClusteringModel(PickleSerializingMixin, ABC):
    """
    Base class for all clustering algorithms

    :param noiseLabel: label that is associated with the noise cluster or None
    """
    def __init__(self, noiseLabel=-1, minClusterSize: int = None, maxClusterSize: int = None):
        self._datapoints = None
        self._labels = None
        self._clusterIdentifiers = None

        if minClusterSize is not None or maxClusterSize is not None:
            if noiseLabel is None:
                raise ValueError("the noise label has to be not None for non-trivial bounds on cluster sizes")
        self.noiseLabel = noiseLabel
        self.maxClusterSize = maxClusterSize if maxClusterSize is not None else np.inf
        self.minClusterSize = minClusterSize if minClusterSize is not None else -np.inf

    class Cluster:
        def __init__(self, datapoints: np.ndarray, identifier: Union[int, str]):
            self.datapoints = datapoints
            self.identifier = identifier

        def __len__(self):
            return len(self.datapoints)

        def __str__(self):
            return f"{self.__class__.__name__}_{self.identifier}"

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.identifier == other.identifier and np.array_equal(self.datapoints, other.datapoints)
            return False

        def centroid(self):
            return np.mean(self.datapoints, axis=0)

        def radius(self):
            return np.max(distance_matrix([self.centroid()], self.datapoints))

        def summaryDict(self):
            """
            Dictionary containing coarse information about the cluster (e.g. num_members and centroid)

            :return: A dict
            """
            return {
                "identifier": self.identifier,
                "centroid": self.centroid(),
                "num_members": len(self),
                "radius": self.radius()
            }

    @classmethod
    def __str__(cls):
        return cls.__name__

    def clusters(self, condition: Callable[[Cluster], bool] = None) -> Iterable[Cluster]:
        """
        :param condition: if provided, only clusters fulfilling the condition will be included
        :return: A generator of clusters
        """
        percentageToLog = 0
        for i, clusterId in enumerate(self.clusterIdentifiers.difference({self.noiseLabel})):
            # logging process through the loop
            percentageGenerated = int(100 * i / self.numClusters)
            if percentageGenerated == percentageToLog:
                log.info(f"Processed {percentageToLog}% of clusters")
                percentageToLog += 5

            cluster = self.getCluster(clusterId)
            if condition is None or condition(cluster):
                yield cluster

    def noiseCluster(self):
        if self.noiseLabel is None:
            return NotImplementedError(f"The algorithm {self} does not provide a noise cluster")
        return self.getCluster(self.noiseLabel)

    def summaryDF(self, condition=None):
        """
        Data frame containing coarse information about the clusters

        :param condition: if provided, only clusters fulfilling the condition will be included
        :return: pandas DataFrame
        """
        summary_dicts = [cluster.summaryDict() for cluster in self.clusters(condition=condition)]
        return pd.DataFrame(summary_dicts).set_index("identifier", drop=True)

    def fit(self, data: np.ndarray) -> None:
        log.info(f"Fitting {self} to {len(data)} coordinate datapoints.")
        self._fit(data)
        self._datapoints = data
        self._labels = self.getLabels()
        if len(self._labels) != len(data):
            raise Exception(f"Number of labels does not match number of datapoints")
        self._clusterIdentifiers = set(self._labels)
        log.info(f"{self} found {self.numClusters} clusters")

    @property
    def isFitted(self):
        return self._datapoints is not None

    @property
    def datapoints(self) -> np.ndarray:
        assert self.isFitted
        return self._datapoints

    def getLabels(self) -> np.ndarray:
        assert self.isFitted
        labels = self._getLabels()
        # Relabel clusters that do not fulfill size bounds as noise
        if self.minClusterSize != -np.inf or self.maxClusterSize != np.inf:
            for clusterId, cluster_size in zip(*np.unique(labels, return_counts=True)):
                if not self.minClusterSize <= cluster_size <= self.maxClusterSize:
                    labels[labels == clusterId] = self.noiseLabel
        return labels

    @property
    def clusterIdentifiers(self) -> Set[int]:
        assert self.isFitted
        return self._clusterIdentifiers

    # unfortunately, there seems to be no way to annotate the return type correctly
    # https://github.com/python/mypy/issues/3993
    def getCluster(self, clusterId: int) -> Cluster:
        if clusterId not in self.getLabels():
            raise KeyError(f"no cluster for id {clusterId}")
        return self.Cluster(self.datapoints[self.getLabels() == clusterId], identifier=clusterId)

    @property
    def numClusters(self) -> int:
        return len(self.clusterIdentifiers.difference({self.noiseLabel}))

    @abstractmethod
    def _fit(self, x: np.ndarray) -> None:
        pass

    @abstractmethod
    def _getLabels(self) -> np.ndarray:
        """
        :return: list of the same length as the input datapoints; it represents the mapping coordinate -> cluster_id
        """
        pass


class SKLearnTypeClusterer(Protocol):
    """
    Only used for type hints, do not instantiate
    """
    def fit(self, x: np.ndarray): ...

    labels_: np.ndarray


class SKLearnClusteringModel(ClusteringModel):
    """
    Wrapper around an sklearn-type clustering algorithm

    :param clusterer: a clusterer object compatible the sklearn API
    :param noiseLabel: label that is associated with the noise cluster or None
    """

    @staticmethod
    def _validateInput(data_points: np.ndarray) -> bool:
        pass

    def __init__(self, clusterer: SKLearnTypeClusterer, noiseLabel=-1,
             minClusterSize: int = None, maxClusterSize: int = None):
        super().__init__(noiseLabel=noiseLabel, minClusterSize=minClusterSize, maxClusterSize=maxClusterSize)
        self.clusterer = clusterer

    def _fit(self, x: np.ndarray):
        self.clusterer.fit(x)

    def _getLabels(self):
        return self.clusterer.labels_

    def __str__(self):
        return f"{super().__str__()}_{self.clusterer.__class__.__name__}"
