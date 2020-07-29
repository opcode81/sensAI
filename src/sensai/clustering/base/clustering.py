import logging
from abc import ABC, abstractmethod
from typing import Union, Set, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from typing_extensions import Protocol

from ...util.cache import PickleLoadSaveMixin

log = logging.getLogger(__name__)


# TODO at some point in the future: generalize to other input and deal with algorithms that allow prediction of labels
class ClusteringModel(PickleLoadSaveMixin, ABC):
    """
    Base class for all clustering algorithms. Supports noise clusters and relabelling of identified clusters as noise
    based on their size.

    :param noiseLabel: label that is associated with the noise cluster or None
    :param minClusterSize: if not None, clusters below this size will be labeled as noise
    :param maxClusterSize: if not None, clusters above this size will be labeled as noise
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
            :return: dictionary containing coarse information about the cluster (e.g. num_members and centroid)
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
        :return: generator of clusters
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
            raise NotImplementedError(f"The algorithm {self} does not provide a noise cluster")
        return self.getCluster(self.noiseLabel)

    def summaryDF(self, condition: Callable[[Cluster], bool] = None):
        """
        :param condition: if provided, only clusters fulfilling the condition will be included
        :return: pandas DataFrame containing coarse information about the clusters
        """
        summary_dicts = [cluster.summaryDict() for cluster in self.clusters(condition=condition)]
        return pd.DataFrame(summary_dicts).set_index("identifier", drop=True)

    def fit(self, data: np.ndarray) -> None:
        log.info(f"Fitting {self} to {len(data)} coordinate datapoints.")
        labels = self._computeLabels(data)
        if len(labels) != len(data):
            raise Exception(f"Bad Implementation: number of labels does not match number of datapoints")
        # Relabel clusters that do not fulfill size bounds as noise
        if self.minClusterSize != -np.inf or self.maxClusterSize != np.inf:
            for clusterId, clusterSize in zip(*np.unique(labels, return_counts=True)):
                if not self.minClusterSize <= clusterSize <= self.maxClusterSize:
                    labels[labels == clusterId] = self.noiseLabel

        self._datapoints = data
        self._clusterIdentifiers = set(labels)
        self._labels = labels
        log.info(f"{self} found {self.numClusters} clusters")

    @property
    def isFitted(self):
        return self._datapoints is not None

    @property
    def datapoints(self) -> np.ndarray:
        assert self.isFitted
        return self._datapoints

    @property
    def labels(self) -> np.ndarray:
        assert self.isFitted
        return self._labels

    @property
    def clusterIdentifiers(self) -> Set[int]:
        assert self.isFitted
        return self._clusterIdentifiers

    # unfortunately, there seems to be no way to annotate the return type correctly
    # https://github.com/python/mypy/issues/3993
    def getCluster(self, clusterId: int) -> Cluster:
        if clusterId not in self.labels:
            raise KeyError(f"no cluster for id {clusterId}")
        return self.Cluster(self.datapoints[self.labels == clusterId], identifier=clusterId)

    @property
    def numClusters(self) -> int:
        return len(self.clusterIdentifiers.difference({self.noiseLabel}))

    @abstractmethod
    def _computeLabels(self, x: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model and return an array of cluster labels

        :param x: the datapoints
        :return: list of the same length as the input datapoints; it represents the mapping coordinate -> cluster_id
        """
        pass


class SKLearnClustererProtocol(Protocol):
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
    :param minClusterSize: if not None, clusters below this size will be labeled as noise
    :param maxClusterSize: if not None, clusters above this size will be labeled as noise
    """

    def __init__(self, clusterer: SKLearnClustererProtocol, noiseLabel=-1,
             minClusterSize: int = None, maxClusterSize: int = None):
        super().__init__(noiseLabel=noiseLabel, minClusterSize=minClusterSize, maxClusterSize=maxClusterSize)
        self.clusterer = clusterer

    def _computeLabels(self, x: np.ndarray):
        self.clusterer.fit(x)
        return self.clusterer.labels_

    def __str__(self):
        return f"{super().__str__()}_{self.clusterer.__class__.__name__}"
