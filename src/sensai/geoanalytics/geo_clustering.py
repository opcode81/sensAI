import collections
import itertools
import math
from abc import abstractmethod, ABC
from typing import List, Tuple

import numpy as np
import sklearn.cluster

from .geo_coords import GeoCoord
from .local_coords import LocalCoordinateSystem
from ..clustering import GreedyAgglomerativeClustering


class GeoCoordClusterer(ABC):
    @abstractmethod
    def fitGeoCoords(self, geoCoords: List[GeoCoord]):
        """
        :param geoCoords: the coordinates to be clustered
        """
        pass

    @abstractmethod
    def clustersIndices(self) -> Tuple[List[List[int]], List[int]]:
        """
        :return: a tuple (clusters, outliers), where clusters is a dictionary mapping from cluster index to
            the list of original point indices within the cluster and outliers is the list of indices of points not within
            clusters
        """
        pass


class GreedyAgglomerativeGeoCoordClusterer(GeoCoordClusterer):
    def __init__(self, maxMinDistanceForMergeM: float, maxDistanceM: float, minClusterSize: int, lcs: LocalCoordinateSystem = None):
        """
        :param maxMinDistanceForMergeM: the maximum distance, in metres, for the minimum distance between two existing clusters for a merge
            to be admissible
        :param maxDistanceM: the maximum distance, in metres, between any two points for the points to be allowed to be in the same cluster
        :param minClusterSize: the minimum number of points any valid cluster must ultimately contain; the points in any smaller clusters
            shall be considered as outliers
        :param lcs: the local coordinate system to use for clustering; if None, compute based on mean coordinates passed when fitting
        """
        self.lcs = lcs
        self.minClusterSize = minClusterSize
        self.maxMinDistanceForMerge = maxMinDistanceForMergeM
        self.squaredMaxMinDistanceForMerge = maxMinDistanceForMergeM * maxMinDistanceForMergeM
        self.squaredMaxDistance = maxDistanceM * maxDistanceM
        self.localPoints = None

    class LocalPoint:
        def __init__(self, xy: np.ndarray, idx: int):
            self.idx = idx
            self.xy = xy

    class Cluster(GreedyAgglomerativeClustering.Cluster):
        def __init__(self, point: "GreedyAgglomerativeGeoCoordClusterer.LocalPoint", clusterer: 'GreedyAgglomerativeGeoCoordClusterer'):
            self.clusterer = clusterer
            self.points = [point]

        def mergeCost(self, other):
            cartesianProduct = itertools.product(self.points, other.points)
            minSquaredDistance = math.inf
            for p1, p2 in cartesianProduct:
                diff = p1.xy - p2.xy
                squaredDistance = np.dot(diff, diff)
                if squaredDistance > self.clusterer.squaredMaxDistance:
                    return math.inf
                else:
                    minSquaredDistance = min(squaredDistance, minSquaredDistance)
            if minSquaredDistance <= self.clusterer.squaredMaxMinDistanceForMerge:
                return minSquaredDistance
            return math.inf

        def merge(self, other):
            self.points += other.points

    def fitGeoCoords(self, geoCoords: List[GeoCoord]) -> None:
        if self.lcs is None:
            meanCoord = GeoCoord.meanCoord(geoCoords)
            self.lcs = LocalCoordinateSystem(meanCoord.lat, meanCoord.lon)
        self.localPoints = [self.LocalPoint(np.array(self.lcs.getLocalCoords(p.lat, p.lon)), idx) for idx, p in enumerate(geoCoords)]
        clusters = [self.Cluster(lp, self) for lp in self.localPoints]
        clusters = GreedyAgglomerativeClustering(clusters).applyClustering()
        self.clusters = clusters

    def clustersIndices(self) -> Tuple[List[List[int]], List[int]]:
        outliers = []
        clusters = []
        for c in self.clusters:
            indices = [p.idx for p in c.points]
            if len(c.points) < self.minClusterSize:
                outliers.extend(indices)
            else:
                clusters.append(indices)
        return clusters, outliers


class SkLearnGeoCoordClusterer(GeoCoordClusterer):
    def __init__(self, clusterer, lcs: LocalCoordinateSystem = None):
        """
        :param clusterer: a clusterer from sklearn.cluster
        :param lcs: the local coordinate system to use for Euclidian conversion; if None, determine from data (using mean coordinate as centre)
        """
        self.lcs = lcs
        self.clusterer = clusterer
        self.localPoints = None

    def fitGeoCoords(self, geoCoords: List[GeoCoord]):
        if self.lcs is None:
            meanCoord = GeoCoord.meanCoord(geoCoords)
            self.lcs = LocalCoordinateSystem(meanCoord.lat, meanCoord.lon)
        self.localPoints = [self.lcs.getLocalCoords(p.lat, p.lon) for p in geoCoords]
        self.clusterer.fit(self.localPoints)

    def _clusters(self, mode):
        clusters = collections.defaultdict(list)
        outliers = []
        for idxPoint, idxCluster in enumerate(self.clusterer.labels_):
            if mode == "localPoints":
                item = self.localPoints[idxPoint]
            elif mode == "indices":
                item = idxPoint
            else:
                raise ValueError()
            if idxCluster >= 0:
                clusters[idxCluster].append(item)
            else:
                outliers.append(item)
        return list(clusters.values()), outliers

    def clustersLocalPoints(self) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float]]]:
        """
        :return: a tuple (clusters, outliers), where clusters is a dictionary mapping from cluster index to
            the list of local points within the cluster and outliers is a list of local points not within
            clusters
        """
        return self._clusters("localPoints")

    def clustersIndices(self) -> Tuple[List[List[int]], List[int]]:
        return self._clusters("indices")


class DBSCANGeoCoordClusterer(SkLearnGeoCoordClusterer):
    def __init__(self, eps, min_samples, lcs: LocalCoordinateSystem = None, **kwargs):
        """
        :param eps: the maximum distance between two samples for one to be considered as in the neighbourhood of the other
        :param min_samples: the minimum number of samples that must be within a neighbourhood for a cluster to be formed
        :param lcs: the local coordinate system for conversion to a Euclidian space
        :param kwargs: additional arguments to pass to DBSCAN (see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
        """
        super().__init__(sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, **kwargs), lcs)