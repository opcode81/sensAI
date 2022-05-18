import itertools
import math
from typing import List, Tuple

import numpy as np

from .geo_coords import GeoCoord
from .local_coords import LocalCoordinateSystem
from ..clustering import GreedyAgglomerativeClustering


class GreedyAgglomerativeGeoCoordClusterer:
    def __init__(self, maxMinDistanceForMergeM: float, maxDistanceM: float, minClusterSize: int):
        """
        :param maxMinDistanceForMergeM: the maximum distance, in metres, for the minimum distance between two existing clusters for a merge
            to be admissible
        :param maxDistanceM: the maximum distance, in metres, between any two points for the points to be allowed to be in the same cluster
        :param minClusterSize: the minimum number of points any valid cluster must ultimately contain; the points in any smaller clusters
            shall be considered as outliers
        """
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

    def fitGeoCoords(self, geoCoords: List[GeoCoord], lcs: LocalCoordinateSystem = None) -> None:
        """
        :param geoCoords: the coordinates to be clustered
        :param lcs: the local coordinate system to use for clustering; if None, compute based on mean coordinates
        """
        if lcs is None:
            meanCoord = GeoCoord.meanCoord(geoCoords)
            lcs = LocalCoordinateSystem(meanCoord.lat, meanCoord.lon)
        self.localPoints = [self.LocalPoint(np.array(lcs.getLocalCoords(p.lat, p.lon)), idx) for idx, p in enumerate(geoCoords)]
        clusters = [self.Cluster(lp, self) for lp in self.localPoints]
        clusters = GreedyAgglomerativeClustering(clusters).applyClustering()
        self.clusters = clusters

    def clustersIndices(self) -> Tuple[List[List[int]], List[int]]:
        """
        :return: a tuple (clusters, outliers), where clusters is a list of lists of original point indices where each inner list forms
         a cluster and outliers is the list of indices of points not within a cluster
        """
        outliers = []
        clusters = []
        for c in self.clusters:
            indices = [p.idx for p in c.points]
            if len(c.points) < self.minClusterSize:
                outliers.extend(indices)
            else:
                clusters.append(indices)
        return clusters, outliers
