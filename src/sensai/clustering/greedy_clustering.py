# -*- coding: utf-8 -*-
import logging
import math
import queue
from abc import ABC, abstractmethod
from typing import List, Sequence

log = logging.getLogger(__name__)


class GreedyAgglomerativeClustering(object):
    """
    An implementation of greedy agglomerative clustering which avoids unnecessary 
    recomputations of merge costs through the management of a priority queue of 
    potential merges.

    Greedy agglomerative clustering works as follows. Starting with an initial
    set of clusters (where each cluster typically contains a single data point),
    the method successively merges the two clusters where the merge cost is lowest (greedy),
    until no further merges are admissible.
    The merge operation is a mutating operation, i.e. the initial clusters are modified.

    To apply the method, the Cluster class must be subclassed, so as to define
    what the cost of a merge in your application shall be and how two clusters can be merged.
    For example, if data points are points in a Cartesian coordinate system, then the merge cost
    can be defined as the minimum or maximum distance among all pairs of points in the two clusters,
    admissibility being determined by a threshold that must not be exceeded;
    the merge operation can simply concatenate lists of data points.
    """
    log = log.getChild(__qualname__)

    class Cluster(ABC):
        """
        Base class for clusters that can be merged via GreedyAgglomerativeClustering
        """
        @abstractmethod
        def mergeCost(self, other) -> float:
            """
            Computes the cost of merging the given cluster with this cluster

            :return: the (non-negative) merge cost or math.inf if a merge is inadmissible"""
            pass

        @abstractmethod
        def merge(self, other):
            """
            Merges the given cluster into this cluster"

            :param other: the cluster that is to be merged into this cluster
            """
            pass

    def __init__(self, clusters: Sequence[Cluster]):
        """
        :param clusters: the initial clusters, which are to be agglomerated into larger clusters
        """
        self.prioritisedMerges = queue.PriorityQueue()
        self.wrappedClusters = []
        for idx, c in enumerate(clusters):
            self.wrappedClusters.append(GreedyAgglomerativeClustering.WrappedCluster(c, idx, self))
        
    def applyClustering(self) -> List[Cluster]:
        """
        Applies greedy agglomerative clustering to the clusters given at construction, merging
        clusters until no further merges are admissible

        :return: the list of agglomerated clusters (subset of the original clusters, which may have had other
            clusters merged into them)
        """
        # compute all possible merges, adding them to the priority queue
        self.log.debug("Computing initial merges")
        for idx, wc in enumerate(self.wrappedClusters):
            self.log.debug("Computing potential merges for cluster index %d" % idx)
            wc.computeMerges(False)
        
        # greedily apply the least-cost merges
        steps = 0
        while not self.prioritisedMerges.empty():
            self.log.debug("Clustering step %d" % (steps+1))
            haveMerge = False
            while not haveMerge and not self.prioritisedMerges.empty():
                merge = self.prioritisedMerges.get()
                if not merge.evaporated:
                    haveMerge = True
            if haveMerge:
                merge.apply()
            steps += 1
        
        result = filter(lambda wc: not wc.isMerged, self.wrappedClusters)
        result = list(map(lambda wc: wc.cluster, result))
        return result
        
    class WrappedCluster(object):
        """
        Wrapper for clusters which stores additional data required for clustering (internal use only)
        """
        def __init__(self, cluster, idx, clusterer):
            self.isMerged = False
            self.merges = []
            self.cluster = cluster
            self.idx = idx
            self.clusterer = clusterer
            
        def removeMerges(self):
            for merge in self.merges:
                merge.evaporated = True
            self.merges = []

        def computeMerges(self, allPairs):
            # add new merges to queue
            startIdx = 0 if allPairs else self.idx + 1
            otherIdx = startIdx
            wrappedClusters = self.clusterer.wrappedClusters
            while otherIdx < len(wrappedClusters):
                if otherIdx != self.idx:
                    other = wrappedClusters[otherIdx]
                    if not other.isMerged:
                        mergeCost = self.cluster.mergeCost(other.cluster)
                        if not math.isinf(mergeCost):
                            merge = GreedyAgglomerativeClustering.ClusterMerge(self, other, mergeCost)
                            self.merges.append(merge)
                            other.merges.append(merge)
                            self.clusterer.prioritisedMerges.put(merge)
                otherIdx += 1
        
        def __str__(self):
            return "Cluster[idx=%d]" % self.idx
        
    class ClusterMerge(object):
        """
        Represents a potential merge
        """
        log = log.getChild(__qualname__)

        def __init__(self, c1, c2, mergeCost):
            self.c1 = c1
            self.c2 = c2
            self.mergeCost = mergeCost
            self.evaporated = False

        def apply(self):
            c1, c2 = self.c1, self.c2
            self.log.debug("Merging %s into %s..." % (str(c1), str(c2)))
            c1.cluster.merge(c2.cluster)
            c2.isMerged = True
            c1.removeMerges()
            c2.removeMerges()
            self.log.debug("Computing new merge costs for %s..." % str(c1))
            c1.computeMerges(True)
        
        def __lt__(self, other):
            return self.mergeCost < other.mergeCost

