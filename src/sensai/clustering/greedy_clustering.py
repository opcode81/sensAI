# -*- coding: utf-8 -*-
import logging
import math
import queue

log = logging.getLogger(__name__)
         

class GreedyAgglomerativeClustering(object):
    """
    An implementation of greedy agglomerative clustering which avoids unnecessary 
    recomputations of merge costs through the management of a priority queue of 
    potential merges
    """
    log = log.getChild(__qualname__)

    class Cluster(object):
        """
        Base class for clusters that can be merged via GreedyAgglomerativeClustering
        """

        def mergeCost(self, other):
            """Computes the cost of merging the clusters other and self; if a merge is admissible, returns math.inf"""
            raise Exception("Not implemented")
        
        def merge(self, other):
            """Merges the given cluster into this cluster"""
            raise Exception("Not implemented")
        
    def __init__(self, clusters):
        """
        Parameters:
            clusters: the initial clusters which are to be aggregated into larger clusters
        """
        self.prioritisedMerges = queue.PriorityQueue()
        self.wrappedClusters = []
        for idx, c in enumerate(clusters):
            self.wrappedClusters.append(GreedyAgglomerativeClustering.WrappedCluster(c, idx, self))
        
    def applyClustering(self):
        """
        Applies greedy agglomerative clustering and returns the list of clusters
        """
        
        # compute all possible merges, adding them to the queue
        self.log.info("Computing initial merges")
        for idx, wc in enumerate(self.wrappedClusters):
            self.log.info("Computing potential merges for cluster index %d" % idx)
            wc.computeMerges(False)
        
        # perform greedy agglomerative clustering
        steps = 0
        while not self.prioritisedMerges.empty():
            self.log.info("Clustering step %d" % (steps+1))
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
        """Wrapper for clusters which stores additional data required for clustering (internal use only)"""
        
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
        """Represents a potential merge"""
        log = log.getChild(__qualname__)

        def __init__(self, c1, c2, mergeCost):
            self.c1 = c1
            self.c2 = c2
            self.mergeCost = mergeCost
            self.evaporated = False

        def apply(self):
            c1, c2 = self.c1, self.c2
            self.log.info("Merging %s into %s..." % (str(c1), str(c2)))
            c1.cluster.merge(c2.cluster)
            c2.isMerged = True
            c1.removeMerges()
            c2.removeMerges()
            self.log.info("Computing new merge costs for %s..." % str(c1))
            c1.computeMerges(True)
        
        def __lt__(self, other):
            return self.mergeCost < other.mergeCost

