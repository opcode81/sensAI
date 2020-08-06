import geopandas as gp
import logging
import networkx as nx
import numpy as np
import scipy
from itertools import combinations
from scipy.spatial.distance import euclidean
from scipy.spatial.qhull import Delaunay
from shapely.geometry import MultiLineString, Polygon
from shapely.ops import polygonize, unary_union
from typing import Callable, Dict

from .coordinates import extractCoordinatesArray, TCoordinates
from ..clustering.coordinate_clustering import GeoDataFrameWrapper

log = logging.getLogger(__name__)


def delaunayGraph(data: np.ndarray, edge_weight: Callable[[np.ndarray, np.ndarray], float] = euclidean):
    """
    The Delaunay triangulation of the data as networkx.Graph

    :param data:
    :param edge_weight: function to compute weight given two coordinate points
    :return: instance of networx.Graph where the edges contain additional datapoints entries for
        "weight" and for constants.COORDINATE_PAIR_KEY
    """
    tri = scipy.spatial.Delaunay(data)
    graph = nx.Graph()

    for simplex in tri.simplices:
        for vertex_id_pair in combinations(simplex, 2):
            coordinate_pair = tri.points[
                np.array(vertex_id_pair)]  # vertex_id_pair is a tuple and needs to be cast to an array
            graph.add_edge(*vertex_id_pair, weight=edge_weight(*coordinate_pair))
    return graph


# after already having implemented this, I found the following package: https://github.com/bellockk/alphashape
# It will compute the same polygons (I have verified it). It also contains an optimizer for alpha, which is, however,
# extremely slow and therefore unusable in most practical applications.
def alphaShape(coordinates: TCoordinates, alpha=0.5):
    """
    Compute the `alpha shape`_ of a set of points. Based on `this implementation`_. In contrast to the standard
    definition of the parameter alpha here we normalize it by the mean edge size of the cluster. This results in
    similar "concavity properties" of the resulting shapes for different coordinate sets and a fixed alpha.

    .. _this implementation: https://sgillies.net/2012/10/13/the-fading-shape-of-alpha.html
    .. _alpha shape: https://en.wikipedia.org/wiki/Alpha_shape

    :param coordinates: a suitable iterable of 2-dimensional coordinates
    :param alpha: alpha value to influence the gooeyness of the border. Larger numbers
        don't fall inward as much as smaller numbers.
    :return: a shapely Polygon
    """
    coordinates = extractCoordinatesArray(coordinates)

    edge_index_pairs = set()
    edge_vertex_pairs = []
    graph = delaunayGraph(coordinates)
    mean_edge_size = graph.size(weight="weight") / graph.number_of_edges()

    def add_edge(edge_index_pair, edge_vertex_pair):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        edge_index_pair = tuple(sorted(edge_index_pair))
        if edge_index_pair in edge_index_pairs:
            # already added
            return
        edge_index_pairs.add(edge_index_pair)
        edge_vertex_pairs.append(edge_vertex_pair)

    tri = Delaunay(coordinates)
    for simplex in tri.simplices:
        vertices = tri.points[simplex]
        area = Polygon(vertices).area
        edges = combinations(vertices, 2)
        product_edges_lengths = 1
        for vertex_1, vertex_2 in edges:
            product_edges_lengths *= euclidean(vertex_1, vertex_2)
        # this is the radius of the circumscribed circle of the triangle
        # see https://en.wikipedia.org/wiki/Circumscribed_circle#Triangles
        circum_r = product_edges_lengths / (4.0 * area)

        if circum_r < mean_edge_size/alpha:
            for index_pair in combinations(simplex, 2):
                add_edge(index_pair, tri.points[np.array(index_pair)])

    remaining_edges = MultiLineString(edge_vertex_pairs)

    return unary_union(list(polygonize(remaining_edges)))


class SpanningTree:
    """
    Wrapper around a tree-finding algorithm that will be applied on the Delaunay graph of the datapoints
    """
    def __init__(self, datapoints: np.ndarray, tree_finder: Callable[[nx.Graph], nx.Graph] = nx.minimum_spanning_tree):
        """
        :param datapoints:
        :param tree_finder: function mapping a graph to a subgraph. The default is minimum_spanning_tree
        """
        datapoints = extractCoordinatesArray(datapoints)
        self.tree = tree_finder(delaunayGraph(datapoints))
        edgeWeights = []
        self.coordinatePairs = []
        for edge in self.tree.edges.data():
            edgeCoordinateIndices, edgeData = [edge[0], edge[1]], edge[2]
            edgeWeights.append(edgeData["weight"])
            self.coordinatePairs.append(datapoints[edgeCoordinateIndices])
        self.edgeWeights = np.array(edgeWeights)

    def totalWeight(self):
        return self.edgeWeights.sum()

    def numEdges(self):
        return len(self.tree.edges)

    def meanEdgeWeight(self):
        return self.edgeWeights.mean()

    def summaryDict(self) -> Dict[str, float]:
        """
        Dictionary containing coarse information about the tree
        """
        return {
            "numEdges": self.numEdges(),
            "totalWeight": self.totalWeight(),
            "meanEdgeWeight": self.meanEdgeWeight()
        }


class CoordinateSpanningTree(SpanningTree, GeoDataFrameWrapper):
    """
    Wrapper around a tree-finding algorithm that will be applied on the Delaunay graph of the coordinates.
    Enhances the :class:`SpanningTree` class by adding methods and validation specific to geospatial coordinates.
    """
    def __init__(self, datapoints: np.ndarray, tree_finder: Callable[[nx.Graph], nx.Graph] = nx.minimum_spanning_tree):
        datapoints = extractCoordinatesArray(datapoints)
        super().__init__(datapoints, tree_finder=tree_finder)

    def multiLineString(self):
        return MultiLineString(self.coordinatePairs)

    def toGeoDF(self, crs='epsg:3857'):
        """
        :param crs: projection. By default pseudo-mercator
        :return: GeoDataFrame of length 1 with the tree as MultiLineString instance
        """
        gdf = gp.GeoDataFrame({"geometry": [self.multiLineString()]})
        gdf.crs = crs
        return gdf
