import geopandas as gp
import numpy as np
from abc import ABC, abstractmethod
from shapely.geometry import MultiPoint
from typing import Union, Callable

from sensai.clustering import EuclideanClusterer
from sensai.util.graph import SpanningTree

TCoordinates = Union[np.ndarray, MultiPoint, gp.GeoDataFrame, EuclideanClusterer.Cluster]


def validateCoordinates(coordinates: np.ndarray):
    # for the moment we only support 2-dim coordinates. We can adjust it in the future when needed
    if not len(coordinates.shape) == 2 or coordinates.shape[1] != 2:
        raise Exception(f"Coordinates must be of shape (n, 2), instead got: {coordinates.shape}")


def extractCoordinatesArray(coordinates: TCoordinates) -> np.ndarray:
    """
    Extract coordinates as numpy array
    """
    if isinstance(coordinates, gp.GeoDataFrame):
        try:
            coordinates = np.array(MultiPoint(list(coordinates.geometry)))
        except Exception:
            raise ValueError(f"Could not extract coordinates from GeoDataFrame. "
                             f"Is the geometry column a sequence of Points?")
    elif isinstance(coordinates, MultiPoint):
        coordinates = np.array(coordinates)
    elif isinstance(coordinates, EuclideanClusterer.Cluster):
        coordinates = coordinates.datapoints
    validateCoordinates(coordinates)
    return coordinates


class GeoDataFrameWrapper(ABC):
    @abstractmethod
    def toGeoDF(self, *args, **kwargs) -> gp.GeoDataFrame:
        pass

    def plot(self, *args, **kwargs):
        self.toGeoDF().plot(*args, **kwargs)


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