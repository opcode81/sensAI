import geopandas as gp
from shapely.geometry import MultiPoint
import numpy as np


def validateCoordinates(coordinates: np.ndarray):
    # for the moment we only support 2-dim coordinates. We can adjust it in the future when needed
    if not len(coordinates.shape) == 2 or coordinates.shape[1] != 2:
        raise Exception(f"Coordinates must be of shape (n, 2), instead got: {coordinates.shape}")


def coordinatesFromGeoDF(geodf: gp.GeoDataFrame) -> np.ndarray:
    """
    Extract coordinates as numpy array from a GeoDataFrame.

    :param geodf: A GeoDataFrame with one point per row
    :return: coordinates as array
    """
    try:
        return np.array(MultiPoint(list(geodf.geometry)))
    except Exception:
        raise ValueError(f"Could not extract coordinates from GeoDataFrame. Is the geometry column a sequence of Points?")
