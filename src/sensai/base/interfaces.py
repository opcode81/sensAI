"""
Contains basic interfaces, mainly for enforcing naming conventions across different classes
"""

from abc import ABC, abstractmethod
from typing import TypeVar

import geopandas as gp

T = TypeVar("T")


# TODO: move this once more clustering-related stuff has been merged
# Not a pure interface but this is still an appropriate place for it as the default plot method will often be overrode
class GeoDataFrameWrapper(ABC):
    @abstractmethod
    def toGeoDF(self, *args, **kwargs) -> gp.GeoDataFrame:
        pass

    def plot(self, *args, **kwargs):
        self.toGeoDF().plot(*args, **kwargs)
