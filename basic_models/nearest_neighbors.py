import logging
from abc import ABC, abstractmethod
import collections
import datetime
from typing import Callable, List, Union

import numpy as np
import pandas as pd

from .distance_metric import DistanceMetric
from .basic_models_base import VectorClassificationModel, VectorRegressionModel
from .util.tracking import stringRepr

log = logging.getLogger(__name__)


class Neighbor:
    def __init__(self, identifier, value: float, distance):
        self.identifier = identifier
        self.distance = distance
        self.value = value


class NeighborProvider(ABC):
    @abstractmethod
    def iterPotentialNeighbors(self, id, value):
        pass


class AllNeighborsProvider(NeighborProvider):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def iterPotentialNeighbors(self, id, value):
        for nt in self.df.itertuples():
            if nt.Index != id:
                yield nt.Index, nt

    def __str__(self):
        return str(self.__class__.__name__)


class TimerangeNeighborsProvider(NeighborProvider):
    def __init__(self, df: pd.DataFrame, timestampsColumn="timestamps", pastTimeRangeDays=120, futureTimeRangeDays=120):
        if not pd.core.dtypes.common.is_datetime64_any_dtype(df[timestampsColumn]):
            raise Exception(f"Column {timestampsColumn} does not have a compatible datatype")

        self.df = df
        self.timestampsColumn = timestampsColumn
        self.pastTimeRangeDays = pastTimeRangeDays
        self.futureTimeRangeDays = futureTimeRangeDays
        self.pastTimeDelta = datetime.timedelta(days=pastTimeRangeDays)
        self.futureTimeDelta = datetime.timedelta(days=futureTimeRangeDays)

    def iterPotentialNeighbors(self, id, value: collections.namedtuple):
        inputTime = getattr(value, self.timestampsColumn)
        for nt in self.df.itertuples():
            if nt.Index != id:
                maxTime, minTime = inputTime + self.futureTimeDelta, inputTime - self.pastTimeDelta
                if minTime <= inputTime <= maxTime:
                    yield nt.Index, nt

    def __str__(self):
        return stringRepr(self, ["pastTimeRangeDays", "futureTimeRangeDays"])


class KNearestNeighboursFinder:
    def __init__(self, df: pd.DataFrame, distanceMetric: DistanceMetric, neighborProvider: NeighborProvider):
        self.neighborProvider = neighborProvider
        self.distanceMetric = distanceMetric
        if any(df.index.duplicated()):
            raise Exception("Found duplicates in the DataFrame's index")

    def findNeighbors(self, id: Union[str, int], value, n_neighbors=20) -> List[Neighbor]:
        result = []
        log.debug(f"Finding neighbors for {id}")
        for neighborId, neighborValue in self.neighborProvider.iterPotentialNeighbors(id, value):
            distance = self.distanceMetric.distance(id, value, neighborId, neighborValue)
            result.append(Neighbor(neighborId, neighborValue, distance))
        result.sort(key=lambda n: n.distance)
        return result[:n_neighbors]

    def findNeighborIds(self, id: Union[str, int], value, n_neighbors=20) -> List[Union[str, int]]:
        return [n.identifier for n in self.findNeighbors(id, value, n_neighbors=n_neighbors)]


class KNearestNeighboursClassificationModel(VectorClassificationModel):
    def __init__(self, numNeighbors: int, distanceMetric: DistanceMetric,
            neighborProviderFactory: Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider,
            distanceBasedWeighting=False, distanceEpsilon=1e-3, **kwargs):
        """
        :param numNeighbors: the number of nearest neighbors to consider
        :param distanceMetric: the distance metric to use
        :param neighborProviderFactory: a factory with which a neighbor provider can be constructed using data
        :param distanceBasedWeighting: whether to weight neighbors according to their distance (inverse); if False, use democratic vote
        :param distanceEpsilon: a distance that is added to all distances for distance-based weighting (in order to avoid 0 distances);
        :param kwargs: parameters to pass on to super-classes
        """
        super().__init__(**kwargs)
        self.distanceEpsilon = distanceEpsilon
        self.distanceBasedWeighting = distanceBasedWeighting
        self.neighborProviderFactory = neighborProviderFactory
        self.neighborProvider: NeighborProvider = None
        self.n_neighbors = numNeighbors
        self.distance_metric = distanceMetric
        self.df = None
        self.y = None
        self.knnFinder = None

    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        assert len(y.columns) == 1, "Expected exactly one column in label set Y"
        self.df = X.merge(y, how="inner", left_index=True, right_index=True)
        self.y = y
        self.neighborProvider = self.neighborProviderFactory(self.df)
        self.knnFinder = KNearestNeighboursFinder(self.df, self.distance_metric, self.neighborProvider)

    def _predictClassProbabilities(self, X: pd.DataFrame):
        outputDf = pd.DataFrame({label: np.nan for label in self._labels}, index=X.index)
        for nt in X.itertuples():
            neighbors = self.findNeighbors(nt)
            probabilities = self._predictClassProbabilityVectorFromNeighbors(neighbors)
            outputDf.loc[nt.Index] = probabilities
        return outputDf

    def _predictClassProbabilityVectorFromNeighbors(self, neighbors: List['Neighbor']):
        weights = collections.defaultdict(lambda: 0)
        total = 0
        for neigh in neighbors:
            if self.distanceBasedWeighting:
                weight = 1.0 / (neigh.distance + self.distanceEpsilon)
            else:
                weight = 1
            weights[self._getLabel(neigh)] += weight
            total += weight
        return [weights[label] / total for label in self._labels]

    def _getLabel(self, neighbor: 'Neighbor'):
        return self.y.iloc[:, 0].loc[neighbor.identifier]

    def findNeighbors(self, namedTuple):
        return self.knnFinder.findNeighbors(namedTuple.Index, namedTuple, self.n_neighbors)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.convertClassProbabilitiesToPredictions(self._predictClassProbabilities(x))

    def __str__(self):
        return stringRepr(self, ["n_neighbors", "distance_metric", "distanceBasedWeighting", "neighborProvider"])



class KNearestNeighboursRegressionModel(VectorRegressionModel):
    def __init__(self, numNeighbors: int, distanceMetric: DistanceMetric,
            neighborProviderFactory: Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider,
            distanceBasedWeighting=False, distanceEpsilon=1e-3, **kwargs):
        """
        :param numNeighbors: the number of nearest neighbors to consider
        :param distanceMetric: the distance metric to use
        :param neighborProviderFactory: a factory with which a neighbor provider can be constructed using data
        :param distanceBasedWeighting: whether to weight neighbors according to their distance (inverse); if False, use democratic vote
        :param distanceEpsilon: a distance that is added to all distances for distance-based weighting (in order to avoid 0 distances);
        :param kwargs: parameters to pass on to super-classes
        """
        super().__init__(**kwargs)
        self.distanceEpsilon = distanceEpsilon
        self.distanceBasedWeighting = distanceBasedWeighting
        self.neighbor_provider_factory = neighborProviderFactory
        self.numNeighbors = numNeighbors
        self.distanceMetric = distanceMetric
        self.df = None
        self.y = None
        self.knnFinder = None

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame):
        assert len(y.columns) == 1, "Expected exactly one column in label set Y"
        self.df = X.merge(y, how="inner", left_index=True, right_index=True)
        self.y = y
        neighbor_provider = self.neighbor_provider_factory(self.df)
        self.knnFinder = KNearestNeighboursFinder(self.df, self.distanceMetric, neighbor_provider)

    def _getTarget(self, neighbor: Neighbor):
        return self.y.iloc[:, 0].loc[neighbor.identifier]

    def _predictSingleInput(self, namedTuple):
        neighbors = self.knnFinder.findNeighbors(namedTuple.Index, namedTuple, self.numNeighbors)
        neighborTargets = np.array([self._getTarget(n) for n in neighbors])
        if self.distanceBasedWeighting:
            neighborWeights = np.array([1.0 / (n.distance + self.distanceEpsilon) for n in neighbors])
            return np.sum(neighborTargets * neighborWeights) / np.sum(neighborWeights)
        else:
            return np.mean(neighborTargets)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        predictedValues = []
        for i, nt in enumerate(x.itertuples()):
            predictedValues.append(self._predictSingleInput(nt))
        return pd.DataFrame({self._predictedVariableNames[0]: predictedValues}, index=x.index)

    def __str__(self):
        return stringRepr(self, ["n_neighbors", "distance_metric", "distanceBasedWeighting", "neighborProviderFactory"])
