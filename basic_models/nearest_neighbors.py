import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import Callable, List, Union

import numpy as np
import pandas as pd

from .distance_metric import DistanceMetric
from .basic_models_base import VectorClassificationModel, VectorRegressionModel

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


class KNearestNeighboursFinder:
    def __init__(self, df: pd.DataFrame, distanceMetric: DistanceMetric, neighborProvider: NeighborProvider):
        self.neighborProvider = neighborProvider
        self.distanceMetric = distanceMetric
        if any(df.index.duplicated()):
            raise Exception("Found duplicates in the DataFrame's index")

    def findNeighbors(self, id: Union[str, int], value, n_neighbors=20) -> List[Neighbor]:
        result = []
        for neighborId, neighborValue in self.neighborProvider.iterPotentialNeighbors(id, value):
            distance = self.distanceMetric.distance(id, value, neighborId, neighborValue)
            result.append(Neighbor(neighborId, neighborValue, distance))
        result.sort(key=lambda n: n.distance)
        return result[:n_neighbors]

    def findNeighborIds(self, id: Union[str, int], value, n_neighbors=20) -> List[Union[str, int]]:
        return [n.identifier for n in self.findNeighbors(id, value, n_neighbors=n_neighbors)]


class KNearestNeighboursClassificationModel(VectorClassificationModel):
    def __init__(self, n_neighbors: int, distance_metric: DistanceMetric,
            neighbor_provider_factory: Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_provider_factory = neighbor_provider_factory
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.df = None
        self.y = None
        self.knn_finder = None

    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        assert len(y.columns) == 1, "Expected exactly one column in label set Y"
        self.df = X.merge(y, how="inner", left_index=True, right_index=True)
        self.y = y
        neighbor_provider = self.neighbor_provider_factory(self.df)
        self.knn_finder = KNearestNeighboursFinder(self.df, self.distance_metric, neighbor_provider)

    def _predict_proba(self, X: pd.DataFrame):
        outputDf = pd.DataFrame({label: np.nan for label in self._labels}, index=X.index)
        for nt in X.itertuples():
            neighbors = self.findNeighbors(nt)
            probabilities = self._predict_proba_from_neighbors(neighbors)
            outputDf.loc[nt.Index] = probabilities
        return outputDf

    def _predict_proba_from_neighbors(self, neighbors: List['Neighbor']):
        neighborLabels = []
        probabilities = []
        for neigh in neighbors:
            neighborLabels.append(self._getLabel(neigh))
        for label in self._labels:
            probabilities.append(neighborLabels.count(label) / len(neighborLabels))
        return probabilities

    def _getLabel(self, neighbor: 'Neighbor'):
        return self.y.iloc[:, 0].loc[neighbor.identifier]

    def _predict_single_input(self, namedTuple):
        neighbors = self.findNeighbors(namedTuple)
        c = Counter([self._getLabel(neigh) for neigh in neighbors])
        mostCommonLabel, count = c.most_common(1)[0]
        return mostCommonLabel

    def findNeighbors(self, namedTuple):
        return self.knn_finder.findNeighbors(namedTuple.Index, namedTuple, self.n_neighbors)

    def get_params(self):
        return {"n_neighbors": self.n_neighbors, "metric": self.metric.__name__, "dataset_size": len(self)}

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        predictedValues = []
        for i, nt in enumerate(x.itertuples()):
            log.info(f"Computing prediction for #{i+1}/{len(x)}: {nt}")
            predictedValues.append(self._predict_single_input(nt))
        return pd.DataFrame({"category_range": predictedValues}, index=x.index)


class KNearestNeighboursRegressionModel(VectorRegressionModel):
    def __init__(self, numNeighbors: int, distance_metric: DistanceMetric,
            neighbor_provider_factory: Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_provider_factory = neighbor_provider_factory
        self.numNeighbors = numNeighbors
        self.distance_metric = distance_metric
        self.df = None
        self.y = None
        self.knnFinder = None

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame):
        assert len(y.columns) == 1, "Expected exactly one column in label set Y"
        self.df = X.merge(y, how="inner", left_index=True, right_index=True)
        self.y = y
        neighbor_provider = self.neighbor_provider_factory(self.df)
        self.knnFinder = KNearestNeighboursFinder(self.df, self.distance_metric, neighbor_provider)

    def _getTarget(self, neighbor: 'Neighbor'):
        return self.y.iloc[:, 0].loc[neighbor.identifier]

    def _predictSingleInput(self, namedTuple):
        neighbors = self.knnFinder.findNeighbors(namedTuple.Index, namedTuple, self.numNeighbors)
        neighborTargets = np.array([self._getTarget(n) for n in neighbors])
        neighborWeights = np.array([(1.0/(1e-5+n.distance)) for n in neighbors])
        return np.sum(neighborTargets * neighborWeights) / np.sum(neighborWeights)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        predictedValues = []
        for i, nt in enumerate(x.itertuples()):
            log.info(f"Computing prediction for #{i+1}/{len(x)}: {nt}")
            predictedValues.append(self._predictSingleInput(nt))
        return pd.DataFrame({self._predictedVariableNames[0]: predictedValues}, index=x.index)
