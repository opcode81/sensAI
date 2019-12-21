import logging
import typing
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from . import distance_metric, nearest_neighbors, util, data_transformation

log = logging.getLogger(__name__)


class FeatureGenerator(ABC):
    """
    Base class for feature generators that create a new DataFrame containing feature values
    from an input DataFrame
    """
    def __init__(self, categoricalFeatureNames: typing.Sequence[str] = (),
            normalisationRules: typing.Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        self._normalisationRules = normalisationRules
        self._categoricalFeatureNames = categoricalFeatureNames

    def getNormalisationRules(self) -> typing.Sequence[data_transformation.DFTNormalisation.Rule]:
        return self._normalisationRules

    def getCategoricalFeatureNames(self) -> typing.Sequence[str]:
        return self._categoricalFeatureNames

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    @abstractmethod
    def generateFeatures(self, df: pd.DataFrame, ctx) -> pd.DataFrame:
        """
        Generates features for the data points in the given data frame

        :param df: the data frame for which to generate features
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        :return: a data frame containing the generated features, which uses the same index as df
        """
        pass


class RuleBasedFeatureGenerator(FeatureGenerator, ABC):
    """
    A feature generator which does not require fitting
    """
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass


class MultiFeatureGenerator(FeatureGenerator):
    def __init__(self, featureGenerators: typing.Sequence[FeatureGenerator]):
        self.featureGenerators = featureGenerators
        categoricalFeatureNames = util.concatSequences([fg.getCategoricalFeatureNames() for fg in featureGenerators])
        normalisationRules = util.concatSequences([fg.getNormalisationRules() for fg in featureGenerators])
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)

    def generateFeatures(self, inputDF: pd.DataFrame, ctx=None):
        dfs = [fg.generateFeatures(inputDF, ctx) for fg in self.featureGenerators]
        return pd.concat(dfs, axis=1)

    def getNormalisationRules(self, withCategoricalOneHotRule=True):
        """
        :param withCategoricalOneHotRule: if True, adds a rule stating that all one-hot encoded categorical features are to be skipped/not to be normalised
        :return: the list of rules
        """
        rules = list(super().getNormalisationRules())
        if withCategoricalOneHotRule:
            rules.append(data_transformation.DFTNormalisation.Rule(r"(%s)_\d+" % "|".join(self.getCategoricalFeatureNames())))
        return rules

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        for fg in self.featureGenerators:
            fg.fit(X, Y)


class FeatureGeneratorFromNamedTuples(FeatureGenerator, ABC):
    """
    Generates feature values for one data point at a time, creating a dictionary with
    feature values from each named tuple
    """
    def __init__(self, cache: util.cache.PersistentKeyValueCache = None, categoricalFeatureNames: typing.Sequence[str] = (),
            normalisationRules: typing.Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        self.cache = cache

    def generateFeatures(self, df: pd.DataFrame, ctx):
        dicts = []
        for idx, nt in enumerate(df.itertuples()):
            if idx % 100 == 0:
                log.debug(f"Generating feature via {self.__class__.__name__} for index {idx}")
            value = None
            if self.cache is not None:
                value = self.cache.get(nt.Index)
            if value is None:
                value = self._generateFeatureDict(nt)
                if self.cache is not None:
                    self.cache.set(nt.Index, value)
            dicts.append(value)
        return pd.DataFrame(dicts, index=df.index)

    @abstractmethod
    def _generateFeatureDict(self, namedTuple) -> typing.Dict[str, typing.Any]:
        """
        Creates a dictionary with feature values from a named tuple

        :param namedTuple: the data point for which to generate features
        :return: a dictionary mapping feature names to values
        """
        pass


class FeatureGeneratorTakeColumns(RuleBasedFeatureGenerator):
    def __init__(self, columns: typing.Union[str, typing.Sequence[str]], categoricalFeatureNames: typing.Sequence[str] = (),
            normalisationRules: typing.Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

    def generateFeatures(self, df: pd.DataFrame, ctx) -> pd.DataFrame:
        return df[self.columns]


class FeatureGeneratorTakeAllColumns(RuleBasedFeatureGenerator):
    def generateFeatures(self, df: pd.DataFrame, ctx) -> pd.DataFrame:
        return df


class FeatureGeneratorFlattenColumns(RuleBasedFeatureGenerator):
    """
    Instances of this class take columns with vectors and creates a dataframe with columns containing entries of
    these vectors.

    For example, if columns "vec1", "vec2" contain vectors of dimensions dim1, dim2, a datafrane dim1+dim2 new columns
    will be created. It will contain the columns "vec1_<i1>", "vec2_<i2>" with i1, i2 ranging in (0, dim1), (0, dim2).

    """
    def __init__(self, columns: typing.Union[str, typing.Sequence[str]], categoricalFeatureNames: typing.Sequence[str] = (),
            normalisationRules: typing.Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

    def generateFeatures(self, df: pd.DataFrame, ctx) -> pd.DataFrame:
        resultDf = pd.DataFrame(index=df.index)
        for col in self.columns:
            log.info(f"Flattening column {col}")
            values = np.stack(df[col].values)
            if len(values.shape) != 2:
                raise ValueError(f"Column {col} was expected to contain one dimensional vectors, something went wrong")
            dimension = values.shape[1]
            new_columns = [f"{col}_{i}" for i in range(dimension)]
            log.info(f"Adding {len(new_columns)} new columns to feature dataframe")
            resultDf[new_columns] = pd.DataFrame(values, index=df.index)
        return resultDf


class ChainedFeatureGenerator(FeatureGenerator):
    """
    Chains feature generators such that they are executed one after another. The output of generator i>=1 is the input of
    generator i+1 in the generator sequence.
    """
    def __init__(self, *featureGenerators: FeatureGenerator, categoricalFeatureNames: typing.Sequence[str] = (),
            normalisationRules: typing.Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        self.featureGenerators = featureGenerators

    def generateFeatures(self, df: pd.DataFrame, ctx) -> pd.DataFrame:
        for featureGen in self.featureGenerators:
            df = featureGen.generateFeatures(df, ctx)
        return df

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        for fg in self.featureGenerators:
            fg.fit(X, Y)


class FeatureGeneratorNeighbors(FeatureGeneratorFromNamedTuples):
    """
    Generates features based on nearest neighbors. For each neighbor, a set of features is added to the output data frame.
    Each feature has the name "n{0-based neighbor index}_{feature name}", where the feature names are configurable
    at construction. The feature name "distance", which indicates the distance of the neighbor to the data point is
    always present.
    """
    def __init__(self, numNeighbors: int,
            neighborAttributes: typing.List[str],
            distanceMetric: distance_metric.DistanceMetric,
            neighborProviderFactory: typing.Callable[[pd.DataFrame], nearest_neighbors.NeighborProvider] = nearest_neighbors.AllNeighborsProvider,
            cache: util.cache.PersistentKeyValueCache = None,
            categoricalFeatureNames: typing.Sequence[str] = (),
            normalisationRules: typing.Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        """
        :param numNeighbors: the number of neighbors for to generate features
        :param neighborAttributes: the attributes of the neighbor's named tuple to include as features (in addition to "distance")
        :param distanceMetric: the distance metric defining which neighbors are near
        :param neighborProviderFactory: a factory for the creation of neighbor provider
        :param cache: an optional key-value cache in which feature values are stored by data point identifier (as given by the DataFrame's index)
        """
        super().__init__(cache=cache, categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        self.neighborAttributes = neighborAttributes
        self.distanceMetric = distanceMetric
        self.neighborProviderFactory = neighborProviderFactory
        self.numNeighbors = numNeighbors
        self._knnFinder: nearest_neighbors.KNearestNeighboursFinder = None
        self._trainX = None

    def generateFeatures(self, df: pd.DataFrame, ctx):
        if self._trainX is None:
            raise Exception("Feature generator has not been fitted")
        neighborProvider = self.neighborProviderFactory(self._trainX)
        self._knnFinder = nearest_neighbors.KNearestNeighboursFinder(self._trainX, self.distanceMetric, neighborProvider)
        return super().generateFeatures(df, ctx)

    def _generateFeatureDict(self, namedTuple) -> typing.Dict[str, typing.Any]:
        neighbors = self._knnFinder.findNeighbors(namedTuple.Index, namedTuple, self.numNeighbors)
        result = {}
        for i, neighbor in enumerate(neighbors):
            result[f"n{i}_distance"] = neighbor.distance
            for attr in self.neighborAttributes:
                result[f"n{i}_{attr}"] = getattr(neighbor.value, attr)
        return result

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self._trainX = X


################################
#
# generator registry
#
################################


class FeatureGeneratorRegistry:
    """
    Represents a registry for named feature generators which can be instantiated via factories.
    Each named feature generator is a singleton, i.e. each factory will be called at most once.
    """
    def __init__(self):
        self._featureGeneratorFactories = {}
        self._featureGeneratorSingletons = {}

    def registerFactory(self, name, factory: typing.Callable[[], FeatureGenerator]):
        """
        Registers a feature generator factory which can subsequently be referenced by models via their name
        :param name: the name
        :param factory: the factory
        """
        if name in self._featureGeneratorFactories:
            raise ValueError(f"Generator for name '{name}' already registered")
        self._featureGeneratorFactories[name] = factory

    def getFeatureGenerator(self, name):
        """
        Creates a feature generator from a name, which must

        :param name: the name of the generator
        :return: a new feature generator instance
        """
        generator = self._featureGeneratorSingletons.get(name)
        if generator is None:
            factory = self._featureGeneratorFactories.get(name)
            if factory is None:
                raise ValueError(f"No factory registered for name '{name}': known names: {list(self._featureGeneratorFactories.keys())}. Use registerFeatureGeneratorFactory to register a new feature generator factory.")
            generator = factory()
            self._featureGeneratorSingletons[name] = generator
        return generator
