import logging
from typing import Sequence, List, Union, Callable, Any, Dict, TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from . import util, data_transformation
from .columngen import ColumnGenerator

if TYPE_CHECKING:
    from .vector_model import VectorModel

log = logging.getLogger(__name__)


class DuplicateColumnNamesException(Exception):
    pass


class FeatureGenerator(ABC):
    """
    Base class for feature generators that create a new DataFrame containing feature values
    from an input DataFrame
    """
    def __init__(self, categoricalFeatureNames: Sequence[str] = (),
            normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = (), addCategoricalDefaultRules=True):
        """
        :param categoricalFeatureNames: if provided, will ensure that the respective columns in the generated data frames will
            have dtype 'category'.
            Furthermore, presence of meta-information can later be leveraged for further transformations, e.g. one-hot encoding.
        :param normalisationRules: Rules to be used by DFTNormalisation (e.g. for constructing an input transformer for a model).
            These rules are only relevant if a DFTNormalisation object consuming them is instantiated and used
            within a data processing pipeline. They do not affect feature generation.
        :param addCategoricalDefaultRules:
            If True, normalisation rules for categorical features (which are unsupported by normalisation) and their corresponding one-hot
            encoded features (with "_<index>" appended) will be added.
        """
        self._categoricalFeatureNames = categoricalFeatureNames
        normalisationRules = list(normalisationRules)
        if addCategoricalDefaultRules and len(categoricalFeatureNames) > 0:
            normalisationRules.append(data_transformation.DFTNormalisation.Rule(r"(%s)" % "|".join(categoricalFeatureNames), unsupported=True))
            normalisationRules.append(data_transformation.DFTNormalisation.Rule(r"(%s)_\d+" % "|".join(categoricalFeatureNames), skip=True))
        self._normalisationRules = normalisationRules

    def getNormalisationRules(self) -> Sequence[data_transformation.DFTNormalisation.Rule]:
        return self._normalisationRules

    def getCategoricalFeatureNames(self) -> Sequence[str]:
        return self._categoricalFeatureNames

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        """
        Fits the feature generator based on the given data

        :param X: the input/features data frame for the learning problem
        :param Y: the corresponding output data frame for the learning problem
            (which will typically contain regression or classification target columns)
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        """
        pass

    def generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        """
        Generates features for the data points in the given data frame

        :param df: the input data frame for which to generate features
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        :return: a data frame containing the generated features, which uses the same index as X (and Y)
        """
        resultDF = self._generate(df, ctx=ctx)

        isColumnDuplicatedArray = resultDF.columns.duplicated()
        if any(isColumnDuplicatedArray):
            duplicatedColumns = set(resultDF.columns[isColumnDuplicatedArray])
            raise DuplicateColumnNamesException(f"Feature data frame contains duplicate column names: {duplicatedColumns}")

        # ensure that categorical columns have dtype 'category'
        if len(self._categoricalFeatureNames) > 0:
            resultDF = resultDF.copy()  # resultDF we got might be a view of some other DF, so before we modify it, we must copy it
            for colName in self._categoricalFeatureNames:
                series = resultDF[colName].copy()
                if series.dtype.name != 'category':
                    resultDF[colName] = series.astype('category', copy=False)

        return resultDF

    @abstractmethod
    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        """
        Generates features for the data points in the given data frame.

        :param df: the input data frame for which to generate features
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        :return: a data frame containing the generated features, which uses the same index as X (and Y).
            The data frame's columns holding categorical columns are not required to have dtype 'category';
            this will be ensured by the encapsulating call.
        """
        pass

    def fitGenerate(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None) -> pd.DataFrame:
        """
        Fits the feature generator and subsequently generates features for the data points in the given data frame

        :param X: the input data frame for the learning problem and for which to generate features
        :param Y: the corresponding output data frame for the learning problem
            (which will typically contain regression or classification target columns)
        :param ctx: a context object whose functionality may be required for feature generation;
            this is typically the model instance that this feature generator is to generate inputs for
        :return: a data frame containing the generated features, which uses the same index as X (and Y)
        """
        self.fit(X, Y, ctx)
        return self.generate(X, ctx)


class RuleBasedFeatureGenerator(FeatureGenerator, ABC):
    """
    A feature generator which does not require fitting
    """
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        pass


class MultiFeatureGenerator(FeatureGenerator):
    def __init__(self, featureGenerators: Sequence[FeatureGenerator]):
        self.featureGenerators = featureGenerators
        categoricalFeatureNames = util.concatSequences([fg.getCategoricalFeatureNames() for fg in featureGenerators])
        normalisationRules = util.concatSequences([fg.getNormalisationRules() for fg in featureGenerators])
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules,
            addCategoricalDefaultRules=False)

    def _generateFromMultiple(self, generateFeatures: Callable[[FeatureGenerator], pd.DataFrame], index) -> pd.DataFrame:
        dfs = []
        for fg in self.featureGenerators:
            df = generateFeatures(fg)
            dfs.append(df)
        if len(dfs) == 0:
            return pd.DataFrame(index=index)
        else:
            return pd.concat(dfs, axis=1)

    def _generate(self, inputDF: pd.DataFrame, ctx=None):
        def generateFeatures(fg: FeatureGenerator):
            return fg.generate(inputDF, ctx=ctx)
        return self._generateFromMultiple(generateFeatures, inputDF.index)

    def fitGenerate(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None) -> pd.DataFrame:
        def generateFeatures(fg: FeatureGenerator):
            return fg.fitGenerate(X, Y, ctx)
        return self._generateFromMultiple(generateFeatures, X.index)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        for fg in self.featureGenerators:
            fg.fit(X, Y)


class FeatureGeneratorFromNamedTuples(FeatureGenerator, ABC):
    """
    Generates feature values for one data point at a time, creating a dictionary with
    feature values from each named tuple
    """
    def __init__(self, cache: util.cache.PersistentKeyValueCache = None, categoricalFeatureNames: Sequence[str] = (),
            normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        self.cache = cache

    def _generate(self, df: pd.DataFrame, ctx=None):
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
    def _generateFeatureDict(self, namedTuple) -> Dict[str, Any]:
        """
        Creates a dictionary with feature values from a named tuple

        :param namedTuple: the data point for which to generate features
        :return: a dictionary mapping feature names to values
        """
        pass


class FeatureGeneratorTakeColumns(RuleBasedFeatureGenerator):
    def __init__(self, columns: Union[str, List[str]], categoricalFeatureNames: Sequence[str] = (),
            normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        missingCols = set(self.columns) - set(df.columns)
        if len(missingCols) > 0:
            raise Exception(f"Columns {missingCols} not present in data frame; available columns: {list(df.columns)}")
        return df[self.columns]


class FeatureGeneratorTakeAllColumns(RuleBasedFeatureGenerator):
    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return df


class FeatureGeneratorFlattenColumns(RuleBasedFeatureGenerator):
    """
    Instances of this class take columns with vectors and creates a dataframe with columns containing entries of
    these vectors.

    For example, if columns "vec1", "vec2" contain vectors of dimensions dim1, dim2, a datafrane dim1+dim2 new columns
    will be created. It will contain the columns "vec1_<i1>", "vec2_<i2>" with i1, i2 ranging in (0, dim1), (0, dim2).

    """
    def __init__(self, columns: Union[str, Sequence[str]], categoricalFeatureNames: Sequence[str] = (),
            normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
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


class FeatureGeneratorFromColumnGenerator(RuleBasedFeatureGenerator):
    """
    Implements a feature generator via a column generator
    """
    log = log.getChild(__qualname__)

    def __init__(self, columnGen: ColumnGenerator, takeInputColumnIfPresent=False, categoricalFeatureNames: Sequence[str] = (),
            normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        """
        :param columnGen: the underlying column generator
        :param takeInputColumnIfPresent: if True, then if a column whose name corresponds to the column to generate exists
            in the input data, simply copy it to generate the output (without using the column generator); if False, always
            apply the columnGen to generate the output
        """
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        self.takeInputColumnIfPresent = takeInputColumnIfPresent
        self.columnGen = columnGen

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        colName = self.columnGen.generatedColumnName
        if self.takeInputColumnIfPresent and colName in df.columns:
            self.log.debug(f"Taking column '{colName}' from input data frame")
            series = df[colName]
        else:
            self.log.debug(f"Generating column '{colName}' via {self.columnGen}")
            series = self.columnGen.generateColumn(df)
        return pd.DataFrame({colName: series})


class ChainedFeatureGenerator(FeatureGenerator):
    """
    Chains feature generators such that they are executed one after another. The output of generator i>=1 is the input of
    generator i+1 in the generator sequence.
    """
    def __init__(self, *featureGenerators: FeatureGenerator, categoricalFeatureNames: Sequence[str] = None,
                 normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = None):
        """
        :param featureGenerators: the list of feature generators to apply in order
        :param categoricalFeatureNames: the list of categorical feature names being generated; if None, use the ones
            indicated by the last feature generator in the list
        :param normalisationRules: normalisation rules to use; if None, use rules of the last feature generator in the list
        """
        if len(featureGenerators) == 0:
            raise ValueError("Empty list of feature generators")
        if categoricalFeatureNames is None:
            categoricalFeatureNames = featureGenerators[-1].getCategoricalFeatureNames()
        if normalisationRules is None:
            normalisationRules = featureGenerators[-1].getNormalisationRules()
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        self.featureGenerators = featureGenerators

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        for featureGen in self.featureGenerators:
            df = featureGen.generate(df, ctx)
        return df

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        self.fitGenerate(X, Y, ctx)

    def fitGenerate(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None) -> pd.DataFrame:
        for fg in self.featureGenerators:
            X = fg.fitGenerate(X, Y, ctx)
        return X


class FeatureGeneratorTargetDistribution(FeatureGenerator):
    """
    A feature generator, which, for a column T (typically the categorical target column of a classification problem
    or the continuous target column of a regression problem),
      * can ensure that T takes on limited set of values t_1, ..., t_n by allowing the user to apply binning
        using given bin boundaries
      * computes for each value c of a categorical column C the conditional empirical distribution
        P(T | C=c) in the training data during the training phase,
      * generates, for each requested column C and value c in the column,
        n features '<C>_<T>_distribution_<t_i>' = P(T=t_i | C=c) if flatten=True
        or one feature '<C>_<T>_distribution' = [P(T=t_i | C=c), ..., P(T=t_n | C=c)] if flatten=False
     Being probability values, the features generated by this feature generator are already normalised.
    """
    def __init__(self, columns: Union[str, Sequence[str]], targetColumn: str,
            targetColumnBins: Optional[Union[Sequence[float], int, pd.IntervalIndex]], targetColumnInFeaturesDf=False,
            flatten=True):
        """
        :param columns: the categorical columns for which to generate distribution features
        :param targetColumn: the column the distributions over which will make up the features.
            If targetColumnBins is not None, this column will be discretised before computing the conditional distributions
        :param targetColumnBins: if not None, specifies the binning to apply via pandas.cut
            (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html).
            Note that if a value should match no bin, NaN will generated. To avoid this when specifying bin boundaries in a list,
            -inf and +inf should be used as the first and last entries.
        :param targetColumnInFeaturesDf: if True, when fitting will look for targetColumn in the features data frame (X) instead of in target data frame (Y)
        :param flatten: whether to generate a separate scalar feature per distribution value rather than one feature
            with all of the distribution's values
        """
        self.flatten = flatten
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns
        self.targetColumn = targetColumn
        self.targetColumnInInputDf = targetColumnInFeaturesDf
        self.targetColumnBins = targetColumnBins
        newColumnNamesRegex = r'(%s)_%s_distribution' % ("|".join(columns), targetColumn)
        if self.flatten:
            normalisationRule = data_transformation.DFTNormalisation.Rule(newColumnNamesRegex + r'_.+', skip=True)
        else:
            normalisationRule = data_transformation.DFTNormalisation.Rule(newColumnNamesRegex, unsupported=True)
        super().__init__(normalisationRules=[normalisationRule])
        self._targetColumnValues = None
        # This will hold the mapping: column -> featureValue -> targetValue -> targetValueEmpiricalProbability
        self._discreteTargetDistributionsByColumn: Dict[str, Dict[Any, Dict[Any, float]]] = None

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        """
        This will persist the empirical target probability distributions for all unique values in the specified columns
        """
        if self.targetColumnInInputDf:
            target = X[self.targetColumn]
        else:
            target = Y[self.targetColumn]
        if self.targetColumnBins is not None:
            discretisedTarget = pd.cut(target, self.targetColumnBins)
        else:
            discretisedTarget = target
        self._targetColumnValues = discretisedTarget.unique()

        self._discreteTargetDistributionsByColumn = {}
        for column in self.columns:
            self._discreteTargetDistributionsByColumn[column] = {}
            columnTargetDf = pd.DataFrame()
            columnTargetDf[column] = X[column]
            columnTargetDf["target"] = discretisedTarget.values
            for value, valueTargetsDf in columnTargetDf.groupby(column):
                # The normalized value_counts contain targetValue -> targetValueEmpiricalProbability for the current value
                self._discreteTargetDistributionsByColumn[column][value] = valueTargetsDf["target"].value_counts(normalize=True).to_dict()

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        if self._discreteTargetDistributionsByColumn is None:
            raise Exception("Feature generator has not been fitted")
        resultDf = pd.DataFrame(index=df.index)
        for column in self.columns:
            targetDistributionByValue = self._discreteTargetDistributionsByColumn[column]
            if self.flatten:
                for targetValue in self._targetColumnValues:
                    # Important: pd.Series.apply should not be used here, as it would label the resulting column as categorical
                    resultDf[f"{column}_{self.targetColumn}_distribution_{targetValue}"] = [targetDistributionByValue[value].get(targetValue, 0.0) for value in df[column]]
            else:
                distributions = [[targetDistributionByValue[value].get(targetValue, 0.0) for targetValue in self._targetColumnValues] for value in df[column]]
                resultDf[f"{column}_{self.targetColumn}_distribution"] = pd.Series(distributions, index=df[column].index)
        return resultDf


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

    def registerFactory(self, name, factory: Callable[[], FeatureGenerator]):
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


class FeatureCollector(object):
    """
    A feature collector which can provide a multi-feature generator from a list of names/generators and registry
    """

    def __init__(self, *featureGeneratorsOrNames: Union[str, FeatureGenerator], registry=None):
        """
        :param featureGeneratorsOrNames: generator names (known to articleFeatureGeneratorRegistry) or generator instances.
        :param registry: the feature generator registry for the case where names are passed
        """
        self._featureGeneratorsOrNames = featureGeneratorsOrNames
        self._registry = registry
        self._multiFeatureGenerator = self._createMultiFeatureGenerator()

    def getMultiFeatureGenerator(self) -> MultiFeatureGenerator:
        return self._multiFeatureGenerator

    def _createMultiFeatureGenerator(self):
        featureGenerators = []
        for f in self._featureGeneratorsOrNames:
            if isinstance(f, FeatureGenerator):
                featureGenerators.append(f)
            elif type(f) == str:
                if self._registry is None:
                    raise Exception(f"Received feature name '{f}' instead of instance but no registry to perform the lookup")
                featureGenerators.append(self._registry.getFeatureGenerator(f))
            else:
                raise ValueError(f"Unexpected type {type(f)} in list of features")
        return MultiFeatureGenerator(featureGenerators)


class FeatureGeneratorFromVectorModel(FeatureGenerator):
    def __init__(self, vectorModel: "VectorModel", targetFeatureGenerator: FeatureGenerator, categoricalFeatureNames: Sequence[str] = (),
            normalisationRules: Sequence[data_transformation.DFTNormalisation.Rule] = (),
            inputFeatureGenerator: FeatureGenerator = None):
        """
        Provides a feature via predictions of a given model
        :param vectorModel: model used for generate features from predictions
        :param targetFeatureGenerator: generator for target to be predicted
        :param categoricalFeatureNames:
        :param normalisationRules:
        :param inputFeatureGenerator: optional feature generator to be applied to input of vectorModel's fit and predict
        """
        super().__init__(categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)

        self.targetFeatureGenerator = targetFeatureGenerator
        self.inputFeatureGenerator = inputFeatureGenerator
        self.vectorModel = vectorModel

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, ctx=None):
        targetDF = self.targetFeatureGenerator.fitGenerate(X, Y)
        if self.inputFeatureGenerator:
            X = self.inputFeatureGenerator.fitGenerate(X, Y)
        self.vectorModel.fit(X, targetDF)

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        if self.inputFeatureGenerator:
            df = self.inputFeatureGenerator.generate(df)
        return self.vectorModel.predict(df)


