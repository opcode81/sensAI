import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Sequence, Union, Dict, Callable, Any, Optional, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .columngen import ColumnGenerator
from .util.string import orRegexGroup

_log = logging.getLogger(__name__)


class DataFrameTransformer(ABC):
    """
    Base class for data frame transformers, i.e. objects which can transform one data frame into another
    (possibly applying the transformation to the original data frame - in-place transformation).
    A data frame transformer may require being fitted using training data.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class InvertibleDataFrameTransformer(DataFrameTransformer, ABC):
    @abstractmethod
    def applyInverse(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class RuleBasedDataFrameTransformer(DataFrameTransformer, ABC):
    """Base class for transformers whose logic is entirely based on rules and does not need to be fitted to data"""

    def fit(self, df: pd.DataFrame):
        pass


class DataFrameTransformerChain:
    """Supports the application of a chain of data frame transformers"""

    def __init__(self, dataFrameTransformers: Sequence[DataFrameTransformer]):
        self.dataFrameTransformers = dataFrameTransformers

    def apply(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        """
        Applies the chain of transformers to the given DataFrame, optionally fitting each transformer before applying it.
        Each transformer in the chain receives the transformed output of its predecessor.

        :param df: the data frame
        :param fit: whether to fit the transformers before applying them
        :return: the transformed data frame
        """
        for transformer in self.dataFrameTransformers:
            if fit:
                transformer.fit(df)
            df = transformer.apply(df)
        return df

    def fit(self, df: pd.DataFrame):
        for transformer in self.dataFrameTransformers:
            transformer.fit(df)


class DFTRenameColumns(RuleBasedDataFrameTransformer):
    def __init__(self, columnsMap: Dict[str, str]):
        """
        :param columnsMap: dictionary mapping old column names to new names
        """
        self.columnsMap = columnsMap

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.columnsMap)


class DFTConditionalRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a boolean function to one of the columns and retaining only the rows
    for which the function returns True
    """
    def __init__(self, column: str, condition: Callable[[Any], bool]):
        self.column = column
        self.condition = condition

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.column].apply(self.condition)]


class DFTInSetComparisonRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a boolean function to one of the columns and retaining only the rows
    for which the function returns True
    """
    def __init__(self, column: str, setToKeep: Set):
        self.setToKeep = setToKeep
        self.column = column

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.column].isin(self.setToKeep)]


class DFTNotInSetComparisonRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a boolean function to one of the columns and retaining only the rows
    for which the function returns True
    """
    def __init__(self, column: str, setToDrop: Set):
        self.setToDrop = setToDrop
        self.column = column

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[~df[self.column].isin(self.setToDrop)]


class DFTVectorizedConditionalRowFilterOnColumn(RuleBasedDataFrameTransformer):
    def __init__(self, column: str, vectorizedCondition: Callable[[Any], Sequence[bool]]):
        """

        :param column:
        :param vectorizedCondition:
        """
        self.column = column
        self.vectorizedCondition = vectorizedCondition

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.vectorizedCondition(df[self.column])]


class DFTRowFilter(RuleBasedDataFrameTransformer):
    def __init__(self, condition: Callable[[Any], bool]):
        """
        Filters a data frame by applying a boolean function to each row and retaining only the rows
        for which the function returns True
        :param condition:
        """
        self.condition = condition

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.apply(self.condition, axis=1)]


class DFTModifyColumn(RuleBasedDataFrameTransformer):
    def __init__(self, column: str, columnTransform: Callable):
        """
        Modifies a column specified by 'column' using 'columnTransform'
        :param column:
        :param columnTransform:
        """
        self.columnTransform = columnTransform
        self.column = column

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = df[self.column].apply(self.columnTransform)
        return df


class DFTModifyColumnVectorized(DFTModifyColumn):

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = self.columnTransform(df[self.column].values)
        return df


class DFTOneHotEncoder(DataFrameTransformer):
    def __init__(self, columns: Sequence[str], categoriesList: List[np.ndarray] = None, inplace=False, ignoreUnknown=False):
        """
        One hot encode categorical variables

        :param columns: names of original columns that are to be replaced by a list one-hot encoded columns each
        :param categoriesList: numpy arrays containing the possible values of each of the specified columns.
            If None, the possible values will be inferred from the columns
        :param ignoreUnknown: if True and an unknown category is encountered during transform, the resulting one-hot
            encoded columns for this feature will be all zeros. if False, an unknown category will raise an error.
        """
        self.oneHotEncoders = None
        self.columnNamesToProcess = columns
        self.inplace = inplace
        self.handleUnknown = "ignore" if ignoreUnknown else "error"
        if categoriesList is not None:
            if len(columns) != len(categoriesList):
                raise ValueError(f"Length of categories is not the same as length of columnNamesToProcess")
            self.oneHotEncoders = [OneHotEncoder(categories=[np.sort(categories)], sparse=False, handle_unknown=self.handleUnknown) for categories in categoriesList]

    def fit(self, df: pd.DataFrame):
        if self.oneHotEncoders is None:
            self.oneHotEncoders = [OneHotEncoder(categories=[np.sort(df[column].unique())], sparse=False, handle_unknown=self.handleUnknown) for column in self.columnNamesToProcess]
        for encoder, columnName in zip(self.oneHotEncoders, self.columnNamesToProcess):
            encoder.fit(df[[columnName]])

    def apply(self, df: pd.DataFrame):
        if not self.inplace:
            df = df.copy()
        for encoder, columnName in zip(self.oneHotEncoders, self.columnNamesToProcess):
            encodedArray = encoder.transform(df[[columnName]])
            df = df.drop(columns=columnName)
            for i in range(encodedArray.shape[1]):
                df["%s_%d" % (columnName, i)] = encodedArray[:, i]
        return df


class DFTColumnFilter(RuleBasedDataFrameTransformer):
    """
    A DataFrame transformer that filters columns by retaining or dropping specified columns
    """

    def __init__(self, keep: Union[str, Sequence[str]] = None, drop: Union[str, Sequence[str]] = None):
        self.keep = [keep] if type(keep) == str else keep
        self.drop = drop

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.keep is not None:
            df = df[self.keep]
        if self.drop is not None:
            df = df.drop(columns=self.drop)
        return df


class DFTKeepColumns(DFTColumnFilter):

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.keep]


class DFTDRowFilterOnIndex(RuleBasedDataFrameTransformer):
    def __init__(self, keep: Set = None, drop: Set = None):
        self.drop = drop
        self.keep = keep

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.keep is not None:
            df = df.loc[self.keep]
        if self.drop is not None:
            df = df.drop(self.drop)
        return df


class DFTNormalisation(DataFrameTransformer):
    """
    Applies normalisation/scaling to a data frame by applying a set of transformation rules, where each
    rule defines a set of columns to which it applies (learning a single transformer based on the values
    of all applicable columns)
    """

    class RuleTemplate:
        def __init__(self, skip=False, unsupported=False, transformer: Callable = None):
            """
            :param skip: flag indicating whether no transformation shall be performed on the matching column(s)
            :param unsupported: flag indicating whether normalisation of the matching column(s) is unsupported (shall trigger an exception if attempted)
            :param transformer: a transformer instance (from sklearn.preprocessing, e.g. StandardScaler) to apply to the matching column(s).
                If None the default transformer will be used.
            """
            if skip and transformer is not None:
                raise ValueError("skip==True while transformer is not None")
            self.skip = skip
            self.unsupported = unsupported
            self.transformer = transformer

        def toRule(self, regex: str):
            """
            Convert the template to a rule for all columns matching the regex

            :param regex: a regular expression defining the column the rule applies to
            :return: the resulting Rule
            """
            return DFTNormalisation.Rule(regex, skip=self.skip, unsupported=self.unsupported, transformer=self.transformer)

    class Rule:
        def __init__(self, regex: str, skip=False, unsupported=False, transformer=None):
            """
            :param regex: a regular expression defining the column the rule applies to
            :param skip: flag indicating whether no transformation shall be performed on the matching column(s)
            :param unsupported: flag indicating whether normalisation of the matching column(s) is unsupported (shall trigger an exception if attempted)
            :param transformer: a transformer instance (from sklearn.preprocessing, e.g. StandardScaler) to apply to the matching column(s).
                If None the default transformer will be used.
            """
            if skip and transformer is not None:
                raise ValueError("skip==True while transformer is not None")
            self.regex = re.compile(regex)
            self.skip = skip
            self.unsupported = unsupported
            self.transformer = transformer

        def matches(self, column: str):
            return self.regex.fullmatch(column) is not None

        def matchingColumns(self, columns: Sequence[str]):
            return [col for col in columns if self.matches(col)]

        def __str__(self):
            return f"{self.__class__.__name__}[regex='{self.regex}', unsupported={self.unsupported}, skip={self.skip}, transformer={self.transformer}]"

    def __init__(self, rules: Sequence[Rule], defaultTransformerFactory=None, requireAllHandled=True, inplace=False):
        """
        :param rules: the set of rules to apply
        :param defaultTransformerFactory: a factory for the creation of transformer instances (from sklearn.preprocessing, e.g. StandardScaler)
            that shall be used to create a transformer for all rules that don't specify a particular transformer
        :param requireAllHandled: whether to raise an exception if not all columns are matched by a rule
        :param inplace: whether to apply data frame transformations in-place
        """
        self.requireAllHandled = requireAllHandled
        self.inplace = inplace
        self._userRules = rules
        self._defaultTransformerFactory = defaultTransformerFactory
        self._rules = None

    def fit(self, df: pd.DataFrame):
        matchedRulesByColumn = {}
        self._rules = []
        for rule in self._userRules:
            matchingColumns = rule.matchingColumns(df.columns)
            for c in matchingColumns:
                if c in matchedRulesByColumn:
                    raise Exception(f"More than one rule applies to column '{c}': {matchedRulesByColumn[c]}, {rule}")
                matchedRulesByColumn[c] = rule

            # fit transformer
            if len(matchingColumns) > 0:
                if rule.unsupported:
                    raise Exception(f"Normalisation of columns {matchingColumns} is unsupported according to {rule}")
                if not rule.skip:
                    if rule.transformer is None:
                        if self._defaultTransformerFactory is None:
                            raise Exception(f"No transformer to fit: {rule} defines no transformer and instance has no transformer factory")
                        rule.transformer = self._defaultTransformerFactory()
                    applicableDF = df[matchingColumns]
                    flatValues = applicableDF.values.flatten()
                    rule.transformer.fit(flatValues.reshape((len(flatValues), 1)))
            else:
                _log.log(logging.DEBUG - 1, f"{rule} matched no columns")

            # collect specialised rule for application
            specialisedRule = copy.copy(rule)
            r = orRegexGroup(matchingColumns)
            try:
                specialisedRule.regex = re.compile(r)
            except Exception as e:
                raise Exception(f"Could not compile regex '{r}': {e}")
            self._rules.append(specialisedRule)

    def _checkUnhandledColumns(self, df, matchedRulesByColumn):
        if self.requireAllHandled:
            unhandledColumns = set(df.columns) - set(matchedRulesByColumn.keys())
            if len(unhandledColumns) > 0:
                raise Exception(f"The following columns are not handled by any rules: {unhandledColumns}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        matchedRulesByColumn = {}
        for rule in self._rules:
            for c in rule.matchingColumns(df.columns):
                matchedRulesByColumn[c] = rule
                if not rule.skip:
                    df[c] = rule.transformer.transform(df[[c]].values)
        self._checkUnhandledColumns(df, matchedRulesByColumn)
        return df


class DFTFromColumnGenerators(RuleBasedDataFrameTransformer):
    def __init__(self, columnGenerators: Sequence[ColumnGenerator], inplace=False):
        self.columnGenerators = columnGenerators
        self.inplace = inplace

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        for cg in self.columnGenerators:
            series = cg.generateColumn(df)
            df[series.name] = series
        return df


class DFTCountEntries(RuleBasedDataFrameTransformer):
    def __init__(self, columnForEntryCount: str, columnNameForResultingCounts: str = "counts"):
        self.columnNameForResultingCounts = columnNameForResultingCounts
        self.columnForEntryCount = columnForEntryCount

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        series = df[self.columnForEntryCount].value_counts()
        return pd.DataFrame({self.columnForEntryCount: series.index, self.columnNameForResultingCounts: series.values})


class DFTAggregationOnColumn(RuleBasedDataFrameTransformer):
    def __init__(self, columnForAggregation: str, aggregation: Callable):
        self.columnForAggregation = columnForAggregation
        self.aggregation = aggregation

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(self.columnForAggregation).agg(self.aggregation)


class DFTRoundFloats(RuleBasedDataFrameTransformer):

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.round(df.values), columns=df.columns, index=df.index)


class DFTSkLearnTransformer(InvertibleDataFrameTransformer):
    """
    Applies a transformer from sklearn.preprocessing to (a subset of the columns of) a data frame
    """
    def __init__(self, sklearnTransformer, columns: Optional[List[str]] = None, inplace=False):
        """
        :param sklearnTransformer: the transformer instance (from sklearn.preprocessing) to use (which will be fitted & applied)
        :param columns: the set of column names to which the transformation shall apply; if None, apply it to all columns
        :param inplace: whether to apply the transformation in-place
        """
        self.sklearnTransformer = sklearnTransformer
        self.columns = columns
        self.inplace = inplace

    def fit(self, df: pd.DataFrame):
        cols = self.columns
        if cols is None:
            cols = df.columns
        self.sklearnTransformer.fit(df[cols].values)

    def _apply(self, df: pd.DataFrame, inverse: bool) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        cols = self.columns
        if cols is None:
            cols = df.columns
        if inverse:
            df[cols] = self.sklearnTransformer.inverse_transform(df[cols].values)
        else:
            df[cols] = self.sklearnTransformer.transform(df[cols].values)
        return df

    def apply(self, df):
        return self._apply(df, False)

    def applyInverse(self, df):
        return self._apply(df, True)
