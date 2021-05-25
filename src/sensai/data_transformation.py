import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Sequence, Union, Dict, Callable, Any, Optional, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Protocol

from .columngen import ColumnGenerator
from .util import flattenArguments
from .util.pandas import DataFrameColumnChangeTracker
from .util.string import orRegexGroup

log = logging.getLogger(__name__)


class DataFrameTransformer(ABC):
    """
    Base class for data frame transformers, i.e. objects which can transform one data frame into another
    (possibly applying the transformation to the original data frame - in-place transformation).
    A data frame transformer may require being fitted using training data.
    """
    def __init__(self):
        self._name = f"{self.__class__.__name__}-{id(self)}"
        self._isFitted = False
        self._columnChangeTracker: Optional[DataFrameColumnChangeTracker] = None
        self._paramInfo = {}  # arguments passed to init that are not saved otherwise can be persisted here

    # for backwards compatibility with persisted DFTs based on code prior to commit 7088cbbe
    # They lack the __isFitted attribute and we assume that each such DFT was fitted
    def __setstate__(self, d):
        d["_name"] = d.get("_name", f"{self.__class__.__name__}-{id(self)}")
        d["_isFitted"] = d.get("_isFitted", True)
        d["_columnChangeTracker"] = d.get("_columnChangeTracker", None)
        d["_paramInfo"] = d.get("_paramInfo", {})
        self.__dict__ = d

    def getName(self) -> str:
        """
        :return: the name of this dft transformer, which may be a default name if the name has not been set.
        """
        return self._name

    def setName(self, name):
        self._name = name

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._columnChangeTracker = DataFrameColumnChangeTracker(df)
        if not self.isFitted():
            raise Exception(f"Cannot apply a DataFrameTransformer which is not fitted: "
                            f"the df transformer {self.getName()} requires fitting")
        df = self._apply(df)
        self._columnChangeTracker.trackChange(df)
        return df

    def info(self):
        return {
            "name": self.getName(),
            "changeInColumnNames": self._columnChangeTracker.columnChangeString() if self._columnChangeTracker is not None else None,
            "isFitted": self.isFitted(),
        }

    def fit(self, df: pd.DataFrame):
        self._fit(df)
        self._isFitted = True

    def isFitted(self):
        return self._isFitted

    def fitApply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.apply(df)


class InvertibleDataFrameTransformer(DataFrameTransformer, ABC):
    @abstractmethod
    def applyInverse(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class RuleBasedDataFrameTransformer(DataFrameTransformer, ABC):
    """Base class for transformers whose logic is entirely based on rules and does not need to be fitted to data"""

    def _fit(self, df: pd.DataFrame):
        pass

    def fit(self, df: pd.DataFrame):
        pass

    def isFitted(self):
        return True


class DataFrameTransformerChain(DataFrameTransformer):
    """
    Supports the application of a chain of data frame transformers.
    During fit and apply each transformer in the chain receives the transformed output of its predecessor.
    """

    def __init__(self, *dataFrameTransformers: Union[DataFrameTransformer, List[DataFrameTransformer]]):
        super().__init__()
        self.dataFrameTransformers = flattenArguments(dataFrameTransformers)

    def __len__(self):
        return len(self.dataFrameTransformers)

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformer in self.dataFrameTransformers:
            df = transformer.apply(df)
        return df

    def _fit(self, df: pd.DataFrame):
        if len(self.dataFrameTransformers) == 0:
            return
        for transformer in self.dataFrameTransformers[:-1]:
            df = transformer.fitApply(df)
        self.dataFrameTransformers[-1].fit(df)

    def isFitted(self):
        return all([dft.isFitted() for dft in self.dataFrameTransformers])

    def getNames(self) -> List[str]:
        """
        :return: the list of names of all contained feature generators
        """
        return [transf.getName() for transf in self.dataFrameTransformers]

    def info(self):
        info = super().info()
        info["chainedDFTTransformerNames"] = self.getNames()
        info["length"] = len(self)
        return info


class DFTRenameColumns(RuleBasedDataFrameTransformer):
    def __init__(self, columnsMap: Dict[str, str]):
        """
        :param columnsMap: dictionary mapping old column names to new names
        """
        super().__init__()
        self.columnsMap = columnsMap

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.columnsMap)


class DFTConditionalRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a boolean function to one of the columns and retaining only the rows
    for which the function returns True
    """
    def __init__(self, column: str, condition: Callable[[Any], bool]):
        super().__init__()
        self.column = column
        self.condition = condition

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.column].apply(self.condition)]


class DFTInSetComparisonRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame on the selected column and retains only the rows for which the value is in the setToKeep
    """
    def __init__(self, column: str, setToKeep: Set):
        super().__init__()
        self.setToKeep = setToKeep
        self.column = column

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.column].isin(self.setToKeep)]

    def info(self):
        info = super().info()
        info["column"] = self.column
        info["setToKeep"] = self.setToKeep
        return info


class DFTNotInSetComparisonRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame on the selected column and retains only the rows for which the value is not in the setToDrop
    """
    def __init__(self, column: str, setToDrop: Set):
        super().__init__()
        self.setToDrop = setToDrop
        self.column = column

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[~df[self.column].isin(self.setToDrop)]

    def info(self):
        info = super().info()
        info["column"] = self.column
        info["setToDrop"] = self.setToDrop
        return info


class DFTVectorizedConditionalRowFilterOnColumn(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a vectorized condition on the selected column and retaining only the rows
    for which it returns True
    """
    def __init__(self, column: str, vectorizedCondition: Callable[[pd.Series], Sequence[bool]]):
        super().__init__()
        self.column = column
        self.vectorizedCondition = vectorizedCondition

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.vectorizedCondition(df[self.column])]

    def info(self):
        info = super().info()
        info["column"] = self.column
        return info


class DFTRowFilter(RuleBasedDataFrameTransformer):
    """
    Filters a data frame by applying a condition function to each row and retaining only the rows
    for which it returns True
    """
    def __init__(self, condition: Callable[[Any], bool]):
        super().__init__()
        self.condition = condition

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.apply(self.condition, axis=1)]


class DFTModifyColumn(RuleBasedDataFrameTransformer):
    def __init__(self, column: str, columnTransform: Callable):
        """
        Modifies a column specified by 'column' using 'columnTransform'
        :param column:
        :param columnTransform:
        """
        super().__init__()
        self.columnTransform = columnTransform
        self.column = column

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = df[self.column].apply(self.columnTransform)
        return df


class DFTModifyColumnVectorized(DFTModifyColumn):

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.column] = self.columnTransform(df[self.column].values)
        return df


class DFTOneHotEncoder(DataFrameTransformer):
    def __init__(self, columns: Optional[Union[str, Sequence[str]]],
            categories: Union[List[np.ndarray], Dict[str, np.ndarray]] = None, inplace=False, ignoreUnknown=False,
            arrayValuedResult=False):
        """
        One hot encode categorical variables

        :param columns: list of names or regex matching names of columns that are to be replaced by a list one-hot encoded columns each
            (or an array-valued column for the case where useArrayValues=True);
            If None, then no columns are actually to be one-hot-encoded
        :param categories: numpy arrays containing the possible values of each of the specified columns (for case where sequence is specified
            in 'columns') or dictionary mapping column name to array of possible categories for the column name.
            If None, the possible values will be inferred from the columns
        :param inplace: whether to perform the transformation in-place
        :param ignoreUnknown: if True and an unknown category is encountered during transform, the resulting one-hot
            encoded columns for this feature will be all zeros. if False, an unknown category will raise an error.
        :param arrayValuedResult: whether to replace the input columns by columns of the same name containing arrays as values
            instead of creating a separate column per original value
        """
        super().__init__()
        self._paramInfo["columns"] = columns
        self._paramInfo["inferCategories"] = categories is None
        self.oneHotEncoders = None
        if columns is None:
            self._columnsToEncode = []
            self._columnNameRegex = "$"
        elif type(columns) == str:
            self._columnNameRegex = columns
            self._columnsToEncode = None
        else:
            self._columnNameRegex = orRegexGroup(columns)
            self._columnsToEncode = columns
        self.inplace = inplace
        self.arrayValuedResult = arrayValuedResult
        self.handleUnknown = "ignore" if ignoreUnknown else "error"
        if categories is not None:
            if type(categories) == dict:
                self.oneHotEncoders = {col: OneHotEncoder(categories=[np.sort(categories)], sparse=False, handle_unknown=self.handleUnknown) for col, categories in categories.items()}
            else:
                if len(columns) != len(categories):
                    raise ValueError(f"Given categories must have the same length as columns to process")
                self.oneHotEncoders = {col: OneHotEncoder(categories=[np.sort(categories)], sparse=False, handle_unknown=self.handleUnknown) for col, categories in zip(columns, categories)}

    def __setstate__(self, state):
        if "arrayValuedResult" not in state:
            state["arrayValuedResult"] = False
        super().__setstate__(state)

    def _fit(self, df: pd.DataFrame):
        if self._columnsToEncode is None:
            self._columnsToEncode = [c for c in df.columns if re.fullmatch(self._columnNameRegex, c) is not None]
            if len(self._columnsToEncode) == 0:
                log.warning(f"{self} does not apply to any columns, transformer has no effect; regex='{self._columnNameRegex}'")
        if self.oneHotEncoders is None:
            self.oneHotEncoders = {column: OneHotEncoder(categories=[np.sort(df[column].unique())], sparse=False, handle_unknown=self.handleUnknown) for column in self._columnsToEncode}
        for columnName in self._columnsToEncode:
            self.oneHotEncoders[columnName].fit(df[[columnName]])

    def _apply(self, df: pd.DataFrame):
        if len(self._columnsToEncode) == 0:
            return df

        if not self.inplace:
            df = df.copy()
        for columnName in self._columnsToEncode:
            encodedArray = self.oneHotEncoders[columnName].transform(df[[columnName]])
            if not self.arrayValuedResult:
                df = df.drop(columns=columnName)
                for i in range(encodedArray.shape[1]):
                    df["%s_%d" % (columnName, i)] = encodedArray[:, i]
            else:
                df[columnName] = list(encodedArray)
        return df

    def info(self):
        info = super().info()
        info["inplace"] = self.inplace
        info["handleUnknown"] = self.handleUnknown
        info["arrayValuedResult"] = self.arrayValuedResult
        info.update(self._paramInfo)
        return info


class DFTColumnFilter(RuleBasedDataFrameTransformer):
    """
    A DataFrame transformer that filters columns by retaining or dropping specified columns
    """
    def __init__(self, keep: Union[str, Sequence[str]] = None, drop: Union[str, Sequence[str]] = None):
        super().__init__()
        self.keep = [keep] if type(keep) == str else keep
        self.drop = drop

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.keep is not None:
            df = df[self.keep]
        if self.drop is not None:
            df = df.drop(columns=self.drop)
        return df

    def info(self):
        info = super().info()
        info["keep"] = self.keep
        info["drop"] = self.drop
        return info


class DFTKeepColumns(DFTColumnFilter):
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.keep]


class DFTDRowFilterOnIndex(RuleBasedDataFrameTransformer):
    def __init__(self, keep: Set = None, drop: Set = None):
        super().__init__()
        self.drop = drop
        self.keep = keep

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
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
        def __init__(self, skip=False, unsupported=False, transformer=None):
            """
            :param skip: flag indicating whether no transformation shall be performed on all of the columns
            :param unsupported: flag indicating whether normalisation of all columns is unsupported (shall trigger an exception if attempted)
            :param transformer: a transformer instance (from sklearn.preprocessing, e.g. StandardScaler) to apply to all of the columns.
                If None, the default transformer will be used (as specified in DFTNormalisation instance).
            """
            if skip and transformer is not None:
                raise ValueError("skip==True while transformer is not None")
            self.skip = skip
            self.unsupported = unsupported
            self.transformer = transformer

        def toRule(self, regex: Optional[str]):
            """
            Convert the template to a rule for all columns matching the regex

            :param regex: a regular expression defining the column the rule applies to
            :return: the resulting Rule
            """
            return DFTNormalisation.Rule(regex, skip=self.skip, unsupported=self.unsupported, transformer=self.transformer)

        def toPlaceholderRule(self):
            return self.toRule(None)

    class Rule:
        def __init__(self, regex: Optional[str], skip=False, unsupported=False, transformer=None, arrayValued=False, fit=True):
            """
            :param regex: a regular expression defining the column(s) the rule applies to.
                If None, the rule is a placeholder rule and the regex must be set later via setRegex or the rule will not be applicable.
            :param skip: flag indicating whether no transformation shall be performed on the matching column(s)
            :param unsupported: flag indicating whether normalisation of the matching column(s) is unsupported (shall trigger an exception if attempted)
            :param transformer: a transformer instance (from sklearn.preprocessing, e.g. StandardScaler) to apply to the matching column(s).
                If None the default transformer will be used.
            :param arrayValued: whether the column values are not scalars but arrays (of arbitrary lengths).
                It is assumed that all entries in such arrays are to be normalised in the same way.
            :param fit: whether the rule's transformer shall be fitted
            """
            if skip and transformer is not None:
                raise ValueError("skip==True while transformer is not None")
            self.regex = re.compile(regex) if regex is not None else None
            self.skip = skip
            self.unsupported = unsupported
            self.transformer = transformer
            self.arrayValued = arrayValued
            self.fit = fit

        def __setstate__(self, d):
            if "arrayValued" not in d:
                d["arrayValued"] = False
            if "fit" not in d:
                d["fit"] = True
            self.__dict__ = d

        def setRegex(self, regex: str):
            self.regex = re.compile(regex)

        def matches(self, column: str):
            if self.regex is None:
                raise Exception("Attempted to apply a placeholder rule. Perhaps the feature generator from which the rule originated was never applied in order to have the rule instantiated.")
            return self.regex.fullmatch(column) is not None

        def matchingColumns(self, columns: Sequence[str]):
            return [col for col in columns if self.matches(col)]

        def __str__(self):
            return f"{self.__class__.__name__}[regex='{self.regex.pattern}', unsupported={self.unsupported}, skip={self.skip}, transformer={self.transformer}]"

    def __init__(self, rules: Sequence[Rule], defaultTransformerFactory=None, requireAllHandled=True, inplace=False):
        """
        :param rules: the set of rules; rules are always fitted and applied in the given order
        :param defaultTransformerFactory: a factory for the creation of transformer instances (from sklearn.preprocessing, e.g. StandardScaler)
            that shall be used to create a transformer for all rules that don't specify a particular transformer.
            The default transformer will only be applied to columns matched by such rules, unmatched columns will
            not be transformed.
        :param requireAllHandled: whether to raise an exception if not all columns are matched by a rule
        :param inplace: whether to apply data frame transformations in-place
        """
        super().__init__()
        self.requireAllHandled = requireAllHandled
        self.inplace = inplace
        self._userRules = rules
        self._defaultTransformerFactory = defaultTransformerFactory
        self._rules = None

    def _fit(self, df: pd.DataFrame):
        matchedRulesByColumn = {}
        self._rules = []
        for rule in self._userRules:
            matchingColumns = rule.matchingColumns(df.columns)
            for c in matchingColumns:
                if c in matchedRulesByColumn:
                    raise Exception(f"More than one rule applies to column '{c}': {matchedRulesByColumn[c]}, {rule}")
                matchedRulesByColumn[c] = rule

            if len(matchingColumns) > 0:
                if rule.unsupported:
                    raise Exception(f"Normalisation of columns {matchingColumns} is unsupported according to {rule}")
                if not rule.skip:
                    if rule.transformer is None:
                        if self._defaultTransformerFactory is None:
                            raise Exception(f"No transformer to fit: {rule} defines no transformer and instance has no transformer factory")
                        rule.transformer = self._defaultTransformerFactory()
                    if rule.fit:
                        # fit transformer
                        applicableDF = df[matchingColumns]
                        if not rule.arrayValued:
                            flatValues = applicableDF.values.flatten()
                        else:
                            flatValues = np.concatenate(applicableDF.values.flatten())
                        rule.transformer.fit(flatValues.reshape((len(flatValues), 1)))
            else:
                log.log(logging.DEBUG - 1, f"{rule} matched no columns")

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

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        matchedRulesByColumn = {}
        for rule in self._rules:
            for c in rule.matchingColumns(df.columns):
                matchedRulesByColumn[c] = rule
                if not rule.skip:
                    if not rule.arrayValued:
                        df[c] = rule.transformer.transform(df[[c]].values)
                    else:
                        df[c] = [rule.transformer.transform([x])[0] for x in df[c]]
        self._checkUnhandledColumns(df, matchedRulesByColumn)
        return df

    def info(self):
        info = super().info()
        info["requireAllHandled"] = self.requireAllHandled
        info["inplace"] = self.inplace
        return info


class DFTFromColumnGenerators(RuleBasedDataFrameTransformer):
    def __init__(self, columnGenerators: Sequence[ColumnGenerator], inplace=False):
        super().__init__()
        self.columnGenerators = columnGenerators
        self.inplace = inplace

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.inplace:
            df = df.copy()
        for cg in self.columnGenerators:
            series = cg.generateColumn(df)
            df[series.name] = series
        return df

    def info(self):
        info = super().info()
        info["inplace"] = self.inplace
        return info


class DFTCountEntries(RuleBasedDataFrameTransformer):
    """
    Adds a new column with counts of the values on a selected column
    """
    def __init__(self, columnForEntryCount: str, columnNameForResultingCounts: str = "counts"):
        super().__init__()
        self.columnNameForResultingCounts = columnNameForResultingCounts
        self.columnForEntryCount = columnForEntryCount

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        series = df[self.columnForEntryCount].value_counts()
        return pd.DataFrame({self.columnForEntryCount: series.index, self.columnNameForResultingCounts: series.values})

    def info(self):
        info = super().info()
        info["columnNameForResultingCounts"] = self.columnNameForResultingCounts
        info["columnForEntryCount"] = self.columnForEntryCount
        return info


class DFTAggregationOnColumn(RuleBasedDataFrameTransformer):
    def __init__(self, columnForAggregation: str, aggregation: Callable):
        super().__init__()
        self.columnForAggregation = columnForAggregation
        self.aggregation = aggregation

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(self.columnForAggregation).agg(self.aggregation)


class DFTRoundFloats(RuleBasedDataFrameTransformer):
    def __init__(self, decimals=0):
        super().__init__()
        self.decimals = decimals

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.round(df.values, self.decimals), columns=df.columns, index=df.index)

    def info(self):
        info = super().info()
        info["decimals"] = self.decimals
        return info


class SklearnTransformerProtocol(Protocol):
    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        pass

    def transform(self, arr: np.ndarray) -> np.ndarray:
        pass

    def fit(self, arr: np.ndarray):
        pass


class DFTSkLearnTransformer(InvertibleDataFrameTransformer):
    """
    Applies a transformer from sklearn.preprocessing to (a subset of the columns of) a data frame
    """
    def __init__(self, sklearnTransformer: SklearnTransformerProtocol, columns: Optional[List[str]] = None, inplace=False):
        """
        :param sklearnTransformer: the transformer instance (from sklearn.preprocessing) to use (which will be fitted & applied)
        :param columns: the set of column names to which the transformation shall apply; if None, apply it to all columns
        :param inplace: whether to apply the transformation in-place
        """
        super().__init__()
        self.setName(f"{self.__class__.__name__}_wrapped_{sklearnTransformer.__class__.__name__}")
        self.sklearnTransformer = sklearnTransformer
        self.columns = columns
        self.inplace = inplace

    def _fit(self, df: pd.DataFrame):
        cols = self.columns
        if cols is None:
            cols = df.columns
        self.sklearnTransformer.fit(df[cols].values)

    def _apply_transformer(self, df: pd.DataFrame, inverse: bool) -> pd.DataFrame:
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

    def _apply(self, df):
        return self._apply_transformer(df, False)

    def applyInverse(self, df):
        return self._apply_transformer(df, True)

    def info(self):
        info = super().info()
        info["columns"] = self.columns
        info["inplace"] = self.inplace
        info["sklearnTransformerClass"] = self.sklearnTransformer.__class__.__name__
        return info


class DFTSortColumns(RuleBasedDataFrameTransformer):
    """
    Sorts a data frame's columns in ascending order
    """
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[sorted(df.columns)]
