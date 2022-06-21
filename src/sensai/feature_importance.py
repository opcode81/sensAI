import collections
import re
from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, List, Tuple

import seaborn as sns
from matplotlib import pyplot as plt

from .util.plot import MATPLOTLIB_DEFAULT_FIGURE_SIZE


class FeatureImportance:
    def __init__(self, featureImportances: Union[Dict[str, float], Dict[str, Dict[str, float]]]):
        self.featureImportances = featureImportances
        self._isMultiVar = self._isDict(next(iter(featureImportances.values())))

    @staticmethod
    def _isDict(x):
        return hasattr(x, "get")

    def getFeatureImportanceDict(self, predictedVarName=None) -> Dict[str, float]:
        if self._isMultiVar:
            self.featureImportances: Dict[str, Dict[str, float]]
            if predictedVarName is not None:
                return self.featureImportances[predictedVarName]
            else:
                if len(self.featureImportances) > 1:
                    raise ValueError("Must provide predicted variable name (multiple output variables)")
                else:
                    return next(iter(self.featureImportances.values()))
        else:
            return self.featureImportances

    def getSortedTuples(self, predictedVarName=None) -> List[Tuple[str, float]]:
        # noinspection PyTypeChecker
        tuples: List[Tuple[str, float]] = list(self.getFeatureImportanceDict(predictedVarName).items())
        tuples.sort(key=lambda t: t[1])
        return tuples


class FeatureImportanceProvider(ABC):
    """
    Interface for models that can provide feature importance values
    """
    @abstractmethod
    def getFeatureImportances(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Gets the feature importance values

        :return: either a dictionary mapping feature names to importance values or (for models predicting multiple
            variables (independently)) a dictionary which maps predicted variable names to such dictionaries
        """
        pass

    def getFeatureImportance(self) -> FeatureImportance:
        return FeatureImportance(self.getFeatureImportances())


def plotFeatureImportance(featureImportanceDict: Dict[str, float], subtitle: str = None) -> plt.Figure:
    numFeatures = len(featureImportanceDict)
    defaultWidth, defaultHeight = MATPLOTLIB_DEFAULT_FIGURE_SIZE
    height = max(defaultHeight, defaultHeight * numFeatures / 20)
    fig, ax = plt.subplots(figsize=(defaultWidth, height))
    sns.barplot(x=list(featureImportanceDict.values()), y=list(featureImportanceDict.keys()), ax=ax)
    title = "Feature Importance"
    if subtitle is not None:
        title += "\n" + subtitle
    plt.title(title)
    plt.tight_layout()
    return fig


class AggregatedFeatureImportances:
    """
    Aggregates feature importance values (e.g. from models implementing FeatureImportanceProvider, such as sklearn's RandomForest
    models and compatible models from lightgbm, etc.)
    """
    def __init__(self, *featureImportances: Union[FeatureImportanceProvider, Dict[str, float], Dict[str, Dict[str, float]]],
            featureAggRegEx: Sequence[str] = ()):
        r"""
        :param featureImportances: (optional) initial list of feature importance providers or dictionaries to aggregate; further
            values can be added via method add
        :param featureAggRegEx: a sequence of regular expressions describing which feature names to sum as one. Each regex must
            contain exactly one group. If a regex matches a feature name, the feature importance will be summed under the key
            of the matched group instead of the full feature name. For example, the regex r"(\w+)_\d+$" will cause "foo_1" and "foo_2"
            to be summed under "foo" and similarly "bar_1" and "bar_2" to be summed under "bar".
        """
        self.aggDict = None
        self._isNested = None
        self._numDictsAdded = 0
        self._featureAggRegEx = [re.compile(p) for p in featureAggRegEx]
        for d in featureImportances:
            self.add(d)

    @staticmethod
    def _isDict(x):
        return hasattr(x, "get")

    def add(self, featureImportance: Union[FeatureImportanceProvider, Dict[str, float], Dict[str, Dict[str, float]]]):
        """
        Adds the feature importance values from the given dictionary

        :param featureImportance: the dictionary obtained via a model's getFeatureImportances method
        """
        if isinstance(featureImportance, FeatureImportanceProvider):
            featureImportance = featureImportance.getFeatureImportances()
        if self._isNested is None:
            self._isNested = self._isDict(next(iter(featureImportance.values())))
        if self._isNested:
            if self.aggDict is None:
                self.aggDict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
            for targetName, d in featureImportance.items():
                d: dict
                for featureName, value in d.items():
                    self.aggDict[targetName][self._aggFeatureName(featureName)] += value
        else:
            if self.aggDict is None:
                self.aggDict = collections.defaultdict(lambda: 0)
            for featureName, value in featureImportance.items():
                self.aggDict[self._aggFeatureName(featureName)] += value
        self._numDictsAdded += 1

    def _aggFeatureName(self, featureName: str):
        for regex in self._featureAggRegEx:
            m = regex.match(featureName)
            if m is not None:
                return m.group(1)
        return featureName

    def getAggregatedFeatureImportanceDict(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        return self.aggDict

    def getAggregatedFeatureImportance(self) -> FeatureImportance:
        return FeatureImportance(self.aggDict)
