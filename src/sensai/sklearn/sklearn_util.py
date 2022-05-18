import collections
from typing import Dict, Union


class AggregatedFeatureImportances:
    """
    Aggregates feature importance values from models that suppurt method getFeatureImportances
    (e.g. sklearn's RandomForest models and compatible models from lightgbm, etc.)
    """
    def __init__(self, *featureImportanceDicts: Union[Dict[str, float], Dict[str, Dict[str, float]]]):
        self.aggDict = None
        self._isNested = None
        self._numDictsAdded = 0
        for d in featureImportanceDicts:
            self.add(d)

    @staticmethod
    def _isDict(x):
        return hasattr(x, "get")

    def add(self, featureImportance: Union[Dict[str, float], Dict[str, Dict[str, float]]]):
        """
        Adds the feature importance values from the given dictionary

        :param featureImportance: the dictionary obtained via a model's getFeatureImportances method
        """
        if self._isNested is None:
            self._isNested = self._isDict(next(iter(featureImportance.values())))
        if self._isNested:
            if self.aggDict is None:
                self.aggDict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
                for targetName, d in featureImportance.items():
                    d: dict
                    for featureName, value in d.items():
                        self.aggDict[targetName][featureName] += value
        else:
            if self.aggDict is None:
                self.aggDict = collections.defaultdict(lambda: 0)
            for featureName, value in featureImportance.items():
                self.aggDict[featureName] += value
        self._numDictsAdded += 1

    def getFeatureImportanceSum(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        return self.aggDict
