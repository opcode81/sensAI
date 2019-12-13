import collections
from math import log

import pandas as pd

from .basic_models_base import VectorClassificationModel


class NaiveBayesVectorClassificationModel(VectorClassificationModel):
    """
    Naive Bayes with categorical features
    """
    def __init__(self, inputTransformers=(), pseudoCount=0.1):
        """

        :param inputTransformers: the sequence of input transformers
        :param pseudoCount: the count to add to each empirical count in order to avoid overfitting
        """
        super().__init__(inputTransformers=inputTransformers)
        self.prior = None
        self.conditionals = None
        self.pseudoCount = pseudoCount

    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        self.prior = collections.defaultdict(lambda: 0)
        self.conditionals = collections.defaultdict(lambda: [collections.defaultdict(lambda: 0) for _ in range(X.shape[1])])
        increment = 1
        for idxRow in range(X.shape[0]):
            cls = y.iloc[idxRow,0]
            self.prior[cls] += increment
            for idxFeature in range(X.shape[1]):
                value = X.iloc[idxRow, idxFeature]
                self.conditionals[cls][idxFeature][value] += increment

    def _predictClassProbabilities(self, X: pd.DataFrame):
        pass

    def _probability(self, counts, value):
        valueCount = counts.get(value, 0.0)
        totalCount = sum(counts.values())
        return (valueCount + self.pseudoCount) / (totalCount + self.pseudoCount)

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, features in X.iterrows():
            bestCls = None
            bestLp = None
            for cls in self.prior:
                lp = log(self._probability(self.prior, cls))
                for idxFeature, value in enumerate(features):
                    lp += log(self._probability(self.conditionals[cls][idxFeature], value))
                if bestLp is None or lp > bestLp:
                    bestLp = lp
                    bestCls = cls
            results.append(bestCls)
        return pd.DataFrame(results, columns=self.getModelOutputVariableNames())
