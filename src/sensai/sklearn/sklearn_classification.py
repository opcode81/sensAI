import logging
from typing import Sequence

import lightgbm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.tree

from .sklearn_base import AbstractSkLearnVectorClassificationModel, DataFrameTransformer


log = logging.getLogger(__name__)


class SkLearnDecisionTreeVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, min_samples_leaf=8, random_state=42, **modelArgs):
        super().__init__(sklearn.tree.DecisionTreeClassifier,
            min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)


class SkLearnRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, min_samples_leaf=8, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestClassifier,
            random_state=random_state, min_samples_leaf=min_samples_leaf, **modelArgs)


class SkLearnMLPVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.neural_network.MLPClassifier, **modelArgs)


class SkLearnMultinomialNBVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.naive_bayes.MultinomialNB, **modelArgs)


class SkLearnLightGBMVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, random_state=42, num_leaves=31, **modelArgs):
        super().__init__(lightgbm.sklearn.LGBMClassifier, random_state=random_state, num_leaves=num_leaves, **modelArgs)

