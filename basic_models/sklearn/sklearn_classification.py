import logging

import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.tree

from .sklearn_base import AbstractSkLearnVectorClassificationModel


log = logging.getLogger(__name__)


class SkLearnDecisionTreeVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, inputTransformers=(), min_samples_leaf=8, random_state=42, **modelArgs):
        super().__init__(sklearn.tree.DecisionTreeClassifier, inputTransformers=inputTransformers, min_samples_leaf=min_samples_leaf,
                         random_state=random_state, **modelArgs)


class SkLearnRandomForrestVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, min_samples_leaf=8, random_state=42, inputTransformers=(), **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestClassifier, inputTransformers=inputTransformers, random_state=random_state, min_samples_leaf=min_samples_leaf, **modelArgs)


class SkLearnMLPVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, inputTransformers=(), **modelArgs):
        super().__init__(sklearn.neural_network.MLPClassifier, inputTransformers=inputTransformers, **modelArgs)


class SkLearnMultinomialNBVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, inputTransformers=(), **modelArgs):
        super().__init__(sklearn.naive_bayes.MultinomialNB, inputTransformers=inputTransformers, **modelArgs)